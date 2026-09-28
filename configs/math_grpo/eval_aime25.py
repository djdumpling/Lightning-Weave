# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "datasets>=4.0.0,<5",
#   "math-verify==0.9.0",
#   "modal-training-gym @ git+https://github.com/modal-projects/training-gym.git@8899342e709189e2a09a6f9bdadc1e36a28dae79",
# ]
# ///
"""Evaluate the Qwen3-4B math trajectory on the repository's AIME25 task."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

MODEL_REPO = "Qwen/Qwen3-4B"
AIME25_REPO = "math-ai/aime25"
AIME25_SPLIT = "test"
SOURCE_RUN_ID = "vibrato-heat-e3db4b3269c6"
TRAINING_GYM_COMMIT = "8899342e709189e2a09a6f9bdadc1e36a28dae79"
VLLM_VERSION = "0.13.0"

# Match evaluation/math_tasks/weave_aime25.yaml. The engine limit reserves an
# additional 1,024 tokens for the AIME prompt and chat-template overhead.
PROMPT_TEMPLATE = "Question: {problem}\nAnswer:"
MAX_RESPONSE_TOKENS = 32768
MAX_MODEL_LEN = 33792
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
SAMPLES_PER_PROBLEM = 8
STOP = ["Question:", "</s>", "<|im_end|>", "<|eot_id|>"]

EVAL_GPU = "H100"
EVAL_CPU = 4.0
EVAL_MEMORY_MB = 32768
EVAL_TIMEOUT_SECONDS = 24 * 60 * 60
GPU_MEMORY_UTILIZATION = 0.85
VLLM_SEED = 42
EVAL_RESULTS_VOLUME = "lightning-weave-aime25-results"
REMOTE_RESULTS_ROOT = Path("/eval-results")


def require_python_312() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "Modal Training Gym requires Python 3.12 for serialized functions. "
            "Run this through bash scripts/eval_math_aime25.sh."
        )


def format_aime25_prompt(problem: str) -> str:
    return PROMPT_TEMPLATE.format(problem=problem.strip())


def score_aime25_response(answer: str, response: str) -> float:
    """Apply the same math-verify parse/verify rule as the repository task."""

    from math_verify import parse, verify

    try:
        target = parse("$" + answer + "$", parsing_timeout=None)
        prediction = parse(response, parsing_timeout=None)
        return float(bool(target and prediction and verify(target, prediction, timeout_seconds=None)))
    except Exception:  # noqa: BLE001 - malformed generations score zero
        return 0.0


def generation_kwargs() -> dict[str, Any]:
    return {
        "max_tokens": MAX_RESPONSE_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "stop": STOP,
    }


def protocol_hash(protocol: dict[str, Any]) -> str:
    semantic_protocol = {
        key: value for key, value in protocol.items() if key not in {"iterations", "include_base", "environment"}
    }
    return hashlib.sha256(json.dumps(semantic_protocol, sort_keys=True).encode()).hexdigest()[:12]


def remote_result_path(*, run_id: str, target: str, protocol: dict[str, Any]) -> str:
    safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "-", target)
    return f"{run_id}/{protocol_hash(protocol)}/{safe_target}.json"


def load_aime25_rows(max_problems: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(AIME25_REPO, split=AIME25_SPLIT)
    if len(dataset) != 30:
        raise ValueError(f"Expected 30 AIME25 problems, found {len(dataset)}.")
    limit = len(dataset) if max_problems is None else max_problems
    return [
        {
            "prompt": format_aime25_prompt(str(row["problem"])),
            "answer": str(row["answer"]),
            "problem_index": problem_index,
        }
        for problem_index, row in enumerate(dataset.select(range(limit)))
    ]


def run_vllm_batch(
    *, model_path: str, rows: list[dict[str, Any]], samples_per_problem: int
) -> list[Any]:
    """Load vLLM and generate every remaining problem in one offline batch."""

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        seed=VLLM_SEED,
    )
    sampling = SamplingParams(n=samples_per_problem, **generation_kwargs())
    conversations = [[{"role": "user", "content": str(row["prompt"])}] for row in rows]
    outputs = llm.chat(messages=conversations, sampling_params=sampling, use_tqdm=True)
    if len(outputs) != len(rows):
        raise RuntimeError(f"vLLM returned {len(outputs)} requests for {len(rows)} AIME problems")
    return outputs


def build_problem_result(
    example: dict[str, Any], request_output: Any, *, samples_per_problem: int
) -> dict[str, Any]:
    completions = sorted(request_output.outputs, key=lambda output: output.index)
    if len(completions) != samples_per_problem:
        raise RuntimeError(
            f"Problem {int(example['problem_index']) + 1} returned {len(completions)} "
            f"of {samples_per_problem} requested samples"
        )

    samples = []
    for sample_index, completion in enumerate(completions):
        response = str(completion.text)
        samples.append(
            {
                "sample_index": sample_index,
                "score": score_aime25_response(str(example["answer"]), response),
                "response": response,
                "finish_reason": completion.finish_reason,
                "completion_tokens": len(completion.token_ids),
            }
        )

    correct = sum(sample["score"] for sample in samples)
    problem_index = int(example["problem_index"])
    print(
        f"AIME25 problem {problem_index + 1:02d}: {correct:g}/{samples_per_problem} correct",
        flush=True,
    )
    return {
        "score": correct / samples_per_problem,
        "prompt": str(example["prompt"]),
        "response": samples[0]["response"],
        "metadata": {
            "answer": str(example["answer"]),
            "problem_index": problem_index,
            "samples": samples,
        },
    }


def _write_remote_payload(path: Path, payload: dict[str, Any], commit: Callable[[], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    commit()


def run_inprocess_evaluation(
    job: dict[str, Any], *, results_root: Path, commit: Callable[[], None]
) -> dict[str, Any]:
    """Generate, score, and persist one checkpoint inside one GPU container."""

    result_path = results_root / str(job["result_path"])
    payload: dict[str, Any]
    if result_path.exists() and not job.get("force", False):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            print(f"Using completed remote result at {job['result_path']}")
            return {"result_path": job["result_path"], "checkpoint": payload["checkpoint"]}
    else:
        payload = {
            "protocol": job["protocol"],
            "target": job["target"],
            "status": "running",
            "rows": [],
        }

    payload["status"] = "running"
    payload.pop("error", None)
    payload["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    _write_remote_payload(result_path, payload, commit)

    try:
        rows = load_aime25_rows(job.get("max_problems"))
        completed = {int(row["metadata"]["problem_index"]): row for row in payload.get("rows", [])}
        remaining = [row for row in rows if int(row["problem_index"]) not in completed]

        if remaining:
            outputs = run_vllm_batch(
                model_path=str(job["model_path"]),
                rows=remaining,
                samples_per_problem=int(job["samples_per_problem"]),
            )
            for example, request_output in zip(remaining, outputs, strict=True):
                problem_index = int(example["problem_index"])
                completed[problem_index] = build_problem_result(
                    example,
                    request_output,
                    samples_per_problem=int(job["samples_per_problem"]),
                )
                payload["rows"] = [completed[key] for key in sorted(completed)]
                payload["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
                _write_remote_payload(result_path, payload, commit)

        if len(completed) != len(rows):
            raise RuntimeError(f"Only completed {len(completed)} of {len(rows)} AIME25 problems")

        correct_samples = sum(sample["score"] for row in completed.values() for sample in row["metadata"]["samples"])
        total_samples = len(rows) * int(job["samples_per_problem"])
        iteration = job.get("checkpoint_iteration")
        checkpoint = {
            "target": job["target"],
            "checkpoint_iteration": iteration,
            "completed_rollout_step": 0 if iteration is None else iteration + 1,
            "accuracy": correct_samples / total_samples,
            "correct_samples": int(correct_samples),
            "total_samples": total_samples,
            "problems": len(rows),
            "samples_per_problem": int(job["samples_per_problem"]),
        }
        payload.update(
            {
                "status": "completed",
                "checkpoint": checkpoint,
                "rows": [completed[key] for key in sorted(completed)],
                "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
        )
        _write_remote_payload(result_path, payload, commit)
        return {"result_path": job["result_path"], "checkpoint": checkpoint}
    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "error": repr(exc),
                "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
        )
        _write_remote_payload(result_path, payload, commit)
        raise


def result_volume(environment: str) -> Any:
    import modal

    return modal.Volume.from_name(
        EVAL_RESULTS_VOLUME,
        environment_name=environment,
        create_if_missing=True,
    )


def build_gpu_evaluator(
    *, environment: str, app_name: str, checkpoints_volume_name: str = "", checkpoints_mount_path: str = ""
) -> tuple[Any, Any]:
    """Build one in-process vLLM evaluator on a single H100."""

    import modal
    from modal_training_gym.common import hf_secrets

    source_path = Path(__file__).resolve()
    image = (
        modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        .uv_pip_install(
            f"vllm=={VLLM_VERSION}",
            "huggingface-hub==0.36.0",
            "datasets>=4.0.0,<5",
            "math-verify==0.9.0",
        )
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
        .add_local_file(str(source_path), remote_path="/root/eval_aime25.py", copy=True)
    )

    volumes: dict[str, Any] = {
        "/root/.cache/huggingface": modal.Volume.from_name(
            "huggingface-cache", environment_name=environment, create_if_missing=True
        ),
        "/root/.cache/vllm": modal.Volume.from_name(
            "vllm-cache", environment_name=environment, create_if_missing=True
        ),
        str(REMOTE_RESULTS_ROOT): result_volume(environment),
    }
    if checkpoints_volume_name:
        volumes[checkpoints_mount_path or "/checkpoints"] = modal.Volume.from_name(
            checkpoints_volume_name,
            environment_name=environment,
            create_if_missing=False,
        )

    app = modal.App(
        app_name,
        tags={
            "_modal_source": "lightning-weave",
            "_modal_job_type": "evaluation",
            "_modal_framework": "vllm-offline",
        },
    )
    results = volumes[str(REMOTE_RESULTS_ROOT)]

    @app.function(
        name="evaluate",
        image=image,
        gpu=EVAL_GPU,
        cpu=EVAL_CPU,
        memory=EVAL_MEMORY_MB,
        timeout=EVAL_TIMEOUT_SECONDS,
        volumes=volumes,
        secrets=hf_secrets(),
        serialized=True,
    )
    def evaluate_on_gpu(job: dict[str, Any]) -> dict[str, Any]:
        import importlib.util

        spec = importlib.util.spec_from_file_location("lightning_weave_aime25_remote", "/root/eval_aime25.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load the AIME25 evaluator in Modal")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.run_inprocess_evaluation(
            job,
            results_root=module.REMOTE_RESULTS_ROOT,
            commit=results.commit,
        )

    return app, evaluate_on_gpu


def build_conversion_recipe(environment: str) -> Any:
    """Use Training Gym's one-GPU recipe only for Megatron-to-HF conversion."""

    from modal_training_gym import Qwen3_4B_VllmRecipe

    return Qwen3_4B_VllmRecipe(gpu=EVAL_GPU, n_gpu=1, environment_name=environment)


def checkpoint_iteration(checkpoint: Any) -> int:
    match = re.fullmatch(r"iter_(\d+)", checkpoint.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint name: {checkpoint.name!r}")
    return int(match.group(1))


def select_checkpoints(checkpoints: list[Any], iterations: str) -> list[Any]:
    available = {checkpoint_iteration(checkpoint): checkpoint for checkpoint in checkpoints}
    if iterations.strip().lower() == "all":
        return [available[key] for key in sorted(available)]
    requested = [int(value.strip()) for value in iterations.split(",") if value.strip()]
    missing = sorted(set(requested) - available.keys())
    if missing:
        raise ValueError(f"Missing checkpoint iteration(s) {missing}; available: {sorted(available)}")
    return [available[key] for key in requested]


def list_checkpoints_with_retry(source_run: Any, *, max_attempts: int = 6) -> list[Any]:
    for attempt in range(1, max_attempts + 1):
        try:
            return source_run.checkpoints()
        except Exception as exc:
            if "rate limit" not in str(exc).lower() or attempt == max_attempts:
                raise
            delay = min(2**attempt, 30)
            print(
                f"Checkpoint listing was rate-limited; retrying in {delay}s ({attempt}/{max_attempts})...",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def target_name(checkpoint: Any | None) -> str:
    return "base" if checkpoint is None else checkpoint.name


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_remote_payload(volume: Any, path: str) -> dict[str, Any] | None:
    try:
        return json.loads(b"".join(volume.read_file(path)))
    except FileNotFoundError:
        return None


def save_collected_result(*, output_dir: Path, protocol: dict[str, Any], payload: dict[str, Any]) -> None:
    target = str(payload["target"])
    write_json(output_dir / f"{target}.json", payload)
    curve_path = output_dir / "curve.json"
    curve = json.loads(curve_path.read_text(encoding="utf-8")) if curve_path.exists() else {
        "protocol": protocol,
        "results": [],
    }
    if protocol_hash(curve.get("protocol", {})) != protocol_hash(protocol):
        curve = {"protocol": protocol, "results": []}
    records = {str(record["target"]): record for record in curve.get("results", []) if "target" in record}
    if payload.get("status") == "completed":
        records[target] = payload["checkpoint"]
    else:
        records[target] = {
            "target": target,
            "status": payload.get("status", "unknown"),
            "completed_problems": len(payload.get("rows", [])),
            "error": payload.get("error"),
        }
    order = {"base": -1}
    curve = {
        "protocol": protocol,
        "results": sorted(
            records.values(),
            key=lambda record: order.get(str(record["target"]), int(record.get("checkpoint_iteration") or 0)),
        ),
    }
    write_json(curve_path, curve)


def save_launch_record(output_dir: Path, record: dict[str, Any]) -> None:
    path = output_dir / "launches.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"launches": []}
    launches = {str(item["target"]): item for item in payload.get("launches", []) if "target" in item}
    launches[str(record["target"])] = record
    write_json(path, {"launches": list(launches.values())})


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-id", default=SOURCE_RUN_ID)
    result.add_argument(
        "--iterations",
        default="all",
        help="Comma-separated zero-based checkpoint iterations, or 'all'.",
    )
    result.add_argument("--samples-per-problem", type=int, default=SAMPLES_PER_PROBLEM)
    result.add_argument("--environment", default=os.environ.get("MODAL_ENVIRONMENT", "alex-dev-2"))
    result.add_argument("--output-dir", type=Path, default=None, help="Defaults to results/math/aime25/<run-id>.")
    result.add_argument("--no-base", action="store_true", help="Skip the untrained Qwen3-4B baseline.")
    result.add_argument(
        "--detach", action="store_true", help="Launch detached Modal H100 evaluators and return without waiting."
    )
    result.add_argument(
        "--collect",
        action="store_true",
        help="Download remote progress/results without allocating GPUs.",
    )
    result.add_argument("--force", action="store_true", help="Ignore a completed remote result and evaluate again.")
    result.add_argument(
        "--max-problems", type=int, default=None, help="Smoke-test only: evaluate the first N of 30 problems."
    )
    result.add_argument(
        "--list-checkpoints", action="store_true", help="List committed checkpoints without allocating GPUs."
    )
    result.add_argument("--dry-run", action="store_true", help="Print the static protocol without contacting Modal.")
    return result


def protocol_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_run_id": args.run_id,
        "model": MODEL_REPO,
        "dataset": AIME25_REPO,
        "split": AIME25_SPLIT,
        "prompt_template": PROMPT_TEMPLATE,
        "response_tokens": MAX_RESPONSE_TOKENS,
        "model_context": MAX_MODEL_LEN,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "stop": STOP,
        "samples_per_problem": args.samples_per_problem,
        "max_problems": args.max_problems or 30,
        "metric": "Avg@N exact-answer accuracy",
        "engine": f"vLLM {VLLM_VERSION} offline batch",
        "gpu": EVAL_GPU,
        "gpu_count_per_checkpoint": 1,
        "durable_results_volume": EVAL_RESULTS_VOLUME,
        "environment": args.environment,
        "iterations": args.iterations,
        "include_base": not args.no_base,
    }


def main() -> None:
    require_python_312()
    args = parser().parse_args()
    if args.samples_per_problem < 1:
        raise ValueError("--samples-per-problem must be at least 1")
    if args.max_problems is not None and not 1 <= args.max_problems <= 30:
        raise ValueError("--max-problems must be between 1 and 30")

    protocol = protocol_summary(args)
    print(json.dumps(protocol, indent=2))
    if args.dry_run:
        return

    from modal_training_gym import Qwen3_4B, TrainingRun, convert_megatron_checkpoint_to_hf

    source_run = TrainingRun.from_id(args.run_id)
    checkpoints = select_checkpoints(list_checkpoints_with_retry(source_run), args.iterations)
    print(
        "Committed checkpoints:",
        ", ".join(f"{checkpoint.name} (step {checkpoint_iteration(checkpoint) + 1})" for checkpoint in checkpoints),
    )
    if args.list_checkpoints:
        return

    output_dir = (args.output_dir or Path("results/math/aime25") / args.run_id).resolve()
    targets: list[Any | None] = ([] if args.no_base else [None]) + checkpoints
    results = result_volume(args.environment)
    failures: list[str] = []

    for checkpoint in targets:
        name = target_name(checkpoint)
        iteration = None if checkpoint is None else checkpoint_iteration(checkpoint)
        result_path = remote_result_path(run_id=args.run_id, target=name, protocol=protocol)
        existing = read_remote_payload(results, result_path)
        if args.collect or (existing is not None and existing.get("status") == "completed" and not args.force):
            if existing is None:
                print(f"{name}: no remote result at {result_path}")
                failures.append(name)
                continue
            save_collected_result(output_dir=output_dir, protocol=protocol, payload=existing)
            status = existing.get("status", "unknown")
            completed = len(existing.get("rows", []))
            if status == "completed":
                print(
                    f"{name}: accuracy={existing['checkpoint']['accuracy']:.4f} "
                    f"({completed} problems; collected from Modal)",
                    flush=True,
                )
            else:
                print(f"{name}: {status} ({completed}/{args.max_problems or 30} problems persisted)")
            continue

        print(f"\nLaunching {name}...", flush=True)
        try:
            model_path = MODEL_REPO
            checkpoints_volume_name = ""
            checkpoints_mount_path = ""
            if checkpoint is not None:
                one_gpu_checkpoint = dataclasses.replace(checkpoint, training_run_id="")
                hf_checkpoint = convert_megatron_checkpoint_to_hf(
                    one_gpu_checkpoint,
                    Qwen3_4B(),
                    recipe=build_conversion_recipe(args.environment),
                )
                model_path = hf_checkpoint.path
                checkpoints_volume_name = hf_checkpoint.checkpoints_volume_name
                checkpoints_mount_path = hf_checkpoint.checkpoints_mount_path

            app_name = f"lw-aime25-{name.replace('_', '-')}"
            evaluator_app, evaluate_on_gpu = build_gpu_evaluator(
                environment=args.environment,
                app_name=app_name,
                checkpoints_volume_name=checkpoints_volume_name,
                checkpoints_mount_path=checkpoints_mount_path,
            )
            job = {
                "protocol": protocol,
                "target": name,
                "checkpoint_iteration": iteration,
                "model_path": model_path,
                "result_path": result_path,
                "samples_per_problem": args.samples_per_problem,
                "max_problems": args.max_problems,
                "force": args.force,
            }

            import modal

            with (
                modal.enable_output(),
                evaluator_app.run(name=app_name, detach=True, environment_name=args.environment),
            ):
                function_call = evaluate_on_gpu.spawn(job)
                save_launch_record(
                    output_dir,
                    {
                        "target": name,
                        "function_call_id": function_call.object_id,
                        "evaluator": "Modal H100 in-process vLLM",
                        "modal_app_id": evaluator_app.app_id,
                        "remote_result_path": result_path,
                        "launched_at": datetime.datetime.now(datetime.UTC).isoformat(),
                    },
                )
                print(
                    f"{name}: H100 evaluator {function_call.object_id} launched; "
                    f"results are durable at {EVAL_RESULTS_VOLUME}/{result_path}",
                    flush=True,
                )

            if args.detach:
                continue
            function_call.get()
            payload = read_remote_payload(results, result_path)
            if payload is None:
                raise RuntimeError(f"Modal evaluator completed without writing {result_path}")
            save_collected_result(output_dir=output_dir, protocol=protocol, payload=payload)
            if payload.get("status") != "completed":
                raise RuntimeError(
                    f"Remote evaluator status is {payload.get('status')}: "
                    f"{payload.get('error', 'no error recorded')}"
                )
            record = payload["checkpoint"]
            print(
                f"{name}: accuracy={record['accuracy']:.4f} "
                f"({record['problems']} problems x {args.samples_per_problem} samples)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - continue the checkpoint sweep
            failures.append(name)
            payload = read_remote_payload(results, result_path)
            if payload is not None:
                save_collected_result(output_dir=output_dir, protocol=protocol, payload=payload)
            print(f"{name} failed: {exc!r}", file=sys.stderr, flush=True)

    curve_path = output_dir / "curve.json"
    if curve_path.exists() and not args.detach:
        curve = json.loads(curve_path.read_text(encoding="utf-8"))
        print("\nAIME25 checkpoint curve")
        for record in curve["results"]:
            if "accuracy" in record:
                print(f"  {record['target']:>16}: {100 * record['accuracy']:6.2f}%")
            else:
                print(f"  {record['target']:>16}: {str(record.get('status', 'unknown')).upper()}")
        print(f"Results: {curve_path}")
    if failures and not args.detach:
        raise RuntimeError(f"Evaluation failed for: {', '.join(failures)}")


if __name__ == "__main__":
    main()
