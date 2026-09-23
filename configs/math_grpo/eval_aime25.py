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
import random
import re
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

MODEL_REPO = "Qwen/Qwen3-4B"
AIME25_REPO = "math-ai/aime25"
AIME25_SPLIT = "test"
SOURCE_RUN_ID = "vibrato-heat-e3db4b3269c6"
TRAINING_GYM_COMMIT = "8899342e709189e2a09a6f9bdadc1e36a28dae79"

# Match evaluation/math_tasks/weave_aime25.yaml. The serving limit reserves an
# additional 1,024 tokens for the AIME prompt and chat-template overhead.
PROMPT_TEMPLATE = "Question: {problem}\nAnswer:"
MAX_RESPONSE_TOKENS = 32768
MAX_MODEL_LEN = 33792
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
SAMPLES_PER_PROBLEM = 8
STOP = ["Question:", "</s>", "<|im_end|>", "<|eot_id|>"]

EVAL_RESULTS_VOLUME = "lightning-weave-aime25-results"
REMOTE_RESULTS_ROOT = Path("/eval-results")
REMOTE_DRIVER_CPU = 4.0
REMOTE_DRIVER_MEMORY_MB = 8192
REMOTE_DRIVER_TIMEOUT_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT_SECONDS = 10 * 60
REQUEST_MAX_ATTEMPTS = 8


def require_python_312() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "Modal Training Gym requires Python 3.12 for serialized functions. "
            "Run this through `bash scripts/eval_math_aime25.sh`."
        )


def format_aime25_prompt(problem: str) -> str:
    return PROMPT_TEMPLATE.format(problem=problem.strip())


def score_aime25_response(answer: str, response: str) -> float:
    """Apply the same math-verify parse/verify rule as the repository task."""

    from math_verify import parse, verify

    try:
        # EvalConfig grades on worker threads. SIGALRM-based timeouts are only
        # legal on the main thread, so disable them without changing parsing.
        target = parse(f"${answer}$", parsing_timeout=None)
        prediction = parse(response, parsing_timeout=None)
        return float(bool(target and prediction and verify(target, prediction, timeout_seconds=None)))
    except Exception:  # noqa: BLE001 - malformed generations score zero
        return 0.0


def make_aime25_dataset(max_problems: int | None = None) -> Any:
    from modal_training_gym import DatasetConfig

    class AIME25Dataset(DatasetConfig):
        def cache_key(self) -> str:
            return f"{AIME25_REPO}:{AIME25_SPLIT}:lightning-weave-v1"

        def input_key(self) -> str:
            return "prompt"

        def label_key(self) -> str:
            return "answer"

        def rows(self):
            from datasets import load_dataset

            dataset = load_dataset(AIME25_REPO, split=AIME25_SPLIT)
            if len(dataset) != 30:
                raise ValueError(f"Expected 30 AIME25 problems, found {len(dataset)}.")
            limit = len(dataset) if max_problems is None else max_problems
            for problem_index, row in enumerate(dataset.select(range(limit))):
                yield {
                    "prompt": format_aime25_prompt(str(row["problem"])),
                    "answer": str(row["answer"]),
                    "problem_index": problem_index,
                }

    return AIME25Dataset()


def make_eval_function(
    *, samples_per_problem: int, generate_kwargs: dict[str, Any]
) -> Callable[[Any, dict[str, Any]], Any]:
    """Return one compact EvalConfig row containing all samples for a problem."""

    def evaluate_problem(deployment: Any, example: dict[str, Any]) -> Any:
        from modal_training_gym import EvalRowResult

        prompt = str(example["prompt"])
        answer = str(example["answer"])
        samples = []
        for sample_index in range(samples_per_problem):
            response = deployment.generate(prompt, **generate_kwargs)
            score = score_aime25_response(answer, response)
            samples.append(
                {
                    "sample_index": sample_index,
                    "score": score,
                    "response": response,
                }
            )

        score = sum(sample["score"] for sample in samples) / len(samples)
        print(
            f"AIME25 problem {int(example['problem_index']) + 1:02d}: "
            f"{sum(sample['score'] for sample in samples):g}/{len(samples)} correct",
            flush=True,
        )
        return EvalRowResult(
            score=score,
            prompt=prompt,
            response=samples[0]["response"],
            metadata={
                "answer": answer,
                "problem_index": int(example["problem_index"]),
                "samples": samples,
            },
        )

    return evaluate_problem


def generation_kwargs() -> dict[str, Any]:
    return {
        "max_tokens": MAX_RESPONSE_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "stop": STOP,
    }


def protocol_hash(protocol: dict[str, Any]) -> str:
    # Launch selection and workspace routing do not change model outputs.
    semantic_protocol = {
        key: value for key, value in protocol.items() if key not in {"iterations", "include_base", "environment"}
    }
    return hashlib.sha256(json.dumps(semantic_protocol, sort_keys=True).encode()).hexdigest()[:12]


def remote_result_path(*, run_id: str, target: str, protocol: dict[str, Any]) -> str:
    safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "-", target)
    return f"{run_id}/{protocol_hash(protocol)}/{safe_target}.json"


def generate_from_endpoint(
    *,
    endpoint_url: str,
    served_model_name: str,
    prompt: str,
    generate_kwargs: dict[str, Any],
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
    max_attempts: int = REQUEST_MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Call the OpenAI-compatible server with bounded transient retries."""

    import requests

    body = {
        "model": served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        **generate_kwargs,
    }
    retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                f"{endpoint_url.rstrip('/')}/v1/chat/completions",
                json=body,
                timeout=(15, timeout_seconds),
            )
            if response.status_code in retryable_statuses:
                response.raise_for_status()
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content")
            if content is None:
                content = message.get("reasoning_content", "")
            usage = payload.get("usage") or {}
            return {
                "response": str(content),
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_error = exc
            retryable = not isinstance(exc, requests.HTTPError) or (
                exc.response is not None and exc.response.status_code in retryable_statuses
            )
            if not retryable or attempt == max_attempts:
                raise
            delay = min(2 ** (attempt - 1), 30) + random.random()
            print(
                f"Transient generation failure; retrying in {delay:.1f}s ({attempt}/{max_attempts}): {exc}",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError("Generation retry loop exhausted") from last_error


def wait_until_endpoint_ready(endpoint_url: str, *, timeout_seconds: int = 50 * 60) -> None:
    """Wait for the deployed vLLM server from inside Modal's network."""

    import requests

    deadline = time.monotonic() + timeout_seconds
    last_status = "no response"
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{endpoint_url.rstrip('/')}/v1/models", timeout=(10, 30))
            if response.ok and response.json().get("data"):
                return
            last_status = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_status = f"{type(exc).__name__}: {exc}"
        print(f"Waiting for vLLM ({last_status})...", flush=True)
        time.sleep(10)
    raise TimeoutError(f"vLLM endpoint was not ready after {timeout_seconds}s ({last_status})")


def evaluate_problem_from_endpoint(
    *,
    endpoint_url: str,
    served_model_name: str,
    example: dict[str, Any],
    samples_per_problem: int,
    generate_kwargs: dict[str, Any],
) -> dict[str, Any]:
    samples = []
    for sample_index in range(samples_per_problem):
        generation = generate_from_endpoint(
            endpoint_url=endpoint_url,
            served_model_name=served_model_name,
            prompt=str(example["prompt"]),
            generate_kwargs=generate_kwargs,
        )
        response = generation.pop("response")
        samples.append(
            {
                "sample_index": sample_index,
                "score": score_aime25_response(str(example["answer"]), response),
                "response": response,
                **generation,
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


def _write_remote_payload(path: Path, payload: dict[str, Any], commit: Callable[[], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    commit()


def _stop_modal_app(app_id: str) -> None:
    """Best-effort endpoint cleanup from inside the Modal CPU container."""

    if not app_id:
        return
    try:
        from modal._utils.async_utils import synchronizer
        from modal.client import _Client
        from modal_proto import api_pb2

        async def stop() -> None:
            client = await _Client.from_env()
            await client.stub.AppStop(
                api_pb2.AppStopRequest(
                    app_id=app_id,
                    source=api_pb2.APP_STOP_SOURCE_PYTHON_CLIENT,
                )
            )

        synchronizer.create_blocking(stop)()
    except Exception as exc:  # noqa: BLE001 - cleanup must not erase the result
        print(f"WARNING: could not stop serving app {app_id}: {exc!r}", flush=True)


def run_remote_evaluation(job: dict[str, Any], *, results_root: Path, commit: Callable[[], None]) -> dict[str, Any]:
    """Run and durably checkpoint the whole evaluator inside Modal."""

    result_path = results_root / str(job["result_path"])
    payload: dict[str, Any]
    if result_path.exists() and not job.get("force", False):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            print(f"Using completed remote result at {job['result_path']}")
            if job.get("stop_deployment", True):
                _stop_modal_app(str(job.get("modal_app_id", "")))
            return {
                "result_path": job["result_path"],
                "checkpoint": payload["checkpoint"],
            }
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
        wait_until_endpoint_ready(str(job["endpoint_url"]))
        rows = load_aime25_rows(job.get("max_problems"))
        completed = {int(row["metadata"]["problem_index"]): row for row in payload.get("rows", [])}
        remaining = [row for row in rows if int(row["problem_index"]) not in completed]
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=int(job["max_concurrency"])) as executor:
            futures = {
                executor.submit(
                    evaluate_problem_from_endpoint,
                    endpoint_url=str(job["endpoint_url"]),
                    served_model_name=str(job["served_model_name"]),
                    example=row,
                    samples_per_problem=int(job["samples_per_problem"]),
                    generate_kwargs=dict(job["generate_kwargs"]),
                ): int(row["problem_index"])
                for row in remaining
            }
            for future in as_completed(futures):
                problem_index = futures[future]
                try:
                    completed[problem_index] = future.result()
                    payload["rows"] = [completed[key] for key in sorted(completed)]
                    payload["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
                    _write_remote_payload(result_path, payload, commit)
                except Exception as exc:  # noqa: BLE001 - finish/persist other rows
                    message = f"problem {problem_index + 1}: {exc!r}"
                    errors.append(message)
                    print(f"ERROR: {message}", flush=True)

        if errors:
            raise RuntimeError("; ".join(errors))
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
    finally:
        if job.get("stop_deployment", True):
            _stop_modal_app(str(job.get("modal_app_id", "")))


def build_remote_driver(environment: str | None = None) -> tuple[Any, Any, Any]:
    """Build the detached CPU evaluator and its durable result volume."""

    import modal

    source_path = Path(__file__).resolve()
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            "datasets>=4.0.0,<5",
            "math-verify==0.9.0",
            "requests>=2.32.0,<3",
        )
        .add_local_file(str(source_path), remote_path="/root/eval_aime25.py", copy=True)
    )
    volume = modal.Volume.from_name(
        EVAL_RESULTS_VOLUME,
        environment_name=environment,
        create_if_missing=True,
    )
    app = modal.App("lightning-weave-aime25-driver")

    @app.function(
        image=image,
        cpu=REMOTE_DRIVER_CPU,
        memory=REMOTE_DRIVER_MEMORY_MB,
        timeout=REMOTE_DRIVER_TIMEOUT_SECONDS,
        volumes={str(REMOTE_RESULTS_ROOT): volume},
        serialized=True,
    )
    def evaluate_on_modal(job: dict[str, Any]) -> dict[str, Any]:
        import importlib.util

        spec = importlib.util.spec_from_file_location("lightning_weave_aime25_remote", "/root/eval_aime25.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load the AIME25 evaluator in Modal")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.run_remote_evaluation(
            job,
            results_root=module.REMOTE_RESULTS_ROOT,
            commit=volume.commit,
        )

    return app, evaluate_on_modal, volume


def build_eval_config(*, samples_per_problem: int, max_problems: int | None = None) -> Any:
    from modal_training_gym import EvalConfig

    kwargs = generation_kwargs()
    protocol = {
        "dataset": AIME25_REPO,
        "split": AIME25_SPLIT,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "samples_per_problem": samples_per_problem,
        "max_problems": max_problems,
    }
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    return EvalConfig(
        dataset=make_aime25_dataset(max_problems=max_problems),
        eval_fn=make_eval_function(
            samples_per_problem=samples_per_problem,
            generate_kwargs=kwargs,
        ),
        prompt_column="prompt",
        generate_kwargs=kwargs,
        eval_config_id=f"lightning-weave-aime25-{protocol_hash}",
    )


def build_deploy_recipe(environment: str) -> Any:
    from modal_training_gym import Qwen3_4B_VllmRecipe

    return Qwen3_4B_VllmRecipe(
        gpu="H100",
        n_gpu=1,
        environment_name=environment,
        extra_vllm_args=[
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--gpu-memory-utilization",
            "0.85",
            "--seed",
            "42",
        ],
    )


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


def stop_deployment(deployment: Any, environment: str) -> None:
    command = [
        sys.executable,
        "-m",
        "modal",
        "app",
        "stop",
        deployment.modal_app_id,
        "--yes",
        "--env",
        environment,
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(
            f"WARNING: could not stop deployment {deployment.modal_app_id}; stop it from the Modal dashboard.",
            file=sys.stderr,
        )


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
    if curve_path.exists():
        curve = json.loads(curve_path.read_text(encoding="utf-8"))
    else:
        curve = {"protocol": protocol, "results": []}
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
            key=lambda record: order.get(
                str(record["target"]),
                int(record.get("checkpoint_iteration") or 0),
            ),
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
    result.add_argument(
        "--max-concurrency",
        type=int,
        default=30,
        help="Concurrent AIME problems; each problem samples sequentially.",
    )
    result.add_argument(
        "--environment",
        default=os.environ.get("MODAL_ENVIRONMENT", "alex-dev-2"),
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/math/aime25/<run-id>.",
    )
    result.add_argument(
        "--no-base",
        action="store_true",
        help="Skip the untrained Qwen3-4B baseline.",
    )
    result.add_argument(
        "--keep-deployments",
        action="store_true",
        help="Leave inference deployments running after evaluation.",
    )
    result.add_argument(
        "--detach",
        action="store_true",
        help="Launch the Modal CPU evaluator and return without waiting.",
    )
    result.add_argument(
        "--collect",
        action="store_true",
        help="Download remote progress/results without launching GPUs.",
    )
    result.add_argument(
        "--force",
        action="store_true",
        help="Ignore a completed remote result and evaluate again.",
    )
    result.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Smoke-test only: evaluate the first N of 30 problems.",
    )
    result.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List committed checkpoints without allocating GPUs.",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the static protocol without contacting Modal.",
    )
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
        "driver": "Modal CPU",
        "driver_cpu": REMOTE_DRIVER_CPU,
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
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be at least 1")
    if args.max_problems is not None and not 1 <= args.max_problems <= 30:
        raise ValueError("--max-problems must be between 1 and 30")

    protocol = protocol_summary(args)
    print(json.dumps(protocol, indent=2))
    if args.dry_run:
        return

    from modal_training_gym import (
        CustomDeployment,
        Qwen3_4B,
        TrainingRun,
        convert_megatron_checkpoint_to_hf,
    )

    source_run = TrainingRun.from_id(args.run_id)
    checkpoints = select_checkpoints(list_checkpoints_with_retry(source_run), args.iterations)
    print(
        "Committed checkpoints:",
        ", ".join(f"{checkpoint.name} (step {checkpoint_iteration(checkpoint) + 1})" for checkpoint in checkpoints),
    )
    if args.list_checkpoints:
        return

    output_dir = args.output_dir or Path("results/math/aime25") / args.run_id
    output_dir = output_dir.resolve()
    targets: list[Any | None] = ([] if args.no_base else [None]) + checkpoints
    driver_app, evaluate_on_modal, result_volume = build_remote_driver(args.environment)
    failures: list[str] = []

    for checkpoint in targets:
        name = target_name(checkpoint)
        iteration = None if checkpoint is None else checkpoint_iteration(checkpoint)
        result_path = remote_result_path(run_id=args.run_id, target=name, protocol=protocol)

        existing = read_remote_payload(result_volume, result_path)
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
                print(f"{name}: {status} ({completed}/30 problems persisted)")
            continue

        deployment = None
        driver_started = False
        print(f"\nLaunching {name}...", flush=True)
        try:
            model = Qwen3_4B()
            deploy_recipe = build_deploy_recipe(args.environment)
            serving_checkpoint = checkpoint
            if checkpoint is not None:
                # Qwen3-4B is TP=1. Training Gym otherwise infers the training
                # run's eight-GPU allocation for this single-process export.
                one_gpu_checkpoint = dataclasses.replace(checkpoint, training_run_id="")
                serving_checkpoint = convert_megatron_checkpoint_to_hf(
                    one_gpu_checkpoint,
                    model,
                    recipe=deploy_recipe,
                )

            deployment = CustomDeployment.launch(
                model=model,
                checkpoint=serving_checkpoint,
                recipe=deploy_recipe,
                app_name=f"qwen3-4b-aime25-{name.replace('_', '-')}",
                served_model_name=f"qwen3-4b-aime25-{name.replace('_', '-')}",
                unauthenticated=True,
            )
            job = {
                "protocol": protocol,
                "target": name,
                "checkpoint_iteration": iteration,
                "endpoint_url": deployment.url,
                "served_model_name": deployment.served_model_name,
                "modal_app_id": deployment.modal_app_id,
                "result_path": result_path,
                "samples_per_problem": args.samples_per_problem,
                "max_problems": args.max_problems,
                "max_concurrency": min(args.max_concurrency, args.max_problems or 30),
                "generate_kwargs": generation_kwargs(),
                "stop_deployment": not args.keep_deployments,
                "force": args.force,
            }
            import modal

            with (
                modal.enable_output(),
                driver_app.run(
                    name=f"lw-aime25-driver-{name.replace('_', '-')}",
                    detach=True,
                    environment_name=args.environment,
                ),
            ):
                function_call = evaluate_on_modal.spawn(job)
                driver_started = True
                launch_record = {
                    "target": name,
                    "function_call_id": function_call.object_id,
                    "driver": "Modal CPU",
                    "serving_app_id": deployment.modal_app_id,
                    "remote_result_path": result_path,
                    "launched_at": datetime.datetime.now(datetime.UTC).isoformat(),
                }
                save_launch_record(output_dir, launch_record)
                print(
                    f"{name}: Modal CPU driver {function_call.object_id} launched; "
                    f"progress is durable at {EVAL_RESULTS_VOLUME}/{result_path}",
                    flush=True,
                )
                if args.detach:
                    continue
                function_call.get()

            payload = read_remote_payload(result_volume, result_path)
            if payload is None:
                raise RuntimeError(f"Modal driver completed without writing {result_path}")
            save_collected_result(output_dir=output_dir, protocol=protocol, payload=payload)
            if payload.get("status") != "completed":
                raise RuntimeError(
                    f"Remote evaluator status is {payload.get('status')}: {payload.get('error', 'no error recorded')}"
                )
            record = payload["checkpoint"]
            print(
                f"{name}: accuracy={record['accuracy']:.4f} "
                f"({record['problems']} problems x {args.samples_per_problem} samples)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - continue the checkpoint sweep
            failures.append(name)
            payload = read_remote_payload(result_volume, result_path)
            if payload is not None:
                save_collected_result(output_dir=output_dir, protocol=protocol, payload=payload)
            print(f"{name} failed: {exc!r}", file=sys.stderr, flush=True)
        finally:
            # Once the durable CPU call starts it owns endpoint cleanup. Stopping
            # it here on a laptop disconnect would recreate the original bug.
            if deployment is not None and not driver_started and not args.keep_deployments:
                stop_deployment(deployment, args.environment)

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
