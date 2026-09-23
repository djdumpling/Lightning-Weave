import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "configs/math_grpo/eval_aime25.py"
SPEC = importlib.util.spec_from_file_location("math_grpo_eval_aime25", EVAL_SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeConfig:
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            setattr(self, key, value)


def install_fake_training_gym(monkeypatch):
    module = types.ModuleType("modal_training_gym")
    for name in (
        "DatasetConfig",
        "EvalConfig",
        "EvalRowResult",
        "Qwen3_4B_VllmRecipe",
    ):
        setattr(module, name, type(name, (FakeConfig,), {}))
    monkeypatch.setitem(sys.modules, "modal_training_gym", module)
    return module


def test_protocol_matches_repo_aime25_with_requested_response_cap(monkeypatch):
    install_fake_training_gym(monkeypatch)
    args = MODULE.parser().parse_args([])
    config = MODULE.build_eval_config(samples_per_problem=64)
    recipe = MODULE.build_deploy_recipe("alex-dev-2")

    assert MODULE.AIME25_REPO == "math-ai/aime25"
    assert MODULE.AIME25_SPLIT == "test"
    assert MODULE.format_aime25_prompt("  2 + 2?  ") == "Question: 2 + 2?\nAnswer:"
    assert config.prompt_column == "prompt"
    assert config.generate_kwargs == {
        "max_tokens": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "stop": ["Question:", "</s>", "<|im_end|>", "<|eot_id|>"],
    }
    assert recipe.gpu == "H100"
    assert recipe.n_gpu == 1
    assert recipe.environment_name == "alex-dev-2"
    assert recipe.extra_vllm_args == [
        "--max-model-len",
        "33792",
        "--gpu-memory-utilization",
        "0.85",
        "--seed",
        "42",
    ]
    assert args.samples_per_problem == 8
    assert args.max_concurrency == 30
    assert args.no_base is False


def test_dataset_validates_30_rows_and_formats_prompts(monkeypatch):
    install_fake_training_gym(monkeypatch)

    class FakeDataset(list):
        def select(self, indices):
            return FakeDataset(self[index] for index in indices)

    rows = FakeDataset({"problem": f" Problem {index} ", "answer": index} for index in range(30))
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda repo, split: rows
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    dataset = MODULE.make_aime25_dataset(max_problems=2)
    materialized = list(dataset.rows())

    assert dataset.input_key() == "prompt"
    assert dataset.label_key() == "answer"
    assert materialized == [
        {
            "prompt": "Question: Problem 0\nAnswer:",
            "answer": "0",
            "problem_index": 0,
        },
        {
            "prompt": "Question: Problem 1\nAnswer:",
            "answer": "1",
            "problem_index": 1,
        },
    ]


def test_math_verify_scoring_disables_thread_unsafe_timeouts(monkeypatch):
    calls = []
    math_verify = types.ModuleType("math_verify")

    def parse(value, *, parsing_timeout):
        calls.append(("parse", parsing_timeout))
        return [value.replace("$", "").strip()]

    def verify(target, prediction, *, timeout_seconds):
        calls.append(("verify", timeout_seconds))
        return target == prediction

    math_verify.parse = parse
    math_verify.verify = verify
    monkeypatch.setitem(sys.modules, "math_verify", math_verify)

    assert MODULE.score_aime25_response("42", "42") == 1.0
    assert calls == [
        ("parse", None),
        ("parse", None),
        ("verify", None),
    ]


def test_checkpoint_selection_uses_exact_iteration_not_latest():
    checkpoints = [
        types.SimpleNamespace(name="iter_0000079"),
        types.SimpleNamespace(name="iter_0000019"),
        types.SimpleNamespace(name="iter_0000059"),
        types.SimpleNamespace(name="iter_0000039"),
    ]

    assert [checkpoint.name for checkpoint in MODULE.select_checkpoints(checkpoints, "19,59")] == [
        "iter_0000019",
        "iter_0000059",
    ]
    assert [checkpoint.name for checkpoint in MODULE.select_checkpoints(checkpoints, "all")] == [
        "iter_0000019",
        "iter_0000039",
        "iter_0000059",
        "iter_0000079",
    ]


def test_checkpoint_listing_retries_modal_rate_limit(monkeypatch):
    calls = 0

    class SourceRun:
        def checkpoints(self):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("VolumeListFiles rate limit exceeded")
            return ["checkpoint"]

    delays = []
    monkeypatch.setattr(MODULE.time, "sleep", delays.append)

    assert MODULE.list_checkpoints_with_retry(SourceRun()) == ["checkpoint"]
    assert calls == 3
    assert delays == [2, 4]


def test_eval_function_averages_repeats_and_keeps_raw_samples(monkeypatch):
    install_fake_training_gym(monkeypatch)

    responses = iter(["7", "8"])
    deployment = types.SimpleNamespace(generate=lambda prompt, **kwargs: next(responses))
    monkeypatch.setattr(
        MODULE,
        "score_aime25_response",
        lambda answer, response: float(answer == response),
    )
    evaluate = MODULE.make_eval_function(
        samples_per_problem=2,
        generate_kwargs=MODULE.generation_kwargs(),
    )

    result = evaluate(
        deployment,
        {"prompt": "Question: x\nAnswer:", "answer": "7", "problem_index": 3},
    )

    assert result.score == 0.5
    assert result.metadata["answer"] == "7"
    assert result.metadata["samples"] == [
        {"sample_index": 0, "score": 1.0, "response": "7"},
        {"sample_index": 1, "score": 0.0, "response": "8"},
    ]


def test_launcher_is_pinned_and_uses_modal_cpu_driver():
    source = EVAL_SCRIPT.read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts/eval_math_aime25.sh").read_text(encoding="utf-8")

    assert MODULE.TRAINING_GYM_COMMIT in source
    assert '# requires-python = "==3.12.*"' in source
    assert "uv run --python 3.12" in wrapper
    assert "convert_megatron_checkpoint_to_hf" in source
    assert "CustomDeployment.launch" in source
    assert 'modal.App("lightning-weave-aime25-driver")' in source
    assert "cpu=REMOTE_DRIVER_CPU" in source
    assert "evaluate_on_modal.spawn(job)" in source
    assert "detach=True" in source
    assert MODULE.REMOTE_DRIVER_CPU == 4.0
    assert MODULE.REMOTE_DRIVER_TIMEOUT_SECONDS == 24 * 60 * 60


def test_remote_result_key_ignores_launch_selection():
    first = MODULE.protocol_summary(MODULE.parser().parse_args(["--iterations", "19"]))
    second = MODULE.protocol_summary(MODULE.parser().parse_args(["--iterations", "all", "--no-base"]))

    assert MODULE.remote_result_path(run_id="run", target="iter_0000019", protocol=first) == MODULE.remote_result_path(
        run_id="run", target="iter_0000019", protocol=second
    )


def test_collected_curve_does_not_mix_smoke_and_full_protocols(tmp_path):
    smoke_protocol = {"samples_per_problem": 1, "max_problems": 1}
    full_protocol = {"samples_per_problem": 64, "max_problems": 30}
    MODULE.save_collected_result(
        output_dir=tmp_path,
        protocol=smoke_protocol,
        payload={
            "target": "iter_0000019",
            "status": "completed",
            "checkpoint": {"target": "iter_0000019", "accuracy": 1.0},
            "rows": [{}],
        },
    )
    MODULE.save_collected_result(
        output_dir=tmp_path,
        protocol=full_protocol,
        payload={
            "target": "iter_0000039",
            "status": "completed",
            "checkpoint": {"target": "iter_0000039", "accuracy": 0.5},
            "rows": [{}] * 30,
        },
    )

    curve = json.loads((tmp_path / "curve.json").read_text())
    assert curve["protocol"] == full_protocol
    assert [record["target"] for record in curve["results"]] == ["iter_0000039"]


def test_endpoint_problem_evaluator_keeps_generation_metadata(monkeypatch):
    generations = iter(
        [
            {"response": "7", "finish_reason": "stop", "completion_tokens": 5},
            {"response": "8", "finish_reason": "length", "completion_tokens": 9},
        ]
    )
    monkeypatch.setattr(MODULE, "generate_from_endpoint", lambda **kwargs: next(generations))
    monkeypatch.setattr(
        MODULE,
        "score_aime25_response",
        lambda answer, response: float(answer == response),
    )

    result = MODULE.evaluate_problem_from_endpoint(
        endpoint_url="https://example.invalid",
        served_model_name="qwen",
        example={
            "prompt": "Question: x\nAnswer:",
            "answer": "7",
            "problem_index": 0,
        },
        samples_per_problem=2,
        generate_kwargs=MODULE.generation_kwargs(),
    )

    assert result["score"] == 0.5
    assert result["metadata"]["samples"][0]["finish_reason"] == "stop"
    assert result["metadata"]["samples"][1]["completion_tokens"] == 9
