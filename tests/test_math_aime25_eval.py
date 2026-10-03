import json
import sys
import types
from pathlib import Path

from configs.math_grpo import eval_aime25 as MODULE

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_matches_repo_aime25_with_requested_response_cap(fake_training_gym):
    args = MODULE.parser().parse_args([])
    recipe = MODULE.build_conversion_recipe("alex-dev-2")

    assert MODULE.AIME25_REPO == "math-ai/aime25"
    assert MODULE.AIME25_SPLIT == "test"
    assert MODULE.format_aime25_prompt("  2 + 2?  ") == "Question: 2 + 2?\nAnswer:"
    assert MODULE.generation_kwargs() == {
        "max_tokens": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "stop": ["Question:", "</s>", "<|im_end|>", "<|eot_id|>"],
    }
    assert recipe.gpu == "H100"
    assert recipe.n_gpu == 1
    assert recipe.environment_name == "alex-dev-2"
    assert args.samples_per_problem == 8
    assert args.no_base is False


def test_dataset_validates_30_rows_and_formats_prompts(monkeypatch):
    class FakeDataset(list):
        def select(self, indices):
            return FakeDataset(self[index] for index in indices)

    rows = FakeDataset({"problem": f" Problem {index} ", "answer": index} for index in range(30))
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda repo, split: rows
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    materialized = MODULE.load_aime25_rows(max_problems=2)
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


def test_problem_result_averages_repeats_and_keeps_generation_metadata(monkeypatch):
    monkeypatch.setattr(
        MODULE,
        "score_aime25_response",
        lambda answer, response: float(answer == response),
    )
    output = types.SimpleNamespace(
        outputs=[
            types.SimpleNamespace(index=0, text="7", finish_reason="stop", token_ids=[1, 2]),
            types.SimpleNamespace(index=1, text="8", finish_reason="length", token_ids=[3, 4, 5]),
        ]
    )
    result = MODULE.build_problem_result(
        {"prompt": "Question: x\nAnswer:", "answer": "7", "problem_index": 3},
        output,
        samples_per_problem=2,
    )

    assert result["score"] == 0.5
    assert result["metadata"]["answer"] == "7"
    assert result["metadata"]["samples"] == [
        {
            "sample_index": 0,
            "score": 1.0,
            "response": "7",
            "finish_reason": "stop",
            "completion_tokens": 2,
        },
        {
            "sample_index": 1,
            "score": 0.0,
            "response": "8",
            "finish_reason": "length",
            "completion_tokens": 3,
        },
    ]


def test_vllm_batch_submits_all_problems_together(monkeypatch):
    calls = {}

    class SamplingParams:
        def __init__(self, **kwargs):
            calls["sampling"] = kwargs

    class LLM:
        def __init__(self, **kwargs):
            calls["engine"] = kwargs

        def chat(self, **kwargs):
            calls["chat"] = kwargs
            return ["first", "second"]

    vllm = types.ModuleType("vllm")
    vllm.LLM = LLM
    vllm.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)

    outputs = MODULE.run_vllm_batch(
        model_path="/checkpoints/model",
        rows=[{"prompt": "one"}, {"prompt": "two"}],
        samples_per_problem=8,
    )

    assert outputs == ["first", "second"]
    assert calls["sampling"]["n"] == 8
    assert calls["sampling"]["max_tokens"] == 32768
    assert calls["chat"]["messages"] == [
        [{"role": "user", "content": "one"}],
        [{"role": "user", "content": "two"}],
    ]


def test_inprocess_evaluator_scores_and_persists_one_batch(tmp_path, monkeypatch):
    rows = [
        {"prompt": "Question: one\nAnswer:", "answer": "1", "problem_index": 0},
        {"prompt": "Question: two\nAnswer:", "answer": "2", "problem_index": 1},
    ]
    outputs = [
        types.SimpleNamespace(
            outputs=[types.SimpleNamespace(index=0, text="1", finish_reason="stop", token_ids=[1])]
        ),
        types.SimpleNamespace(
            outputs=[types.SimpleNamespace(index=0, text="0", finish_reason="stop", token_ids=[2])]
        ),
    ]
    calls = []
    monkeypatch.setattr(MODULE, "load_aime25_rows", lambda max_problems: rows)
    monkeypatch.setattr(
        MODULE,
        "run_vllm_batch",
        lambda **kwargs: calls.append(kwargs) or outputs,
    )
    monkeypatch.setattr(
        MODULE,
        "score_aime25_response",
        lambda answer, response: float(answer == response),
    )

    result = MODULE.run_inprocess_evaluation(
        {
            "protocol": {"name": "test"},
            "target": "iter_0000019",
            "checkpoint_iteration": 19,
            "model_path": "/checkpoints/model",
            "result_path": "run/protocol/iter_0000019.json",
            "samples_per_problem": 1,
            "max_problems": 2,
            "force": False,
        },
        results_root=tmp_path,
        commit=lambda: None,
    )

    assert len(calls) == 1
    assert calls[0]["rows"] == rows
    assert result["checkpoint"]["accuracy"] == 0.5
    payload = json.loads((tmp_path / "run/protocol/iter_0000019.json").read_text())
    assert payload["status"] == "completed"
    assert len(payload["rows"]) == 2


def test_launcher_is_pinned_and_uses_inprocess_modal_gpu():
    source = Path(MODULE.__file__).read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts/eval_math_aime25.sh").read_text(encoding="utf-8")

    assert MODULE.TRAINING_GYM_COMMIT in source
    assert '# requires-python = "==3.12.*"' in source
    assert "--python 3.12 --script" in wrapper
    assert "convert_megatron_checkpoint_to_hf" in source
    assert "from vllm import LLM, SamplingParams" in source
    assert "llm.chat(" in source
    assert "gpu=EVAL_GPU" in source
    assert "evaluate_on_gpu.spawn(job)" in source
    assert "detach=True" in source
    assert "CustomDeployment.launch" not in source
    assert "requests.post" not in source
    assert MODULE.EVAL_GPU == "H100"
    assert MODULE.EVAL_TIMEOUT_SECONDS == 24 * 60 * 60


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
