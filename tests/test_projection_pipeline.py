"""The decision-projection stages of configs/agent_eff/modal_pipeline.py, run locally end to end on a tiny cache.

Every stage runs its real command, except the two GPU programs: vLLM sampling is stubbed inside the real collector,
and donor scoring writes synthetic scores. Training stops after its command is built, which is then checked against
the launcher and the sealed generation config, as slime checks it.
"""

from __future__ import annotations

import importlib.util
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("modal")

REPO = Path(__file__).resolve().parents[1]
PROMPTS = 6
RESPONSES = 8


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) % 200 + 1 for character in text[:6]]


def completion(prompt_index: int, response_index: int):
    """Reasoning of varying length; most responses of a prompt make the same call."""
    words = 2 + (prompt_index + 3 * response_index) % 7
    call = "f" if response_index % 4 else "g"
    text = "<think>" + " step" * words + "</think>\n\n" + f'<tool_call>\n{{"name": "{call}", "arguments": {{}}}}\n</tool_call>'
    tokens = list(range(10, 10 + words + 4))
    logprobs = [  # the sampled token plus 15 alternatives: the cache keeps 16 candidates per position
        {token: SimpleNamespace(logprob=-0.1), **{token + 100 * k: SimpleNamespace(logprob=-1.0 - k) for k in range(1, 16)}}
        for token in tokens
    ]
    return SimpleNamespace(token_ids=tokens, logprobs=logprobs, text=text, finish_reason="stop")


class StopTraining(Exception):
    pass


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("agent_eff_pipeline", REPO / "configs/agent_eff/modal_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    for name in ("data_volume", "model_volume", "checkpoint_volume"):
        monkeypatch.setattr(module, name, volume)
    for key in module.PROJECTION:
        monkeypatch.setitem(module.PROJECTION, key, str(tmp_path / key))
    monkeypatch.setattr(module.agent_eff, "ASSET_LOCK", str(tmp_path / "assets.json"))
    monkeypatch.setattr(module.agent_eff, "PROJECTION_CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setattr(module, "PROMPT_DATA", str(tmp_path / "prompts.parquet"))
    original_require = module.require_paths
    monkeypatch.setattr(module, "require_paths", lambda paths: original_require(
        [path for path in paths if str(path).startswith(str(tmp_path))]  # donor weights live on Modal volumes
    ))

    pq.write_table(pa.table({"prompt": [f"prompt {index}" for index in range(PROMPTS)], "label": [""] * PROMPTS,
                             "prompt_id": [f"p{index}" for index in range(PROMPTS)]}), tmp_path / "prompts.parquet")
    (tmp_path / "assets.json").write_text(json.dumps({
        "tokenizer_hash": "t" * 64,
        "token_id_compatibility": {"mode": "official_input_tokenizer_null", "model_vocab_size": 300,
                                   "normalization_vocab_size": 300},
        "models": {"student": {"path": "/base", "revision": "base", "model_vocab_size": 300, "tokenizer_hash": "t"}},
    }))
    recipient = tmp_path / "recipient_hf"
    recipient.mkdir()
    (recipient / module.PROVENANCE_FILE).write_text(json.dumps(
        {"variant": "acc-legacy", "seed": 1234, "target_revision": "r" * 64}
    ))

    commands = []

    def run(command, *, env=None):
        commands.append(command)
        script = command[1] if command[0] == "python" else None
        if script == "data_curation/collect_direct_opd_rollouts.py":
            class LLM:
                def __init__(self, **kwargs):
                    pass

                def generate(self, prompts, sampling, **kwargs):
                    assert sampling.top_k == 20 and sampling.n == RESPONSES
                    return [SimpleNamespace(outputs=[completion(int(str(row["prompt_token_ids"][0])), j)
                                                     for j in range(sampling.n)]) for row in prompts]

            monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="test", LLM=LLM,
                                                                     SamplingParams=SimpleNamespace))
            monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(CompilationConfig=SimpleNamespace(mode=None)))
            monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(
                    encode=lambda text, add_special_tokens=False: [int(text.split()[-1])]))
            ))
            monkeypatch.setattr(sys, "argv", command[1:])
            runpy.run_path(str(REPO / script), run_name="__main__")
        elif script == "data_curation/score_reasoning.py":
            # Synthetic donor scores that favor short reasoning.
            output = Path(command[command.index("--output") + 1])
            rollouts = Path(command[command.index("--rollouts") + 1])
            rows = [row for path in sorted(rollouts.glob("*.parquet")) for row in pq.read_table(path).to_pylist()]
            output.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({"sample_id": [row["metadata"]["sample_id"] for row in rows],
                                     "score": [-0.5 * row["metadata"]["response_length"] for row in rows]}), output)
        elif script == "configs/lightning_weave/train.py":
            raise StopTraining
        elif script == "evaluation/projection_probe.py" and command[2] == "sample":
            from data_curation.decision_projection import decision_key
            from evaluation import projection_probe as probe_module

            model = command[command.index("--model") + 1]
            output = Path(command[command.index("--output") + 1])
            probe_dir = Path(command[command.index("--probe") + 1])
            moved = "ordinary" in model
            visible = "Sure." if moved else '<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'
            tokens = 100 if "recipient" in model else 90
            rows = [{"set": item["set"], "prompt_id": item["prompt_id"], "responses": [
                {"tokens": tokens, "finish_reason": "stop", "decision": decision_key(visible, "stop"),
                 **({"verdict": "match"} if item["set"] == "heldout" else {})} for _ in range(8)]}
                for item in probe_module.read_jsonl(probe_dir / "prompts.jsonl")]
            probe_module.write_jsonl(rows, output)
            module.probe_seeds.append(int(command[command.index("--seed") + 1]))
        elif script == "evaluation/projection_probe.py" and command[2] == "fit":
            from evaluation import projection_probe as probe_module

            output = Path(command[command.index("--output") + 1])
            probe_dir = Path(command[command.index("--probe") + 1])
            module.probe_fits.append(output.name)
            probe_module.write_jsonl([{"prompt_id": r["prompt_id"], "sample_id": r["sample_id"], "logprob": -10.0}
                                      for r in probe_module.read_jsonl(probe_dir / "fit_rows.jsonl")], output)
        elif script == "evaluation/decision_drift.py" and command[2] == "sample":
            from evaluation import decision_drift

            model = command[command.index("--model") + 1]
            histories = decision_drift.read_jsonl(Path(command[command.index("--histories") + 1]))
            moved = "ordinary" in model  # this arm's decisions move; the others keep the recipient's
            decision = '{"visible":"other","finish_reason":"stop"}' if moved else '{"visible":"same","finish_reason":"stop"}'
            responses = int(command[command.index("--responses") + 1])
            module.drift_seed_blocks.append(range(int(command[command.index("--seed") + 1]),
                                                  int(command[command.index("--seed") + 1]) + responses))
            decision_drift.write_jsonl([{"history_id": item["history_id"], "kind": item["kind"], "lane": item["lane"],
                                         "decisions": [decision] * responses, "tokens": [5] * responses}
                                        for item in histories],
                                       Path(command[command.index("--output") + 1]))
        else:
            subprocess.run([sys.executable, *command[1:]], check=True, cwd=REPO, env=env)

    monkeypatch.setattr(module, "run", run)
    module.commands = commands
    module.drift_seed_blocks = []
    module.probe_seeds = []
    module.probe_fits = []
    return module


def test_stages_from_sampling_to_the_training_command(pipeline, tmp_path, monkeypatch):
    assert pipeline.projection_sample_shard.local(0, 1) == "rank 0: complete"
    assert pipeline.projection_sample_shard.local(0, 1) == "rank 0: existing"  # a finished rank is reused
    sample_command = pipeline.commands[0]
    for flag, value in (("--temperature", "0.6"), ("--top-p", "0.95"), ("--sampling-top-k", "20"),
                        ("--responses-per-prompt", str(RESPONSES)), ("--top-k", "16")):
        assert sample_command[sample_command.index(flag) + 1] == value
    assert sample_command[sample_command.index("--model-revision") + 1] == "agent-eff/acc-legacy/seed1234/" + "r" * 64

    assert pipeline.projection_score_shard.local(0, 1) == "rank 0: complete"
    score_command = pipeline.commands[-1]
    decs = pipeline.DONORS["decs"]
    assert score_command[score_command.index("--post-revision") + 1] == decs.post.revision
    assert score_command[score_command.index("--pre-revision") + 1] == decs.pre.revision

    diagnostics = pipeline.projection_weights.local(1)
    assert diagnostics["calibration"]["implied_savings"] == pytest.approx(0.15, abs=1e-6)
    assert diagnostics["weights"]["projected_group_mass_max_error"] < 1e-9
    targets = [pipeline.projection_seal.local(arm) for arm in pipeline.agent_eff.PROJECTION_ARMS]
    from slime.rollout.offline_direct_opd import validate_sealed_manifest

    manifests = [validate_sealed_manifest(Path(target) / "manifest.json", expected_top_k=16) for target in targets]
    rows = manifests[0]["total_rows"]
    assert rows == PROMPTS * RESPONSES and len({m["post_teacher_model"]["revision"] for m in manifests}) == 3

    # Training: the sealed rows fill whole batches here, and the command must pass slime's generation-config lock.
    monkeypatch.setitem(pipeline.agent_eff.PROJECTION_TRAINING, "rollout_batch_size", rows)
    monkeypatch.setitem(pipeline.agent_eff.PROJECTION_TRAINING, "global_batch_size", RESPONSES)
    initial = Path(pipeline.PROJECTION["recipient_megatron"])
    initial.mkdir()
    (initial / "modal_conversion.json").write_text("{}")
    with pytest.raises(StopTraining):
        pipeline.projection_train.local("projected", 1234)
    command = pipeline.commands[-1]
    sys.path.insert(0, str(REPO / "configs/lightning_weave"))
    try:
        launcher = importlib.import_module("train")
    finally:
        sys.path.pop(0)
    launch = launcher.parser().parse_args(command[2:])
    options = launcher.build_train_args(launch)
    assert "--calculate-per-token-loss" not in options
    assert options[options.index("--offline-direct-opd-loss-mode") + 1] == "sequence_weighted"
    assert options[options.index("--hf-checkpoint") + 1] == str((tmp_path / "recipient_hf").resolve())
    generation = manifests[2]["generation_config"]
    expected = {"temperature": launch.temperature, "top_p": launch.top_p, "top_k": launch.top_k,
                "max_response_length": launch.max_response_length, "max_prompt_length": launch.max_prompt_length,
                "responses_per_prompt": launch.responses_per_prompt}
    assert {key: generation[key] for key in expected} == expected
    assert manifests[2]["student_model"]["path"] == launch.student


def test_drift_stages_replay_one_logged_run_to_every_model(pipeline, tmp_path, monkeypatch):
    import gzip

    from configs.bfcl_eval import config as bfcl

    monkeypatch.setattr(pipeline, "BFCL_RESULTS", str(tmp_path / "results"))
    requests = tmp_path / "results" / "run-id" / "ae.joint.acc-legacy" / bfcl.REQUEST_LOG_DIR
    requests.mkdir(parents=True)
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
    seen: set[str] = set()
    with gzip.open(requests / "start.jsonl.gz", "wt", encoding="utf-8") as handle:
        for index in range(3):
            body = {"messages": [{"role": "user", "content": f"task {index}"}], "tools": tools}
            record = bfcl.request_log_record("multi_turn_base", body, body, {"choices": []}, seen)
            handle.write(json.dumps(record) + "\n")
    frozen = pipeline.projection_drift_histories.local("run-id/ae.joint.acc-legacy")
    assert frozen["histories"] == 3 and len(frozen["sha256"]) == 64
    jobs = pipeline.drift_jobs([1234])
    assert [label for label, _, _ in jobs[:2]] == ["recipient", "recipient-null"]
    for label, model, seed in jobs:
        Path(model).mkdir(parents=True, exist_ok=True)
        (Path(model) / "config.json").write_text("{}")
        pipeline.projection_drift_sample.local(label, model, seed)
    # Two recipient draws and the arms never share a per-sample seed; the arms share one block.
    reference, null, *arms = pipeline.drift_seed_blocks
    assert not set(reference) & set(null) and not (set(reference) | set(null)) & set(arms[0])
    assert all(block == arms[0] for block in arms)
    report = pipeline.projection_drift_compare.local([label for label, _, _ in jobs[2:]])
    exact = report["views"]["exact"]["arms"]
    assert report["histories"] == 3 and report["histories_frozen"]["sha256"] == frozen["sha256"]
    assert exact["projected-seed1234"]["all"]["excess_over_null"] == pytest.approx(0.0)
    assert exact["ordinary-seed1234"]["all"]["excess_over_null"] == pytest.approx(1.0)


def test_weights_fall_back_to_the_lower_target_when_the_first_is_out_of_reach(pipeline, monkeypatch):
    pipeline.projection_sample_shard.local(0, 1)
    pipeline.projection_score_shard.local(0, 1)
    monkeypatch.setattr(pipeline.agent_eff, "PROJECTION_TARGET_SAVINGS", 0.99)
    diagnostics = pipeline.projection_weights.local(1)
    plan = diagnostics["target_plan"]
    assert [attempt["reached"] for attempt in plan["attempts"]] == [False, True]
    assert plan["fallback"] == pipeline.agent_eff.PROJECTION_FALLBACK_TARGET_SAVINGS
    assert diagnostics["calibration"]["implied_savings"] == pytest.approx(plan["fallback"], abs=1e-6)
    failed = json.loads(Path(plan["attempts"][0]["diagnostics"]).read_text())
    assert "error" in failed["calibration"]  # the failed attempt's diagnostics are kept


def test_probe_stages_sample_every_model_and_compare(pipeline, tmp_path, monkeypatch):
    pipeline.projection_sample_shard.local(0, 1)
    pipeline.projection_score_shard.local(0, 1)
    pipeline.projection_weights.local(1)
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text("".join(json.dumps({"prompt_id": f"held-{i}", "prompt_token_ids": [7, i],
                                           "target": {"tool_calls": [{"function": {"name": "f", "arguments": {}}}]}})
                               + "\n" for i in range(4)))
    monkeypatch.setitem(pipeline.PROBE, "heldout_prompts", str(heldout))
    monkeypatch.setitem(pipeline.PROBE, "train_prompts", 4)
    summary = pipeline.projection_probe_prompts.local()
    assert summary == {"train": 4, "heldout": 4, "fit_rows": 4 * RESPONSES}
    jobs = pipeline.probe_jobs([1234])
    assert [job[0] for job in jobs[:2]] == ["recipient", "recipient-null"] and jobs[1][3] is False
    for label, model, seed, teacher_force in jobs:
        Path(model).mkdir(parents=True, exist_ok=True)
        (Path(model) / "config.json").write_text("{}")
        pipeline.projection_probe_sample.local(label, model, seed, teacher_force)
    assert len(set(pipeline.probe_seeds[:2])) == 2 and len(set(pipeline.probe_seeds[2:])) == 1
    assert "recipient-null.fit.jsonl" not in pipeline.probe_fits and len(pipeline.probe_fits) == len(jobs) - 1
    # A rerun after a teacher-forcing failure keeps the samples and redoes only the teacher forcing.
    label, model, seed, teacher_force = jobs[2]
    (Path(pipeline.PROJECTION["probe"]) / "samples" / f"{label}.fit.jsonl").unlink()
    sampled = len(pipeline.probe_seeds)
    pipeline.projection_probe_sample.local(label, model, seed, teacher_force)
    assert len(pipeline.probe_seeds) == sampled and pipeline.probe_fits[-1] == f"{label}.fit.jsonl"
    report = pipeline.projection_probe_compare.local()
    assert set(report["students"]) == {f"{arm}-seed1234" for arm in pipeline.agent_eff.PROJECTION_ARMS}
    projected, ordinary = report["students"]["projected-seed1234"], report["students"]["ordinary-seed1234"]
    assert projected["savings_train"]["point"] == pytest.approx(0.1)
    assert ordinary["squared_distance_calls_heldout"]["detectable_movement"]
    assert projected["realization"]["reading"] in ("substantially realized", "weakly realized", "inconclusive")
    assert "slope_on_log_weight" in projected["fit"]
