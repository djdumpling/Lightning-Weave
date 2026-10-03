import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_curation import decision_projection as projection
from evaluation import projection_probe as probe

SETTINGS = {"thresholds": {"substantial": 0.7, "weak": 0.4}, "equivalence_fraction": 0.5}


def call(name):
    return f'<tool_call>\n{{"name": "{name}", "arguments": {{}}}}\n</tool_call>'


def test_the_squared_distance_estimate_is_unbiased_and_cannot_cancel():
    assert probe.unbiased_squared_distance(["a"] * 8, ["a"] * 8) == 0.0
    assert probe.unbiased_squared_distance(["a"] * 8, ["b"] * 8) == pytest.approx(2.0)
    rng = np.random.default_rng(0)
    p, q = np.array([0.5, 0.5]), np.array([0.6, 0.4])
    estimates = [
        probe.unbiased_squared_distance(list(rng.choice(["x", "y"], 8, p=p)), list(rng.choice(["x", "y"], 8, p=q)))
        for _ in range(40_000)
    ]
    assert np.mean(estimates) == pytest.approx(((p - q) ** 2).sum(), abs=0.002)  # 0.02, not drowned by a floor
    same = [probe.unbiased_squared_distance(list(rng.choice(["x", "y"], 8, p=p)), list(rng.choice(["x", "y"], 8, p=p)))
            for _ in range(40_000)]
    assert np.mean(same) == pytest.approx(0.0, abs=0.002)
    with pytest.raises(ValueError):
        probe.unbiased_squared_distance(["a"], ["a", "b"])


def test_target_distance_and_readings():
    assert probe.target_squared_distance(["a", "a", "b", "b"], [1, 1, 1, 1]) == 0.0
    # Weight moves from decision b to a: q = (0.75, 0.25) against p̂ = (0.5, 0.5).
    assert probe.target_squared_distance(["a", "a", "b", "b"], [1.5, 1.5, 0.5, 0.5]) == pytest.approx(0.125)
    names = ("high", "low", "inconclusive")
    assert probe.reading([0.75, 0.9], high=0.7, low=0.4, names=names) == "high"
    assert probe.reading([0.1, 0.35], high=0.7, low=0.4, names=names) == "low"
    assert probe.reading([0.3, 0.8], high=0.7, low=0.4, names=names) == "inconclusive"


def write_rollouts(path, prompts=3, responses=4):
    rows = []
    for p in range(prompts):
        for r in range(responses):
            tokens = list(range(5 + 3 * r))
            name = "f" if r < 3 else "g"
            rows.append({"prompt": f"p{p}", "label": "", "metadata": {
                "prompt_id": f"p{p}", "sample_id": f"p{p}-{r}", "prompt_tokens": [1, 2, p],
                "response_tokens": tokens, "response": "<think>" + "x" * len(tokens) + "</think>\n\n" + call(name),
                "finish_reason": "stop", "response_length": len(tokens)}})
    pq.write_table(pa.Table.from_pylist(rows), path / "rollouts.parquet")
    return rows


@pytest.fixture
def probe_dir(tmp_path):
    rows = write_rollouts(tmp_path)
    pq.write_table(pa.table({"sample_id": [r["metadata"]["sample_id"] for r in rows],
                             "score": [-float(r["metadata"]["response_length"]) for r in rows]}),
                   tmp_path / "scores.parquet")
    projection.main(["weights", "--rollouts", str(tmp_path / "rollouts.parquet"), "--scores",
                     str(tmp_path / "scores.parquet"), "--alpha", "2.0", "--output", str(tmp_path / "weights.parquet")])
    heldout = [{"prompt_id": f"h{i}", "prompt_token_ids": [9, i], "conversation_kind": "single_turn",
                "target": {"tool_calls": [{"function": {"name": "f", "arguments": {}}}]}} for i in range(3)]
    (tmp_path / "heldout.jsonl").write_text("".join(json.dumps(row) + "\n" for row in heldout))
    summary = probe.build_prompts(tmp_path / "rollouts.parquet", tmp_path / "weights.parquet", tmp_path / "heldout.jsonl",
                                  tmp_path / "probe", train_prompts=2)
    assert summary == {"train": 2, "heldout": 3, "fit_rows": 8}
    return tmp_path


def test_prompts_record_the_chosen_targets_and_refuse_overlap(probe_dir, tmp_path):
    targets = json.loads((probe_dir / "probe/targets.json").read_text())
    assert len(targets) == 2 and all(len(t["weights"]["projected"]) == 4 for t in targets.values())
    first = next(iter(targets.values()))
    # The projected target keeps each decision group's total weight; the ordinary one need not.
    calls = [json.loads(d)["visible"].count('"g"') for d in first["decisions"]]
    assert sum(w for w, c in zip(first["weights"]["projected"], calls) if c) == pytest.approx(1.0)
    (tmp_path / "bad.jsonl").write_text(json.dumps({"prompt_id": "p0", "prompt_token_ids": [1]}) + "\n")
    with pytest.raises(ValueError, match="training prompt"):
        probe.build_prompts(tmp_path / "rollouts.parquet", tmp_path / "weights.parquet", tmp_path / "bad.jsonl",
                            tmp_path / "other", train_prompts=1)


def test_sampling_and_teacher_forcing_with_a_stub_engine(probe_dir):
    prompts = probe.read_jsonl(probe_dir / "probe/prompts.jsonl")

    class LLM:
        def generate(self, requests, sampling, use_tqdm=False):
            if getattr(sampling, "prompt_logprobs", None) == 0:
                return [SimpleNamespace(prompt_logprobs=[None] + [{token: SimpleNamespace(logprob=-0.5)}
                                                                  for token in r["prompt_token_ids"][1:]])
                        for r in requests]
            return [SimpleNamespace(outputs=[SimpleNamespace(text="<think>a</think>\n\n" + call("f"), finish_reason="stop",
                                                             token_ids=[1, 2, 3])] * sampling.n) for _ in requests]

    rows = probe.sample_rows(LLM(), SimpleNamespace(n=2), prompts)
    held = [row for row in rows if row["set"] == "heldout"]
    assert held and all(r["verdict"] == "match" for row in held for r in row["responses"])
    fit = probe.teacher_force(LLM(), SimpleNamespace(prompt_logprobs=0), probe.read_jsonl(probe_dir / "probe/fit_rows.jsonl"))
    first = probe.read_jsonl(probe_dir / "probe/fit_rows.jsonl")[0]
    assert fit[0]["logprob"] == pytest.approx(-0.5 * len(first["response_tokens"]))


def write_samples(directory, name, prompts, tokens, decision_for, fit=None):
    rows = []
    for prompt in prompts:
        responses = [{"tokens": tokens(prompt, k), "finish_reason": "stop",
                      "decision": projection.decision_key(decision_for(prompt, k), "stop"),
                      **({"verdict": "match"} if prompt["set"] == "heldout" else {})} for k in range(8)]
        rows.append({"set": prompt["set"], "prompt_id": prompt["prompt_id"], "responses": responses})
    probe.write_jsonl(rows, directory / f"{name}.jsonl")
    if fit is not None:
        probe.write_jsonl(fit, directory / f"{name}.fit.jsonl")


def test_compare_reads_savings_realization_and_decision_movement(probe_dir):
    prompts = probe.read_jsonl(probe_dir / "probe/prompts.jsonl")
    fit_rows = probe.read_jsonl(probe_dir / "probe/fit_rows.jsonl")
    targets = json.loads((probe_dir / "probe/targets.json").read_text())
    samples = probe_dir / "samples"
    samples.mkdir()

    def same(prompt, k):  # one decision throughout, so every estimate between such draws is exactly 0
        return call("f")

    base_fit = [{"prompt_id": r["prompt_id"], "sample_id": r["sample_id"], "logprob": -10.0} for r in fit_rows]
    write_samples(samples, "recipient", prompts, lambda p, k: 100, same, fit=base_fit)
    write_samples(samples, "recipient-null", prompts, lambda p, k: 100, same)
    # A student that keeps the recipient's decisions and realizes all of its target's savings, on both prompt sets.
    saving = 1 - sum(sum(w * t for w, t in zip(t["weights"]["projected"], t["tokens"])) for t in targets.values()) / sum(
        sum(t["tokens"]) for t in targets.values())
    weight = {r["sample_id"]: w for p, t in targets.items() for r, w in
              zip([r for r in fit_rows if r["prompt_id"] == p], t["weights"]["projected"])}
    student_fit = [{**row, "logprob": -10.0 + np.log(weight[row["sample_id"]])} for row in base_fit]
    write_samples(samples, "projected-seed1234", prompts, lambda p, k: 100 * (1 - saving), same, fit=student_fit)
    # A student whose every decision changes and that saves nothing.
    write_samples(samples, "ordinary-seed1234", prompts, lambda p, k: 100, lambda p, k: "Sure.")
    report = probe.compare(probe_dir / "probe", samples, SETTINGS, draws=200)
    projected, ordinary = report["students"]["projected-seed1234"], report["students"]["ordinary-seed1234"]
    assert report["null"]["train"]["calls"] == pytest.approx(0.0)
    assert projected["realization"]["point"] == pytest.approx(1.0)
    assert projected["generalization"]["point"] == pytest.approx(1.0)
    assert report["equivalence_bound_calls"] > 0
    movement = projected["squared_distance_calls_train"]
    assert movement["within_margin"] and not movement["detectable_movement"]
    assert projected["fit"]["slope_on_log_weight"] == pytest.approx(1.0)
    assert ordinary["squared_distance_calls_heldout"]["detectable_movement"]
    assert not ordinary["squared_distance_calls_heldout"]["within_margin"]
    assert ordinary["generalization"]["reading"].startswith("undefined")  # it saved nothing on training prompts
    assert ordinary["realization"]["reading"] == "weakly realized"
    assert projected["heldout_call_accuracy"] == {"student": 1.0, "recipient": 1.0}


def test_generalization_is_undefined_without_positive_training_savings():
    """The reported failure: training savings 0% [−1%, +1%], held-out 10% must not read as generalizing."""
    rng = np.random.default_rng(1)
    train_tokens = 100 + rng.normal(0, 5, 400)
    train = (train_tokens, train_tokens + rng.normal(0, 5, 400))  # about 0% saved, interval around 0
    heldout = (np.full(600, 90.0), np.full(600, 100.0))  # 10% saved
    saved = 1 - train[0] / train[1].mean()
    interval = [float(np.percentile(saved, 2.5)), float(np.percentile(saved, 97.5))]
    result = probe.generalization(train, heldout, {"ci95": [-0.01, 0.01]}, SETTINGS["thresholds"], draws=200, seed=0)
    assert result["reading"].startswith("undefined") and result["point"] is None
    assert interval[0] < 0 < interval[1]


def test_generalization_contrasts_read_both_directions():
    thresholds = SETTINGS["thresholds"]
    train = (np.full(400, 90.0), np.full(400, 100.0))  # 10% saved
    clear = {"ci95": [0.09, 0.11]}
    keeps = probe.generalization(train, (np.full(600, 91.0), np.full(600, 100.0)), clear, thresholds, draws=50, seed=0)
    assert keeps["reading"] == "generalizes" and keeps["point"] == pytest.approx(0.9)
    loses = probe.generalization(train, (np.full(600, 98.0), np.full(600, 100.0)), clear, thresholds, draws=50, seed=0)
    assert loses["reading"] == "training-specific"
    middle = probe.generalization(train, (np.full(600, 95.0), np.full(600, 100.0)), clear, thresholds, draws=50, seed=0)
    assert middle["reading"] == "inconclusive"


def test_detectable_movement_and_the_margin_are_separate_flags(probe_dir):
    """A small but detectable movement can also lie within the margin; both flags are reported."""
    prompts = probe.read_jsonl(probe_dir / "probe/prompts.jsonl")
    samples = probe_dir / "flags"
    samples.mkdir()
    write_samples(samples, "recipient", prompts, lambda p, k: 100, lambda p, k: call("f"))
    write_samples(samples, "recipient-null", prompts, lambda p, k: 100, lambda p, k: call("f"))
    write_samples(samples, "projected-seed1234", prompts, lambda p, k: 90,
                  lambda p, k: call("g") if k < 2 else call("f"))  # 2 of 8 samples move, on every prompt
    report = probe.compare(probe_dir / "probe", samples, {**SETTINGS, "equivalence_fraction": 0.99}, draws=200)
    movement = report["students"]["projected-seed1234"]["squared_distance_calls_train"]
    assert movement["ci95"][0] > 0 and movement["ci95"][1] < report["equivalence_bound_calls"]
    assert movement["detectable_movement"] and movement["within_margin"]
