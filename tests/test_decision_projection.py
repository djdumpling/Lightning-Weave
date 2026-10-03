import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_curation import decision_projection as projection


def call(name, **arguments):
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


def visible_of(decision):
    return json.loads(decision)["visible"]


def test_a_decision_is_the_exact_visible_output_and_how_the_response_ended():
    parsed = projection.parse_response(f"<think>\nplan\n</think>\n\n{call('cd', folder='a')}", "stop")
    assert parsed.reasoning == "\nplan\n" and parsed.finished
    assert visible_of(parsed.decision) == f"\n\n{call('cd', folder='a')}"
    assert json.loads(parsed.decision)["finish_reason"] == "stop"

    def key(visible):
        return projection.parse_response("<think>r</think>" + visible, "stop").decision

    # Nothing is normalized: whitespace inside text, argument key order and call order all distinguish decisions.
    assert key("print('a  b')") != key("print('a b')")
    reordered = '<tool_call>\n{"arguments": {"b": 2, "a": 1}, "name": "f"}\n</tool_call>'
    assert key(reordered) != key('<tool_call>\n{"name": "f", "arguments": {"a": 1, "b": 2}}\n</tool_call>')
    first, second = call("cd", folder="a"), call("mv", source="x", destination="y")
    assert key(first + second) != key(second + first)
    assert key(call("cd", folder="a")) == key(call("cd", folder="a"))
    # The same reasoning-independent output ending differently is a different decision.
    assert projection.parse_response("<think>r</think>Done.", "stop").decision != (
        projection.parse_response("<think>r</think>Done.", "length").decision
    )


def test_decisions_follow_the_servers_reasoning_split():
    """vLLM 0.11.0's qwen3 parser: unless both tags are present, the whole output is visible content."""
    calls_f = projection.parse_response("<think>a</think>\n" + call("f"), "length")
    calls_g = projection.parse_response("<think>b</think>\n" + call("g"), "length")
    assert calls_f.decision != calls_g.decision and not calls_f.finished
    # Unclosed reasoning is shown as text, so two different unclosed responses are two different decisions.
    unclosed = [projection.parse_response(f"<think>\nstill {word}", "length") for word in ("thinking", "going")]
    assert unclosed[0].decision != unclosed[1].decision
    assert unclosed[0].visible == "<think>\nstill thinking" and unclosed[0].reasoning is None
    assert not unclosed[0].finished
    # A closing tag without an opening one (a prompt-opened template) is also all content for this parser.
    assert projection.served_split("plan\n</think>\n\nDone.") == (None, "plan\n</think>\n\nDone.")
    assert projection.served_split("<think>plan</think>") == ("plan", None)  # empty content is None
    assert projection.served_split("x<think>plan</think>Done.") == ("plan", "Done.")  # text before <think> is dropped
    plain = projection.parse_response("plain answer", "stop")
    assert plain.reasoning is None and plain.visible == "plain answer" and not plain.finished
    assert projection.reasoning_opened("<|im_start|>assistant\n<think>\n")
    assert not projection.reasoning_opened("<|im_start|>assistant\n")


def test_projection_reproduces_the_preservation_example():
    """A-only, A-and-B and two B-only responses at 25/25/50%; the donor favors both B kinds by e^1."""
    decisions = ["keeps A", "keeps A", "loses A", "loses A"]
    scores = np.array([0.0, 2.0, 2.0, 2.0])
    ordinary = projection.arm_weights(scores, decisions, arm="ordinary", alpha=2.0) / 4
    projected = projection.arm_weights(scores, decisions, arm="projected", alpha=2.0) / 4
    np.testing.assert_allclose(ordinary[:2].sum(), (1 + math.e) / (1 + 3 * math.e))  # A falls to 40.6%
    np.testing.assert_allclose([ordinary[0], ordinary[1], ordinary[2:].sum()], [0.1092, 0.2969, 0.5938], atol=1e-4)
    np.testing.assert_allclose([projected[0], projected[1], projected[2:].sum()], [0.1345, 0.3655, 0.5], atol=1e-4)


@pytest.mark.parametrize("eta", [0.0, 0.3, 1.0])
def test_projected_weights_keep_every_decision_groups_mass(eta):
    rng = np.random.default_rng(0)
    decisions = list("aabbbcd")
    scores = rng.normal(size=len(decisions)) * 5
    weights = projection.arm_weights(scores, decisions, arm="projected", alpha=0.7, eta=eta)
    for decision in set(decisions):
        members = [index for index, value in enumerate(decisions) if value == decision]
        assert weights[members].sum() == pytest.approx(len(members))
    assert weights[decisions.index("c")] == pytest.approx(1.0)  # a singleton keeps its weight
    ordinary = projection.arm_weights(scores, decisions, arm="ordinary", alpha=0.7, eta=eta)
    assert ordinary.sum() == pytest.approx(len(decisions))
    np.testing.assert_array_equal(projection.arm_weights(None, decisions, arm="uniform"), np.ones(len(decisions)))


def test_weights_reject_bad_inputs():
    with pytest.raises(ValueError, match="alpha"):
        projection.arm_weights([0.0, 1.0], "ab", arm="projected", alpha=0.0)
    with pytest.raises(ValueError, match="finite"):
        projection.arm_weights([0.0, math.nan], "ab", arm="ordinary", alpha=1.0)
    with pytest.raises(ValueError, match="eta"):
        projection.arm_weights([0.0, 1.0], "ab", arm="ordinary", alpha=1.0, eta=1.5)
    with pytest.raises(ValueError, match="arm"):
        projection.arm_weights([0.0, 1.0], "ab", arm="other", alpha=1.0)


def toy_samples():
    """Two prompts. In p1 the shorter reasoning scores higher within each decision; the short decision is 'quick'."""
    rows = [
        ("p1", "careful", 100, -10.0),
        ("p1", "careful", 60, -6.0),
        ("p1", "quick", 20, -2.0),
        ("p1", "quick", 30, -3.0),
        ("p2", "only", 50, -5.0),
        ("p2", "only", 50, -5.0),
        ("p2", "single", 80, -8.0),
    ]
    return [
        projection.Sample(prompt, f"{prompt}-{index}", decision, tokens, score)
        for index, (prompt, decision, tokens, score) in enumerate(rows)
    ]


def test_structure_report_counts_shared_decisions_and_the_shortest_bound():
    report = projection.structure_report(toy_samples())
    assert report["prompts"] == 2 and report["samples"] == 7
    assert report["samples_in_shared_decision_fraction"] == pytest.approx(6 / 7)
    assert report["largest_group_size"] == {2: 2}
    # shortest within decision: careful 160 -> 120, quick 50 -> 40, only 100 -> 100: saves 50 of 390 tokens.
    assert report["shortest_within_decision_savings"] == pytest.approx(50 / 390)


def test_weight_report_and_calibration():
    samples = toy_samples()
    report = projection.weight_report(samples, alpha=1.0)
    assert report["projected_group_mass_max_error"] < 1e-12
    assert report["arms"]["uniform"]["implied_savings"] == pytest.approx(0.0)
    # The ordinary tilt also moves weight toward the cheap decision, so it saves more than the projection at equal α.
    assert report["arms"]["ordinary"]["implied_savings"] > report["arms"]["projected"]["implied_savings"] > 0
    assert report["ordinary_decision_tv"]["mean"] > 0
    assert report["within_decision_score_length_spearman"]["median"] == pytest.approx(-1.0)

    calibration = projection.calibrate_alpha(samples, 0.05, arm="projected")
    assert calibration["implied_savings"] == pytest.approx(0.05, abs=1e-6)
    with pytest.raises(ValueError, match="reaches at most"):
        projection.calibrate_alpha(samples, 0.5, arm="projected")
    curve = projection.savings_curve(samples, [0.5, 5.0])
    assert curve[0]["projected"] > curve[1]["projected"]


def test_savings_match_uses_each_arms_pooled_savings_against_the_recipient():
    pooled = {
        "projected vs recipient": {"pooled": {"total_tokens": {"delta": -12.0}}},
        "ordinary vs recipient": {"pooled": {"total_tokens": {"delta": -14.5}}},
    }
    result = projection.savings_matched(pooled, "projected", "ordinary", 3.0)
    assert result["savings"] == {"projected": 12.0, "ordinary": 14.5} and result["matched"]
    assert not projection.savings_matched(pooled, "projected", "ordinary", 2.0)["matched"]


def write_rollouts(path, rows):
    metadata = [
        {"prompt_id": prompt, "sample_id": sample, "response": response, "response_length": length,
         "finish_reason": "stop"}
        for prompt, sample, response, length in rows
    ]
    pq.write_table(pa.table({"prompt": ["x"] * len(rows), "metadata": metadata}), path)


def test_cli_writes_weights_for_every_arm(tmp_path):
    rows = [
        ("p", "s0", "<think>long</think>" + call("f", a=1), 40),
        ("p", "s1", "<think>short</think>" + call("f", a=1), 20),
        ("p", "s2", "<think>x</think>Sure.", 10),
    ]
    write_rollouts(tmp_path / "rollouts.parquet", rows)
    pq.write_table(pa.table({"sample_id": ["s0", "s1", "s2"], "score": [-4.0, -2.0, -1.0]}), tmp_path / "s.parquet")
    output = tmp_path / "weights.parquet"
    projection.main(["weights", "--rollouts", str(tmp_path / "rollouts.parquet"), "--scores",
                     str(tmp_path / "s.parquet"), "--alpha", "1.0", "--output", str(output)])
    table = pq.read_table(output).to_pydict()
    assert table["sample_id"] == ["s0", "s1", "s2"]
    assert table["weight_uniform"] == [1.0, 1.0, 1.0]
    np.testing.assert_allclose(table["weight_projected"], [2 / (1 + math.e**2), 2 * math.e**2 / (1 + math.e**2), 1.0])
    assert sum(table["weight_ordinary"]) == pytest.approx(3.0)
    assert pq.read_schema(output).metadata[b"schema"] == b"decision_projection_weights_v1"

    with pytest.raises(ValueError, match="no score"):
        pq.write_table(pa.table({"sample_id": ["s0"], "score": [0.0]}), tmp_path / "partial.parquet")
        projection.attach_scores(projection.load_rollouts([tmp_path / "rollouts.parquet"]), tmp_path / "partial.parquet")


def test_calibration_finds_a_target_reached_only_at_a_moderate_tilt():
    """Scores need not rank by length: the strongest tilt picks the long top-scored response and saves nothing."""
    samples = [
        projection.Sample("p", f"s{index}", "same", tokens, score)
        for index, (tokens, score) in enumerate([(100, 0.0), (1, 19.0), (100, 20.0)])
    ]
    strongest = projection.implied_savings(samples, projection.dataset_weights(samples, arm="projected", alpha=1e-3))
    assert strongest < 0.01
    calibration = projection.calibrate_alpha(samples, 0.15, arm="projected")
    assert calibration["implied_savings"] == pytest.approx(0.15, abs=1e-6)
    peak = projection.implied_savings(samples, projection.dataset_weights(samples, arm="projected", alpha=6.79))
    assert peak > 0.17 and calibration["alpha"] > 6.79  # the weakest tilt that reaches the target
    with pytest.raises(ValueError, match="weakest tilt"):
        projection.calibrate_alpha(samples, 0.0, arm="projected")


def test_spearman_gives_ties_their_mean_rank():
    np.testing.assert_array_equal(projection._average_ranks(np.array([3.0, 1.0, 3.0, 2.0])), [2.5, 0.0, 2.5, 1.0])
    assert projection._spearman(np.array([1.0, 1.0, 2.0]), np.array([5.0, 5.0, 9.0])) == pytest.approx(1.0)
    assert projection._spearman(np.array([1.0, 1.0]), np.array([2.0, 3.0])) is None


def test_scores_join_from_a_directory_of_rank_files(tmp_path):
    write_rollouts(tmp_path / "rollouts.parquet", [("p", "s0", "<think>a</think>x", 3), ("p", "s1", "<think>b</think>x", 3)])
    scores = tmp_path / "scores"
    scores.mkdir()
    pq.write_table(pa.table({"sample_id": ["s0"], "score": [1.0]}), scores / "r0.parquet")
    pq.write_table(pa.table({"sample_id": ["s1"], "score": [2.0]}), scores / "r1.parquet")
    joined = projection.attach_scores(projection.load_rollouts([tmp_path / "rollouts.parquet"]), scores)
    assert [sample.score for sample in joined] == [1.0, 2.0]
    pq.write_table(pa.table({"sample_id": ["s1"], "score": [2.0]}), scores / "r2.parquet")
    with pytest.raises(ValueError, match="twice"):
        projection.attach_scores(projection.load_rollouts([tmp_path / "rollouts.parquet"]), scores)
