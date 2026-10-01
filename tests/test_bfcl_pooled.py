"""Pooled paired BFCL comparisons: metric definitions, pooling over pairs, and the decision rules."""

import numpy as np
import pytest

from evaluation.bfcl_efficiency import MULTI_TURN, V3_CATEGORIES
from evaluation.bfcl_pooled import METRICS, analyze, verdict

N = 4


def run(correct=None, tokens=None):
    """Every category with N entries: all correct and 100 tokens each, unless overridden per category."""
    correct, tokens = correct or {}, tokens or {}
    return {
        category: (
            [f"{category}_{i}" for i in range(N)],
            np.array(correct.get(category, [1.0] * N)),
            np.array(tokens.get(category, [100.0] * N)),
        )
        for category in V3_CATEGORIES
    }


def test_metrics_split_multi_turn_from_single_turn():
    reference = run()
    arm = run(correct={"multi_turn_base": [0.0, 1.0, 1.0, 1.0]}, tokens={"simple_python": [200.0] * N})
    pooled = analyze({"arm": arm, "ref": reference}, {"c": {"pairs": [("arm", "ref")]}}, draws=50)["c"]["pooled"]
    # one of 4 entries in one of the 4 equally weighted multi-turn categories; overall is a third of multi-turn
    assert pooled["multi_turn_accuracy"]["delta"] == pytest.approx(-100 / 4 / 4)
    assert pooled["accuracy"]["delta"] == pytest.approx(-100 / 4 / 4 / 3)
    assert pooled["single_turn_accuracy"]["delta"] == pytest.approx(0.0)
    single_turn = len(V3_CATEGORIES) - len(MULTI_TURN)
    assert pooled["single_turn_tokens"]["delta"] == pytest.approx(100 / single_turn)
    assert pooled["total_tokens"]["delta"] == pytest.approx(100 / len(V3_CATEGORIES))
    assert pooled["multi_turn_tokens"]["delta"] == pytest.approx(0.0)


def test_pooling_averages_pairs_and_identical_runs_give_zero_intervals():
    reference = run()
    worse = run(correct={category: [0.0, 1.0, 1.0, 1.0] for category in MULTI_TURN})
    comparisons = {"pooled": {"pairs": [("worse", "ref"), ("ref", "ref")]}, "null": {"pairs": [("ref", "ref")]}}
    report = analyze({"worse": worse, "ref": reference}, comparisons, draws=200)
    pooled = report["pooled"]
    assert [row["multi_turn_accuracy"] for row in pooled["per_pair"]] == pytest.approx([-25.0, 0.0])
    assert pooled["pooled"]["multi_turn_accuracy"]["delta"] == pytest.approx(-12.5)
    low, high = pooled["pooled"]["multi_turn_accuracy"]["ci95"]
    assert low <= -12.5 <= high
    assert all(report["null"]["pooled"][metric]["ci95"] == [0.0, 0.0] for metric in METRICS)


def test_runs_must_share_entries():
    other = run()
    other["live_simple"] = (["x", "y", "z", "w"], *other["live_simple"][1:])
    with pytest.raises(ValueError, match="different live_simple entries"):
        analyze({"a": run(), "b": other}, {"c": {"pairs": [("a", "b")]}}, draws=10)


@pytest.mark.parametrize(
    ("rule", "interval", "expected"),
    [
        (["recovery", "m", 1.0], [0.2, 2.5], "recovers"),
        (["recovery", "m", 1.0], [0.1, 0.8], "partial"),
        (["recovery", "m", 1.0], [-1.5, 0.9], "does not recover"),
        (["recovery", "m", 1.0], [-1.0, 1.5], "inconclusive"),
        (["non_inferior", "m", 1.0], [-0.8, 0.4], "no detectable cost"),
        (["non_inferior", "m", 1.0], [-0.8, -0.1], "small cost within margin"),
        (["non_inferior", "m", 1.0], [-1.6, -0.2], "costs"),
        (["non_inferior", "m", 1.0], [-1.6, 0.3], "inconclusive"),
        (["reduces", "m"], [-9.0, -2.0], "reduced"),
        (["reduces", "m"], [-3.0, 0.5], "not shown"),
    ],
)
def test_verdicts(rule, interval, expected):
    assert verdict(rule, {"m": interval}) == expected


def test_rules_are_reported_by_name():
    report = analyze(
        {"a": run(), "b": run()},
        {"c": {"pairs": [("a", "b")], "rules": [["reduces", "single_turn_tokens"]]}},
        draws=10,
    )
    assert report["c"]["verdicts"] == {"reduces single_turn_tokens": "not shown"}
