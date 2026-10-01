"""Overlap analysis: disjoint runs, repeat-based reliability under unequal noise, identification, and nulls."""

import numpy as np
import pytest

from evaluation.bfcl_shared_factor import R0, R1, R2, analyze, disjoint

N = 6_000


def synthetic(seed=0):
    """Levers A and B share a susceptibility u with very different noise; C follows v; D has none."""
    rng = np.random.default_rng(seed)
    base = rng.normal(6.0, 1.0, N)
    u, v = rng.normal(0, 1, N), rng.normal(0, 1, N)

    def run(signal, noise):
        return base + signal + rng.normal(0, noise, N)

    tokens = {R1: run(0, 0.5), R2: run(0, 0.5), R0: run(0, 0.5)}
    tokens |= {"A1": run(-0.3 * u, 0.2), "A2": run(-0.3 * u, 0.2)}  # low noise
    tokens |= {"B1": run(-0.6 * u, 0.8), "B2": run(-0.6 * u, 0.8)}  # high noise
    tokens |= {"C1": run(-0.6 * v, 0.5), "C2": run(-0.6 * v, 0.5)}
    tokens |= {"D1": run(0, 0.5), "D2": run(0, 0.5)}  # no signal at all
    correct = {tag: np.ones(N) for tag in tokens}
    correct["B1"] = (rng.random(N) > np.clip(0.5 * u, 0, 1)).astype(float)  # accuracy falls where u cuts most
    groups = np.where(np.arange(N) % 4 == 0, "multi_turn", "live")
    arms = {"A1": R1, "A2": R2, "B1": R0, "B2": R2, "C1": R1, "C2": R0, "D1": R1, "D2": R2}
    levers = {"A": ["A1", "A2"], "B": ["B1", "B2"], "C": ["C1", "C2"], "D": ["D1", "D2"]}
    return tokens, correct, groups, arms, levers


def test_repeat_based_correction_recovers_the_latent_correlation_under_unequal_noise():
    tokens, correct, groups, arms, levers = synthetic()
    report = analyze(tokens, correct, groups, arms, levers, draws=100)
    overall = report["by_subset"]["all"]
    reliability = overall["reliability"]
    # A: 0.09 / (0.09 + 0.04 + 0.25); B: 0.36 / (0.36 + 0.64 + 0.25).
    assert reliability["A"]["r"] == pytest.approx(0.09 / 0.38, abs=0.04)
    assert reliability["B"]["r"] == pytest.approx(0.36 / 1.25, abs=0.04)
    latent = overall["latent"]
    assert latent["A ~ B"]["identified"] and latent["A ~ B"]["latent_r"] == pytest.approx(1.0, abs=0.15)
    low, high = latent["A ~ B"]["ci95"]
    assert low < 1.1 and high > 0.9 and high - low < 0.4  # an interval around the truth, not a degenerate one
    assert abs(latent["A ~ C"]["latent_r"]) < 0.2
    # A lever with no signal has no identified reliability, so nothing involving it is estimated.
    assert not latent["A ~ D"]["identified"] and latent["A ~ D"]["ci95"] == [None, None]
    for item in report["null"].values():
        assert abs(item["r"]) < 0.06


def test_only_disjoint_runs_are_compared_and_consensus_is_descriptive():
    tokens, correct, groups, arms, levers = synthetic(1)
    report = analyze(tokens, correct, groups, arms, levers, draws=30)
    assert "A1 ~ C1" not in report["raw"] and "A2 ~ B2" not in report["raw"]  # shared references
    assert "A1 ~ B1" in report["raw"]
    consensus = report["consensus"]
    assert sorted(consensus["arms"]) == ["A1", "C1", "D1"]
    assert consensus["predicts"]["B1"]["tokens"]["all"]["r"] > 0.05
    assert consensus["predicts"]["B1"]["correct"]["all"]["r"] > 0.0
    assert len(consensus["quintiles"]) == 5
    with pytest.raises(ValueError, match="without references"):
        analyze(tokens, correct, groups, arms, {"X": ["A1", "missing"]}, draws=5)
    assert not disjoint(("x", R1), ("y", R1)) and disjoint(("x", R1), ("y", R2))
