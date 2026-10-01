"""Difficulty strata on BFCL: independent stratification runs, per-stratum changes, and the pooled check."""

import numpy as np
import pytest

from evaluation.bfcl_difficulty_strata import stratify
from evaluation.bfcl_shared_factor import R0, R1, R2


def test_strata_come_from_independent_runs_and_measure_where_accuracy_changes():
    # Entries 0-3: both stratification runs right; 4-5: split; 6-7: both wrong.
    s_a = np.array([1, 1, 1, 1, 1, 0, 0, 0], float)
    s_b = np.array([1, 1, 1, 1, 0, 1, 0, 0], float)
    reference = np.ones(8)
    arm = np.array([1, 1, 1, 1, 0, 0, 1, 1], float)  # loses the split entries, gains the hard ones
    correct = {R0: s_a, R2: s_b, R1: reference, "arm": arm}
    tokens = {tag: np.log1p(np.full(8, 100.0)) for tag in correct}
    tokens["arm"] = np.log1p(np.full(8, 50.0))
    groups = np.array(["live"] * 4 + ["multi_turn"] * 4)
    report = stratify(tokens, correct, groups, [("arm", R1, (R0, R2))], draws=50)
    strata = report["comparisons"]["arm"]["strata"]["all"]
    assert strata["both_right"]["entries"] == 4 and strata["both_right"]["delta_accuracy"] == 0.0
    assert strata["split"]["delta_accuracy"] == -100.0 and strata["both_wrong"]["delta_accuracy"] == 0.0
    assert strata["split"]["total_token_ratio"] == pytest.approx(-0.5)
    assert report["comparisons"]["arm"]["strata"]["multi_turn"]["both_right"]["entries"] == 0
    with pytest.raises(ValueError, match="disjoint"):
        stratify(tokens, correct, groups, [("arm", R1, (R1, R2))], draws=5)


def test_pooled_seeds_are_stratified_by_a_run_outside_every_pair():
    rng = np.random.default_rng(0)
    runs = {tag: (rng.random(20) > 0.3).astype(float) for tag in (R0, R1, R2, "a1", "a2")}
    tokens = {tag: np.log1p(rng.integers(50, 500, 20).astype(float)) for tag in runs}
    groups = np.array(["live"] * 20)
    pooled = {"name": "pooled", "pairs": [("a1", R1), ("a2", R2)], "by": R0}
    report = stratify(tokens, runs, groups, [], pooled, draws=20)
    right = report["pooled"]["strata"]["all"]["right"]["entries"]
    wrong = report["pooled"]["strata"]["all"]["wrong"]["entries"]
    assert right + wrong == 20 and right == int(runs[R0].sum())
    with pytest.raises(ValueError, match="disjoint"):
        stratify(tokens, runs, groups, [], {**pooled, "by": R1}, draws=5)
