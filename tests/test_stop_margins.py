"""Stop states: gate classification, the composed-target view, consistent masking, and censored thoughts."""

import numpy as np
import pytest

from data_curation.shift_geometry import CachedRow, bucket_probs, log_tilted_target
from data_curation.shift_states import StateLabeler, TokenTable
from data_curation.stop_margins import Totals, gate_names, row_statistics

VOCAB = {"Ġa": 0, ".Ċ": 1, ".ĊĊ": 2, "Wait": 3, "So": 4}
ADDED = {"<think>": 20, "</think>": 21, "<tool_call>": 22, "</tool_call>": 23, "<|im_end|>": 24}
CLOSE = 21
TAUS = (0.25, 1.0)

# position: (generated token, candidates, behavior probabilities)
STEPS = [
    (20, [20, 0, 1], [0.97, 0.01, 0.01]),  # 0 <think>: not a thinking state
    (0, [0, 1, 2], [0.34, 0.33, 0.30]),  # 1 stop fork; </think> absent, K-th candidate 0.125 below top
    (1, [1, CLOSE, 0], [0.60, 0.30, 0.05]),  # 2 </think> at margin log 2 = 0.69
    (2, [2, 1, 0], [0.97, 0.01, 0.01]),  # 3 boundary (after ".\n"); </think> absent, certified far
    (3, [3, 4, 0], [0.50, 0.40, 0.05]),  # 4 boundary; first reflection fork
    (0, [0, CLOSE, 1], [0.50, 0.45, 0.02]),  # 5 </think> at margin log(0.5 / 0.45) = 0.105
    (2, [2, 1, 0], [0.97, 0.01, 0.01]),  # 6
    (3, [3, 4, 0], [0.50, 0.40, 0.05]),  # 7 boundary; second reflection fork
    (CLOSE, [CLOSE, 0, 1], [0.90, 0.05, 0.02]),  # 8 </think> already the top candidate
    (24, [24, 0, 1], [0.97, 0.01, 0.01]),  # 9 after the thought
]


def fixture_row(loss_mask=None):
    tokens = np.asarray([step[0] for step in STEPS])
    candidates = np.asarray([step[1] for step in STEPS])
    probs = np.asarray([step[2] for step in STEPS])
    at_close = np.where(candidates == CLOSE, 1.0, 0.0)
    ones = np.ones(candidates.shape, dtype=bool)
    sources = {"acc": -2.0 * at_close, "eff": at_close}  # accuracy pushes </think> down; efficiency up
    return CachedRow(
        "s0", "p0", tokens, candidates, np.log(probs),
        np.ones(len(tokens), dtype=bool) if loss_mask is None else np.asarray(loss_mask),
        shifts=sources, mapped=dict.fromkeys(sources, ones), trained=dict.fromkeys(sources, ones),
        support=dict.fromkeys(sources, np.ones(len(tokens))),
    )


def labeler():
    return StateLabeler(TokenTable.from_vocab(VOCAB, ADDED))


def statistics(row=None, **target):
    return row_statistics(row or fixture_row(), labeler(), "eff", TAUS, **target)


def test_gates_partition_thinking_positions_by_how_close_think_close_is():
    gates = statistics()["gates"]
    assert set(gates) == set(gate_names(TAUS))

    def count(name):
        return gates[name]["positions"]

    assert (count("all:close_top"), count("all:near@0.25"), count("all:far@0.25")) == (1, 1, 1)
    assert (count("all:absent_certified@0.25"), count("all:absent_unresolved@0.25")) == (4, 1)
    assert (count("all:near@1.0"), count("all:far@1.0")) == (2, 0)
    for tau in TAUS:  # the five close gates partition the eight thinking positions
        kinds = ("near", "far", "absent_certified", "absent_unresolved")
        assert count("all:close_top") + sum(count(f"all:{kind}@{tau}") for kind in kinds) == 8
    assert count("boundary:absent_certified@0.25") == 3 and count("boundary:near@1.0") == 0
    assert count("stop_fork") == 1 and gates["stop_fork"]["stop_b"] == pytest.approx(0.33)
    assert gates["all:near@0.25"]["stop_b"] == pytest.approx(0.45)
    assert gates["all:near@0.25"]["close_shift"] == pytest.approx(0.55)  # centered: 1 - b(</think>)


def test_the_composed_target_is_measured_against_the_accuracy_target():
    stats = statistics(accuracy_source="acc", coef=2.0, alpha=2.0)
    near = stats["gates"]["all:near@0.25"]  # position 5
    # The efficiency term (2 x +1) cancels the accuracy push (-2), so the composed target is the behavior.
    assert near["stop_new"] == pytest.approx(0.45)
    b = bucket_probs(np.log(np.asarray([[0.50, 0.45, 0.02]])))
    q_acc = np.exp(log_tilted_target(b, np.asarray([[0.0, -2.0, 0.0, 0.0]]), 2.0))
    assert near["stop_acc"] == pytest.approx(q_acc[0, 1])
    assert near["kl"] == pytest.approx(float((b * (np.log(b) - np.log(q_acc))).sum()))
    fork = stats["gates"]["stop_fork"]  # no </think> candidate: both targets equal the behavior
    assert (fork["stop_acc"], fork["stop_new"], fork["kl"]) == pytest.approx((0.33, 0.33, 0.0))


def test_removable_suffix_uses_the_same_mask_as_every_gate():
    stats = statistics()
    assert stats["suffix"] == {
        "all:near@0.25": 3, "all:near@1.0": 6, "boundary:near@0.25": 0, "boundary:near@1.0": 0,
    }
    masked = statistics(fixture_row(loss_mask=[False] * len(STEPS)))
    assert masked["thinking_positions"] == 0 and set(masked["suffix"].values()) == {0}


def test_totals_separate_censored_thoughts_and_report_shares():
    totals = Totals(TAUS, has_target=True)
    totals.add("multi_turn", statistics(accuracy_source="acc", coef=2.0))
    row = fixture_row()
    row.response_tokens = np.where(row.response_tokens == CLOSE, 0, row.response_tokens)  # never closes
    totals.add("single_turn", statistics(row, accuracy_source="acc", coef=2.0))
    summary = totals.summary()
    assert set(summary) == {"all", "multi_turn", "single_turn"}
    everything = summary["all"]
    assert (everything["closed_rows"], everything["unclosed_rows"]) == (1, 1)
    assert everything["mean_unclosed_thinking_tokens"] == 9  # positions 1-9 stay inside the thought
    near = everything["gates"]["all:near@1.0"]
    assert near["removable_suffix_share_of_closed_thinking"] == pytest.approx(6 / 8)  # closed thoughts only
    assert summary["single_turn"]["gates"]["all:near@1.0"]["removable_suffix_share_of_closed_thinking"] is None
    assert sum(item["energy_share"] for item in everything["states"].values()) == pytest.approx(1.0)
    assert everything["mean_added_kl"] > 0 and near["max_added_kl"] >= near["mean_added_kl"]
