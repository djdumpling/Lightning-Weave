"""Reflection probe: selecting first vs later reflection forks, and the reflect-minus-conclude report."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_curation.reflection_probe import best_member, evaluate, parse_quotas, select_states, state_values
from data_curation.shift_geometry import prompt_fold
from data_curation.shift_states import StateLabeler, TokenTable

VOCAB = {"Ġa": 0, ".Ċ": 1, ".ĊĊ": 2, "Wait": 3, "So": 4}
ADDED = {"<think>": 20, "</think>": 21, "<tool_call>": 22, "</tool_call>": 23, "<|im_end|>": 24}
FORK, PLAIN = [3, 4, 0], [0, 1, 2]


def labeler():
    return StateLabeler(TokenTable.from_vocab(VOCAB, ADDED))


def shard(root, prompt_id):
    """One row whose response has reflection forks at positions 3 and 6 (the second lacks "So")."""
    tokens = [20, 0, 2, 3, 0, 2, 3, 0, 21]
    candidates = [[20, 0, 1], PLAIN, [2, 1, 0], FORK, PLAIN, [2, 1, 0], [3, 0, 1], PLAIN, [21, 0, 1]]
    probs = np.full((len(tokens), 3), [0.97, 0.01, 0.01])
    probs[3] = probs[6] = [0.5, 0.4, 0.05]
    floats = pa.list_(pa.list_(pa.float32()))
    fields = {
        "sample_id": pa.array(["s0"]),
        "prompt_id": pa.array([prompt_id]),
        "prompt_tokens": pa.array([[7, 8, 9]], type=pa.list_(pa.int32())),
        "response_tokens": pa.array([tokens], type=pa.list_(pa.int32())),
        "candidate_ids": pa.array([candidates], type=pa.list_(pa.list_(pa.int32()))),
        "behavior_topk_log_probs": pa.array([np.log(probs).tolist()], type=floats),
        "loss_mask": pa.array([[True] * len(tokens)], type=pa.list_(pa.bool_())),
        "pre_teacher_log_probs": pa.array([np.zeros((len(tokens), 3)).tolist()], type=floats),
        "post_teacher_log_probs": pa.array([np.zeros((len(tokens), 3)).tolist()], type=floats),
    }
    metadata = pa.StructArray.from_arrays(list(fields.values()), names=list(fields))
    root.mkdir()
    table = pa.table({"prompt": ["p"], "label": [""], "metadata": metadata})
    pq.write_table(table, root / "rollouts-r00000-00000.parquet")
    return root


def test_select_keeps_forks_where_both_markers_are_live_and_labels_their_order(tmp_path):
    base = shard(tmp_path / "base", "prompt-a")
    fold = prompt_fold("prompt-a")
    meta = {"prompt-a": {"conversation": "multi_turn"}}
    quotas = parse_quotas("multi_turn:first=5,multi_turn:later=5")
    states = select_states(base, labeler(), meta, folds={fold}, quotas=quotas, min_mass=0.02, seed=0)
    # Position 3 is the first fork and has both markers; position 6 is a fork but "So" is not a candidate.
    assert [(s["order"], s["position"]) for s in states] == [("first", 3)]
    (state,) = states
    assert state["prefix_tokens"] == [7, 8, 9, 20, 0, 2] and state["response_prefix_length"] == 3
    assert state["forced"] == {"reflect": 3, "conclude": 4}
    assert state["behavior_probs"] == pytest.approx({"reflect": 0.5, "conclude": 0.4})
    other_fold = select_states(base, labeler(), meta, folds={(fold + 1) % 5}, quotas=quotas, min_mass=0.02, seed=0)
    assert other_fold == []
    assert best_member(labeler(), "conclusion", np.asarray(PLAIN), np.ones(3)) is None


def record(sample_id, position, alternative, match, tokens, finished=True):
    return {"sample_id": sample_id, "position": position, "alternative": alternative, "finished": finished,
            "think_tokens": tokens, "total_tokens": tokens, "tool_calls": 1, "call_match": match, "malformed": False}


def test_report_measures_reflect_minus_conclude_per_stratum_and_first_minus_later():
    states = [
        {"sample_id": "a", "prompt_id": "p1", "kind": "multi_turn", "order": "first", "position": 3},
        {"sample_id": "a", "prompt_id": "p1", "kind": "multi_turn", "order": "later", "position": 9},
        {"sample_id": "b", "prompt_id": "p2", "kind": "multi_turn", "order": "first", "position": 4},
    ]
    records = [
        # a:3 reflecting always matches, concluding matches half the time (one censored miss).
        record("a", 3, "reflect", True, 100), record("a", 3, "reflect", True, 120),
        record("a", 3, "conclude", True, 40), record("a", 3, "conclude", False, 60, finished=False),
        # a:9 no difference.
        record("a", 9, "reflect", True, 80), record("a", 9, "conclude", True, 30),
        # b:4 reflecting helps fully.
        record("b", 4, "reflect", True, 90), record("b", 4, "conclude", False, 50),
    ]
    values = state_values(states, records)
    assert values[0]["conclude"]["match"] == 0.5 and values[0]["conclude"]["censored"] == 0.5
    report = evaluate(states, records, draws=200, seed=0)["strata"]
    first = report["multi_turn:first"]
    assert first["delta_match"]["mean"] == pytest.approx((0.5 + 1.0) / 2)
    assert first["delta_tokens"]["mean"] == pytest.approx(((110 - 50) + (90 - 50)) / 2)
    assert report["multi_turn:later"]["delta_match"]["mean"] == pytest.approx(0.0)
    assert report["multi_turn:first_minus_later"]["delta_match"]["mean"] == pytest.approx(0.75)
    with pytest.raises(ValueError, match="no records"):
        state_values(states, records[:-2])
