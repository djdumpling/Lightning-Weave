"""Turn positions of cached prompts, and the prompt weights that gate a term by them."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_curation.turn_positions import RULES, position, positions, provenance, rule_weights, summary


def message(role):
    return {"role": role, "content": ""}


def test_position_reads_what_the_target_answers():
    assert position([message("system"), message("user")]) == "first_turn"
    assert position([message("user"), message("assistant"), message("user")]) == "turn_start"
    assert position([message("user"), message("assistant"), message("tool")]) == "after_tool"
    later = [message("user"), message("assistant"), message("tool"), message("assistant"), message("user")]
    assert position(later) == "turn_start"
    for bad in ([message("user"), message("assistant")], [message("system")], []):
        with pytest.raises(ValueError, match="must answer"):
            position(bad)


def write_inputs(tmp_path, rows):
    canonical = tmp_path / "canonical.jsonl"
    extra = {"messages": [message("user")], "metadata": {"source_index": 99, "conversation_kind": "single_turn"}}
    lines = [{"messages": m, "metadata": {"source_index": s, "conversation_kind": k}} for _, s, k, m in rows]
    canonical.write_text("\n".join(json.dumps(row) for row in [*lines, extra]))
    prompts = tmp_path / "prompts.parquet"
    pq.write_table(pa.table({"prompt_id": [p for p, *_ in rows], "source_index": [s for _, s, *_ in rows]}), prompts)
    return prompts, canonical


def test_positions_join_prompts_to_canonical_rows_and_weights_gate_turn_starts(tmp_path):
    rows = [
        ("p0", 10, "single_turn", [message("user")]),
        ("p1", 11, "multi_turn", [message("user"), message("assistant"), message("user")]),
        ("p2", 12, "multi_turn", [message("user"), message("assistant"), message("tool")]),
    ]
    prompts, canonical = write_inputs(tmp_path, rows)
    found, kinds = positions(prompts, canonical)
    assert found == {"p0": "first_turn", "p1": "turn_start", "p2": "after_tool"}
    weights = rule_weights(found, kinds)
    assert set(weights) == set(RULES)
    assert weights["protect_turn_starts"] == {"p0": 1.0, "p1": 0.0, "p2": 1.0}
    report = summary(found, kinds, weights)
    assert report["positions"] == {"first_turn": 1, "turn_start": 1, "after_tool": 1}
    assert report["by_kind"] == {"multi_turn:after_tool": 1, "multi_turn:turn_start": 1, "single_turn:first_turn": 1}
    assert set(provenance(prompts, canonical)) == {"prompts", "canonical", "code"}
    pq.write_table(pa.table({"prompt_id": ["p0", "lost"], "source_index": [10, 77]}), prompts)
    with pytest.raises(KeyError, match="no canonical row"):
        positions(prompts, canonical)


def test_the_random_control_gates_as_many_multi_turn_prompts_at_random():
    found = {f"t{i}": "turn_start" for i in range(30)} | {f"a{i}": "after_tool" for i in range(10)}
    found |= {f"f{i}": "first_turn" for i in range(20)}
    kinds = {p: "single_turn" if p.startswith("f") else "multi_turn" for p in found}
    weights = rule_weights(found, kinds)
    control = {p for p, w in weights["random_multi_turn_control"].items() if w == 0.0}
    protect = {p for p, w in weights["protect_turn_starts"].items() if w == 0.0}
    assert len(control) == len(protect) == 30 and all(kinds[p] == "multi_turn" for p in control)
    assert any(found[p] == "after_tool" for p in control)  # a random choice, not the turn starts
    assert rule_weights(found, kinds) == weights  # deterministic
    report = summary(found, kinds, weights)
    assert report["gated"]["random_multi_turn_control"]["prompts"] == 30
    assert report["gated_by_both_rules"] == len(control & protect)
    with pytest.raises(ValueError, match="multi-turn prompts"):
        rule_weights({"t": "turn_start", "u": "turn_start"}, {"t": "multi_turn", "u": "single_turn"})
