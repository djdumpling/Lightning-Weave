"""Per-prompt difficulty: verified outcomes, text references, truncation, rules, coverage, and provenance."""

import json

import pytest

from data_curation.prompt_difficulty import prompt_statistics, provenance, rule_weights, summary, verify


def call(name, **arguments):
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def test_verify_checks_calls_exactly_and_never_marks_text_references_correct():
    target = {"content": "", "tool_calls": [call("book", city="Paris")]}
    right = '<think>\nok\n</think>\n\n<tool_call>\n{"name": "book", "arguments": {"city": "Paris"}}\n</tool_call>'
    wrong = right.replace("Paris", "Rome")
    assert verify(right, target) == "match" and verify(wrong, target) == "miss"
    assert verify("<think>\nok\n</think>\n\nBooked your flight!", target) == "miss"
    text_target = {"content": "Which city?", "tool_calls": []}
    assert verify("<think>\nok\n</think>\n\nPlease provide the city.", text_target) == "unverifiable"
    assert verify("<think>\nok\n</think>\n\nBooked your flight!", text_target) == "unverifiable"


def test_verified_solved_needs_a_call_reference_and_every_response_finished_and_matched():
    kinds = {
        "easy": {"conversation_kind": "single_turn", "target_kind": "single_call"},
        "mixed": {"conversation_kind": "multi_turn", "target_kind": "single_call"},
        "cut": {"conversation_kind": "multi_turn", "target_kind": "parallel_call"},
        "text": {"conversation_kind": "single_turn", "target_kind": "text_no_call"},
    }
    outcomes = {
        "easy": ["match", "match"],
        "mixed": ["match", "miss"],
        "cut": ["match", "unfinished"],
        "text": ["unverifiable", "unverifiable"],
    }
    responses = [(prompt, [index, len(prompt)]) for prompt, items in outcomes.items() for index in range(len(items))]
    stats = prompt_statistics(responses, lambda prompt, tokens: outcomes[prompt][tokens[0]], kinds)
    assert {p: item["verified_solved"] for p, item in stats.items()} == {
        "easy": True,
        "mixed": False,
        "cut": False,
        "text": False,
    }
    assert stats["cut"]["unfinished"] == 1 and stats["cut"]["call_pass_rate"] == 0.5
    assert stats["text"]["call_pass_rate"] is None
    weights = rule_weights(stats)
    assert weights["verified_solved"] == {"easy": 1.0, "mixed": 0.0, "cut": 0.0, "text": 0.0}
    assert weights["call_pass_rate"]["text"] == 0.0
    report = summary(stats)
    assert report["all"]["prompts"] == 4 and report["all"]["verified_solved_prompts"] == 1
    assert report["single_turn:text_no_call"]["unverifiable_share"] == 1.0
    assert report["multi_turn:parallel_call"]["unfinished_share"] == 0.5
    with pytest.raises(ValueError, match="unknown outcome"):
        prompt_statistics([("easy", [0])], lambda *_: "solved", kinds)


def test_provenance_changes_with_any_input(tmp_path):
    base = tmp_path / "cache"
    base.mkdir()
    (base / "manifest.json").write_text("{}")
    inputs = {name: tmp_path / name for name in ("canonical.jsonl", "prompts.parquet", "tokenizer.json")}
    for path in inputs.values():
        path.write_text("x")
    before = provenance(base, *inputs.values())
    inputs["canonical.jsonl"].write_text(json.dumps({"changed": True}))
    after = provenance(base, *inputs.values())
    assert before["canonical"] != after["canonical"] and before["cache_manifest"] == after["cache_manifest"]
    assert set(before["code"]) == {"data_curation/prompt_difficulty.py", "data_curation/looptool.py"}
