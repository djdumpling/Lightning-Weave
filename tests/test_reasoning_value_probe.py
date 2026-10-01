"""Reasoning-value probe: held-out selection and the shortening effect by verified-solved label."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_curation.reasoning_value_probe import evaluate, per_prompt, select_prompts


class FakeTokenizer:
    """Renders "messages|tools" and tokenizes one token per character."""

    def apply_chat_template(self, messages, tools, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and enable_thinking
        return json.dumps(messages) + "|" + json.dumps(tools)

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": list(range(len(text)))}


def row(prompt_id, kind, content, calls=True):
    target = {
        "content": "" if calls else "Which city?",
        "tool_calls": [{"function": {"name": "f", "arguments": {}}}] if calls else [],
    }
    return {
        "id": prompt_id,
        "messages": [{"role": "user", "content": content}],
        "tools": [],
        "target": target,
        "metadata": {
            "conversation_kind": kind,
            "target_kind": "single_call" if calls else "text_no_call",
            "source_index": 0,
        },
    }


def test_selection_is_held_out_call_referenced_unique_and_fills_quotas(tmp_path):
    tokenizer = FakeTokenizer()
    rows = [
        row("trained", "multi_turn", "a"),  # used in training by id
        row("same_text", "multi_turn", "b"),  # its rendered prompt was trained under another id
        row("text", "multi_turn", "c", calls=False),  # text reference: not verifiable
        row("m1", "multi_turn", "d"),
        row("m2", "multi_turn", "e"),
        row("m3", "multi_turn", "d"),  # duplicate rendered prompt of m1
        row("s1", "single_turn", "f"),
        row("long", "single_turn", "x" * 9_000),  # over the prompt budget
    ]
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text("\n".join(json.dumps(r) for r in rows))
    trained_prompt = tokenizer.apply_chat_template(rows[1]["messages"], rows[1]["tools"], False, True, True)
    pq.write_table(
        pa.table({"prompt_id": ["trained", "other"], "prompt": ["unused", trained_prompt]}), tmp_path / "p.parquet"
    )
    selected = select_prompts(
        canonical, tmp_path / "p.parquet", tokenizer, quotas={"multi_turn": 2, "single_turn": 1}, seed=0
    )
    ids = {item["prompt_id"] for item in selected}
    assert ids <= {"m1", "m2", "m3", "s1"} and len(selected) == 3 and "s1" in ids
    assert not {"m1", "m3"} <= ids  # the two identical prompts are never both kept
    with pytest.raises(ValueError, match="not enough"):
        select_prompts(canonical, tmp_path / "p.parquet", tokenizer, quotas={"single_turn": 2}, seed=0)


def records(prompt_id, policy, outcomes, tokens):
    return [
        {
            "prompt_id": prompt_id,
            "policy": policy,
            "sample": i,
            "finished": o != "unfinished",
            "outcome": o,
            "tokens": tokens,
            "think_tokens": 0,
        }
        for i, o in enumerate(outcomes)
    ]


def test_evaluate_measures_the_shortening_effect_by_label():
    prompts = [{"prompt_id": p, "conversation_kind": "multi_turn"} for p in ("easy", "hard")]
    log = [
        *records("easy", "base", ["match"] * 4, 500),  # verified solved
        *records("easy", "full", ["match"] * 4, 400),
        *records("easy", "short", ["match"] * 4, 200),  # shortening is free here
        *records("hard", "base", ["match", "miss", "miss", "unfinished"], 900),
        *records("hard", "full", ["match", "match", "miss", "miss"], 800),
        *records("hard", "short", ["miss", "miss", "miss", "unfinished"], 600),  # and costly here
    ]
    values = per_prompt(prompts, log, ("base", "full", "short"))
    assert values[1]["short"]["unfinished"] == 0.25 and values[1]["full"]["success"] == 0.5
    report = evaluate(prompts, log, draws=50)
    groups = report["groups"]
    assert groups["solved"]["prompts"] == 1 and groups["solved"]["shortening_effect"]["mean"] == 0.0
    assert groups["not_solved"]["shortening_effect"]["mean"] == -0.5
    assert groups["solved"]["total_token_ratio"] == pytest.approx(-0.5)
    assert report["solved_minus_not_solved_effect"]["mean"] == 0.5
    assert report["label_predicts_full_success"] == {"solved": 1.0, "not_solved": 0.5}
    with pytest.raises(ValueError, match="no short samples"):
        per_prompt(
            prompts,
            [r for r in log if not (r["policy"] == "short" and r["prompt_id"] == "hard")],
            ("base", "full", "short"),
        )
