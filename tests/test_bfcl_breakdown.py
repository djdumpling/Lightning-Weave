"""Step-0 BFCL diagnostics: category flips and overflow, the template check, and trajectory-consistent growth."""

import pytest

from evaluation.bfcl_breakdown import (
    TEMPLATE_MARKERS,
    category_rows,
    common_no_overflow,
    history_pairs,
    template_retention,
    trajectory_consistent,
)


def entry(index, correct, overflow=False, requests=None, steps=None):
    item = {"id": f"e{index}", "is_correct": correct, "out_of_context": overflow}
    if requests is not None:
        item["requests"] = requests
        item["step_tokens"] = steps if steps is not None else [r["completion_tokens"] for r in requests]
    return item


def test_category_rows_keep_the_full_comparison_primary_and_report_overflow():
    reference = {"multi_turn_base": [entry(0, True), entry(1, True), entry(2, False), entry(3, False, overflow=True)]}
    model = {"multi_turn_base": [entry(0, True), entry(1, False, overflow=True), entry(2, True), entry(3, False)]}
    common = common_no_overflow([reference, model], ["multi_turn_base"])
    assert common == {"multi_turn_base": {"e0", "e2"}}  # one fixed subset for every compared run
    (row,) = category_rows(reference, model, ["multi_turn_base"], common=common, samples=50)
    assert (row["entries"], row["lost"], row["gained"]) == (4, 1, 1)
    assert row["delta"] == pytest.approx(0.0)
    assert row["overflow"] == {"reference_only": 1, "model_only": 1, "both": 0}
    assert row["entries_common_subset"] == 2 and row["delta_common_subset"] == pytest.approx(50.0)


class QwenLikeTemplate:
    """Keeps reasoning only for assistant messages after the last user message, as Qwen3's template does."""

    def apply_chat_template(self, messages, **_):
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        parts = []
        for index, message in enumerate(messages):
            content = message.get("content") or ""
            if message["role"] == "assistant":
                if "</think>" in content:
                    reasoning, content = content.split("</think>", 1)
                    reasoning = reasoning.replace("<think>", "")
                else:
                    reasoning = message.get("reasoning_content") or ""
                if index > last_user:
                    content = f"<think>{reasoning}</think>{content}"
            parts.append(f"{message['role']}: {content}")
        return "\n".join(parts)


def test_template_retention_reports_which_earlier_reasoning_survives():
    kept = template_retention(QwenLikeTemplate())
    assert set(kept) == set(TEMPLATE_MARKERS)
    assert kept == {
        "field_before_last_user": False,
        "field_after_last_user": True,
        "inline_before_last_user": False,
        "inline_after_last_user": True,
    }


def request(users, prompt, completion, reasoning):
    return {"users": users, "prompt_tokens": prompt, "completion_tokens": completion, "reasoning_tokens": reasoning}


def test_trajectory_consistency_needs_the_recorded_order_and_extending_users():
    ordered = [request(["a"], 100, 600, 500), request(["a", "b"], 900, 50, 10)]
    assert trajectory_consistent(entry(0, True, requests=ordered))
    assert not trajectory_consistent(entry(0, True, requests=ordered, steps=[50, 600]))  # counts match, order not
    crossed = [request(["a"], 100, 600, 500), request(["z", "b"], 900, 50, 10)]
    assert not trajectory_consistent(entry(0, True, requests=crossed))


def test_history_pairs_use_only_consistent_entries_and_do_not_claim_proof():
    kept_steps = [
        request(["a"], 100, 600, 500),  # reasoned 500 tokens
        request(["a"], 300, 400, 300),  # the prompt grew 200 < 500
        request(["a", "b"], 1_000, 50, 10),  # new turn; grew 700 >= 300
        request(["a", "b"], 1_020, 50, 10),  # previous step reasoned too little to be informative
    ]
    mixed = [request(["a"], 100, 600, 500), request(["q"], 150, 400, 300)]
    run = {"multi_turn_base": [entry(0, True, requests=kept_steps), entry(1, True), entry(2, True, requests=mixed)]}
    summary = history_pairs(run, min_reasoning=256)
    assert (summary["unreconciled_entries"], summary["ambiguous_entries"]) == (1, 1)
    within, new = summary["within_turn"], summary["new_turn"]
    assert within["pairs"] == 1 and within["growth_below_previous_reasoning_fraction"] == 1.0
    assert new["pairs"] == 1 and new["growth_below_previous_reasoning_fraction"] == 0.0
    assert "dropped_fraction" not in within
    # (growth - previous visible tokens) / previous reasoning = (700 - 100) / 300.
    assert new["median_growth_beyond_visible_over_reasoning"] == pytest.approx(2.0)
