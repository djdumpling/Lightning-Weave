"""BFCL cost vectors, leaderboard weights, the usage join, paired comparisons, and the proxy's request handling."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation import bfcl_efficiency as efficiency
from configs.bfcl_eval import config as requests


def counts(value=10):
    return {category: value for category in efficiency.V3_CATEGORIES}


def test_group_weights_follow_the_v3_leaderboard_scorer():
    sizes = counts()
    sizes["live_multiple"] = 30
    weights = efficiency.group_weights(sizes)
    for group in ("non_live", "live", "multi_turn", "overall"):
        assert sum(weights[group].values()) == pytest.approx(1.0)
    assert weights["non_live"]["simple_python"] == pytest.approx(1 / 15)
    assert weights["live"]["live_multiple"] == pytest.approx(30 / 80)
    assert weights["overall"]["multi_turn_base"] == pytest.approx(1 / 12)
    with pytest.raises(ValueError, match="missing"):
        efficiency.group_weights({"simple_python": 1})


MULTI_TURN_ROW = {
    "id": "multi_turn_base_0",
    "is_correct": False,
    "num_generated_tokens": 33_118,
    "num_generated_tokens_list": [100, 200, 32_768, 50],
    "generation": [[[{"ls": '{"a": 1}'}, {"ls": '{"a":1}'}], "done"], [[{"cd": "{}"}], "ok"]],
    "question": [[{"role": "user", "content": "first"}], [{"role": "user", "content": "second"}]],
}


def test_entry_costs_count_steps_calls_duplicates_turns_and_overflow():
    costs = efficiency.entry_costs(MULTI_TURN_ROW)
    assert costs["steps"] == 4 and costs["user_turns"] == 2 and costs["steps_per_turn"] == 2
    assert costs["tool_calls"] == 3 and costs["duplicate_calls"] == 1 and costs["text_steps"] == 2
    assert costs["turn_tokens"] == [[100, 200], [32_768, 50]] and costs["runaway"]
    assert len(costs["users"]) == 2
    with pytest.raises(ValueError, match="token counts"):
        efficiency.entry_costs({**MULTI_TURN_ROW, "num_generated_tokens_list": [1]})
    efficiency.entry_costs({**MULTI_TURN_ROW, "num_generated_tokens_list": [1], "error": efficiency.OUT_OF_CONTEXT})


def test_visible_text_uses_the_hermes_call_format():
    assert efficiency.visible_text([{"f": json.dumps({"x": 1})}]) == '<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call>'


def test_aes_matches_the_lightning_weave_definition():
    assert efficiency.aes_term(0.5, 0.55, 1000, 900) == pytest.approx(0.1 + 3 * 0.1)
    assert efficiency.aes_term(0.5, 0.45, 1000, 900) == pytest.approx(0.1 - 5 * 0.1)


def fake_run(seed, shift=0.0):
    rng = np.random.default_rng(seed)
    return {
        category: [
            {
                "id": f"{category}_{index}",
                "is_correct": bool(rng.random() < 0.6),
                "gen_tokens": float(rng.integers(100, 1000) + shift),
                "steps": 1, "steps_per_turn": 1.0, "tool_calls": 1, "duplicate_calls": 0, "max_step_tokens": 100,
                "runaway": False, "out_of_context": False, "turn_tokens": [[100]],
            }
            for index in range(20)
        ]
        for category in efficiency.V3_CATEGORIES
    }


def test_paired_bootstrap_is_exact_for_constant_differences_and_restricts_to_subsets():
    base = fake_run(0)
    shifted = {c: [{**e, "gen_tokens": e["gen_tokens"] + 50} for e in entries] for c, entries in base.items()}
    weights = efficiency.group_weights({c: len(e) for c, e in base.items()})["overall"]
    result = efficiency.paired_bootstrap(base, shifted, efficiency._metric("gen_tokens"), categories=weights, samples=200)
    assert result["delta"] == pytest.approx(50.0) and result["ci95"] == pytest.approx([50.0, 50.0])
    both = efficiency.paired_bootstrap(
        base, shifted, efficiency._metric("gen_tokens"), categories=weights, keep=lambda b, m: b["is_correct"], samples=20
    )
    assert both["delta"] == pytest.approx(50.0) and both["pairs"] == sum(e["is_correct"] for es in base.values() for e in es)


def test_summary_uses_leaderboard_weights_for_means_and_quantiles():
    run = fake_run(1)
    summary = efficiency.summarize(run)
    weights = efficiency.group_weights({c: len(e) for c, e in run.items()})["overall"]
    expected = sum(w * np.mean([x["is_correct"] for x in run[c]]) for c, w in weights.items())
    assert summary["groups"]["overall"]["accuracy"] == pytest.approx(expected)
    values = np.asarray([1.0, 2.0, 3.0, 100.0])
    assert efficiency.weighted_quantile(values, np.asarray([0.7, 0.1, 0.1, 0.1]), 0.5) == 1.0
    assert efficiency.weighted_quantile(values, np.ones(4), 0.5) == 2.0


def test_comparison_reports_the_paired_log_ratio_strata_and_a_decision():
    base = fake_run(2)
    cheaper = {c: [{**e, "gen_tokens": (e["gen_tokens"] + 1) * 0.5 - 1} for e in entries] for c, entries in base.items()}
    report = efficiency.compare(base, {"cheaper": cheaper}, samples=50)["models"]["cheaper"]
    assert report["gen_tokens_relative"]["overall"]["relative"] == pytest.approx(-0.5)
    assert report["gen_tokens_relative"]["overall"]["ci95"] == pytest.approx([-0.5, -0.5])
    assert report["decision"]["passes"]
    assert report["decision"]["rule"]["guard_tolerances"] == efficiency.GUARD_TOLERANCES
    assert sum(report["outcome_strata"]["overall"].values()) == pytest.approx(1.0)
    assert report["multi_turn_common_horizon_tokens"]["delta"] == 0.0


def test_guards_require_the_upper_bound_within_tolerance():
    def deltas(**guards):
        overall = {"is_correct": {"ci95": [0.0, 0.0]}}
        overall.update({metric: {"ci95": list(guards.get(metric, (0.0, 0.0)))} for metric in efficiency.GUARD_TOLERANCES})
        return {"overall": overall}

    options = {"irrelevance": {"ci95": [0.0, 0.0]}, "tokens": {"ci95": [-0.2, -0.1]}, "margin": 0.01, "min_reduction": 0.05}
    assert efficiency.decision(deltas(), **options)["passes"]
    # an interval that includes zero but permits a large increase is not non-inferior
    wide = efficiency.decision(deltas(duplicate_calls=(-0.1, 5.0)), **options)
    assert not wide["non_inferior_duplicate_calls"] and not wide["passes"]
    tolerated = efficiency.GUARD_TOLERANCES["runaway"]
    assert efficiency.decision(deltas(runaway=(0.001, tolerated)), **options)["non_inferior_runaway"]
    assert not efficiency.decision(deltas(runaway=(0.001, 2 * tolerated)), **options)["non_inferior_runaway"]


def request(users, tools, lane, completion, time, messages=None):
    body = {"messages": messages or [{"role": "user", "content": text} for text in users], "tools": tools}
    return {"lane": lane, **requests.request_identity(body), "prompt_tokens": 10, "completion_tokens": completion,
            "reasoning_tokens": 3, "finish_reason": "stop", "time": time}


def test_history_reasoning_counts_what_the_harness_sends_back():
    body = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "reasoning_content": "abcd", "tool_calls": []},
            {"role": "tool", "content": "1"},
            {"role": "assistant", "content": "<think>xyz</think>answer"},
            {"role": "assistant", "content": "tail</think>answer"},  # an opening tag the template already emitted
            {"role": "assistant", "content": [{"type": "text", "text": "plain"}]},
        ]
    }
    assert requests.history_reasoning(body) == {"history_assistant_messages": 4, "history_reasoning_chars": 4 + 3 + 4}
    assert requests.history_reasoning({"messages": [{"role": "user", "content": "q"}]})["history_reasoning_chars"] == 0


def test_request_identity_matches_the_entry_side_digests():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    body = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}], "tools": tools}
    identity = requests.request_identity(body)
    entry = efficiency.entry_costs(
        {"id": "a", "is_correct": True, "num_generated_tokens": 5, "generation": "x",
         "question": [[{"role": "user", "content": "q"}]], "tools": tools}
    )
    assert identity["users"] == entry["users"] and identity["tools_digest"] == entry["tools_digest"]


def test_usage_join_handles_tools_identical_first_turns_retries_and_injected_turns(tmp_path):
    tools_a = [{"type": "function", "function": {"name": "f", "description": "a"}}]
    tools_b = [{"type": "function", "function": {"name": "f", "description": "b"}}]
    single = lambda i, tools, tokens: efficiency.entry_costs(  # noqa: E731
        {"id": i, "is_correct": True, "num_generated_tokens": tokens, "generation": "x",
         "question": [[{"role": "user", "content": "same"}]], "tools": tools}
    )
    multi = lambda i, turns, tokens: efficiency.entry_costs(  # noqa: E731
        {"id": i, "is_correct": True, "num_generated_tokens": sum(tokens), "num_generated_tokens_list": tokens,
         "generation": [[f"t{k}"] for k in range(len(tokens))],
         "question": [[{"role": "user", "content": text}] if text else [] for text in turns]}
    )
    run = {
        "live_simple": [single("a", tools_a, 5), single("b", tools_b, 7)],
        "multi_turn_base": [multi("m1", ["first", "x"], [11, 12]), multi("m2", ["first", "y"], [13, 14])],
        # miss_func: the empty second turn is filled with a message the harness writes itself.
        "multi_turn_miss_func": [multi("f1", ["go", None, "more"], [21, 22, 23])],
        # A single turn may hold several user messages.
        "live_irrelevance": [
            efficiency.entry_costs(
                {"id": "r", "is_correct": True, "num_generated_tokens": 9, "generation": "x", "tools": tools_a,
                 "question": [[{"role": "user", "content": "u1"}, {"role": "user", "content": "u2"}]]}
            )
        ],
    }
    log = [
        request(["same"], tools_a, "live_simple", 5, 1.0),
        request(["same"], tools_a, "live_simple", 900, 2.0),  # a duplicate attempt the harness did not keep
        request(["same"], tools_b, "live_simple", 7, 1.0),
        # m1 and m2 share their first turn, so they send byte-identical first requests.
        request(["first"], [], "multi_turn_base", 11, 1.0),
        request(["first"], [], "multi_turn_base", 13, 1.5),
        request(["first", "x"], [], "multi_turn_base", 12, 2.0),
        request(["first", "y"], [], "multi_turn_base", 14, 2.0),
        request(["go"], [], "multi_turn_miss_func", 21, 1.0),
        request(["go", "injected"], [], "multi_turn_miss_func", 22, 2.0),
        request(["go", "injected", "more"], [], "multi_turn_miss_func", 23, 3.0),
        request(["u1", "u2"], tools_a, "live_irrelevance", 9, 1.0),
    ]
    path = tmp_path / "usage.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in log))
    coverage = efficiency.attach_usage(run, path)
    assert all(item["reconciled"] == item["entries"] for item in coverage.values())
    assert coverage["live_simple"]["surplus_requests"] == 1
    assert run["multi_turn_base"][0]["prompt_tokens"] == 20 and run["live_simple"][0]["prompt_tokens"] == 10
    # Reconciled entries keep the requests the harness kept, in time order (the retry is not among them).
    assert [r["completion_tokens"] for r in run["multi_turn_miss_func"][0]["requests"]] == [21, 22, 23]
    assert [r["completion_tokens"] for r in run["live_simple"][0]["requests"]] == [5]
    path.write_text("\n".join(json.dumps(record) for record in log[:-2]))
    with pytest.raises(ValueError, match="reconcile"):
        efficiency.attach_usage({k: [dict(e) for e in v] for k, v in run.items()}, path)


def test_proxy_applies_the_cap_and_system_prompt_without_mutating_the_request():
    body = {"messages": [{"role": "user", "content": "q"}], "max_completion_tokens": 32_768}
    capped = requests.rewrite_request(body, SimpleNamespace(max_completion_tokens=4096, system_prompt="be brief"))
    assert capped["max_completion_tokens"] == 4096 and capped["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["max_completion_tokens"] == 32_768 and len(body["messages"]) == 1


def entry(index, correct, overflow=False):
    return {"id": f"e{index}", "is_correct": correct, "out_of_context": overflow}


def test_category_rows_keep_the_full_comparison_primary_and_report_overflow():
    reference = {"multi_turn_base": [entry(0, True), entry(1, True), entry(2, False), entry(3, False, overflow=True)]}
    model = {"multi_turn_base": [entry(0, True), entry(1, False, overflow=True), entry(2, True), entry(3, False)]}
    common = efficiency.common_no_overflow([reference, model], ["multi_turn_base"])
    assert common == {"multi_turn_base": {"e0", "e2"}}  # one fixed subset for every compared run
    (row,) = efficiency.category_rows(reference, model, ["multi_turn_base"], common=common, samples=50)
    assert (row["entries"], row["lost"], row["gained"]) == (4, 1, 1)
    assert row["delta"] == pytest.approx(0.0)
    assert row["overflow"] == {"reference_only": 1, "model_only": 1, "both": 0}
    assert row["entries_common_subset"] == 2 and row["delta_common_subset"] == pytest.approx(50.0)


def test_token_only_loading_skips_the_usage_join(tmp_path):
    output = tmp_path / "bfcl_v3.multi_turn_base" / "output.jsonl"
    output.parent.mkdir()
    output.write_text(json.dumps(MULTI_TURN_ROW) + "\n")
    (tmp_path / "usage.jsonl").write_text("an incomplete usage log")
    with pytest.raises(json.JSONDecodeError):
        efficiency.load_run(tmp_path)
    (entry,) = efficiency.load_run(tmp_path, include_usage=False)["multi_turn_base"]
    assert entry["gen_tokens"] == MULTI_TURN_ROW["num_generated_tokens"]
    assert entry["is_correct"] == MULTI_TURN_ROW["is_correct"]
