import gzip
import json
from types import SimpleNamespace

import pytest

from configs.bfcl_eval import config as bfcl
from evaluation import decision_drift as drift

TOOLS = [{"type": "function", "function": {"name": "cd", "parameters": {"type": "object", "properties": {}}}}]


def request(lane, messages):
    return {"model": "m", "messages": messages, "tools": TOOLS, "max_completion_tokens": 10}


USER = [{"role": "user", "content": "go"}]
AFTER_TOOL = USER + [
    {"role": "assistant", "content": "<think>\nx\n</think>\n\n", "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "cd", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "1", "content": "ok"},
]


def test_histories_are_distinct_multi_turn_requests_by_kind(tmp_path):
    directory = tmp_path / bfcl.REQUEST_LOG_DIR
    directory.mkdir()
    seen: set[str] = set()
    response = {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}
    with gzip.open(directory / "a.jsonl.gz", "wt", encoding="utf-8") as handle:
        for lane, messages in (("multi_turn_base", USER), ("multi_turn_base", USER),  # a retry: one history
                               ("multi_turn_miss_func", AFTER_TOOL), ("simple_python", USER + [{"role": "user",
                                                                                               "content": "x"}])):
            body = request(lane, messages)
            handle.write(json.dumps(bfcl.request_log_record(lane, body, body, response, seen)) + "\n")
    drift.main(["histories", "--requests", str(directory), "--output", str(tmp_path / "h.jsonl")])
    histories = drift.read_jsonl(tmp_path / "h.jsonl")
    assert [(item["lane"], item["kind"]) for item in histories] == [("multi_turn_base", "turn_start"),
                                                                   ("multi_turn_miss_func", "after_tool")]
    assert all(item["tools"] == TOOLS for item in histories)
    assert drift.select_histories(bfcl.read_request_logs(directory), per_kind=0) == []


def test_sampled_responses_become_decisions_grouped_by_tool_list():
    histories = [
        {"history_id": "h1", "kind": "turn_start", "lane": "multi_turn_base", "messages": USER, "tools": TOOLS},
        {"history_id": "h2", "kind": "after_tool", "lane": "multi_turn_base", "messages": AFTER_TOOL, "tools": []},
    ]
    calls = []

    class LLM:
        def chat(self, conversations, sampling, tools=None, use_tqdm=False):
            calls.append(tools)
            return [SimpleNamespace(prompt="<|im_start|>assistant\n", outputs=[
                SimpleNamespace(text="<think>a</think>\n\nDone.", finish_reason="stop", token_ids=[1, 2, 3]),
                SimpleNamespace(text="<think>still", finish_reason="length", token_ids=[1] * 8),
            ]) for _ in conversations]

    rows = drift.sample_decisions(LLM(), SimpleNamespace(n=2), histories)
    assert len(calls) == 2 and TOOLS in calls and [] in calls  # one call per distinct tool list
    first = next(row for row in rows if row["history_id"] == "h1")
    assert json.loads(first["decisions"][0]) == {"visible": "\n\nDone.", "finish_reason": "stop"}
    assert json.loads(first["decisions"][1]) == {"visible": "<think>still", "finish_reason": "length"}
    assert first["tokens"] == [3, 8]


def rows(decisions_by_history):
    return [{"history_id": f"h{index}", "kind": drift.KINDS[index % 2], "lane": "multi_turn_base",
             "decisions": decisions, "tokens": [10] * len(decisions)}
            for index, decisions in enumerate(decisions_by_history)]


def decision(visible, finish="stop"):
    return json.dumps({"visible": visible, "finish_reason": finish}, separators=(",", ":"))


def test_drift_is_measured_above_the_recipients_own_sampling_floor():
    assert drift.total_variation(["a", "a"], ["a", "b"]) == pytest.approx(0.5)
    assert drift.match_rate(["a", "a"], ["a", "b"]) == pytest.approx(0.5)
    a, b, c = (decision(text) for text in ("A", "B", "C"))
    reference = rows([[a, a, b, a]] * 20)
    null = rows([[a, b, a, a]] * 20)
    report = drift.drift_report(reference, null, {
        "same": rows([[a, a, a, b]] * 20),
        "moved": rows([[c, c, a, c]] * 20),
    }, draws=200)
    exact = report["views"]["exact"]
    assert report["exploratory"] and report["histories"] == 20
    assert exact["null"]["total_variation"] == pytest.approx(0.0) and exact["null"]["saturated_fraction"] == 0.0
    assert exact["arms"]["same"]["all"]["excess_over_null"] == pytest.approx(0.0)
    moved = exact["arms"]["moved"]["all"]
    assert moved["excess_over_null"] == pytest.approx(0.75) and moved["excess_ci95"][0] > 0.5
    assert moved["match_deficit"] == pytest.approx(0.625 - 0.1875)
    assert set(exact["arms"]["moved"]) == {"all", "turn_start", "after_tool"}
    with pytest.raises(ValueError, match="every model"):
        drift.drift_report(reference, [], {"same": reference})


def test_saturation_is_reported_because_unique_responses_hide_any_difference():
    """Every response unique: two draws of one policy and two different policies both sit at distance 1."""
    def unique(prefix):
        return rows([[decision(f"{prefix}{history} {sample}") for sample in range(4)] for history in range(10)])

    report = drift.drift_report(unique("ref"), unique("null"), {"different": unique("other")}, draws=100)
    exact = report["views"]["exact"]
    assert exact["arms"]["different"]["all"]["excess_over_null"] == pytest.approx(0.0)  # no evidence either way
    assert exact["null"]["saturated_fraction"] == 1.0
    assert exact["repeating_fraction"] == {"reference": 0.0, "null": 0.0, "different": 0.0}


def test_the_call_level_view_ignores_wording_but_not_calls():
    call = '<tool_call>\n{"name": "cd", "arguments": {"folder": "a"}}\n</tool_call>'
    reordered = '<tool_call>\n{"arguments": {"folder": "a"}, "name": "cd"}\n</tool_call>'
    assert drift.call_signature(decision("I will move.\n\n" + call)) == drift.call_signature(decision(reordered))
    assert drift.call_signature(decision(call)) != drift.call_signature(decision(call.replace('"a"', '"b"')))
    assert drift.call_signature(decision("Done.")) == drift.call_signature(decision("Finished.")) == "reply"
    assert drift.call_signature(decision("<think>still", "length")) == "truncated"
    replies = rows([[decision(f"reply {k}") for k in range(4)]] * 6)
    report = drift.drift_report(replies, replies, {"same": replies}, draws=50)
    assert report["views"]["calls"]["null"]["saturated_fraction"] == 0.0  # replies repeat at the call level


def test_histories_are_frozen_once_chosen_and_checked_when_compared(tmp_path):
    directory = tmp_path / bfcl.REQUEST_LOG_DIR
    directory.mkdir()
    with gzip.open(directory / "a.jsonl.gz", "wt", encoding="utf-8") as handle:
        body = request("multi_turn_base", USER)
        handle.write(json.dumps(bfcl.request_log_record("multi_turn_base", body, body, {"choices": []}, set())) + "\n")
    histories = tmp_path / "histories.jsonl"
    drift.main(["histories", "--requests", str(directory), "--output", str(histories)])
    frozen = json.loads(histories.with_suffix(".frozen.json").read_text())
    assert frozen["histories"] == 1
    with pytest.raises(FileExistsError, match="frozen"):
        drift.main(["histories", "--requests", str(directory), "--output", str(histories)])
    samples = tmp_path / "s.jsonl"
    drift.write_jsonl(rows([[decision("A")] * 4]), samples)
    drift.main(["compare", "--reference", str(samples), "--null", str(samples), "--arm", f"x={samples}",
                "--histories", str(histories), "--output", str(tmp_path / "report.json")])
    assert json.loads((tmp_path / "report.json").read_text())["histories_frozen"]["sha256"] == frozen["sha256"]
    histories.write_text(histories.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        drift.main(["compare", "--reference", str(samples), "--null", str(samples), "--arm", f"x={samples}",
                    "--histories", str(histories), "--output", str(tmp_path / "report.json")])
