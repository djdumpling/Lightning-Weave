import gzip
import json

from configs.bfcl_eval import config

TOOLS = [{"type": "function", "function": {"name": "cd", "parameters": {"type": "object", "properties": {}}}}]


def harness_request(step: int) -> dict:
    messages = [{"role": "user", "content": "Go to the folder."}]
    if step:
        messages += [
            {"role": "assistant", "content": "", "reasoning_content": "check the cwd",
             "tool_calls": [{"id": "1", "type": "function", "function": {"name": "cd", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
        ]
    return {"model": "m", "messages": messages, "tools": TOOLS, "max_completion_tokens": 100}


RESPONSE = {
    "choices": [{"message": {"role": "assistant", "content": "Done.", "reasoning_content": "all set"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 5},
}


def test_records_keep_the_sent_request_and_store_each_tool_list_once():
    seen: set[str] = set()
    records = []
    for step in (0, 1):
        received = harness_request(step)
        sent = config.inline_history_reasoning(received)
        records.append(config.request_log_record("multi_turn_base", received, sent, RESPONSE, seen))
    first, second = records
    assert first["tools"] == TOOLS and "tools" not in second
    assert first["tools_digest"] == second["tools_digest"] == config.request_identity(harness_request(0))["tools_digest"]
    # The identity joins usage.jsonl (computed on the harness body); the request is what the model saw.
    assert second["body_digest"] == config.request_identity(harness_request(1))["body_digest"]
    assistant = second["request"]["messages"][1]
    assert assistant["content"].startswith("<think>\ncheck the cwd\n</think>") and "reasoning_content" not in assistant
    assert "tools" not in second["request"]
    assert second["choices"] == [{"message": RESPONSE["choices"][0]["message"], "finish_reason": "stop"}]
    assert second["usage"] == RESPONSE["usage"]


def write_log(path, start):
    seen: set[str] = set()
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for step in (0, 1):
            received = harness_request(step)
            record = config.request_log_record("lane", received, received, RESPONSE, seen)
            handle.write(json.dumps({**record, "time": start}) + "\n")


def test_each_proxy_start_writes_its_own_file_and_a_cut_off_file_keeps_its_complete_lines(tmp_path):
    directory = tmp_path / config.REQUEST_LOG_DIR
    directory.mkdir()
    write_log(directory / "20261002T000000-aaaa.jsonl.gz", 0)
    first = (directory / "20261002T000000-aaaa.jsonl.gz").read_bytes()
    # A preempted container cut its file mid-data; the relaunch wrote a new file, which stays fully readable.
    (directory / "20261002T000000-aaaa.jsonl.gz").write_bytes(first[: len(first) // 2])
    write_log(directory / "20261002T010000-bbbb.jsonl.gz", 1)
    records = config.read_request_logs(directory)
    assert [record["time"] for record in records][-2:] == [1, 1] and len(records) < 4
    assert all(record["request"]["tools"] == TOOLS for record in records)
    # Losing only the gzip trailer keeps every line.
    (directory / "20261002T000000-aaaa.jsonl.gz").write_bytes(first[:-8])
    assert len(config.read_request_logs(directory)) == 4


def test_logging_is_not_part_of_the_protocol():
    assert "log_requests" not in config.PROTOCOL.resolved()
