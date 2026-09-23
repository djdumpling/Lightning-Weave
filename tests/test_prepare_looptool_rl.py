"""Offline tests for the LoopTool-23k RL preprocessing pipeline."""

import hashlib
import json

import pytest

pytest.importorskip("jsonschema")
pytest.importorskip("datasketch")

from data_curation import looptool as L  # noqa: E402
from data_curation import prepare_looptool_rl as P  # noqa: E402

WEATHER = {
    "name": "get_weather",
    "description": "Weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "days": {"type": "integer", "default": 1}},
        "required": ["city"],
        "optional": [],
    },
    "category": "weather",
}
SEARCH = {
    "name": "search",
    "description": "Search the web.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}
POLICY = "You are an expert in composing functions.\n\nThe current time is 2024-01-01, Monday."


def instruction(tools=(WEATHER, SEARCH), policy=POLICY):
    body = "".join(json.dumps(tool) + "\n" for tool in tools)
    return f"{policy}\n{L.TOOLS_HEADER}{body}{L.TOOLS_FOOTERS[0]}"


def call(name, **arguments):
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


def response(text):
    return f"<tool_response>\n{text}\n</tool_response>"


def record(input_text, output, tools=(WEATHER, SEARCH)):
    return {"instruction": instruction(tools), "input": input_text, "output": output}


def parse(rec, index=0, **kwargs):
    return L.parse_example(rec, index=index, dataset="ds", revision="rev", checker=L.SchemaChecker(), **kwargs)


def reject_reason(rec, **kwargs):
    with pytest.raises(L.Reject) as info:
        parse(rec, **kwargs)
    return info.value.reason


MULTI_TURN = (
    "What is the weather in Paris?<|im_end|>\n"
    f"<|im_start|>assistant\n{call('get_weather', city='Paris')}\n{call('search', query='Paris news')}<|im_end|>\n"
    f"<|im_start|>user\n{response('{"temp": 20}')}\n{response('{"hits": []}')}<|im_end|>\n"
    "<|im_start|>assistant\nIt is 20 degrees in Paris.<|im_end|>\n"
    "<|im_start|>user\nAnd in Rome for 3 days?"
)


class FakeTokenizer:
    """Deterministic stand-in for a Qwen chat template: one token per whitespace word."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        self.calls.append({"messages": messages, "tools": tools, "tokenize": tokenize, "gen": add_generation_prompt})
        parts = []
        if tools:
            parts.append("<|im_start|>system\n" + " ".join(json.dumps(tool) for tool in tools) + "<|im_end|>\n")
        for message in messages:
            calls = " ".join(
                f"<tool_call> {call['function']['name']} {json.dumps(call['function']['arguments'])} </tool_call>"
                for call in message.get("tool_calls", [])
            )
            parts.append(f"<|im_start|>{message['role']}\n{message.get('content', '')} {calls}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n<think>\n")
        text = "".join(parts)
        return list(range(len(text.split()))) if tokenize else text


def args_for(tmp_path, *extra):
    return P.parse_args(["--output-dir", str(tmp_path), "--bfcl-audit", "off", "--workers", "1", *extra])


def run_pipeline(tmp_path, records, *extra):
    return P.run(
        args_for(tmp_path, *extra),
        records=records,
        source_info={
            "dataset": "ds",
            "requested_revision": "rev",
            "resolved_revision": "rev",
            "source_rows": len(records),
            "processed_rows": len(records),
            "source_content_sha256": "x",
        },
        tokenizer=FakeTokenizer(),
        tokenizer_info={"tokenizer": "fake", "requested_revision": None, "resolved_revision": None},
    )


# ---------------------------------------------------------------------------
# Parsing


def test_plain_single_turn_input_and_tool_extraction():
    row = parse(record("  What is the weather in Paris?\r\n", call("get_weather", city="Paris"))).row
    system, user = row["messages"]
    assert system == {"role": "system", "content": POLICY}
    assert "<tools>" not in system["content"] and "# Tools" not in system["content"]
    assert user == {"role": "user", "content": "What is the weather in Paris?"}
    assert [tool["function"]["name"] for tool in row["tools"]] == ["get_weather", "search"]
    assert row["tools"][0]["function"]["category"] == "weather"
    assert row["tools"][0]["function"]["parameters"]["properties"]["days"]["default"] == 1
    assert row["metadata"]["conversation_kind"] == "single_turn"
    assert row["metadata"]["target_kind"] == "single_call"
    assert row["target"] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
    }


def test_input_without_control_tokens_is_not_split_on_prose_labels():
    text = "Historical dialogue:\nInquirer: hi\nAssistant: hello\nInquirer: weather?"
    row = parse(record(text, call("get_weather", city="Oslo"))).row
    assert [message["role"] for message in row["messages"]] == ["system", "user"]
    assert row["messages"][1]["content"] == text


def test_qwen_multi_turn_with_plain_prefix_history_calls_and_responses():
    row = parse(record(MULTI_TURN, call("get_weather", city="Rome", days=3))).row
    roles = [message["role"] for message in row["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant", "user"]
    assistant = row["messages"][2]
    assert assistant["content"] == ""
    assert [call["id"] for call in assistant["tool_calls"]] == ["call_0", "call_1"]
    assert [call["function"]["name"] for call in assistant["tool_calls"]] == ["get_weather", "search"]
    assert [row["messages"][3]["tool_call_id"], row["messages"][4]["tool_call_id"]] == ["call_0", "call_1"]
    assert row["messages"][3]["content"] == '{"temp": 20}'
    assert row["messages"][5] == {"role": "assistant", "content": "It is 20 degrees in Paris."}
    meta = row["metadata"]
    assert (meta["conversation_kind"], meta["num_user_messages"], meta["num_history_tool_calls"]) == (
        "multi_turn",
        2,
        2,
    )
    assert meta["num_tool_responses"] == 2
    for message in row["messages"]:
        assert all(token not in message.get("content", "") for token in L.FRAMING_TOKENS)
    assert L.check_row({**row, "metadata": {**meta, "prompt_tokens": 1}}) == []


def test_input_starting_with_role_marker_and_deterministic_call_ids():
    text = "<|im_start|>user\nWeather?<|im_end|>\n" + MULTI_TURN.split("<|im_end|>\n", 1)[1]
    first = parse(record(text, call("search", query="x")), index=1).row
    second = parse(record(text, call("search", query="x")), index=99).row
    assert first["messages"][1] == {"role": "user", "content": "Weather?"}
    assert first["messages"] == second["messages"]
    assert first["id"] != second["id"]
    assert L.prompt_hash(first) == L.prompt_hash(second)


def test_parallel_text_and_mixed_targets():
    parallel = parse(record("Hi", call("get_weather", city="A") + "\n" + call("get_weather", city="B"))).row
    assert parallel["metadata"]["target_kind"] == "parallel_call"
    assert [c["function"]["arguments"]["city"] for c in parallel["target"]["tool_calls"]] == ["A", "B"]

    text = parse(record("Book me a flight", "  Which city are you flying from?\r\n")).row
    assert text["target"] == {"role": "assistant", "content": "Which city are you flying from?", "tool_calls": []}
    assert text["metadata"]["target_kind"] == "text_no_call"

    example = parse(record("Hi", "Let me check.\n" + call("search", query="q")))
    assert example.row["target"]["content"] == "Let me check."
    assert example.row["metadata"]["target_has_text_and_calls"] is True
    assert "target_text_with_tool_calls" in {event.code for event in example.events}


def test_explicit_think_blocks_are_removed_and_unclosed_rejected():
    text = "Hi<|im_end|>\n<|im_start|>assistant\n<think>\nprivate\n</think>\n\nHello there.<|im_end|>\n<|im_start|>user\nWeather?"
    example = parse(record(text, "<think>plan</think>\n" + call("get_weather", city="X")))
    assert example.row["messages"][2] == {"role": "assistant", "content": "Hello there."}
    assert "private" not in json.dumps(example.row) and "plan" not in json.dumps(example.row["target"])
    codes = {event.code for event in example.events}
    assert {"dialogue_think_removed", "target_think_removed"} <= codes
    unclosed = (
        "Hi<|im_end|>\n<|im_start|>assistant\n<think>\nreasoning that never ends<|im_end|>\n<|im_start|>user\nOk"
    )
    assert reject_reason(record(unclosed, "Sure.")) == "unbalanced_think_tags"


@pytest.mark.parametrize(
    "text,reason",
    [
        ("Hi<|im_end|>\n\nstray text<|im_end|>\n<|im_start|>user\nx", "im_end_without_im_start"),
        ("Hi<|im_start|>assistant\nx<|im_end|>\n<|im_start|>user\ny", "im_start_inside_open_message"),
        ("Hi<|im_end|>\nloose<|im_start|>user\ny", "text_outside_role_block"),
        ("Hi<|im_end|>\n<|im_start|>narrator\ny<|im_end|>\n<|im_start|>user\nz", "unsupported_role"),
        ("Hi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n", "prompt_ends_with_assistant"),
    ],
)
def test_malformed_role_structure_is_rejected(text, reason):
    assert reject_reason(record(text, "Fine.")) == reason


@pytest.mark.parametrize(
    "output,reason",
    [
        ('<tool_call>\n{"name": "search", "arguments": {}}', "unbalanced_tool_call_tags"),
        ('{"name": "search"}</tool_call>', "unbalanced_tool_call_tags"),
        ("<tool_call>\n{name: search}\n</tool_call>", "tool_call_invalid_json"),
        ('<tool_call>\n{"name": "search", "arguments": "{}"}\n</tool_call>', "tool_call_arguments_not_object"),
        ('<tool_call>\n{"name": "search", "arguments": [1]}\n</tool_call>', "tool_call_arguments_not_object"),
        ('<tool_call>\n{"name": "search", "arguments": {}, "id": 1}\n</tool_call>', "tool_call_unexpected_keys"),
        (call("launch_rocket", target="moon"), "target_call_unknown_tool"),
        (call("search", query="a") + "\ntrailing words", "assistant_text_after_tool_call"),
        ("   ", "empty_target"),
    ],
)
def test_malformed_targets_are_rejected(output, reason):
    assert reject_reason(record("Hi", output)) == reason


def test_history_pairing_failures():
    unknown = f"Hi<|im_end|>\n<|im_start|>assistant\n{call('nope')}<|im_end|>\n<|im_start|>user\n{response('x')}"
    assert reject_reason(record(unknown, "Done.")) == "history_call_unknown_tool"
    mismatch = f"Hi<|im_end|>\n<|im_start|>assistant\n{call('search', query='a')}<|im_end|>\n<|im_start|>user\n{response('x')}{response('y')}"
    assert reject_reason(record(mismatch, "Done.")) == "tool_response_count_mismatch"
    orphan = f"Hi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n<|im_start|>user\n{response('x')}"
    assert reject_reason(record(orphan, "Done.")) == "tool_response_without_tool_call"
    unanswered = (
        f"Hi<|im_end|>\n<|im_start|>assistant\n{call('search', query='a')}<|im_end|>\n<|im_start|>user\nthanks"
    )
    assert reject_reason(record(unanswered, "Done.")) == "tool_call_without_tool_response"
    mixed = f"Hi<|im_end|>\n<|im_start|>assistant\n{call('search', query='a')}<|im_end|>\n<|im_start|>user\nnote {response('x')}"
    assert reject_reason(record(mixed, "Done.")) == "tool_response_mixed_with_user_text"


def test_duplicate_tool_definitions():
    conflicting = dict(SEARCH, description="Different.")
    assert reject_reason(record("Hi", "Ok.", tools=(SEARCH, conflicting))) == "conflicting_duplicate_tool_name"
    example = parse(record("Hi", "Ok.", tools=(SEARCH, WEATHER, SEARCH)))
    assert [tool["function"]["name"] for tool in example.row["tools"]] == ["search", "get_weather"]
    assert example.row["metadata"]["num_tools"] == 2
    assert "identical_duplicate_tool_definition_removed" in example.row["metadata"]["repair_codes"]


@pytest.mark.parametrize(
    "parameters,reason",
    [
        ({"type": "object", "properties": {"x": {"type": "Set[str]"}}}, "tool_schema_invalid_draft7"),
        ({"type": "object", "properties": [], "required": []}, "tool_schema_invalid_draft7"),
        ("object", "tool_parameters_not_object"),
        ({"type": "string"}, "tool_schema_root_not_object"),
        (
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["b"]},
            "tool_schema_required_properties_mismatch",
        ),
        (
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a", "a"]},
            "tool_schema_invalid_draft7",
        ),
    ],
)
def test_invalid_schemas_are_rejected(parameters, reason):
    tool = {"name": "bad", "description": "", "parameters": parameters}
    assert reject_reason(record("Hi", "Ok.", tools=(tool,))) == reason


def test_schema_error_details_are_order_independent():
    parameters = {"type": "object", "properties": {name: {"type": "str, optional"} for name in "zyxabc"}}
    details = L.SchemaChecker().schema_error(parameters)
    assert details.endswith("at /properties/a/type")
    assert details == L.SchemaChecker().schema_error(json.loads(json.dumps(parameters)))


def test_nameless_and_non_object_tools_are_rejected():
    assert (
        reject_reason(record("Hi", "Ok.", tools=({"name": "", "parameters": {"type": "object"}},)))
        == "tool_name_missing_or_empty"
    )
    rec = {"instruction": f"{POLICY}\n{L.TOOLS_HEADER}[1, 2]\n{L.TOOLS_FOOTERS[0]}", "input": "Hi", "output": "Ok."}
    assert reject_reason(rec) == "tool_definition_not_object"
    assert reject_reason({"instruction": POLICY, "input": "Hi", "output": "Ok."}) == "tool_block_header_not_recognized"


def test_schema_repairs_are_logged_and_optional():
    tool = {
        "name": "calc",
        "description": "d",
        "parameters": {
            "type": "dict",
            "properties": {"x": {"type": "float"}, "tags": {"type": "list", "items": {"type": "str"}}},
        },
    }
    example = parse(record("Hi", call("calc", x=1.5, tags=["a"]), tools=(tool,)))
    parameters = example.row["tools"][0]["function"]["parameters"]
    assert parameters == {
        "type": "object",
        "properties": {"x": {"type": "number"}, "tags": {"type": "array", "items": {"type": "string"}}},
    }
    repairs = sorted(event.details for event in example.events if event.code == "schema_type_alias")
    assert repairs == [
        "calc/:dict->object",
        "calc/properties/tags/items:str->string",
        "calc/properties/tags:list->array",
        "calc/properties/x:float->number",
    ]
    assert example.row["metadata"]["repair_codes"] == ["schema_type_alias"]
    assert reject_reason(record("Hi", "Ok.", tools=(tool,)), type_aliases=False) == "tool_schema_invalid_draft7"

    empty = parse(
        record("Hi", call("ping"), tools=({"name": "ping", "description": "", "parameters": {"type": "object"}},))
    )
    assert empty.row["tools"][0]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert "schema_empty_object_properties_added" in empty.row["metadata"]["repair_codes"]

    compound = {
        "name": "compound",
        "description": "d",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "str, optional"},
                "matrix": {"type": "List[List[float]]"},
                "point": {"type": "Tuple[float, float]"},
                "vertices": {"type": "List[Tuple[float, float]]"},
            },
            "required": ["matrix"],
        },
    }
    repaired = parse(record("Hi", call("compound", matrix=[[1.0]]), tools=(compound,)))
    parameters = repaired.row["tools"][0]["function"]["parameters"]
    assert parameters["properties"]["label"] == {"type": "string"}
    assert parameters["properties"]["matrix"] == {
        "type": "array",
        "items": {"type": "array", "items": {"type": "number"}},
    }
    assert parameters["properties"]["vertices"] == {
        "type": "array",
        "items": {
            "type": "array",
            "items": [{"type": "number"}, {"type": "number"}],
            "minItems": 2,
            "maxItems": 2,
        },
    }
    assert parameters["properties"]["point"] == {
        "type": "array",
        "items": [{"type": "number"}, {"type": "number"}],
        "minItems": 2,
        "maxItems": 2,
    }
    assert parameters["required"] == ["matrix"]
    assert {
        "schema_type_alias",
        "schema_optional_suffix_removed",
        "schema_list_type_expanded",
        "schema_tuple_type_expanded",
    } <= set(repaired.row["metadata"]["repair_codes"])
    assert reject_reason(record("Hi", "Ok.", tools=(compound,)), type_aliases=False) == "tool_schema_invalid_draft7"


def test_target_schema_violations_are_rejected_but_history_violations_are_warnings():
    assert (
        reject_reason(record("Hi", call("get_weather", city="Paris", days="3"))) == "target_arguments_schema_mismatch"
    )

    bad_history = MULTI_TURN.replace(
        call("get_weather", city="Paris"),
        call("get_weather", city="Paris", days="3", extra=True),
    )
    example = parse(record(bad_history, call("get_weather", city="Rome", days=3)))
    first_history_call = next(message for message in example.row["messages"] if message["role"] == "assistant")[
        "tool_calls"
    ][0]["function"]
    assert first_history_call["arguments"] == {"city": "Paris", "days": "3", "extra": True}
    codes = set(example.row["metadata"]["warning_codes"])
    assert {"history_arguments_schema_type", "history_arguments_undeclared_parameter"} <= codes


# ---------------------------------------------------------------------------
# Conflicts and deduplication


def rows_from(*records):
    return [parse(rec, index=index).row for index, rec in enumerate(records)]


def test_conflicting_targets_are_excluded_but_parallel_order_is_not_a_conflict():
    two = call("get_weather", city="A") + call("get_weather", city="B")
    swapped = call("get_weather", city="B") + call("get_weather", city="A")
    rows = rows_from(
        record("Weather in A and B?", two),
        record("Weather in A and B?", swapped),
        record("Weather in C?", call("get_weather", city="C")),
        record("Weather in C?", call("get_weather", city="D")),
        record("Weather in C?", "Which C?"),
    )
    kept, conflicts, groups = P.find_conflicts(rows)
    assert [row["metadata"]["source_index"] for row in kept] == [0, 1]
    assert sorted(item["row"]["metadata"]["source_index"] for item in conflicts) == [2, 3, 4]
    assert len(groups) == 1 and groups[0]["target_kinds"] == ["single_call", "text_no_call"]
    assert conflicts[0]["row"]["messages"] == rows[2]["messages"]


def test_exact_dedup_keeps_lowest_index_and_audits():
    rows = rows_from(
        record("Weather?", call("get_weather", city="A")),
        record("Weather?", call("get_weather", city="A")),
        record("Weather?", call("get_weather", city="A"), tools=(SEARCH, WEATHER)),
    )
    kept, removed = P.exact_dedup(list(reversed(rows)))
    assert [row["metadata"]["source_index"] for row in kept] == [0, 2]
    assert removed == [
        {
            "removed_id": rows[1]["id"],
            "removed_source_index": 1,
            "representative_id": rows[0]["id"],
            "representative_source_index": 0,
            "canonical_hash": L.example_hash(rows[0]),
        }
    ]


def long_request(words, changes=()):
    tokens = [f"w{index}" for index in range(words)]
    for position in changes:
        tokens[position] = f"changed{position}"
    return " ".join(tokens)


def measured(rows):
    for row in rows:
        row["metadata"]["prompt_tokens"] = P.measure_row(FakeTokenizer(), row)["prompt_tokens"]
    return rows


def near(rows, **kwargs):
    options = {"threshold": 0.95, "num_perm": 256, "lsh_threshold": 0.5, "seed": 1, "shingle_size": 5}
    return P.near_dedup(rows, **{**options, **kwargs})


def test_near_dedup_verifies_lsh_candidates_with_exact_jaccard():
    rows = measured(
        rows_from(
            record(long_request(300), "Noted."),
            record(long_request(300, [150]), "Noted."),
            record(long_request(300, [30, 90, 150, 210, 270]), "Noted."),
        )
    )
    kept, clusters, stats = near(rows)
    assert [row["metadata"]["source_index"] for row in kept] == [0, 2]
    [cluster] = clusters
    assert cluster["representative"]["source_index"] == 0
    sets = [P.shingles(P.words(P.dialogue_text(row)), 5) for row in rows]
    assert [(m["source_index"], m["jaccard"]) for m in cluster["removed"]] == [
        (1, round(P.jaccard(sets[0], sets[1]), 6))
    ]
    assert 0.5 < P.jaccard(sets[0], sets[2]) < 0.95
    assert stats["lsh_candidate_pairs_checked"] >= 2 and stats["verified_pairs_removed"] == 1


def test_near_dedup_is_not_transitive():
    # J(A,B) = J(B,C) = 291/301 >= 0.95, but J(A,C) = 286/306 < 0.95.
    rows = measured(
        rows_from(
            record(long_request(300), "Noted."),
            record(long_request(300, [100]), "Noted."),
            record(long_request(300, [100, 200]), "Noted."),
        )
    )
    kept, clusters, _ = near(rows)
    assert [row["metadata"]["source_index"] for row in kept] == [0, 2]
    assert [member["source_index"] for member in clusters[0]["removed"]] == [1]


def test_near_dedup_respects_behavior_strata():
    text = long_request(200)
    rows = measured(
        rows_from(
            record(text, call("search", query="q")),
            record(text + " w9999", call("search", query="q") + call("search", query="q")),
            record(text + " w9998", "What do you mean?"),
            record(text + " w9997", call("search", query="q"), tools=(SEARCH, WEATHER)),
            record(text + " w9996", call("get_weather", city="q")),
        )
    )
    kept, clusters, _ = near(rows)
    assert len(kept) == 5 and clusters == []


def test_near_dedup_representative_is_shortest_prompt_then_bytes_then_index():
    rows = measured(rows_from(record(long_request(300) + " tail", "Noted."), record(long_request(300), "Noted.")))
    kept, clusters, _ = near(rows)
    assert [row["metadata"]["source_index"] for row in kept] == [1]
    assert clusters[0]["representative"]["source_index"] == 1
    tied = measured(rows_from(record(long_request(300, [5]), "Noted."), record(long_request(300, [6]), "Noted.")))
    assert [row["metadata"]["source_index"] for row in near(tied, threshold=0.9)[0]] == [0]


def test_short_texts_are_not_near_deduplicated():
    rows = measured(rows_from(record("hi", "ok"), record("hi", "ok.")))
    kept, _, stats = near(rows, shingle_size=5)
    assert len(kept) == 2 and stats["rows_below_min_words"] == 2


# ---------------------------------------------------------------------------
# Tokenizer rendering and prompt limits


def test_measure_row_uses_native_template_without_target():
    row = parse(record(MULTI_TURN, call("get_weather", city="Rome"))).row
    tokenizer = FakeTokenizer()
    result = P.measure_row(tokenizer, row)
    prompt_call, appended = tokenizer.calls[0], tokenizer.calls[-1]
    assert prompt_call["tokenize"] is True and prompt_call["gen"] is True
    assert prompt_call["messages"] == row["messages"] and prompt_call["tools"] == row["tools"]
    assert all(
        message["role"] != "assistant" or "Rome" not in json.dumps(message) for message in prompt_call["messages"]
    )
    rendered = tokenizer.apply_chat_template(row["messages"], tools=row["tools"], add_generation_prompt=True)
    assert result["prompt_tokens"] == len(rendered.split())
    assert result["system_tool_tokens"] + result["history_tokens"] == result["prompt_tokens"]
    assert result["error"] is None
    assert appended["gen"] is False and appended["messages"][-1]["tool_calls"] == row["target"]["tool_calls"]


def test_measure_row_detects_unrenderable_target():
    class Broken(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            text = super().apply_chat_template(messages, **kwargs)
            return (
                text.replace("system", "changed")
                if isinstance(text, str) and messages[-1]["role"] == "assistant"
                else text
            )

    row = parse(record("Hi", call("search", query="q"))).row
    assert P.measure_row(Broken(), row)["error"] == "target rendering changes the prompt prefix"


def synthetic(meta_rows):
    return [
        {
            "id": f"r{index}",
            "metadata": {
                "prompt_tokens": tokens,
                "conversation_kind": conversation,
                "target_kind": target,
                "num_user_messages": 1,
                "num_history_tool_calls": 0,
                "num_target_tool_calls": 0 if target == "text_no_call" else 1,
            },
        }
        for index, (tokens, conversation, target) in enumerate(meta_rows)
    ]


def test_automatic_threshold_selection_and_fallback():
    rows = synthetic(
        [(1000, "single_turn", "single_call")] * 400
        + [(3000, "multi_turn", "single_call")] * 400
        + [(3000, "multi_turn", "parallel_call")] * 180
        + [(3000, "single_turn", "text_no_call")] * 20
        + [(6000, "multi_turn", "parallel_call")] * 1
    )
    evaluations = P.evaluate_limits(rows, (2048, 4096, 8192, 16384))
    by_limit = {item["limit"]: item for item in evaluations}
    assert not by_limit[2048]["satisfies_all"]
    assert by_limit[4096]["satisfies_all"] and by_limit[4096]["excluded"] == 1
    selected = P.select_limit(evaluations, None)
    assert (selected["automatic"], selected["override"], selected["effective"]) == (4096, None, 4096)
    override = P.select_limit(evaluations, 2048)
    assert (override["automatic"], override["override"], override["effective"]) == (4096, 2048, 2048)

    skewed = synthetic([(1000, "single_turn", "single_call")] * 50 + [(20000, "multi_turn", "single_call")] * 50)
    fallback = P.select_limit(P.evaluate_limits(skewed, (2048, 16384)), None)
    assert fallback["automatic"] == 16384
    assert set(fallback["fallback_misses"]) == {
        "overall_retention>=95%",
        "each_stratum_retention>=90%",
        "max_abs_pp_shift<=0.5",
    }


def test_default_length_candidates_include_selected_10k_cap():
    assert P.LENGTH_CANDIDATES == (2048, 4096, 8192, 10240, 12288, 16384)
    assert P.RECOMMENDED_MAX_RESPONSE_TOKENS == 32768
    assert P.MODEL_NATIVE_CONTEXT_TOKENS == 262144


def test_source_row_count_is_enforced():
    args = P.parse_args([])
    with pytest.raises(SystemExit, match="expected 23040"):
        P.check_source_rows(23_039, args)
    P.check_source_rows(23_040, args)


# ---------------------------------------------------------------------------
# End-to-end with the fake tokenizer


def fixture_records():
    return [
        record("What is the weather in Paris?", call("get_weather", city="Paris")),
        record(MULTI_TURN, call("get_weather", city="Rome", days=3)),
        record("Book a table", "For how many people?"),
        record("Weather in A and B?", call("get_weather", city="A") + call("get_weather", city="B")),
        record("What is the weather in Paris?", call("get_weather", city="Paris")),  # exact duplicate of 0
        record("Book a table", "Which restaurant?"),  # conflicts with 2
        record("Hi<|im_end|>\n\noops<|im_end|>\n<|im_start|>user\nx", "Ok."),  # malformed
        record(long_request(400), "Noted."),
        record(long_request(400, [200]), "Noted."),  # near duplicate of 7
    ]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_end_to_end_is_byte_identical_and_excludes_full_rows(tmp_path):
    first = run_pipeline(tmp_path / "a", fixture_records(), "--max-prompt-tokens", "100")
    second = run_pipeline(tmp_path / "b", fixture_records(), "--max-prompt-tokens", "100")
    for name in P.ARTIFACTS:
        if name in first["artifacts"]:
            assert first["artifacts"][name]["sha256"] == second["artifacts"][name]["sha256"], name
    assert first["stage_counts"]["structural_validation"] == 8
    assert first["stage_counts"]["conflict_removal"] == 6
    assert first["stage_counts"]["exact_dedup"] == 5
    assert first["stage_counts"]["near_dedup"] == 4
    assert first["prompt_limit"]["effective"] == 100 and first["prompt_limit"]["override"] == 100
    assert first["rollout_budget"] == {
        "max_prompt_tokens": 100,
        "max_response_tokens": 32768,
        "minimum_combined_context_tokens": 32868,
        "model_native_context_tokens": 262144,
    }

    out = tmp_path / "a"
    canonical = [json.loads(line) for line in (out / P.ARTIFACTS["canonical"]).read_text().splitlines()]
    train_lines = (out / P.ARTIFACTS["train"]).read_text().splitlines()
    filtered = [json.loads(line) for line in (out / P.ARTIFACTS["length_filtered"]).read_text().splitlines()]
    canonical_lines = {
        row["id"]: line for row, line in zip(canonical, (out / P.ARTIFACTS["canonical"]).read_text().splitlines())
    }
    assert filtered and all(item["prompt_tokens"] > 100 for item in filtered)
    assert {item["id"] for item in filtered}.isdisjoint(json.loads(line)["id"] for line in train_lines)
    assert set(train_lines) | {canonical_lines[item["id"]] for item in filtered} == set(canonical_lines.values())
    assert [row["metadata"]["source_index"] for row in canonical] == sorted(
        row["metadata"]["source_index"] for row in canonical
    )
    assert all(L.check_row(row) == [] for row in canonical)
    assert first["output_validation"][P.ARTIFACTS["train"]]["passed"]
    rejections = [json.loads(line) for line in (out / P.ARTIFACTS["rejections"]).read_text().splitlines()]
    assert [(item["source_index"], item["reason"]) for item in rejections] == [(6, "im_end_without_im_start")]
    assert (out / "looptool_summary.md").read_text().startswith("# LoopTool-23k RL preprocessing summary")


def test_check_row_flags_contract_violations():
    row = parse(record(MULTI_TURN, call("get_weather", city="Rome"))).row
    row["metadata"]["prompt_tokens"] = 10
    broken = json.loads(json.dumps(row))
    broken["messages"][3]["tool_call_id"] = "call_9"
    broken["messages"][-1]["content"] += "<|im_end|>"
    assert {"tool_response_unpaired", "user_framing_token"} <= set(L.check_row(broken))


# ---------------------------------------------------------------------------
# BFCL audit


def test_bfcl_audit_removes_only_high_confidence_overlaps(tmp_path):
    bfcl_function = {"name": "get_weather", "description": "x", "parameters": WEATHER["parameters"]}
    (tmp_path / "BFCL_v4_simple.json").write_text(
        "\n".join(
            json.dumps(item)
            for item in [
                {
                    "id": "simple_0",
                    "question": [[{"role": "user", "content": "What is the weather in Paris?"}]],
                    "function": [bfcl_function],
                },
                {
                    "id": "simple_1",
                    "question": [[{"role": "user", "content": "Book a table"}]],
                    "function": [bfcl_function],
                },
                {
                    "id": "irrelevance_0",
                    "question": [[{"role": "user", "content": "Split this string"}]],
                    "function": [],
                },
                {
                    "id": "multi_0",
                    "question": [[{"role": "user", "content": "Tell a joke"}]],
                    "involved_classes": ["Unmapped"],
                },
            ]
        )
    )
    entries, info = P.load_bfcl(tmp_path)
    assert info["benchmark_prompts"] == 4 and info["categories"] == ["simple"]
    assert [entry["signature"] is None for entry in entries] == [False, False, False, True]
    rows = rows_from(
        record("What is the weather in Paris?", call("get_weather", city="Paris"), tools=(WEATHER,)),
        record("Book a table", "How many?", tools=(WEATHER, SEARCH)),
        record("Unrelated question entirely", "Sure."),
        record("Split this string", "I can't solve this problem.", tools=()),
        record("Tell a joke", "Why did the chicken cross the road?", tools=()),
    )
    kept, records = P.bfcl_audit(
        rows, entries, shingle_size=5, num_perm=128, seed=1, lsh_threshold=0.5, remove_jaccard=0.99, report_jaccard=0.8
    )
    assert [row["metadata"]["source_index"] for row in kept] == [1, 2, 4]
    assert [(item["source_index"], item["disposition"], item["reason"]) for item in records] == [
        (0, "removed", "exact_prompt_and_tool_schema"),
        (1, "audit_only", "exact_prompt_tool_schema_differs"),
        (3, "removed", "exact_prompt_and_tool_schema"),
        (4, "audit_only", "exact_prompt_tool_schema_differs"),
    ]


def test_requested_bfcl_audit_never_silently_skips(tmp_path):
    args = P.parse_args(
        ["--output-dir", str(tmp_path), "--bfcl-data-dir", str(tmp_path / "missing"), "--workers", "1"]
    )
    assert args.bfcl_audit == "on"
    with pytest.raises(SystemExit):
        P.run(
            args,
            records=fixture_records()[:1],
            source_info={
                "dataset": "ds",
                "requested_revision": "rev",
                "source_rows": 1,
                "processed_rows": 1,
                "source_content_sha256": "x",
            },
            tokenizer=FakeTokenizer(),
            tokenizer_info={},
        )


# ---------------------------------------------------------------------------
# Optional real-tokenizer integration (offline; skipped when not cached)


def test_real_qwen_tokenizer_rendering_if_cached():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("jinja2")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            P.TOKENIZER, revision=P.TOKENIZER_REVISION, local_files_only=True
        )
    except Exception:  # noqa: BLE001
        pytest.skip("pinned Qwen tokenizer is not in the local Hugging Face cache")
    row = parse(record(MULTI_TURN, call("get_weather", city="Rome", days=3))).row
    result = P.measure_row(tokenizer, row)
    assert result["error"] is None and result["prompt_tokens"] > result["system_tool_tokens"] > 0
    text = tokenizer.apply_chat_template(
        row["messages"], tools=row["tools"], tokenize=False, add_generation_prompt=True
    )
    assert text.count("<|im_start|>system") == 1 and text.endswith("<|im_start|>assistant\n<think>\n")
    assert POLICY in text and '"name": "get_weather"' in text
    assert text.count("<tool_response>") == 2 and "Rome" in text.split("<|im_start|>user")[-1]


def test_pp_changes_warn_from_raw_markers_onward():
    def stage(name, text_rows, call_rows):
        metas = [
            {
                "conversation_kind": "single_turn",
                "target_kind": "text_no_call",
                "num_user_messages": 1,
                "num_history_tool_calls": 0,
                "num_target_tool_calls": 0,
            }
        ] * text_rows
        metas += [
            {
                "conversation_kind": "multi_turn",
                "target_kind": "single_call",
                "num_user_messages": 2,
                "num_history_tool_calls": 1,
                "num_target_tool_calls": 1,
            }
        ] * call_rows
        return {"stage": name, "rows": len(metas), "distribution": P.distribution(metas)}

    changes, warnings = P.pp_changes(
        [stage("source_raw_markers", 100, 900), stage("structural_validation", 50, 900), stage("near_dedup", 50, 899)]
    )
    assert [change["stage"] for change in changes] == ["structural_validation", "near_dedup"]
    assert any(item.startswith("structural_validation: text_no_call changed -4.737") for item in warnings)
    assert not any(item.startswith("near_dedup") for item in warnings)
