"""Deterministic parsing and normalization for ``zhangkangning/LoopTool-23k``.

Every source row has three strings:

* ``instruction``: a task policy, a ``The current time is ...`` line, and the
  Qwen tool block (``# Tools ... <tools> one JSON object per line </tools> ...``).
* ``input``: either plain user text, or a plain user prefix followed by
  Qwen-serialized ``<|im_start|>ROLE ... <|im_end|>`` turns. Tool observations
  are ``user`` turns made only of ``<tool_response>`` blocks and the final user
  turn is left open (no ``<|im_end|>``).
* ``output``: ``<tool_call>`` blocks, or ordinary assistant text.

The parser here is intentionally strict: anything it does not explicitly
recognize becomes a :class:`Reject` with a stable reason code, never a guess.
Nothing in this module loads a tokenizer, a model, or the network.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from typing import Any

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
FRAMING_TOKENS = (IM_START, IM_END, "<|endoftext|>")
TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<tool_call>", "</tool_call>"
TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE = "<tool_response>", "</tool_response>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
SOURCE_ROLES = ("user", "assistant")

# The Qwen tool boilerplate exactly as it appears in the pinned source. The
# target chat template regenerates it from ``tools=`` (without the trailing
# period), so it is the only instruction text this parser removes.
TOOLS_HEADER = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
)
TOOLS_FOOTERS = tuple(
    "</tools>\n\nFor each function call, return a json object with function name and arguments within "
    '<tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>" + ending
    for ending in (".", "")
)

# Closed, exact table of Python/typing spellings of JSON Schema primitive types.
# The same primitive mapping is used by configs/bfcl_grpo/bfcl_adapter.py for
# BFCL docs. ``repair_schema`` additionally recognizes the exact recursive
# grammar ``List[T]`` / ``Tuple[T, ...]`` and the exact suffix
# ``, optional``; no other type syntax is inferred.
TYPE_ALIASES = {
    "dict": "object",
    "Dict": "object",
    "float": "number",
    "int": "integer",
    "str": "string",
    "bool": "boolean",
    "list": "array",
    "List": "array",
    "tuple": "array",
}
JSON_SCHEMA_TYPES = frozenset(("array", "boolean", "integer", "null", "number", "object", "string"))

# Keywords whose values are (maps of / lists of) subschemas in JSON Schema draft 7.
_SCHEMA_MAP_KEYWORDS = ("properties", "patternProperties", "definitions", "$defs")
_SCHEMA_LIST_KEYWORDS = ("allOf", "anyOf", "oneOf")
_SCHEMA_KEYWORDS = (
    "additionalProperties",
    "additionalItems",
    "contains",
    "not",
    "propertyNames",
    "if",
    "then",
    "else",
)


class Reject(Exception):
    """A mechanical failure that excludes a source row."""

    def __init__(self, stage: str, reason: str, details: str = ""):
        super().__init__(f"{stage}/{reason}: {details}")
        self.stage = stage
        self.reason = reason
        self.details = details[:500]


@dataclass
class Event:
    kind: str  # "repair" | "warning" | "info"
    code: str
    details: str = ""


@dataclass
class Example:
    row: dict[str, Any]
    events: list[Event] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Canonical JSON and text


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_hex(canonical_json(value))


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_text(text: str) -> str:
    """NFC, LF line endings, outer whitespace stripped; internal text untouched."""
    return unicodedata.normalize("NFC", normalize_newlines(text)).strip()


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant {value}")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json(text: str) -> Any:
    """Standard JSON only: no NaN/Infinity and no silently overwritten duplicate keys."""
    return json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)


# ---------------------------------------------------------------------------
# Instruction and tools


def split_instruction(instruction: str) -> tuple[str, list[str]]:
    """Return (task-specific system text, raw tool-definition lines)."""
    text = normalize_newlines(instruction)
    if text.count(TOOLS_HEADER) != 1:
        raise Reject("instruction", "tool_block_header_not_recognized", f"header count {text.count(TOOLS_HEADER)}")
    start = text.index(TOOLS_HEADER)
    footer = next((footer for footer in TOOLS_FOOTERS if text.endswith(footer)), None)
    if footer is None:
        raise Reject("instruction", "tool_block_footer_not_recognized", repr(text[-120:]))
    body = text[start + len(TOOLS_HEADER) : len(text) - len(footer)]
    if body and not body.endswith("\n"):
        raise Reject("instruction", "tool_block_not_line_delimited", repr(body[-80:]))
    lines = body[:-1].split("\n") if body else []
    if any(not line.strip() for line in lines):
        raise Reject("instruction", "tool_block_blank_line")
    return normalize_text(text[:start]), lines


def _walk_subschemas(schema: Any, path: str):
    """Yield (json_pointer, subschema) for every schema position under ``schema``."""
    if not isinstance(schema, dict):
        return
    yield path, schema
    for keyword in _SCHEMA_MAP_KEYWORDS:
        value = schema.get(keyword)
        if isinstance(value, dict):
            for name, sub in value.items():
                yield from _walk_subschemas(sub, f"{path}/{keyword}/{name}")
    for keyword in _SCHEMA_LIST_KEYWORDS:
        value = schema.get(keyword)
        if isinstance(value, list):
            for position, sub in enumerate(value):
                yield from _walk_subschemas(sub, f"{path}/{keyword}/{position}")
    for keyword in _SCHEMA_KEYWORDS:
        yield from _walk_subschemas(schema.get(keyword), f"{path}/{keyword}")
    items = schema.get("items")
    if isinstance(items, list):
        for position, sub in enumerate(items):
            yield from _walk_subschemas(sub, f"{path}/items/{position}")
    else:
        yield from _walk_subschemas(items, f"{path}/items")
    dependencies = schema.get("dependencies")
    if isinstance(dependencies, dict):
        for name, sub in dependencies.items():
            yield from _walk_subschemas(sub, f"{path}/dependencies/{name}")


def _split_type_arguments(text: str) -> list[str] | None:
    parts: list[str] = []
    depth = 0
    start = 0
    for position, character in enumerate(text):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth < 0:
                return None
        elif character == "," and depth == 0:
            parts.append(text[start:position].strip())
            start = position + 1
    if depth:
        return None
    parts.append(text[start:].strip())
    return parts if all(parts) else None


def _type_expression_schema(kind: str) -> tuple[dict[str, Any], bool, str | None] | None:
    """Parse the documented closed type grammar.

    Return ``(schema_fragment, had_optional_suffix, collection_expression)``.
    Optionality is intentionally not encoded in the returned fragment: JSON
    Schema determines it solely from the containing object's ``required`` list.
    """
    optional = kind.endswith(", optional")
    base = kind[: -len(", optional")] if optional else kind
    if base in TYPE_ALIASES:
        return {"type": TYPE_ALIASES[base]}, optional, None
    if base in JSON_SCHEMA_TYPES:
        return {"type": base}, optional, None
    if base.startswith("List[") and base.endswith("]"):
        inner = _type_expression_schema(base[5:-1])
        if inner is None or inner[1]:
            return None
        return {"type": "array", "items": inner[0]}, optional, "list"
    if base.startswith("Tuple[") and base.endswith("]"):
        arguments = _split_type_arguments(base[6:-1])
        if arguments is None:
            return None
        items = [_type_expression_schema(argument) for argument in arguments]
        if any(item is None or item[1] for item in items):
            return None
        fragments = [item[0] for item in items if item is not None]
        return (
            {
                "type": "array",
                "items": fragments,
                "minItems": len(fragments),
                "maxItems": len(fragments),
            },
            optional,
            "tuple",
        )
    return None


def repair_schema(parameters: dict[str, Any], *, type_aliases: bool = True) -> tuple[dict[str, Any], list[Event]]:
    """Apply only the documented mechanical repairs; return (schema, repair events)."""
    schema = json.loads(json.dumps(parameters))
    events: list[Event] = []
    if type_aliases:
        for path, sub in _walk_subschemas(schema, ""):
            kind = sub.get("type")
            if not isinstance(kind, str):
                continue
            parsed = _type_expression_schema(kind)
            if parsed is None:
                continue
            fragment, optional, collection_expression = parsed
            # An existing, incompatible ``items`` schema makes expanding
            # List[T] ambiguous. Leave it untouched so Draft 7 validation
            # rejects the non-standard type instead of guessing.
            if "items" in fragment and "items" in sub and sub["items"] != fragment["items"]:
                continue
            if fragment == {"type": kind}:
                continue
            sub.update(fragment)
            rendered = canonical_json(fragment) if collection_expression else fragment["type"]
            events.append(Event("repair", "schema_type_alias", f"{path or '/'}:{kind}->{rendered}"))
            if optional:
                events.append(Event("repair", "schema_optional_suffix_removed", f"{path or '/'}:{kind}"))
            if collection_expression == "list":
                events.append(Event("repair", "schema_list_type_expanded", f"{path or '/'}:{kind}->{rendered}"))
            elif collection_expression == "tuple":
                events.append(Event("repair", "schema_tuple_type_expanded", f"{path or '/'}:{kind}->{rendered}"))
    object_keywords = {"properties", "patternProperties", "additionalProperties", "required", "$ref"}
    object_keywords |= set(_SCHEMA_LIST_KEYWORDS)
    if schema.get("type") == "object" and not object_keywords & set(schema):
        schema["properties"] = {}
        events.append(Event("repair", "schema_empty_object_properties_added", "/"))
    return schema, events


class SchemaChecker:
    """Draft 7 meta-schema checks, cached by canonical schema text."""

    def __init__(self):
        from jsonschema import Draft7Validator

        self._draft = Draft7Validator
        self._checked: dict[str, str | None] = {}
        self._validators: dict[str, Any] = {}

    def schema_error(self, schema: dict[str, Any]) -> str | None:
        key = canonical_json(schema)
        if key not in self._checked:
            # jsonschema's error order is not stable across processes; report the first error in a sorted order.
            errors = sorted(
                (f"/{'/'.join(map(str, error.absolute_path))}", error.validator, error.message[:200])
                for error in self._draft(self._draft.META_SCHEMA).iter_errors(schema)
            )
            self._checked[key] = f"{errors[0][2]} at {errors[0][0]}" if errors else None
        return self._checked[key]

    def argument_errors(self, schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
        key = canonical_json(schema)
        if key not in self._validators:
            self._validators[key] = self._draft(schema)
        return sorted(
            f"{error.validator}@/{'/'.join(map(str, error.absolute_path))}"
            for error in self._validators[key].iter_errors(arguments)
        )


def _structural_schema_problem(schema: dict[str, Any]) -> str | None:
    """Reject duplicate ``required`` names and required names with no declaration.

    A required name is accepted when ``properties`` declares it, or when the
    object has no ``properties`` map, or when ``patternProperties`` / a schema-valued
    ``additionalProperties`` could supply it.
    """
    for path, sub in _walk_subschemas(schema, ""):
        required, properties = sub.get("required"), sub.get("properties")
        if not isinstance(required, list) or not required:
            continue
        if len(set(required)) != len(required):
            return f"duplicate required names at {path or '/'}"
        if not isinstance(properties, dict) or "patternProperties" in sub:
            continue
        if isinstance(sub.get("additionalProperties"), dict):
            continue
        missing = [name for name in required if name not in properties]
        if missing:
            return f"required {missing} not declared in properties at {path or '/'}"
    return None


def parse_tools(
    lines: list[str], checker: SchemaChecker, *, type_aliases: bool = True
) -> tuple[list[dict], list[Event]]:
    tools: list[dict[str, Any]] = []
    events: list[Event] = []
    seen: dict[str, str] = {}
    for position, line in enumerate(lines):
        try:
            definition = strict_json(line)
        except ValueError as exc:
            raise Reject("tools", "tool_definition_invalid_json", f"tool {position}: {exc}") from None
        if not isinstance(definition, dict):
            raise Reject("tools", "tool_definition_not_object", f"tool {position}")
        name = definition.get("name")
        if not isinstance(name, str) or not name.strip():
            raise Reject("tools", "tool_name_missing_or_empty", f"tool {position}")
        signature = canonical_json(definition)
        if name in seen:
            if seen[name] != signature:
                raise Reject("tools", "conflicting_duplicate_tool_name", name)
            events.append(
                Event("repair", "identical_duplicate_tool_definition_removed", f"{name}:position={position}")
            )
            continue
        seen[name] = signature
        if "description" in definition and not isinstance(definition["description"], str):
            raise Reject("tools", "tool_description_not_string", name)
        if "description" not in definition:
            events.append(Event("warning", "tool_description_missing", name))
        if "parameters" not in definition:
            raise Reject("schema", "tool_parameters_missing", name)
        parameters = definition["parameters"]
        if not isinstance(parameters, dict):
            raise Reject("schema", "tool_parameters_not_object", name)
        parameters, repairs = repair_schema(parameters, type_aliases=type_aliases)
        events.extend(Event(item.kind, item.code, f"{name}{item.details}") for item in repairs)
        error = checker.schema_error(parameters)
        if error is not None:
            raise Reject("schema", "tool_schema_invalid_draft7", f"{name}: {error}")
        if parameters.get("type") != "object":
            raise Reject("schema", "tool_schema_root_not_object", f"{name}: type={parameters.get('type')!r}")
        problem = _structural_schema_problem(parameters)
        if problem is not None:
            raise Reject("schema", "tool_schema_required_properties_mismatch", f"{name}: {problem}")
        if "required" in definition:
            events.append(Event("warning", "tool_level_required_outside_parameters", name))
        function = {key: value for key, value in definition.items() if key != "parameters"}
        function["parameters"] = parameters
        tools.append({"type": "function", "function": function})
    return tools, events


# ---------------------------------------------------------------------------
# Dialogue state machine


def split_role_blocks(text: str) -> list[tuple[str, str]]:
    """Split Qwen-style serialization into (role, raw content) blocks.

    A plain prefix before the first ``<|im_start|>`` is a user message. Text with
    no control tokens is one user message. The final block may be left open.
    """
    if IM_START not in text and IM_END not in text:
        return [("user", text)]
    blocks: list[tuple[str, str]] = []
    position = 0
    role: str | None = None
    if not text.startswith(IM_START):
        role, content_start = "user", 0
    while True:
        if role is not None:
            next_start = text.find(IM_START, content_start)
            next_end = text.find(IM_END, content_start)
            if next_end == -1:
                if next_start != -1:
                    raise Reject("dialogue", "im_start_inside_open_message", f"block {len(blocks)}")
                blocks.append((role, text[content_start:]))
                return blocks
            if next_start != -1 and next_start < next_end:
                raise Reject("dialogue", "im_start_inside_open_message", f"block {len(blocks)}")
            blocks.append((role, text[content_start:next_end]))
            position, role = next_end + len(IM_END), None
            continue
        next_start = text.find(IM_START, position)
        next_end = text.find(IM_END, position)
        if next_end != -1 and (next_start == -1 or next_end < next_start):
            raise Reject("dialogue", "im_end_without_im_start", f"after block {len(blocks)}")
        gap = text[position:] if next_start == -1 else text[position:next_start]
        if gap.strip():
            raise Reject("dialogue", "text_outside_role_block", repr(gap.strip()[:80]))
        if next_start == -1:
            return blocks
        header_end = text.find("\n", next_start + len(IM_START))
        if header_end == -1:
            raise Reject("dialogue", "role_header_unterminated")
        role = text[next_start + len(IM_START) : header_end]
        if role not in SOURCE_ROLES:
            raise Reject("dialogue", "unsupported_role", repr(role[:40]))
        content_start = header_end + 1


def _tagged_segments(text: str, open_tag: str, close_tag: str, stage: str, label: str) -> tuple[list[str], list[str]]:
    """Return (outside segments, inside segments); tags must strictly alternate."""
    outside, inside = [], []
    position = 0
    while True:
        start = text.find(open_tag, position)
        stray_close = text.find(close_tag, position)
        if start == -1:
            if stray_close != -1:
                raise Reject(stage, f"unbalanced_{label}_tags", "close without open")
            outside.append(text[position:])
            return outside, inside
        if stray_close != -1 and stray_close < start:
            raise Reject(stage, f"unbalanced_{label}_tags", "close before open")
        end = text.find(close_tag, start + len(open_tag))
        if end == -1:
            raise Reject(stage, f"unbalanced_{label}_tags", "open without close")
        body = text[start + len(open_tag) : end]
        if open_tag in body:
            raise Reject(stage, f"unbalanced_{label}_tags", "nested open")
        outside.append(text[position:start])
        inside.append(body)
        position = end + len(close_tag)


def strip_think(text: str, stage: str, events: list[Event]) -> str:
    """Remove only explicitly delimited ``<think>...</think>`` blocks."""
    if THINK_OPEN not in text and THINK_CLOSE not in text:
        return text
    outside, inside = _tagged_segments(text, THINK_OPEN, THINK_CLOSE, stage, "think")
    events.append(Event("info", f"{stage}_think_removed", f"{len(inside)} block(s)"))
    return "".join(outside)


def parse_call(body: str, stage: str) -> dict[str, Any]:
    try:
        call = strict_json(body.strip())
    except ValueError as exc:
        raise Reject(stage, "tool_call_invalid_json", str(exc)) from None
    if not isinstance(call, dict):
        raise Reject(stage, "tool_call_not_object")
    if set(call) - {"name", "arguments"}:
        raise Reject(stage, "tool_call_unexpected_keys", str(sorted(set(call) - {"name", "arguments"})))
    name = call.get("name")
    if not isinstance(name, str) or not name.strip():
        raise Reject(stage, "tool_call_name_missing_or_empty")
    if "arguments" not in call:
        raise Reject(stage, "tool_call_arguments_missing", name)
    if not isinstance(call["arguments"], dict):
        raise Reject(stage, "tool_call_arguments_not_object", f"{name}: {type(call['arguments']).__name__}")
    return {"name": name, "arguments": call["arguments"]}


def parse_assistant(text: str, stage: str, events: list[Event]) -> tuple[str, list[dict[str, Any]]]:
    """Return (normalized text, calls). Text is allowed only before the first call."""
    text = strip_think(normalize_newlines(text), stage, events)
    outside, inside = _tagged_segments(text, TOOL_CALL_OPEN, TOOL_CALL_CLOSE, stage, "tool_call")
    if any(segment.strip() for segment in outside[1:]):
        raise Reject(stage, "assistant_text_after_tool_call", repr("".join(outside[1:]).strip()[:80]))
    return normalize_text(outside[0]), [parse_call(body, stage) for body in inside]


def _check_plain_text(text: str, stage: str, label: str) -> None:
    for token in FRAMING_TOKENS:
        if token in text:
            raise Reject(stage, "residual_framing_token", f"{label}: {token}")
    for tag in (TOOL_CALL_OPEN, TOOL_CALL_CLOSE, TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE, THINK_OPEN, THINK_CLOSE):
        if tag in text:
            raise Reject(stage, "residual_control_tag", f"{label}: {tag}")


def parse_dialogue(text: str, events: list[Event]) -> list[dict[str, Any]]:
    """Convert ``input`` into structured user/assistant/tool messages."""
    messages: list[dict[str, Any]] = []
    pending: list[str] | None = None
    call_count = 0
    for index, (role, raw) in enumerate(split_role_blocks(normalize_newlines(text))):
        is_observation = role == "user" and (TOOL_RESPONSE_OPEN in raw or TOOL_RESPONSE_CLOSE in raw)
        if is_observation:
            outside, inside = _tagged_segments(
                raw, TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE, "dialogue", "tool_response"
            )
            if any(segment.strip() for segment in outside):
                raise Reject("dialogue", "tool_response_mixed_with_user_text", f"block {index}")
            if pending is None:
                raise Reject("dialogue", "tool_response_without_tool_call", f"block {index}")
            if len(inside) != len(pending):
                raise Reject(
                    "dialogue", "tool_response_count_mismatch", f"{len(pending)} calls, {len(inside)} responses"
                )
            for call_id, body in zip(pending, inside):
                content = normalize_text(body)
                _check_plain_text(content, "dialogue", f"tool response {call_id}")
                messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
            pending = None
            continue
        if pending is not None:
            raise Reject("dialogue", "tool_call_without_tool_response", f"block {index}")
        if role == "user":
            content = normalize_text(raw)
            if not content:
                raise Reject("dialogue", "empty_user_message", f"block {index}")
            _check_plain_text(content, "dialogue", f"user block {index}")
            messages.append({"role": "user", "content": content})
            continue
        content, calls = parse_assistant(raw, "dialogue", events)
        _check_plain_text(content, "dialogue", f"assistant block {index}")
        if not content and not calls:
            raise Reject("dialogue", "empty_assistant_message", f"block {index}")
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            ids = [f"call_{call_count + offset}" for offset in range(len(calls))]
            call_count += len(calls)
            message["tool_calls"] = [
                {"id": call_id, "type": "function", "function": call} for call_id, call in zip(ids, calls)
            ]
            if content:
                events.append(Event("info", "history_assistant_text_with_tool_calls", f"block {index}"))
            pending = ids
        messages.append(message)
    if pending is not None:
        raise Reject("dialogue", "prompt_ends_with_unanswered_tool_calls")
    if not messages or messages[0]["role"] != "user":
        raise Reject("dialogue", "dialogue_does_not_start_with_user")
    if messages[-1]["role"] == "assistant":
        raise Reject("dialogue", "prompt_ends_with_assistant")
    for previous, current in zip(messages, messages[1:]):
        if previous["role"] == current["role"] and current["role"] != "tool":
            events.append(Event("warning", f"consecutive_{current['role']}_messages"))
    return messages


def parse_target(output: str, events: list[Event]) -> dict[str, Any]:
    content, calls = parse_assistant(output, "target", events)
    _check_plain_text(content, "target", "target text")
    if not content and not calls:
        raise Reject("target", "empty_target")
    if content and calls:
        events.append(Event("info", "target_text_with_tool_calls"))
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"type": "function", "function": call} for call in calls],
    }


# ---------------------------------------------------------------------------
# Whole-row parsing and mechanical classification


def target_kind(target: dict[str, Any]) -> str:
    count = len(target["tool_calls"])
    return "parallel_call" if count >= 2 else "single_call" if count == 1 else "text_no_call"


def target_key(target: dict[str, Any]) -> str:
    """Conflict key: calls of one assistant turn compared as an unordered multiset."""
    calls = sorted(canonical_json(call["function"]) for call in target["tool_calls"])
    return canonical_hash({"content": target["content"], "calls": calls})


def prompt_hash(row: dict[str, Any]) -> str:
    return canonical_hash({"tools": row["tools"], "messages": row["messages"]})


def example_hash(row: dict[str, Any]) -> str:
    return canonical_hash({"tools": row["tools"], "messages": row["messages"], "target": row["target"]})


def source_row_id(record: dict[str, Any], *, dataset: str, revision: str, index: int) -> str:
    payload = {
        "dataset": dataset,
        "revision": revision,
        "index": index,
        "instruction": record.get("instruction"),
        "input": record.get("input"),
        "output": record.get("output"),
    }
    return "sha256:" + canonical_hash(payload)


def _check_arguments(
    call: dict[str, Any],
    tools: dict[str, dict],
    checker: SchemaChecker,
    prefix: str,
    events: list[Event],
    *,
    reject_schema_errors: bool = False,
):
    parameters = tools[call["name"]]["parameters"]
    errors = checker.argument_errors(parameters, call["arguments"])
    if errors and reject_schema_errors:
        raise Reject("target", "target_arguments_schema_mismatch", f"{call['name']}: {errors[0]}")
    for error in errors:
        validator = error.split("@", 1)[0]
        events.append(Event("warning", f"{prefix}_arguments_schema_{validator}", f"{call['name']}: {error}"))
    properties = parameters.get("properties")
    if isinstance(properties, dict) and "additionalProperties" not in parameters:
        unknown = sorted(set(call["arguments"]) - set(properties))
        if unknown:
            events.append(Event("warning", f"{prefix}_arguments_undeclared_parameter", f"{call['name']}: {unknown}"))


def parse_example(
    record: dict[str, Any],
    *,
    index: int,
    dataset: str,
    revision: str,
    checker: SchemaChecker,
    type_aliases: bool = True,
) -> Example:
    for key in ("instruction", "input", "output"):
        if not isinstance(record.get(key), str):
            raise Reject("source", "source_field_missing_or_not_string", key)
    events: list[Event] = []
    system, tool_lines = split_instruction(record["instruction"])
    _check_plain_text(system, "instruction", "system text")
    tools, tool_events = parse_tools(tool_lines, checker, type_aliases=type_aliases)
    events.extend(tool_events)
    dialogue = parse_dialogue(record["input"], events)
    target = parse_target(record["output"], events)

    inventory = {tool["function"]["name"]: tool["function"] for tool in tools}
    history_calls = [call["function"] for message in dialogue for call in message.get("tool_calls", [])]
    for call in history_calls:
        if call["name"] not in inventory:
            raise Reject("references", "history_call_unknown_tool", call["name"])
        _check_arguments(call, inventory, checker, "history", events)
    for call in target["tool_calls"]:
        if call["function"]["name"] not in inventory:
            raise Reject("references", "target_call_unknown_tool", call["function"]["name"])
        _check_arguments(call["function"], inventory, checker, "target", events, reject_schema_errors=True)

    messages = ([{"role": "system", "content": system}] if system else []) + dialogue
    roles = [message["role"] for message in messages]
    num_users = roles.count("user")
    kind = target_kind(target)
    metadata = {
        "source_dataset": dataset,
        "source_revision": revision,
        "source_index": index,
        "conversation_kind": "single_turn" if num_users == 1 and not history_calls else "multi_turn",
        "target_kind": kind,
        "num_user_messages": num_users,
        "num_assistant_messages": roles.count("assistant"),
        "num_history_tool_calls": len(history_calls),
        "num_tool_responses": roles.count("tool"),
        "num_target_tool_calls": len(target["tool_calls"]),
        "num_tools": len(tools),
        "target_has_text_and_calls": bool(target["content"] and target["tool_calls"]),
        "repair_codes": sorted({event.code for event in events if event.kind == "repair"}),
        "warning_codes": sorted({event.code for event in events if event.kind == "warning"}),
        "prompt_tokens": None,
    }
    row = {
        "id": source_row_id(record, dataset=dataset, revision=revision, index=index),
        "messages": messages,
        "tools": tools,
        "target": target,
        "metadata": metadata,
    }
    # Round-trip so in-memory key order is exactly what a JSONL reader will see.
    return Example(row=json.loads(canonical_json(row)), events=events)


# ---------------------------------------------------------------------------
# Output contract


def _text_problems(text: Any, label: str) -> list[str]:
    if not isinstance(text, str):
        return [f"{label}_content_not_string"]
    problems = [f"{label}_framing_token" for token in FRAMING_TOKENS if token in text]
    if THINK_OPEN in text or THINK_CLOSE in text:
        problems.append(f"{label}_think_tag")
    return problems


def _call_problems(call: Any, inventory: set[str], label: str, *, needs_id: bool) -> list[str]:
    if not isinstance(call, dict) or call.get("type") != "function" or not isinstance(call.get("function"), dict):
        return [f"{label}_call_malformed"]
    problems = []
    if needs_id and not isinstance(call.get("id"), str):
        problems.append(f"{label}_call_missing_id")
    function = call["function"]
    if function.get("name") not in inventory:
        problems.append(f"{label}_call_unknown_tool")
    if not isinstance(function.get("arguments"), dict):
        problems.append(f"{label}_call_arguments_not_object")
    return problems


def check_row(row: Any) -> list[str]:
    """Return contract violations for one canonical output row (empty when valid)."""
    if not isinstance(row, dict) or set(row) != {"id", "messages", "tools", "target", "metadata"}:
        return ["row_keys"]
    problems = []
    if not isinstance(row["id"], str) or not row["id"].startswith("sha256:") or len(row["id"]) != 71:
        problems.append("id_format")
    tools = row["tools"]
    if not isinstance(tools, list) or not all(
        isinstance(tool, dict)
        and tool.get("type") == "function"
        and isinstance(tool.get("function"), dict)
        and isinstance(tool["function"].get("name"), str)
        and isinstance(tool["function"].get("parameters"), dict)
        for tool in tools
    ):
        return problems + ["tools_malformed"]
    tool_names = [tool["function"]["name"] for tool in tools]
    if len(tool_names) != len(set(tool_names)):
        problems.append("duplicate_tool_name")
    inventory = set(tool_names)
    messages = row["messages"]
    if not isinstance(messages, list) or not messages:
        return problems + ["messages_malformed"]
    pending: list[str] = []
    for position, message in enumerate(messages):
        role = message.get("role") if isinstance(message, dict) else None
        if role not in ("system", "user", "assistant", "tool"):
            problems.append("message_role")
            continue
        problems += _text_problems(message.get("content"), role)
        if role == "system" and position != 0:
            problems.append("system_not_first")
        if role == "tool":
            if not pending or message.get("tool_call_id") != pending[0]:
                problems.append("tool_response_unpaired")
            else:
                pending.pop(0)
            continue
        if pending:
            problems.append("tool_call_unanswered")
            pending = []
        if role == "assistant":
            for call in message.get("tool_calls", []):
                problems += _call_problems(call, inventory, "history", needs_id=True)
                if isinstance(call, dict) and isinstance(call.get("id"), str):
                    pending.append(call["id"])
    if pending:
        problems.append("tool_call_unanswered")
    if messages[-1].get("role") not in ("user", "tool"):
        problems.append("prompt_last_role")
    target = row["target"]
    if (
        not isinstance(target, dict)
        or target.get("role") != "assistant"
        or not isinstance(target.get("tool_calls"), list)
    ):
        return problems + ["target_malformed"]
    problems += _text_problems(target.get("content"), "target")
    for call in target["tool_calls"]:
        problems += _call_problems(call, inventory, "target", needs_id=False)
    if not target["tool_calls"] and not target.get("content"):
        problems.append("target_empty")
    metadata = row["metadata"]
    if not isinstance(metadata, dict) or not isinstance(metadata.get("prompt_tokens"), int):
        return problems + ["metadata_prompt_tokens"]
    if metadata.get("target_kind") != target_kind(target):
        problems.append("metadata_target_kind")
    users = sum(message.get("role") == "user" for message in messages)
    history = sum(len(message.get("tool_calls", [])) for message in messages)
    expected = "single_turn" if users == 1 and not history else "multi_turn"
    if metadata.get("conversation_kind") != expected:
        problems.append("metadata_conversation_kind")
    return sorted(set(problems))
