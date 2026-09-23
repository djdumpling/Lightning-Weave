"""Small BFCL-v4 executable-environment adapter for Slime rollouts.

Derived from Training Gym's ``tutorials/cross_tok_distill`` BFCL example at
commit 8899342. Episodes cover one BFCL user turn so completed-turn thinking is
not replayed, while tool calls and observations inside the active turn remain
autoregressive.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

BFCL_PACKAGE_VERSION = "2026.3.23"
BFCL_CATEGORY = "multi_turn_base"
MAX_ASSISTANT_STEPS = 20
MAX_PROMPT_TOKENS = 8192
MAX_STEP_RESPONSE_TOKENS = 4096
MAX_MODEL_TOKENS = 32768
OBSERVATION_CHARS = 2000

SYSTEM_PROMPT = """You are a tool-using agent. Complete the current user request with the tools provided.
Call exactly one tool per assistant step. Emit the native tool call with no extra prose. Check each tool result, continue until the request is complete, then stop calling tools. Never invent tools or arguments."""

_TYPE_MAP = {
    "dict": "object",
    "list": "array",
    "tuple": "array",
    "float": "number",
    "integer": "integer",
    "string": "string",
    "boolean": "boolean",
}
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
_TMP_RE = re.compile(r"/tmp/[^\s/]+")


@dataclass
class Observation:
    text: str
    is_error: bool = False


def _data_dir() -> str:
    import bfcl_eval

    return os.path.join(os.path.dirname(bfcl_eval.__file__), "data")


def _jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_call(call: str) -> dict[str, Any]:
    node = ast.parse(call.strip(), mode="eval").body
    if not isinstance(node, ast.Call):
        raise TypeError(f"Not a call expression: {call!r}")
    name = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
    args = {f"_pos{i}": ast.literal_eval(arg) for i, arg in enumerate(node.args)}
    args.update({kw.arg: ast.literal_eval(kw.value) for kw in node.keywords})
    return {"name": name, "arguments": args}


def _owner(instances: dict[str, Any], name: str) -> Any | None:
    if name.startswith("_"):
        return None
    return next(
        (instance for instance in instances.values() if hasattr(type(instance), name)),
        None,
    )


def _normalize_args(owner: Any, name: str, args: dict[str, Any]) -> dict[str, Any]:
    positional = sorted(
        ((key, value) for key, value in args.items() if key.startswith("_pos")),
        key=lambda item: int(item[0][4:]),
    )
    if not positional:
        return args
    try:
        names = [parameter for parameter in inspect.signature(getattr(owner, name)).parameters if parameter != "self"]
    except (TypeError, ValueError):
        names = []
    normalized = dict(zip(names, (value for _, value in positional)))
    normalized.update({key: value for key, value in args.items() if not key.startswith("_pos")})
    return normalized


def execute(instances: dict[str, Any], call: dict[str, Any]) -> Observation:
    owner = _owner(instances, call["name"])
    if owner is None:
        return Observation(f"Error during execution: unknown function {call['name']!r}", True)
    args = _normalize_args(owner, call["name"], call.get("arguments") or {})
    try:
        value = getattr(owner, call["name"])(**deepcopy(args))
    except Exception as exc:  # noqa: BLE001 - BFCL backends expose heterogeneous errors.
        return Observation(f"Error during execution: {exc}", True)
    if isinstance(value, dict):
        try:
            value = json.dumps(value)
        except TypeError:
            value = str(value)
    return Observation(str(value))


def replay(label: dict[str, Any], calls: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    import importlib

    from bfcl_eval.constants.executable_backend_config import (
        CLASS_FILE_PATH_MAPPING,
        STATELESS_CLASSES,
    )

    instances: dict[str, Any] = {}
    for class_name in label["involved_classes"]:
        module = importlib.import_module(CLASS_FILE_PATH_MAPPING[class_name])
        instance = getattr(module, class_name)()
        if class_name not in STATELESS_CLASSES:
            instance._load_scenario(
                deepcopy(label["initial_config"].get(class_name, {})),
                long_context=False,
            )
        instances[class_name] = instance
    observations = [execute(instances, deepcopy(call)).text for call in calls]
    return instances, observations


def _json_schema(value: Any) -> Any:
    if isinstance(value, dict):
        result = {key: _json_schema(item) for key, item in value.items() if key != "default"}
        if result.get("type") in _TYPE_MAP:
            result["type"] = _TYPE_MAP[result["type"]]
        return result
    if isinstance(value, list):
        return [_json_schema(item) for item in value]
    return value


def load_tool_schemas(classes: list[str], excluded: list[str]) -> dict[str, Any]:
    from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING

    schemas: dict[str, Any] = {}
    for class_name in classes:
        filename = MULTI_TURN_FUNC_DOC_FILE_MAPPING.get(class_name)
        if not filename:
            continue
        path = os.path.join(_data_dir(), "multi_turn_func_doc", filename)
        for doc in _jsonl(path):
            if doc["name"] not in excluded:
                schemas[doc["name"]] = {
                    "description": doc.get("description", ""),
                    "parameters": _json_schema(doc.get("parameters", {})),
                }
    return schemas


def openai_tools(schemas: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec.get("description", ""),
                "parameters": spec.get("parameters") or {"type": "object", "properties": {}},
            },
        }
        for name, spec in sorted(schemas.items())
    ]


def prefix_messages(label: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    shown = 0
    stop = label["start_step"]
    for turn in label["turns"]:
        messages.append({"role": "user", "content": turn["user"]})
        for call in turn["calls"]:
            if shown >= stop:
                return messages
            call_id = f"history_{shown}"
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": call,
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": label["observations"][shown],
                    },
                ]
            )
            shown += 1
    return messages


def make_dataset_class(dataset_base: type) -> type:
    class BfclTurnDataset(dataset_base):
        """BFCL v4 multi-turn-base, expanded to one executable user turn per row."""

        def __init__(self, *, split: str = "train", eval_tasks: int = 30, task_limit: int | None = None):
            self.split = split
            self.eval_tasks = eval_tasks
            self.task_limit = task_limit

        def cache_key(self) -> str:
            payload = (
                f"{BFCL_PACKAGE_VERSION}:{BFCL_CATEGORY}:{self.split}:{self.eval_tasks}:{self.task_limit}:turn-v1"
            )
            return "bfcl-" + hashlib.sha256(payload.encode()).hexdigest()[:20]

        def input_key(self) -> str:
            return "prompt"

        def label_key(self) -> str:
            return "label"

        def apply_chat_template(self) -> bool:
            return False

        def rows(self) -> list[dict[str, str]]:
            from bfcl_eval.constants.category_mapping import VERSION_PREFIX

            filename = f"{VERSION_PREFIX}_{BFCL_CATEGORY}.json"
            entries = {row["id"]: row for row in _jsonl(os.path.join(_data_dir(), filename))}
            answers = {
                row["id"]: row["ground_truth"]
                for row in _jsonl(os.path.join(_data_dir(), "possible_answer", filename))
            }
            task_ids = list(entries)
            if self.eval_tasks:
                task_ids = task_ids[-self.eval_tasks :] if self.split == "eval" else task_ids[: -self.eval_tasks]
            elif self.split == "eval":
                task_ids = []
            if self.task_limit is not None:
                task_ids = task_ids[: self.task_limit]

            rows: list[dict[str, str]] = []
            for task_id in task_ids:
                entry, ground_truth = entries[task_id], answers.get(task_id, [])
                if not ground_truth or not any(ground_truth):
                    continue
                turns = []
                for messages, calls in zip(entry["question"], ground_truth):
                    user = next(
                        (
                            str(message["content"])
                            for message in messages
                            if message.get("role") == "user" and str(message.get("content", "")).strip()
                        ),
                        "",
                    )
                    turns.append({"user": user, "calls": [parse_call(call) for call in calls]})
                all_calls = [call for turn in turns for call in turn["calls"]]
                base_label = {
                    "task_id": task_id,
                    "initial_config": entry["initial_config"],
                    "involved_classes": entry["involved_classes"],
                    "turns": turns,
                    "all_calls": all_calls,
                    "tool_schemas": load_tool_schemas(entry["involved_classes"], entry.get("excluded_function", [])),
                }
                _, observations = replay(base_label, all_calls)
                base_label["observations"] = [text[:OBSERVATION_CHARS] for text in observations]

                start = 0
                for turn_index, turn in enumerate(turns):
                    if not turn["calls"]:
                        continue
                    label = {
                        **base_label,
                        "turn_index": turn_index,
                        "start_step": start,
                        "target_steps": len(turn["calls"]),
                    }
                    rows.append({"prompt": f"{task_id}:turn-{turn_index}", "label": json.dumps(label)})
                    start += len(turn["calls"])
            return rows

    return BfclTurnDataset


@dataclass
class BfclEnvironment:
    label: dict[str, Any]
    instances: dict[str, Any]
    observations: list[str] = field(default_factory=list)

    def step(self, action: Any) -> Observation:
        observation = execute(
            self.instances,
            {"name": action.name, "arguments": action.arguments or {}},
        )
        self.observations.append(observation.text)
        return observation

    def passed(self) -> bool:
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
            response_checker,
            state_checker,
        )

        start = self.label["start_step"]
        end = start + self.label["target_steps"]
        expected_instances, expected = replay(self.label, self.label["all_calls"][:end])
        state = state_checker(self.instances, expected_instances)
        if not state["valid"]:
            return False
        return bool(response_checker(self.observations, expected[start:end], 0)["valid"])


def _coerce_args(call: dict[str, Any] | None) -> dict[str, Any]:
    if not call:
        return {}
    args = call.get("arguments", {})
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _normalized(value: Any) -> str:
    text = str(value).strip().lower()
    text = _UUID_RE.sub("<uuid>", text)
    text = _TIMESTAMP_RE.sub("<timestamp>", text)
    return _TMP_RE.sub("/tmp/<tmp>", text)


def _structural_match(student: dict[str, Any] | None, expert: dict[str, Any]) -> float:
    if not student or student.get("name") != expert.get("name"):
        return 0.0
    score = 0.4
    student_args, expert_args = _coerce_args(student), _coerce_args(expert)
    student_keys, expert_keys = set(student_args), set(expert_args)
    if student_keys == expert_keys:
        score += 0.3
    elif student_keys & expert_keys:
        score += 0.15
    else:
        return score
    shared = student_keys & expert_keys
    if not shared:
        return score + 0.3
    matches = sum(_normalized(student_args[key]) == _normalized(expert_args[key]) for key in shared)
    return score + 0.3 * matches / len(shared)


def _partial(student: dict[str, Any] | None, expert: dict[str, Any], schemas: dict[str, Any]) -> float:
    if not student or "name" not in student:
        return 0.0
    score = 0.2
    spec = schemas.get(student.get("name"))
    if spec is not None:
        score += 0.15
        schema = spec.get("parameters", spec) if isinstance(spec, dict) else spec
        try:
            import jsonschema

            jsonschema.validate(_coerce_args(student), schema)
        except (jsonschema.ValidationError, jsonschema.SchemaError):
            valid = False
        else:
            valid = True
        if valid:
            score += 0.15
    score += 0.5 * _structural_match(student, expert)
    return min(score, 1.0)


def trajectory_reward(
    calls: list[dict[str, Any]],
    successes: list[bool],
    experts: list[dict[str, Any]],
    schemas: dict[str, Any],
    passed: bool,
) -> float:
    steps = max(len(experts), 1)
    partial = sum(_partial(call, expert, schemas) for call, expert in zip(calls, experts)) / steps
    first = _partial(calls[0], experts[0], schemas) if calls and experts else 0.0
    execution = min(sum(successes), steps) / steps
    return 0.25 * partial + 0.20 * first + 0.10 * execution + 0.45 * float(passed)


def _sampling_request(sampling_params: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    """Preserve Slime's replay metadata request while applying the per-step cap."""

    request = deepcopy(sampling_params)
    request["max_new_tokens"] = max_new_tokens
    if float(request.get("top_p", 1.0)) != 1.0:
        custom_params = dict(request.get("custom_params") or {})
        custom_params["return_top_p_token_ids"] = True
        request["custom_params"] = custom_params
    return request


def _generated_tokens(output: dict[str, Any]) -> tuple[list[int], list[float]]:
    """Read exact sampled tokens/log-probs instead of retokenizing response text."""

    token_logprobs = output.get("meta_info", {}).get("output_token_logprobs")
    if not token_logprobs:
        raise RuntimeError("SGLang did not return output_token_logprobs for a BFCL rollout.")
    return (
        [int(item[1]) for item in token_logprobs],
        [float(item[0]) for item in token_logprobs],
    )


async def bfcl_turn_rollout(args: Any, sample: Any, sampling_params: dict[str, Any]) -> Any:
    """Run one BFCL user turn; mask capped trajectories without changing reward."""

    from modal_training_gym.common.models.base import parse_qwen3_response

    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    state = GenerateState(args)
    tokenizer = state.tokenizer
    label = json.loads(sample.label)
    messages = prefix_messages(label)
    tools = openai_tools(label["tool_schemas"])
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    def abort() -> Any:
        token = tokenizer.pad_token_id or tokenizer.eos_token_id or prompt_ids[-1]
        sample.tokens = prompt_ids + [token]
        sample.response = ""
        sample.response_length = 1
        sample.loss_mask = [0]
        sample.rollout_log_probs = None
        sample.rollout_top_p_token_ids = None
        sample.rollout_top_p_token_offsets = None
        sample.status = Sample.Status.ABORTED
        return sample

    if not prompt_ids or len(prompt_ids) > int(getattr(args, "bfcl_max_prompt_tokens", MAX_PROMPT_TOKENS)):
        return abort()

    # Infer the exact Qwen separators instead of hard-coding special tokens.
    base = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
    generation_suffix = prompt[len(base) :]
    probe = tokenizer.apply_chat_template(
        messages
        + [
            {"role": "assistant", "content": "\x01A\x01"},
            {"role": "tool", "content": "\x00OBS\x00"},
        ],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    after_assistant = probe.split("\x01A\x01", 1)[1]
    observation_open, rest = after_assistant.split("\x00OBS\x00", 1)
    observation_close = rest[: len(rest) - len(generation_suffix)]
    stop_token = observation_open.split("\n", 1)[0]

    try:
        env = BfclEnvironment(
            label=label,
            instances=replay(label, label["all_calls"][: label["start_step"]])[0],
        )
    except Exception:  # noqa: BLE001 - an invalid BFCL sandbox must abort cleanly.
        return abort()

    sample.prompt = prompt
    sample.tokens = list(prompt_ids)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    trajectory = ""
    calls: list[dict[str, Any]] = []
    successes: list[bool] = []
    consecutive_errors = 0
    capped = False
    max_context = int(getattr(args, "bfcl_max_model_tokens", MAX_MODEL_TOKENS))
    step_cap = int(getattr(args, "bfcl_step_response_tokens", MAX_STEP_RESPONSE_TOKENS))
    max_steps = int(getattr(args, "bfcl_max_assistant_steps", MAX_ASSISTANT_STEPS))
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    for turn in range(max_steps):
        remaining = max_context - len(prompt_ids) - sample.response_length
        if remaining <= 0:
            capped = True
            break
        request = _sampling_request(sampling_params, min(step_cap, remaining))
        output = await post(
            url,
            {
                "input_ids": sample.tokens,
                "sampling_params": request,
                "return_logprob": True,
            },
        )
        finish = output["meta_info"]["finish_reason"]["type"]
        if finish == "abort":
            return abort()

        model_text = output["text"]
        model_ids, model_log_probs = _generated_tokens(output)
        if len(model_ids) > remaining:
            raise RuntimeError(f"SGLang returned {len(model_ids)} BFCL tokens with only {remaining} tokens remaining.")
        sample.append_response_tokens(
            args,
            tokens=model_ids,
            log_probs=model_log_probs,
            trainable=True,
            meta_info=output["meta_info"],
            text=model_text,
            update_terminal_info=False,
        )
        if float(request.get("top_p", 1.0)) != 1.0 and sample.rollout_top_p_token_offsets is None:
            raise RuntimeError("SGLang omitted top-p replay metadata for a BFCL rollout.")
        trajectory += model_text

        parsed = parse_qwen3_response(model_text)
        action = parsed.tool_calls[0] if parsed.tool_calls else None
        if action is None or finish == "length":
            capped = finish == "length"
            break

        calls.append({"name": action.name, "arguments": action.arguments})
        observation = env.step(action)
        successes.append(not observation.is_error)
        consecutive_errors = consecutive_errors + 1 if observation.is_error else 0

        opening = observation_open[len(stop_token) :] if model_text.endswith(stop_token) else observation_open
        observation_text = observation.text[:OBSERVATION_CHARS]
        segment = opening + observation_text + observation_close + generation_suffix
        segment_ids = tokenizer(segment, add_special_tokens=False)["input_ids"]
        remaining = max_context - len(prompt_ids) - sample.response_length
        if len(segment_ids) > remaining:
            segment_ids = segment_ids[: max(remaining, 0)]
            segment = tokenizer.decode(segment_ids, skip_special_tokens=False)
            capped = True
        sample.append_response_tokens(
            args,
            tokens=segment_ids,
            trainable=False,
            text=segment,
            update_terminal_info=False,
        )
        trajectory += segment
        if capped or consecutive_errors >= 3:
            break

    if not sample.response_length or not any(sample.loss_mask):
        return abort()

    try:
        passed = env.passed()
    except Exception:  # noqa: BLE001 - checker failures are failed episodes, not crashed batches.
        passed = False
    start = label["start_step"]
    experts = label["all_calls"][start : start + label["target_steps"]]
    reward = trajectory_reward(calls, successes, experts, label["tool_schemas"], passed)

    if capped:
        sample.loss_mask = [0] * sample.response_length
    sample.remove_sample = capped
    sample.status = Sample.Status.TRUNCATED if capped else Sample.Status.COMPLETED
    sample.reward = reward
    sample.metadata = {
        "bfcl_passed": float(passed),
        "bfcl_reward": reward,
        "bfcl_calls": len(calls),
        "bfcl_exec_successes": sum(successes),
        "bfcl_overlong_masked": float(capped),
        "bfcl_task": label["task_id"],
        "bfcl_turn": label["turn_index"],
    }
    return sample
