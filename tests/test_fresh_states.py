"""The fresh-state pilot's CPU pieces: task audit, state extraction, collection records, tag resolution, contrasts."""

from __future__ import annotations

import collections
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.bfcl_eval import config as BFCL  # noqa: E402
from configs.tau_bench_eval import harness  # noqa: E402
from data_curation import areal_tau2_tasks as tasks  # noqa: E402
from data_curation import fresh_states  # noqa: E402
from evaluation import bfcl_pooled  # noqa: E402


def task(task_id: str, domain: str, text: str, actions=None) -> dict:
    return {
        "id": task_id,
        "user_scenario": {"instructions": {"domain": domain, "task_instructions": text}},
        "evaluation_criteria": {"actions": actions or []},
        "db_path": f"tau2_rl_database/tau2_{domain}_db.json",
    }


def test_audit_excludes_shared_objects_users_and_text():
    evaluation = {
        "airline": [task("0", "airline", "Cancel reservation EHGLP3. You are emma_kim_9957.")] * 50,
        "retail": [task("0", "retail", "Return order #W2378156 for a refund please today.")] * 114,
    }
    training = [
        task("a1", "airline", "Change reservation EHGLP3 to tomorrow."),  # shared reservation
        task("a2", "airline", "I am emma_kim_9957 and want a new booking."),  # shared user
        task("a3", "airline", "Book flight HAT084 for john_doe_1234 on reservation ZZZ9ZZ."),  # flight numbers are shared catalog
        task("a4", "airline", "Help me.", [{"name": "get_user_details", "arguments": {"user_id": "emma_kim_9957"}}]),
        task("r1", "retail", "Return order #W2378156 for a refund please today."),  # shared order and text
        task("r2", "retail", "Exchange order #W1111111 for a blue one."),
    ]
    report = tasks.audit(training, evaluation)
    assert report["eligible"] == {"airline": ["a3"], "retail": ["r2"]}
    reasons = {item["id"]: item for item in report["excluded"]}
    assert reasons["a1"]["shared_object"] == ["EHGLP3"] and reasons["a2"]["shared_user"] == ["emma_kim_9957"]
    assert reasons["a4"]["shared_user"] == ["emma_kim_9957"]  # found in the training task's own gold actions
    assert reasons["r1"]["text_overlap"] == 1.0


def test_tokens_round_trip_and_state_labels():
    ids = [0, 1, 151935, 42] * 1000
    assert harness.unpack_tokens(harness.pack_tokens(ids)) == ids
    assert fresh_states.action_kind({"tool_calls": [{"name": "get_user_details"}], "finish_reason": "tool_calls"}) == "read"
    assert fresh_states.action_kind({"tool_calls": [{"name": "cancel_reservation"}], "finish_reason": "tool_calls"}) == "write"
    assert fresh_states.action_kind({"tool_calls": [], "finish_reason": "stop"}) == "text"
    assert fresh_states.action_kind({"tool_calls": [], "finish_reason": "length"}) == "truncated"
    messages = [{"role": "system"}, {"role": "assistant"}, {"role": "user"}]
    assert fresh_states.state_type(messages) == "first_user_turn"
    assert fresh_states.state_type(messages + [{"role": "assistant"}, {"role": "tool"}]) == "after_tool"
    assert fresh_states.state_type(messages + [{"role": "assistant"}, {"role": "user"}]) == "later_user_turn"


class Tokenizer:
    """Two tokens: 3 is "x", 7 is the generation prompt."""

    PROMPT = "<|im_start|>assistant\n"

    def decode(self, ids, **_):
        return "".join("x" if token == 3 else self.PROMPT for token in ids)

    def encode(self, text, **_):
        ids = []
        for index, part in enumerate(text.split(self.PROMPT)):
            ids += ([7] if index else []) + [3] * len(part)
        return ids


def test_episode_states_require_the_servers_exact_tokens():
    def request(ids, count=None):
        return {
            "messages": [{"role": "system"}, {"role": "user"}],
            "response": {"tool_calls": [], "finish_reason": "stop"},
            "prompt_tokens": len(ids) if count is None else count,
            "server_render": {"count": len(ids), "tokens": harness.pack_tokens(ids)},
        }

    record = {
        "requests": [request([3, 3, 7]), request([3, 7], count=5), request([3, 3]), {"capture_error": "x"}, request([3] * 9 + [7])]
    }
    counts = collections.Counter()
    states = fresh_states.episode_states(record, Tokenizer(), 5, counts)
    assert [state["request_index"] for state in states] == [0]
    assert counts["render_count_mismatch"] == counts["not_a_generation_prompt"] == 1
    assert counts["no_server_render"] == counts["over_length"] == 1


def test_run_conversation_records_a_collection_episode():
    domain = SimpleNamespace(name="tau2_airline", suite="tau2", domain="airline", user_model="user")
    router = harness.Router(
        "model", harness.Endpoint("a", "k"), harness.Endpoint("b", "k"), {}, {}, harness.Retry(1, 0, 0)
    )
    job = harness.Job("airline_1", None, 0, 7)

    def play():
        router.local.requests.append({"messages": []})
        return {"reward": -1.0, "termination": "user_stop", "result": {}, "fields": {"db_path": "db.json"}}

    scored = harness.run_conversation(domain, job, None, router, play=play)
    assert "requests" not in scored and scored["db_path"] == "db.json" and scored["seed"] == 7
    router.capture = lambda kwargs, response: {}
    collected = harness.run_conversation(domain, job, None, router, play=play)
    assert collected["requests"] == [{"messages": []}] and not collected["success"]
    failed = harness.run_conversation(domain, job, None, router, play=lambda: 1 / 0)
    assert failed["termination"] == "agent_error" and failed["reward"] == 0.0


def test_tau_tags_resolve_like_bfcl(monkeypatch):
    # A private copy of the tau config under its own profile, so the shared module is untouched.
    monkeypatch.setenv("TAU_PROFILE", "user235b-4b")
    spec = importlib.util.spec_from_file_location("tau_config_user235b_4b", ROOT / "configs/tau_bench_eval/config.py")
    config = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, config)
    spec.loader.exec_module(config)
    for tag in ("ae.joint.acc-legacy", "ae.joint.acc-legacy+decs.s5678", "ae.fresh.acc-legacy+decs-fixed.s5678"):
        assert config.model_path(tag) == BFCL.resolve_model(tag).path
        config.check_agent_window(config.PROTOCOL, tag)
    with pytest.raises(ValueError):
        config.model_path("ae.joint")
    assert config.PROTOCOL.max_model_len == 40_960 and config.PROTOCOL.user_server is not None


def test_four_run_contrast_is_a_difference_of_changes():
    values = {
        name: np.array([0.5 + delta, 0.5, 0.5, 100.0, 50.0, 50.0])
        for name, delta in (("fd", 0.0), ("fa", 0.1), ("cd", 0.0), ("ca", 0.3))
    }
    out = bfcl_pooled.contrast(values, ["fd", "fa", "cd", "ca"])
    assert out[0] == pytest.approx(100 * ((0.0 - 0.1) - (0.0 - 0.3)))


def test_decision_states_are_writes_transfers_and_the_first_failed_call():
    def request(ids, calls, messages):
        return {
            "messages": messages,
            "response": {"tool_calls": [{"name": name} for name in calls], "finish_reason": "tool_calls" if calls else "stop"},
            "prompt_tokens": len(ids),
            "server_render": {"count": len(ids), "tokens": harness.pack_tokens(ids)},
        }

    user = [{"role": "system"}, {"role": "user"}]
    tool = lambda text: {"role": "tool", "content": text}  # noqa: E731
    requests = [
        request([3, 7], ["get_user_details"], user),
        request([3, 3, 7], ["get_reservation_details"], user + [{"role": "assistant"}, tool("{}")]),
        request([3, 3, 3, 7], ["cancel_reservation"], user + [{"role": "assistant"}, tool("Error: not found")]),
        request([3, 3, 3, 3, 7], ["get_reservation_details"], user + [{"role": "assistant"}, tool("ok")]),
        request([3] * 5 + [7], [], user + [{"role": "assistant"}, tool("Error: again")]),
        request([3] * 6 + [7], ["transfer_to_human_agents"], user + [{"role": "assistant"}, {"role": "user"}]),
    ]
    assert fresh_states.error_steps(requests) == [1, 3]
    states = fresh_states.episode_states({"requests": requests}, Tokenizer(), 100, collections.Counter())
    assert [state["decision_state"] for state in states] == [False, True, True, False, False, True]
    assert [state["action_kind"] for state in states] == ["read", "read", "write", "read", "text", "transfer"]


def test_a_thinking_template_generation_prompt_is_required_when_given():
    class ThinkingTokenizer(Tokenizer):
        PROMPT = "<|im_start|>assistant\n<think>\n"

    def request(ids):
        return {
            "messages": [{"role": "system"}, {"role": "user"}],
            "response": {"tool_calls": [], "finish_reason": "stop"},
            "prompt_tokens": len(ids),
            "server_render": {"count": len(ids), "tokens": harness.pack_tokens(ids)},
        }

    record = {"requests": [request([3, 3, 7])]}
    counts = collections.Counter()
    assert fresh_states.episode_states(record, ThinkingTokenizer(), 10, counts) == []  # default prompt: refused
    assert counts["not_a_generation_prompt"] == 1
    (state,) = fresh_states.episode_states(record, ThinkingTokenizer(), 10, counts, ThinkingTokenizer.PROMPT)
    assert state["request_index"] == 0 and not state["decision_state"]


def budget_request(ids, completion, calls=()):
    return {
        "messages": [{"role": "system"}, {"role": "user"}],
        "response": {"tool_calls": [{"name": name} for name in calls], "finish_reason": "tool_calls" if calls else "stop"},
        "prompt_tokens": len(ids),
        "completion_tokens": completion,
        "server_render": {"count": len(ids), "tokens": harness.pack_tokens(ids)},
    }


def test_an_episode_budget_keeps_states_reached_with_budget_left():
    # generated before each request: 0, 900, 1,700, 2,047, 2,048 (the last two straddle the 2,048 budget)
    record = {"requests": [
        budget_request([3, 7], 900, ["get_user_details"]),
        budget_request([3, 3, 7], 800, ["get_order_details"]),
        budget_request([3, 3, 3, 7], 347, ["get_product_details"]),
        budget_request([3, 3, 3, 3, 7], 1, ["cancel_pending_order"]),
        budget_request([3] * 5 + [7], 400, ["return_delivered_order_items"]),
    ]}
    counts = collections.Counter()
    states = fresh_states.episode_states(record, Tokenizer(), 100, counts, episode_token_budget=2_048)
    assert [s["request_index"] for s in states] == [0, 1, 2, 3]
    assert [s["generated_before"] for s in states] == [0, 900, 1_700, 2_047]
    assert [s["response_budget"] for s in states] == [2_048, 1_148, 348, 1]
    assert counts["over_episode_budget"] == 1
    uncapped = fresh_states.episode_states(record, Tokenizer(), 100, collections.Counter())
    assert len(uncapped) == 5 and "response_budget" not in uncapped[0] and uncapped[4]["generated_before"] == 2_048
    # an ineligible earlier request still spends budget
    record["requests"][0]["server_render"] = {}
    states = fresh_states.episode_states(record, Tokenizer(), 100, collections.Counter(), episode_token_budget=2_048)
    assert [s["request_index"] for s in states] == [1, 2, 3] and states[0]["response_budget"] == 1_148
    record["requests"][1]["completion_tokens"] = None
    with pytest.raises(ValueError, match="completion token count"):
        fresh_states.episode_states(record, Tokenizer(), 100, collections.Counter(), episode_token_budget=2_048)


def test_repeated_copies_fill_the_quota_with_new_prompt_ids():
    pool = [{"collector": "c", "task_id": str(i), "request_index": 0} for i in range(3)]
    filled = fresh_states.fill_quota(pool, 7)
    assert [s["task_id"] for s in filled] == ["0", "1", "2", "0", "1", "2", "0"]
    assert [s["copy_index"] for s in filled] == [0, 0, 0, 1, 1, 1, 2]
    assert [s["copy_index"] for s in fresh_states.fill_quota(pool, 2)] == [0, 0]
    ids = [fresh_states.prompt_id(s) for s in filled]
    assert len(set(ids)) == 7
    # the first copy keeps the id an uncapped pool gives the same state
    assert ids[0] == fresh_states.rank("tau2-fresh", "c", "0", 0)[:24]
    with pytest.raises(RuntimeError):
        fresh_states.fill_quota([], 1)


def test_rollout_requests_take_each_prompts_budget_and_a_spaced_seed():
    spec = importlib.util.spec_from_file_location("collect", ROOT / "data_curation" / "collect_direct_opd_rollouts.py")
    collect = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collect)
    plain = SimpleNamespace(max_response_length=8_192, seed=42, max_tokens_key=None, seed_offset_key=None)
    assert collect.request_options({}, plain) == (8_192, 42)
    budgeted = SimpleNamespace(max_response_length=8_192, seed=42, max_tokens_key="response_budget", seed_offset_key="copy_index")
    assert collect.request_options({"response_budget": 348, "copy_index": 0}, budgeted) == (348, 42)
    assert collect.request_options({"response_budget": 9_000, "copy_index": 1}, budgeted) == (8_192, 42 + collect.SEED_OFFSET_STRIDE)
    # a copy's n consecutive sample seeds never overlap the first copy's
    assert collect.SEED_OFFSET_STRIDE > 64
    with pytest.raises(ValueError):
        collect.request_options({"response_budget": 0, "copy_index": 0}, budgeted)


def test_generation_prompt_ids_must_agree_with_independent_rendering():
    request = {
        "messages": [{"role": "user"}],
        "response": {"tool_calls": [], "finish_reason": "stop"},
        "prompt_tokens": 2,
        "server_render": {"count": 2, "tokens": harness.pack_tokens([3, 7])},
        "actual_prompt_token_ids": harness.pack_tokens([3, 7]),
    }
    record = {"task_id": "retail_smoke", "requests": [request]}
    assert len(fresh_states.episode_states(record, Tokenizer(), 5, collections.Counter())) == 1
    request["actual_prompt_token_ids"] = harness.pack_tokens([7, 3])
    with pytest.raises(ValueError, match="served/rendered prompt mismatch"):
        fresh_states.episode_states(record, Tokenizer(), 5, collections.Counter())
