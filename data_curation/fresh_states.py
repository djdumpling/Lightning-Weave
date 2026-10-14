"""Training prompts from the decision states trained students visited on AReaL tau2 training tasks.

Each agent request a collector served (configs/tau_bench_eval collection mode) is one decision state:
the history (system policy, greeting, user turns, earlier calls and their results, plus any reasoning
retained by the collection profile) exactly as the agent saw it. The server's own prompt token ids are stored with each request,
so a prompt here is their decoding, kept only when it re-encodes to the identical ids and ends with the
student's generation prompt. Every visited step is eligible (no WRITE/error filtering); up to
``--states-per-episode`` are drawn per episode by hash (with ``--prioritize-decisions``, the episode's
decision states first: its write and transfer steps and the step whose call first returned an error, as in
the mentor's focused turns), then each (domain, collector) cell draws its quota, so the pool keeps the
collectors' visitation within a fixed allocation. Rows are written in hash order, so stored-order training
batches mix the cells.

``--episode-token-budget`` reproduces a cumulative per-episode generation budget (the mentor's collection stops
an episode once its assistant turns have generated 2,048 tokens, reasoning included): only states reached with
budget left are eligible, and each keeps the remainder as its response budget. A budget leaves fewer states than
an uncapped pool, so ``--repeat-to-quota`` fills the quota with further copies of the cell's states in its order,
each copy a new prompt id sampled with its own seed (as repeated episodes of one task revisit its early states).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from configs.tau_bench_eval.harness import unpack_tokens

SCHEMA_VERSION = "tau2_fresh_states_v1"
READ_PREFIXES = ("get_", "search_", "list_", "find_", "calculate")


def rank(*parts: object) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode()).hexdigest()


def state_type(messages: list[dict]) -> str:
    last = messages[-1]["role"]
    if last == "tool":
        return "after_tool"
    users = sum(message["role"] == "user" for message in messages)
    return "first_user_turn" if users == 1 else "later_user_turn"


def action_kind(response: dict) -> str:
    if response.get("finish_reason") == "length":
        return "truncated"
    names = [call["name"] for call in response.get("tool_calls") or []]
    if not names:
        return "text"
    if any(name.startswith("transfer_to_human") for name in names):
        return "transfer"
    return "read" if all(name.startswith(READ_PREFIXES) or name == "think" for name in names) else "write"


def error_steps(requests: list[dict]) -> list[int]:
    """Indices of the requests whose tool calls returned an error, read from the next request's tool messages."""
    steps = []
    for index in range(len(requests) - 1):
        if not ((requests[index].get("response") or {}).get("tool_calls")):
            continue
        results = []
        for message in reversed(requests[index + 1].get("messages") or []):
            if message.get("role") != "tool":
                break
            results.append(message)
        if any(str(message.get("content") or "").startswith("Error") for message in results):
            steps.append(index)
    return steps


def episode_states(
    record: dict,
    tokenizer,
    max_prompt_tokens: int,
    counts: collections.Counter,
    generation_prompt: str = "<|im_start|>assistant\n",
    episode_token_budget: int | None = None,
) -> list[dict]:
    """Every eligible state of one episode, in request order, each marked if it is a decision state.

    ``generated_before`` counts the tokens the episode's earlier requests generated; with an episode budget, a
    state is eligible only while some budget remains, and ``response_budget`` is what remains."""
    states = []
    requests = record.get("requests") or []
    errors = error_steps(requests)
    generated = 0
    for index, request in enumerate(requests):
        counts["requests"] += 1
        generated_before = generated
        completion = request.get("completion_tokens")
        if completion is None and episode_token_budget is not None:
            raise ValueError(f"{record.get('task_id')}: request {index} has no completion token count")
        generated += completion or 0
        if episode_token_budget is not None and generated_before >= episode_token_budget:
            counts["over_episode_budget"] += 1
            continue
        render = request.get("server_render") or {}
        if "capture_error" in request or "tokens" not in render:
            counts["no_server_render"] += 1
            continue
        ids = unpack_tokens(render["tokens"])
        # Qwen3.5 collection returns the IDs actually consumed by generation.
        # The separate rendering is retained as an independent serialization check.
        if request.get("actual_prompt_token_ids") is not None:
            actual_ids = unpack_tokens(request["actual_prompt_token_ids"])
            if actual_ids != ids:
                raise ValueError(f"{record.get('task_id')}: request {index} served/rendered prompt mismatch")
            ids = actual_ids
        if len(ids) != render["count"] or (request.get("prompt_tokens") is not None and len(ids) != request["prompt_tokens"]):
            counts["render_count_mismatch"] += 1
            continue
        prompt = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if tokenizer.encode(prompt, add_special_tokens=False) != ids:
            counts["round_trip_mismatch"] += 1
            continue
        if not prompt.endswith(generation_prompt):
            counts["not_a_generation_prompt"] += 1
            continue
        if len(ids) > max_prompt_tokens:
            counts["over_length"] += 1
            continue
        counts["eligible"] += 1
        kind = action_kind(request["response"])
        states.append(
            {
                "prompt": prompt,
                "prompt_token_length": len(ids),
                "request_index": index,
                "state_type": state_type(request["messages"]),
                "action_kind": kind,
                "decision_state": kind in ("write", "transfer") or (bool(errors) and index == errors[0]),
                "generated_before": generated_before,
                **({"response_budget": episode_token_budget - generated_before} if episode_token_budget is not None else {}),
            }
        )
    return states


def fill_quota(pool: list[dict], quota: int) -> list[dict]:
    """The first ``quota`` states, cycling through further copies (copy_index 1, 2, ...) when the pool is short."""
    if not pool:
        raise RuntimeError("no eligible states to repeat")
    return [{**pool[i % len(pool)], "copy_index": i // len(pool)} for i in range(quota)]


def prompt_id(state: dict) -> str:
    """A state's id; a repeated copy's adds its copy index, so the first copy keeps the uncapped pools' id."""
    parts = ["tau2-fresh", state["collector"], state["task_id"], state["request_index"]]
    if state.get("copy_index"):
        parts += ["copy", state["copy_index"]]
    return rank(*parts)[:24]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True, help="<tau results>/<collection run id>")
    parser.add_argument("--collectors", required=True, help="comma-separated collector model tags")
    parser.add_argument("--domains", required=True, help="comma-separated domains, e.g. tau2_airline,tau2_retail")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-prompt-tokens", type=int, required=True)
    parser.add_argument("--states-per-episode", type=int, required=True)
    parser.add_argument("--generation-prompt", default="<|im_start|>assistant\n", help="the student template's")
    parser.add_argument("--prioritize-decisions", action="store_true", help="draw each episode's decision states first")
    parser.add_argument("--episode-token-budget", type=int, default=None, help="cumulative generated tokens per episode")
    parser.add_argument("--repeat-to-quota", action="store_true", help="fill a short cell with further copies of its states")
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.episode_token_budget is not None and args.episode_token_budget <= 0:
        raise ValueError("--episode-token-budget must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    collectors = [tag for tag in args.collectors.split(",") if tag]
    domains = [name for name in args.domains.split(",") if name]
    cells = [(domain, tag) for domain in domains for tag in collectors]
    if args.num_prompts % len(cells):
        raise ValueError(f"{args.num_prompts} prompts do not split evenly over {len(cells)} cells")
    quota = args.num_prompts // len(cells)

    selected, report = [], {}
    for domain, tag in cells:
        counts: collections.Counter = collections.Counter()
        pool = []
        files = sorted((args.collection / tag / domain).glob("*.json"))
        for path in files:
            record = json.loads(path.read_text(encoding="utf-8"))
            counts["episodes"] += 1
            counts[f"termination:{record['termination']}"] += 1
            states = episode_states(
                record, tokenizer, args.max_prompt_tokens, counts, args.generation_prompt, args.episode_token_budget
            )
            states.sort(
                key=lambda state: (
                    args.prioritize_decisions and not state["decision_state"],
                    rank(args.seed, tag, record["task_id"], state["request_index"]),
                )
            )
            for state in states[: args.states_per_episode]:
                pool.append({**state, "domain": domain, "collector": tag, "task_id": record["task_id"]})
        pool.sort(key=lambda state: rank(args.seed, "cell", tag, state["task_id"], state["request_index"]))
        report[f"{domain}|{tag}"] = {
            "counts": dict(counts),
            "pool": len(pool),
            "quota": quota,
            "pool_state_types": dict(collections.Counter(state["state_type"] for state in pool)),
            "pool_decision_states": sum(state["decision_state"] for state in pool),
            "repeated_states": max(quota - len(pool), 0) if args.repeat_to_quota else 0,
        }
        if len(pool) < quota and not args.repeat_to_quota:
            raise RuntimeError(f"{domain} x {tag}: {len(pool)} eligible states for a quota of {quota}; collect more episodes")
        selected.extend(fill_quota(pool, quota))

    selected.sort(
        key=lambda state: rank(args.seed, "order", state["collector"], state["task_id"], state["request_index"], state["copy_index"])
        if state["copy_index"]
        else rank(args.seed, "order", state["collector"], state["task_id"], state["request_index"])
    )
    rows = []
    for index, state in enumerate(selected):
        rows.append(
            {
                "prompt": state["prompt"],
                "label": "",
                "prompt_id": prompt_id(state),
                "source_index": index,
                "prompt_token_length": state["prompt_token_length"],
                "domain": state["domain"],
                "collector": state["collector"],
                "task_id": state["task_id"],
                "request_index": state["request_index"],
                "state_type": state["state_type"],
                "action_kind": state["action_kind"],
                "decision_state": state["decision_state"],
                "generated_before": state["generated_before"],
                "copy_index": state["copy_index"],
                **({"response_budget": state["response_budget"]} if "response_budget" in state else {}),
            }
        )
    if len({row["prompt_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate prompt ids")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    lengths = sorted(row["prompt_token_length"] for row in rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "collection": str(args.collection),
        "collectors": collectors,
        "domains": domains,
        "max_prompt_length": args.max_prompt_tokens,
        "states_per_episode": args.states_per_episode,
        "generation_prompt": args.generation_prompt,
        "prioritize_decisions": args.prioritize_decisions,
        "episode_token_budget": args.episode_token_budget,
        "repeat_to_quota": args.repeat_to_quota,
        "selection_seed": args.seed,
        "selected_prompts": len(rows),
        "targets_in_output": False,
        "cells": report,
        "selected_distribution": {
            key: dict(collections.Counter(row[key] for row in rows))
            for key in ("domain", "collector", "state_type", "action_kind", "decision_state", "copy_index")
        },
        "student_prompt_token_lengths": {
            "min": lengths[0],
            "p50": lengths[len(lengths) // 2],
            "p90": lengths[int(0.9 * len(lengths))],
            "max": lengths[-1],
        },
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("selected_prompts", "selected_distribution", "student_prompt_token_lengths")}, indent=2))


if __name__ == "__main__":
    main()
