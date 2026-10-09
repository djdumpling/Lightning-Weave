"""Training prompts from the decision states trained students visited on AReaL tau2 training tasks.

Each agent request a collector served (configs/tau_bench_eval collection mode) is one decision state:
the public history (system policy, greeting, user turns, earlier calls and their results; no earlier
reasoning) exactly as the agent saw it. The server's own prompt token ids are stored with each request,
so a prompt here is their decoding, kept only when it re-encodes to the identical ids. Every visited step
is eligible (no WRITE/error filtering); up to ``--states-per-episode`` are drawn per episode by hash, then
each (domain, collector) cell draws its quota, so the pool keeps the collectors' visitation within a
fixed allocation. Rows are written in hash order, so stored-order training batches mix the cells.
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


def episode_states(record: dict, tokenizer, max_prompt_tokens: int, counts: collections.Counter) -> list[dict]:
    """Every eligible state of one episode, in request order."""
    states = []
    for index, request in enumerate(record.get("requests") or []):
        counts["requests"] += 1
        render = request.get("server_render") or {}
        if "capture_error" in request or "tokens" not in render:
            counts["no_server_render"] += 1
            continue
        ids = unpack_tokens(render["tokens"])
        if len(ids) != render["count"] or (request.get("prompt_tokens") is not None and len(ids) != request["prompt_tokens"]):
            counts["render_count_mismatch"] += 1
            continue
        prompt = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if tokenizer.encode(prompt, add_special_tokens=False) != ids:
            counts["round_trip_mismatch"] += 1
            continue
        if not prompt.endswith("<|im_start|>assistant\n"):
            counts["not_a_generation_prompt"] += 1
            continue
        if len(ids) > max_prompt_tokens:
            counts["over_length"] += 1
            continue
        counts["eligible"] += 1
        states.append(
            {
                "prompt": prompt,
                "prompt_token_length": len(ids),
                "request_index": index,
                "state_type": state_type(request["messages"]),
                "action_kind": action_kind(request["response"]),
            }
        )
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True, help="<tau results>/<collection run id>")
    parser.add_argument("--collectors", required=True, help="comma-separated collector model tags")
    parser.add_argument("--domains", required=True, help="comma-separated domains, e.g. tau2_airline,tau2_retail")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-prompt-tokens", type=int, required=True)
    parser.add_argument("--states-per-episode", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

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
            states = episode_states(record, tokenizer, args.max_prompt_tokens, counts)
            states.sort(key=lambda state: rank(args.seed, tag, record["task_id"], state["request_index"]))
            for state in states[: args.states_per_episode]:
                pool.append({**state, "domain": domain, "collector": tag, "task_id": record["task_id"]})
        pool.sort(key=lambda state: rank(args.seed, "cell", tag, state["task_id"], state["request_index"]))
        report[f"{domain}|{tag}"] = {
            "counts": dict(counts),
            "pool": len(pool),
            "quota": quota,
            "pool_state_types": dict(collections.Counter(state["state_type"] for state in pool)),
        }
        if len(pool) < quota:
            raise RuntimeError(f"{domain} x {tag}: {len(pool)} eligible states for a quota of {quota}; collect more episodes")
        selected.extend(pool[:quota])

    selected.sort(key=lambda state: rank(args.seed, "order", state["collector"], state["task_id"], state["request_index"]))
    rows = []
    for index, state in enumerate(selected):
        rows.append(
            {
                "prompt": state["prompt"],
                "label": "",
                "prompt_id": rank("tau2-fresh", state["collector"], state["task_id"], state["request_index"])[:24],
                "source_index": index,
                "prompt_token_length": state["prompt_token_length"],
                "domain": state["domain"],
                "collector": state["collector"],
                "task_id": state["task_id"],
                "request_index": state["request_index"],
                "state_type": state["state_type"],
                "action_kind": state["action_kind"],
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
        "selection_seed": args.seed,
        "selected_prompts": len(rows),
        "targets_in_output": False,
        "cells": report,
        "selected_distribution": {
            key: dict(collections.Counter(row[key] for row in rows))
            for key in ("domain", "collector", "state_type", "action_kind")
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
