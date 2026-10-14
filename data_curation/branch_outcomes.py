"""Executed-branch outcomes for cached responses at collected tau2 states (CPU pieces).

Each cached behavior response at a fresh tau2 state is a branch: the episode's history up to that state, then the
response as the agent's message, then the rest of the episode played by the student and the user simulator
(configs/tau_bench_eval ``branch`` mode). A branch's outcome is tau2's reward for the whole trajectory on the task's
own database. Within one state, a response's advantage is the log-ratio of its success probability to the state's,

    A_k = log(p_k / p̄),   p_k = (R_k + 1/2) / 2,   p̄ = mean over the state's responses with an outcome,

the Bayes form of conditioning on success (π(a | s, success) = π(a | s) · P(success | s, a) / P(success | s)),
with each one-sample probability smoothed by half a success. It is in nats, like the anchor's log-ratio, and enters
the training target as an extra reward on the response's own sampled tokens
(data_curation/build_synthetic_shift_target.py, ``outcome`` transform):

    q ∝ b · exp((δ_anchor + A) / α).

Where every response of a state ends the same way, A = 0 and the target is the anchor's alone.

This module turns a sampled completion into the message vLLM would have returned, cuts the episode's history at
the state, writes the branch plan, and turns branch outcomes into per-response advantages.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA_VERSION = "tau2_branch_outcomes_v1"
THINK_END = "</think>"
# vLLM's hermes tool parser: closed blocks, or a final block left open.
TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)


def parse_generated_turn(text: str, finish_reason: str) -> dict | None:
    """The assistant message vLLM would return for a sampled 2507 turn (deepseek_r1 reasoning, hermes tools).

    The reasoning ends at ``</think>`` (the template opened ``<think>`` in the prompt). After it, any
    ``<tool_call>`` blocks become calls, with the text before the first block as content; if a block is not a JSON
    object with a name and object arguments, the parser returns the whole text as content and no calls, as vLLM's
    does. A truncated turn, a turn that never closes its reasoning, or one with neither text nor calls is not a
    branch (None).
    """

    if finish_reason == "length" or THINK_END not in text:
        return None
    after = text.split(THINK_END, 1)[1]
    calls = []
    if "<tool_call>" in after:
        for closed, open_ in TOOL_CALL.findall(after):
            try:
                call = json.loads(closed or open_)
            except json.JSONDecodeError:
                calls = None
                break
            if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(
                call.get("arguments", {}), dict
            ):
                calls = None
                break
            calls.append({"name": call["name"], "arguments": call.get("arguments", {})})
    if calls:
        content = after[: after.index("<tool_call>")]
        return {"content": content or None, "tool_calls": calls}
    if not after.strip():
        return None
    return {"content": after, "tool_calls": None}


def episode_prefix(messages: list[dict], request_index: int) -> list[dict]:
    """The tau2 trajectory before the agent's ``request_index``-th request (the scripted greeting is message 0)."""

    assistants = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    if request_index + 1 >= len(assistants):
        raise ValueError(f"request {request_index} has no recorded response in this trajectory")
    return messages[: assistants[request_index + 1]]


def branch_message(turn: dict, branch_id: str) -> dict:
    """A tau2 AssistantMessage (as JSON) for a parsed turn, with deterministic call ids."""

    calls = None
    if turn["tool_calls"]:
        calls = [
            {"id": f"{branch_id}-{index}", "name": call["name"], "arguments": call["arguments"], "requestor": "assistant"}
            for index, call in enumerate(turn["tool_calls"])
        ]
    return {"role": "assistant", "content": turn["content"], "tool_calls": calls}


def branch_seed(branch_id: str) -> int:
    return int(hashlib.sha256(f"branch:{branch_id}".encode()).hexdigest()[:8], 16) % 1_000_000


def build_plan(prompts: list[dict], rollouts: list[dict], records: dict[tuple[str, str, str], dict]) -> tuple[list, dict]:
    """One branch per parseable cached response; ``records`` maps (collector, domain, task id) to its episode."""

    by_prompt = {row["prompt_id"]: row for row in prompts}
    branches, counts = [], collections.Counter()
    for row in sorted(rollouts, key=lambda item: item["sample_id"]):
        counts["responses"] += 1
        state = by_prompt[row["prompt_id"]]
        turn = parse_generated_turn(row["response"], row["finish_reason"])
        if turn is None:
            counts["truncated" if row["finish_reason"] == "length" else "unparsed"] += 1
            continue
        record = records[(state["collector"], state["domain"], state["task_id"])]
        # An episode the model ended with an error kept its requests (so its states) but no tau2 trajectory to
        # replay; a state whose response was never recorded has no history to cut either.
        if not record.get("result"):
            counts["no_trajectory"] += 1
            continue
        try:
            prefix = episode_prefix(record["result"]["messages"], int(state["request_index"]))
        except ValueError:
            counts["no_recorded_response"] += 1
            continue
        branch_id = row["sample_id"][:24]
        counts["tool_call" if turn["tool_calls"] else "text"] += 1
        branches.append(
            {
                "branch_id": branch_id,
                "sample_id": row["sample_id"],
                "prompt_id": row["prompt_id"],
                "domain": state["domain"],
                "task_id": state["task_id"],
                "seed": branch_seed(branch_id),
                "history": prefix,
                "branch": branch_message(turn, branch_id),
            }
        )
    return branches, dict(counts)


def advantages(rollouts: list[dict], outcomes: dict[str, float]) -> tuple[dict[str, float], dict]:
    """Per response: log(p_k / p̄) with p_k = (R_k + 1/2) / 2 (module docstring); 0 without an outcome.

    A state contributes only when at least two of its responses have outcomes and they differ (otherwise every
    p_k equals p̄ anyway).
    """

    groups: dict[str, list[str]] = collections.defaultdict(list)
    for row in rollouts:
        groups[row["prompt_id"]].append(row["sample_id"])
    result, counts = {}, collections.Counter()
    for prompt_id, samples in groups.items():
        scored = [sample for sample in samples if sample in outcomes]
        probability = {sample: (outcomes[sample] + 0.5) / 2.0 for sample in scored}
        mean = sum(probability.values()) / len(scored) if scored else 1.0
        informative = len(scored) >= 2 and len({outcomes[sample] for sample in scored}) > 1
        counts["informative_states" if informative else "flat_states"] += 1
        for sample in samples:
            result[sample] = math.log(probability[sample] / mean) if informative and sample in outcomes else 0.0
    counts["responses_with_outcome"] = sum(sample in outcomes for samples in groups.values() for sample in samples)
    counts["responses"] = len(result)
    return result, dict(counts)


# Failures the evaluation scores as the model's (0); the user simulator's or a replay's have no outcome.
MODEL_FAILURES = ("agent_error", "context_window_exceeded", "bad_request")


def branch_outcome(record: dict) -> float | None:
    """A branch record's reward, 0 for the model's own failures, None when the branch has no outcome."""

    if record["termination"] in MODEL_FAILURES:
        return 0.0
    if "outcome" not in record:
        return None
    return None if record["outcome"] is None else float(record["outcome"])


ROLLOUT_FIELDS = ("sample_id", "prompt_id", "response", "finish_reason")


def read_rollouts(directory: Path) -> list[dict]:
    """The cached responses' identities, text and finish reasons (rollout shards keep them in a metadata struct)."""
    import pyarrow.parquet as pq

    rows = []
    for path in sorted(directory.glob("*.parquet")):
        for item in pq.read_table(path, columns=["metadata"]).column("metadata").to_pylist():
            rows.append({field: item[field] for field in ROLLOUT_FIELDS})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan", help="write the branch plan (JSONL) for a fresh-state cache")
    plan.add_argument("--prompts", type=Path, required=True)
    plan.add_argument("--rollouts", type=Path, required=True, help="the cache's rollouts directory")
    plan.add_argument("--collection", type=Path, required=True, help="<tau results>/<collection run id>")
    plan.add_argument("--output", type=Path, required=True)
    adv = sub.add_parser("advantages", help="per-response advantages from finished branch records")
    adv.add_argument("--rollouts", type=Path, required=True)
    adv.add_argument("--branches", type=Path, nargs="+", required=True, help="each shard's <tau results>/<run>/<tag>")
    adv.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.action == "plan":
        import pyarrow.parquet as pq

        prompts = pq.read_table(args.prompts).to_pylist()
        records = {}
        for state in prompts:
            key = (state["collector"], state["domain"], state["task_id"])
            if key not in records:
                (path,) = (args.collection / key[0] / key[1]).glob(f"{key[2]}-*.json")
                records[key] = json.loads(path.read_text(encoding="utf-8"))
        branches, counts = build_plan(prompts, read_rollouts(args.rollouts), records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for branch in branches:
                handle.write(json.dumps(branch, sort_keys=True) + "\n")
        summary = {"schema_version": SCHEMA_VERSION, "branches": len(branches), "counts": counts,
                   "by_domain": dict(collections.Counter(branch["domain"] for branch in branches))}
        args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        rollouts = read_rollouts(args.rollouts)
        samples = {row["sample_id"][:24]: row["sample_id"] for row in rollouts}
        outcomes, terminations = {}, collections.Counter()
        for path in sorted(item for root in args.branches for item in root.glob("tau2_*/*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            terminations[record["termination"]] += 1
            outcome = branch_outcome(record)
            if outcome is not None:
                outcomes[samples[record["task_id"]]] = outcome
        result, counts = advantages(rollouts, outcomes)
        counts["terminations"] = dict(terminations)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        summary = {"schema_version": SCHEMA_VERSION, "outcomes": len(outcomes), "counts": counts,
                   "mean_outcome": sum(outcomes.values()) / max(len(outcomes), 1)}
        args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
