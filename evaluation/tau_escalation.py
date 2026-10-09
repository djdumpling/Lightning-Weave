#!/usr/bin/env python3
"""Transfers to a human in tau2 records: when they are needed, what they cost, and how two models differ (CPU only).

A transfer is necessary when the task's gold actions include transfer_to_human_agents; any other transfer is
unnecessary by that standard. tau2 v0.1.3 grades the final database (and communicated information), so a transfer
on a task that needs no write can still pass. Reports, per model group and domain:

- success by task type (gold writes or none) and decision (transfer or stay);
- for each (arm, reference) tag pair, matched by task and trial, the net change in successes split by how the
  transfer decision changed.

    python evaluation/tau_escalation.py --root RUNS/tau-98179d00b25d-full --tau2-domains TAU2/data/tau2/domains \\
        --spec configs/agent_eff/fresh_tau_comparisons.json --pairs ae.fresh.acc-legacy:ae.joint.acc-legacy
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

READ_PREFIXES = ("get_", "search_", "list_", "find_", "calculate", "transfer_to_human")


def gold(domains_dir: Path, domain: str) -> dict[str, dict]:
    """Per task: whether its gold actions include a transfer, and whether they include any write."""
    tasks = json.loads((domains_dir / domain.split("_", 1)[1] / "tasks.json").read_text(encoding="utf-8"))
    out = {}
    for task in tasks:
        names = [action["name"] for action in (task.get("evaluation_criteria") or {}).get("actions") or []]
        out[task["id"]] = {
            "needs_transfer": "transfer_to_human_agents" in names,
            "writes": any(not name.startswith(READ_PREFIXES) for name in names),
        }
    return out


def episodes(root: Path, tag: str, domain: str) -> dict[tuple[str, int], dict]:
    out = {}
    for path in (root / tag / domain).glob("trial*/*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        messages = (record.get("result") or {}).get("messages") or []
        transferred = any(
            call["name"] == "transfer_to_human_agents"
            for message in messages
            if message.get("role") == "assistant"
            for call in message.get("tool_calls") or []
        )
        out[record["task_id"], record["trial"]] = {"success": bool(record["success"]), "transferred": transferred}
    return out


def change(arm: dict, reference: dict, needs_transfer: bool) -> str:
    if needs_transfer:
        return "necessary-transfer tasks"
    if arm["transferred"] == reference["transferred"]:
        return "transfer unchanged"
    return "arm drops an unneeded transfer" if reference["transferred"] else "arm adds an unneeded transfer"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tau2-domains", type=Path, required=True, help="tau2-bench data/tau2/domains at the eval pin")
    parser.add_argument("--spec", type=Path, required=True, help="a tau_pooled spec: its groups and domains are used")
    parser.add_argument("--pairs", default="", help="comma-separated arm_tag:reference_tag pairs to compare")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    pairs = [tuple(item.split(":")) for item in args.pairs.split(",") if item]
    report = {"by_task_type": {}, "paired": {}}
    for domain in spec["domains"]:
        info = gold(args.tau2_domains, domain)
        cache = {}

        def load(tag: str) -> dict:
            if tag not in cache:
                cache[tag] = episodes(args.root, tag, domain)
            return cache[tag]

        for group, tags in spec["groups"].items():
            cells = collections.defaultdict(lambda: [0, 0])
            for tag in tags:
                for (task_id, _), episode in load(tag).items():
                    key = f"{'write' if info[task_id]['writes'] else 'no-write'} task / {'transfer' if episode['transferred'] else 'stay'}"
                    cells[key][0] += episode["success"]
                    cells[key][1] += 1
            report["by_task_type"][f"{group}|{domain}"] = {key: {"success": s / n, "n": n} for key, (s, n) in sorted(cells.items())}
        for arm, reference in pairs:
            net, n = collections.Counter(), collections.Counter()
            a, r = load(arm), load(reference)
            for key in a.keys() & r.keys():
                bucket = change(a[key], r[key], info[key[0]]["needs_transfer"])
                n[bucket] += 1
                net[bucket] += a[key]["success"] - r[key]["success"]
            report["paired"][f"{arm} vs {reference}|{domain}"] = {bucket: {"net": net[bucket], "pairs": n[bucket]} for bucket in sorted(n)}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key, cells in report["by_task_type"].items():
        print(key.ljust(36), "  ".join(f"{cell}: {100 * value['success']:.0f}% of {value['n']}" for cell, value in cells.items()))
    for key, buckets in report["paired"].items():
        print(key, "|", "  ".join(f"{bucket}: {value['net']:+d} (of {value['pairs']})" for bucket, value in buckets.items()))


if __name__ == "__main__":
    main()
