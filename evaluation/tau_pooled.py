#!/usr/bin/env python3
"""Paired tau2 comparisons pooled over training seeds and trials, with a task bootstrap (CPU only).

Each model's records live at ``<root>/<tag>/<domain>/trial<k>/*.json`` (configs/tau_bench_eval). A model
group is a list of tags (e.g. a variant's two training seeds); its per-task success is the mean over its
tags and trials. A contrast is a signed sum of groups, e.g. ``{"fresh acc": 1, "cache acc": -1}`` or the
2x2 interaction ``{"fresh decs": 1, "fresh acc": -1, "cache decs": -1, "cache acc": 1}``. Its value is the
mean over tasks of that signed sum (points), reported per domain and over all tasks; the 95% interval
resamples tasks within each domain, so every group is resampled with the same tasks.

    python evaluation/tau_pooled.py --root RUNS/tau-98179d00b25d-full --spec spec.json --output tau_pooled.json

spec: {"groups": {name: [tags]}, "contrasts": {name: {group: sign}}, "domains": [...]}
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def task_success(root: Path, tags: list[str], domain: str) -> tuple[dict[str, float], dict]:
    """Mean success per task over the group's tags and trials, and token/termination bookkeeping.

    Every tag must hold the same trials for every task (a missing conversation would silently reweight a task).
    """
    outcomes: dict[str, list[float]] = collections.defaultdict(list)
    tokens, conversations, terminations = 0, 0, collections.Counter()
    for tag in tags:
        trials: dict[str, set[int]] = collections.defaultdict(set)
        for path in sorted((root / tag / domain).glob("trial*/*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            trials[record["task_id"]].add(record["trial"])
            outcomes[record["task_id"]].append(float(record["success"]))
            tokens += sum(call.get("completion_tokens") or 0 for call in record["agent_calls"])
            conversations += 1
            terminations[record["termination"]] += 1
        if len({frozenset(value) for value in trials.values()}) > 1:
            short = sorted(task for task, value in trials.items() if len(value) < max(map(len, trials.values())))
            raise ValueError(f"{tag}/{domain}: incomplete trial sets for tasks {short[:10]}")
    stats = {
        "conversations": conversations,
        "tokens_per_conversation": tokens / conversations if conversations else None,
        "terminations": dict(terminations),
    }
    return {task: float(np.mean(values)) for task, values in outcomes.items()}, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    domains = spec["domains"]
    success, stats = {}, {}
    for name, tags in spec["groups"].items():
        for domain in domains:
            success[name, domain], stats[f"{name}|{domain}"] = task_success(args.root, tags, domain)
    rng = np.random.default_rng(args.seed)
    tasks = {domain: sorted(set.intersection(*(set(success[name, domain]) for name in spec["groups"]))) for domain in domains}
    for domain in domains:
        union = set.union(*(set(success[name, domain]) for name in spec["groups"]))
        if union != set(tasks[domain]):
            raise ValueError(f"{domain}: groups cover different tasks ({len(union) - len(tasks[domain])} not shared)")
    draws = {domain: rng.integers(0, len(tasks[domain]), size=(args.draws, len(tasks[domain]))) for domain in domains if tasks[domain]}
    report = {"groups": spec["groups"], "tasks": {domain: len(ids) for domain, ids in tasks.items()}, "stats": stats, "contrasts": {}}
    for name, signs in spec["contrasts"].items():
        rows, pooled_values, pooled_boot = {}, [], []
        for domain in domains:
            if not tasks[domain]:
                continue
            values = sum(sign * np.array([success[group, domain][task] for task in tasks[domain]]) for group, sign in signs.items())
            boot = values[draws[domain]].mean(axis=1)
            rows[domain] = {
                "delta": 100 * values.mean(),
                "low": 100 * np.quantile(boot, 0.025),
                "high": 100 * np.quantile(boot, 0.975),
            }
            pooled_values.append(values)
            pooled_boot.append(values[draws[domain]].sum(axis=1))
        total = sum(len(values) for values in pooled_values)
        overall = np.sum(pooled_boot, axis=0) / total
        rows["all_tasks"] = {
            "delta": 100 * np.concatenate(pooled_values).mean(),
            "low": 100 * np.quantile(overall, 0.025),
            "high": 100 * np.quantile(overall, 0.975),
        }
        report["contrasts"][name] = rows
    report["pass1"] = {
        name: {domain: 100 * np.mean(list(success[name, domain].values())) for domain in domains if success[name, domain]}
        for name in spec["groups"]
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, rows in report["contrasts"].items():
        print(name.ljust(44), "  ".join(f"{d}: {r['delta']:+.1f} [{r['low']:+.1f}, {r['high']:+.1f}]" for d, r in rows.items()))


if __name__ == "__main__":
    main()
