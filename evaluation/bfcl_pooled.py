#!/usr/bin/env python3
"""Paired BFCL v3 comparisons pooled over replicate pairs, with pre-registered decision rules (CPU only).

A comparison is a list of (arm, reference) run directories, relative to ``--root``. Each pair shares a training seed
and a decoding seed, so the arm is only ever compared with a reference decoded under the same sampling seed. For each
pair, the metrics are:
- accuracy changes in points (leaderboard-weighted): overall, multi-turn, and single-turn (the mean of the non-live
  and live groups);
- relative changes in total generated tokens (%): all entries, multi-turn entries, and single-turn entries.

The pooled value is the mean over pairs, so every entry counts equally across its replicates. One
category-stratified bootstrap over entries resamples every run with the same indices; its interval reflects entry
sampling, with the replicates averaged.

A comparison may carry rules that turn a metric's 95% interval [lo, hi] into a verdict:
- ``["recovery", metric, m]``: "recovers" if lo > 0 (but "partial" if also hi < m), "does not recover" if hi < m,
  otherwise "inconclusive";
- ``["non_inferior", metric, m]``: "no detectable cost" if lo > -m (but "small cost within margin" if also hi < 0),
  "costs" if hi < 0, otherwise "inconclusive";
- ``["reduces", metric]``: "reduced" if hi < 0, otherwise "not shown".

The comparisons file is ``{name: {"pairs": [[arm, reference], ...], "rules": [...]}}``. A four-run item
``[arm, reference, control_arm, control_reference]`` is a difference of changes,
change(arm, reference) - change(control_arm, control_reference): for example, whether a recipe changes an efficiency
term's cost (an interaction), with all four runs resampled with the same entries.

    python evaluation/bfcl_pooled.py --root RESULTS --comparisons comparisons.json --output pooled.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.bfcl_efficiency import MULTI_TURN, V3_CATEGORIES, group_weights, load_run

SINGLE_TURN = tuple(category for category in V3_CATEGORIES if category not in MULTI_TURN)
ACCURACY_METRICS = ("accuracy", "multi_turn_accuracy", "single_turn_accuracy")
TOKEN_METRICS = ("total_tokens", "multi_turn_tokens", "single_turn_tokens")
METRICS = ACCURACY_METRICS + TOKEN_METRICS
TOKEN_SETS = {"total_tokens": V3_CATEGORIES, "multi_turn_tokens": MULTI_TURN, "single_turn_tokens": SINGLE_TURN}


def accuracy_weights(counts: dict[str, int]) -> dict[str, dict[str, float]]:
    groups = group_weights(counts)
    single = {category: weight / 2 for name in ("non_live", "live") for category, weight in groups[name].items()}
    return {"accuracy": groups["overall"], "multi_turn_accuracy": groups["multi_turn"], "single_turn_accuracy": single}


def run_arrays(path: Path) -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
    """{category: (entry ids, correctness, generated tokens)} in entry-id order; every v3 category is required."""
    run = load_run(path, include_usage=False)
    missing = [category for category in V3_CATEGORIES if not run.get(category)]
    if missing:
        raise ValueError(f"{path} is missing categories {missing}")
    arrays = {}
    for category in V3_CATEGORIES:
        entries = sorted(run[category], key=lambda entry: entry["id"])
        arrays[category] = (
            [entry["id"] for entry in entries],
            np.array([float(entry["is_correct"]) for entry in entries]),
            np.array([float(entry["gen_tokens"]) for entry in entries]),
        )
    return arrays


def run_statistics(arrays: dict, weights: dict, indices: dict[str, np.ndarray] | None) -> np.ndarray:
    """Each metric's raw value (accuracy fraction or token sum): shape (len(METRICS),), or (draws, len(METRICS))."""
    means, sums = {}, {}
    for category, (_, correct, tokens) in arrays.items():
        pick = correct if indices is None else correct[indices[category]]
        spent = tokens if indices is None else tokens[indices[category]]
        means[category], sums[category] = pick.mean(axis=-1), spent.sum(axis=-1)
    values = [
        sum(weight * means[category] for category, weight in weights[metric].items()) for metric in ACCURACY_METRICS
    ]
    values += [sum(sums[category] for category in TOKEN_SETS[metric]) for metric in TOKEN_METRICS]
    return np.stack(values, axis=-1)


def change(arm: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Accuracy changes in points, then token changes in percent."""
    accuracy = 100 * (arm[..., :3] - reference[..., :3])
    tokens = 100 * (arm[..., 3:] / reference[..., 3:] - 1)
    return np.concatenate([accuracy, tokens], axis=-1)


def contrast(values: dict[str, np.ndarray], runs: list[str]) -> np.ndarray:
    """change(arm, reference), or for four runs the difference of two changes."""
    if len(runs) == 2:
        return change(values[runs[0]], values[runs[1]])
    if len(runs) == 4:
        return change(values[runs[0]], values[runs[1]]) - change(values[runs[2]], values[runs[3]])
    raise ValueError(f"a comparison item names 2 or 4 runs, not {runs}")


def verdict(rule: list, interval: dict[str, list[float]]) -> str:
    kind, metric = rule[0], rule[1]
    low, high = interval[metric]
    if kind == "recovery":
        margin = rule[2]
        if low > 0:
            return "partial" if high < margin else "recovers"
        return "does not recover" if high < margin else "inconclusive"
    if kind == "non_inferior":
        margin = rule[2]
        if low > -margin:
            return "small cost within margin" if high < 0 else "no detectable cost"
        return "costs" if high < 0 else "inconclusive"
    if kind == "reduces":
        return "reduced" if high < 0 else "not shown"
    raise ValueError(f"unknown rule {rule}")


def analyze(runs: dict[str, dict], comparisons: dict[str, dict], *, draws: int = 2_000, seed: int = 0) -> dict:
    """Pooled changes, bootstrap intervals, and verdicts for every comparison over the loaded ``runs``."""
    first = next(iter(runs.values()))
    for name, arrays in runs.items():
        for category in V3_CATEGORIES:
            if arrays[category][0] != first[category][0]:
                raise ValueError(f"{name} has different {category} entries from the other runs")
    weights = accuracy_weights({category: len(first[category][0]) for category in V3_CATEGORIES})
    rng = np.random.default_rng(seed)
    indices = {
        category: rng.integers(0, len(first[category][0]), (draws, len(first[category][0])))
        for category in V3_CATEGORIES
    }
    point = {name: run_statistics(arrays, weights, None) for name, arrays in runs.items()}
    boot = {name: run_statistics(arrays, weights, indices) for name, arrays in runs.items()}
    report = {}
    for name, item in comparisons.items():
        pairs = item["pairs"]
        per_pair = np.stack([contrast(point, runs) for runs in pairs])
        pooled = np.mean([contrast(boot, runs) for runs in pairs], axis=0)
        interval = {
            metric: [float(np.percentile(pooled[:, k], 2.5)), float(np.percentile(pooled[:, k], 97.5))]
            for k, metric in enumerate(METRICS)
        }
        report[name] = {
            "pairs": [list(pair) for pair in pairs],
            "per_pair": [dict(zip(METRICS, map(float, row), strict=True)) for row in per_pair],
            "pooled": {
                metric: {"delta": float(per_pair[:, k].mean()), "ci95": interval[metric]}
                for k, metric in enumerate(METRICS)
            },
            "verdicts": {" ".join(map(str, rule)): verdict(rule, interval) for rule in item.get("rules", [])},
        }
    return report


def table(report: dict) -> str:
    def cell(item, metric, unit):
        low, high = item["pooled"][metric]["ci95"]
        return f"{item['pooled'][metric]['delta']:+.2f}{unit} [{low:+.2f}, {high:+.2f}]"

    lines = [
        "| comparison | pairs | Δacc | Δacc multi-turn | Δacc single-turn | Δtotal tokens | Δmulti-turn tokens "
        "| Δsingle-turn tokens | verdicts |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, item in report.items():
        cells = [cell(item, metric, "") for metric in ACCURACY_METRICS] + [cell(item, m, "%") for m in TOKEN_METRICS]
        verdicts = "; ".join(f"{rule}: {result}" for rule, result in item["verdicts"].items())
        lines.append(f"| {name} | {len(item['pairs'])} | " + " | ".join(cells) + f" | {verdicts} |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="directory holding the run trees")
    parser.add_argument("--comparisons", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--draws", type=int, default=2_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparisons = json.loads(args.comparisons.read_text(encoding="utf-8"))
    names = sorted({run for item in comparisons.values() for pair in item["pairs"] for run in pair})
    runs = {name: run_arrays(args.root / name) for name in names}
    report = analyze(runs, comparisons, draws=args.draws)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(table(report))


if __name__ == "__main__":
    main()
