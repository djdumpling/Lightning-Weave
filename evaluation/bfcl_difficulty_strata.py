#!/usr/bin/env python3
"""Where does shortening cost accuracy on BFCL? Efficiency arms' changes, split by how hard each entry is.

The premise behind a difficulty gate is that shortening reasoning is safe where the accuracy-trained student
reliably succeeds, and costly where its outcome is uncertain. This tests the premise on BFCL itself, multi-turn
included, from finished runs (CPU only).

For an arm compared with its same-seed reference, entries are stratified by the accuracy-only student's
correctness in two other accuracy-only runs: ``both_right``, ``split``, or ``both_wrong``. Those runs are
disjoint from the compared pair, so the stratification is independent of both. Regression to the mean
affects the arm and its reference alike, and the per-stratum change stays unbiased. Per stratum and group
(all, single-turn, multi-turn), the report gives the entry-mean accuracy change with an entry-bootstrap
interval, the mean per-entry token log-ratio, and the total-token ratio. DECS's two seeds are also pooled,
stratified by V0 alone, which is independent of all four runs.

Two stratification runs give only three coarse strata. The strata describe observed accuracy-only outcomes;
they do not measure what reasoning is worth (see ``data_curation/reasoning_value_probe.py`` for that). Entry
means are unweighted, not leaderboard-weighted. Entries that overflow the serving window in any run are
dropped.

    python evaluation/bfcl_difficulty_strata.py --root RUNS --output strata.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.bfcl_efficiency import load_run
from evaluation.bfcl_shared_factor import R0, R1, R2, J, aligned_arrays

STRATA = {1.0: "both_right", 0.5: "split", 0.0: "both_wrong"}
# (arm, same-seed reference, two accuracy-only stratification runs disjoint from both)
DEFAULT_COMPARISONS = [
    *(
        (f"{J}{name}", R1, (R0, R2))
        for name in (
            "acc-legacy+decs",
            "acc-legacy+decs7b",
            "acc-legacy+l1max",
            "acc-legacy+donor-mean",
            "acc-legacy+decs@low",
            "acc-legacy+decs@high",
            "acc-legacy+heuristic",
            "acc-legacy-scaled",
            "acc-legacy+decs-protect-first",
            "acc-legacy.concise",
        )
    ),
    (f"{J}acc-legacy+decs.s5678", R2, (R0, R1)),
    (f"{J}acc-legacy+decs-protect-first.s5678", R2, (R0, R1)),
    ("opd-concise", R0, (R1, R2)),
]
POOLED = {
    "name": "decs, both seeds",
    "pairs": [(f"{J}acc-legacy+decs", R1), (f"{J}acc-legacy+decs.s5678", R2)],
    "by": R0,
}


def interval(values: np.ndarray, *, draws: int, rng) -> list[float] | list[None]:
    if len(values) < 2:
        return [None, None]
    boot = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)]
    return np.percentile(boot, [2.5, 97.5]).tolist()


def stratum_row(delta_correct, arm_tokens, ref_tokens, keep, *, draws, rng) -> dict:
    """Accuracy (points) and token change over one stratum's entries."""
    if not keep.any():
        return {"entries": 0}
    d = delta_correct[keep]
    return {
        "entries": int(keep.sum()),
        "delta_accuracy": 100 * float(d.mean()),
        "ci95": [None if v is None else 100 * v for v in interval(d, draws=draws, rng=rng)],
        "mean_token_log_ratio": float((arm_tokens[keep] - ref_tokens[keep]).mean()),
        "total_token_ratio": float(np.expm1(arm_tokens[keep]).sum() / np.expm1(ref_tokens[keep]).sum() - 1),
    }


def stratify(log_tokens, correct, groups, comparisons, pooled=None, *, draws: int = 1_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    subsets = {
        "all": np.ones(len(groups), dtype=bool),
        "single_turn": groups != "multi_turn",
        "multi_turn": groups == "multi_turn",
    }
    report = {"entries": len(groups), "comparisons": {}}
    for arm, ref, by in comparisons:
        if len({arm, ref, *by}) != 2 + len(by):
            raise ValueError(f"{arm}: stratification runs must be disjoint from the compared pair")
        rate = np.mean([correct[run] for run in by], axis=0)
        delta = correct[arm] - correct[ref]
        report["comparisons"][arm] = {
            "reference": ref,
            "stratified_by": list(by),
            "strata": {
                group: {
                    label: stratum_row(
                        delta, log_tokens[arm], log_tokens[ref], keep & (rate == value), draws=draws, rng=rng
                    )
                    for value, label in STRATA.items()
                }
                for group, keep in subsets.items()
            },
        }
    if pooled:
        if any(pooled["by"] in pair for pair in pooled["pairs"]):
            raise ValueError("the pooled stratification run must be disjoint from every pooled pair")
        delta = np.mean([correct[a] - correct[r] for a, r in pooled["pairs"]], axis=0)
        arm_tokens = np.mean([log_tokens[a] for a, _ in pooled["pairs"]], axis=0)
        ref_tokens = np.mean([log_tokens[r] for _, r in pooled["pairs"]], axis=0)
        right = correct[pooled["by"]] == 1.0
        report["pooled"] = {
            "name": pooled["name"],
            "stratified_by": pooled["by"],
            "strata": {
                group: {
                    label: stratum_row(delta, arm_tokens, ref_tokens, keep & mask, draws=draws, rng=rng)
                    for label, mask in (("right", right), ("wrong", ~right))
                }
                for group, keep in subsets.items()
            },
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="directory holding one downloaded run per tag")
    parser.add_argument("--draws", type=int, default=1_000)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tags = sorted({tag for arm, ref, by in DEFAULT_COMPARISONS for tag in (arm, ref, *by)})
    runs = {tag: load_run(args.root / tag, require_usage=False) for tag in tags}
    log_tokens, correct, groups, dropped = aligned_arrays(runs)
    report = stratify(log_tokens, correct, groups, DEFAULT_COMPARISONS, POOLED, draws=args.draws)
    report["dropped_overflow_entries"] = dropped
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"entries {report['entries']} (dropped {dropped} overflowing); Δaccuracy in points, tokens as total ratio")
    for arm, item in [*report["comparisons"].items(), (report["pooled"]["name"], report["pooled"])]:
        for group in ("all", "multi_turn"):
            cells = []
            for label, row in item["strata"][group].items():
                if row["entries"]:
                    low, high = row["ci95"]
                    bounds = "" if low is None else f" [{low:+.1f}, {high:+.1f}]"
                    cells.append(
                        f"{label} n={row['entries']}: {row['delta_accuracy']:+.1f}{bounds}, "
                        f"{100 * row['total_token_ratio']:+.0f}%"
                    )
            print(f"{arm.replace(J, '')} [{group}] " + " | ".join(cells))


if __name__ == "__main__":
    main()
