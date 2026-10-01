#!/usr/bin/env python3
"""Do different ways of cutting reasoning cut the same BFCL entries? A descriptive overlap analysis.

Every efficiency lever we tried trades accuracy for tokens at about the same aggregate rate. That rate is a
descriptive fit. Many different parameter changes can share a ratio of accuracy change to token change, so
neither a shared slope nor shared entries would establish a shared mechanism, and different entries would not
guarantee that combining levers helps. This script measures how much the levers' per-entry effects overlap,
as descriptive evidence only (CPU, from finished runs).

For an arm a evaluated against a reference run R(a) (the accuracy-only student of the same seed, or V0 for a
decoding change on V0), the per-entry token change is d_a(e) = log(1 + tokens_a(e)) − log(1 + tokens_R(e)).
Two arms measured against the same reference share −log(1 + tokens_R(e)), which by itself makes their changes
correlate, so every correlation is computed only between arms whose runs are all distinct. Distinct runs are
not guaranteed to be independent, though: the null checks below test that.

- ``raw``: correlations between arms with disjoint runs. Attenuated by per-entry noise; exploratory.
- ``reliability``: for a lever run twice with disjoint runs (two training seeds, or the concise prompt on two
  independently trained students), the correlation of its two repeats. This is the only reliability used; it
  makes no assumption about the lever's noise level.
- ``latent``: for two levers that both have repeats, the mean cross-lever correlation over disjoint pairs divided
  by sqrt(reliability_L · reliability_M). All three quantities come from the same bootstrap resample of entries,
  so the interval covers the correction too. It is marked unidentified unless both reliabilities are
  significantly positive; latent correlations of levers without repeats are not estimated.
- ``null``: a no-change difference between two accuracy-only runs, correlated with each arm. Nonzero values
  mean the accuracy-only runs are not exchangeable (e.g. V0 came from an earlier pipeline), which weakens
  every correction.
- ``consensus``: u(e), the mean token change over the arms sharing the first reference. It is related to the
  token and accuracy changes of arms measured against other references, by quintile and overall. This asks
  whether the accuracy cost falls where reasoning is cut most. u is also set against baseline length, taken
  from runs outside u.

Entries that overflow the serving window in any run are dropped, because their token counts are truncated.

    python evaluation/bfcl_shared_factor.py --root RUNS --output shared_factor.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.bfcl_efficiency import GROUPS, V3_CATEGORIES, load_run

J = "ae.joint."
R1, R2, R0 = f"{J}acc-legacy", f"{J}acc-legacy.s5678", "opd"
REFERENCES = (R1, R2, R0)
DEFAULT_ARMS = {
    **{
        f"{J}{name}": R1
        for name in (
            "acc-legacy+decs",
            "acc-legacy+decs7b",
            "acc-legacy+l1max",
            "acc-legacy+nemotron",
            "acc-legacy+donor-mean",
            "acc-legacy+decs@low",
            "acc-legacy+decs@high",
            "acc-legacy+heuristic",
            "acc-legacy-scaled",
            "acc-legacy+decs-protect-first",
            "acc-legacy.concise",
        )
    },
    f"{J}acc-legacy+decs.s5678": R2,
    f"{J}acc-legacy+decs-protect-first.s5678": R2,
    "opd-concise": R0,
    "opd-cap4k": R0,
}
# Levers run twice with disjoint runs: the only source of reliability.
DEFAULT_LEVERS = {
    "decs": [f"{J}acc-legacy+decs", f"{J}acc-legacy+decs.s5678"],
    "decs_protect_first": [f"{J}acc-legacy+decs-protect-first", f"{J}acc-legacy+decs-protect-first.s5678"],
    "concise_prompt": [f"{J}acc-legacy.concise", "opd-concise"],
}
QUINTILES = 5


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def correlation(x: np.ndarray, y: np.ndarray, *, draws: int, rng) -> dict:
    """Pearson r with an entry-bootstrap 95% interval (None when either side is constant)."""
    point = pearson(x, y)
    if np.isnan(point):
        return {"r": None, "ci95": [None, None]}
    boot = []
    for _ in range(draws):
        index = rng.integers(0, len(x), len(x))
        boot.append(pearson(x[index], y[index]))
    return {"r": point, "ci95": np.nanpercentile(boot, [2.5, 97.5]).tolist()}


def disjoint(first: tuple[str, str], second: tuple[str, str]) -> bool:
    """Two (arm, reference) differences share no run."""
    return not set(first) & set(second)


def latent_correlations(change: dict, arms: dict, levers: dict, *, draws: int, rng) -> dict:
    """Repeat-based reliabilities and disattenuated cross-lever correlations, bootstrapped jointly."""
    usable = {}
    for name, tags in levers.items():
        repeats = [(a, b) for a, b in itertools.combinations(tags, 2) if disjoint((a, arms[a]), (b, arms[b]))]
        if repeats:
            usable[name] = repeats
    crosses = {}
    for first, second in itertools.combinations(sorted(usable), 2):
        combos = [(a, b) for a in levers[first] for b in levers[second] if disjoint((a, arms[a]), (b, arms[b]))]
        if combos:
            crosses[(first, second)] = combos
    size = len(next(iter(change.values())))

    def mean_r(combos, index) -> float:
        return float(np.nanmean([pearson(change[a][index], change[b][index]) for a, b in combos]))

    def estimate(index: np.ndarray) -> tuple[dict, dict]:
        reliability = {name: mean_r(repeats, index) for name, repeats in usable.items()}
        latent = {}
        for key, combos in crosses.items():
            scale = reliability[key[0]] * reliability[key[1]]
            latent[key] = mean_r(combos, index) / np.sqrt(scale) if scale > 0 else float("nan")
        return reliability, latent

    point_rel, point_latent = estimate(np.arange(size))
    boot = [estimate(rng.integers(0, size, size)) for _ in range(draws)]
    report = {"reliability": {}, "latent": {}}
    for name, value in point_rel.items():
        interval = np.nanpercentile([rel[name] for rel, _ in boot], [2.5, 97.5]).tolist()
        report["reliability"][name] = {"r": value, "ci95": interval, "repeats": usable[name]}
    for key, value in point_latent.items():
        identified = all(report["reliability"][name]["ci95"][0] > 0 for name in key)
        interval = np.nanpercentile([lat[key] for _, lat in boot], [2.5, 97.5]).tolist() if identified else None
        report["latent"][" ~ ".join(key)] = {
            "latent_r": value,
            "ci95": interval or [None, None],
            "identified": identified,
            "cross_pairs": crosses[key],
        }
    return report


def analyze(
    log_tokens: dict[str, np.ndarray],
    correct: dict[str, np.ndarray],
    groups: np.ndarray,
    arms: dict[str, str],
    levers: dict[str, list[str]],
    *,
    draws: int = 1_000,
    seed: int = 0,
) -> dict:
    """The overlap report over aligned per-entry arrays (see the module docstring)."""
    rng = np.random.default_rng(seed)
    unknown = {tag for tags in levers.values() for tag in tags} - set(arms)
    if unknown:
        raise ValueError(f"levers name arms without references: {sorted(unknown)}")
    change = {arm: log_tokens[arm] - log_tokens[ref] for arm, ref in arms.items()}
    accuracy = {arm: correct[arm] - correct[ref] for arm, ref in arms.items()}
    subsets = {"all": np.ones(len(groups), dtype=bool), **{g: groups == g for g in sorted(set(groups))}}
    report: dict = {"entries": len(groups), "raw": {}, "null": {}, "by_subset": {}, "consensus": {}}

    for name, keep in subsets.items():
        values = {arm: delta[keep] for arm, delta in change.items()}
        report["by_subset"][name] = latent_correlations(values, arms, levers, draws=draws, rng=rng)

    names = sorted(arms)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if disjoint((a, arms[a]), (b, arms[b])):
                report["raw"][f"{a} ~ {b}"] = {
                    name: correlation(change[a][keep], change[b][keep], draws=draws, rng=rng)
                    for name, keep in subsets.items()
                }

    for arm, ref in arms.items():
        others = [r for r in REFERENCES if r not in (arm, ref) and r in log_tokens]
        if len(others) >= 2:
            null = log_tokens[others[0]] - log_tokens[others[1]]
            report["null"][arm] = correlation(change[arm], null, draws=draws, rng=rng)

    consensus_arms = [arm for arm, ref in arms.items() if ref == R1]
    held_out = [arm for arm, ref in arms.items() if ref != R1]
    if consensus_arms and held_out:
        u = np.mean([change[arm] for arm in consensus_arms], axis=0)
        outside = [r for r in (R2, R0) if r in log_tokens]
        baseline = np.mean([log_tokens[r] for r in outside], axis=0)
        edges = np.quantile(u, np.linspace(0, 1, QUINTILES + 1))
        bins = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, QUINTILES - 1)  # 0 = most cut
        consensus = {
            "arms": consensus_arms,
            "u_vs_baseline_length": correlation(u, baseline, draws=draws, rng=rng),
            "quintiles": [],
            "predicts": {},
        }
        for q in range(QUINTILES):
            keep = bins == q
            row = {
                "entries": int(keep.sum()),
                "mean_u": float(u[keep].mean()),
                "mean_baseline_log_tokens": float(baseline[keep].mean()),
            }
            row.update({f"delta_tokens:{arm}": float(change[arm][keep].mean()) for arm in held_out})
            row.update({f"delta_correct:{arm}": float(accuracy[arm][keep].mean()) for arm in held_out})
            consensus["quintiles"].append(row)
        for arm in held_out:
            consensus["predicts"][arm] = {
                target: {
                    name: correlation(u[keep], values[keep], draws=draws, rng=rng) for name, keep in subsets.items()
                }
                for target, values in (("tokens", change[arm]), ("correct", accuracy[arm]))
            }
        report["consensus"] = consensus
    return report


def aligned_arrays(runs: dict[str, dict]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, int]:
    """Per-entry log-token and correctness arrays in one entry order, without entries that overflow anywhere."""
    group_of = {category: group for group, members in GROUPS.items() for category in members}
    keys, dropped = [], 0
    for category in V3_CATEGORIES:
        by_run = [{entry["id"]: entry for entry in run[category]} for run in runs.values()]
        if any(items.keys() != by_run[0].keys() for items in by_run):
            raise ValueError(f"{category}: runs do not cover the same entries")
        for entry_id in sorted(by_run[0]):
            if any(items[entry_id]["out_of_context"] for items in by_run):
                dropped += 1
                continue
            keys.append((category, entry_id))
    lookup = {tag: {(c, e["id"]): e for c in V3_CATEGORIES for e in run[c]} for tag, run in runs.items()}
    log_tokens = {tag: np.log1p([lookup[tag][key]["gen_tokens"] for key in keys]) for tag in runs}
    correct = {tag: np.asarray([float(lookup[tag][key]["is_correct"]) for key in keys]) for tag in runs}
    groups = np.asarray([group_of[category] for category, _ in keys])
    return log_tokens, correct, groups, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="directory holding one downloaded run per tag")
    parser.add_argument("--arms", type=json.loads, default=DEFAULT_ARMS, help="JSON {arm tag: reference tag}")
    parser.add_argument("--levers", type=json.loads, default=DEFAULT_LEVERS, help="JSON {lever: [repeat arm tags]}")
    parser.add_argument("--draws", type=int, default=1_000)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tags = sorted(set(args.arms) | set(args.arms.values()) | set(REFERENCES))
    runs = {tag: load_run(args.root / tag, require_usage=False) for tag in tags}
    log_tokens, correct, groups, dropped = aligned_arrays(runs)
    report = analyze(log_tokens, correct, groups, args.arms, args.levers, draws=args.draws)
    report["dropped_overflow_entries"] = dropped
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fmt(value, interval) -> str:
        if value is None or np.isnan(value):
            return "undefined"
        bounds = "unidentified" if interval[0] is None else f"[{interval[0]:+.3f}, {interval[1]:+.3f}]"
        return f"{value:+.3f} {bounds}"

    print(f"entries {report['entries']} (dropped {dropped} overflowing)")
    for subset, item in report["by_subset"].items():
        for lever, rel in item["reliability"].items():
            print(f"[{subset}] reliability {lever}: {fmt(rel['r'], rel['ci95'])}")
        for pair, lat in item["latent"].items():
            print(f"[{subset}] latent {pair}: {fmt(lat['latent_r'], lat['ci95'])}")
    nulls = [item for item in report["null"].values() if item["r"] is not None]
    excluding = sum(1 for item in nulls if item["ci95"][0] > 0 or item["ci95"][1] < 0)
    print(f"null checks: {excluding} of {len(nulls)} nominal intervals exclude 0 (nonzero: runs not exchangeable)")
    for arm, item in report["consensus"].get("predicts", {}).items():
        tokens, accuracy = item["tokens"]["all"], item["correct"]["all"]
        print(
            f"consensus -> {arm}: tokens {fmt(tokens['r'], tokens['ci95'])}, "
            f"accuracy {fmt(accuracy['r'], accuracy['ci95'])}"
        )


if __name__ == "__main__":
    main()
