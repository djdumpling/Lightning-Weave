"""Does a training signal favor the student's own responses that succeed? (CPU probe over executed branches.)

For each cached response y at a fresh-state prompt s, a signal's tilted target q ∝ b·exp(δ/α) (the same per-token
target Offline Direct-OPD trains toward) assigns it the sequence log-ratio

    log q(y | s) − log b(y | s) = Σ_t [ δ_t(y_t)/α − log Σ_k b_tk exp(δ_tk/α) ],

on the K+1 buckets (the sampled token's bucket, or the remaining-vocabulary bucket). A signal is a delta source
(raw post − pre, as the ``raw`` transform), a teacher (the post model's own distribution, as the ``teacher``
transform), or a control: the student's own log-probability of y, or its length. Within each state whose responses
ended differently (data_curation/branch_outcomes.py), the AUC is the probability that a successful response scores
above a failed one (ties count half); 0.5 is no information. Intervals bootstrap states.

    python data_curation/branch_signal_probe.py --base CACHE/anchor/final --donor agentforge=CACHE/donors/agentforge/pre \\
        --prompts CACHE/prompts.parquet --branches RUN-s0/thinking2507 ... --domain tau2_retail --output probe.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import random
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.branch_outcomes import branch_outcome
from data_curation.build_synthetic_shift_target import teacher_vector
from data_curation.shift_geometry import CachedRow, bucket_probs, bucket_shift, iter_joined_rows, relevel


def sampled_buckets(row: CachedRow) -> np.ndarray:
    """Each position's bucket for the sampled token: its candidate index, or K for the remaining vocabulary."""
    candidates = np.asarray(row.candidate_ids)
    hit = candidates == np.asarray(row.response_tokens)[:, None]
    return np.where(hit.any(axis=-1), hit.argmax(axis=-1), candidates.shape[-1])


# Qwen3's end-of-reasoning token: positions after it are the turn's action (text and tool calls).
THINK_END_ID = 151668
SPANS = ("all", "reasoning", "action")


def span_masks(row: CachedRow) -> dict[str, np.ndarray]:
    """Loss positions of the whole response, of the reasoning up to and including </think>, and of the action after it."""
    tokens = np.asarray(row.response_tokens)
    ends = np.nonzero(tokens == THINK_END_ID)[0]
    after = np.zeros(len(tokens), dtype=bool)
    if ends.size:
        after[ends[0] + 1 :] = True
    mask = np.asarray(row.loss_mask, dtype=bool)
    return {"all": mask, "reasoning": mask & ~after, "action": mask & after}


def position_log_ratios(row: CachedRow, delta: np.ndarray, alpha: float) -> np.ndarray:
    """Per position, log q(y_t) − log b(y_t) for the tilted target of a [T, K+1] releveled delta."""
    probs = bucket_probs(row.behavior_log_probs)
    scaled = delta / float(alpha)
    log_normalizer = np.log((probs * np.exp(scaled - scaled.max(axis=-1, keepdims=True))).sum(axis=-1)) + scaled.max(axis=-1)
    return scaled[np.arange(len(scaled)), sampled_buckets(row)] - log_normalizer


def sequence_log_ratio(row: CachedRow, delta: np.ndarray, alpha: float) -> float:
    """log q(y) − log b(y) for the tilted target of a [T, K+1] releveled delta, over the loss positions."""
    return float(position_log_ratios(row, delta, alpha)[row.loss_mask].sum())


def signals(row: CachedRow, deltas: list[str], teachers: list[str], alpha: float) -> dict[str, float]:
    """Each signal's log-ratio summed over each span (``name@span``); the student's log-probability; the length."""
    masks = span_masks(row)
    per_position = {f"delta:{name}": position_log_ratios(row, relevel(bucket_shift(row.shifts[name])), alpha) for name in deltas}
    per_position |= {f"teacher:{name}": position_log_ratios(row, relevel(teacher_vector(row, name, alpha)), alpha) for name in teachers}
    probs = bucket_probs(row.behavior_log_probs)
    per_position["student_logprob"] = np.log(probs[np.arange(len(probs)), sampled_buckets(row)])
    out = {}
    for name, values in per_position.items():
        for span, mask in masks.items():
            out[f"{name}@{span}"] = float(values[mask].sum())
    out["shorter"] = -float(masks["all"].sum())
    out["shorter_action"] = -float(masks["action"].sum())
    return out


def pair_counts(scores: list[float], outcomes: list[float]) -> tuple[float, int]:
    """(wins + ties/2, pairs) over every (success, failure) pair of one state."""
    wins, pairs = 0.0, 0
    for i, oi in enumerate(outcomes):
        for j, oj in enumerate(outcomes):
            if oi > oj:
                pairs += 1
                wins += 1.0 if scores[i] > scores[j] else 0.5 if scores[i] == scores[j] else 0.0
    return wins, pairs


def auc_table(states: dict[str, list[tuple[dict[str, float], float]]], names: list[str], draws: int, seed: int) -> dict:
    """Pooled within-state AUC per signal, with a state-bootstrap 95% interval."""
    keys = sorted(states)
    per_state = {
        name: [pair_counts([s[name] for s, _ in states[key]], [o for _, o in states[key]]) for key in keys]
        for name in names
    }
    rng = random.Random(seed)
    samples = [[rng.randrange(len(keys)) for _ in keys] for _ in range(draws)]
    table = {}
    for name, counts in per_state.items():
        wins, pairs = sum(w for w, _ in counts), sum(p for _, p in counts)
        boots = sorted(sum(counts[i][0] for i in sample) / max(sum(counts[i][1] for i in sample), 1) for sample in samples)
        table[name] = {"auc": wins / pairs if pairs else math.nan, "lo": boots[int(0.025 * draws)],
                       "hi": boots[int(0.975 * draws) - 1], "pairs": pairs}
    return table


def read_outcomes(roots: list[Path], samples: dict[str, str]) -> dict[str, float]:
    outcomes = {}
    for root in roots:
        for path in glob.glob(str(root / "tau2_*" / "*.json")):
            record = json.loads(Path(path).read_text(encoding="utf-8"))
            outcome = branch_outcome(record)
            if outcome is not None and record["task_id"] in samples:
                outcomes[samples[record["task_id"]]] = outcome
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--donor", action="append", default=[], help="NAME=DIR of a scored donor chain")
    parser.add_argument("--teacher", action="append", default=[], help="a scored source whose post model is a teacher")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--branches", type=Path, nargs="+", required=True)
    parser.add_argument("--retest", type=Path, nargs="*", default=[], help="a second continuation of some branches")
    parser.add_argument("--domain", default="tau2_retail")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    prompts = {row["prompt_id"]: row for row in pq.read_table(args.prompts).to_pylist()}
    donors = {name: Path(path) for name, path in (item.split("=", 1) for item in args.donor)}
    deltas = ["agent_acc", *donors]
    rows = {}
    for row in iter_joined_rows(args.base, donors):
        if prompts[row.prompt_id]["domain"] == args.domain:
            rows[row.sample_id] = (row.prompt_id, signals(row, deltas, args.teacher, args.alpha))
    samples = {sample_id[:24]: sample_id for sample_id in rows}
    outcomes = read_outcomes(args.branches, samples)
    retest = read_outcomes(args.retest, samples) if args.retest else {}
    names = sorted(next(iter(rows.values()))[1])

    def states(outcome_of) -> dict:
        grouped = collections.defaultdict(list)
        for sample_id, (prompt_id, scores) in rows.items():
            if outcome_of(sample_id) is not None:
                grouped[prompt_id].append((scores, outcome_of(sample_id)))
        return {key: items for key, items in grouped.items() if len({o for _, o in items}) > 1}

    informative = states(outcomes.get)
    report = {
        "domain": args.domain,
        "responses": len(rows),
        "responses_with_outcome": sum(sample in outcomes for sample in rows),
        "states": len({prompt for prompt, _ in rows.values()}),
        "informative_states": len(informative),
        "mean_outcome": sum(outcomes.get(sample, 0.0) for sample in rows) / max(len(rows), 1),
        "auc": auc_table(informative, names, args.draws, 0),
        "auc_by_decision_state": {
            str(flag): auc_table({k: v for k, v in informative.items() if bool(prompts[k]["decision_state"]) == flag},
                                 names, args.draws, 0)
            for flag in (True, False)
        },
    }
    if retest:
        paired = [sample for sample in retest if sample in outcomes]
        agree = sum(outcomes[s] == retest[s] for s in paired)
        both = {s: (outcomes[s] + retest[s]) / 2 for s in paired}
        retested_states = {rows[s][0] for s in paired}
        first = {k for k in states(outcomes.get) if k in retested_states}
        second = {k for k in states(retest.get) if k in retested_states}
        report["retest"] = {
            "branches": len(paired),
            "agreement": agree / max(len(paired), 1),
            "first_success_rate": sum(outcomes[s] for s in paired) / max(len(paired), 1),
            "second_success_rate": sum(retest[s] for s in paired) / max(len(paired), 1),
            "states": len(retested_states),
            "informative_first": len(first),
            "informative_second": len(second),
            "informative_both": len(first & second),
            "auc_second_continuation": auc_table(states(retest.get), names, args.draws, 1),
            "auc_two_continuation_mean": auc_table(states(both.get), names, args.draws, 2),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name in names:
        item = report["auc"][name]
        print(f"{name:32s} AUC {item['auc']:.3f} [{item['lo']:.3f}, {item['hi']:.3f}]  pairs {item['pairs']}")


if __name__ == "__main__":
    main()
