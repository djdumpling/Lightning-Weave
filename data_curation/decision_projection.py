#!/usr/bin/env python3
"""Decision-preserving projection of a sequence tilt over a recipient's own samples (CPU only).

For a prompt h, the recipient p0 gave N sampled responses. Each response is private reasoning z followed by a decision
a: its complete visible output, exactly as the BFCL server shows it (the tool calls with their arguments, in order, and
any visible text), together with how the response ended (its finish reason). A donor score s(z) rates the reasoning.
Three targets over the same N samples differ only in their weights:

    uniform     w_i = 1
    ordinary    w_i = N   · exp(s_i / α) / Σ_{j in prompt}  exp(s_j / α)
    projected   w_i = n_g · exp(s_i / α) / Σ_{j in group g} exp(s_j / α)

where g is the group of responses with the same decision as i and n_g its size. The ordinary tilt also moves weight
between decisions, toward decisions whose reasoning scores well; the projected tilt keeps every group's total weight
at n_g, so the recipient's empirical decision frequencies are unchanged and only the reasoning within each decision is
reweighted. An optional anchor w_η = (1 − η) · w + η keeps that property for every η.

Training fits each target with weighted, summed log-likelihood over complete responses (``sequence_weighted`` in
``slime/rollout/offline_direct_opd.py``); the weight of a sample is stored as ``metadata.sequence_weight``.

    python data_curation/decision_projection.py diagnose --rollouts DIR [--scores SCORES] [--target-savings 0.15]
    python data_curation/decision_projection.py weights --rollouts DIR --scores SCORES --alpha A --output W.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ARMS = ("uniform", "ordinary", "projected")
# α values scanned by ``calibrate_alpha``, from the weakest tilt to the strongest (20 per decade).
ALPHA_GRID = tuple(float(value) for value in np.geomspace(1e4, 1e-3, 141))


@dataclass(frozen=True)
class ParsedResponse:
    reasoning: str | None  # what the server returns as reasoning; None unless both thinking tags are present
    visible: str | None  # what the server returns as content: everything it shows (None if empty)
    finish_reason: str | None
    decision: str

    @property
    def finished(self) -> bool:
        return self.reasoning is not None and self.finish_reason != "length"


def reasoning_opened(prompt: str) -> bool:
    """Whether the prompt itself opens the thinking block (e.g. Qwen3-2507 templates end with ``<think>``)."""
    return prompt.rstrip().endswith(THINK_OPEN)


def served_split(text: str) -> tuple[str | None, str | None]:
    """(reasoning, content) exactly as BFCL's server splits a response: vLLM 0.11.0's ``qwen3`` reasoning parser.

    Unless both tags are present, everything is content: an unclosed reasoning block is shown to the harness (and
    kept in the history) as visible text. Empty content is None.
    """
    if THINK_OPEN not in text or THINK_CLOSE not in text:
        return None, text
    rest = text.partition(THINK_OPEN)[2]
    if THINK_CLOSE not in rest:
        return None, rest
    reasoning, _, content = rest.partition(THINK_CLOSE)
    return reasoning, content or None


def decision_key(visible: str | None, finish_reason: str | None) -> str:
    """The decision: the exact content the server shows and how the response ended.

    Nothing is normalized, so two responses share a decision only if what they show is byte-identical.
    """
    return json.dumps({"visible": visible, "finish_reason": finish_reason}, ensure_ascii=False, separators=(",", ":"))


def parse_response(text: str, finish_reason: str | None = None) -> ParsedResponse:
    reasoning, visible = served_split(text)
    return ParsedResponse(reasoning, visible, finish_reason, decision_key(visible, finish_reason))


def _softmax_scaled(scores: np.ndarray, alpha: float) -> np.ndarray:
    logits = scores / alpha
    logits = logits - logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def arm_weights(scores, decisions, *, arm: str, alpha: float | None = None, eta: float = 0.0) -> np.ndarray:
    """Weights for one prompt's samples. They sum to N; projected weights sum to n_g within each decision group."""
    decisions = list(decisions)
    count = len(decisions)
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta}")
    if arm == "uniform":
        return np.ones(count)
    if alpha is None or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"alpha must be positive and finite, got {alpha}")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (count,) or not np.isfinite(scores).all():
        raise ValueError("scores must be finite, one per sample")
    if arm == "ordinary":
        weights = count * _softmax_scaled(scores, alpha)
    else:
        weights = np.empty(count)
        groups = defaultdict(list)
        for index, decision in enumerate(decisions):
            groups[decision].append(index)
        for members in groups.values():
            weights[members] = len(members) * _softmax_scaled(scores[members], alpha)
    return (1.0 - eta) * weights + eta


@dataclass(frozen=True)
class Sample:
    prompt_id: str
    sample_id: str
    decision: str
    tokens: int  # response tokens under the recipient's tokenizer (reasoning, decision and the stop token)
    score: float | None = None
    finished: bool = True  # the reasoning closed and the response did not hit the length limit


def by_prompt(samples: list[Sample]) -> dict[str, list[Sample]]:
    prompts: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        prompts[sample.prompt_id].append(sample)
    return dict(prompts)


def dataset_weights(samples: list[Sample], *, arm: str, alpha: float | None, eta: float = 0.0) -> dict[str, float]:
    result = {}
    for group in by_prompt(samples).values():
        scores = [sample.score for sample in group] if arm != "uniform" else None
        weights = arm_weights(scores, [sample.decision for sample in group], arm=arm, alpha=alpha, eta=eta)
        result.update({sample.sample_id: float(weight) for sample, weight in zip(group, weights, strict=True)})
    return result


def implied_savings(samples: list[Sample], weights: dict[str, float]) -> float:
    """1 − (target-weighted tokens / uniform tokens): the cut in response tokens the target asks for."""
    total = sum(sample.tokens for sample in samples)
    weighted = sum(weights[sample.sample_id] * sample.tokens for sample in samples)
    return 1.0 - weighted / total


def calibrate_alpha(
    samples: list[Sample], target: float, *, arm: str = "projected", eta: float = 0.0,
    grid: tuple[float, ...] = ALPHA_GRID, iterations: int = 60,
) -> dict:
    """The weakest tilt (largest α) at which ``arm``'s implied savings reach ``target``.

    Savings need not grow monotonically as α shrinks: donor scores need not rank responses by length, so the strongest
    tilt can save less than a moderate one. The fixed grid is scanned from the weakest tilt to the strongest; at the
    first grid point that reaches the target, the bracket with the point before it is bisected in log α. A target that
    no grid point reaches raises, reporting the best one.
    """

    def savings(alpha: float) -> float:
        return implied_savings(samples, dataset_weights(samples, arm=arm, alpha=alpha, eta=eta))

    if list(grid) != sorted(grid, reverse=True):
        raise ValueError("the alpha grid must run from the weakest tilt (largest alpha) to the strongest")
    scanned = []
    for index, alpha in enumerate(grid):
        scanned.append((alpha, savings(alpha)))
        if scanned[-1][1] < target:
            continue
        if index == 0:
            raise ValueError(f"the weakest tilt on the grid (alpha={alpha}) already saves {scanned[-1][1]:.4f}")
        log_reach, log_short = math.log(alpha), math.log(grid[index - 1])
        for _ in range(iterations):
            middle = 0.5 * (log_reach + log_short)
            if savings(math.exp(middle)) >= target:
                log_reach = middle
            else:
                log_short = middle
        found = math.exp(log_reach)
        return {"arm": arm, "target": target, "alpha": found, "implied_savings": savings(found), "eta": eta}
    best_alpha, best = max(scanned, key=lambda item: item[1])
    raise ValueError(f"{arm} reaches at most {best:.4f} implied savings (alpha={best_alpha:.4g}); target {target}")


def _ess(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.square(weights).sum())


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 0, with tied values sharing their mean rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return None
    return float(np.corrcoef(_average_ranks(first), _average_ranks(second))[0, 1])


def structure_report(samples: list[Sample]) -> dict:
    """Decision structure, needing no scores: how much within-decision variation there is to reweight."""
    prompts = by_prompt(samples)
    largest, group_counts, in_shared, shortest_saved = [], [], 0, 0
    for group in prompts.values():
        sizes = Counter(sample.decision for sample in group)
        largest.append(max(sizes.values()))
        group_counts.append(len(sizes))
        for decision, size in sizes.items():
            if size < 2:
                continue
            in_shared += size
            lengths = [sample.tokens for sample in group if sample.decision == decision]
            shortest_saved += sum(lengths) - size * min(lengths)
    total_tokens = sum(sample.tokens for sample in samples)
    return {
        "prompts": len(prompts),
        "samples": len(samples),
        "responses_per_prompt": dict(sorted(Counter(len(group) for group in prompts.values()).items())),
        "unfinished_fraction": sum(not sample.finished for sample in samples) / len(samples),
        "largest_group_size": dict(sorted(Counter(largest).items())),
        "decisions_per_prompt": dict(sorted(Counter(group_counts).items())),
        "samples_in_shared_decision_fraction": in_shared / len(samples),
        # The most a projected target can cut: every decision group's weight on its shortest member.
        "shortest_within_decision_savings": shortest_saved / total_tokens,
        "shortest_within_prompt_savings": sum(
            sum(s.tokens for s in group) - len(group) * min(s.tokens for s in group) for group in prompts.values()
        ) / total_tokens,
    }


def weight_report(samples: list[Sample], *, alpha: float, eta: float = 0.0) -> dict:
    """Target diagnostics at one α: implied savings, weight concentration, decision movement, score–length relation."""
    prompts = by_prompt(samples)
    report: dict = {"alpha": alpha, "eta": eta, "arms": {}}
    decision_tv, mass_error, correlations = [], 0.0, []
    for arm in ARMS:
        weights = dataset_weights(samples, arm=arm, alpha=alpha, eta=eta)
        prompt_ess, group_ess = [], []
        for group in prompts.values():
            values = np.array([weights[sample.sample_id] for sample in group])
            prompt_ess.append(_ess(values) / len(group))
            for decision in {sample.decision for sample in group}:
                members = np.array([weights[s.sample_id] for s in group if s.decision == decision])
                if len(members) >= 2:
                    group_ess.append(_ess(members) / len(members))
            if arm == "projected":
                for decision in {sample.decision for sample in group}:
                    members = [weights[s.sample_id] for s in group if s.decision == decision]
                    mass_error = max(mass_error, abs(sum(members) - len(members)))
        report["arms"][arm] = {
            "implied_savings": implied_savings(samples, weights),
            "prompt_ess_fraction": {"median": float(np.median(prompt_ess)), "p10": float(np.percentile(prompt_ess, 10))},
            "group_ess_fraction": (
                {"median": float(np.median(group_ess)), "p10": float(np.percentile(group_ess, 10))} if group_ess else None
            ),
            "max_weight": max(weights.values()),
        }
    ordinary = dataset_weights(samples, arm="ordinary", alpha=alpha, eta=eta)
    for group in prompts.values():
        moved = defaultdict(float)
        for sample in group:
            moved[sample.decision] += (ordinary[sample.sample_id] - 1.0) / len(group)
        decision_tv.append(0.5 * sum(abs(value) for value in moved.values()))
        for decision in {sample.decision for sample in group}:
            members = [s for s in group if s.decision == decision]
            rho = _spearman(np.array([s.score for s in members]), np.array([s.tokens for s in members], dtype=float))
            if rho is not None:
                correlations.append(rho)
    report["projected_group_mass_max_error"] = mass_error
    # How far the ordinary target moves decision frequencies (total variation per prompt); the projection's is 0.
    report["ordinary_decision_tv"] = {"mean": float(np.mean(decision_tv)), "p90": float(np.percentile(decision_tv, 90))}
    report["within_decision_score_length_spearman"] = (
        {"median": float(np.median(correlations)), "groups": len(correlations)} if correlations else None
    )
    return report


def savings_curve(samples: list[Sample], alphas, *, eta: float = 0.0) -> list[dict]:
    rows = []
    for alpha in alphas:
        row = {"alpha": float(alpha)}
        for arm in ("ordinary", "projected"):
            row[arm] = implied_savings(samples, dataset_weights(samples, arm=arm, alpha=alpha, eta=eta))
        rows.append(row)
    return rows


def savings_matched(pooled: dict, first: str, second: str, tolerance: float) -> dict:
    """Pre-registered check: do two arms' realized total-token savings against the recipient agree within tolerance?

    ``pooled`` is a ``evaluation/bfcl_pooled.py`` report with comparisons named "<arm> vs recipient".
    """
    savings = {arm: -pooled[f"{arm} vs recipient"]["pooled"]["total_tokens"]["delta"] for arm in (first, second)}
    difference = savings[first] - savings[second]
    return {"savings": savings, "difference": difference, "tolerance": tolerance, "matched": abs(difference) <= tolerance}


# --- I/O ---------------------------------------------------------------------------------------------------------


def load_rollouts(paths: list[Path]) -> list[Sample]:
    """Samples from offline Direct-OPD rollout shards (the decision is parsed from each response)."""
    import pyarrow.parquet as pq

    files = sorted(file for path in paths for file in ([path] if path.is_file() else path.glob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet shards under {paths}")
    columns = [f"metadata.{name}" for name in ("prompt_id", "sample_id", "response", "response_length", "finish_reason")]
    samples = []
    for file in files:
        table = pq.read_table(file, columns=columns)
        rows = zip(*(table.column(index).to_pylist() for index in range(len(columns))), strict=True)
        for prompt_id, sample_id, response, length, finish_reason in rows:
            parsed = parse_response(response, finish_reason)
            samples.append(
                Sample(str(prompt_id), str(sample_id), parsed.decision, int(length), finished=parsed.finished)
            )
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("duplicate sample_id in the rollout shards")
    return samples


def attach_scores(samples: list[Sample], scores_path: Path) -> list[Sample]:
    """Join ``data_curation/score_reasoning.py`` output (one file, or a directory of rank files) by sample_id.

    Every sample needs exactly one score.
    """
    import pyarrow.parquet as pq

    files = sorted(scores_path.glob("*.parquet")) if scores_path.is_dir() else [scores_path]
    scores: dict[str, float] = {}
    for file in files:
        table = pq.read_table(file, columns=["sample_id", "score"])
        for sample_id, score in zip(map(str, table.column(0).to_pylist()), table.column(1).to_pylist(), strict=True):
            if sample_id in scores:
                raise ValueError(f"sample {sample_id} is scored twice in {scores_path}")
            scores[sample_id] = float(score)
    missing = [sample.sample_id for sample in samples if sample.sample_id not in scores]
    if missing:
        raise ValueError(f"{len(missing)} samples have no score, e.g. {missing[:3]}")
    return [replace(sample, score=scores[sample.sample_id]) for sample in samples]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("diagnose", "weights"):
        command = commands.add_parser(name)
        command.add_argument("--rollouts", type=Path, nargs="+", required=True)
        command.add_argument("--scores", type=Path, required=name == "weights")
        command.add_argument("--alpha", type=float)
        command.add_argument("--eta", type=float, default=0.0)
        command.add_argument("--output", type=Path, required=name == "weights")
    commands.choices["diagnose"].add_argument("--target-savings", type=float)
    commands.choices["diagnose"].add_argument("--calibrate-arm", choices=("ordinary", "projected"), default="projected")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    samples = load_rollouts(args.rollouts)
    if args.scores is not None:
        samples = attach_scores(samples, args.scores)
    if args.command == "diagnose":
        report = {"structure": structure_report(samples)}
        failure = None
        if args.scores is not None:
            alpha = args.alpha
            if args.target_savings is not None:
                try:
                    report["calibration"] = calibrate_alpha(samples, args.target_savings, arm=args.calibrate_arm,
                                                            eta=args.eta)
                    alpha = report["calibration"]["alpha"]
                except ValueError as error:  # the pre-registered stop: record the diagnostics, then fail
                    failure = report["calibration"] = {"target": args.target_savings, "error": str(error)}
                    alpha = None
            if alpha is not None:
                report["weights"] = weight_report(samples, alpha=alpha, eta=args.eta)
                report["savings_curve"] = savings_curve(samples, alpha * np.array([0.25, 0.5, 1.0, 2.0, 4.0]),
                                                        eta=args.eta)
            else:
                report["savings_curve"] = savings_curve(samples, ALPHA_GRID[::10], eta=args.eta)
        text = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        if failure is not None:
            sys.exit(f"calibration failed: {failure['error']}")
        return
    if args.alpha is None:
        sys.exit("weights requires --alpha")
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = {"sample_id": [s.sample_id for s in samples], "prompt_id": [s.prompt_id for s in samples],
               "decision": [s.decision for s in samples]}
    for arm in ARMS:
        weights = dataset_weights(samples, arm=arm, alpha=args.alpha, eta=args.eta)
        columns[f"weight_{arm}"] = [weights[s.sample_id] for s in samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(columns).replace_schema_metadata(
        {"alpha": repr(args.alpha), "eta": repr(args.eta), "schema": "decision_projection_weights_v1"}
    )
    pq.write_table(table, args.output)
    print(f"wrote {len(samples)} weights per arm to {args.output}")


if __name__ == "__main__":
    main()
