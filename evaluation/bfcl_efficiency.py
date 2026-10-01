#!/usr/bin/env python3
"""Agent cost vectors, leaderboard-weighted aggregates, paired comparisons, and a decision rule for BFCL v3.

A run directory is one ``<run_id>/<tag>`` tree from the BFCL Modal eval: one
``bfcl_v3.<category>/output.jsonl`` per category (NeMo-Skills generations with
``is_correct`` merged in), and ``usage.jsonl`` from the serving proxy. Every
per-entry metric is aggregated with the vendored v3 leaderboard's category
weights, so ``accuracy`` here reproduces ``aggregate.json``. Distribution
summaries (median, p90, trimmed mean) use the same weights per entry.

Per-entry cost vector:

- ``gen_tokens``: ``num_generated_tokens``, completion tokens with thinking.
- ``steps`` (assistant generations), ``user_turns``, ``steps_per_turn``.
- ``tool_calls`` and ``duplicate_calls`` (an identical name+arguments call
  repeated within one user turn).
- ``max_step_tokens``, ``runaway`` (a step reached the per-step cap), and
  ``out_of_context`` (NeMo-Skills ``_ran_out_of_context_``).
- From the proxy log: ``prompt_tokens`` (summed over steps),
  ``reasoning_tokens`` (the re-tokenized ``reasoning_content``), and
  ``length_stops``. Requests are joined to entries by their user-message
  sequence (and, for single-turn entries, the tool schemas), then by exact
  per-step completion counts, which also separates retried attempts. Each entry
  must reconcile exactly: its logged completion tokens are its
  ``num_generated_tokens_list``.

Comparisons against a baseline run (same entries):

- the primary cost estimate: the paired log-ratio of generated tokens, i.e. the
  leaderboard-weighted mean over entries of log(1 + model) − log(1 + base),
  reported as a relative change. A few runaway entries dominate differences of
  mean tokens; the log-ratio measures the typical entry's change and is several
  times more precise;
- paired, category-stratified bootstrap intervals of every difference;
- cost on entries both models got right, and the outcome-transition strata;
- multi-turn cost over the common horizon (turns both runs completed);
- cost per success;
- a decision rule: accuracy non-inferiority, a real generated-token reduction,
  and non-inferiority on irrelevance accuracy, duplicate calls, runaways, and
  context overflows: each guard's upper interval bound must stay within its
  tolerated increase (``GUARD_TOLERANCES``). AES is reported, but only as a
  description.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.bfcl_eval.config import _digest, message_text

SIMPLE_AST = ("simple_python", "simple_java", "simple_javascript")
OTHER_NON_LIVE = ("parallel", "multiple", "parallel_multiple", "irrelevance")
LIVE = ("live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance")
MULTI_TURN = ("multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context")
V3_CATEGORIES = SIMPLE_AST + OTHER_NON_LIVE + LIVE + MULTI_TURN
GROUPS = {"non_live": SIMPLE_AST + OTHER_NON_LIVE, "live": LIVE, "multi_turn": MULTI_TURN}
IRRELEVANCE = ("irrelevance", "live_irrelevance")
OUT_OF_CONTEXT = "_ran_out_of_context_"
DEFAULT_STEP_CAP = 32_768
SUMMARY_METRICS = (
    "gen_tokens",
    "steps",
    "steps_per_turn",
    "tool_calls",
    "duplicate_calls",
    "max_step_tokens",
    "runaway",
    "out_of_context",
)
USAGE_METRICS = ("prompt_tokens", "reasoning_tokens", "length_stops")
COMPARED_METRICS = ("is_correct", "gen_tokens", "steps", "tool_calls", "duplicate_calls", "runaway", "out_of_context")
OUTCOMES = {(True, True): "both", (False, True): "model_only", (True, False): "base_only", (False, False): "neither"}


def group_weights(counts: dict[str, int]) -> dict[str, dict[str, float]]:
    """Weights w_c with group(x) = sum_c w_c * mean_c(x), matching the v3 leaderboard scorer.

    Non-live is the unweighted mean of {simple AST (itself the unweighted mean
    of three languages), parallel, multiple, parallel_multiple, irrelevance};
    live is entry-weighted over its six categories; multi-turn is the
    unweighted mean of its four; overall is the unweighted mean of the groups.
    """
    missing = [category for category in V3_CATEGORIES if not counts.get(category)]
    if missing:
        raise ValueError(f"missing or empty categories: {missing}")
    non_live = {category: 1 / 15 for category in SIMPLE_AST}
    non_live.update({category: 1 / 5 for category in OTHER_NON_LIVE})
    live_total = sum(counts[category] for category in LIVE)
    live = {category: counts[category] / live_total for category in LIVE}
    multi_turn = {category: 1 / 4 for category in MULTI_TURN}
    groups = {"non_live": non_live, "live": live, "multi_turn": multi_turn}
    groups["overall"] = {
        category: weight / 3 for weights in (non_live, live, multi_turn) for category, weight in weights.items()
    }
    return groups


# ---------------------------------------------------------------------------
# Per-entry costs
# ---------------------------------------------------------------------------


def _steps(generation) -> list[list]:
    """Per user turn, the list of steps; a step is a list of call dicts or a text string."""
    if isinstance(generation, list) and generation and all(isinstance(turn, list) for turn in generation):
        if all(isinstance(step, (list, str)) for turn in generation for step in turn):
            return generation
    return [[generation]]


def _calls(step) -> list[tuple[str, str]]:
    if isinstance(step, list):
        return [(name, _canonical_arguments(arguments)) for call in step for name, arguments in call.items()]
    return []


def _canonical_arguments(arguments) -> str:
    try:
        value = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        return str(arguments)
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def visible_text(step) -> str:
    """The step's content as the hermes tool-call format renders it (reasoning excluded)."""
    if isinstance(step, str):
        return step
    parts = []
    for call in step:
        for name, arguments in call.items():
            try:
                value = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                value = arguments
            parts.append(
                "<tool_call>\n" + json.dumps({"name": name, "arguments": value}, ensure_ascii=False) + "\n</tool_call>"
            )
    return "\n".join(parts)


def user_digests(question: list[list[dict]]) -> tuple[list[str | None], list[int]]:
    """Digests of an entry's user messages in order, and the turn each belongs to.

    An empty turn (multi_turn_miss_func adds the missing function there) is sent
    with a user message NeMo-Skills writes itself, so it becomes a wildcard (None).
    """
    digests, turns = [], []
    for index, turn in enumerate(question):
        users = [_digest(message_text(m)) for m in turn if m.get("role") == "user"] or [None]
        digests.extend(users)
        turns.extend([index] * len(users))
    return digests, turns


def entry_costs(row: dict, *, step_cap: int = DEFAULT_STEP_CAP, tokenizer=None) -> dict:
    turns = _steps(row["generation"])
    step_tokens = [int(t) for t in row.get("num_generated_tokens_list") or [row["num_generated_tokens"]]]
    steps = [step for turn in turns for step in turn]
    if row.get("error") != OUT_OF_CONTEXT and len(steps) != len(step_tokens):
        raise ValueError(f"{row['id']}: {len(steps)} steps but {len(step_tokens)} token counts")
    calls = duplicates = 0
    for turn in turns:
        seen = Counter(call for step in turn for call in _calls(step))
        calls += sum(seen.values())
        duplicates += sum(count - 1 for count in seen.values())
    users, user_turn_index = user_digests(row.get("question") or [])
    turn_tokens, cursor = [], 0
    for turn in turns:
        turn_tokens.append(step_tokens[cursor : cursor + len(turn)])
        cursor += len(turn)
    costs = {
        "id": row["id"],
        "is_correct": bool(row["is_correct"]),
        "gen_tokens": int(row["num_generated_tokens"]),
        "steps": len(step_tokens),
        "user_turns": len(turns),
        "steps_per_turn": len(step_tokens) / len(turns),
        "tool_calls": calls,
        "duplicate_calls": duplicates,
        "text_steps": sum(isinstance(step, str) for step in steps),
        "max_step_tokens": max(step_tokens),
        "runaway": any(tokens >= step_cap for tokens in step_tokens),
        "out_of_context": row.get("error") == OUT_OF_CONTEXT,
        "step_tokens": step_tokens,
        "turn_tokens": turn_tokens,
        "users": users,
        "user_turn_index": user_turn_index,
        "tools_digest": _digest(row.get("tools") or []),
    }
    if tokenizer is not None:
        visible = sum(len(tokenizer.encode(visible_text(step), add_special_tokens=False)) for step in steps)
        costs["visible_tokens"] = visible
        costs["reasoning_tokens_est"] = max(costs["gen_tokens"] - visible, 0)
    return costs


# ---------------------------------------------------------------------------
# Joining the serving proxy's request log
# ---------------------------------------------------------------------------


def _request_candidates(category: str, entries: list[dict]):
    """The entries a request can belong to: its users are a prefix of theirs (wildcards match anything)."""
    single_turn = category not in MULTI_TURN
    by_first: dict[str | None, list[int]] = {}
    for index, entry in enumerate(entries):
        by_first.setdefault(entry["users"][0] if entry["users"] else None, []).append(index)

    def matches(users: list[str], expected: list[str | None]) -> bool:
        return len(users) <= len(expected) and all(e is None or e == u for u, e in zip(users, expected))

    def candidates(record: dict) -> list[int]:
        pool = by_first.get(record["users"][0], []) + by_first.get(None, []) if record["users"] else []
        found = [i for i in pool if matches(record["users"], entries[i]["users"])]
        if single_turn:
            found = [i for i in found if entries[i]["tools_digest"] == record["tools_digest"]]
        return found

    return candidates


def join_usage(category: str, entries: list[dict], records: list[dict]) -> dict:
    """Assign one category's requests to entries by exact per-step token counts; attach usage to those that reconcile.

    A request goes to a candidate entry that still expects a step with its
    completion-token count in that user turn. This settles both entries that are
    identical up to a turn (they send identical requests) and retried requests:
    the attempt the harness kept is the one whose count the entry recorded, and
    any other attempt is surplus. A reconciled entry also keeps its requests, in
    time order, under ``requests``.
    """
    candidates = _request_candidates(category, entries)
    remaining = [[Counter(turn) for turn in entry["turn_tokens"]] for entry in entries]
    assigned: dict[int, list[dict]] = {i: [] for i in range(len(entries))}
    unmatched = surplus = 0
    for record in sorted(records, key=lambda r: r["time"]):
        found = candidates(record)
        if not found:
            unmatched += 1
            continue
        tokens = record["completion_tokens"]
        turn = {i: entries[i]["user_turn_index"][len(record["users"]) - 1] for i in found}
        owners = [i for i in found if turn[i] < len(remaining[i]) and remaining[i][turn[i]][tokens] > 0]
        if owners:
            assigned[owners[0]].append(record)
            remaining[owners[0]][turn[owners[0]]][tokens] -= 1
        else:
            surplus += 1
    reconciled = 0
    for i, entry in enumerate(entries):
        logged = Counter(r["completion_tokens"] for r in assigned[i] if r["completion_tokens"])
        if logged == Counter(t for t in entry["step_tokens"] if t):
            entry.update(
                {
                    "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in assigned[i]),
                    "reasoning_tokens": sum(r["reasoning_tokens"] or 0 for r in assigned[i]),
                    "length_stops": sum(r["finish_reason"] == "length" for r in assigned[i]),
                    "requests": assigned[i],
                }
            )
            reconciled += 1
    return {"entries": len(entries), "reconciled": reconciled, "unmatched_requests": unmatched, "surplus_requests": surplus}


def attach_usage(run: dict[str, list[dict]], usage_path: Path, *, require_complete: bool = True) -> dict:
    by_lane: dict[str, list[dict]] = {}
    with usage_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            by_lane.setdefault(record.get("lane", ""), []).append(record)
    coverage = {category: join_usage(category, entries, by_lane.get(category, [])) for category, entries in run.items()}
    incomplete = {category: item for category, item in coverage.items() if item["reconciled"] != item["entries"]}
    if incomplete and require_complete:
        raise ValueError(f"usage log does not reconcile with every entry: {incomplete}")
    return coverage


def load_run(
    path: Path, *, step_cap: int = DEFAULT_STEP_CAP, tokenizer=None,
    require_usage: bool = True, include_usage: bool = True,
) -> dict:
    """{category: [entry costs]}; set ``include_usage=False`` to skip proxy-log reconciliation."""
    run = {}
    for category in V3_CATEGORIES:
        output = Path(path) / f"bfcl_v3.{category}" / "output.jsonl"
        if not output.exists():
            continue
        with output.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        entries = [entry_costs(row, step_cap=step_cap, tokenizer=tokenizer) for row in rows]
        ids = [entry["id"] for entry in entries]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate entry ids in {output}")
        run[category] = entries
    usage = Path(path) / "usage.jsonl"
    if include_usage and usage.exists():
        attach_usage(run, usage, require_complete=require_usage)
    return run


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def _values(entries: list[dict], metric: str) -> np.ndarray:
    return np.asarray([float(entry[metric]) for entry in entries if metric in entry], dtype=np.float64)


def weighted_aggregate(run: dict[str, list[dict]], metric: str, weights: dict[str, float]) -> float:
    return float(sum(weight * _values(run[category], metric).mean() for category, weight in weights.items()))


def weighted_entries(run: dict[str, list[dict]], metric: str, weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Every entry's value and its leaderboard weight (its category's weight over the category size)."""
    values, masses = [], []
    for category, weight in weights.items():
        x = _values(run[category], metric)
        values.append(x)
        masses.append(np.full(len(x), weight / len(x)))
    return np.concatenate(values), np.concatenate(masses)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][min(np.searchsorted(cumulative, quantile * cumulative[-1]), len(values) - 1)])


def weighted_trimmed_mean(values: np.ndarray, weights: np.ndarray, fraction: float) -> float:
    low, high = weighted_quantile(values, weights, fraction), weighted_quantile(values, weights, 1 - fraction)
    keep = (values >= low) & (values <= high)
    return float(np.average(values[keep], weights=weights[keep]))


def summarize(run: dict[str, list[dict]]) -> dict:
    weights = group_weights({category: len(entries) for category, entries in run.items()})
    optional = USAGE_METRICS + ("reasoning_tokens_est",)
    metrics = ("is_correct",) + SUMMARY_METRICS
    metrics += tuple(m for m in optional if all(m in e for entries in run.values() for e in entries))
    summary = {"categories": {}, "groups": {}}
    for category, entries in run.items():
        summary["categories"][category] = {"entries": len(entries), **{m: float(_values(entries, m).mean()) for m in metrics}}
    for group, group_weights_ in weights.items():
        gen, mass = weighted_entries(run, "gen_tokens", group_weights_)
        values = {
            "entries": len(gen),
            "accuracy": weighted_aggregate(run, "is_correct", group_weights_),
            **{metric: weighted_aggregate(run, metric, group_weights_) for metric in metrics[1:]},
            "gen_tokens_median": weighted_quantile(gen, mass, 0.5),
            "gen_tokens_p90": weighted_quantile(gen, mass, 0.9),
            "gen_tokens_trimmed10": weighted_trimmed_mean(gen, mass, 0.10),
        }
        if "prompt_tokens" in metrics:
            for rho in (0.1, 0.25):
                values[f"cost_rho{rho}"] = values["gen_tokens"] + rho * values["prompt_tokens"]
        summary["groups"][group] = values
    return summary


def aes_term(base_accuracy: float, accuracy: float, base_length: float, length: float) -> float:
    """Lightning Weave's AES for one benchmark: ΔL + 3ΔA if ΔA >= 0, else ΔL - 5|ΔA| (relative changes)."""
    delta_length = (base_length - length) / base_length
    delta_accuracy = (accuracy - base_accuracy) / base_accuracy
    return delta_length + 3 * delta_accuracy if delta_accuracy >= 0 else delta_length - 5 * abs(delta_accuracy)


def aes(base: dict, model: dict, *, length_metric: str = "gen_tokens") -> dict:
    per_group = {
        group: aes_term(
            base["groups"][group]["accuracy"],
            model["groups"][group]["accuracy"],
            base["groups"][group][length_metric],
            model["groups"][group][length_metric],
        )
        for group in GROUPS
    }
    return {"per_group": per_group, "mean": float(np.mean(list(per_group.values())))}


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------


def _metric(name: str):
    return lambda base, model: (float(base[name]), float(model[name]))


def log_tokens(base: dict, model: dict) -> tuple[float, float]:
    return float(np.log1p(base["gen_tokens"])), float(np.log1p(model["gen_tokens"]))


def relative(result: dict) -> dict:
    """A bootstrapped log-ratio as a relative change (exp(·) − 1)."""
    to_relative = lambda value: float(np.expm1(value))  # noqa: E731
    return {"relative": to_relative(result["delta"]), "ci95": [to_relative(v) for v in result["ci95"]], "pairs": result["pairs"]}


def common_horizon_tokens(base: dict, model: dict) -> tuple[float, float]:
    """Generated tokens over the user turns both runs completed."""
    horizon = min(len(base["turn_tokens"]), len(model["turn_tokens"]))
    return float(sum(map(sum, base["turn_tokens"][:horizon]))), float(sum(map(sum, model["turn_tokens"][:horizon])))


def paired_bootstrap(
    base: dict[str, list[dict]],
    model: dict[str, list[dict]],
    values,
    *,
    categories: dict[str, float],
    keep=None,
    samples: int = 2_000,
    seed: int = 0,
) -> dict:
    """Stratified (per-category) paired bootstrap of Σ_c w_c · mean_c(model − base) over entry ids.

    ``values(base_entry, model_entry)`` returns the pair of values; ``keep``
    optionally restricts each category to a subset of pairs, and category
    weights are renormalized over the categories that keep any pair.
    """
    rng = np.random.default_rng(seed)
    differences = []
    for category, weight in categories.items():
        base_by_id = {entry["id"]: entry for entry in base[category]}
        model_by_id = {entry["id"]: entry for entry in model[category]}
        if base_by_id.keys() != model_by_id.keys():
            raise ValueError(f"{category}: runs do not cover the same entries")
        pairs = [(base_by_id[i], model_by_id[i]) for i in sorted(base_by_id)]
        pairs = [pair for pair in pairs if keep is None or keep(*pair)]
        if pairs:
            differences.append((weight, np.asarray([m - b for b, m in (values(*pair) for pair in pairs)])))
    if not differences:
        return {"delta": float("nan"), "ci95": [float("nan"), float("nan")], "pairs": 0}
    total = sum(weight for weight, _ in differences)
    point, draws = 0.0, np.zeros(samples)
    for weight, difference in differences:
        point += weight / total * difference.mean()
        draws += weight / total * difference[rng.integers(0, len(difference), size=(samples, len(difference)))].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"delta": float(point), "ci95": [float(low), float(high)], "pairs": sum(len(d) for _, d in differences)}


def common_no_overflow(runs: list[dict], categories) -> dict[str, set[str]]:
    """Entry ids that overflow in none of the runs, per category."""
    common = {}
    for category in categories:
        ids = [{entry["id"] for entry in run[category] if not entry["out_of_context"]} for run in runs]
        common[category] = set.intersection(*ids)
    return common


def category_rows(
    reference: dict,
    model: dict,
    categories,
    *,
    common: dict[str, set[str]] | None = None,
    samples: int = 2_000,
    seed: int = 0,
) -> list[dict]:
    """One row per category: Δaccuracy (points), flips, overflow transitions, and the common-subset delta."""
    rows = []
    for category in categories:
        before = {entry["id"]: entry for entry in reference[category]}
        pairs = [(before[entry["id"]], entry) for entry in model[category]]
        options = {"categories": {category: 1.0}, "samples": samples, "seed": seed}
        full = paired_bootstrap(reference, model, _metric("is_correct"), **options)
        row = {
            "category": category,
            "entries": len(pairs),
            "lost": sum(a["is_correct"] and not b["is_correct"] for a, b in pairs),
            "gained": sum(b["is_correct"] and not a["is_correct"] for a, b in pairs),
            "overflow": {
                "reference_only": sum(a["out_of_context"] and not b["out_of_context"] for a, b in pairs),
                "model_only": sum(b["out_of_context"] and not a["out_of_context"] for a, b in pairs),
                "both": sum(a["out_of_context"] and b["out_of_context"] for a, b in pairs),
            },
            "delta": 100 * full["delta"],
            "ci95": [100 * value for value in full["ci95"]],
        }
        if common is not None:
            keep = common[category]

            def in_subset(base: dict, _: dict, keep=keep) -> bool:
                return base["id"] in keep

            subset = paired_bootstrap(reference, model, _metric("is_correct"), keep=in_subset, **options)
            row.update(
                {
                    "delta_common_subset": 100 * subset["delta"],
                    "ci95_common_subset": [100 * value for value in subset["ci95"]],
                    "entries_common_subset": subset["pairs"],
                }
            )
        rows.append(row)
    return rows


def outcome_strata(base: dict, model: dict, categories: dict[str, float]) -> dict:
    """Leaderboard-weighted shares of both / model-only / base-only / neither correct."""
    shares = Counter()
    for category, weight in categories.items():
        base_correct = {entry["id"]: entry["is_correct"] for entry in base[category]}
        for entry in model[category]:
            shares[OUTCOMES[(base_correct[entry["id"]], entry["is_correct"])]] += weight / len(model[category])
    return dict(shares)


# The largest tolerated increase of each guard metric (leaderboard-weighted per-entry mean): duplicate calls per
# entry (base Qwen3-4B averages about 0.22), and the shares of entries that run away or overflow the window.
GUARD_TOLERANCES = {"duplicate_calls": 0.05, "runaway": 0.005, "out_of_context": 0.005}


def decision(deltas: dict, irrelevance: dict, tokens: dict, *, margin: float, min_reduction: float) -> dict:
    """Non-inferior accuracy, a real token reduction (paired log-ratio), and non-inferiority on every guard.

    A guard is non-inferior when its interval's upper bound stays within ``GUARD_TOLERANCES``: an interval that
    merely includes zero (e.g. [-0.1, +5]) does not pass.
    """
    overall = deltas["overall"]
    result = {
        "accuracy_non_inferior": overall["is_correct"]["ci95"][0] > -margin,
        "token_reduction": tokens["ci95"][1] < -min_reduction,
        "irrelevance_non_inferior": irrelevance["ci95"][0] > -margin,
        **{
            f"non_inferior_{metric}": overall[metric]["ci95"][1] <= tolerance
            for metric, tolerance in GUARD_TOLERANCES.items()
        },
    }
    rule = {"margin": margin, "min_reduction": min_reduction, "guard_tolerances": GUARD_TOLERANCES}
    return {**result, "passes": all(result.values()), "rule": rule}


def compare(
    base_run: dict, runs: dict[str, dict], *, samples: int = 2_000, margin: float = 0.01, min_reduction: float = 0.05
) -> dict:
    base_summary = summarize(base_run)
    weights = group_weights({category: len(entries) for category, entries in base_run.items()})
    groups = ("overall", *GROUPS)
    report = {"base": base_summary, "models": {}}
    common = common_no_overflow([base_run, *runs.values()], V3_CATEGORIES)
    for name, run in runs.items():
        summary = summarize(run)

        def bootstrap(values, categories, keep=None):
            return paired_bootstrap(base_run, run, values, categories=categories, keep=keep, samples=samples)

        deltas = {group: {metric: bootstrap(_metric(metric), weights[group]) for metric in COMPARED_METRICS} for group in groups}
        irrelevance = bootstrap(_metric("is_correct"), {category: 1.0 for category in IRRELEVANCE})
        both_correct = lambda b, m: b["is_correct"] and m["is_correct"]  # noqa: E731
        tokens = {group: relative(bootstrap(log_tokens, weights[group])) for group in groups}
        report["models"][name] = {
            "summary": summary,
            "categories": category_rows(base_run, run, V3_CATEGORIES, common=common, samples=samples),
            "gen_tokens_relative": tokens,
            "both_correct_gen_tokens_relative": {
                group: relative(bootstrap(log_tokens, weights[group], both_correct)) for group in groups
            },
            "deltas": deltas,
            "both_correct_gen_tokens": {
                group: bootstrap(_metric("gen_tokens"), weights[group], both_correct) for group in groups
            },
            "outcome_strata": {group: outcome_strata(base_run, run, weights[group]) for group in groups},
            "multi_turn_common_horizon_tokens": bootstrap(common_horizon_tokens, weights["multi_turn"]),
            "irrelevance_accuracy": irrelevance,
            "gen_tokens_per_success": {
                group: {
                    "base": base_summary["groups"][group]["gen_tokens"] / base_summary["groups"][group]["accuracy"],
                    "model": summary["groups"][group]["gen_tokens"] / summary["groups"][group]["accuracy"],
                }
                for group in groups
            },
            "decision": decision(deltas, irrelevance, tokens["overall"], margin=margin, min_reduction=min_reduction),
            "aes_descriptive": aes(base_summary, summary),
        }
    return report


def format_table(report: dict) -> str:
    columns = ("accuracy", "gen_tokens", "gen_tokens_median", "gen_tokens_p90", "steps", "tool_calls", "runaway")
    rows = [("base", report["base"])] + [(name, item["summary"]) for name, item in report["models"].items()]
    lines = []
    for group in ("overall", *GROUPS):
        lines.append(f"\n{group}\n| model | " + " | ".join(columns) + " |")
        lines.append("|---" * (len(columns) + 1) + "|")
        for name, summary in rows:
            values = summary["groups"][group]
            cells = [f"{100 * values[c]:.2f}" if c in ("accuracy", "runaway") else f"{values[c]:.1f}" for c in columns]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")

    def interval(result, scale=1.0):
        return f"{scale * result['delta']:+.1f} [{scale * result['ci95'][0]:+.1f}, {scale * result['ci95'][1]:+.1f}]"

    def percent(result):
        return f"{100 * result['relative']:+.1f}% [{100 * result['ci95'][0]:+.1f}, {100 * result['ci95'][1]:+.1f}]"

    for name, item in report["models"].items():
        lines.append(f"\n{name} vs base: decision {item['decision']}")
        for group, deltas in item["deltas"].items():
            lines.append(
                f"  {group}: tokens {percent(item['gen_tokens_relative'][group])}"
                f" (both correct {percent(item['both_correct_gen_tokens_relative'][group])})"
                f"  Δacc {interval(deltas['is_correct'], 100)}  Δmean gen {interval(deltas['gen_tokens'])}"
            )
        lines.append(f"  multi-turn common-horizon Δgen {interval(item['multi_turn_common_horizon_tokens'])}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", type=Path, help="run directory used as the reference")
    parser.add_argument("models", nargs="*", type=Path, help="run directories compared against the base")
    parser.add_argument("--tokenizer", help="tokenizer path/id for a reasoning/visible split without usage logs")
    parser.add_argument("--step-cap", type=int, default=DEFAULT_STEP_CAP)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--margin", type=float, default=0.01, help="accuracy non-inferiority margin (fraction)")
    parser.add_argument("--min-reduction", type=float, default=0.05, help="required generated-token reduction")
    parser.add_argument("--allow-partial-usage", action="store_true", help="tolerate entries without reconciled usage")
    parser.add_argument("--output", type=Path, help="write the full JSON report here")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def load(path):
        return load_run(path, step_cap=args.step_cap, tokenizer=tokenizer, require_usage=not args.allow_partial_usage)

    report = compare(
        load(args.base),
        {path.name: load(path) for path in args.models},
        samples=args.bootstrap,
        margin=args.margin,
        min_reduction=args.min_reduction,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(format_table(report))


if __name__ == "__main__":
    main()
