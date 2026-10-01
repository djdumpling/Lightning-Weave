#!/usr/bin/env python3
"""Step-0 BFCL diagnostics for the agent-efficiency experiments (CPU only, from finished runs).

``categories``: paired Δaccuracy per category for each model against a reference run (primary), with the
entries lost and gained, and the overflow transitions (entries that exceed the 65k serving window in the
reference only, the model only, or both). As a sensitivity analysis only, the same delta is also given on
one fixed subset: entries that overflow in none of the runs compared in the invocation. The subset is
selected by policy-affected outcomes, so it is not the primary comparison.

``template``: renders a tool conversation through a served model's exact chat template and reports which
earlier reasoning survives. It checks reasoning passed as a ``reasoning_content`` field and reasoning
inlined as ``<think>`` text, for an assistant turn before the last user message and for one after it.
Whether reasoning reaches the model also depends on what the harness sends back; for new runs, the usage
proxy logs that per request (``history_reasoning_chars``).

``history``: an exploratory cross-check from old usage logs. It asks whether a multi-turn request's prompt
grew by less than the previous step's reasoning. That is consistent with the reasoning being dropped, but
does not prove it: other text can be removed at the same time, and large tool outputs can mask a drop. The
usage join matches token counts, not trajectories. So only entries whose attached requests follow the
recorded step order, with each request's user messages extending the previous request's, are used; the
rest are counted as ambiguous. Even a confirmed drop would not show that the reasoning was dispensable.

    python evaluation/bfcl_breakdown.py --root RUNS --reference ae.joint.acc-legacy \\
        --models ae.joint.acc-legacy+decs --template /path/to/served/model --output breakdown.json
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

from evaluation.bfcl_efficiency import IRRELEVANCE, MULTI_TURN, V3_CATEGORIES, _metric, load_run, paired_bootstrap

DEFAULT_CATEGORIES = (*MULTI_TURN, *IRRELEVANCE, "live_relevance")
MIN_REASONING = 256  # only pairs whose previous step reasoned this much are informative
TEMPLATE_MARKERS = {
    "field_before_last_user": "RSN_FIELD_BEFORE",
    "field_after_last_user": "RSN_FIELD_AFTER",
    "inline_before_last_user": "RSN_INLINE_BEFORE",
    "inline_after_last_user": "RSN_INLINE_AFTER",
}


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


def template_retention(tokenizer) -> dict[str, bool]:
    """Which earlier reasoning a chat template keeps in a rendered two-turn tool conversation."""
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value.",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
        },
    }

    def call(key: str) -> list[dict]:
        return [{"id": key, "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"key": key})}}]

    def turn(user: str, field: str, inline: str, key: str) -> list[dict]:
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "reasoning_content": field, "tool_calls": call(key)},
            {"role": "tool", "tool_call_id": key, "content": "1"},
            {"role": "assistant", "content": f"<think>\n{inline}\n</think>\n\n", "tool_calls": call(key + "2")},
            {"role": "tool", "tool_call_id": key + "2", "content": "2"},
        ]

    marks = TEMPLATE_MARKERS
    messages = [
        *turn("first question", marks["field_before_last_user"], marks["inline_before_last_user"], "a"),
        {"role": "assistant", "content": "The values are 1 and 2."},
        *turn("second question", marks["field_after_last_user"], marks["inline_after_last_user"], "b"),
    ]
    rendered = tokenizer.apply_chat_template(messages, tools=[tool], tokenize=False, add_generation_prompt=True)
    return {name: marker in rendered for name, marker in marks.items()}


def trajectory_consistent(entry: dict) -> bool:
    """Attached requests follow the recorded step order, and each extends the previous request's user messages."""
    requests = entry.get("requests") or []
    recorded = [tokens for tokens in entry.get("step_tokens", []) if tokens]
    if [r["completion_tokens"] for r in requests if r["completion_tokens"]] != recorded:
        return False
    return all(b["users"][: len(a["users"])] == a["users"] for a, b in itertools.pairwise(requests))


def history_pairs(run: dict, *, min_reasoning: int = MIN_REASONING) -> dict:
    """Prompt growth between consecutive requests of trajectory-consistent multi-turn entries."""
    pairs = {"within_turn": [], "new_turn": []}
    unreconciled = ambiguous = 0
    for category in MULTI_TURN:
        for entry in run.get(category, []):
            if entry.get("requests") is None:
                unreconciled += 1
                continue
            if not trajectory_consistent(entry):
                ambiguous += 1
                continue
            for previous, current in itertools.pairwise(entry["requests"]):
                reasoning = previous.get("reasoning_tokens")
                prompts = (previous.get("prompt_tokens"), current.get("prompt_tokens"))
                if reasoning is None or reasoning < min_reasoning or None in prompts:
                    continue
                visible = previous["completion_tokens"] - reasoning
                growth = prompts[1] - prompts[0]
                same_turn = len(current["users"]) == len(previous["users"])
                pairs["within_turn" if same_turn else "new_turn"].append((growth, reasoning, visible))
    summary = {"unreconciled_entries": unreconciled, "ambiguous_entries": ambiguous, "min_reasoning": min_reasoning}
    for kind, values in pairs.items():
        if not values:
            summary[kind] = {"pairs": 0}
            continue
        growth, reasoning, visible = (np.asarray(column, dtype=np.float64) for column in zip(*values))
        summary[kind] = {
            "pairs": len(values),
            "growth_below_previous_reasoning_fraction": float((growth < reasoning).mean()),
            "negative_growth_fraction": float((growth < 0).mean()),
            "median_growth": float(np.median(growth)),
            "median_previous_reasoning": float(np.median(reasoning)),
            "median_growth_beyond_visible_over_reasoning": float(np.median((growth - visible) / reasoning)),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, help="directory holding one downloaded run per tag")
    parser.add_argument("--reference")
    parser.add_argument("--models", default="", help="comma-separated tags compared with --reference")
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES), help="comma-separated, or 'all'")
    parser.add_argument("--template", default="", help="served model or tokenizer directory to render")
    parser.add_argument("--history", default="", help="comma-separated tags for the exploratory growth check")
    parser.add_argument("--samples", type=int, default=2_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = V3_CATEGORIES if args.categories == "all" else tuple(c for c in args.categories.split(",") if c)
    unknown = sorted(set(categories) - set(V3_CATEGORIES))
    if unknown:
        raise ValueError(f"unknown categories {unknown}")
    models = [tag for tag in args.models.split(",") if tag]
    histories = [tag for tag in args.history.split(",") if tag]
    if (models or histories) and args.root is None:
        raise ValueError("--models and --history need --root")
    if models and not args.reference:
        raise ValueError("--models needs --reference")
    runs: dict[str, dict] = {}

    def run(tag: str) -> dict:
        if tag not in runs:
            # Categories need no usage log; the history check counts entries whose log did not reconcile.
            runs[tag] = load_run(args.root / tag, require_usage=False)
        return runs[tag]

    report = {"reference": args.reference, "categories": {}, "template": None, "history": {}}
    if models:
        common = common_no_overflow([run(args.reference), *map(run, models)], categories)
        header = ["model", "category", "entries", "lost / gained", "overflow ref / model / both", "Δacc"]
        print("| " + " | ".join([*header, "Δacc, common subset"]) + " |")
        print("|---|---|---|---|---|---|---|")
        for tag in models:
            rows = category_rows(run(args.reference), run(tag), categories, common=common, samples=args.samples)
            report["categories"][tag] = rows
            for row in rows:
                ci, sub, flow = row["ci95"], row["ci95_common_subset"], row["overflow"]
                print(
                    f"| {tag} | {row['category']} | {row['entries']} | {row['lost']} / {row['gained']} | "
                    f"{flow['reference_only']} / {flow['model_only']} / {flow['both']} | "
                    f"{row['delta']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | "
                    f"{row['delta_common_subset']:+.2f} [{sub[0]:+.2f}, {sub[1]:+.2f}] "
                    f"(n={row['entries_common_subset']}) |"
                )
    if args.template:
        from transformers import AutoTokenizer

        report["template"] = template_retention(AutoTokenizer.from_pretrained(args.template))
        print(f"\nchat template {args.template}: {json.dumps(report['template'], indent=2)}")
    for tag in histories:
        report["history"][tag] = history_pairs(run(tag))
        print(f"\nprompt growth (exploratory), {tag}: {json.dumps(report['history'][tag], indent=2)}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
