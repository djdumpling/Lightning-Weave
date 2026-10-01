#!/usr/bin/env python3
"""Where do efficiency arms lose multi-turn BFCL accuracy? First failures and turn-level behavior (CPU only).

BFCL grades a multi-turn entry turn by turn and stops at the first failing turn. This reads that failure from
the score file and gives it one class, checked in this order:

- ``step_limit``: the harness stopped the conversation at a turn's step cap (a loop). Serving-window
  overflows are not failures of this kind; entries that overflow in any loaded run are dropped.
- ``acted_early``: on an unanswerable turn just before the failing turn, the model already called a function
  that the failing turn's reference calls, instead of asking or waiting. An unanswerable turn is one whose
  reference is empty: a needed function is missing (miss_func) or a needed parameter is (miss_param).
- ``no_action``: no executable call on a turn whose reference calls functions.
- The failing turn's calls compared with its reference by function name, as multisets:
  - ``missing_calls``: only fewer;
  - ``different_calls``: some missing and some not in the reference;
  - ``repeated_calls``: every reference call, and some of them again;
  - ``extra_calls``: every reference call plus other functions;
  - ``wrong_arguments``: the same names.

The last five classes are name-level heuristics. The reference is one valid path, and state failures are
graded on the environment's state, not on the calls.

For an arm compared with its reference over the same entries, the per-class changes in failure rate (arm minus
reference) sum to the multi-turn accuracy change with the sign flipped, so together they decompose the loss.
The same split is given by the failing turn's index. Two seed pairs are pooled by averaging per-entry
differences, and intervals resample entries. At sampling temperature, flips in both directions are frequent,
so the lost entries alone mostly show noise. The decomposition is the signal, and the accuracy-only student's
two training seeds give a null comparison.

Turn-level behavior compares rates. Correctness rates cover turns up to and including the first failure (every
turn of a correct entry):
- action turns without a call;
- action turns missing reference calls (among turns with a call);
- action turns with calls beyond the reference;
- action turns calling a reference function more often than the reference;
- identical calls (same function and arguments) repeated within a turn, per call;
- unanswerable turns with a call.

Cost rates cover every generated turn:
- steps and calls per turn;
- completion tokens of a turn's first step (the response to the user's message), overall and separately for the
  conversation's first turn and for later turns (which answer a new user message mid-conversation);
- completion tokens of its later steps (responses to tool results).

``--comparisons`` replaces the built-in comparisons with a JSON file ``{name: {"pairs": [[arm, reference], ...]}}``
(the format of ``evaluation/bfcl_pooled.py``; other keys are ignored), naming run directories relative to ``--root``.

The reference answers must come from the gorilla pin (``GORILLA_COMMIT`` in ``configs/bfcl_eval/config.py``),
``berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_<category>.json``. They are checked
against every graded failure.

    python evaluation/bfcl_multiturn_failures.py --root RUNS --ground-truth DIR --output failures.json \\
        --examples examples.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.bfcl_efficiency import MULTI_TURN, _calls, _steps, entry_costs
from evaluation.bfcl_shared_factor import R1, R2, J

CLASSES = (
    "step_limit",
    "acted_early",
    "no_action",
    "missing_calls",
    "different_calls",
    "repeated_calls",
    "extra_calls",
    "wrong_arguments",
)
TURN_BUCKETS = ("0", "1", "2", "3+")
# name: (numerator, denominator, scope); "on_path" stops at the first failure, "all" covers every generated turn
RATES = {
    "no_call": ("no_call", "action_turns", "on_path"),
    "missing_calls": ("missing", "called_action_turns", "on_path"),
    "extra_calls": ("extra", "action_turns", "on_path"),
    "repeated_reference_call": ("repeated", "action_turns", "on_path"),
    "identical_repeats": ("identical_repeats", "on_path_calls", "on_path"),
    "acted_on_unanswerable": ("early", "unanswerable_turns", "on_path"),
    "steps_per_turn": ("steps", "turns", "all"),
    "calls_per_turn": ("calls", "turns", "all"),
    "first_step_tokens": ("first_step_tokens", "turns", "all"),
    "first_step_tokens_first_turn": ("first_step_tokens_first_turn", "first_turns", "all"),
    "first_step_tokens_later_turns": ("first_step_tokens_later_turns", "later_turns", "all"),
    "later_step_tokens": ("later_step_tokens", "later_steps", "all"),
}
COST_RATES = (
    "steps_per_turn",
    "calls_per_turn",
    "first_step_tokens",
    "first_step_tokens_first_turn",
    "first_step_tokens_later_turns",
    "later_step_tokens",
)
COMPARISONS = {
    "decs": [(f"{J}acc-legacy+decs", R1), (f"{J}acc-legacy+decs.s5678", R2)],
    "decs, protect first": [
        (f"{J}acc-legacy+decs-protect-first", R1),
        (f"{J}acc-legacy+decs-protect-first.s5678", R2),
    ],
    "decs7b": [(f"{J}acc-legacy+decs7b", R1)],
    "l1max": [(f"{J}acc-legacy+l1max", R1)],
    "donor sum": [(f"{J}acc-legacy+donor-mean", R1)],
    "concise prompt": [(f"{J}acc-legacy.concise", R1)],
    "null: accuracy-only seed 5678 vs 1234": [(R2, R1)],
}
EXAMPLE_COMPARISON = "decs"
REFERENCE_NAME = re.compile(r"\s*([A-Za-z_][\w.]*)\s*\(")
TURN = re.compile(r"for turn (\d+)")
STATE, EMPTY, RESPONSE, FORCED = (
    f"multi_turn:{kind}"
    for kind in (
        "instance_state_mismatch",
        "empty_turn_model_response",
        "execution_response_mismatch",
        "force_terminated",
    )
)


def reference_names(calls: list[str]) -> Counter:
    names = Counter()
    for call in calls:
        match = REFERENCE_NAME.match(call)
        if match is None:
            raise ValueError(f"unparseable reference call {call!r}")
        names[match.group(1)] += 1
    return names


def turn_names(turn: list) -> Counter:
    return Counter(name for step in turn for name, _ in _calls(step))


def failure_turn(error: dict, generated_turns: int) -> int:
    kind = error["error_type"]
    if kind == STATE:
        return len(error["execution_result"]) - 1
    if kind in (EMPTY, RESPONSE):
        return int(TURN.search(error["error_message"]).group(1))
    if kind == FORCED:
        return generated_turns - 1  # the turn that hit the step cap, or was cut by the window
    raise ValueError(f"unexpected multi-turn error {kind!r}")


def classify(turns: list[list], reference: list[list[str]], error: dict, *, overflow: bool) -> tuple[str, int]:
    """(class, failing turn) of a failed entry; ``overflow`` marks serving-window overflows."""
    turn = failure_turn(error, len(turns))
    if error["error_type"] == FORCED:
        return ("overflow" if overflow else "step_limit"), turn
    expected = reference_names(reference[turn])
    early = Counter()
    previous = turn - 1
    while previous >= 0 and not reference[previous]:
        early += turn_names(turns[previous])
        previous -= 1
    if set(early) & set(expected):
        return "acted_early", turn
    if error["error_type"] == EMPTY:
        return "no_action", turn
    made = turn_names(turns[turn])
    missing, extra = expected - made, made - expected
    if missing:
        return ("different_calls" if extra else "missing_calls"), turn
    if not extra:
        return "wrong_arguments", turn
    return ("repeated_calls" if set(extra) <= set(expected) else "extra_calls"), turn


def behavior(turns: list[list], turn_tokens: list[list[int]], reference: list[list[str]], last: int) -> dict:
    """Per-entry counts for RATES: correctness counts through turn ``last``, cost counts over all turns."""
    counts = Counter()
    for index, (turn, tokens) in enumerate(zip(turns, turn_tokens, strict=True)):
        made = turn_names(turn)
        counts["turns"] += 1
        counts["steps"] += len(turn)
        counts["calls"] += sum(made.values())
        counts["first_step_tokens"] += tokens[0] if tokens else 0
        first = index == 0
        counts["first_turns" if first else "later_turns"] += 1
        counts["first_step_tokens_first_turn" if first else "first_step_tokens_later_turns"] += (
            tokens[0] if tokens else 0
        )
        counts["later_steps"] += max(len(tokens) - 1, 0)
        counts["later_step_tokens"] += sum(tokens[1:])
        if index > last:
            continue
        counts["on_path_calls"] += sum(made.values())
        identical = Counter(call for step in turn for call in _calls(step))
        counts["identical_repeats"] += sum(count - 1 for count in identical.values())
        expected = reference_names(reference[index])
        if expected:
            counts["action_turns"] += 1
            counts["no_call"] += not made
            counts["called_action_turns"] += bool(made)
            counts["missing"] += bool(made) and bool(expected - made)
            counts["extra"] += bool(made - expected)
            counts["repeated"] += any(made[name] > count for name, count in expected.items())
        else:
            counts["unanswerable_turns"] += 1
            counts["early"] += bool(made)
    return dict(counts)


def load_ground_truth(directory: Path) -> dict[str, list[list[str]]]:
    truth = {}
    for category in MULTI_TURN:
        with (directory / f"BFCL_v4_{category}.json").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    truth[row["id"]] = row["ground_truth"]
    return truth


def load_multiturn(path: Path, truth: dict[str, list[list[str]]]) -> dict[str, dict]:
    """{entry id: record} over the multi-turn categories of one run."""
    records = {}
    for category in MULTI_TURN:
        with (path / f"bfcl_v3.{category}" / "output.jsonl").open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        lines = (path / "scores" / f"BFCL_v4_{category}_score.json").read_text(encoding="utf-8").splitlines()
        failures = {row["id"]: row for row in map(json.loads, lines[1:])}
        for row in rows:
            entry_id, reference = row["id"], truth[row["id"]]
            costs = entry_costs(row)
            turns = _steps(row["generation"])
            failed = failures.get(entry_id)
            if (failed is None) != costs["is_correct"]:
                raise ValueError(f"{path.name} {entry_id}: output and score file disagree on correctness")
            record = {
                "category": category,
                "correct": costs["is_correct"],
                "overflow": costs["out_of_context"],
                "turns": turns,
                "turn_tokens": costs["turn_tokens"],
                "reference": reference,
                "class": None,
                "failure_turn": None,
            }
            last = len(turns) - 1
            if failed is not None:
                if failed["possible_answer"] != reference:
                    raise ValueError(f"{entry_id}: graded reference differs from --ground-truth (wrong gorilla pin?)")
                record["class"], record["failure_turn"] = classify(
                    turns, reference, failed["error"], overflow=costs["out_of_context"]
                )
                record["error"] = failed["error"]
                last = record["failure_turn"]
            elif len(turns) != len(reference):
                raise ValueError(f"{entry_id}: a correct entry has {len(turns)} turns for {len(reference)}")
            if not record["overflow"]:
                record["behavior"] = behavior(turns, costs["turn_tokens"], reference, last)
            records[entry_id] = record
    return records


def aligned_ids(runs: dict[str, dict[str, dict]]) -> tuple[list[str], int]:
    """Entry ids present in every run and overflowing in none, and how many were dropped."""
    first = next(iter(runs.values()))
    if any(records.keys() != first.keys() for records in runs.values()):
        raise ValueError("runs do not cover the same multi-turn entries")
    keep = sorted(e for e in first if not any(records[e]["overflow"] for records in runs.values()))
    return keep, len(first) - len(keep)


def turn_bucket(turn: int) -> str:
    return TURN_BUCKETS[min(turn, len(TURN_BUCKETS) - 1)]


def indicators(records: dict[str, dict], ids: list[str], labels, label_of) -> np.ndarray:
    """(entries, labels) 0/1 matrix: which label each entry's first failure has (all zero when correct)."""
    matrix = np.zeros((len(ids), len(labels)))
    position = {label: j for j, label in enumerate(labels)}
    for i, entry_id in enumerate(ids):
        record = records[entry_id]
        if not record["correct"]:
            matrix[i, position[label_of(record)]] = 1.0
    return matrix


def interval(draws: np.ndarray) -> list[float]:
    return np.percentile(draws, [2.5, 97.5], axis=0).T.tolist()


def decompose(runs, pairs, ids, *, weights: np.ndarray) -> dict:
    """Per-class and per-turn change in failure rate (points), pooled over ``pairs``."""
    report = {}
    splits = {
        "by_class": (CLASSES, lambda record: record["class"]),
        "by_failure_turn": (TURN_BUCKETS, lambda record: turn_bucket(record["failure_turn"])),
    }
    for split, (labels, label_of) in splits.items():
        change = np.mean(
            [
                indicators(runs[arm], ids, labels, label_of) - indicators(runs[ref], ids, labels, label_of)
                for arm, ref in pairs
            ],
            axis=0,
        )
        point = 100 * change.mean(axis=0)
        draws = 100 * (weights @ change) / len(ids)
        bounds = interval(draws)
        report[split] = {label: {"delta": float(point[j]), "ci95": bounds[j]} for j, label in enumerate(labels)}
    correct = np.mean(
        [[float(runs[a][e]["correct"]) - float(runs[r][e]["correct"]) for e in ids] for a, r in pairs], axis=0
    )
    report["accuracy_change"] = {
        "delta": float(100 * correct.mean()),
        "ci95": interval(100 * (weights @ correct) / len(ids)),
    }
    return report


def compare_behavior(runs, pairs, ids, *, weights: np.ndarray) -> dict:
    """Per rate: reference and arm values (pooled means over pairs), difference, and relative change."""
    report = {}
    for name, (numerator, denominator, _) in RATES.items():
        values = {"ref": [], "arm": []}
        for arm, ref in pairs:
            for side, tag in (("arm", arm), ("ref", ref)):
                num = np.asarray([runs[tag][e]["behavior"].get(numerator, 0) for e in ids], dtype=np.float64)
                den = np.asarray([runs[tag][e]["behavior"].get(denominator, 0) for e in ids], dtype=np.float64)
                values[side].append((num.sum() / den.sum(), (weights @ num) / (weights @ den)))
        point = {side: float(np.mean([v[0] for v in items])) for side, items in values.items()}
        draws = {side: np.mean([v[1] for v in items], axis=0) for side, items in values.items()}
        report[name] = {
            "reference": point["ref"],
            "arm": point["arm"],
            "delta": point["arm"] - point["ref"],
            "ci95": interval(draws["arm"] - draws["ref"]),
        }
        if name in COST_RATES:
            report[name]["relative"] = point["arm"] / point["ref"] - 1
            report[name]["relative_ci95"] = interval(draws["arm"] / draws["ref"] - 1)
    return report


def calls_text(turn: list) -> list[str]:
    return [f"{name}({arguments})" for step in turn for name, arguments in _calls(step)]


def example(entry_id: str, arm_record: dict, ref_record: dict, question: list[list[dict]], arm: str) -> dict:
    turn = arm_record["failure_turn"]
    first = turn
    while first > 0 and not arm_record["reference"][first - 1]:
        first -= 1
    arm_turns, ref_turns = arm_record["turns"], ref_record["turns"]
    return {
        "id": entry_id,
        "arm": arm,
        "class": arm_record["class"],
        "failure_turn": turn,
        "error": arm_record["error"]["error_type"],
        "user": {
            t: [m.get("content") for m in question[t] if m.get("role") == "user"] for t in range(first, turn + 1)
        },
        "reference_calls": {t: arm_record["reference"][t] for t in range(first, turn + 1)},
        "reference_run_calls": {t: calls_text(ref_turns[t]) for t in range(first, turn + 1)},
        "arm_calls": {t: calls_text(arm_turns[t]) for t in range(first, min(turn + 1, len(arm_turns)))},
        "arm_text": [step[:500] for step in arm_turns[turn] if isinstance(step, str)][:3]
        if turn < len(arm_turns)
        else [],
        "tokens": {"reference_run": ref_record["turn_tokens"][turn], "arm": arm_record["turn_tokens"][turn]},
    }


def examples(runs, questions, pairs, ids, *, per_class: int) -> dict[str, list[dict]]:
    """Up to ``per_class`` entries per class that the reference solved and the arm failed, in id order."""
    found: dict[str, list[dict]] = {label: [] for label in CLASSES}
    for arm, ref in pairs:
        for entry_id in ids:
            arm_record, ref_record = runs[arm][entry_id], runs[ref][entry_id]
            if ref_record["correct"] and not arm_record["correct"]:
                items = found[arm_record["class"]]
                if len(items) < per_class:
                    items.append(example(entry_id, arm_record, ref_record, questions[entry_id], arm))
    return found


def analyze(runs, comparisons, *, draws: int = 2_000, seed: int = 0) -> dict:
    ids, dropped = aligned_ids(runs)
    weights = np.random.default_rng(seed).multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=draws)
    report = {"entries": len(ids), "dropped_overflow_entries": dropped, "comparisons": {}}
    for name, pairs in comparisons.items():
        report["comparisons"][name] = {
            "pairs": [list(pair) for pair in pairs],
            **decompose(runs, pairs, ids, weights=weights),
            "behavior": compare_behavior(runs, pairs, ids, weights=weights),
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="directory holding one downloaded run per tag")
    parser.add_argument("--ground-truth", required=True, type=Path, help="gorilla-pinned possible_answer directory")
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--examples", type=Path, help="write lost entries of the DECS pairs here")
    parser.add_argument("--examples-per-class", type=int, default=8)
    parser.add_argument("--comparisons", type=Path, help='JSON {name: {"pairs": [[arm, reference], ...]}}')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = load_ground_truth(args.ground_truth)
    comparisons = COMPARISONS
    if args.comparisons:
        loaded = json.loads(args.comparisons.read_text(encoding="utf-8"))
        comparisons = {name: [tuple(pair) for pair in item["pairs"]] for name, item in loaded.items()}
    tags = sorted({tag for pairs in comparisons.values() for pair in pairs for tag in pair})
    runs = {tag: load_multiturn(args.root / tag, truth) for tag in tags}
    report = analyze(runs, comparisons, draws=args.draws)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"multi-turn entries {report['entries']} (dropped {report['dropped_overflow_entries']} overflowing)")
    for name, item in report["comparisons"].items():
        low, high = item["accuracy_change"]["ci95"]
        print(f"\n{name}: Δaccuracy {item['accuracy_change']['delta']:+.2f} [{low:+.2f}, {high:+.2f}]")
        for split in ("by_class", "by_failure_turn"):
            cells = [
                f"{k} {v['delta']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]" for k, v in item[split].items()
            ]
            print(f"  Δfailure {split}: " + " | ".join(cells))
        cells = []
        for rate, row in item["behavior"].items():
            if rate in COST_RATES:
                cells.append(f"{rate} {row['reference']:.3g}→{row['arm']:.3g} ({100 * row['relative']:+.1f}%)")
            else:
                cells.append(f"{rate} {row['reference']:.3f}→{row['arm']:.3f}")
        print("  behavior: " + " | ".join(cells))
    if args.examples:
        pairs = comparisons.get(EXAMPLE_COMPARISON) or next(iter(comparisons.values()))
        questions = {}
        for category in MULTI_TURN:
            path = args.root / pairs[0][1] / f"bfcl_v3.{category}" / "output.jsonl"
            with path.open(encoding="utf-8") as handle:
                questions.update({row["id"]: row["question"] for row in map(json.loads, handle)})
        ids, _ = aligned_ids(runs)
        found = examples(runs, questions, pairs, ids, per_class=args.examples_per_class)
        args.examples.write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
