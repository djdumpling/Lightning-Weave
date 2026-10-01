#!/usr/bin/env python3
"""Validate the token-state taxonomy against manual annotation.

``export`` writes a balanced, deterministic sample of cached states per
heuristic type. Each record holds the decoded end of the prefix, the student's
top candidates with probabilities, the heuristic label, and an empty
``annotation`` field for the state type a reader assigns. ``score`` reports, per
heuristic type, the precision against those annotations and the confusions.
Run the analysis at several ``--fork-mass`` thresholds for sensitivity.
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

from data_curation.shift_geometry import iter_joined_rows, prompt_unit
from data_curation.shift_states import FORK_MASS, STATE_TYPES, StateLabeler, TokenTable


def export(base: Path, labeler: StateLabeler, *, per_type: int, window: int, candidates: int) -> list[dict]:
    pool: dict[str, list[tuple[float, dict]]] = {name: [] for name in STATE_TYPES}
    table = labeler.table
    for row in iter_joined_rows(base, {}):
        probs = np.exp(row.behavior_log_probs)
        codes = labeler.label(row.response_tokens, row.candidate_ids, probs)
        for position in np.nonzero(row.loss_mask)[0]:
            state = STATE_TYPES[codes[position]]
            order = np.argsort(-probs[position])[:candidates]
            pool[state].append(
                (
                    prompt_unit(f"{row.sample_id}:{position}", "annotation"),
                    {
                        "sample_id": row.sample_id,
                        "position": int(position),
                        "label": state,
                        "prefix_tail": "".join(table.text(t) for t in row.response_tokens[max(0, position - window) : position]),
                        "candidates": [[table.text(row.candidate_ids[position][k]), round(float(probs[position][k]), 4)] for k in order],
                        "annotation": "",
                    },
                )
            )
    return [item for state in STATE_TYPES for _, item in sorted(pool[state], key=lambda pair: pair[0])[:per_type]]


def score(records: list[dict]) -> dict:
    annotated = [record for record in records if record.get("annotation")]
    report = {}
    for state in STATE_TYPES:
        labeled = [record for record in annotated if record["label"] == state]
        if labeled:
            report[state] = {
                "annotated": len(labeled),
                "precision": sum(record["annotation"] == state for record in labeled) / len(labeled),
                "confusions": dict(Counter(record["annotation"] for record in labeled if record["annotation"] != state)),
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    exporter = sub.add_parser("export")
    exporter.add_argument("--base", required=True, type=Path)
    exporter.add_argument("--tokenizer-json", required=True, type=Path)
    exporter.add_argument("--output", required=True, type=Path)
    exporter.add_argument("--per-type", type=int, default=40)
    exporter.add_argument("--window", type=int, default=48)
    exporter.add_argument("--candidates", type=int, default=6)
    exporter.add_argument("--fork-mass", type=float, default=FORK_MASS)
    scorer = sub.add_parser("score")
    scorer.add_argument("annotations", type=Path)
    args = parser.parse_args()
    if args.stage == "export":
        labeler = StateLabeler(TokenTable.from_tokenizer_json(args.tokenizer_json), fork_mass=args.fork_mass)
        records = export(args.base, labeler, per_type=args.per_type, window=args.window, candidates=args.candidates)
        args.output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
        print(json.dumps(Counter(record["label"] for record in records), indent=2))
    else:
        records = [json.loads(line) for line in args.annotations.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(json.dumps(score(records), indent=2))


if __name__ == "__main__":
    main()
