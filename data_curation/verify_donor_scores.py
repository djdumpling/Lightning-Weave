#!/usr/bin/env python3
"""Check a donor pair's scored shards against the base cache and report its evidence.

Checks, per shard: the same file names as the base, ``sample_id`` order,
candidate ids and response lengths, finite scores on every mapped candidate, and
the expected post/pre revisions. It then reports, per token-state type:

- the evidence the composer will use (mapped by the scorer AND trained by both
  anchors): whole-position coverage and behavior-mass coverage;
- the donor's own support mass on those candidates (median and p10);
- how often a structural token's scored log-probability exactly equals another
  special candidate's at the same state, the signature of untrained rows.

On success it writes ``evidence.json``, which the Modal composition and
analysis stages require before they read the pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import write_json
from data_curation.shift_geometry import iter_joined_rows, load_untrained, read_shard_fields, shard_names
from data_curation.shift_states import STATE_TYPES, StateLabeler, TokenTable

def check_shards(base: Path, donor: Path, revisions: dict[str, str]) -> int:
    """Raise unless ``donor`` holds exactly the base's shards and rows, scored by the expected anchors."""
    names = shard_names(base)
    extra = sorted({path.name for path in donor.glob("*.parquet")} - set(names))
    if extra:
        raise ValueError(f"{donor} has shards the base lacks: {extra}")
    rows = 0
    for name in names:
        path = donor / name
        if not path.exists():
            raise FileNotFoundError(f"{donor} lacks shard {name}")
        base_columns = read_shard_fields(base / name, ("sample_id", "candidate_ids"))
        columns = read_shard_fields(
            path,
            ("sample_id", "candidate_ids", "post_teacher_revision", "pre_teacher_revision", "post_teacher_log_probs", "pre_teacher_log_probs"),
        )
        if columns["sample_id"] != base_columns["sample_id"]:
            raise ValueError(f"{path}: rows are not the base's rows in the base's order")
        for role, expected in revisions.items():
            observed = set(columns[f"{role}_teacher_revision"])
            if observed != {expected}:
                raise ValueError(f"{path}: {role} revisions {sorted(observed)} != {expected}")
        for index, (candidates, reference) in enumerate(zip(columns["candidate_ids"], base_columns["candidate_ids"])):
            if not np.array_equal(candidates, reference):
                raise ValueError(f"{path}: row {index} candidates differ from the base")
            for role in revisions:
                scores = columns[f"{role}_teacher_log_probs"][index]
                if scores.shape != reference.shape:
                    raise ValueError(f"{path}: row {index} {role} scores have shape {scores.shape}")
        rows += len(columns["sample_id"])
    return rows


def evidence_report(rows, name: str, labeler: StateLabeler, structural: dict[int, str]) -> dict:
    states = ("all", *STATE_TYPES)
    counts = {state: np.zeros(4) for state in states}
    support = {state: [] for state in states}
    ties = {symbol: np.zeros(2) for symbol in structural.values()}
    special = np.asarray(sorted(structural))
    for row in rows:
        valid, mapped = row.valid(name), row.mapped[name]
        shift = row.shifts[name]
        if not np.isfinite(shift[mapped]).all():
            raise ValueError(f"{row.sample_id}: non-finite scores on mapped candidates")
        probs = np.exp(row.behavior_log_probs)
        codes = labeler.label(row.response_tokens, row.candidate_ids, probs)
        for state in states:
            selected = row.loss_mask if state == "all" else row.loss_mask & (codes == STATE_TYPES.index(state))
            counts[state] += [
                selected.sum(),
                valid[selected].all(axis=-1).sum(),
                (probs[selected] * valid[selected]).sum(),
                probs[selected].sum(),
            ]
            support[state].append(row.support[name][selected])
        # Untrained rows give identical log-probabilities to every such token at a state, so
        # their shifts tie exactly; trained tokens essentially never do.
        is_special = np.isin(row.candidate_ids, special) & mapped
        for position in np.nonzero(is_special.sum(axis=-1) >= 2)[0]:
            slots = np.nonzero(is_special[position])[0]
            values = shift[position, slots]
            for slot, value in zip(slots, values):
                symbol = structural[int(row.candidate_ids[position, slot])]
                ties[symbol] += [np.sum(np.isclose(values, value, rtol=0, atol=1e-6)) > 1, 1]
    report = {}
    for state in states:
        total, kept, mass, total_mass = counts[state]
        if not total:
            continue
        values = np.concatenate(support[state])
        report[state] = {
            "positions": int(total),
            "whole_position_evidence": float(kept / total),
            "behavior_mass_evidence": float(mass / max(total_mass, 1e-12)),
            "support_median": float(np.median(values)),
            "support_p10": float(np.percentile(values, 10)),
        }
    return {
        "by_state": report,
        "structural_shift_ties": {symbol: {"states": int(n), "tie_rate": float(t / n)} for symbol, (t, n) in ties.items() if n},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path, help="the pair's pre-stage directory (post and pre)")
    parser.add_argument("--post-revision", required=True)
    parser.add_argument("--pre-revision", required=True)
    parser.add_argument("--untrained", type=Path)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--report-rows", type=int, default=2_000)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_rows = check_shards(args.base, args.scores, {"post": args.post_revision, "pre": args.pre_revision})
    table = TokenTable.from_tokenizer_json(args.tokenizer_json)
    structural = {token_id: table.symbols[token_id] for token_id in table.added}
    rows = iter_joined_rows(
        args.base, {args.name: args.scores}, untrained=load_untrained(args.untrained), max_rows=args.report_rows
    )
    report = {
        "name": args.name,
        "rows": total_rows,
        "revisions": {"post": args.post_revision, "pre": args.pre_revision},
        "evidence": evidence_report(rows, args.name, StateLabeler(table), structural),
    }
    write_json(report, args.output)
    print(json.dumps({"rows": total_rows, "evidence": report["evidence"]["by_state"].get("all")}, indent=2))


if __name__ == "__main__":
    main()
