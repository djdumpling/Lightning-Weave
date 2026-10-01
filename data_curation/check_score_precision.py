#!/usr/bin/env python3
"""Does a pair's scored shift survive different numerics? Compare it with a re-score of the same rows.

The main scores come from bfloat16 with flash-attention-2. The check re-scores a few rows of one cache shard
with other numerics (float32 and SDPA attention in the Modal stage). A shift that is mostly rounding noise
changes under the re-score; a real one does not. Both scores are read through the composer's own path
(``iter_joined_rows`` and ``direction_vectors``): candidates without evidence carry no shift, and each position is
centered under the behavior policy over its K+1 buckets. The shifts are compared over the base loss mask:

- ``fisher_correlation``: Σ b·u·v / sqrt(Σ b·u² · Σ b·v²), with u the main and v the check shift;
- ``relative_rms_error``: sqrt(Σ b·(u − v)² / Σ b·u²);
- the RMS sizes of both, and the RMS and maximum difference of each anchor's log-probabilities on candidates
  both runs can score.

    python data_curation/check_score_precision.py --base CACHE --main DONOR/pre --check DONOR/precision/pre \\
        --name decs --untrained untrained.json --output precision.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import write_json
from data_curation.shift_geometry import (
    Direction,
    bucket_probs,
    direction_vectors,
    iter_joined_rows,
    load_untrained,
    read_shard_fields,
)


def aligned_subset(base: Path, main: Path, check: Path, directory: Path) -> str:
    """Write the base and main rows of ``check``'s one shard, in its row order, as three aligned directories."""
    shards = sorted(check.glob("*.parquet"))
    if len(shards) != 1:
        raise ValueError(f"{check} must hold exactly one re-scored shard, found {len(shards)}")
    name = shards[0].name
    wanted = read_shard_fields(shards[0], ("sample_id",))["sample_id"]
    if not wanted or len(set(wanted)) != len(wanted):
        raise ValueError(f"{shards[0]} has no rows or repeats a sample_id")
    for label, source in (("base", base), ("main", main)):
        ids = read_shard_fields(source / name, ("sample_id",))["sample_id"]
        position = {sample_id: index for index, sample_id in enumerate(ids)}
        missing = [sample_id for sample_id in wanted if sample_id not in position]
        if missing:
            raise KeyError(f"{source / name} lacks {len(missing)} re-scored rows")
        (directory / label).mkdir(parents=True)
        table = pq.read_table(source / name).take([position[sample_id] for sample_id in wanted])
        pq.write_table(table, directory / label / name)
    (directory / "check").mkdir()
    (directory / "check" / name).symlink_to(shards[0].resolve())
    return name


def compare(base: Path, main: Path, check: Path, *, untrained: set[int] | None = None) -> dict:
    sums = dict.fromkeys(("uv", "uu", "vv", "dd"), 0.0)
    anchor = {role: {"squares": 0.0, "max": 0.0, "count": 0} for role in ("post", "pre")}
    rows = positions = 0
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        name = aligned_subset(base, main, check, root)
        sources = {"main": root / "main", "check": root / "check"}
        blocked = {source: set(untrained or ()) for source in sources}
        directions = [Direction("main", (("main", 1.0),)), Direction("check", (("check", 1.0),))]
        fields = ("post_teacher_log_probs", "pre_teacher_log_probs")
        scores = {source: read_shard_fields(path / name, fields) for source, path in sources.items()}
        for index, row in enumerate(iter_joined_rows(root / "base", sources, untrained=blocked)):
            probs = bucket_probs(row.behavior_log_probs)
            vectors = direction_vectors(row, directions, probs)[row.loss_mask]
            weights = probs[row.loss_mask]
            u, v = vectors[..., 0], vectors[..., 1]
            sums["uv"] += float((weights * u * v).sum())
            sums["uu"] += float((weights * u * u).sum())
            sums["vv"] += float((weights * v * v).sum())
            sums["dd"] += float((weights * (u - v) ** 2).sum())
            rows += 1
            positions += int(row.loss_mask.sum())
            both = (row.valid("main") & row.valid("check"))[row.loss_mask]
            for role, field in zip(("post", "pre"), fields, strict=True):
                main_scores = np.asarray(scores["main"][field][index], dtype=np.float64)
                check_scores = np.asarray(scores["check"][field][index], dtype=np.float64)
                difference = np.abs(main_scores - check_scores)[row.loss_mask][both]
                anchor[role]["squares"] += float((difference**2).sum())
                # np.max propagates NaN, so a non-finite score on a valid candidate stays visible
                anchor[role]["max"] = float(np.max([anchor[role]["max"], difference.max(initial=0.0)]))
                anchor[role]["count"] += int(difference.size)
    if not positions or sums["uu"] <= 0 or sums["vv"] <= 0:
        raise ValueError("no scored positions with a shift to compare")
    return {
        "rows": rows,
        "positions": positions,
        "fisher_correlation": sums["uv"] / float(np.sqrt(sums["uu"] * sums["vv"])),
        "relative_rms_error": float(np.sqrt(sums["dd"] / sums["uu"])),
        "shift_rms": {"main": float(np.sqrt(sums["uu"] / positions)), "check": float(np.sqrt(sums["vv"] / positions))},
        "log_prob_difference": {
            role: {"rms": float(np.sqrt(item["squares"] / max(item["count"], 1))), "max": item["max"]}
            for role, item in anchor.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path, help="sealed base cache")
    parser.add_argument("--main", required=True, type=Path, help="the pair's scored chain (its pre stage)")
    parser.add_argument("--check", required=True, type=Path, help="the re-scored chain: one shard's subset")
    parser.add_argument("--name", required=True, help="the pair's name in --untrained")
    parser.add_argument("--untrained", type=Path)
    parser.add_argument("--settings", default="{}", help="JSON describing the check's numerics, recorded as is")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    untrained = set()
    if args.untrained:
        lists = load_untrained(args.untrained)
        if args.name not in lists:
            raise KeyError(f"{args.untrained} has no untrained list for {args.name!r}")
        untrained = lists[args.name]
    report = {"name": args.name, "settings": json.loads(args.settings)}
    report.update(compare(args.base, args.main, args.check, untrained=untrained))
    write_json(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
