#!/usr/bin/env python3
"""Merge pre/post score columns from matching, row-aligned source shards."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import staged_directory, write_parquet
from data_curation.composition import replace_metadata_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--pre-scores", required=True, type=Path)
    parser.add_argument("--post-scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def merge_one_shard(task: tuple[Path, Path, Path, Path]) -> int:
    base_path, pre_path, post_path, output_path = task
    base = pq.read_table(base_path)
    replacements = {
        "is_offline_direct_opd": pa.array([True] * base.num_rows, type=pa.bool_()),
        "offline_direct_opd_stage": pa.array(["fully_scored"] * base.num_rows, type=pa.string()),
    }
    # Read only teacher columns, not another copy of the cached behavior payload.
    for role, path in (("pre", pre_path), ("post", post_path)):
        fields = [f"{role}_teacher_log_probs", f"{role}_teacher_revision"]
        scores = pq.read_table(path, columns=[f"metadata.{field}" for field in fields])
        replacements.update({field: scores[field].combine_chunks() for field in fields})
    write_parquet(replace_metadata_fields(base, replacements), output_path, compression="zstd", row_group_size=128)
    return base.num_rows


def main() -> None:
    args = parse_args()
    with staged_directory(args.output_dir) as temporary:
        tasks = [
            (path, args.pre_scores / path.name, args.post_scores / path.name, temporary / path.name)
            for path in sorted(args.base.glob("*.parquet"))
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = sum(executor.map(merge_one_shard, tasks))
    print(f"MERGED {args.output_dir} rows={rows} shards={len(tasks)}")


if __name__ == "__main__":
    main()
