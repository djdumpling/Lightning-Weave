#!/usr/bin/env python3
"""Build a sealed, group-preserving mixture from aligned Offline Direct-OPD datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import (
    canonical_hash,
    file_sha256,
    read_manifest,
    staged_directory,
    write_json,
    write_parquet,
)
from data_curation.composition import table_chunks, trainable_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-manifest", required=True, type=Path)
    parser.add_argument("--right-manifest", required=True, type=Path)
    parser.add_argument("--left-name", default="performance")
    parser.add_argument("--right-name", default="brevity")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=12_800)
    parser.add_argument("--rows-per-output-shard", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [args.left_manifest, args.right_manifest]
    manifests = [read_manifest(path) for path in paths]

    source_rows = [0, 0]
    total_tokens = 0
    shard_records = []
    streams = [
        table_chunks(path, manifest, rows=args.rows, chunk_rows=args.rows_per_output_shard)
        for path, manifest in zip(paths, manifests)
    ]
    with staged_directory(args.output_dir) as temporary:
        global_group = 0
        for shard_index, tables in enumerate(zip(*streams)):
            pieces = []
            for offset in range(0, tables[0].num_rows, args.group_size):
                component = global_group % 2
                pieces.append(tables[component].slice(offset, args.group_size))
                source_rows[component] += args.group_size
                global_group += 1
            mixed = pa.concat_tables(pieces)
            output = temporary / f"mix-{shard_index:05d}.parquet"
            write_parquet(mixed, output, compression="zstd", row_group_size=args.rows_per_output_shard)
            total_tokens += trainable_tokens(mixed)
            shard_records.append({"path": output.name, "rows": mixed.num_rows, "sha256": file_sha256(output)})

        components = [
            {
                "name": name,
                "weight": 0.5,
                "rows": rows,
                "groups": rows // args.group_size,
                "model": manifest["post_teacher_model"],
                "source_manifest": str(path.resolve()),
                "source_manifest_sha256": file_sha256(path),
            }
            for name, rows, manifest, path in zip((args.left_name, args.right_name), source_rows, manifests, paths)
        ]
        selection = {
            "unit": "prompt_group",
            "strategy": "round_robin",
            "group_size": args.group_size,
            "rows_per_output_shard": args.rows_per_output_shard,
            "rollout_batch_size": args.rollout_batch_size,
            "rows_per_component_per_rollout": args.rollout_batch_size // 2,
            "duplicate_trajectories": False,
        }
        revision = canonical_hash({"components": components, "selection": selection})
        manifest = {
            **manifests[0],
            "post_teacher_model": {
                "model_type": "mixture",
                "revision": revision,
                "components": components,
                "selection": selection,
            },
            "total_rows": sum(source_rows),
            "total_trainable_tokens": total_tokens,
            "shards": shard_records,
        }
        manifest_path = temporary / "manifest.json"
        write_json(manifest, manifest_path)
    print(f"SEALED_MIXTURE {args.output_dir} rows={args.rows} tokens={total_tokens} revision={revision}")


if __name__ == "__main__":
    main()
