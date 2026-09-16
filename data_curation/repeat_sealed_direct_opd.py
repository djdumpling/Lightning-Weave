#!/usr/bin/env python3
"""Repeat complete shards cyclically; request a total that ends at a shard boundary."""

from __future__ import annotations

import argparse
import os
import sys
from itertools import cycle
from pathlib import Path

import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import canonical_hash, read_manifest, staged_directory, write_json
from data_curation.composition import trainable_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--total-rows", required=True, type=int)
    parser.add_argument("--student-model-path", type=Path)
    return parser.parse_args()


def repeated_shard_plan(shards, *, total_rows):
    emitted = 0
    for index, shard in enumerate(cycle(shards)):
        if emitted >= total_rows:
            return
        yield index // len(shards), index % len(shards), shard
        emitted += shard["rows"]


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.manifest)
    output_shards = []
    total_tokens = 0
    token_cache = {}
    with staged_directory(args.output_dir) as temporary:
        for index, (cycle_index, source_index, shard) in enumerate(
            repeated_shard_plan(manifest["shards"], total_rows=args.total_rows)
        ):
            source = args.manifest.parent / shard["path"]
            name = f"repeat-c{cycle_index:02d}-s{source_index:05d}-o{index:05d}.parquet"
            os.link(source, temporary / name)
            if shard["path"] not in token_cache:
                token_cache[shard["path"]] = trainable_tokens(pq.read_table(source, columns=["metadata.loss_mask"]))
            total_tokens += token_cache[shard["path"]]
            output_shards.append({**shard, "path": name})

        manifest["total_rows"] = sum(shard["rows"] for shard in output_shards)
        manifest["total_trainable_tokens"] = total_tokens
        manifest["shards"] = output_shards
        if args.student_model_path is not None:
            manifest["student_model"]["path"] = str(args.student_model_path.resolve())
        for role in ("pre", "post"):
            model = manifest[f"{role}_teacher_model"]
            if model.get("model_type") == "mixture":
                rows = manifest["total_rows"] // len(model["components"])
                for component in model["components"]:
                    component["rows"] = rows
                    component["groups"] = rows // model["selection"]["group_size"]
                model["selection"]["duplicate_trajectories"] = True
                model["revision"] = canonical_hash(
                    {
                        "teacher_role": role,
                        "components": model["components"],
                        "selection": model["selection"],
                    }
                )
        write_json(manifest, temporary / "manifest.json")
    print(f"REPEATED {args.output_dir} rows={manifest['total_rows']} tokens={total_tokens}")


if __name__ == "__main__":
    main()
