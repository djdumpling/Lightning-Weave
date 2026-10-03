#!/usr/bin/env python3
"""Repeat complete shards cyclically; request a total that ends at a shard boundary.

Shards are hard-linked by default. ``--copy`` writes byte-identical copies instead, for filesystems without
hard links (e.g. Modal Volumes); the manifest is the same either way."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from itertools import cycle
from pathlib import Path

import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import read_manifest, staged_directory, write_json
from data_curation.composition import trainable_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--total-rows", required=True, type=int)
    parser.add_argument("--student-model-path", type=Path)
    parser.add_argument("--copy", action="store_true", help="copy shards instead of hard-linking them")
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
    if any(manifest[f"{role}_teacher_model"].get("model_type") == "mixture" for role in ("pre", "post")):
        raise ValueError("mixture-teacher targets are not supported; repeat a single-teacher target")
    place = shutil.copyfile if args.copy else os.link
    output_shards = []
    total_tokens = 0
    token_cache = {}
    with staged_directory(args.output_dir) as temporary:
        for index, (cycle_index, source_index, shard) in enumerate(
            repeated_shard_plan(manifest["shards"], total_rows=args.total_rows)
        ):
            source = args.manifest.parent / shard["path"]
            name = f"repeat-c{cycle_index:02d}-s{source_index:05d}-o{index:05d}.parquet"
            place(source, temporary / name)
            if shard["path"] not in token_cache:
                token_cache[shard["path"]] = trainable_tokens(pq.read_table(source, columns=["metadata.loss_mask"]))
            total_tokens += token_cache[shard["path"]]
            output_shards.append({**shard, "path": name})

        manifest["total_rows"] = sum(shard["rows"] for shard in output_shards)
        manifest["total_trainable_tokens"] = total_tokens
        manifest["shards"] = output_shards
        if args.student_model_path is not None:
            manifest["student_model"]["path"] = str(args.student_model_path.resolve())
        write_json(manifest, temporary / "manifest.json")
    print(f"REPEATED {args.output_dir} rows={manifest['total_rows']} tokens={total_tokens}")


if __name__ == "__main__":
    main()
