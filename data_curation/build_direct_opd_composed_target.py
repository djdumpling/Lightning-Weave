#!/usr/bin/env python3
"""Compose aligned cached targets using sum_j weight_j * (log p_post_j - log p_pre_j).

Weights are raw, not normalized. ``weighted_post_log_prob_sum`` omits the
reference terms. Both rules encode the resulting signed shift as the
difference of finite, non-positive scores in the existing Direct-OPD schema.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
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
from data_curation.composition import flatten_metadata, replace_metadata_fields, table_chunks, trainable_tokens

COMPOSITION_SCHEMA_VERSION = "offline_direct_opd_multi_anchor_composition_v1"
COMPOSITION_RULES = {"weighted_log_density_ratio_sum", "weighted_post_log_prob_sum"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-manifest", action="append", required=True, type=Path)
    parser.add_argument("--anchor-name", action="append", required=True)
    parser.add_argument("--anchor-weight", action="append", required=True, type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=12_800)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--rows-per-output-shard", type=int, default=128)
    parser.add_argument(
        "--composition-rule", choices=sorted(COMPOSITION_RULES), default="weighted_log_density_ratio_sum"
    )
    return parser.parse_args()


def compose_anchor_deltas(
    pre_log_probs: list[np.ndarray],
    post_log_probs: list[np.ndarray],
    weights: list[float],
    *,
    composition_rule: str = "weighted_log_density_ratio_sum",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return synthetic pre/post scores and their exact weighted delta."""
    delta = np.zeros_like(pre_log_probs[0], dtype=np.float64)
    for pre, post, weight in zip(pre_log_probs, post_log_probs, weights):
        pre, post = np.asarray(pre, dtype=np.float64), np.asarray(post, dtype=np.float64)
        delta += float(weight) * (post - pre if composition_rule == "weighted_log_density_ratio_sum" else post)
    synthetic_post = np.minimum(delta, 0.0).astype(np.float32)
    synthetic_pre = np.minimum(-delta, 0.0).astype(np.float32)
    return synthetic_pre, synthetic_post, delta.astype(np.float32)


def nested_score_array(rows: list[np.ndarray], arrow_type: pa.DataType) -> pa.Array:
    """Build Arrow lists directly from buffers, avoiding millions of Python floats."""
    width = rows[0].shape[1]
    flat_rows = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
    values = pa.array(flat_rows.reshape(-1), type=arrow_type.value_type.value_type, from_pandas=False)
    token_offsets = np.arange(flat_rows.shape[0] + 1, dtype=np.int32) * width
    tokens = pa.ListArray.from_arrays(token_offsets, values, type=arrow_type.value_type)
    row_offsets = np.zeros(len(rows) + 1, dtype=np.int32)
    np.cumsum([row.shape[0] for row in rows], out=row_offsets[1:])
    return pa.ListArray.from_arrays(row_offsets, tokens, type=arrow_type)


def replace_score_fields(
    table: pa.Table,
    *,
    pre_rows: list[np.ndarray],
    post_rows: list[np.ndarray],
    loss_masks: list[list[bool]],
    projection_masks: list[list[bool]] | None,
    revision: str,
) -> pa.Table:
    metadata = flatten_metadata(table)
    replacements = {
        field: nested_score_array(rows, metadata.type.field(field).type)
        for field, rows in (("pre_teacher_log_probs", pre_rows), ("post_teacher_log_probs", post_rows))
    }
    values = {
        "loss_mask": loss_masks,
        "pre_teacher_revision": [revision] * table.num_rows,
        "post_teacher_revision": [revision] * table.num_rows,
    }
    if projection_masks is not None:
        values["token_projection_valid_mask"] = projection_masks
    replacements.update(
        {field: pa.array(rows, type=metadata.type.field(field).type) for field, rows in values.items()}
    )
    return replace_metadata_fields(table, replacements)


def compose_tables(tables: Sequence[pa.Table], weights: list[float], *, rule: str, revision: str) -> pa.Table:
    metadata = [flatten_metadata(table) for table in tables]
    pre_rows, post_rows, loss_masks = [], [], []
    projection_masks = [] if "token_projection_valid_mask" in metadata[0].type.names else None
    for row_index in range(tables[0].num_rows):
        pre_scores, post_scores = (
            [np.asarray(item.field(field)[row_index].as_py(), dtype=np.float32) for item in metadata]
            for field in ("pre_teacher_log_probs", "post_teacher_log_probs")
        )
        pre, post, _ = compose_anchor_deltas(pre_scores, post_scores, weights, composition_rule=rule)
        pre_rows.append(pre)
        post_rows.append(post)
        anchor_masks = [np.asarray(item.field("loss_mask")[row_index].as_py(), dtype=bool) for item in metadata]
        # Zero-weight anchors still participate in mask intersection.
        mask = np.logical_and.reduce(anchor_masks)
        loss_masks.append(mask.tolist())
        if projection_masks is not None:
            masks = []
            for item, anchor_mask in zip(metadata, anchor_masks):
                projected = (
                    np.asarray(item.field("token_projection_valid_mask")[row_index].as_py(), dtype=bool)
                    if "token_projection_valid_mask" in item.type.names
                    else np.ones_like(anchor_mask)
                )
                masks.append(projected)
            projection_masks.append(np.logical_and.reduce(masks).tolist())
    return replace_score_fields(
        tables[0],
        pre_rows=pre_rows,
        post_rows=post_rows,
        loss_masks=loss_masks,
        projection_masks=projection_masks,
        revision=revision,
    )


def main() -> None:
    args = parse_args()
    count = len(args.anchor_manifest)
    manifests = [read_manifest(path) for path in args.anchor_manifest]
    reference = manifests[0]
    anchors = [
        {
            "name": name,
            "weight": float(weight),
            "pre_teacher_model": manifest["pre_teacher_model"],
            "post_teacher_model": manifest["post_teacher_model"],
        }
        for name, weight, manifest in zip(args.anchor_name, args.anchor_weight, manifests)
    ]
    composition_spec = {
        "schema_version": COMPOSITION_SCHEMA_VERSION,
        "rule": args.composition_rule,
        "anchors": anchors,
        "source_rows": args.rows,
        "repeat": args.repeat,
    }
    revision = canonical_hash(composition_spec)
    streams = [
        table_chunks(path, manifest, rows=args.rows, chunk_rows=args.rows_per_output_shard)
        for path, manifest in zip(args.anchor_manifest, manifests)
    ]
    source_rows = source_tokens = 0
    base_shards = []
    with staged_directory(args.output_dir) as temporary:
        for shard_index, tables in enumerate(zip(*streams)):
            composed = compose_tables(tables, args.anchor_weight, rule=args.composition_rule, revision=revision)
            output = temporary / f"composed-c00-{shard_index:05d}.parquet"
            write_parquet(composed, output, compression="zstd", row_group_size=args.rows_per_output_shard)
            base_shards.append({"path": output.name, "rows": composed.num_rows, "sha256": file_sha256(output)})
            source_rows += composed.num_rows
            source_tokens += trainable_tokens(composed)
        shards = []
        for cycle in range(args.repeat):
            for shard_index, base in enumerate(base_shards):
                name = f"composed-c{cycle:02d}-{shard_index:05d}.parquet"
                if cycle:
                    os.link(temporary / base["path"], temporary / name)
                shards.append({**base, "path": name})
        model = {
            "revision": revision,
            "composition_schema_version": COMPOSITION_SCHEMA_VERSION,
            "rule": args.composition_rule,
        }
        manifest = {
            **reference,
            "post_teacher_model": {**model, "model_type": "multi_anchor_composition", "anchors": anchors},
            "pre_teacher_model": {
                **model,
                "model_type": (
                    "multi_anchor_composition_reference"
                    if args.composition_rule == "weighted_log_density_ratio_sum"
                    else "zero_log_score_reference"
                ),
            },
            "composition_schedule": {"source_rows": args.rows, "repeat": args.repeat},
            "total_rows": source_rows * args.repeat,
            "total_trainable_tokens": source_tokens * args.repeat,
            "shards": shards,
        }
        path = temporary / "manifest.json"
        write_json(manifest, path)
    print(
        f"SEALED_MULTI_ANCHOR_COMPOSITION {args.output_dir} anchors={count} "
        f"rows={manifest['total_rows']} tokens={manifest['total_trainable_tokens']} "
        f"rule={args.composition_rule} revision={revision}"
    )


if __name__ == "__main__":
    main()
