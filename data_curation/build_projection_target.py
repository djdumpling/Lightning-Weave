#!/usr/bin/env python3
"""Seal one decision-projection arm: the recipient's own rollouts, each carrying its weight (CPU only).

The rows keep the recipient's tokens, candidates and behavior log-probabilities exactly as collected. Each gets
``metadata.sequence_weight`` from ``data_curation/decision_projection.py weights`` for one arm (uniform, ordinary or
projected), which the ``sequence_weighted`` loss fits. That loss reads no teacher shift, so the schema's teacher fields
hold the behavior log-probabilities themselves (a zero shift) and the reference field the behavior's sampled
log-probabilities. The manifest locks the recipient as the student, so training must start from that checkpoint.

    python data_curation/build_projection_target.py --rollouts DIR --weights W.parquet --arm projected \
        --recipient PATH --recipient-revision REV --asset-lock assets.json --source-dataset prompts.parquet \
        --output-dir OUT
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import (
    canonical_hash,
    file_sha256,
    parquet_paths,
    staged_directory,
    write_json,
    write_parquet,
)
from data_curation.composition import flatten_metadata
from data_curation.decision_projection import ARMS
from data_curation.precompute_direct_opd_scores import cast_offline_direct_opd_storage
from slime.rollout.offline_direct_opd import (
    SCHEMA_VERSION,
    SEQUENCE_WEIGHT_FIELD,
    validate_offline_direct_opd_metadata,
)

STAGE = "decision_projection"
WEIGHTS_SCHEMA = "decision_projection_weights_v1"


def read_weights(path: Path, arm: str) -> tuple[dict[str, float], dict]:
    table = pq.read_table(path)
    metadata = {key.decode(): value.decode() for key, value in (table.schema.metadata or {}).items()}
    if metadata.get("schema") != WEIGHTS_SCHEMA:
        raise ValueError(f"{path} is not a {WEIGHTS_SCHEMA} file")
    weights = dict(zip(map(str, table["sample_id"].to_pylist()), table[f"weight_{arm}"].to_pylist(), strict=True))
    if len(weights) != table.num_rows:
        raise ValueError(f"{path} lists a sample_id twice")
    bad = [sample for sample, weight in weights.items() if not (math.isfinite(weight) and weight >= 0)]
    if bad:
        raise ValueError(f"{len(bad)} weights are negative or non-finite, e.g. {bad[:3]}")
    return weights, metadata


def with_metadata(table: pa.Table, replacements: dict[str, pa.Array]) -> pa.Table:
    """Replace or add top-level fields of the ``metadata`` struct column."""
    metadata = flatten_metadata(table)
    names = [field.name for field in metadata.type]
    arrays = [replacements.get(name, metadata.field(name)) for name in names]
    arrays += [array for name, array in replacements.items() if name not in names]
    names += [name for name in replacements if name not in names]
    rebuilt = pa.StructArray.from_arrays(arrays, names=names)
    return table.set_column(table.schema.get_field_index("metadata"), "metadata", rebuilt)


def projection_rows(table: pa.Table, weights: dict[str, float], revision: str) -> pa.Table:
    metadata = flatten_metadata(table)
    sample_ids = [str(value) for value in metadata.field("sample_id").to_pylist()]
    missing = [sample for sample in sample_ids if sample not in weights]
    if missing:
        raise ValueError(f"{len(missing)} rollouts have no weight, e.g. {missing[:3]}")
    rows = table.num_rows
    behavior = metadata.field("behavior_topk_log_probs")
    return with_metadata(
        table,
        {
            "is_offline_direct_opd": pa.array([True] * rows),
            "offline_direct_opd_stage": pa.array([STAGE] * rows),
            "post_teacher_log_probs": behavior,
            "pre_teacher_log_probs": behavior,
            "student_ref_sampled_log_probs": metadata.field("behavior_sampled_log_probs"),
            "post_teacher_revision": pa.array([revision] * rows),
            "pre_teacher_revision": pa.array([revision] * rows),
            SEQUENCE_WEIGHT_FIELD: pa.array([weights[sample] for sample in sample_ids], type=pa.float64()),
        },
    )


def check_rows(table: pa.Table, top_k: int) -> None:
    """Every row's shapes, cheaply; two rows per shard through the exhaustive validator."""
    metadata = flatten_metadata(table)
    lengths = metadata.field("response_length")
    for field in ("response_tokens", "loss_mask", "candidate_ids", "behavior_topk_log_probs",
                  "behavior_sampled_log_probs"):
        if not pc.all(pc.equal(pc.list_value_length(metadata.field(field)), lengths)).as_py():
            raise ValueError(f"{field} does not match response_length in some row")
    for index in {0, table.num_rows - 1}:
        validate_offline_direct_opd_metadata(table.slice(index, 1).to_pylist()[0]["metadata"], expected_top_k=top_k,
                                             require_provenance=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--recipient", required=True, help="the checkpoint that sampled the rollouts")
    parser.add_argument("--recipient-revision", required=True)
    parser.add_argument("--asset-lock", type=Path, required=True, help="tokenizer and vocabulary of the recipient")
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, help="JSON recorded in the manifest (calibration, scores, ...)")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    weights, weight_metadata = read_weights(args.weights, args.arm)
    shards = parquet_paths(args.rollouts)
    if not shards:
        raise FileNotFoundError(f"no rollout shards under {args.rollouts}")
    asset_lock = json.loads(args.asset_lock.read_text(encoding="utf-8"))
    provenance = json.loads(args.provenance.read_text(encoding="utf-8")) if args.provenance else {}
    revision = canonical_hash(
        {
            "stage": STAGE,
            "arm": args.arm,
            "weights_sha256": file_sha256(args.weights),
            "rollout_sha256": [file_sha256(path) for path in shards],
            "recipient_revision": args.recipient_revision,
        }
    )
    first = flatten_metadata(pq.read_table(shards[0], columns=["metadata"])).slice(0, 1).to_pylist()[0]
    generation_config, generation_hash = first["generation_config"], first["generation_config_hash"]
    if first["student_revision"] != args.recipient_revision:
        raise ValueError(f"rollouts were sampled by {first['student_revision']}, not {args.recipient_revision}")
    seen: set[str] = set()
    records = []
    with staged_directory(args.output_dir) as staging:
        for index, path in enumerate(shards):
            table = pq.read_table(path)
            metadata = flatten_metadata(table)
            if set(metadata.field("generation_config_hash").to_pylist()) != {generation_hash}:
                raise ValueError(f"{path} mixes generation configs")
            if set(metadata.field("student_revision").to_pylist()) != {args.recipient_revision}:
                raise ValueError(f"{path} holds rollouts from another sampler")
            ids = [str(value) for value in metadata.field("sample_id").to_pylist()]
            if seen.intersection(ids) or len(set(ids)) != len(ids):
                raise ValueError(f"{path} repeats a sample_id")
            seen.update(ids)
            sealed = cast_offline_direct_opd_storage(projection_rows(table, weights, revision))
            check_rows(sealed, generation_config["top_k"])
            output = staging / f"projection-{index:05d}.parquet"
            write_parquet(sealed, output, compression="zstd")
            records.append({"path": output.name, "rows": sealed.num_rows, "sha256": file_sha256(output)})
        unused = set(weights) - seen
        if unused:
            raise ValueError(f"{len(unused)} weights match no rollout, e.g. {sorted(unused)[:3]}")
        student = asset_lock["models"]["student"]
        reference = {"model_type": "behavior", "revision": revision, "note": "the recipient's own log-probabilities"}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "sealed": True,
            "top_k": generation_config["top_k"],
            "student_model": {**student, "path": args.recipient, "revision": args.recipient_revision},
            "post_teacher_model": {
                "model_type": STAGE,
                "revision": revision,
                "arm": args.arm,
                "weights_sha256": file_sha256(args.weights),
                "weights": weight_metadata,
                "provenance": provenance,
            },
            "pre_teacher_model": reference,
            "asset_lock_sha256": file_sha256(args.asset_lock),
            "tokenizer_hash": first["tokenizer_hash"],
            "model_vocab_size": student["model_vocab_size"],
            "token_id_compatibility": asset_lock["token_id_compatibility"],
            "generation_config": generation_config,
            "generation_config_hash": generation_hash,
            "loss_mask_storage_dtype": "bool",
            "score_storage_dtype": "float32",
            "token_storage_dtype": "int32",
            "source_dataset_sha256": file_sha256(args.source_dataset),
            "total_rows": sum(record["rows"] for record in records),
            "shards": records,
        }
        write_json(manifest, staging / "manifest.json")
    print(f"sealed {manifest['total_rows']} {args.arm} rows at {args.output_dir} (revision {revision[:12]})")


if __name__ == "__main__":
    main()
