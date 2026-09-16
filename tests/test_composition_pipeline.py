"""Small real-parquet regressions for composition, mixing, repeating, and merging."""

from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_curation import build_direct_opd_composed_target as compose
from data_curation import build_direct_opd_mixture as mixture
from data_curation import merge_direct_opd_score_columns as merge
from data_curation import repeat_sealed_direct_opd as repeat
from data_curation.common import file_sha256, write_json
from data_curation.composition import flatten_metadata, replace_metadata_fields
from slime.rollout.offline_direct_opd import SCHEMA_VERSION, validate_sealed_manifest


def source_table(rows=16, *, pre=-2.0, post=-1.0, post_revision="post", mask=(True, True), projection=None):
    scores = pa.list_(pa.list_(pa.float32()))
    values = {
        "sample_id": [f"s{index}" for index in range(rows)],
        "group_id": [index // 4 for index in range(rows)],
        "response_id": [index % 4 for index in range(rows)],
        "loss_mask": [list(mask)] * rows,
        "pre_teacher_log_probs": [[[pre, pre - 1], [pre - 2, pre - 3]]] * rows,
        "post_teacher_log_probs": [[[post, post - 1], [post - 2, post - 3]]] * rows,
        "pre_teacher_revision": ["pre"] * rows,
        "post_teacher_revision": [post_revision] * rows,
        "is_offline_direct_opd": [True] * rows,
        "offline_direct_opd_stage": ["fully_scored"] * rows,
    }
    if projection is not None:
        values["token_projection_valid_mask"] = [list(projection)] * rows
        values["loss_mask_before_token_projection"] = [[True, True]] * rows
    metadata = pa.StructArray.from_arrays(
        [pa.array(value, type=scores if field.endswith("_log_probs") else None) for field, value in values.items()],
        names=list(values),
    )
    return pa.table(
        {
            "prompt": [f"prompt-{index // 4}" for index in range(rows)],
            "label": ["label"] * rows,
            "metadata": metadata,
        }
    )


def sealed_source(tmp_path, name, *, row_group_rows=4, shard_rows=8, **table_kwargs):
    root = tmp_path / name
    root.mkdir()
    table = source_table(**table_kwargs)
    shards = []
    for index, offset in enumerate(range(0, table.num_rows, shard_rows)):
        path = root / f"part-{index:03d}.parquet"
        shard = table.slice(offset, shard_rows)
        pq.write_table(shard, path, row_group_size=row_group_rows)
        shards.append({"path": path.name, "rows": shard.num_rows, "sha256": file_sha256(path)})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sealed": True,
        "top_k": 2,
        "student_model": {"revision": "student", "path": "/models/student"},
        "pre_teacher_model": {"revision": "pre"},
        "post_teacher_model": {"revision": table_kwargs.get("post_revision", "post")},
        "asset_lock_sha256": "a" * 64,
        "tokenizer_hash": "tokenizer",
        "model_vocab_size": 32,
        "generation_config": {"responses_per_prompt": 4},
        "generation_config_hash": "g" * 64,
        "loss_mask_storage_dtype": "bool",
        "score_storage_dtype": "float32",
        "token_storage_dtype": "int32",
        "source_dataset_sha256": "s" * 64,
        "total_rows": table.num_rows,
        "total_trainable_tokens": table.num_rows * sum(table_kwargs.get("mask", (True, True))),
        "shards": shards,
    }
    path = root / "manifest.json"
    write_json(manifest, path)
    return path


def read_output(path):
    manifest = validate_sealed_manifest(path / "manifest.json")
    table = pa.concat_tables([pq.read_table(path / item["path"]) for item in manifest["shards"]])
    return manifest, flatten_metadata(table)


def composition_args(left, right, output, **kwargs):
    return Namespace(
        **{
            "anchor_manifest": [left, right],
            "anchor_name": ["left", "right"],
            "anchor_weight": [2.0, 0.0],
            "output_dir": output,
            "rows": 16,
            "repeat": 2,
            "rows_per_output_shard": 8,
            "composition_rule": "weighted_log_density_ratio_sum",
            **kwargs,
        }
    )


@pytest.mark.parametrize("rule", sorted(compose.COMPOSITION_RULES))
def test_composition_preserves_raw_weights_masks_chunks_and_repeat(tmp_path, monkeypatch, rule):
    left = sealed_source(tmp_path, "left", pre=-3.0, post=-1.0, projection=(True, True))
    right = sealed_source(
        tmp_path,
        "right",
        pre=-5.0,
        post=-4.0,
        row_group_rows=8,
        shard_rows=16,
        mask=(True, False),
        projection=(True, False),
    )
    output = tmp_path / "composed"
    monkeypatch.setattr(compose, "parse_args", lambda: composition_args(left, right, output, composition_rule=rule))
    compose.main()
    manifest, metadata = read_output(output)
    assert manifest["total_rows"] == 32
    assert manifest["total_trainable_tokens"] == 32
    assert [item["rows"] for item in manifest["shards"]] == [8] * 4
    assert metadata.field("sample_id").to_pylist() == [f"s{index}" for index in range(16)] * 2
    assert metadata.field("loss_mask").to_pylist() == [[True, False]] * 32
    assert metadata.field("token_projection_valid_mask").to_pylist() == [[True, False]] * 32
    before = np.asarray(metadata.field("pre_teacher_log_probs")[0].as_py())
    after = np.asarray(metadata.field("post_teacher_log_probs")[0].as_py())
    expected = np.full((2, 2), 4.0) if rule == "weighted_log_density_ratio_sum" else np.array([[-2, -4], [-6, -8]])
    np.testing.assert_allclose(after - before, expected)
    assert metadata.type.field("pre_teacher_log_probs").type.value_type.value_type == pa.float32()
    assert manifest["composition_schedule"] == {"source_rows": 16, "repeat": 2}
    assert [item["weight"] for item in manifest["post_teacher_model"]["anchors"]] == [2.0, 0.0]
    assert set(metadata.field("pre_teacher_revision").to_pylist()) == {manifest["pre_teacher_model"]["revision"]}
    for first, second in zip(manifest["shards"][:2], manifest["shards"][2:], strict=True):
        assert first["sha256"] == second["sha256"]
        assert (output / first["path"]).stat().st_ino == (output / second["path"]).stat().st_ino
    assert sorted(path.suffix for path in output.iterdir()) == [".json"] + [".parquet"] * 4


def test_zero_weight_anchor_still_participates_in_empty_mask_intersection(tmp_path, monkeypatch):
    left = sealed_source(tmp_path, "left", mask=(True, False))
    right = sealed_source(tmp_path, "right", mask=(False, True))
    output = tmp_path / "composed"
    monkeypatch.setattr(compose, "parse_args", lambda: composition_args(left, right, output))
    compose.main()
    manifest, metadata = read_output(output)
    assert manifest["total_trainable_tokens"] == 0
    assert metadata.field("loss_mask").to_pylist() == [[False, False]] * 32


def test_composition_discards_staged_shards_after_write_failure(tmp_path, monkeypatch):
    left, right = sealed_source(tmp_path, "left"), sealed_source(tmp_path, "right")
    output = tmp_path / "composed"
    monkeypatch.setattr(compose, "parse_args", lambda: composition_args(left, right, output))
    original_write = compose.write_parquet
    calls = 0

    def fail_second_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected failure")
        original_write(*args, **kwargs)

    monkeypatch.setattr(compose, "write_parquet", fail_second_write)
    with pytest.raises(OSError, match="injected failure"):
        compose.main()
    assert calls == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".composed.*"))


def test_mixture_and_repetition_preserve_prompt_groups(tmp_path, monkeypatch):
    left = sealed_source(tmp_path, "left", post_revision="left", post=-1.0)
    right = sealed_source(tmp_path, "right", post_revision="right", post=-1.5)
    output = tmp_path / "mixed"
    monkeypatch.setattr(
        mixture,
        "parse_args",
        lambda: Namespace(
            left_manifest=left,
            right_manifest=right,
            left_name="left",
            right_name="right",
            output_dir=output,
            rows=8,
            rows_per_output_shard=4,
            group_size=4,
            rollout_batch_size=8,
        ),
    )
    mixture.main()
    manifest, metadata = read_output(output)
    assert metadata.field("sample_id").to_pylist() == [f"s{index}" for index in range(8)]
    assert metadata.field("post_teacher_revision").to_pylist() == ["left"] * 4 + ["right"] * 4
    assert [item["rows"] for item in manifest["post_teacher_model"]["components"]] == [4, 4]
    repeated = tmp_path / "repeated"
    monkeypatch.setattr(
        repeat,
        "parse_args",
        lambda: Namespace(
            manifest=output / "manifest.json",
            output_dir=repeated,
            total_rows=16,
            student_model_path=None,
        ),
    )
    repeat.main()
    repeated_manifest, repeated_metadata = read_output(repeated)
    assert repeated_manifest["total_trainable_tokens"] == 32
    assert repeated_metadata.field("sample_id").to_pylist() == [f"s{index}" for index in range(8)] * 2
    assert repeated_metadata.field("post_teacher_revision").to_pylist() == (["left"] * 4 + ["right"] * 4) * 2
    assert [item["rows"] for item in repeated_manifest["post_teacher_model"]["components"]] == [8, 8]
    for index, shard in enumerate(repeated_manifest["shards"]):
        source = output / manifest["shards"][index % 2]["path"]
        assert (repeated / shard["path"]).stat().st_ino == source.stat().st_ino


def test_repeated_plan_preserves_partial_cycle():
    shards = [{"path": "a", "rows": 4}, {"path": "b", "rows": 8}]
    plan = repeat.repeated_shard_plan(shards, total_rows=16)
    assert [(cycle, index) for cycle, index, _ in plan] == [(0, 0), (0, 1), (1, 0)]


def test_chunking_streams_a_prefix_within_a_source_row_group(tmp_path):
    path = sealed_source(tmp_path, "source", row_group_rows=8)
    manifest = validate_sealed_manifest(path)
    chunks = list(compose.table_chunks(path, manifest, rows=4, chunk_rows=4))
    assert len(chunks) == 1
    assert flatten_metadata(chunks[0]).field("sample_id").to_pylist() == ["s0", "s1", "s2", "s3"]


def test_merge_updates_only_teacher_fields_and_preserves_order(tmp_path):
    tables = [
        source_table(rows=4, pre=-3, post=-4, post_revision="base"),
        source_table(rows=4, pre=-2, post=-4, post_revision="base"),
        source_table(rows=4, pre=-3, post=-1, post_revision="post"),
    ]
    paths = [tmp_path / f"{name}.parquet" for name in ("base", "pre", "post")]
    for table, path in zip(tables, paths, strict=True):
        pq.write_table(table, path)
    output = tmp_path / "merged.parquet"
    count = merge.merge_one_shard((*paths, output))
    merged = pq.read_table(output)
    metadata = flatten_metadata(merged)
    assert count == 4
    assert metadata.field("sample_id").to_pylist() == [f"s{index}" for index in range(4)]
    for field in ("prompt", "label"):
        assert merged[field].equals(tables[0][field])
    for index, role in ((1, "pre"), (2, "post")):
        field = f"{role}_teacher_log_probs"
        assert metadata.field(field).equals(flatten_metadata(tables[index]).field(field))
    assert metadata.field("loss_mask").equals(flatten_metadata(tables[0]).field("loss_mask"))


def test_merge_main_preserves_shard_order(tmp_path, monkeypatch):
    sources = [sealed_source(tmp_path, name) for name in ("base", "pre", "post")]
    output = tmp_path / "merged"
    monkeypatch.setattr(merge, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(
        merge,
        "parse_args",
        lambda: Namespace(
            base=sources[0].parent,
            pre_scores=sources[1].parent,
            post_scores=sources[2].parent,
            output_dir=output,
            workers=1,
        ),
    )
    merge.main()
    shards = sorted(output.glob("*.parquet"))
    assert [path.name for path in shards] == ["part-000.parquet", "part-001.parquet"]
    metadata = flatten_metadata(pa.concat_tables([pq.read_table(path) for path in shards]))
    assert metadata.field("sample_id").to_pylist() == [f"s{index}" for index in range(16)]


def test_metadata_replacement_preserves_struct_nulls_and_field_metadata():
    field = pa.field("score", pa.int32(), nullable=False, metadata={b"unit": b"logprob"})
    metadata = pa.StructArray.from_arrays(
        [pa.array([1, 2], type=pa.int32())],
        fields=[field],
        mask=pa.array([False, True]),
    )
    table = pa.table({"metadata": metadata})
    updated = replace_metadata_fields(table, {"score": pa.array([3.0, 4.0], type=pa.float32())})
    result = flatten_metadata(updated)
    assert result.is_null().to_pylist() == [False, True]
    assert result.type.field("score").metadata == field.metadata
    assert result.type.field("score").nullable is False
