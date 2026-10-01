"""The numerical check: a re-score of a few rows compared with the main scores through the composer's path."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_build_synthetic_shift_target import LENGTH, cache

from data_curation.check_score_precision import compare


def subset(full, directory, rows):
    """The listed rows of ``full``'s first shard, as a one-shard re-scored chain."""
    shard = min(full.glob("*.parquet"))
    directory.mkdir()
    pq.write_table(pq.read_table(shard).take(rows), directory / shard.name)
    return directory


def test_identical_rescores_agree_exactly_and_rows_join_by_sample_id(tmp_path):
    cache(tmp_path / "base", seed=0)
    main = cache(tmp_path / "main", seed=1, sealed=False)
    same = subset(cache(tmp_path / "again", seed=1, sealed=False), tmp_path / "check", [2, 0])
    report = compare(tmp_path / "base", main, same)
    assert report["rows"] == 2 and report["fisher_correlation"] == pytest.approx(1.0)
    assert report["relative_rms_error"] == pytest.approx(0.0, abs=1e-6)
    assert report["log_prob_difference"]["post"]["max"] == pytest.approx(0.0, abs=1e-6)


def test_a_rescaled_shift_is_aligned_but_off_and_an_unrelated_one_disagrees(tmp_path):
    cache(tmp_path / "base", seed=0)
    main = cache(tmp_path / "main", seed=1, sealed=False)
    doubled = subset(
        cache(tmp_path / "doubled", seed=1, sealed=False, shift_scale=2.0), tmp_path / "check2", [0, 1, 3]
    )
    report = compare(tmp_path / "base", main, doubled)
    assert report["fisher_correlation"] == pytest.approx(1.0, abs=1e-6)
    assert report["relative_rms_error"] == pytest.approx(1.0, abs=1e-4)
    assert report["shift_rms"]["check"] == pytest.approx(2 * report["shift_rms"]["main"], rel=1e-4)
    assert report["log_prob_difference"]["pre"]["max"] == pytest.approx(0.0, abs=1e-6)
    other = subset(cache(tmp_path / "other", seed=7, sealed=False), tmp_path / "check7", [0, 1, 2, 3])
    assert abs(compare(tmp_path / "base", main, other)["fisher_correlation"]) < 0.9


def test_the_check_must_be_one_shard_of_known_rows(tmp_path):
    cache(tmp_path / "base", seed=0)
    main = cache(tmp_path / "main", seed=1, sealed=False)
    with pytest.raises(ValueError, match="exactly one"):
        compare(tmp_path / "base", main, main)
    stranger = tmp_path / "stranger"
    cache(stranger, seed=1, sealed=False)
    table = pq.read_table(min(stranger.glob("*.parquet")))
    metadata = table.column("metadata").combine_chunks()
    renamed = [f"x{i}" for i in range(table.num_rows)]
    fields = [metadata.field(name) for name in metadata.type.names]
    fields[metadata.type.names.index("sample_id")] = pa.array(renamed)
    table = table.set_column(
        table.schema.get_field_index("metadata"),
        "metadata",
        pa.StructArray.from_arrays(fields, names=metadata.type.names),
    )
    (tmp_path / "check").mkdir()
    pq.write_table(table, tmp_path / "check" / min(stranger.glob("*.parquet")).name)
    with pytest.raises(KeyError, match="lacks"):
        compare(tmp_path / "base", main, tmp_path / "check")
    assert np.isfinite(compare(tmp_path / "base", main, subset(main, tmp_path / "ok", [1]))["fisher_correlation"])


def perturbed(path, field, change):
    """Rewrite one score field of a shard with ``change`` applied to every row's [T][K] list."""
    table = pq.read_table(path)
    metadata = table.column("metadata").combine_chunks()
    arrays = [metadata.field(name) for name in metadata.type.names]
    index = metadata.type.names.index(field)
    arrays[index] = pa.array([change(row) for row in arrays[index].to_pylist()], type=arrays[index].type)
    rebuilt = pa.StructArray.from_arrays(arrays, names=metadata.type.names)
    pq.write_table(table.set_column(table.schema.get_field_index("metadata"), "metadata", rebuilt), path)


def test_scores_without_evidence_or_outside_the_loss_mask_are_ignored(tmp_path):
    valid = [[True, False, True]] * LENGTH  # candidate 2 (token 2) is unmapped
    cache(tmp_path / "base", seed=0, loss_mask=[True, True, False, True, True])
    main = cache(tmp_path / "main", seed=1, sealed=False, candidate_valid=valid)
    check = subset(cache(tmp_path / "again", seed=1, sealed=False, candidate_valid=valid), tmp_path / "check", [3, 1])
    shard = min(check.glob("*.parquet"))
    clean = compare(tmp_path / "base", main, check, untrained={3})
    # placeholders on the unmapped candidate, a position outside the base loss mask, and the untrained token 3
    perturbed(shard, "post_teacher_log_probs", lambda row: [[a, float("nan"), c] for a, _, c in row])
    perturbed(shard, "pre_teacher_log_probs", lambda row: [[a, 55.0, c] for a, _, c in row])
    perturbed(
        shard, "post_teacher_log_probs", lambda row: [x if t != 2 else [v + 10 for v in x] for t, x in enumerate(row)]
    )
    perturbed(shard, "post_teacher_log_probs", lambda row: [[a, b, c + 7] for a, b, c in row])
    assert compare(tmp_path / "base", main, check, untrained={3}) == clean
    # without the untrained list, token 3's perturbation is visible
    assert compare(tmp_path / "base", main, check)["fisher_correlation"] < 1 - 1e-6
