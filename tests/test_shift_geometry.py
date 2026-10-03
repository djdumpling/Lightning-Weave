"""Invariances of the K+1 bucket geometry, evidence handling, and state labeling."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from data_curation import shift_geometry as geometry
from data_curation.shift_states import STATE_CODE, StateLabeler, TokenTable
from slime.rollout.offline_direct_opd import _topk_plus_other_distribution

K = 3


def random_state(rng, states=6):
    logits = rng.normal(size=(states, 8))
    log_probs = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    return -np.sort(-log_probs, axis=-1)[:, :K]


def test_bucket_probs_match_the_training_loss():
    topk = random_state(np.random.default_rng(0))
    expected = _topk_plus_other_distribution(torch.tensor(topk, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(geometry.bucket_probs(topk), expected, rtol=1e-6, atol=1e-7)  # the loss runs in float32


def test_target_is_invariant_to_state_constants_over_all_buckets_only():
    rng = np.random.default_rng(1)
    probs = geometry.bucket_probs(random_state(rng))
    delta = geometry.bucket_shift(rng.normal(size=(6, K)))
    constant = rng.normal(size=(6, 1))
    base = geometry.log_tilted_target(probs, delta, 2.0)
    np.testing.assert_allclose(geometry.log_tilted_target(probs, delta + constant, 2.0), base, atol=1e-12)
    shifted_candidates = delta.copy()
    shifted_candidates[:, :-1] += constant
    assert not np.allclose(geometry.log_tilted_target(probs, shifted_candidates, 2.0), base)
    centered = geometry.center(delta, probs)
    np.testing.assert_allclose((probs * centered).sum(-1), 0.0, atol=1e-12)
    releveled = geometry.relevel(centered)
    np.testing.assert_array_equal(releveled[:, -1], 0.0)
    np.testing.assert_allclose(geometry.log_tilted_target(probs, releveled, 2.0), base, atol=1e-12)


def test_fisher_inner_product_is_the_second_order_target_kl():
    rng = np.random.default_rng(2)
    probs = geometry.bucket_probs(random_state(rng))
    u, v = rng.normal(size=(2, 6, K + 1))
    alpha, scale = 2.0, 1e-3
    kl = geometry.target_kl(probs, scale * u, scale * v, alpha)
    difference = geometry.center(scale * (u - v), probs)
    np.testing.assert_allclose(kl, (probs * difference**2).sum(-1) / (2 * alpha**2), rtol=1e-3)


# --- cache fixtures ---------------------------------------------------------------------------------


def write_cache(root, rows, *, seed=0, sample_prefix="s", candidate_valid=None, placeholder=None):
    """A tiny scored cache; ``placeholder`` overwrites the scores of invalid candidates."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    names = geometry.BASE_FIELDS + ("candidate_projection_valid_mask", "loss_mask_before_token_projection")
    values = {name: [] for name in names}
    for index in range(rows):
        length = 4
        valid = np.asarray(candidate_valid if candidate_valid is not None else [[True] * K] * length)
        pre = rng.normal(size=(length, K)) - 3
        post = pre + rng.normal(size=(length, K))
        if placeholder is not None:
            pre, post = np.where(valid, pre, placeholder), np.where(valid, post, placeholder)
        values["sample_id"].append(f"{sample_prefix}{index}")
        values["prompt_id"].append(f"p{index // 2}")
        values["response_tokens"].append([1, 2, 3, 4])
        values["candidate_ids"].append([[1, 2, 3]] * length)
        values["behavior_topk_log_probs"].append(random_state(np.random.default_rng(index), length).tolist())
        values["loss_mask"].append([True] * length)
        values["loss_mask_before_token_projection"].append([True] * length)
        values["pre_teacher_log_probs"].append(pre.tolist())
        values["post_teacher_log_probs"].append(post.tolist())
        values["candidate_projection_valid_mask"].append(valid.tolist())
    floats = pa.list_(pa.list_(pa.float32()))
    types = {
        "response_tokens": pa.list_(pa.int32()),
        "candidate_ids": pa.list_(pa.list_(pa.int32())),
        "behavior_topk_log_probs": floats,
        "pre_teacher_log_probs": floats,
        "post_teacher_log_probs": floats,
        "loss_mask": pa.list_(pa.bool_()),
        "loss_mask_before_token_projection": pa.list_(pa.bool_()),
        "candidate_projection_valid_mask": pa.list_(pa.list_(pa.bool_())),
    }
    metadata = pa.StructArray.from_arrays([pa.array(v, type=types.get(n)) for n, v in values.items()], names=list(values))
    pq.write_table(pa.table({"metadata": metadata}), root / "shard-00000.parquet")
    return root


def test_joined_rows_check_alignment_and_separate_mapped_from_trained(tmp_path):
    base = write_cache(tmp_path / "base", 4)
    donor = write_cache(tmp_path / "donor", 4, seed=3, candidate_valid=[[True, True, False]] * 4)
    rows = list(geometry.iter_joined_rows(base, {"donor": donor}, untrained={"donor": {2}}))
    assert [row.sample_id for row in rows] == ["s0", "s1", "s2", "s3"]
    row = rows[0]
    assert row.mapped["donor"][:, 2].sum() == 0 and row.mapped["donor"][:, 1].all()
    assert row.trained["donor"][:, 1].sum() == 0 and row.trained["donor"][:, 2].all()
    assert row.valid("donor")[:, 0].all() and not row.valid("donor")[:, 1:].any()
    assert row.valid("agent_acc").all()
    probs = np.exp(row.behavior_log_probs)
    np.testing.assert_allclose(row.coverage("donor"), probs[:, 0] / probs.sum(-1))
    misaligned = write_cache(tmp_path / "misaligned", 4, sample_prefix="x")
    with pytest.raises(ValueError, match="row-aligned"):
        list(geometry.iter_joined_rows(base, {"donor": misaligned}))


def test_support_never_counts_placeholder_scores(tmp_path):
    base = write_cache(tmp_path / "base", 2)
    donor = write_cache(tmp_path / "donor", 2, seed=4, candidate_valid=[[True, False, False]] * 4, placeholder=-0.01)
    row = next(geometry.iter_joined_rows(base, {"donor": donor}))
    scores = pq.read_table(donor / "shard-00000.parquet").column("metadata").combine_chunks()
    post = np.asarray(scores.field("post_teacher_log_probs")[0].as_py())
    pre = np.asarray(scores.field("pre_teacher_log_probs")[0].as_py())
    np.testing.assert_allclose(row.support["donor"], np.minimum(np.exp(post[:, 0]), np.exp(pre[:, 0])))
    assert (row.support["donor"] < 0.5).all()  # the near-certain placeholder scores are excluded


def test_a_candidate_without_evidence_keeps_its_odds_against_the_remaining_vocabulary(tmp_path):
    base = write_cache(tmp_path / "base", 2)
    donor = write_cache(tmp_path / "donor", 2, seed=5, candidate_valid=[[True, False, True]] * 4)
    row = next(geometry.iter_joined_rows(base, {"donor": donor}))
    probs = geometry.bucket_probs(row.behavior_log_probs)
    direction = geometry.Direction("donor", (("donor", 1.0),))
    delta = geometry.relevel(geometry.direction_vectors(row, [direction], probs)[..., 0])
    target = geometry.log_tilted_target(probs, delta, 2.0)
    base_odds = np.log(probs[:, 1]) - np.log(probs[:, -1])
    np.testing.assert_allclose(target[:, 1] - target[:, -1], base_odds, atol=1e-12)
    assert not np.allclose(target[:, 0] - target[:, -1], np.log(probs[:, 0]) - np.log(probs[:, -1]))


def test_keep_untrained_restores_the_legacy_log_ratio(tmp_path):
    base = write_cache(tmp_path / "base", 2)
    row = next(geometry.iter_joined_rows(base, {}, untrained={"agent_acc": {2}}))
    clean = geometry.evidence_shift(row, "agent_acc")
    legacy = geometry.evidence_shift(row, "agent_acc", keep_untrained=True)
    np.testing.assert_array_equal(clean[:, 1], 0.0)
    np.testing.assert_array_equal(legacy[:, :-1], row.shifts["agent_acc"])


def test_state_labels_follow_think_and_tool_call_structure():
    vocab = {"Ġa": 0, ".Ċ": 1, ".ĊĊ": 2, "Wait": 3, "{\"": 4, "name": 5, "\":": 6, "Ġ\"": 7, "f": 8, "Ċ": 9}
    added = {"<think>": 20, "</think>": 21, "<tool_call>": 22, "</tool_call>": 23, "<|im_end|>": 24}
    labeler = StateLabeler(TokenTable.from_vocab(vocab, added))
    tokens = [20, 0, 1, 21, 9, 22, 4, 5, 6, 7, 8, 23, 24]
    candidates = [[token, 0, 1] for token in tokens]
    probs = [[1.0, 0.0, 0.0] for _ in tokens]
    candidates[2], probs[2] = [1, 2, 0], [0.6, 0.3, 0.1]
    candidates[3] = [21, 0, 1]
    names = {code: name for name, code in STATE_CODE.items()}
    labels = [names[code] for code in labeler.label(tokens, candidates, probs)]
    assert (labels[0], labels[2], labels[3], labels[10], labels[12]) == (
        "think_open",
        "think_stop_fork",
        "think_close",
        "tool_name",
        "call_boundary",
    )
    assert json.dumps(labels)
