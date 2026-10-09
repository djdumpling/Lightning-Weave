from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from data_curation.build_direct_opd_composed_target import compose_anchor_deltas, nested_score_array


def test_composition_reconstructs_weighted_sum_with_distinct_pre_teachers():
    pre_a = np.log(np.array([[0.4, 0.3]], dtype=np.float32))
    post_a = np.log(np.array([[0.6, 0.2]], dtype=np.float32))
    pre_b = np.log(np.array([[0.5, 0.25]], dtype=np.float32))
    post_b = np.log(np.array([[0.35, 0.45]], dtype=np.float32))

    synthetic_pre, synthetic_post, delta = compose_anchor_deltas([pre_a, pre_b], [post_a, post_b], [1.0, 0.75])

    expected = (post_a - pre_a) + 0.75 * (post_b - pre_b)
    np.testing.assert_allclose(delta, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(synthetic_post - synthetic_pre, expected, rtol=2e-6, atol=2e-6)
    assert np.all(synthetic_pre <= 0)
    assert np.all(synthetic_post <= 0)


def test_composition_accepts_more_than_two_anchors():
    pre = [np.array([[-1.0, -2.0]], dtype=np.float32)] * 3
    post = [
        np.array([[-0.5, -2.5]], dtype=np.float32),
        np.array([[-1.5, -1.5]], dtype=np.float32),
        np.array([[-0.8, -2.2]], dtype=np.float32),
    ]
    synthetic_pre, synthetic_post, delta = compose_anchor_deltas(pre, post, [1.0, 2.0, 0.5])
    expected = sum(weight * (after - before) for weight, before, after in zip([1.0, 2.0, 0.5], pre, post, strict=True))
    np.testing.assert_allclose(delta, expected)
    np.testing.assert_allclose(synthetic_post - synthetic_pre, expected)


def test_post_only_composition_removes_pre_teacher_terms():
    pre_a = np.log(np.array([[0.4, 0.3]], dtype=np.float32))
    post_a = np.log(np.array([[0.6, 0.2]], dtype=np.float32))
    pre_b = np.log(np.array([[0.5, 0.25]], dtype=np.float32))
    post_b = np.log(np.array([[0.35, 0.45]], dtype=np.float32))

    synthetic_pre, synthetic_post, delta = compose_anchor_deltas(
        [pre_a, pre_b],
        [post_a, post_b],
        [1.0, 0.75],
        composition_rule="weighted_post_log_prob_sum",
    )

    expected = post_a + 0.75 * post_b
    np.testing.assert_allclose(delta, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(synthetic_post - synthetic_pre, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(synthetic_pre, np.zeros_like(expected))
    assert not np.allclose(delta, (post_a - pre_a) + 0.75 * (post_b - pre_b))


@pytest.mark.parametrize(
    ("weights", "expected_anchor"),
    [
        ([2.0, 0.0], 0),
        ([0.0, 2.0], 1),
    ],
)
def test_composition_accepts_zero_weight_endpoints(weights, expected_anchor):
    pre = [
        np.array([[-1.0, -2.0]], dtype=np.float32),
        np.array([[-0.7, -1.6]], dtype=np.float32),
    ]
    post = [
        np.array([[-0.4, -2.4]], dtype=np.float32),
        np.array([[-1.1, -1.0]], dtype=np.float32),
    ]

    synthetic_pre, synthetic_post, delta = compose_anchor_deltas(pre, post, weights)

    expected = 2.0 * (post[expected_anchor] - pre[expected_anchor])
    np.testing.assert_allclose(delta, expected)
    np.testing.assert_allclose(synthetic_post - synthetic_pre, expected)


def test_nested_score_buffers_preserve_variable_response_lengths():
    rows = [
        np.array([[-1.0, -2.0]], dtype=np.float32),
        np.array([[-3.0, -4.0], [-5.0, -6.0]], dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
    ]
    arrow_type = pa.list_(pa.list_(pa.float32()))
    array = nested_score_array(rows, arrow_type)
    assert array.type == arrow_type
    assert array.to_pylist() == [row.tolist() for row in rows]


@pytest.mark.parametrize("weights", [[1.0], [1.0, 2.0, 3.0], [1.0, float('nan')], [0.0, 0.0]])
def test_composition_rejects_incomplete_or_invalid_weights(weights):
    scores = [np.zeros((1, 2)), np.zeros((1, 2))]
    with pytest.raises(ValueError):
        compose_anchor_deltas(scores, scores, weights)


def test_composition_rejects_broadcastable_shape_mismatch():
    with pytest.raises(ValueError, match="shapes differ"):
        compose_anchor_deltas(
            [np.zeros((2, 2)), np.zeros((1, 2))],
            [np.ones((2, 2)), np.ones((1, 2))],
            [1.0, 1.0],
        )


def _aligned_table(*, candidates=(1, 2), response=(1,), behavior=(-1.0, -2.0), sample_id="s0"):
    return pa.table({
        "prompt": ["same prompt"],
        "label": ["same label"],
        "metadata": [{
            "sample_id": sample_id,
            "candidate_ids": [list(candidates)],
            "response_tokens": list(response),
            "behavior_topk_log_probs": [list(behavior)],
            "loss_mask": [True],
            "pre_teacher_log_probs": [[-2.0, -3.0]],
            "post_teacher_log_probs": [[-1.0, -2.0]],
            "pre_teacher_revision": "pre",
            "post_teacher_revision": "post",
        }],
    })


@pytest.mark.parametrize("change", [
    {"candidates": (2, 1)},
    {"response": (2,)},
    {"behavior": (-2.0, -1.0)},
    {"sample_id": "different-row"},
])
def test_composition_rejects_equal_shaped_but_unaligned_scores(change):
    from data_curation.build_direct_opd_composed_target import compose_tables
    with pytest.raises(ValueError, match="aligned anchors differ"):
        compose_tables(
            [_aligned_table(), _aligned_table(**change)], [1.0, 1.0],
            rule="weighted_log_density_ratio_sum", revision="test",
        )


@pytest.mark.parametrize("both_short", [False, True])
def test_composition_never_seals_truncated_input(tmp_path, monkeypatch, both_short):
    from argparse import Namespace
    from data_curation import build_direct_opd_composed_target as compose

    left, right = tmp_path / "left.json", tmp_path / "right.json"
    output = tmp_path / "composed"
    monkeypatch.setattr(compose, "parse_args", lambda: Namespace(
        anchor_manifest=[left, right], anchor_name=["left", "right"], anchor_weight=[1.0, 1.0],
        output_dir=output, rows=2, repeat=1, rows_per_output_shard=1,
        composition_rule="weighted_log_density_ratio_sum",
    ))
    monkeypatch.setattr("slime.rollout.offline_direct_opd.validate_sealed_manifest", lambda path: {
        "total_rows": 2, "pre_teacher_model": {"revision": "pre"},
        "post_teacher_model": {"revision": "post"},
    })

    def chunks(path, manifest, **kwargs):
        yield _aligned_table()
        if not both_short and path == left:
            yield _aligned_table(sample_id="s1")

    monkeypatch.setattr(compose, "table_chunks", chunks)
    with pytest.raises(ValueError):
        compose.main()
    assert not output.exists()
    assert not list(tmp_path.glob(".composed.*"))
