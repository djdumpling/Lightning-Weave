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
