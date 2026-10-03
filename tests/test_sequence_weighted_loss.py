"""The ``sequence_weighted`` Offline Direct-OPD loss: weighted, summed log-likelihood of cached responses."""

from __future__ import annotations

import sys
import types
from argparse import Namespace

import pytest
import torch

from slime.rollout import offline_direct_opd as direct_opd
from slime.utils.types import Sample


def metadata(weight=None):
    scores = [[-0.2, -1.2], [-0.3, -1.3], [-0.1, -2.0]]
    row = {
        "is_offline_direct_opd": True,
        "schema_version": direct_opd.SCHEMA_VERSION,
        "prompt_tokens": [4, 5],
        "response_tokens": [7, 8, 9],
        "response_length": 3,
        "loss_mask": [1, 1, 1],
        "candidate_ids": [[7, 2], [8, 3], [9, 1]],
        "behavior_topk_log_probs": scores,
        "post_teacher_log_probs": scores,
        "pre_teacher_log_probs": scores,
        "student_ref_sampled_log_probs": [-0.2, -0.3, -0.1],
        "behavior_sampled_log_probs": [-0.2, -0.3, -0.1],
        "finish_reason": "stop",
    }
    if weight is not None:
        row[direct_opd.SEQUENCE_WEIGHT_FIELD] = weight
    return row


def test_a_response_weight_is_validated_and_repeated_on_its_tokens():
    sample = Sample(metadata=metadata(1.75))
    direct_opd.attach_offline_direct_opd_fields(sample, expected_top_k=2)
    assert sample.sequence_weights == [1.75, 1.75, 1.75]
    for trusted in (False, True):
        hydrated = direct_opd.hydrate_offline_direct_opd_sample(
            Sample(prompt=[4, 5], metadata=metadata(0.5)), tokenizer=None, expected_top_k=2, trusted_sealed=trusted
        )
        torch.testing.assert_close(hydrated.sequence_weights, torch.full((3,), 0.5))
        assert hydrated.metadata[direct_opd.SEQUENCE_WEIGHT_FIELD] == 0.5
    unweighted = direct_opd.hydrate_offline_direct_opd_sample(
        Sample(prompt=[4, 5], metadata=metadata()), tokenizer=None, expected_top_k=2
    )
    assert unweighted.sequence_weights is None
    for bad in (-0.1, float("nan"), "heavy"):
        with pytest.raises(direct_opd.OfflineDirectOPDDataError, match="sequence_weight"):
            direct_opd.validate_offline_direct_opd_metadata(metadata(bad), expected_top_k=2)


def test_terms_are_the_weighted_negative_log_likelihood_scaled_back_to_a_sum():
    log_probs = torch.tensor([-0.5, -1.0, -2.0], requires_grad=True)
    weights = torch.tensor([2.0, 2.0, 2.0])
    terms = direct_opd.offline_direct_opd_sequence_weighted_terms(log_probs, weights, response_tokens=3)
    torch.testing.assert_close(terms["sequence_weighted_loss"], torch.tensor([3.0, 6.0, 12.0]))
    # The per-sample reducer divides by the 3 tokens, recovering w * (summed negative log-likelihood).
    torch.testing.assert_close(terms["sequence_weighted_loss"].mean(), torch.tensor(2.0 * 3.5))
    terms["sequence_weighted_loss"].mean().backward()
    torch.testing.assert_close(log_probs.grad, -weights)
    with pytest.raises(ValueError, match="shape"):
        direct_opd.offline_direct_opd_sequence_weighted_terms(log_probs.detach(), torch.ones(2), response_tokens=3)


def test_the_fixed_point_is_the_weighted_empirical_distribution():
    """Three cached one-token responses (tokens 0, 0, 2) with weights 0.5, 0.5, 2: the fit puts 1/3 on 0 and 2/3 on 2."""
    logits = torch.zeros(4, requires_grad=True)
    tokens, weights = torch.tensor([0, 0, 2]), torch.tensor([0.5, 0.5, 2.0])
    optimizer = torch.optim.SGD([logits], lr=0.5)
    for _ in range(3_000):
        optimizer.zero_grad()
        log_probs = logits.log_softmax(-1)[tokens]
        terms = direct_opd.offline_direct_opd_sequence_weighted_terms(log_probs, weights, response_tokens=1)
        terms["sequence_weighted_loss"].sum().backward()
        optimizer.step()
    torch.testing.assert_close(logits.softmax(-1), torch.tensor([1 / 3, 0.0, 2 / 3, 0.0]), atol=2e-3, rtol=0)


@pytest.fixture
def megatron_wrapper(monkeypatch):
    """slime's real Megatron loss wrapper and reducers, with Megatron's parallel state stubbed for one CPU rank."""
    import importlib

    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    core.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_data_parallel_world_size=lambda with_context_parallel=False: 1,
        get_tensor_model_parallel_group=lambda: None,
    )
    megatron.core = core
    misc = types.ModuleType("slime.utils.misc")  # the real module imports ray, which the CPU tests do not need

    def load_function(path):
        module, _, name = path.rpartition(".")
        return getattr(importlib.import_module(module), name)

    misc.load_function = load_function
    for name, module in (("megatron", megatron), ("megatron.core", core), ("slime.utils.misc", misc)):
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("slime.backends.megatron_utils.loss", "slime.backends.megatron_utils.cp_utils"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    from slime.utils import ppo_utils

    monkeypatch.setattr(  # the tensor-parallel kernel, for a single rank
        ppo_utils,
        "calculate_log_probs_and_entropy",
        lambda logits, tokens, group, with_entropy=False, chunk_size=-1: (
            logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)),
            None,
        ),
    )
    return importlib.import_module("slime.backends.megatron_utils.loss")


def wrapper_args(**overrides):
    values = dict(
        loss_type="custom_loss", custom_loss_function_path="slime.rollout.offline_direct_opd.megatron_loss",
        calculate_per_token_loss=False, global_batch_size=1, recompute_loss_function=False, rollout_temperature=1.0,
        offline_direct_opd_loss_mode="sequence_weighted", log_probs_chunk_size=-1,
    )
    return Namespace(**(values | overrides))


def one_response_batch(response, weight):
    """A micro-batch with one cached response after a one-token prompt."""
    length = len(response)
    return {
        "unconcat_tokens": [torch.tensor([9, *response])],
        "total_lengths": [1 + length],
        "response_lengths": [length],
        "loss_masks": [torch.ones(length, dtype=torch.int32)],
        "sequence_weights": [torch.full((length,), weight)],
        **{field: [torch.zeros(length)] for field in direct_opd.DIRECT_OPD_SAMPLE_FIELDS},
    }


def test_the_trainer_fits_summed_likelihood_whatever_the_batch_lengths(megatron_wrapper):
    """Equal-weight responses of 1 and 3 tokens, in separate batches, share the first token's decision 50/50.

    Position-wise logits: both responses compete at their first token (0 or 1); only the long one uses positions 1-2.
    Normalizing each batch by its token count would weight the long response by 1/3 and converge to 75/25.
    """
    batches = [one_response_batch([0], 1.0), one_response_batch([1, 2, 3], 1.0)]
    args = wrapper_args()

    def first_token_split(per_token_normalization: bool) -> torch.Tensor:
        table = torch.zeros(3, 4, requires_grad=True)  # logits per sequence position
        optimizer = torch.optim.SGD([table], lr=1.0)
        decay = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (1.0 + step / 50))
        for step in range(20_000):
            batch = batches[step % 2]
            logits = table[: batch["total_lengths"][0]].unsqueeze(0)  # position t predicts token t + 1
            loss, normalizer, _ = megatron_wrapper.loss_function(args, batch, 1, logits)
            assert int(normalizer) == 1  # per-sample mode: Megatron divides only by the micro-batch count (1)
            if per_token_normalization:  # what dividing by the batch's token count would do
                loss = loss / batch["response_lengths"][0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            decay.step()
        first = table[0].softmax(-1).detach()
        return first[:2] / first[:2].sum()

    torch.testing.assert_close(first_token_split(False), torch.tensor([0.5, 0.5]), atol=5e-3, rtol=0)
    torch.testing.assert_close(first_token_split(True), torch.tensor([0.75, 0.25]), atol=5e-3, rtol=0)

    with pytest.raises(ValueError, match="omit --calculate-per-token-loss"):
        megatron_wrapper.loss_function(wrapper_args(calculate_per_token_loss=True), batches[0], 1,
                                       torch.zeros(1, 2, 4))


def test_megatron_loss_sums_each_responses_weighted_log_likelihood(megatron_wrapper):
    """Two responses in one batch: the wrapper returns sum_i w_i log p(y_i) / global batch size."""
    torch.manual_seed(1)
    responses, weights = [[1, 4, 2], [5]], [1.5, 0.25]
    batch = {
        "unconcat_tokens": [torch.tensor([0, *response]) for response in responses],
        "total_lengths": [1 + len(response) for response in responses],
        "response_lengths": [len(response) for response in responses],
        "loss_masks": [torch.ones(len(response), dtype=torch.int32) for response in responses],
        "sequence_weights": [torch.full((len(response),), weight) for response, weight in zip(responses, weights)],
        **{field: [torch.zeros(len(response)) for response in responses] for field in direct_opd.DIRECT_OPD_SAMPLE_FIELDS},
    }
    logits = torch.randn(1, sum(batch["total_lengths"]), 6)
    loss, _, log = megatron_wrapper.loss_function(wrapper_args(global_batch_size=2), batch, 1, logits)
    log_probs = logits[0].log_softmax(-1)
    expected = 0.0
    offset = 0
    for response, weight in zip(responses, weights, strict=True):
        positions = offset + torch.arange(len(response))  # logits before each response token
        expected -= weight * log_probs[positions, response].sum()
        offset += 1 + len(response)
    torch.testing.assert_close(loss, expected / 2)
    assert log["keys"] == ["loss", "sequence_weighted_loss", "sampled_log_prob", "sequence_weight"]

    batch["sequence_weights"] = None
    with pytest.raises(KeyError, match="sequence_weight"):
        megatron_wrapper.loss_function(wrapper_args(global_batch_size=2), batch, 1, logits)
