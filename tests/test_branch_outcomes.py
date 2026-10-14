"""Executed-branch outcomes: parsing sampled turns, cutting histories, advantages, and the outcome target term."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_curation import branch_outcomes as branches  # noqa: E402
from data_curation.build_synthetic_shift_target import SyntheticComposer, Term, outcome_vector  # noqa: E402
from data_curation.shift_geometry import CachedRow  # noqa: E402


def test_a_sampled_turn_becomes_the_message_vllm_would_return():
    call = '<tool_call>\n{"name": "get_user_details", "arguments": {"user_id": "u1"}}\n</tool_call>'
    turn = branches.parse_generated_turn("Let me look.\n</think>\n\n" + call, "stop")
    assert turn == {"content": "\n\n", "tool_calls": [{"name": "get_user_details", "arguments": {"user_id": "u1"}}]}
    two = branches.parse_generated_turn("r</think>\n\n" + call + "\n" + call.replace("u1", "u2"), "stop")
    assert [item["arguments"]["user_id"] for item in two["tool_calls"]] == ["u1", "u2"] and two["content"] == "\n\n"
    # a final block left open is still parsed, as by the hermes parser
    assert branches.parse_generated_turn("r</think>" + call[: -len("</tool_call>")], "stop")["tool_calls"]
    assert branches.parse_generated_turn("r</think>\n\nPlease confirm.", "stop") == {
        "content": "\n\nPlease confirm.",
        "tool_calls": None,
    }
    # an unparseable call: the whole text is the content, without calls
    broken = branches.parse_generated_turn("r</think>\n\nok <tool_call>{not json}</tool_call>", "stop")
    assert broken == {"content": "\n\nok <tool_call>{not json}</tool_call>", "tool_calls": None}
    listed = branches.parse_generated_turn('r</think>x<tool_call>{"name": "f", "arguments": [1]}</tool_call>', "stop")
    assert listed == {"content": 'x<tool_call>{"name": "f", "arguments": [1]}</tool_call>', "tool_calls": None}
    # no branch: truncated, reasoning never closed, nothing after the reasoning
    assert branches.parse_generated_turn("r</think>\n\nhi", "length") is None
    assert branches.parse_generated_turn("still thinking", "stop") is None
    assert branches.parse_generated_turn("r</think>\n\n  ", "stop") is None


def test_the_history_is_cut_before_the_requests_response():
    messages = [
        {"role": "assistant", "content": "Hi! How can I help you today?"},  # tau2's scripted greeting
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}]},  # response to request 0
        {"role": "tool", "id": "a", "content": "{}"},
        {"role": "assistant", "content": "done"},  # response to request 1
    ]
    assert branches.episode_prefix(messages, 0) == messages[:2]
    assert branches.episode_prefix(messages, 1) == messages[:4]
    with pytest.raises(ValueError):
        branches.episode_prefix(messages, 2)


def test_the_plan_has_one_branch_per_parseable_response():
    prompts = [{"prompt_id": "p", "collector": "c", "domain": "tau2_airline", "task_id": "airline_1", "request_index": 0}]
    messages = [{"role": "assistant", "content": "Hi"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "x"}]
    records = {("c", "tau2_airline", "airline_1"): {"result": {"messages": messages}}}
    call = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
    rollouts = [
        {"sample_id": "s" * 64, "prompt_id": "p", "response": "r</think>" + call, "finish_reason": "stop"},
        {"sample_id": "t" * 64, "prompt_id": "p", "response": "r</think>text", "finish_reason": "stop"},
        {"sample_id": "u" * 64, "prompt_id": "p", "response": "r", "finish_reason": "length"},
    ]
    plan, counts = branches.build_plan(prompts, rollouts, records)
    assert counts == {"responses": 3, "tool_call": 1, "text": 1, "truncated": 1}
    # an episode that ended in the model's error has no trajectory; a state past the last response has no history
    failed = {key: {"result": None} for key in records}
    assert branches.build_plan(prompts, rollouts, failed)[1] == {"responses": 3, "no_trajectory": 2, "truncated": 1}
    late = [{**prompts[0], "request_index": 5}]
    assert branches.build_plan(late, rollouts, records)[1] == {"responses": 3, "no_recorded_response": 2, "truncated": 1}
    first = plan[0]
    assert first["branch_id"] == "s" * 24 and first["history"] == messages[:2] and first["seed"] == branches.branch_seed("s" * 24)
    assert first["branch"] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "s" * 24 + "-0", "name": "f", "arguments": {}, "requestor": "assistant"}],
    }
    assert plan[1]["branch"] == {"role": "assistant", "content": "text", "tool_calls": None}


def test_advantages_are_smoothed_success_log_ratios_within_each_state():
    rollouts = [{"prompt_id": prompt, "sample_id": f"{prompt}{index}"} for prompt in "ab" for index in range(4)]
    rollouts.append({"prompt_id": "c", "sample_id": "c0"})
    outcomes = {"a0": 1.0, "a1": 0.0, "a2": 0.0, "b0": 1.0, "b1": 1.0, "c0": 1.0}  # a3, b2, b3 have no outcome
    result, counts = branches.advantages(rollouts, outcomes)
    mean = (0.75 + 0.25 + 0.25) / 3
    assert result["a0"] == pytest.approx(math.log(0.75 / mean)) and result["a1"] == pytest.approx(math.log(0.25 / mean))
    assert result["a3"] == 0.0  # no outcome
    assert result["b0"] == result["b1"] == result["c0"] == 0.0  # outcomes that agree, or a single one
    assert counts["informative_states"] == 1 and counts["flat_states"] == 2 and counts["responses"] == 9
    # the outcome-weighted mean of p_k / p̄ over a state's scored responses is 1 (log-ratios of a mean)
    assert sum(math.exp(result[f"a{index}"]) for index in range(3)) / 3 == pytest.approx(1.0)


def test_branch_records_score_model_failures_and_skip_the_simulators_and_replays():
    assert branches.branch_outcome({"termination": "user_stop", "outcome": 1.0}) == 1.0
    assert branches.branch_outcome({"termination": "max_steps", "outcome": 0.0}) == 0.0
    assert branches.branch_outcome({"termination": "agent_error", "reward": 0.0}) == 0.0
    assert branches.branch_outcome({"termination": "context_window_exceeded", "reward": 0.0}) == 0.0
    assert branches.branch_outcome({"termination": "user_error", "reward": 0.0}) is None
    assert branches.branch_outcome({"termination": "branch_setup_error", "outcome": None}) is None


def row(tokens, candidates, sample_id="s"):
    candidates = np.asarray(candidates)
    return CachedRow(
        sample_id=sample_id,
        prompt_id="p",
        response_tokens=np.asarray(tokens),
        candidate_ids=candidates,
        behavior_log_probs=np.log(np.full(candidates.shape, 0.2)),
        loss_mask=np.ones(len(tokens), dtype=bool),
    )


def test_the_outcome_term_rewards_each_sampled_token_bucket():
    # position 0 sampled candidate 1; position 1 sampled a token outside the candidates (the remaining bucket)
    vector = outcome_vector(row([7, 9], [[5, 7, 8], [5, 6, 8]]), {"s": 0.5})
    assert vector.tolist() == [[0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5]]
    with pytest.raises(KeyError):
        outcome_vector(row([7], [[7, 8, 9]], sample_id="missing"), {"s": 0.5})


def test_outcome_terms_name_their_advantages_and_take_no_direction_or_gate():
    spec = {"alpha": 2.0, "terms": [{"transform": "outcome", "coef": 1.0, "advantages": "branch"}]}
    composer = SyntheticComposer(spec, labeler=None, advantages={"branch": {"s": 0.5}})
    assert composer.sources == set()
    probs, (vector,) = composer.row_terms(row([7], [[5, 7, 8]]))
    assert vector.tolist() == [[0.0, 0.5, 0.0, 0.0]]
    with pytest.raises(ValueError, match="without --advantages"):
        SyntheticComposer(spec, labeler=None)
    for bad in (
        {"transform": "outcome", "advantages": "b", "direction": {"terms": [{"source": "x", "coef": 1.0}]}},
        {"transform": "outcome"},
        {"transform": "raw", "advantages": "b", "direction": {"terms": [{"source": "x", "coef": 1.0}]}},
        {"transform": "outcome", "advantages": "b", "gate": {"exclude_first_reflection": True}},
    ):
        with pytest.raises(ValueError):
            Term(0, bad)


def test_rollouts_are_read_from_the_shards_metadata_struct(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    item = {"sample_id": "s", "prompt_id": "p", "response": "r</think>x", "finish_reason": "stop", "response_length": 3}
    pq.write_table(pa.Table.from_pylist([{"prompt": "q", "metadata": item}]), tmp_path / "rollouts-r00000-00000.parquet")
    assert branches.read_rollouts(tmp_path) == [{key: item[key] for key in branches.ROLLOUT_FIELDS}]


def test_the_teacher_term_targets_the_post_models_own_distribution():
    from data_curation.build_synthetic_shift_target import teacher_vector
    from data_curation.shift_geometry import bucket_probs, log_tilted_target, relevel

    rng = np.random.default_rng(0)
    behavior = np.log(np.array([[0.6, 0.3, 0.05], [0.2, 0.2, 0.1]]))  # remaining mass 0.05 and 0.5
    teacher = np.log(np.array([[0.1, 0.7, 0.1], [0.5, 0.1, 0.1]]))  # remaining mass 0.1 and 0.3
    item = row([7, 9], [[5, 7, 8], [5, 6, 8]])
    item.behavior_log_probs = behavior
    item.post["t"] = teacher
    item.mapped["t"] = np.ones((2, 3), dtype=bool)
    delta = relevel(teacher_vector(item, "t", 2.0))
    target = np.exp(log_tilted_target(bucket_probs(behavior), delta, 2.0))
    expected = np.array([[0.1, 0.7, 0.1, 0.1], [0.5, 0.1, 0.1, 0.3]])
    assert np.allclose(target, expected) and np.allclose(delta[:, -1], 0.0)
    spec = {"alpha": 2.0, "terms": [{"transform": "teacher", "direction": {"terms": [{"source": "t", "coef": 1.0}]}}]}
    composer = SyntheticComposer(spec, labeler=None)
    assert composer.sources == {"t"}
    _, (vector,) = composer.row_terms(item)
    assert np.allclose(relevel(vector), delta)
    with pytest.raises(ValueError, match="exactly one source"):
        Term(0, {"transform": "teacher", "direction": {"terms": [{"source": "t", "coef": 0.5}]}})
    with pytest.raises(ValueError, match="no gates|take no gates"):
        Term(0, {"transform": "teacher", "direction": {"terms": [{"source": "t", "coef": 1.0}]}, "gate": {"exclude_first_reflection": True}})
    item.mapped["t"][0, 1] = False
    with pytest.raises(ValueError, match="unmapped"):
        teacher_vector(item, "t", 2.0)


def test_the_probe_scores_responses_by_their_tilted_target_log_ratio():
    from data_curation import branch_signal_probe as probe

    item = row([7, 9], [[5, 7, 8], [5, 6, 8]])  # position 1 sampled outside the candidates
    assert probe.sampled_buckets(item).tolist() == [1, 3]
    zero = np.zeros((2, 4))
    assert probe.sequence_log_ratio(item, zero, 2.0) == pytest.approx(0.0)
    favor = zero.copy()
    favor[0, 1] = 2.0  # raises the sampled token at position 0 by one nat before normalization (alpha 2)
    probs = np.array([0.2, 0.2, 0.2, 0.4])
    expected = 1.0 - math.log((probs * np.exp(favor[0] / 2.0)).sum())
    assert probe.sequence_log_ratio(item, favor, 2.0) == pytest.approx(expected)
    assert probe.pair_counts([3.0, 1.0, 1.0], [1.0, 0.0, 0.0]) == (2.0, 2)
    assert probe.pair_counts([1.0, 1.0], [1.0, 0.0]) == (0.5, 1)
    states = {"a": [({"s": 1.0}, 1.0), ({"s": 0.0}, 0.0)], "b": [({"s": 0.0}, 1.0), ({"s": 1.0}, 0.0)]}
    table = probe.auc_table(states, ["s"], draws=200, seed=0)["s"]
    assert table["auc"] == 0.5 and table["pairs"] == 2 and 0.0 <= table["lo"] <= table["hi"] <= 1.0


def test_probe_spans_split_reasoning_from_the_action_at_think_end():
    from data_curation import branch_signal_probe as probe

    item = row([5, probe.THINK_END_ID, 7, 8], [[5, 7, 8]] * 4)
    masks = probe.span_masks(item)
    assert masks["reasoning"].tolist() == [True, True, False, False]
    assert masks["action"].tolist() == [False, False, True, True]
    unfinished = row([5, 6], [[5, 7, 8]] * 2)
    assert probe.span_masks(unfinished)["action"].tolist() == [False, False]
