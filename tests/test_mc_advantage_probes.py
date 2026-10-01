"""MC probes: call-structure matching, candidate choice, censoring, sample counts, and prompt-level inference."""

import json

import numpy as np
import pytest

from data_curation import mc_advantage_probes as probes
from data_curation.shift_states import StateLabeler, TokenTable


def test_call_match_compares_calls_as_an_unordered_multiset():
    target = {
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "f", "arguments": {"a": 1}}},
            {"type": "function", "function": {"name": "g", "arguments": {}}},
        ],
    }
    message = (
        "<think>\nplan\n</think>\n\n<tool_call>\n" + json.dumps({"name": "g", "arguments": {}})
        + "\n</tool_call>\n<tool_call>\n" + json.dumps({"name": "f", "arguments": {"a": 1}}) + "\n</tool_call>"
    )
    assert probes.score_message(message, target) == (2, True, False)
    assert probes.score_message(message.replace('"a": 1', '"a": 2'), target) == (2, False, False)
    assert probes.score_message("<think>\nunfinished", target)[2]
    # A text-only reference matches any call-free message: structure only, not content.
    text_target = {"content": "Which city?", "tool_calls": []}
    assert probes.score_message("<think>\nx\n</think>\n\nAnything at all.", text_target) == (0, True, False)


def test_defining_alternatives_are_always_probed():
    table = TokenTable.from_vocab({".Ċ": 0, ".ĊĊ": 1, "Ġa": 2, "Ġb": 3, "Ġc": 4}, {"<think>": 5, "</think>": 6, "<tool_call>": 7, "</tool_call>": 8, "<|im_end|>": 9})
    labeler = StateLabeler(table)
    candidates = np.asarray([2, 3, 4, 0, 1])
    topk = np.asarray([0.5, 0.2, 0.15, 0.1, 0.05])
    chosen = probes.probe_candidates(labeler, "think_stop_fork", candidates, topk, 3)
    assert {3, 4} <= set(chosen) and len(chosen) == 3
    call = probes.probe_candidates(labeler, "act_vs_talk", np.asarray([2, 7, 3]), np.asarray([0.6, 0.1, 0.3]), 2)
    assert 1 in call and 0 in call


def state(candidates=3):
    return {
        "sample_id": "s", "prompt_id": "p", "position": 4, "state_type": "think_stop_fork",
        "candidate_ids": list(range(candidates)), "behavior_probs": [0.5, 0.3, 0.2][:candidates],
    }


def records(finished=True, samples=4):
    return [
        {"sample_id": "s", "position": 4, "candidate_index": c, "sample_index": i, "think_tokens": 10 * (c + 1) + i,
         "total_tokens": 1, "tool_calls": 1, "call_match": c == 0, "malformed": False, "finished": finished or c != 2}
        for c in range(3) for i in range(samples)
    ]


def test_advantages_are_centered_and_split_into_halves():
    entry = probes.advantages([state()], records(), samples=4)[("s", 4)]
    weights = entry["weights"]
    for key in ("think_tokens:full", "think_tokens:half_a", "call_match:full"):
        assert abs(weights @ entry[key]) < 1e-12
    assert entry["think_tokens:full"][0] < entry["think_tokens:full"][2]
    np.testing.assert_allclose(entry["think_tokens:half_b"] - entry["think_tokens:half_a"], 0.0, atol=1e-12)


def test_sample_counts_must_be_exact():
    with pytest.raises(ValueError, match="continuation counts"):
        probes.advantages([state()], records()[:-1], samples=4)


def test_censored_states_have_no_token_outcomes():
    entry = probes.advantages([state()], records(finished=False), samples=4)[("s", 4)]
    assert not entry["uncensored"] and entry["finish_rate"] == [1.0, 1.0, 0.0]
    assert "think_tokens:full" not in entry and "call_match:full" in entry


def test_fisher_correlation_is_signed_and_the_interval_is_prompt_clustered():
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(20):
        w = rng.dirichlet(np.ones(3))
        u = rng.normal(size=3)
        u -= w @ u
        pairs.append((w, u, 3 * u))
    assert np.isclose(probes.fisher_correlation(probes.fisher_terms(pairs)), 1.0)
    flipped = probes.fisher_terms([(w, u, -u) for w, u, _ in pairs])
    assert np.isclose(probes.fisher_correlation(flipped), -1.0)
    low, high = probes.clustered_interval(flipped, ["a"] * 10 + ["b"] * 10, 50, np.random.default_rng(1))
    assert np.isclose(low, -1.0) and np.isclose(high, -1.0)
