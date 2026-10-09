#!/usr/bin/env python3
"""Exact finite-POMDP witnesses, not LLM benchmark experiments.

No dependencies, model calls, hidden task data, or training. Run with Python 3.
The optional --output path saves the JSON result. These examples show possible
mechanisms and failure modes; they do not establish their prevalence in tau.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def tilt(prior, values, temperature):
    if temperature <= 0 or len(prior) != len(values):
        raise ValueError("Invalid target parameters")
    if any(p <= 0 for p in prior) or not math.isclose(sum(prior), 1.0):
        raise ValueError("Prior must be positive and normalized")
    logits = [math.log(p) + v / temperature for p, v in zip(prior, values)]
    mx = max(logits)
    weights = [math.exp(x - mx) for x in logits]
    return [x / sum(weights) for x in weights]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def acquisition_witness(observation_skill=.8):
    # Hidden bit z has equal prior. Both worlds have the identical public prefix.
    # WRITE_0/1 ends immediately; READ reveals z and then the student writes.
    # A full-information teacher writes z directly, a decision the unprivileged
    # student cannot reproduce better than 50% without first obtaining evidence.
    behavior = [.45, .45, .10]
    values = [.5, .5, observation_skill]
    target = tilt(behavior, values, .1)
    return {
        "world_prior": [.5, .5],
        "actions": ["WRITE_0", "WRITE_1", "READ_then_student"],
        "behavior": behavior,
        "student_continuation_values": values,
        "clairvoyant_action_imitation_success": .5,
        "behavior_success": dot(behavior, values),
        "outcome_tilt_temperature": .1,
        "outcome_tilt": target,
        "outcome_tilt_success": dot(target, values),
    }


def evidence_sensitivity_witness():
    # At two equally likely, valid post-observation histories the correct action
    # reverses. The *action log-odds* capability delta has a common nuisance shift.
    # Centering across the matched histories removes this chosen nuisance exactly.
    nuisance, semantic = 3., 2.
    logits = [nuisance + semantic, nuisance - semantic]
    centered = [x - sum(logits) / len(logits) for x in logits]
    raw_success = (sigmoid(logits[0]) + 1 - sigmoid(logits[1])) / 2
    centered_success = (sigmoid(centered[0]) + 1 - sigmoid(centered[1])) / 2
    return {
        "behavior_action_prior": [.5, .5],
        "log_odds_delta_by_observation": logits,
        "centered_log_odds_delta": centered,
        "raw_success": raw_success,
        "centered_success": centered_success,
        "assumption": "The common shift is nuisance, not legitimate task prior.",
    }


def legitimate_prior_counterexample():
    # Here the world-invariant term is useful prior knowledge, not style.
    # P(z=1)=.9, and the binary observation is correct with probability .6.
    # Bayesian MAP always writes 1; discarding prior uses observation alone.
    prior_one, signal_accuracy = .9, .6
    prior_odds = math.log(prior_one / (1 - prior_one))
    likelihood_odds = math.log(signal_accuracy / (1 - signal_accuracy))
    outcomes = []
    bayes_accuracy = centered_accuracy = 0.
    for z in (0, 1):
        for observation in (0, 1):
            pz = prior_one if z else 1 - prior_one
            po = signal_accuracy if observation == z else 1 - signal_accuracy
            mass = pz * po
            evidence = likelihood_odds if observation else -likelihood_odds
            bayes_action = int(prior_odds + evidence > 0)
            centered_action = int(evidence > 0)
            bayes_accuracy += mass * (bayes_action == z)
            centered_accuracy += mass * (centered_action == z)
            outcomes.append({"hidden_world": z, "observation": observation,
                             "mass": mass, "bayes_action": bayes_action,
                             "action_after_prior_removal": centered_action})
    return {"world_prior_one": prior_one, "observation_accuracy": signal_accuracy,
            "bayesian_MAP_accuracy": bayes_accuracy,
            "accuracy_after_removing_common_prior": centered_accuracy,
            "outcomes": outcomes,
            "lesson": "Matched-world centering is not universally beneficial; identify nuisance empirically."}


def world_is_consistent(public_history, hidden_world):
    # Simplified exact observation validator. A world may vary an unobserved
    # property, but cannot contradict a receipt already contained in history.
    for fact, value in public_history.items():
        if fact not in hidden_world or hidden_world[fact] != value:
            return False
    return True


def invalid_world_witness():
    history = {"reservation_id": "R0", "refundable": False}
    original = {"reservation_id": "R0", "refundable": False, "unobserved_fee": 10}
    valid_twin = {"reservation_id": "R0", "refundable": False, "unobserved_fee": 20}
    impossible_twin = {"reservation_id": "R0", "refundable": True, "unobserved_fee": 10}
    return {"public_history": history,
            "original_valid": world_is_consistent(history, original),
            "changed_unobserved_fact_valid": world_is_consistent(history, valid_twin),
            "changed_already_observed_fact_valid": world_is_consistent(history, impossible_twin),
            "lesson": "Reject twins inconsistent with any existing public evidence; labels and simulator must change together."}


def run():
    acquisition = acquisition_witness()
    sensitivity = evidence_sensitivity_witness()
    prior = legitimate_prior_counterexample()
    invalid = invalid_world_witness()
    untrained = acquisition_witness(.3)
    learned = acquisition_witness(.9)
    # Each check verifies a mathematical property of a finite constructed world.
    assert acquisition["outcome_tilt_success"] > acquisition["behavior_success"]
    assert acquisition["outcome_tilt"][2] > acquisition["behavior"][2]
    assert sensitivity["centered_success"] > sensitivity["raw_success"]
    assert math.isclose(prior["bayesian_MAP_accuracy"], .9)
    assert math.isclose(prior["accuracy_after_removing_common_prior"], .6)
    assert invalid["original_valid"] and invalid["changed_unobserved_fact_valid"]
    assert not invalid["changed_already_observed_fact_valid"]
    assert untrained["outcome_tilt"][2] < untrained["behavior"][2]
    assert learned["outcome_tilt"][2] > learned["behavior"][2]
    return {
        "experiment_kind": "exact hand-constructed finite-POMDP mechanism probes",
        "not_evidence_of": ["benchmark improvement", "prevalence of failures", "novelty", "LLM training success"],
        "information_acquisition": acquisition,
        "evidence_sensitive_delta": sensitivity,
        "useful_prior_removal_failure": prior,
        "invalid_counterfactual_failure": invalid,
        "current_policy_credit_can_suppress_prerequisite": {
            "before_learning_to_use_observations": untrained,
            "after_learning_to_use_observations": learned,
            "lesson": "Current-student branch Q measures exploitability now. Learn a useful suffix before penalizing its prerequisite action; backward curricula are established, not novel here."
        },
        "checks": "9 mathematical assertions passed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
