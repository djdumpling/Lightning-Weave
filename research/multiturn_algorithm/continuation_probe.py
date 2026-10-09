#!/usr/bin/env python3
"""Exact finite-MDP probes for continuation-calibrated checkpoint-delta transfer.

Standard library only; no model calls, benchmark data, training, or external reward
queries. Synthetic task rewards are specified to audit the model-only objective.
Run: python3 research/multiturn_algorithm/continuation_probe.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

TOL = 1e-10
STATES = ("challenge", "root")  # backward topological order


def normalize_logweights(values):
    largest = max(values)
    weights = [math.exp(value - largest) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def logsumexp(values):
    largest = max(values)
    return largest + math.log(sum(math.exp(value - largest) for value in values))


def transition(state, action):
    if state == "root" and action == 1:
        return "challenge"
    return None


def soft_optimal(reference, reward, alpha=1.0):
    value, policy = {}, {}
    for state in STATES:
        logits = []
        for action in range(2):
            successor = transition(state, action)
            future = value[successor] if successor else 0.0
            logits.append(math.log(reference[state][action]) +
                          (reward[state][action] + future) / alpha)
        value[state] = alpha * logsumexp(logits)
        policy[state] = normalize_logweights(logits)
    return policy, value


def evaluate(policy, reward, reference=None, alpha=1.0):
    value = {}
    for state in STATES:
        value[state] = 0.0
        for action, probability in enumerate(policy[state]):
            successor = transition(state, action)
            immediate = reward[state][action]
            if reference is not None:
                immediate -= alpha * math.log(probability / reference[state][action])
            value[state] += probability * (immediate + (value[successor] if successor else 0.0))
    return value


def local_transplant(reference, delta, alpha):
    target, log_z = {}, {}
    for state in STATES:
        logits = [math.log(reference[state][action]) + delta[state][action] / alpha
                  for action in range(2)]
        target[state] = normalize_logweights(logits)
        log_z[state] = logsumexp(logits)
    return target, log_z


def evaluate_from_normalizers(policy, local_target, log_z, alpha):
    """E_pi[alpha log Z - alpha KL(pi || q0)] is the same regularized reward."""
    reward = {}
    for state in STATES:
        kl = sum(probability * math.log(probability / local_target[state][action])
                 for action, probability in enumerate(policy[state]))
        # This is the expected immediate reward, repeated across actions.
        reward[state] = [alpha * (log_z[state] - kl)] * 2
    return evaluate(policy, reward)


def improve(reference, delta, previous_value, alpha):
    target = {}
    for state in STATES:
        logits = []
        for action in range(2):
            successor = transition(state, action)
            future = previous_value[successor] if successor else 0.0
            logits.append(math.log(reference[state][action]) +
                          (delta[state][action] + future) / alpha)
        target[state] = normalize_logweights(logits)
    return target


def close(first, second):
    return abs(first - second) <= TOL


def case(donor_success=0.01, student_success=0.99, safe_reward=0.2, alpha=1.0):
    donor_pre = {"root": [0.5, 0.5], "challenge": [1-donor_success, donor_success]}
    base = {"root": [0.5, 0.5], "challenge": [1-student_success, student_success]}
    # At challenge: action 0 fails, action 1 succeeds. Safe root action terminates.
    task_reward = {"root": [safe_reward, 0.0], "challenge": [0.0, 1.0]}
    donor_post, _ = soft_optimal(donor_pre, task_reward, 1.0)
    delta = {state: [math.log(donor_post[state][a] / donor_pre[state][a])
                     for a in range(2)] for state in STATES}
    local, log_z = local_transplant(base, delta, alpha)
    corrected, _ = soft_optimal(base, delta, alpha)
    true_optimal, _ = soft_optimal(base, task_reward, alpha)

    local_value = evaluate(local, delta, base, alpha)
    normalizer_value = evaluate_from_normalizers(local, local, log_z, alpha)
    base_value = evaluate(base, delta, base, alpha)
    base_normalizer_value = evaluate_from_normalizers(base, local, log_z, alpha)
    one_improvement = improve(base, delta, normalizer_value, alpha)
    correction_value = evaluate(corrected, delta, base, alpha)
    improvement_value = evaluate(one_improvement, delta, base, alpha)

    for state in STATES:
        assert close(local_value[state], normalizer_value[state])
        assert close(base_value[state], base_normalizer_value[state])
        assert improvement_value[state] + TOL >= local_value[state]
        assert close(improvement_value[state], correction_value[state])
        for action in range(2):
            # Full donor delta returns preserve donor reward in this exact setting.
            assert close(corrected[state][action], true_optimal[state][action])
            assert close(one_improvement[state][action], corrected[state][action])
        if close(donor_success, student_success) and close(alpha, 1.0):
            assert close(log_z[state], 0.0)
            assert close(normalizer_value[state], 0.0)
            for action in range(2):
                assert close(local[state][action], donor_post[state][action])
                assert close(corrected[state][action], donor_post[state][action])

    base_return = evaluate(base, task_reward)["root"]
    local_return = evaluate(local, task_reward)["root"]
    corrected_return = evaluate(corrected, task_reward)["root"]
    return {
        "donor_pre_challenge_success": donor_success,
        "student_base_challenge_success": student_success,
        "safe_reward": safe_reward,
        "alpha": alpha,
        "root_challenge_probability_donor_post": donor_post["root"][1],
        "root_challenge_probability_local": local["root"][1],
        "root_challenge_probability_corrected": corrected["root"][1],
        "student_challenge_success_after_transfer": local["challenge"][1],
        "base_task_return": base_return,
        "local_task_return": local_return,
        "corrected_task_return": corrected_return,
        "base_surrogate_value": base_value["root"],
        "local_surrogate_value": local_value["root"],
        "corrected_surrogate_value": correction_value["root"],
        "local_log_z_root": log_z["root"],
        "local_log_z_challenge": log_z["challenge"],
        "normalizer_return_root": normalizer_value["root"],
        "local_harms_task": local_return + TOL < base_return,
        "correction_improves_task_over_local": corrected_return > local_return + TOL,
    }


def stochastic_transition_trap():
    """The agent cannot exponentially tilt an exogenous lottery outcome."""
    safe, win, chance, alpha = 0.6, 1.0, 0.5, 1.0
    lottery_mean = chance * win
    causal_probability = normalize_logweights([safe / alpha, lottery_mean / alpha])[1]
    lottery_log_mgf = alpha * math.log((1-chance) + chance * math.exp(win/alpha))
    noncausal_probability = normalize_logweights([safe/alpha, lottery_log_mgf/alpha])[1]
    tilted_win = chance * math.exp(win/alpha) / ((1-chance) + chance*math.exp(win/alpha))
    assert causal_probability < 0.5 < noncausal_probability
    assert tilted_win > chance
    return {
        "safe_reward": safe,
        "lottery_success_probability_fixed_by_environment": chance,
        "lottery_expected_reward": lottery_mean,
        "correct_causal_soft_bellman_lottery_probability": causal_probability,
        "incorrect_path_tilt_lottery_probability": noncausal_probability,
        "incorrect_path_tilt_conditional_lottery_success": tilted_win,
        "interpretation": "Use E[next value] across exogenous transitions, not log E exp[next value].",
    }


def surrogate_misalignment():
    """Same exact donor pair, but the downstream objective disagrees with its reward."""
    witness = case()
    # Root-safe external reward is .8; challenge outcome reverses the donor reward.
    result = {
        "construction": "External reward equals 1 - donor task return on every complete episode.",
        "base_external_return": 1-witness["base_task_return"],
        "local_external_return": 1-witness["local_task_return"],
        "corrected_external_return": 1-witness["corrected_task_return"],
        "local_surrogate_value": witness["local_surrogate_value"],
        "corrected_surrogate_value": witness["corrected_surrogate_value"],
        "interpretation": "An exact improvement of checkpoint-ratio return can worsen external task return.",
    }
    assert result["corrected_surrogate_value"] > result["local_surrogate_value"]
    assert result["corrected_external_return"] < result["local_external_return"]
    return result


def high_normalizer_loop():
    """An objective-correct normalizer return can favor remaining in a loop."""
    b_stop, b_continue = 0.1, 0.9
    delta_stop, delta_continue = math.log(0.5/0.9), math.log(0.5/0.1)
    stop_weight = b_stop*math.exp(delta_stop)
    continue_weight = b_continue*math.exp(delta_continue)
    local_stop = stop_weight/(stop_weight+continue_weight)
    log_z = math.log(stop_weight+continue_weight)
    rows = []
    for horizon in (1, 2, 4, 8, 16):
        future_value = 0.0
        success = 0.0  # Only STOP before the time limit earns external success.
        expected_turns = 0.0
        for remaining in range(1, horizon+1):
            stop = stop_weight/(stop_weight+continue_weight*math.exp(future_value))
            cont = 1-stop
            future_value = math.log(stop_weight+continue_weight*math.exp(future_value))
            success = stop+cont*success
            expected_turns = 1+cont*expected_turns
        row = {
            "horizon": horizon,
            "local_stop_success": 1-(1-local_stop)**horizon,
            "corrected_stop_success": success,
            "corrected_expected_turns": expected_turns,
            "corrected_ratio_objective": future_value,
        }
        if horizon > 1:
            assert row["corrected_stop_success"] < row["local_stop_success"]
        rows.append(row)
    return {
        "donor_pre_stop_continue": [0.9, 0.1],
        "donor_post_stop_continue": [0.5, 0.5],
        "recipient_base_stop_continue": [0.1, 0.9],
        "alpha": 1.0,
        "local_stop_probability": local_stop,
        "log_z_per_live_state": log_z,
        "rows": rows,
        "interpretation": "Optimizing unnormalized checkpoint shifts rewards visiting high-Z states; this may reward loops or length rather than success.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    probabilities = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    safe_rewards = [0.1, 0.2, 0.4, 0.6, 0.8]
    alphas = [0.25, 0.5, 1.0, 2.0]
    sweep = [case(d, s, r, a) for d in probabilities for s in probabilities
             for r in safe_rewards for a in alphas]
    # Explicit zero-correction identity across the entire donor grid.
    identities = [case(p, p, r, 1.0) for p in probabilities for r in safe_rewards]
    stronger = [row for row in sweep if row["student_base_challenge_success"] >
                row["donor_pre_challenge_success"]]
    worst_local = min(sweep, key=lambda x: x["local_task_return"]-x["base_task_return"])
    largest_fix = max(sweep, key=lambda x: x["corrected_task_return"]-x["local_task_return"])
    largest_reward_tradeoff = min(sweep, key=lambda x: x["corrected_task_return"]-x["local_task_return"])
    report = {
        "kind": "exact synthetic finite-MDP calculation, not a learned-model benchmark",
        "witness": case(),
        "stochastic_transition_trap": stochastic_transition_trap(),
        "surrogate_misalignment": surrogate_misalignment(),
        "high_normalizer_loop": high_normalizer_loop(),
        "sweep": {
            "total_cases": len(sweep),
            "single_pair_same_reference_identity_checks": len(identities),
            "all_exact_normalizer_identity_and_soft_policy_improvement_checks_passed": True,
            "local_task_harm_cases": sum(row["local_harms_task"] for row in sweep),
            "stronger_recipient_cases": len(stronger),
            "stronger_recipient_local_task_harm_cases": sum(row["local_harms_task"] for row in stronger),
            "correction_improves_task_over_local_cases": sum(row["correction_improves_task_over_local"] for row in sweep),
            "correction_decreases_task_over_local_cases": sum(
                row["corrected_task_return"] + TOL < row["local_task_return"] for row in sweep),
            "largest_aligned_reward_tradeoff": largest_reward_tradeoff,
            "worst_local_regression": worst_local,
            "largest_corrected_task_gain": largest_fix,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "continuation_probe_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "continuation_probe_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

