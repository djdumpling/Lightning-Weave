#!/usr/bin/env python3
"""Exact finite-space probes, not a task-accuracy experiment; Python stdlib only.

Run: python3 research/multiturn_algorithm/hierarchical_target_probe.py
All response/action groups here are exact by construction. Real code must verify
that retained reasoning does not distinguish the future policy state.
"""
import json
import math
from collections import defaultdict


def normalize(values):
    total = sum(values)
    if total <= 0 or any(x < 0 for x in values):
        raise ValueError("Need positive total and nonnegative weights")
    return [x / total for x in values]


def marginal(probabilities, groups):
    out = defaultdict(float)
    for probability, group in zip(probabilities, groups, strict=True):
        out[group] += probability
    return dict(out)


def project(probabilities, groups, action_target):
    masses = marginal(probabilities, groups)
    if set(masses) != set(action_target):
        raise ValueError("Action target must have exactly the represented groups")
    if abs(sum(action_target.values()) - 1) > 1e-12:
        raise ValueError("Action target must be normalized")
    if any(masses[g] <= 0 and action_target[g] > 0 for g in masses):
        raise ValueError("Projection cannot create absent support")
    return [p * action_target[g] / masses[g]
            for p, g in zip(probabilities, groups, strict=True)]


def kl(p, q):
    return sum(x * math.log(x / y) for x, y in zip(p, q, strict=True) if x)


def tilt(base, donors, weights, alpha):
    """Finite response/action-space LW ratio tilt; no Top16 approximation."""
    energies = [sum(w * math.log(d[i] / b)
                    for d, w in zip(donors, weights, strict=True)) / alpha
                for i, b in enumerate(base)]
    largest = max(energies)
    return normalize([b * math.exp(e - largest)
                      for b, e in zip(base, energies, strict=True)])


def check_close(left, right, tolerance=1e-12):
    assert abs(left - right) < tolerance, (left, right)


def projection_probe():
    groups = ["read", "read", "stop", "stop"]
    base = [.25] * 4
    energy = [0., 0., 4., -4.]
    donor = normalize([p * math.exp(d) for p, d in zip(base, energy)])
    target = {"read": .8, "stop": .2}
    projected = project(donor, groups, target)
    naive = normalize([p * target[g] for p, g in zip(donor, groups)])
    shifted = normalize([p * math.exp(7 if g == "stop" else 0)
                         for p, g in zip(donor, groups)])
    projected_shifted = project(shifted, groups, target)
    # Any other distribution with the prescribed action marginal.
    feasible = [.4, .4, .03, .17]
    pythagorean_error = kl(feasible, donor) - kl(feasible, projected) - kl(projected, donor)
    terminal_weight = [target[g] / marginal(donor, groups)[g] for g in groups]
    H_root = sum(p * weight for p, weight in zip(donor, terminal_weight))
    doob = [p * weight / H_root for p, weight in zip(donor, terminal_weight)]
    for g, mass in marginal(projected, groups).items():
        check_close(mass, target[g])
    for i in range(4):
        check_close(projected[i], projected_shifted[i])
        check_close(projected[i], doob[i])
    check_close(pythagorean_error, 0.)
    check_close(projected[2] / projected[3], donor[2] / donor[3], 1e-9)
    return {
        "original_action_mass": marginal(donor, groups),
        "naively_multiply_by_desired_action_mass": marginal(naive, groups),
        "exact_projected_action_mass": marginal(projected, groups),
        "projected_response_mass": projected,
        "group_energy_shift_before_projection": marginal(shifted, groups),
        "group_energy_shift_projection_error": max(abs(x-y) for x,y in zip(projected, projected_shifted)),
        "KL_pythagorean_error": pythagorean_error,
        "doob_transform_error": max(abs(x-y) for x,y in zip(projected, doob)),
    }


def composition_order_probe():
    # The two experts agree on the external action but prefer different rationales.
    groups = ["good", "good", "bad"]
    base = [1/3] * 3
    first = [.95, .001, .049]
    second = [.001, .95, .049]
    macro_base = list(marginal(base, groups).values())
    macro_first = list(marginal(first, groups).values())
    macro_second = list(marginal(second, groups).values())
    rows = []
    for weights, alpha in [([1., 1.], 1.), ([1., 1.], 2.), ([.5, .5], 2.)]:
        token_first = marginal(tilt(base, [first, second], weights, alpha), groups)
        action_first = tilt(macro_base, [macro_first, macro_second], weights, alpha)
        rows.append({
            "weights": weights, "alpha": alpha,
            "compose_responses_then_marginalize": token_first,
            "marginalize_then_compose_with_same_induced_base": dict(zip(["good", "bad"], action_first)),
        })
        assert action_first[0] > token_first["good"]
    return {"base_action_mass": marginal(base, groups),
            "each_donor_action_mass": marginal(first, groups), "configurations": rows}


def token_local_composition_probe():
    # Two-token autoregressive tree. At root choose a rationale token. At leaf
    # choose an action. ALL models share leaf conditionals, so leaf delta=0.
    # Every child is represented; this is an exact local target, not an omitted
    # future-normalizer approximation. With <=3 children it fits Top16 support.
    root_base = [1/3] * 3
    root_post1 = [.95, .001, .049]
    root_post2 = [.001, .95, .049]
    good_given_rationale = [.99, .99, .01]
    base_good = sum(p*v for p,v in zip(root_base, good_given_rationale))
    donor_good = sum(p*v for p,v in zip(root_post1, good_given_rationale))
    rows = []
    for weights, alpha in [([1., 1.], 1.), ([1., 1.], 2.), ([.5, .5], 2.)]:
        root = tilt(root_base, [root_post1, root_post2], weights, alpha)
        local_good = sum(p*v for p,v in zip(root, good_given_rationale))
        action_first = tilt([base_good, 1-base_good],
                            [[donor_good, 1-donor_good]]*2, weights, alpha)
        assert local_good < base_good < donor_good
        assert action_first[0] > local_good
        rows.append({"weights": weights, "alpha": alpha,
                     "local_target_rationale_distribution": root,
                     "token_local_composition_good_action": local_good,
                     "action_marginal_first_good_action": action_first[0]})
    return {"base_good_action": base_good,
            "each_donor_good_action": donor_good,
            "all_models_good_action_conditioned_on_rationale": good_given_rationale,
            "leaf_delta": "exactly zero for every represented token",
            "support": "3 root children and 2 leaf children, all represented",
            "configurations": rows}


def complementary_constraints_probe():
    # Each donor knows one binary requirement; action11 satisfies both.
    actions = ["11", "10", "01", "00"]
    marginal1 = [.45, .45, .05, .05]
    marginal2 = [.45, .05, .45, .05]
    # Only on the correct action, donors disagree about how to explain it.
    first_z0 = [.99, .5, .5, .5]
    second_z0 = [.01, .5, .5, .5]
    joint1 = [[a*z, a*(1-z)] for a,z in zip(marginal1, first_z0)]
    joint2 = [[a*z, a*(1-z)] for a,z in zip(marginal2, second_z0)]
    root1 = [sum(row[z] for row in joint1) for z in range(2)]
    root2 = [sum(row[z] for row in joint2) for z in range(2)]
    root_target = tilt([.5,.5], [root1,root2], [1.,1.], 1.)
    leaves = []
    for z in range(2):
        leaves.append(tilt([.25]*4,
                           [[row[z]/root1[z] for row in joint1],
                            [row[z]/root2[z] for row in joint2]],
                           [1.,1.], 1.))
    local_actions = [sum(root_target[z]*leaves[z][a] for z in range(2))
                     for a in range(4)]
    action_first = tilt([.25]*4, [marginal1,marginal2], [1.,1.], 1.)
    mixture = [(p+q)/2 for p,q in zip(marginal1,marginal2)]
    check_close(action_first[0], .81)
    check_close(mixture[0], .45)
    assert local_actions[0] < mixture[0] < action_first[0]
    return {"weights": [1.,1.], "alpha": 1.,
            "donor1_action_marginal": dict(zip(actions,marginal1)),
            "donor2_action_marginal": dict(zip(actions,marginal2)),
            "arithmetic_donor_mixture": dict(zip(actions,mixture)),
            "token_local_product": dict(zip(actions,local_actions)),
            "action_first_product": dict(zip(actions,action_first)),
            "purpose": "A complementary-constraint witness where an action product can outperform a trivial donor mixture."}


def reasoning_retention_counterexample():
    # Same tool call, but different private thoughts survive in the next prompt.
    candidate_reasoning = [.9, .1]
    baseline_reasoning = [.1, .9]
    response_success = [0., 1.]
    return {
        "public_action_mass_both_policies": 1.,
        "baseline_success": sum(p*r for p,r in zip(baseline_reasoning, response_success)),
        "donor_conditional_success": sum(p*r for p,r in zip(candidate_reasoning, response_success)),
        "lesson": "Equal external action marginals do not preserve return when reasoning persists in the next policy state.",
    }


def public_information_probe():
    worlds = {
        "world_0": {"guess_0": 1., "guess_1": 0., "read": .9},
        "world_1": {"guess_0": 0., "guess_1": 1., "read": .9},
    }
    expected = {a: sum(w[a] for w in worlds.values()) / len(worlds)
                for a in worlds["world_0"]}
    return {
        "world_values": worlds,
        "per_world_best_action": {w: max(v, key=v.get) for w, v in worlds.items()},
        "values_conditioned_on_shared_public_history": expected,
        "best_feasible_shared_action": max(expected, key=expected.get),
        "success_of_imitation_of_privileged_argmax": .5,
        "success_of_read_then_observe": .9,
    }


def main():
    print(json.dumps({
        "scope": "Finite mathematical witnesses, not empirical evidence of task gains.",
        "projection": projection_probe(),
        "composition_order": composition_order_probe(),
        "token_local_composition": token_local_composition_probe(),
        "complementary_constraints": complementary_constraints_probe(),
        "reasoning_retention": reasoning_retention_counterexample(),
        "information_acquisition": public_information_probe(),
    }, indent=2))


if __name__ == "__main__":
    main()
