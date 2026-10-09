#!/usr/bin/env python3
"""Exact autoregressive comparisons: token-local, response-product, action-marginal.

These finite-tree calculations demonstrate possibilities, not production defects.
All candidate tokens fit within top-16. No model/benchmark/GPU calls are made.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path


def normalized(log_weights):
    peak = max(log_weights.values())
    raw = {key: math.exp(value-peak) for key, value in log_weights.items()}
    total = sum(raw.values())
    return {key: value/total for key, value in raw.items()}


def joint(policy):
    return {(first, second): p1*p2
            for first, p1 in policy[()].items()
            for second, p2 in policy[(first,)].items()}


def marginal(distribution, action_index):
    result = {}
    for sequence, probability in distribution.items():
        action = sequence[action_index]
        result[action] = result.get(action, 0.0) + probability
    return result


def evaluate_tree(base, pres, posts, action_index, weight=1.0):
    local = {}
    for prefix, probabilities in base.items():
        local[prefix] = normalized({
            token: math.log(probability) + weight*sum(
                math.log(post[prefix][token])-math.log(pre[prefix][token])
                for pre, post in zip(pres, posts))
            for token, probability in probabilities.items()
        })
    base_joint, pre_joints, post_joints = joint(base), list(map(joint,pres)), list(map(joint,posts))
    response_product = normalized({
        sequence: math.log(probability)+weight*sum(
            math.log(post[sequence])-math.log(pre[sequence])
            for pre, post in zip(pre_joints,post_joints))
        for sequence, probability in base_joint.items()
    })
    base_actions = marginal(base_joint, action_index)
    pre_actions = [marginal(p,action_index) for p in pre_joints]
    post_actions = [marginal(p,action_index) for p in post_joints]
    action_target = normalized({
        action: math.log(probability)+weight*sum(
            math.log(post[action])-math.log(pre[action])
            for pre, post in zip(pre_actions,post_actions))
        for action, probability in base_actions.items()
    })
    return {
        "each_donor_weight_over_alpha": weight,
        "base_action_marginal": base_actions,
        "donor_post_action_marginals": post_actions,
        "token_local_action_marginal": marginal(joint(local),action_index),
        "response_product_action_marginal": marginal(response_product,action_index),
        "marginalize_then_compose_action_target": action_target,
        "token_local_root": local[()],
    }


def rationale_before_action():
    conditional = {"R1": .99, "R2": .99, "R0": .01}
    base = {(): dict.fromkeys(conditional,1/3)}
    for rationale, good in conditional.items():
        base[(rationale,)] = {"Good":good,"Bad":1-good}
    post1, post2 = dict(base), dict(base)
    post1[()] = {"R1":.95,"R2":.001,"R0":.049}
    post2[()] = {"R1":.001,"R2":.95,"R0":.049}
    results = [evaluate_tree(base,[base,base],[post1,post2],1,w) for w in (1.,.5)]
    for row in results:
        assert row["token_local_action_marginal"]["Good"] < row["base_action_marginal"]["Good"]
        assert abs(row["token_local_action_marginal"]["Good"] -
                   row["response_product_action_marginal"]["Good"]) < 1e-12
        assert row["marginalize_then_compose_action_target"]["Good"] > .94
    return {
        "description": "Donors prefer different correct rationale modes; shared incorrect mode survives their product.",
        "root_post1": post1[()],
        "root_post2": post2[()],
        "good_probability_given_rationale_under_every_model":conditional,
        "results":results,
    }


def style_after_action():
    base = {():{"Good":.5,"Bad":.5},
            ("Good",):{"X":.5,"Y":.5},
            ("Bad",):{"X":.5,"Y":.5}}
    post1, post2 = dict(base),dict(base)
    post1[()] = {"Good":.95,"Bad":.05}
    post2[()] = {"Good":.95,"Bad":.05}
    post1[("Good",)] = {"X":.9999,"Y":.0001}
    post2[("Good",)] = {"X":.0001,"Y":.9999}
    results = [evaluate_tree(base,[base,base],[post1,post2],0,w) for w in (1.,.5)]
    for row in results:
        assert row["token_local_action_marginal"]["Good"] > .94
        assert row["response_product_action_marginal"]["Good"] < .28
        assert abs(row["token_local_action_marginal"]["Good"] -
                   row["marginalize_then_compose_action_target"]["Good"]) < 1e-12
    return {
        "description":"Irrelevant stylistic disagreement occurs after the semantic decision; local normalization protects that decision.",
        "root_post1_and_post2":post1[()],
        "style_after_good_post1":post1[("Good",)],
        "style_after_good_post2":post2[("Good",)],
        "results":results,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    result = {
        "kind":"exact autoregressive toy; not measured production behavior",
        "rationale_before_action":rationale_before_action(),
        "style_after_action":style_after_action(),
        "semantic_equivalence_assumption":"The action terminates the episode, or rationale/style is erased before subsequent policy state.",
        "production_caveat":"Retained reasoning can change subsequent policy state. Finite exact support does not model top-16 tail approximation, gradient approximation, or training generalization.",
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"token_tree_probe_results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()

