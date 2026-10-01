"""The agent-efficiency registry: pins, provenance, the primary matrix, and geometry directions."""

import importlib.util
import sys
from pathlib import Path

import pytest

from data_curation.build_synthetic_shift_target import SyntheticComposer
from data_curation.shift_states import StateLabeler, TokenTable

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


config = load("agent_eff_config", "configs/agent_eff/config.py")
bfcl = load("bfcl_eval_config_for_agent_eff", "configs/bfcl_eval/config.py")


def tiny_labeler():
    return StateLabeler(
        TokenTable.from_vocab(
            {"a": 0}, {"<think>": 1, "</think>": 2, "<tool_call>": 3, "</tool_call>": 4, "<|im_end|>": 5}
        )
    )


def test_every_model_is_pinned_to_a_full_revision():
    models = [config.AGENT_ACC.pre, config.AGENT_ACC.post]
    models += [model for pair in config.DONORS.values() for model in (pair.pre, pair.post)]
    for model in models:
        assert len(model.revision) == 40 and all(char in "0123456789abcdef" for char in model.revision), model


def test_donor_provenance_and_aliases():
    assert {name for name, pair in config.DONORS.items() if pair.core} == {"klear", "decs", "deepscaler"}
    assert config.DONORS["decs"].pre == config.DONORS["deepscaler"].pre
    assert config.DONORS["deepcoder"].pre == config.R1D_1P5B  # DeepCoder-1.5B was trained from R1-Distill-1.5B
    assert config.DONORS["l1max"].pre == config.DONORS["deepscaler"].post
    for pair in config.DONORS.values():
        targets = list(pair.aliases.values())
        assert len(targets) == len(set(targets)), f"{pair.name} aliases must be injective"
        assert pair.aliases or "DeepSeek" not in pair.pre.repo


def test_the_primary_matrix_is_small_and_pre_registered():
    primary = config.primary_specs()
    assert set(primary) == {
        "acc-legacy",
        "acc-clean",
        "acc-legacy+decs",
        "acc-legacy+decs-deepscaler",
        "acc-clean+decs",
        "acc-clean+decs-deepscaler",
        "acc-clean+decs-deepscaler-flipped",
    }
    budget = config.KL_BUDGETS[config.PRIMARY_BUDGET]
    assert all(term.get("kl_budget") in (None, budget) for spec in primary.values() for term in spec["terms"])
    assert not set(primary) & set(config.exploratory_specs())
    flipped = primary["acc-clean+decs-deepscaler-flipped"]["terms"][1]
    assert (
        flipped["coef"] == -1.0
        and flipped["direction"] == primary["acc-clean+decs-deepscaler"]["terms"][1]["direction"]
    )


def test_controls_and_follow_ups_sit_on_the_legacy_accuracy_base():
    controls, exploratory = config.control_specs(), config.exploratory_specs()
    (scaled,) = controls["acc-legacy-scaled"]["terms"]
    assert scaled["transform"] == "raw" and scaled["coef"] == config.SCALED_ACCURACY_COEF
    accuracy, heuristic = controls["acc-legacy+heuristic"]["terms"]
    assert accuracy == config.ACCURACY_TERMS["acc-legacy"] and heuristic["transform"] == "heuristic"
    assert {"acc-legacy+decs@low", "acc-legacy+decs@high"} <= set(exploratory)
    assert all(spec["terms"][0] == config.ACCURACY_TERMS["acc-legacy"] for spec in exploratory.values())
    names = [*config.primary_specs(), *controls, *exploratory]
    assert len(names) == len(set(names))


def test_donor_round_uses_raw_shifts_at_the_primary_budget():
    donors = config.donor_specs()
    budget = config.KL_BUDGETS[config.PRIMARY_BUDGET]
    assert set(donors) == {"acc-legacy+decs7b", "acc-legacy+l1max", "acc-legacy+nemotron", "acc-legacy+donor-mean"}
    for spec in donors.values():
        accuracy, efficiency = spec["terms"]
        assert accuracy == config.ACCURACY_TERMS["acc-legacy"]
        assert efficiency["transform"] == "evidence" and efficiency["kl_budget"] == budget
    mean = donors["acc-legacy+donor-mean"]["terms"][1]
    assert mean["equalize"] and [t["source"] for t in mean["direction"]["terms"]] == list(config.TRANSFER_DONORS)
    # The single-donor arms have the same form as the primary acc-legacy+decs arm.
    assert donors["acc-legacy+l1max"]["terms"][1]["direction"] == {"terms": [{"source": "l1max", "coef": 1.0}]}
    assert config.primary_specs()["acc-legacy+decs"]["terms"][1]["direction"] == {
        "terms": [{"source": "decs", "coef": 1.0}]
    }


@pytest.mark.parametrize("name", sorted(config.variant_specs()))
def test_every_variant_spec_builds_a_composer(name):
    spec = config.variant_specs()[name]
    weights = {
        term["gate"]["prompt_weights"]: {} for term in spec["terms"] if "prompt_weights" in term.get("gate", {})
    }
    composer = SyntheticComposer(spec, labeler=tiny_labeler(), vocab_size=8, prompt_weights=weights)
    # every arm uses the LoopTool recipe's alpha, which is also Lightning Weave's
    assert composer.alpha == config.RECIPE.alpha == 2.0
    assert composer.sources <= {"agent_acc", *config.DONORS}
    legacy = [term for term in composer.terms if term.transform == "raw"]
    assert all(term.direction.sources == ("agent_acc",) for term in legacy)


def test_the_efficiency_basis_holds_only_efficiency_contrasts():
    core = config.geometry_spec(["klear", "decs", "deepscaler"])
    assert core["basis"] == ["decs_minus_deepscaler"] and "dler_minus_deepscaler" not in core["directions"]
    census = config.geometry_spec(list(config.DONORS))
    assert set(census["basis"]) == set(config.EFFICIENCY_BASIS)
    assert not {"agent_acc", "klear", "decs", "decs_minus_klear"} & set(census["basis"])


def test_bfcl_tags_resolve_agent_efficiency_students_by_seed():
    default = bfcl.resolve_model("ae.joint.acc-clean+decs-deepscaler")
    assert (
        default.path
        == f"{config.CHECKPOINT_ROOT}/joint/acc-clean+decs-deepscaler/seed{config.RECIPE.training_seed}/hf"
    )
    assert bfcl.resolve_model("ae.joint.acc-clean.s7").path.endswith("/acc-clean/seed7/hf")
    assert bfcl.AGENT_EFF_DEFAULT_SEED == config.RECIPE.training_seed
    with pytest.raises(ValueError, match="unknown"):
        bfcl.resolve_model("ae.joint")


def test_code_round_arms_are_the_pre_registered_compositions():
    specs = config.code_round_specs()
    assert set(specs) == set(config.CODE_ROUND_SEEDS) and set(specs) <= set(config.variant_specs())
    sources = {
        "acc-legacy+e1code": {"e1code"},
        "acc-legacy+e1math": {"e1math"},
        "acc-legacy+e1math+e1code": {"e1math", "e1code"},
        "acc-legacy+decs+e1code": {"decs", "e1code"},
    }
    decs = config.variant_specs()["acc-legacy+decs"]
    for name, spec in specs.items():
        assert spec["alpha"] == config.RECIPE.alpha == decs["alpha"]
        accuracy, efficiency = spec["terms"]
        assert accuracy == decs["terms"][0] == config.ACCURACY_TERMS["acc-legacy"]
        assert {item["source"] for item in efficiency["direction"]["terms"]} == sources[name]
        assert all(item["coef"] == 1.0 for item in efficiency["direction"]["terms"])
        # the same efficiency-term rule as the DECS arm: evidence transform, weight 1, the primary KL budget
        assert efficiency["transform"] == decs["terms"][1]["transform"] == "evidence"
        assert efficiency["coef"] == 1.0 and efficiency["kl_budget"] == decs["terms"][1]["kl_budget"]
        assert (efficiency.get("equalize") == "auto") == (len(sources[name]) == 2)
        assert config.variant_sources(spec) == {"agent_acc", *sources[name]}
    assert config.CODE_ROUND_SEEDS["acc-legacy+e1code"] == (config.RECIPE.training_seed, 5678)
    assert all(seeds[0] == config.RECIPE.training_seed for seeds in config.CODE_ROUND_SEEDS.values())


def test_e1_pairs_subtract_the_accuracy_model_each_was_trained_from():
    assert config.DONORS["e1code"].pre == config.DEEPCODER_14B and config.DONORS["e1code"].post == config.E1_CODE
    assert config.DONORS["e1math"].pre == config.DEEPSCALER == config.DONORS["deepscaler"].post
    for name in config.CODE_ROUND_PAIRS:
        assert config.DONORS[name].aliases == config.DONORS["decs"].aliases == config.R1_ALIASES
        assert config.DONORS[name].role == "efficiency" and not config.DONORS[name].core
    assert {"e1code", "e1math"} <= set(config.EFFICIENCY_BASIS)


def test_pair_and_tail_checks():
    def evidence(act=0.008, **values):
        state = {"behavior_mass_evidence": 1.0, "support_median": 0.97, "support_p10": 0.6} | values
        return {"by_state": {config.PAIR_GATES["state"]: state, "act_vs_talk": {"behavior_mass_evidence": act}}}

    assert config.pair_gate(evidence(), {"fisher_correlation": 0.95})["passed"]
    assert not config.pair_gate(evidence(), {"fisher_correlation": 0.5})["passed"]
    for failing in ({"support_median": 0.5}, {"support_p10": 0.1}, {"behavior_mass_evidence": 0.5}):
        assert not config.pair_gate(evidence(**failing), {"fisher_correlation": 0.99})["passed"]
    # a pair that is not silent at the call-or-reply decision (untrained <tool_call> rows missed) fails
    assert not config.pair_gate(evidence(act=1.0), {"fisher_correlation": 0.99})["passed"]
    unmeasured = config.pair_gate(evidence(), None)
    assert unmeasured["passed"] and unmeasured["checks"]["precision_correlation"] == {"value": None, "ok": None}
    assert not config.pair_gate(evidence(support_p10=0.1), None)["passed"]
    reference = {"kl_quantiles": {"99": 0.25}}
    assert config.tail_gate({"kl_quantiles": {"99": 1.0}}, reference)["passed"]
    assert not config.tail_gate({"kl_quantiles": {"99": 1.01}}, reference)["passed"]


def test_bfcl_tags_resolve_code_round_students():
    for variant, seeds in config.CODE_ROUND_SEEDS.items():
        for seed in seeds:
            tag = f"ae.joint.{variant}" + ("" if seed == config.RECIPE.training_seed else f".s{seed}")
            assert bfcl.resolve_model(tag).path == f"{config.CHECKPOINT_ROOT}/joint/{variant}/seed{seed}/hf"


def test_followups_change_one_thing_each_from_the_e1math_arm():
    specs, e1math = config.followup_specs(), config.code_round_specs()["acc-legacy+e1math"]
    assert set(specs) == set(config.FOLLOWUP_SEEDS) and set(specs) <= set(config.variant_specs())
    manifest_coef = 7.9454246558728014  # synthetic/acc-legacy+e1math/manifest.json calibration.effective_coef
    protected, low = specs["acc-legacy+e1math-protected"], specs["acc-legacy+e1math@low"]
    assert protected["terms"][0] == low["terms"][0] == e1math["terms"][0]
    # protection: the same direction and coefficient, no recalibration, silent at the protocol states
    efficiency = protected["terms"][1]
    assert efficiency["direction"] == e1math["terms"][1]["direction"] and efficiency["transform"] == "evidence"
    assert efficiency["coef"] == manifest_coef == config.E1MATH_MID_COEF and "kl_budget" not in efficiency
    assert efficiency["gate"] == {"exclude_state_types": ["act_vs_talk", "call_boundary"]}
    # the low budget: everything else as the E1-Math arm
    changed = {k for k in e1math["terms"][1] if e1math["terms"][1][k] != low["terms"][1].get(k)}
    assert changed == {"kl_budget"} and low["terms"][1]["kl_budget"] == config.KL_BUDGETS["low"]
    for variant, seeds in config.FOLLOWUP_SEEDS.items():
        assert seeds == (config.RECIPE.training_seed, config.REPLICATE_SEED)
        for seed in seeds:
            tag = f"ae.joint.{variant}" + ("" if seed == config.RECIPE.training_seed else f".s{seed}")
            assert bfcl.resolve_model(tag).path == f"{config.CHECKPOINT_ROOT}/joint/{variant}/seed{seed}/hf"


def test_paper_arms_use_lightning_weaves_alpha_normalized_weights_and_two_passes():
    specs, decs = config.paper_specs(), config.variant_specs()["acc-legacy+decs"]
    assert set(specs) == {"paper-acc-legacy", "paper-acc-legacy+decs", "paper-half-acc-legacy"}
    assert set(specs) <= set(config.variant_specs()) and set(specs) <= set(config.NEXT_SEEDS)
    accuracy = decs["terms"][0]
    half = {**accuracy, "coef": 0.5}
    assert all(item["alpha"] == 2.0 for item in specs.values())
    # accuracy alone at weight 1 (the Klear-only analog), the composition 0.5 + 0.5, and its accuracy half alone
    assert specs["paper-acc-legacy"]["terms"] == [accuracy] and specs["paper-half-acc-legacy"]["terms"] == [half]
    composed = specs["paper-acc-legacy+decs"]["terms"]
    assert composed[0] == half and sum(item["coef"] for item in composed) == 1.0
    efficiency = composed[1]
    # DECS at weight 0.5 (no KL calibration), with the evidence transform every DECS arm here uses
    assert efficiency["direction"] == decs["terms"][1]["direction"] and efficiency["coef"] == 0.5
    assert efficiency["transform"] == "evidence" and "kl_budget" not in efficiency and "gate" not in efficiency
    for variant in specs:
        plan = config.training_plan(variant)
        assert plan["passes"] == 2 and plan["alpha"] == 2.0
        # the released default of 400 updates: 100 batches of 256 rows from a twice-repeated 12,800-row target
        assert plan["num_rollout"] * config.RECIPE.rollout_batch_size // config.RECIPE.global_batch_size == 400
        assert (
            plan["rows"]
            == plan["num_rollout"] * config.RECIPE.rollout_batch_size
            == 2 * config.RECIPE.cached_trajectories
        )
        assert plan["final_iteration"] == 99 and plan["data"] == plan["target"] + "-x2"


def test_every_other_arm_keeps_one_pass_at_the_recipe_alpha():
    for variant in config.variant_specs():
        if variant in config.TRAINING_PASSES:
            continue
        plan = config.training_plan(variant)
        assert plan["passes"] == 1 and plan["alpha"] == config.RECIPE.alpha == 2.0, variant
        assert plan["num_rollout"] == config.RECIPE.replay_rounds and plan["final_iteration"] == config.FINAL_ITERATION
        assert plan["data"] == plan["target"] == f"{config.SYNTHETIC_ROOT}/{variant}"
        assert plan["rows"] == config.RECIPE.cached_trajectories


def test_turn_start_arms_differ_from_decs_mid_only_in_which_prompts_lose_decs():
    from data_curation.turn_positions import RULES

    specs, decs = config.turn_start_specs(), config.variant_specs()["acc-legacy+decs"]
    rules = {"acc-legacy+decs-protect-turn-starts": config.TURN_START_RULE}
    rules["acc-legacy+decs-gate-random-multi-turn"] = config.RANDOM_MULTI_TURN_RULE
    assert (
        set(specs) == set(rules) and set(specs) <= set(config.variant_specs()) and set(specs) <= set(config.NEXT_SEEDS)
    )
    assert set(rules.values()) == set(RULES)
    for name, item in specs.items():
        assert item["alpha"] == decs["alpha"] and item["terms"][0] == decs["terms"][0]
        efficiency = item["terms"][1]
        # DECS at its mid coefficient, fixed (no recalibration), gated off on whole prompts by the arm's rule
        assert efficiency["direction"] == decs["terms"][1]["direction"] and efficiency["coef"] == config.DECS_MID_COEF
        assert efficiency["gate"] == {"prompt_weights": rules[name]} and "kl_budget" not in efficiency
        assert config.training_plan(name)["passes"] == 1
    for variant, seeds in config.NEXT_SEEDS.items():
        assert seeds == (config.RECIPE.training_seed, config.REPLICATE_SEED)
        for seed in seeds:
            tag = f"ae.joint.{variant}" + ("" if seed == config.RECIPE.training_seed else f".s{seed}")
            assert bfcl.resolve_model(tag).path == f"{config.CHECKPOINT_ROOT}/joint/{variant}/seed{seed}/hf"


def test_next_comparisons_pair_runs_at_the_same_seeds():
    import json
    from dataclasses import replace

    comparisons = json.loads((ROOT / "configs/agent_eff/next_comparisons.json").read_text())
    decoding = {bfcl.run_id(0, replace(bfcl.PROTOCOL, seed=seed)): seed for seed in (0, *config.EXTRA_DECODING_SEEDS)}
    assert decoding[bfcl.run_id()] == 0  # the default tree holds every earlier run
    used = set()
    for name, item in comparisons.items():
        assert item["pairs"], name
        for pair in item["pairs"]:
            assert len(pair) == 2, name
            seeds = []
            for path in pair:
                tree, tag = path.split("/")
                assert bfcl.resolve_model(tag).path.startswith(f"{config.CHECKPOINT_ROOT}/joint/"), path
                variant = tag.removeprefix("ae.joint.").removesuffix(f".s{config.REPLICATE_SEED}")
                assert variant in config.variant_specs(), path
                if decoding[tree]:
                    assert variant in config.DECODING_REPLICATES, path
                used.add(variant)
                seeds.append((decoding[tree], tag.endswith(f".s{config.REPLICATE_SEED}")))
            (arm_decoding, arm_training), (reference_decoding, reference_training) = seeds
            # an arm is compared with its reference at the same decoding and training seeds, except in the nulls
            assert arm_decoding == reference_decoding or name.startswith("null, decoding seed"), (name, pair)
            assert arm_training == reference_training or name.startswith("null, training seed"), (name, pair)
    assert set(config.NEXT_SEEDS) <= used
