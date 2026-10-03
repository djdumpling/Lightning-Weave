"""Pinned donors, paths, and variant specs for synthesizing an agent-efficiency shift.

The question: can a missing agent-efficiency shift be transferred from public
math/code policy shifts,

    r̂_agent,eff = r_agent,acc + λ · Σ_d β_d · ê_d,   ê_d = transform(r_d,eff − r_d,acc),

evaluated at the frozen student's own LoopTool token states? The construction
is motivated by KL-regularized RL (a converged policy's log-ratio to its
reference is a scaled advantage), but these donors are broad post-training runs
from different recipes, so a shift here is an empirical direction, not a
recovered reward. Every donor pair scores the existing sealed LoopTool cache
(``looptool-qwen3-4b-v1``); variants are composed on CPU and trained with the
LoopTool Offline Direct-OPD recipe (alpha 2.0, 50 replay rounds). Nothing here
imports Modal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


from configs.looptool_opd import config as looptool

MODAL_CHECKPOINT_VOLUME = looptool.MODAL_CHECKPOINT_VOLUME
MODAL_DATA_VOLUME = looptool.MODAL_DATA_VOLUME
MODAL_MODEL_VOLUME = looptool.MODAL_MODEL_VOLUME
POST_TEACHER_MODEL = looptool.POST_TEACHER_MODEL
POST_TEACHER_REVISION = looptool.POST_TEACHER_REVISION
PRE_TEACHER_MODEL = looptool.PRE_TEACHER_MODEL
PRE_TEACHER_REVISION = looptool.PRE_TEACHER_REVISION
RECIPE = looptool.RECIPE
LOOPTOOL_CHECKPOINT_ROOT = looptool.REMOTE_CHECKPOINT_ROOT
REMOTE_DATA_ROOT = looptool.REMOTE_DATA_ROOT
REMOTE_MODEL_ROOT = looptool.REMOTE_MODEL_ROOT
REMOTE_REPO = looptool.REMOTE_REPO
RUNTIME_IMAGE = looptool.RUNTIME_IMAGE
STUDENT_MODEL = looptool.STUDENT_MODEL
STUDENT_REVISION = looptool.STUDENT_REVISION

MODAL_APP_NAME = "lightning-weave-agent-eff"
BASE_CACHE = f"{REMOTE_DATA_ROOT}/anchor/final"
ROLLOUT_DIR = f"{REMOTE_DATA_ROOT}/rollouts"
PROMPT_DATA = f"{REMOTE_DATA_ROOT}/prompts.parquet"
CANONICAL_DATA = f"{REMOTE_DATA_ROOT}/curation/looptool_rl_canonical.jsonl"
DONOR_ROOT = f"{REMOTE_DATA_ROOT}/donors"
ANALYSIS_ROOT = f"{REMOTE_DATA_ROOT}/analysis"
SYNTHETIC_ROOT = f"{REMOTE_DATA_ROOT}/synthetic"
# Turn positions of the cached prompts and the weight files derived from them (data_curation/turn_positions.py).
TURN_POSITION_ROOT = f"{ANALYSIS_ROOT}/turn_positions"
CHECKPOINT_ROOT = "/checkpoints/agent-eff"
INITIAL_MEGATRON = f"{LOOPTOOL_CHECKPOINT_ROOT}/initial-megatron"
STUDENT_TOKENIZER_JSON = (
    f"{REMOTE_MODEL_ROOT}/models--{STUDENT_MODEL.replace('/', '--')}/snapshots/{STUDENT_REVISION}/tokenizer.json"
)
# Megatron names the checkpoint after the last cached batch: 50 replay rounds end at iter_0000049.
FINAL_ITERATION = RECIPE.replay_rounds - 1
PROVENANCE_FILE = "agent_eff_provenance.json"

# DeepSeek-R1-Distill tokenizers keep <think>, </think>, and <tool_call> (though untrained) but not
# Qwen's chat markers. Aliases must be injective, so only the end-of-message marker is mapped.
R1_ALIASES = {"<|im_end|>": "<｜end▁of▁sentence｜>"}

# Efficiency-term strengths as mean per-token KL(q_with ‖ q_without) on a prompt-stratified sample.
# The agent-accuracy target itself is ≈0.054 nats/token from behavior on this cache.
KL_BUDGETS = {"low": 0.005, "mid": 0.0135, "high": 0.027}
PRIMARY_BUDGET = "mid"


@dataclass(frozen=True)
class Model:
    repo: str
    revision: str

    @property
    def path(self) -> str:
        namespace, name = self.repo.split("/", 1)
        return f"{REMOTE_MODEL_ROOT}/models--{namespace}--{name}/snapshots/{self.revision}"


@dataclass(frozen=True)
class Pair:
    """An anchor pair; its shift is log p_post − log p_pre on the cached candidates."""

    name: str
    role: str
    pre: Model
    post: Model
    domain: str
    note: str
    aliases: dict[str, str] = field(default_factory=dict)
    core: bool = False

    @property
    def directory(self) -> str:
        return f"{DONOR_ROOT}/{self.name}"

    @property
    def scores(self) -> str:
        """The scored chain (the pre stage holds both anchors' columns)."""
        return f"{self.directory}/pre"


R1D_1P5B = Model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562")
R1D_7B = Model("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "916b56a44061fd5cd7d6a8fb632557ed4f724f60")
DEEPSCALER = Model("agentica-org/DeepScaleR-1.5B-Preview", "e3f524ce413a296b4d388e7560dd5c82c1c56725")
NEMOTRON_V1 = Model("nvidia/Nemotron-Research-Reasoning-Qwen-1.5B", "b89048893f95246c6b5749b287f0049e6df42ee9")
# Elastic Reasoning (arXiv 2505.05315) trains each E1 model from its domain's accuracy RL model, with a 1K-token
# thinking budget (</think> is forced when it runs out) and a correctness-only reward. Its paper names the starting
# models, and weight slices confirm them: E1-Code is 2x closer to DeepCoder-14B than to R1-Distill-14B, and E1-Math
# is closer to DeepScaleR than to R1-Distill-1.5B. All five R1-family tokenizers here share one tokenizer.json.
DEEPCODER_14B = Model("agentica-org/DeepCoder-14B-Preview", "cfa11a3b8e32163123776df78ab1e95adf0ca58d")
E1_CODE = Model("Salesforce/E1-Code-14B", "9eebf1fbd4df613ef31915344e5f5ffbe50d8d3b")
E1_MATH = Model("Salesforce/E1-Math-1.5B", "3fa7501acb8aec762347e70be9a04a6324c886fc")

AGENT_ACC = Pair(
    "agent_acc",
    "agent_accuracy",
    Model(PRE_TEACHER_MODEL, PRE_TEACHER_REVISION),
    Model(POST_TEACHER_MODEL, POST_TEACHER_REVISION),
    "agent",
    "the LoopTool anchor, already scored in the base cache; a broad multi-stage post-training difference",
    core=True,
)

DONORS = {
    pair.name: pair
    for pair in (
        Pair(
            "klear",
            "accuracy",
            Model("Qwen/Qwen3-8B-Base", "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"),
            Model("Kwai-Klear/Klear-Reasoner-8B", "ce412688b6181fc2fe2e51a3e86c60f08c08b7dc"),
            "math+code",
            "Lightning Weave's accuracy anchor (SFT then RL); a different family, scale, and recipe from DECS",
            core=True,
        ),
        Pair(
            "decs",
            "efficiency",
            R1D_1P5B,
            Model("pixas/DECS_1.5B", "3a4a92d47e3dda584e03e2d08f8164189af6bea6"),
            "math",
            "Lightning Weave's efficiency anchor (R1-Distill-1.5B + DECS RL on DeepScaleR data)",
            R1_ALIASES,
            core=True,
        ),
        Pair(
            "deepscaler",
            "accuracy",
            R1D_1P5B,
            DEEPSCALER,
            "math",
            "quasi-control for DECS: same base and related data, different authors, steps, and reward details",
            R1_ALIASES,
            core=True,
        ),
        Pair(
            "l1max",
            "efficiency",
            DEEPSCALER,
            Model("l3lab/L1-Qwen-1.5B-Max", "8d5eff2725e735114ec3949d1c5ae90d519fa4b6"),
            "math",
            "length-controlled RL from DeepScaleR; still a full policy-training difference",
            R1_ALIASES,
        ),
        Pair(
            "nemotron",
            "efficiency",
            NEMOTRON_V1,
            Model("nvidia/Nemotron-Research-Reasoning-Qwen-1.5B", "c62ac5e70bd578a9235aa9d8e11fff2f1f63d4a0"),
            "math+code+stem",
            "ProRL v1 -> v2: scheduled length penalty plus continued training and algorithm changes",
            R1_ALIASES,
        ),
        Pair(
            "decs7b",
            "efficiency",
            R1D_7B,
            Model("pixas/DECS_7B", "0e9d916a4b4063a2466fb9d2dcbe1a4f1e3a1f09"),
            "math",
            "DECS at 7B",
            R1_ALIASES,
        ),
        Pair(
            "e1code",
            "efficiency",
            DEEPCODER_14B,
            E1_CODE,
            "code",
            "Elastic Reasoning from the code accuracy model (30 GRPO steps): efficiency minus code accuracy",
            R1_ALIASES,
        ),
        Pair(
            "e1math",
            "efficiency",
            DEEPSCALER,
            E1_MATH,
            "math",
            "Elastic Reasoning from the math accuracy model (200 GRPO steps): efficiency minus math accuracy",
            R1_ALIASES,
        ),
    )
}


def term(sources: dict[str, float], *, transform: str = "evidence", coef: float = 1.0, **extra) -> dict:
    return {
        "transform": transform,
        "coef": coef,
        "direction": {"terms": [{"source": name, "coef": weight} for name, weight in sources.items()]},
        **extra,
    }


ACCURACY_TERMS = {
    # The exact target of the trained LoopTool student, untrained special-token rows included.
    "acc-legacy": term({"agent_acc": 1.0}, transform="raw"),
    # The same pair with Qwen3-4B-Base's untrained special tokens carrying no shift.
    "acc-clean": term({"agent_acc": 1.0}),
}
EFFICIENCY_DIRECTIONS = {
    "decs": {"decs": 1.0},  # naive cross-domain Weave
    "decs-deepscaler": {"decs": 1.0, "deepscaler": -1.0},  # quasi-control contrast (primary)
    "decs-klear": {"decs": 1.0, "klear": -1.0},  # the analogy as first stated (exploratory, confounded)
}

# The Fisher projection coefficient of the acc-legacy+decs target on the acc-legacy target over all
# cached states (review of the composed primary targets). A target that keeps only this share tests
# whether the efficiency term acts as a weaker accuracy shift.
SCALED_ACCURACY_COEF = 0.8647
# Hand-coded fork pushes (nats of δ) matched to the acc-legacy+decs target's change over acc-legacy
# on the fork probes: stop_minus_continue +1.03 and reflect_minus_conclude −7.60, split ± between
# the two candidate groups.
HEURISTIC_PUSHES = {"stop": 0.513, "reflect": 3.80}
# Efficiency donors with distinct pre-models and recipes, each used as its raw pre -> post shift (the
# form that transferred best in the primary matrix).
TRANSFER_DONORS = ("decs", "decs7b", "l1max", "nemotron")
# acc-legacy+decs's calibrated efficiency coefficient (from its sealed manifest). Gated DECS arms keep it
# fixed, so removing the push at the gated states is their only change.
DECS_MID_COEF = 3.1135845716525035


def heuristic_term(stop: float, reflect: float) -> dict:
    """Favor stopping over a new paragraph at stop forks and concluding over reflecting at reflection forks."""
    pushes = [
        ("think_stop_fork", "single_newline", stop),
        ("think_stop_fork", "paragraph_break", -stop),
        ("think_reflection_fork", "reflection", -reflect),
        ("think_reflection_fork", "conclusion", reflect),
    ]
    return {
        "transform": "heuristic",
        "coef": 1.0,
        "pushes": [{"state": state, "group": group, "shift": shift} for state, group, shift in pushes],
    }


def spec(*terms: dict) -> dict:
    return {"alpha": RECIPE.alpha, "terms": list(terms)}


def primary_specs() -> dict[str, dict]:
    """The pre-registered diagnostic matrix: one budget, every arm reported."""
    budget = KL_BUDGETS[PRIMARY_BUDGET]
    specs = {name: spec(accuracy) for name, accuracy in ACCURACY_TERMS.items()}
    for accuracy in ACCURACY_TERMS:
        for efficiency in ("decs", "decs-deepscaler"):
            specs[f"{accuracy}+{efficiency}"] = spec(
                ACCURACY_TERMS[accuracy], term(EFFICIENCY_DIRECTIONS[efficiency], kl_budget=budget)
            )
    # Matched-KL control: the same contrast with its sign flipped.
    specs["acc-clean+decs-deepscaler-flipped"] = spec(
        ACCURACY_TERMS["acc-clean"], term(EFFICIENCY_DIRECTIONS["decs-deepscaler"], coef=-1.0, kl_budget=budget)
    )
    return specs


def control_specs() -> dict[str, dict]:
    """Controls for the best primary arm, acc-legacy+decs: its accuracy share alone, and a fork rule."""
    return {
        "acc-legacy-scaled": spec(term({"agent_acc": 1.0}, transform="raw", coef=SCALED_ACCURACY_COEF)),
        "acc-legacy+heuristic": spec(ACCURACY_TERMS["acc-legacy"], heuristic_term(**HEURISTIC_PUSHES)),
    }


def donor_specs() -> dict[str, dict]:
    """Donor generality at the primary budget, and the equal-Fisher-weight sum of all transfer donors."""
    budget = KL_BUDGETS[PRIMARY_BUDGET]
    accuracy = ACCURACY_TERMS["acc-legacy"]
    specs = {
        f"acc-legacy+{name}": spec(accuracy, term({name: 1.0}, kl_budget=budget))
        for name in TRANSFER_DONORS
        if name != "decs"  # acc-legacy+decs is in the primary matrix
    }
    specs["acc-legacy+donor-mean"] = spec(
        accuracy, term({name: 1.0 for name in TRANSFER_DONORS}, kl_budget=budget, equalize=True)
    )
    return specs


def gated_specs() -> dict[str, dict]:
    """The primary DECS arm without its push at each response's first reflection fork (the first check after
    a new tool result or user message)."""
    return {
        "acc-legacy+decs-protect-first": spec(
            ACCURACY_TERMS["acc-legacy"],
            term({"decs": 1.0}, coef=DECS_MID_COEF, gate={"exclude_first_reflection": True}),
        )
    }


# --- The code-efficiency round (pre-registered 2026-09-30; docs/agent_efficiency_synthesis.md) -------------------
# Each arm keeps the agent-accuracy term A at weight 1 and adds one efficiency term, scaled to the primary KL budget:
#   C = log E1-Code − log DeepCoder-14B, M = log E1-Math − log DeepScaleR, D = log DECS − log R1-Distill-1.5B.
# Inside a sum, sources keep their raw (1:1) weights when their Fisher RMS sizes are within EQUALIZE_MAX_RATIO,
# and are equalized to unit RMS otherwise, so a composition is never just its larger part.
EQUALIZE_MAX_RATIO = 2.0
# E1-Code trained for 30 steps, so the budget may need a larger multiplier than DECS's 3.1 (recorded, not gated).
CODE_ROUND_MAX_MULTIPLIER = 50.0
CODE_ROUND_PAIRS = ("e1code", "e1math")
# The second training seed of every replicated arm (BFCL tags end in .s5678).
REPLICATE_SEED = 5678


def code_round_specs() -> dict[str, dict]:
    budget = KL_BUDGETS[PRIMARY_BUDGET]
    accuracy = ACCURACY_TERMS["acc-legacy"]
    single = {"kl_budget": budget, "max_multiplier": CODE_ROUND_MAX_MULTIPLIER}
    summed = single | {"equalize": "auto", "equalize_max_ratio": EQUALIZE_MAX_RATIO}
    return {
        "acc-legacy+e1code": spec(accuracy, term({"e1code": 1.0}, **single)),
        "acc-legacy+e1math": spec(accuracy, term({"e1math": 1.0}, **single)),
        "acc-legacy+e1math+e1code": spec(accuracy, term({"e1math": 1.0, "e1code": 1.0}, **summed)),
        "acc-legacy+decs+e1code": spec(accuracy, term({"decs": 1.0, "e1code": 1.0}, **summed)),
    }


# --- Follow-ups after the external review of the code round (2026-09-30) ---------------------------------------------
# Donor support on the student's candidates is ~0 at the call-or-reply decision and below 6% right after a call
# (median at call boundaries: DECS 0.95%, E1-Code 5.9%, E1-Math 0.47%), so a term there moves a protocol decision on
# unreliable scores; E1-Math pushes "keep going" there by about 10 nats. Masking the unscorable <tool_call> does not
# prevent this (competing candidates still move), so protection removes the efficiency term at those states entirely.
PROTOCOL_STATES = ("act_vs_talk", "call_boundary")
# acc-legacy+e1math's calibrated coefficient (sealed manifest). The protected arm keeps it fixed, so removing the push
# at the protocol states is its only change from acc-legacy+e1math (recalibrating would strengthen the rest).
E1MATH_MID_COEF = 7.9454246558728014


def followup_specs() -> dict[str, dict]:
    """The protocol-state ablation of the E1-Math arm, and E1-Math at the low budget (for matched savings)."""
    accuracy = ACCURACY_TERMS["acc-legacy"]
    return {
        "acc-legacy+e1math-protected": spec(
            accuracy,
            term({"e1math": 1.0}, coef=E1MATH_MID_COEF, gate={"exclude_state_types": list(PROTOCOL_STATES)}),
        ),
        "acc-legacy+e1math@low": spec(
            accuracy, term({"e1math": 1.0}, kl_budget=KL_BUDGETS["low"], max_multiplier=CODE_ROUND_MAX_MULTIPLIER)
        ),
    }


# --- Next: Lightning Weave's published recipe, and turn-start allocation (set up 2026-09-30) -------------------------
# The paper (arXiv 2609.14708, Sec. 4.1) trains every arm at alpha 2.0 with anchor weights normalized to sum to one:
# Klear-only is accuracy at 1.0, and the equal composition is 0.5 accuracy + 0.5 DECS. Our arms instead keep accuracy
# at 1.0 and add a KL-calibrated efficiency term (DECS at 3.1x the accuracy weight), over one pass (200 updates); the
# released code trains two passes (400 updates). The paper's composition also halves the accuracy weight, so a
# half-accuracy control separates DECS's contribution from that. The DECS term keeps the evidence transform used by
# every DECS arm here (a raw composition would intersect masks and delete the agent-accuracy signal at tool-call
# states). The paper's own number of updates is not stated; the released default (400) is used.
PAPER_WEIGHTS = {"accuracy": 0.5, "efficiency": 0.5}
TRAINING_PASSES = {"paper-acc-legacy": 2, "paper-acc-legacy+decs": 2, "paper-half-acc-legacy": 2}  # 1 otherwise
# Turn starts: cached messages that answer a new user message after earlier assistant turns (1,934 of 3,200 prompts,
# 83% of the multi-turn ones). The control removes DECS from the same number of multi-turn prompts chosen at random,
# so the two arms differ only in which multi-turn prompts keep DECS. Weight files: data_curation/turn_positions.py.
TURN_START_RULE = "protect_turn_starts"
RANDOM_MULTI_TURN_RULE = "random_multi_turn_control"


def paper_specs() -> dict[str, dict]:
    accuracy = ACCURACY_TERMS["acc-legacy"]
    half = term({"agent_acc": 1.0}, transform="raw", coef=PAPER_WEIGHTS["accuracy"])
    decs = term({"decs": 1.0}, coef=PAPER_WEIGHTS["efficiency"])
    return {
        "paper-acc-legacy": spec(accuracy),  # Klear-only's analog: accuracy at weight 1.0
        "paper-acc-legacy+decs": spec(half, decs),  # the equal two-anchor composition
        "paper-half-acc-legacy": spec(half),  # the composition without DECS
    }


def turn_start_specs() -> dict[str, dict]:
    """DECS at its mid coefficient (fixed, as in the other gated DECS arms), removed from whole cached responses on
    turn-start prompts, or on as many randomly chosen multi-turn prompts."""
    accuracy = ACCURACY_TERMS["acc-legacy"]
    return {
        f"acc-legacy+decs-{name}": spec(
            accuracy, term({"decs": 1.0}, coef=DECS_MID_COEF, gate={"prompt_weights": rule})
        )
        for name, rule in (
            ("protect-turn-starts", TURN_START_RULE),
            ("gate-random-multi-turn", RANDOM_MULTI_TURN_RULE),
        )
    }


def training_passes(variant: str) -> int:
    return TRAINING_PASSES.get(variant, 1)


def training_plan(variant: str) -> dict:
    """What ``train`` uses for a variant: the spec's alpha, and one or more passes over its sealed target.

    A multi-pass plan trains on a copy of the target repeated that many times (the offline data source refuses to read
    past the end of a sealed cache), for ``RECIPE.replay_rounds`` batches per pass.
    """
    passes = training_passes(variant)
    rounds = RECIPE.replay_rounds * passes
    target = f"{SYNTHETIC_ROOT}/{variant}"
    return {
        "alpha": variant_specs()[variant]["alpha"],
        "passes": passes,
        "num_rollout": rounds,
        "final_iteration": rounds - 1,
        "rows": RECIPE.consumed_trajectories * passes,
        "target": target,
        "data": target if passes == 1 else f"{target}-x{passes}",
    }


def variant_sources(variant_spec: dict) -> set[str]:
    return {
        source["source"]
        for item in variant_spec["terms"]
        for source in item.get("direction", {}).get("terms", [])
    }


def exploratory_specs() -> dict[str, dict]:
    """The two trained DECS budget-sweep arms."""
    return {
        f"acc-legacy+decs@{name}": spec(
            ACCURACY_TERMS["acc-legacy"], term({"decs": 1.0}, kl_budget=KL_BUDGETS[name])
        )
        for name in ("low", "high")
    }


def variant_specs() -> dict[str, dict]:
    return (
        primary_specs()
        | control_specs()
        | donor_specs()
        | gated_specs()
        | code_round_specs()
        | followup_specs()
        | paper_specs()
        | turn_start_specs()
        | exploratory_specs()
    )


def resolved() -> dict:
    return {
        "app": MODAL_APP_NAME,
        "volumes": {"data": MODAL_DATA_VOLUME, "models": MODAL_MODEL_VOLUME, "checkpoints": MODAL_CHECKPOINT_VOLUME},
        "student": {"repo": STUDENT_MODEL, "revision": STUDENT_REVISION},
        "base_cache": BASE_CACHE,
        "agent_acc": asdict(AGENT_ACC),
        "donors": {name: asdict(pair) | {"directory": pair.directory} for name, pair in DONORS.items()},
        "kl_budgets": KL_BUDGETS,
        "primary_variants": sorted(primary_specs()),
        "control_variants": sorted(control_specs()),
        "donor_variants": sorted(donor_specs()),
        "gated_variants": sorted(gated_specs()),
        "code_round": {
            "pairs": list(CODE_ROUND_PAIRS),
            "specs": code_round_specs(),
            "equalize_max_ratio": EQUALIZE_MAX_RATIO,
        },
        "next": {
            "training_passes": TRAINING_PASSES,
            "specs": paper_specs() | turn_start_specs(),
        },
        "followups": {
            "protocol_states": list(PROTOCOL_STATES),
            "specs": followup_specs(),
        },
        "exploratory_variants": sorted(exploratory_specs()),
        "training": {
            "recipe": "LoopTool Offline Direct-OPD; only the target and the checkpoint cadence differ",
            "alpha": RECIPE.alpha,
            "replay_rounds": RECIPE.replay_rounds,
            "final_iteration": FINAL_ITERATION,
            "save": "final checkpoint only",
            "default_seed": RECIPE.training_seed,
            "data_order": "stored order at the default seed; any other seed also permutes the rows with it",
            "initial_checkpoint": INITIAL_MEGATRON,
            "checkpoint_root": CHECKPOINT_ROOT,
        },
        "images": {"runtime": RUNTIME_IMAGE},
        "repo": REMOTE_REPO,
    }


def main() -> None:
    print(json.dumps(resolved(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
