"""Immutable recipe constants and validation for LoopTool Offline Direct-OPD."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.tau_bench_eval.config import AREAL_REPO, AREAL_REVISION  # noqa: E402


DATASET_REPO = "zhangkangning/LoopTool-23k"
DATASET_REVISION = "b6c572d442ed4f2177f23645d8e9a77522e712c3"
EXPECTED_CANONICAL_ROWS = 23_000

# The behavior policy and trainable student are the released post-trained
# Qwen3-4B. The anchor is the change from the base checkpoint to Thinking-2507.
QWEN3_4B = ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c")
PRE_TEACHER_MODEL = "Qwen/Qwen3-4B-Base"
PRE_TEACHER_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
POST_TEACHER_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
POST_TEACHER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"

# This is the environment published by the upstream Lightning-OPD authors.
# Pinning the registry digest prevents the mutable v0.2 tag from changing a run.
RUNTIME_IMAGE = "tonyhao96/jetmoe@sha256:d79db3699f58172088c8897efd4e21aa7960cbd5530c81b02a6bf07132695313"
ROLLOUT_IMAGE = "vllm/vllm-openai@sha256:d8d39b59e909d2378ac4feeb191f7e7b6f1342477dc66b7c47cec89e9985ad8a"

MODAL_APP_NAME = "lightning-weave-looptool-offline-dopd"
MODAL_DATA_VOLUME = "lightning-weave-looptool-opd-data"
MODAL_MODEL_VOLUME = "lightning-weave-hf-models"
MODAL_CHECKPOINT_VOLUME = "lightning-weave-checkpoints"
REMOTE_REPO = "/workspace/Lightning-Weave"


@dataclass(frozen=True)
class StatePool:
    """A cached state pool: where it lives, its prompt cap, and (if its prompts come from collected tau2
    episodes rather than LoopTool) that collection. The frozen student generates, anchors and is trained on
    every pool: Qwen3-4B unless the pool names another (with its Megatron model type, its chat template's
    generation prompt, a response cap that fits its turns, and its own converted-checkpoint root)."""

    root: str
    max_prompt_tokens: int
    collection: dict | None = None
    student: tuple[str, str] = QWEN3_4B
    model_type: str = "qwen3-4B"
    generation_prompt: str = "<|im_start|>assistant\n"
    max_response_tokens: int = 2_048
    checkpoint_root: str = "/checkpoints/looptool-offline-dopd-qwen3-4b-v1"


STATE_POOLS = {
    # The LoopTool prompts: V0 and every agent-efficiency arm.
    "looptool": StatePool("/opd/looptool-qwen3-4b-v1", 8_192),
    # Decision states trained students visited on AReaL tau2 training tasks (configs/tau_bench_eval collection
    # mode, profile user235b-4b), extracted by data_curation/fresh_states.py.
    "tau2-fresh": StatePool(
        "/opd/tau2-fresh-qwen3-4b-v1",
        16_384,
        {
            "results_volume": "lightning-weave-tau-eval",
            "run_id": "collect-98179d00b25d-v1",
            "collectors": ["ae.joint.acc-legacy", "ae.joint.acc-legacy+decs"],
            "domains": ["tau2_airline", "tau2_retail"],
            "tasks": {"repo": AREAL_REPO, "revision": AREAL_REVISION},
            "states_per_episode": 4,
        },
    ),
    # The same collection and recipe with Qwen3-4B-Thinking-2507 as the collector, sampler and student
    # (profile user235b: its 65,536-token window and deepseek_r1 parser), on the pilot's 1,163 AReaL tasks.
    # Episodes are uncapped and every step is eligible; each episode's write, transfer and first-error steps
    # are drawn first (the mentor's focused turns), then the rest. Responses get 8,192 tokens: 2507's turns
    # reach p99.9 = 5,961 tokens on tau2 at evaluation decoding, so 2,048 would cut its reasoning short.
    "tau2-fresh-2507": StatePool(
        "/opd/tau2-fresh-qwen3-4b-thinking-2507-v1",
        16_384,
        {
            "results_volume": "lightning-weave-tau-eval",
            "run_id": "collect-0870ad4ec984-v1",
            "collectors": ["thinking2507"],
            "domains": ["tau2_airline", "tau2_retail"],
            "tasks": {"repo": AREAL_REPO, "revision": AREAL_REVISION},
            "states_per_episode": 4,
            "prioritize_decisions": True,
        },
        student=(POST_TEACHER_MODEL, POST_TEACHER_REVISION),
        model_type="qwen3-4B-Thinking-2507",
        generation_prompt="<|im_start|>assistant\n<think>\n",
        max_response_tokens=8_192,
        checkpoint_root="/checkpoints/opd-qwen3-4b-thinking-2507-v1",
    ),
    # Retail is the main setting (mentor, 2026-10-09). The same 2507 collection's 563 retail episodes, retail states
    # only: up to 8 per episode (decision states first), 3,200 in one cell. Same student, recipe and initial checkpoint.
    "tau2-retail-2507": StatePool(
        "/opd/tau2-retail-qwen3-4b-thinking-2507-v1",
        16_384,
        {
            "results_volume": "lightning-weave-tau-eval",
            "run_id": "collect-0870ad4ec984-v1",
            "collectors": ["thinking2507"],
            "domains": ["tau2_retail"],
            "tasks": {"repo": AREAL_REPO, "revision": AREAL_REVISION},
            "states_per_episode": 8,
            "prioritize_decisions": True,
        },
        student=(POST_TEACHER_MODEL, POST_TEACHER_REVISION),
        model_type="qwen3-4B-Thinking-2507",
        generation_prompt="<|im_start|>assistant\n<think>\n",
        max_response_tokens=8_192,
        checkpoint_root="/checkpoints/opd-qwen3-4b-thinking-2507-v1",
    ),
}
# Why the mentor's 2507 runs did not transfer (results_7.md): the same 563 retail episodes, varying only which states
# are trained. "cap2048" applies his collection's cumulative 2,048-generated-token episode budget: states reached with
# budget left (about 5 per episode, almost never a write), each response capped at the remainder, the quota filled
# with further independently sampled copies. "uniform" is his pipeline without the cap: every turn eligible, up to 8
# per episode by hash, no decision priority.
STATE_POOLS["tau2-retail-2507-cap2048"] = replace(
    STATE_POOLS["tau2-retail-2507"],
    root="/opd/tau2-retail-cap2048-qwen3-4b-thinking-2507-v1",
    collection={
        **STATE_POOLS["tau2-retail-2507"].collection,
        "states_per_episode": 64,
        "prioritize_decisions": False,
        "episode_token_budget": 2_048,
        "repeat_to_quota": True,
    },
)
STATE_POOLS["tau2-retail-2507-uniform"] = replace(
    STATE_POOLS["tau2-retail-2507"],
    root="/opd/tau2-retail-uniform-qwen3-4b-thinking-2507-v1",
    collection={**STATE_POOLS["tau2-retail-2507"].collection, "prioritize_decisions": False},
)
# The launcher's OPD_DATA selects the pool. A non-default pool runs as its own Modal app with OPD_DATA baked into
# its images (PROFILE_ENV), so every remote import resolves the same pool.
DATA_PROFILE = os.environ.get("OPD_DATA", "looptool")
if DATA_PROFILE not in STATE_POOLS:
    raise ValueError(f"OPD_DATA={DATA_PROFILE!r}; choose from {sorted(STATE_POOLS)}")
POOL = STATE_POOLS[DATA_PROFILE]
PROFILE_ENV = {} if DATA_PROFILE == "looptool" else {"OPD_DATA": DATA_PROFILE}
REMOTE_DATA_ROOT = POOL.root
STUDENT_MODEL, STUDENT_REVISION = POOL.student
MODEL_TYPE = POOL.model_type


def app_name(base: str) -> str:
    return f"{base}-{DATA_PROFILE}" if PROFILE_ENV else base


REMOTE_MODEL_ROOT = "/models"
REMOTE_CHECKPOINT_ROOT = POOL.checkpoint_root


@dataclass(frozen=True)
class Recipe:
    """The mentor-provided recipe plus cache-shape choices from Direct-OPD."""

    learning_rate: float = 1e-6
    alpha: float = 2.0
    global_batch_size: int = 64
    rollout_batch_size: int = 256
    replay_rounds: int = 50
    max_prompt_tokens: int = 8_192
    max_response_tokens: int = 2_048
    responses_per_prompt: int = 4
    selected_prompts: int = 3_200
    top_k: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    data_seed: int = 42
    training_seed: int = 1_234
    training_gpus: int = 8
    cache_workers: int = 8

    @property
    def sequence_tokens(self) -> int:
        return self.max_prompt_tokens + self.max_response_tokens

    @property
    def cached_trajectories(self) -> int:
        return self.selected_prompts * self.responses_per_prompt

    @property
    def consumed_trajectories(self) -> int:
        return self.replay_rounds * self.rollout_batch_size

    @property
    def optimizer_updates_per_round(self) -> int:
        return self.rollout_batch_size // self.global_batch_size

    @property
    def optimizer_updates(self) -> int:
        return self.replay_rounds * self.optimizer_updates_per_round

    def validate(self) -> None:
        positive = {
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "global_batch_size": self.global_batch_size,
            "rollout_batch_size": self.rollout_batch_size,
            "replay_rounds": self.replay_rounds,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_response_tokens": self.max_response_tokens,
            "responses_per_prompt": self.responses_per_prompt,
            "selected_prompts": self.selected_prompts,
            "top_k": self.top_k,
            "training_gpus": self.training_gpus,
            "cache_workers": self.cache_workers,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rollout_batch_size % self.global_batch_size:
            raise ValueError("rollout_batch_size must be divisible by global_batch_size")
        if self.global_batch_size % self.training_gpus:
            raise ValueError("global_batch_size must be divisible by training_gpus")
        if self.cached_trajectories != self.consumed_trajectories:
            raise ValueError(
                "the sealed cache must contain exactly one training pass: "
                f"{self.cached_trajectories=} != {self.consumed_trajectories=}"
            )

    def resolved(self) -> dict[str, object]:
        self.validate()
        dataset = {"repo": DATASET_REPO, "revision": DATASET_REVISION}
        if POOL.collection is not None:
            dataset = {"profile": DATA_PROFILE, "collection": POOL.collection}
        # Pools of the default student keep the exact lock their caches were sealed with.
        student = {} if MODEL_TYPE == "qwen3-4B" else {"student_model_type": MODEL_TYPE}
        return student | {
            "algorithm": "single-anchor Offline Direct-OPD (tilted target)",
            "dataset": dataset,
            "student_behavior_and_trainable_policy": {
                "repo": STUDENT_MODEL,
                "revision": STUDENT_REVISION,
            },
            "anchor_pair": {
                "pre": {"repo": PRE_TEACHER_MODEL, "revision": PRE_TEACHER_REVISION},
                "post": {"repo": POST_TEACHER_MODEL, "revision": POST_TEACHER_REVISION},
            },
            "recipe": asdict(self),
            "derived": {
                "sequence_tokens": self.sequence_tokens,
                "cached_trajectories": self.cached_trajectories,
                "consumed_trajectories": self.consumed_trajectories,
                "optimizer_updates_per_replay_round": self.optimizer_updates_per_round,
                "optimizer_updates": self.optimizer_updates,
                "independent_prompt_groups_per_rollout_batch": (
                    self.rollout_batch_size // self.responses_per_prompt
                ),
            },
            "fixed_native_training": {
                "optimizer": "Adam",
                "adam_betas": [0.9, 0.999],
                "weight_decay": 0.01,
                "gradient_clip": 1.0,
                "lr_schedule": "constant",
                "precision": "BF16 parameters with FP32 reductions/softmax",
                "tensor_parallel_size": 1,
                "context_parallel_size": 1,
                "data_parallel_size": self.training_gpus,
                "dropout": 0.0,
                "dynamic_batching": True,
                "max_tokens_per_gpu": self.sequence_tokens,
                "activation_recomputation": "full, uniform, one layer per block",
                "live_rollout_or_teacher_gpus_during_training": 0,
                "wandb": False,
                "save_every_replay_rounds": 5,
            },
            "runtime_image": RUNTIME_IMAGE,
            "rollout_image": ROLLOUT_IMAGE,
            "important_semantics": {
                "alpha": "q(a|s) is proportional to p_student(a|s) * exp((log p_post - log p_pre) / alpha)",
                "fifty_steps": "50 cached replay rounds; at batch 256/global batch 64 this is 200 optimizer updates",
                "source_pool": "all canonical LoopTool prompts are rendered and length-audited before seeded selection",
            },
        }


RECIPE = Recipe(max_prompt_tokens=POOL.max_prompt_tokens, max_response_tokens=POOL.max_response_tokens)
RECIPE.validate()
# Megatron's full-vocabulary logits outgrow an 80 GB H100 above the pilot's 18,432-token sequences (2507's 24,576-token
# pool ran out of memory there on 2026-10-09); longer pools train on H200s with the same computation.
TRAINING_GPU = "H100:8" if RECIPE.sequence_tokens <= 18_432 else "H200:8"


def main() -> None:
    print(json.dumps(RECIPE.resolved(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
