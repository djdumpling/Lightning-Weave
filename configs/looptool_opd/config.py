"""Immutable recipe constants and validation for LoopTool Offline Direct-OPD."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json


DATASET_REPO = "zhangkangning/LoopTool-23k"
DATASET_REVISION = "b6c572d442ed4f2177f23645d8e9a77522e712c3"
EXPECTED_CANONICAL_ROWS = 23_000

# The behavior policy and trainable student are the released post-trained
# Qwen3-4B. The anchor is the change from the base checkpoint to Thinking-2507.
STUDENT_MODEL = "Qwen/Qwen3-4B"
STUDENT_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
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
REMOTE_DATA_ROOT = "/opd/looptool-qwen3-4b-v1"
REMOTE_MODEL_ROOT = "/models"
REMOTE_CHECKPOINT_ROOT = "/checkpoints/looptool-offline-dopd-qwen3-4b-v1"


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
        return {
            "algorithm": "single-anchor Offline Direct-OPD (tilted target)",
            "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION},
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


RECIPE = Recipe()
RECIPE.validate()


def main() -> None:
    print(json.dumps(RECIPE.resolved(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
