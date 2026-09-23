# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "modal-training-gym @ git+https://github.com/modal-projects/training-gym.git@8899342e709189e2a09a6f9bdadc1e36a28dae79",
# ]
# ///
"""Launch Qwen3-4B GRPO on Lightning Weave's math data with Training Gym.

This is ordinary online GRPO on the Skywork math prompts used by Lightning
Weave. It does not use anchors, cached targets, or Offline Direct-OPD.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

MODEL_REPO = "Qwen/Qwen3-4B"
DATASET_REPO = "Skywork/Skywork-OR1-RL-Data"
DATASET_FILE = "data/math-00000-of-00001.parquet"
DATASET_URI = f"hf://datasets/{DATASET_REPO}/{DATASET_FILE}"
DATASET_ADAPTER_VERSION = "lightning-weave-math-v2"
TRAINING_GYM_COMMIT = "8899342e709189e2a09a6f9bdadc1e36a28dae79"
WANDB_SECRET = "wandb-secret"
PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)
PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def adapt_skywork_math_row(row: dict[str, Any]) -> dict[str, Any]:
    """Match the Skywork conversion used by Lightning Weave's data pipeline."""

    question = row["prompt"][0]["content"].strip()
    ground_truth = json.loads(row["reward_model"]["ground_truth"])[0]
    return {
        "prompt": [
            {
                "role": "user",
                "content": f"{PROMPT_PREFIX}{question}{PROMPT_SUFFIX}",
            }
        ],
        "label": str(ground_truth),
    }


async def skywork_math_reward(_args: Any, sample_or_samples: Any, **_kwargs: Any) -> float | list[float]:
    """Grade Skywork's ``Answer:`` format with the repository's math verifier."""

    from math_verify import parse, verify

    def grade(sample: Any) -> float:
        response = str(sample.response)
        matches = re.findall(
            r"(?im)^\s*\*{0,2}Answer\s*:\s*(.+?)\s*\*{0,2}\s*$",
            response,
        )
        if matches:
            prediction_text = matches[-1].strip()
        elif "\\boxed" in response:
            prediction_text = response
        else:
            return 0.0

        try:
            # Slime evaluates rewards on a background event-loop thread.
            # math-verify's default POSIX timeout uses SIGALRM, which is only
            # legal on the main thread and otherwise makes every reward zero.
            target = parse(f"${sample.label}$", parsing_timeout=None)
            prediction = parse(prediction_text, parsing_timeout=None)
            return float(bool(target and prediction and verify(target, prediction, timeout_seconds=None)))
        except Exception:
            return 0.0

    if isinstance(sample_or_samples, list):
        return [grade(sample) for sample in sample_or_samples]
    return grade(sample_or_samples)


def require_python_312() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "Modal Training Gym serializes local functions into a Python 3.12 "
            "image, so the launcher must also use Python 3.12. Run it through "
            "`bash scripts/train_math_grpo.sh`."
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--wandb-project", default="qwen3-4b-math-grpo")
    result.add_argument("--wandb-group", default="qwen3-4b-math-grpo")
    result.add_argument("--wandb-entity", default="")
    result.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use one short rollout before spending on the full run.",
    )
    result.add_argument(
        "--wait",
        action="store_true",
        help="Stream status until completion; the run is detached by default.",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved configuration without using Modal.",
    )
    return result


def build_config(args: argparse.Namespace) -> Any:
    """Build the pinned Training Gym experiment without making remote calls."""

    from modal_training_gym import (
        HuggingFaceDataset,
        Qwen3_4B,
        Qwen3_4B_Recipe,
        TrainConfig,
        WandbConfig,
    )

    class SkyworkMathDataset(HuggingFaceDataset):
        """Select and normalize the exact Skywork math shard used by this repo."""

        data_file = DATASET_FILE

        def cache_key(self) -> str | None:
            base_key = super().cache_key()
            if base_key is None:
                return None
            return f"{base_key}:{self.data_file}:{DATASET_ADAPTER_VERSION}"

        def _load_hf_dataset(self):
            from datasets import load_dataset

            dataset = load_dataset(
                "parquet",
                data_files={"train": DATASET_URI},
                split="train",
            )
            return dataset.map(
                adapt_skywork_math_row,
                remove_columns=dataset.column_names,
                desc="Formatting Lightning Weave Skywork math prompts",
            )

    # Keep the smoke-test delta explicit. The full run matches the repository's
    # math trajectory/update budget where that budget has an online-RL analogue.
    if args.smoke_test:
        num_rollout = save_interval = 1
        rollout_batch_size = samples_per_prompt = 4
        global_batch_size = 16
        max_response_length = 4096
        max_context_length = max_tokens_per_gpu = 5120
    else:
        num_rollout = 100
        save_interval = 20
        rollout_batch_size = 64
        samples_per_prompt = 4
        global_batch_size = 64
        max_response_length = 4096
        max_context_length = max_tokens_per_gpu = 5120

    dataset = SkyworkMathDataset(
        DATASET_REPO,
        hf_split="train",
        input_column="prompt",
        output_column="label",
        input_format="messages",
    )
    recipe = Qwen3_4B_Recipe(
        # One Modal node with exactly eight H100s. Qwen3-4B fits on one H100,
        # so use TP=1 and spend all eight GPUs on data parallelism and eight
        # independent rollout engines. Actor and rollout share the node.
        gpu_type="H100",
        colocate=True,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        # Rollout budget and sampling.
        num_rollout=num_rollout,
        rollout_batch_size=rollout_batch_size,
        n_samples_per_prompt=samples_per_prompt,
        rollout_max_response_len=max_response_length,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_shuffle=True,
        sglang_mem_fraction_static=0.7,
        # Ordinary GRPO with the original asymmetric clip range. The source
        # recipe enabled KL loss with a zero coefficient, which is preserved.
        advantage_estimator="grpo",
        use_kl_loss=True,
        kl_loss_type="low_var_kl",
        kl_loss_coef=0.0,
        kl_coef=0.0,
        entropy_coef=0.0,
        eps_clip=0.2,
        eps_clip_high=0.28,
        calculate_per_token_loss=False,
        balance_data=True,
        # Optimizer and memory settings.
        global_batch_size=global_batch_size,
        lr=1e-6,
        lr_decay_style="constant",
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        optimizer="adam",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_softmax_in_fp32=True,
        accumulate_allreduce_grads_in_fp32=True,
        attention_backend="flash",
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=max_tokens_per_gpu,
        rm_type=None,
        custom_rm_function=skywork_math_reward,
        save_interval=save_interval,
        # The pinned Slime image has protobuf 5.29.6, while its TensorBoard
        # protos were generated by 6.31.1. Megatron imports TensorBoard during
        # HF checkpoint conversion, so upgrade the runtime and fail the image
        # build immediately if that import ever becomes incompatible again.
        image_run_commands=[
            ("python3 -m pip install --no-cache-dir protobuf==7.35.1 math-verify==0.9.0"),
            (
                'python3 -c "from tensorboard.compat.proto import event_pb2; '
                "from math_verify import parse, verify; "
                'import google.protobuf; print(google.protobuf.__version__)"'
            ),
        ],
        metrics=WandbConfig(
            project=args.wandb_project,
            group=args.wandb_group,
            entity=args.wandb_entity,
            exp_name=args.wandb_group,
            modal_wandb_secret_name=WANDB_SECRET,
        ),
        # These are Slime arguments not yet promoted to typed Training Gym
        # fields. Training Gym writes them to its custom YAML config.
        extra_config={
            "rollout_max_prompt_len": 1024,
            "rollout_max_context_len": max_context_length,
            "rollout_top_k": 16,
            "seq_length": max_context_length,
            "max_position_embeddings": max_context_length,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
            "clip_grad": 1.0,
            "seed": 1234,
            "rollout_seed": 42,
            "wandb_always_use_train_step": True,
            "log_passrate": True,
        },
    )
    return TrainConfig(model=Qwen3_4B(), dataset=dataset, recipe=recipe)


def summary(config: Any, smoke_test: bool) -> dict[str, Any]:
    recipe = config.recipe
    extra = recipe.extra_config
    responses_per_rollout = recipe.rollout_batch_size * recipe.n_samples_per_prompt
    return {
        "training_gym_commit": TRAINING_GYM_COMMIT,
        "model": MODEL_REPO,
        "dataset": DATASET_REPO,
        "dataset_file": DATASET_FILE,
        "algorithm": recipe.advantage_estimator,
        "reward": recipe.rm_type or skywork_math_reward.__name__,
        "gpu_request": f"{recipe.gpu_type}:{recipe.actor_num_gpus_per_node}",
        "rollout_steps": recipe.num_rollout,
        "prompts_per_rollout": recipe.rollout_batch_size,
        "responses_per_prompt": recipe.n_samples_per_prompt,
        "responses_per_rollout": responses_per_rollout,
        "total_sampled_responses": responses_per_rollout * recipe.num_rollout,
        "global_batch_size": recipe.global_batch_size,
        "optimizer_updates_per_rollout": responses_per_rollout // recipe.global_batch_size,
        "max_prompt_tokens": extra["rollout_max_prompt_len"],
        "max_response_tokens": recipe.rollout_max_response_len,
        "max_context_tokens": extra["rollout_max_context_len"],
        "megatron_tensor_parallel": recipe.tensor_model_parallel_size,
        "sglang_tensor_parallel": recipe.rollout_num_gpus_per_engine,
        "wandb_project": recipe.metrics.project,
        "wandb_group": recipe.metrics.group,
        "wandb_secret": recipe.metrics.modal_wandb_secret_name,
        "smoke_test": smoke_test,
    }


def main() -> None:
    require_python_312()
    args = parser().parse_args()
    config = build_config(args)
    print(json.dumps(summary(config, args.smoke_test), indent=2), flush=True)
    if args.dry_run:
        return

    run = config.launch(show_output=True)
    print(f"Training Gym run: {run.training_run_id}", flush=True)
    print(f"Modal app: {run.modal_app_url}", flush=True)
    if args.wait:
        run.result()
    else:
        print(
            f"Training is detached. Use `training-gym run logs {run.training_run_id} --follow` to stream progress.",
            flush=True,
        )


if __name__ == "__main__":
    main()
