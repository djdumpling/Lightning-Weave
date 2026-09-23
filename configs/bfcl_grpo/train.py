# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "modal-training-gym @ git+https://github.com/modal-projects/training-gym.git@8899342e709189e2a09a6f9bdadc1e36a28dae79",
#   "bfcl-eval==2026.3.23",
#   "jsonschema==4.25.1",
# ]
# ///
"""Train Qwen3-4B-Thinking-2507 with GRPO in BFCL's executable environment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from bfcl_adapter import (
    BFCL_CATEGORY,
    BFCL_PACKAGE_VERSION,
    MAX_ASSISTANT_STEPS,
    MAX_MODEL_TOKENS,
    MAX_PROMPT_TOKENS,
    MAX_STEP_RESPONSE_TOKENS,
    bfcl_turn_rollout,
    make_dataset_class,
)

MODEL_REPO = "Qwen/Qwen3-4B-Thinking-2507"
TRAINING_GYM_COMMIT = "8899342e709189e2a09a6f9bdadc1e36a28dae79"
WANDB_SECRET = "wandb-secret"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--wandb-project", default="qwen3-4b-thinking-2507-bfcl-grpo")
    result.add_argument("--wandb-group", default="bfcl-v4-multi-turn-base")
    result.add_argument("--wandb-entity", default="")
    result.add_argument("--max-model-tokens", type=int, default=MAX_MODEL_TOKENS)
    result.add_argument("--smoke-test", action="store_true")
    result.add_argument("--wait", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def require_python_312() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use `bash scripts/train_bfcl_grpo.sh`; Training Gym requires Python 3.12.")


def bfcl_image(image):
    return image.pip_install(
        f"bfcl-eval=={BFCL_PACKAGE_VERSION}",
        "jsonschema==4.25.1",
        "protobuf==7.35.1",
    )


def build_config(args: argparse.Namespace):
    from modal_training_gym import (
        DatasetConfig,
        Qwen3_4B,
        Qwen3_4B_Recipe,
        TrainConfig,
        WandbConfig,
    )

    if args.max_model_tokens < MAX_PROMPT_TOKENS + MAX_STEP_RESPONSE_TOKENS:
        raise ValueError("--max-model-tokens must fit the prompt and one assistant response")

    class Qwen3_4B_Thinking_2507(Qwen3_4B):
        model_name = MODEL_REPO
        architecture = replace(Qwen3_4B.architecture, rotary_base=5_000_000)

    dataset_class = make_dataset_class(DatasetConfig)
    dataset = dataset_class(
        split="train",
        eval_tasks=30,
        task_limit=4 if args.smoke_test else None,
    )
    if args.smoke_test:
        num_rollout, rollout_batch, samples, global_batch, save_interval = 1, 4, 2, 8, 1
    else:
        num_rollout, rollout_batch, samples, global_batch, save_interval = 100, 16, 4, 64, 5

    recipe = Qwen3_4B_Recipe(
        gpu_type="H100",
        colocate=True,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        num_rollout=num_rollout,
        rollout_batch_size=rollout_batch,
        n_samples_per_prompt=samples,
        global_batch_size=global_batch,
        rollout_max_response_len=MAX_STEP_RESPONSE_TOKENS,
        rollout_temperature=0.6,
        rollout_top_p=0.95,
        rollout_shuffle=True,
        sglang_mem_fraction_static=0.65,
        advantage_estimator="grpo",
        eps_clip=0.2,
        eps_clip_high=0.28,
        lr=5e-7,
        lr_decay_style="constant",
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_softmax_in_fp32=True,
        accumulate_allreduce_grads_in_fp32=True,
        attention_backend="flash",
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=args.max_model_tokens,
        custom_generate_function=bfcl_turn_rollout,
        save_interval=save_interval,
        image_overlay=bfcl_image,
        metrics=WandbConfig(
            project=args.wandb_project,
            group=args.wandb_group,
            entity=args.wandb_entity,
            exp_name=args.wandb_group,
            modal_wandb_secret_name=WANDB_SECRET,
        ),
        extra_config={
            "rollout_max_prompt_len": MAX_PROMPT_TOKENS,
            "rollout_max_context_len": args.max_model_tokens,
            "rollout_top_k": 20,
            "seq_length": args.max_model_tokens,
            "max_position_embeddings": args.max_model_tokens,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
            "bfcl_max_prompt_tokens": MAX_PROMPT_TOKENS,
            "bfcl_step_response_tokens": MAX_STEP_RESPONSE_TOKENS,
            "bfcl_max_model_tokens": args.max_model_tokens,
            "bfcl_max_assistant_steps": MAX_ASSISTANT_STEPS,
            "seed": 1234,
            "rollout_seed": 42,
            "log_passrate": True,
            "log_multi_turn": True,
            "wandb_always_use_train_step": True,
        },
    )
    return TrainConfig(model=Qwen3_4B_Thinking_2507(), dataset=dataset, recipe=recipe)


def resolved(config, smoke_test: bool) -> dict[str, object]:
    recipe = config.recipe
    return {
        "training_gym_commit": TRAINING_GYM_COMMIT,
        "bfcl_eval": BFCL_PACKAGE_VERSION,
        "model": MODEL_REPO,
        "dataset": f"BFCL v4/{BFCL_CATEGORY}",
        "unit": "one executable BFCL user turn",
        "algorithm": "GRPO",
        "reward": "Training Gym BFCL shaped reward + official terminal checker",
        "gpu_request": f"{recipe.gpu_type}:{recipe.actor_num_gpus_per_node}",
        "rollouts": recipe.num_rollout,
        "prompts_per_rollout": recipe.rollout_batch_size,
        "samples_per_prompt": recipe.n_samples_per_prompt,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_response_tokens_per_step": MAX_STEP_RESPONSE_TOKENS,
        "max_model_tokens": recipe.extra_config["bfcl_max_model_tokens"],
        "max_assistant_steps": MAX_ASSISTANT_STEPS,
        "rope_theta": config.model.architecture.rotary_base,
        "rope_scaling": None,
        "overlong": "reward retained; full trajectory loss-masked",
        "wandb_secret": recipe.metrics.modal_wandb_secret_name,
        "smoke_test": smoke_test,
    }


def main() -> None:
    require_python_312()
    args = parser().parse_args()
    config = build_config(args)
    print(json.dumps(resolved(config, args.smoke_test), indent=2), flush=True)
    if args.dry_run:
        return
    run = config.launch(show_output=True)
    print(f"Training Gym run: {run.training_run_id}", flush=True)
    print(f"Modal app: {run.modal_app_url}", flush=True)
    if args.wait:
        run.result()
    else:
        print("Training is detached; use `training-gym run logs RUN_ID --follow`.", flush=True)


if __name__ == "__main__":
    main()
