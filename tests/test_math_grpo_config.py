import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "configs/math_grpo/train.py"
SPEC = importlib.util.spec_from_file_location("math_grpo_train", TRAIN_SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeConfig:
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def fake_training_gym(monkeypatch):
    module = types.ModuleType("modal_training_gym")
    for name in (
        "HuggingFaceDataset",
        "Qwen3_4B",
        "Qwen3_4B_Recipe",
        "TrainConfig",
        "WandbConfig",
    ):
        setattr(module, name, type(name, (FakeConfig,), {}))
    monkeypatch.setitem(sys.modules, "modal_training_gym", module)
    return module


def test_training_gym_config_preserves_math_recipe(fake_training_gym):
    args = MODULE.parser().parse_args([])
    config = MODULE.build_config(args)
    recipe = config.recipe

    assert config.model.__class__.__name__ == "Qwen3_4B"
    assert config.dataset.args == ("Skywork/Skywork-OR1-RL-Data",)
    assert config.dataset.data_file == "data/math-00000-of-00001.parquet"
    assert config.dataset.input_column == "prompt"
    assert config.dataset.output_column == "label"
    assert config.dataset.input_format == "messages"

    assert recipe.gpu_type == "H100"
    assert recipe.actor_num_gpus_per_node == 8
    assert recipe.colocate is True
    assert recipe.tensor_model_parallel_size == 1
    assert recipe.sequence_parallel is False
    assert recipe.rollout_num_gpus_per_engine == 1
    assert recipe.advantage_estimator == "grpo"
    assert recipe.rm_type is None
    assert recipe.custom_rm_function is MODULE.skywork_math_reward
    assert recipe.num_rollout == 100
    assert recipe.rollout_batch_size == 64
    assert recipe.n_samples_per_prompt == 4
    assert recipe.global_batch_size == 64
    assert recipe.rollout_max_response_len == 4096
    assert recipe.lr == 1e-6
    assert recipe.weight_decay == 0.01
    assert recipe.adam_beta2 == 0.999
    assert recipe.max_tokens_per_gpu == 5120
    assert recipe.eps_clip == 0.2
    assert recipe.eps_clip_high == 0.28
    assert recipe.use_kl_loss is True
    assert recipe.kl_loss_coef == 0.0
    assert recipe.save_interval == 20
    assert recipe.extra_config["rollout_max_prompt_len"] == 1024
    assert recipe.extra_config["rollout_max_context_len"] == 5120
    assert recipe.extra_config["rollout_top_k"] == 16
    assert recipe.extra_config["seq_length"] == 5120
    assert any("protobuf==7.35.1" in command for command in recipe.image_run_commands)
    assert any("math-verify==0.9.0" in command for command in recipe.image_run_commands)
    assert any("tensorboard.compat.proto" in command for command in recipe.image_run_commands)
    assert recipe.metrics.modal_wandb_secret_name == "wandb-secret"

    resolved = MODULE.summary(config, smoke_test=False)
    assert resolved["dataset_file"] == "data/math-00000-of-00001.parquet"
    assert resolved["responses_per_rollout"] == 256
    assert resolved["optimizer_updates_per_rollout"] == 4
    assert resolved["total_sampled_responses"] == 25_600
    assert resolved["reward"] == "skywork_math_reward"


def test_skywork_row_matches_lightning_weave_math_format():
    row = {
        "prompt": [{"role": "user", "content": "  What is 2 + 4?  "}],
        "reward_model": {"ground_truth": '["6"]', "style": "rule"},
    }

    converted = MODULE.adapt_skywork_math_row(row)

    assert converted["label"] == "6"
    assert converted["prompt"] == [
        {
            "role": "user",
            "content": (f"{MODULE.PROMPT_PREFIX}What is 2 + 4?{MODULE.PROMPT_SUFFIX}"),
        }
    ]


def test_skywork_loader_reads_only_the_math_parquet(fake_training_gym, monkeypatch):
    calls = {}

    class FakeDataset:
        column_names = ["prompt", "reward_model"]

        def map(self, function, **kwargs):
            calls["map_function"] = function
            calls["map_kwargs"] = kwargs
            return self

    def fake_load_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return FakeDataset()

    datasets = types.ModuleType("datasets")
    datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    config = MODULE.build_config(MODULE.parser().parse_args([]))
    materialized = config.dataset._load_hf_dataset()

    assert isinstance(materialized, FakeDataset)
    assert calls["args"] == ("parquet",)
    assert calls["kwargs"] == {
        "data_files": {"train": MODULE.DATASET_URI},
        "split": "train",
    }
    assert calls["map_kwargs"]["remove_columns"] == ["prompt", "reward_model"]


def test_skywork_reward_disables_thread_unsafe_math_verify_timeouts(monkeypatch):
    math_verify = types.ModuleType("math_verify")
    calls = []

    def parse(value, *, parsing_timeout):
        calls.append(("parse", parsing_timeout))
        assert parsing_timeout is None
        return [value.replace("$", "").replace("<|im_end|>", "").strip("* ")]

    def verify(target, prediction, *, timeout_seconds):
        calls.append(("verify", timeout_seconds))
        assert timeout_seconds is None
        return target == prediction

    math_verify.parse = parse
    math_verify.verify = verify
    monkeypatch.setitem(sys.modules, "math_verify", math_verify)

    samples = [
        types.SimpleNamespace(
            response="<think>done</think>\n\n**Answer: 23**<|im_end|>",
            label="23",
        ),
        types.SimpleNamespace(response="<think>unfinished</think>", label="23"),
    ]

    rewards = asyncio.run(MODULE.skywork_math_reward(None, samples))

    assert rewards == [1.0, 0.0]
    assert calls == [("parse", None), ("parse", None), ("verify", None)]


def test_smoke_test_is_small_but_parallelism_safe(fake_training_gym):
    args = MODULE.parser().parse_args(["--smoke-test"])
    config = MODULE.build_config(args)

    assert config.recipe.num_rollout == 1
    assert config.recipe.save_interval == 1
    assert config.recipe.rollout_batch_size == 4
    assert config.recipe.n_samples_per_prompt == 4
    assert config.recipe.global_batch_size == 16
    assert config.recipe.rollout_max_response_len == 4096
    assert config.recipe.max_tokens_per_gpu == 5120
    assert config.recipe.extra_config["rollout_max_context_len"] == 5120
    assert config.recipe.extra_config["seq_length"] == 5120


def test_launcher_is_pinned_and_contains_no_hand_rolled_modal_stack():
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts/train_math_grpo.sh").read_text(encoding="utf-8")

    assert MODULE.TRAINING_GYM_COMMIT in source
    assert '# requires-python = "==3.12.*"' in source
    assert "uv run --python 3.12" in wrapper
    assert "modal-training-gym @ git+https://github.com/modal-projects/training-gym.git@" in source
    assert "Offline Direct-OPD" in source
    assert "import subprocess" not in source
    assert "modal.App" not in source
    assert "ray start" not in source
