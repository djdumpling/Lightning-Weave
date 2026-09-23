import importlib.util
import inspect
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/bfcl_grpo"
sys.path.insert(0, str(CONFIG_DIR))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ADAPTER = load("bfcl_adapter", CONFIG_DIR / "bfcl_adapter.py")
TRAIN = load("bfcl_grpo_train", CONFIG_DIR / "train.py")


class FakeConfig:
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def fake_gym(monkeypatch):
    module = types.ModuleType("modal_training_gym")

    @dataclass
    class ModelArchitecture:
        rotary_base: int = 1_000_000

    class Qwen3_4B(FakeConfig):
        architecture = ModelArchitecture()

    for name, value in {
        "DatasetConfig": type("DatasetConfig", (FakeConfig,), {}),
        "Qwen3_4B": Qwen3_4B,
        "Qwen3_4B_Recipe": type("Qwen3_4B_Recipe", (FakeConfig,), {}),
        "TrainConfig": type("TrainConfig", (FakeConfig,), {}),
        "WandbConfig": type("WandbConfig", (FakeConfig,), {}),
    }.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "modal_training_gym", module)
    return module


def test_recipe_uses_requested_model_context_and_modal_shape(fake_gym):
    config = TRAIN.build_config(TRAIN.parser().parse_args([]))
    recipe = config.recipe

    assert config.model.model_name == "Qwen/Qwen3-4B-Thinking-2507"
    assert config.model.architecture.rotary_base == 5_000_000
    assert recipe.gpu_type == "H100"
    assert recipe.actor_num_gpus_per_node == 8
    assert recipe.tensor_model_parallel_size == 1
    assert recipe.custom_generate_function is ADAPTER.bfcl_turn_rollout
    assert recipe.rollout_max_response_len == 4096
    assert recipe.rollout_temperature == 0.6
    assert recipe.rollout_top_p == 0.95
    assert recipe.extra_config["rollout_top_k"] == 20
    assert recipe.extra_config["rollout_max_prompt_len"] == 8192
    assert recipe.extra_config["rollout_max_context_len"] == 32768
    assert recipe.extra_config["bfcl_max_assistant_steps"] == 20
    assert recipe.metrics.modal_wandb_secret_name == "wandb-secret"


def test_smoke_shape_and_dataset_limit(fake_gym):
    config = TRAIN.build_config(TRAIN.parser().parse_args(["--smoke-test"]))
    assert config.recipe.num_rollout == 1
    assert config.recipe.rollout_batch_size == 4
    assert config.recipe.n_samples_per_prompt == 2
    assert config.recipe.global_batch_size == 8
    assert config.dataset.task_limit == 4


def test_context_length_override_is_applied_consistently(fake_gym):
    args = TRAIN.parser().parse_args(["--max-model-tokens", "65536"])
    config = TRAIN.build_config(args)
    recipe = config.recipe

    assert recipe.max_tokens_per_gpu == 65536
    assert recipe.extra_config["rollout_max_context_len"] == 65536
    assert recipe.extra_config["seq_length"] == 65536
    assert recipe.extra_config["max_position_embeddings"] == 65536
    assert recipe.extra_config["bfcl_max_model_tokens"] == 65536
    assert TRAIN.resolved(config, False)["max_model_tokens"] == 65536


def test_dataset_expands_each_user_turn(monkeypatch):
    entries = [
        {
            "id": "multi_turn_base_0",
            "question": [
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}],
            ],
            "initial_config": {},
            "involved_classes": [],
        }
    ]
    answers = [{"id": "multi_turn_base_0", "ground_truth": [["f(x=1)"], ["g(y=2)"]]}]

    monkeypatch.setattr(ADAPTER, "_data_dir", lambda: "/fake")
    monkeypatch.setattr(
        ADAPTER,
        "_jsonl",
        lambda path: answers if "possible_answer" in path else entries,
    )
    monkeypatch.setattr(ADAPTER, "load_tool_schemas", lambda *_: {})
    monkeypatch.setattr(ADAPTER, "replay", lambda label, calls: ({}, [str(i) for i in range(len(calls))]))
    category = types.ModuleType("bfcl_eval.constants.category_mapping")
    category.VERSION_PREFIX = "BFCL_v4"
    monkeypatch.setitem(sys.modules, "bfcl_eval", types.ModuleType("bfcl_eval"))
    monkeypatch.setitem(sys.modules, "bfcl_eval.constants", types.ModuleType("bfcl_eval.constants"))
    monkeypatch.setitem(sys.modules, "bfcl_eval.constants.category_mapping", category)

    dataset_class = ADAPTER.make_dataset_class(type("DatasetConfig", (), {}))
    rows = dataset_class(split="train", eval_tasks=0).rows()
    labels = [json.loads(row["label"]) for row in rows]

    assert len(rows) == 2
    assert [label["start_step"] for label in labels] == [0, 1]
    assert [label["target_steps"] for label in labels] == [1, 1]


def test_launcher_is_pinned_and_does_not_launch_a_hand_rolled_stack():
    source = (CONFIG_DIR / "train.py").read_text(encoding="utf-8")
    wrapper = (ROOT / "scripts/train_bfcl_grpo.sh").read_text(encoding="utf-8")
    assert TRAIN.TRAINING_GYM_COMMIT in source
    assert "bfcl-eval==2026.3.23" in source
    assert "uv run --python 3.12" in wrapper
    assert "alex-dev-2" in wrapper
    assert "modal.App" not in source
    assert "ray start" not in source


def test_sampling_request_preserves_top_p_replay_metadata():
    original = {"temperature": 0.6, "top_p": 0.95, "custom_params": {"existing": True}}
    request = ADAPTER._sampling_request(original, 4096)

    assert request["max_new_tokens"] == 4096
    assert request["custom_params"] == {
        "existing": True,
        "return_top_p_token_ids": True,
    }
    assert original["custom_params"] == {"existing": True}


def test_generated_tokens_uses_sglang_ids_and_log_probs():
    output = {
        "text": "retokenization would be wrong",
        "meta_info": {"output_token_logprobs": [(-0.25, 101, None), (-0.5, 202, None)]},
    }

    assert ADAPTER._generated_tokens(output) == ([101, 202], [-0.25, -0.5])


def test_generated_tokens_requires_training_metadata():
    with pytest.raises(RuntimeError, match="output_token_logprobs"):
        ADAPTER._generated_tokens({"text": "missing", "meta_info": {}})


def test_rollout_uses_slime_response_metadata_contract():
    source = inspect.getsource(ADAPTER.bfcl_turn_rollout)
    assert '"return_logprob": True' in source
    assert source.count("sample.append_response_tokens(") == 2
    assert "tokenizer(model_text" not in source
