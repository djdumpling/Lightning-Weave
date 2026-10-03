"""Keep CPU tests available without installing the Megatron training stack or Modal Training Gym."""

from dataclasses import dataclass
import importlib.util
import sys
import types

import pytest


collect_ignore = []
if importlib.util.find_spec("megatron") is None:
    collect_ignore.append("test_megatron_direct_opd_cp.py")


class FakeConfig:
    def __init__(self, *args, **kwargs):
        self.args = args
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class ModelArchitecture:
    rotary_base: int = 1_000_000


@pytest.fixture
def fake_training_gym(monkeypatch):
    """A ``modal_training_gym`` whose config classes only record the arguments they are built with."""
    module = types.ModuleType("modal_training_gym")
    for name in (
        "DatasetConfig", "HuggingFaceDataset", "Qwen3_4B_Recipe", "Qwen3_4B_VllmRecipe", "TrainConfig", "WandbConfig"
    ):
        setattr(module, name, type(name, (FakeConfig,), {}))
    module.Qwen3_4B = type("Qwen3_4B", (FakeConfig,), {"architecture": ModelArchitecture()})
    monkeypatch.setitem(sys.modules, "modal_training_gym", module)
    return module
