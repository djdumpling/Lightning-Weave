"""Pinned one-arm Qwen3.5 retail transfer experiment (2026-10-12)."""
from configs.looptool_opd.config import RUNTIME_IMAGE
from configs.tau_bench_eval.config import PROFILES

MODELS = {
    "student": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "pre_teacher": ("Qwen/Qwen3.5-9B-Base", "68c46c4b3498877f3ef123c856ecfde50c39f404"),
    "post_teacher": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
}
PROFILE = "userdeepseek-qwen35-mentor"
PROTOCOL = PROFILES[PROFILE]
ROLLOUT_IMAGE = PROTOCOL.serving_image
DATA_ROOT = "/opd/tau2-retail-uniform-qwen35-mentor-20261012-v1"
CHECKPOINT_ROOT = "/checkpoints/qwen35-retail-uniform-20261012-v1"
SEEDS = (1234, 5678, 9012)
MAX_PROMPT = 16384
MAX_RESPONSE = 8192
NUM_PROMPTS = 3200
K = 4
ALPHA = 2.0
LR = 1e-6
UPDATES = 200
GLOBAL_BATCH = 64
TRAIN_GPUS = 4


def snapshot(role):
    repo, revision = MODELS[role]
    return f"/models/models--{repo.replace('/', '--')}/snapshots/{revision}"


def language_model(role):
    return f"/models/qwen35-text/{MODELS[role][1]}"


def data_root(smoke):
    return DATA_ROOT + ("/smoke" if smoke else "/full")


def collection_run(smoke):
    return "collect-qwen35-mentor-20261012-" + ("smoke" if smoke else "full") + "-v1"
