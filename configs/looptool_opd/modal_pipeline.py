# /// script
# requires-python = "==3.12.*"
# dependencies = ["modal==1.5.5"]
# ///
"""Modal orchestration for LoopTool single-anchor Offline Direct-OPD.

Nothing runs at import time. ``prepare`` is CPU-only; ``cache``, ``convert``,
``train``, and ``all`` allocate GPUs only when explicitly selected.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import modal

# Modal copies the entrypoint itself to ``/root/modal_pipeline.py`` while the
# repository is mounted separately.  Resolve imports from whichever layout is
# present instead of deriving the repository root unconditionally from
# ``__file__``.
ENTRYPOINT_DIR = Path(__file__).resolve().parent
MOUNTED_PROJECT_ROOT = Path("/workspace/Lightning-Weave")
if (ENTRYPOINT_DIR / "config.py").is_file():
    CONFIG_DIR = ENTRYPOINT_DIR
    PROJECT_ROOT = CONFIG_DIR.parents[1]
elif (MOUNTED_PROJECT_ROOT / "configs/looptool_opd/config.py").is_file():
    PROJECT_ROOT = MOUNTED_PROJECT_ROOT
    CONFIG_DIR = PROJECT_ROOT / "configs/looptool_opd"
else:
    raise FileNotFoundError("cannot locate configs/looptool_opd/config.py")
sys.path.insert(0, str(CONFIG_DIR))

from config import (
    EXPECTED_CANONICAL_ROWS,
    MODAL_APP_NAME,
    MODAL_CHECKPOINT_VOLUME,
    MODAL_DATA_VOLUME,
    MODAL_MODEL_VOLUME,
    POST_TEACHER_MODEL,
    POST_TEACHER_REVISION,
    PRE_TEACHER_MODEL,
    PRE_TEACHER_REVISION,
    RECIPE,
    REMOTE_CHECKPOINT_ROOT,
    REMOTE_DATA_ROOT,
    REMOTE_MODEL_ROOT,
    REMOTE_REPO,
    ROLLOUT_IMAGE,
    RUNTIME_IMAGE,
    STUDENT_MODEL,
    STUDENT_REVISION,
)

CURATION_DIR = f"{REMOTE_DATA_ROOT}/curation"
CANONICAL_DATA = f"{CURATION_DIR}/looptool_rl_canonical.jsonl"
PROMPT_DATA = f"{REMOTE_DATA_ROOT}/prompts.parquet"
PROMPT_SUMMARY = f"{REMOTE_DATA_ROOT}/prompts_summary.json"
RECIPE_LOCK = f"{REMOTE_DATA_ROOT}/recipe.json"
ASSET_LOCK = f"{REMOTE_DATA_ROOT}/assets.json"
ROLLOUT_DIR = f"{REMOTE_DATA_ROOT}/rollouts"
POST_DIR = f"{REMOTE_DATA_ROOT}/anchor/post"
PRE_DIR = f"{REMOTE_DATA_ROOT}/anchor/pre"
FINAL_DIR = f"{REMOTE_DATA_ROOT}/anchor/final"
MANIFEST = f"{FINAL_DIR}/manifest.json"
CONVERTED_CHECKPOINT = f"{REMOTE_CHECKPOINT_ROOT}/initial-megatron"
TRAINING_CHECKPOINT = f"{REMOTE_CHECKPOINT_ROOT}/train"
EXPORTED_CHECKPOINT = f"{REMOTE_CHECKPOINT_ROOT}/hf"

app = modal.App(MODAL_APP_NAME)
data_volume = modal.Volume.from_name(MODAL_DATA_VOLUME, create_if_missing=True)
model_volume = modal.Volume.from_name(MODAL_MODEL_VOLUME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(MODAL_CHECKPOINT_VOLUME, create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "datasets==4.1.1",
        "transformers==4.57.1",
        "jinja2==3.1.6",
        "jsonschema==4.25.1",
        "datasketch==1.6.5",
        "bfcl-eval==2026.3.23",
    )
    .env({"HF_HOME": REMOTE_MODEL_ROOT, "PYTHONPATH": REMOTE_REPO})
    .add_local_dir(str(PROJECT_ROOT), remote_path=REMOTE_REPO, copy=True)
)

rollout_image = (
    modal.Image.from_registry(ROLLOUT_IMAGE)
    .entrypoint([])
    # The official vLLM image exposes /usr/bin/python3 but no `python` alias.
    # Modal's uv_pip_install resolves its interpreter with `command -v python`.
    .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python")
    .uv_pip_install("pyarrow==20.0.0")
    .env({"HF_HOME": REMOTE_MODEL_ROOT, "PYTHONPATH": REMOTE_REPO, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    .add_local_dir(str(PROJECT_ROOT), remote_path=REMOTE_REPO, copy=True)
)

runtime_image = (
    modal.Image.from_registry(RUNTIME_IMAGE)
    .entrypoint([])
    .env(
        {
            "HF_HOME": REMOTE_MODEL_ROOT,
            "PYTHONPATH": f"{REMOTE_REPO}:/root/Megatron-LM",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_DEBUG": "WARN",
        }
    )
    .add_local_dir(str(PROJECT_ROOT), remote_path=REMOTE_REPO, copy=True)
)

DATA_AND_MODELS = {"/opd": data_volume, REMOTE_MODEL_ROOT: model_volume}
ALL_VOLUMES = {**DATA_AND_MODELS, "/checkpoints": checkpoint_volume}


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=REMOTE_REPO, env=env)


def snapshot_path(repo: str, revision: str) -> str:
    namespace, name = repo.split("/", 1)
    return f"{REMOTE_MODEL_ROOT}/models--{namespace}--{name}/snapshots/{revision}"


def require_paths(paths: Iterable[str]) -> None:
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("missing prerequisite artifacts:\n" + "\n".join(missing))


def write_recipe_lock() -> None:
    expected = RECIPE.resolved()
    path = Path(RECIPE_LOCK)
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError(f"recipe lock differs from this checkout: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@app.function(image=cpu_image, cpu=4, memory=8_192, timeout=21_600, volumes={REMOTE_MODEL_ROOT: model_volume})
def download_models(full: bool = True) -> dict[str, str]:
    """Download immutable snapshots, or only CPU-stage tokenizer/config files."""

    from huggingface_hub import snapshot_download

    resolved = {}
    allow_patterns = None
    if not full:
        allow_patterns = [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
        ]
    for role, repo, revision in (
        ("student", STUDENT_MODEL, STUDENT_REVISION),
        ("pre_teacher", PRE_TEACHER_MODEL, PRE_TEACHER_REVISION),
        ("post_teacher", POST_TEACHER_MODEL, POST_TEACHER_REVISION),
    ):
        path = snapshot_download(
            repo_id=repo,
            revision=revision,
            cache_dir=REMOTE_MODEL_ROOT,
            allow_patterns=allow_patterns,
        )
        if Path(path).name != revision:
            raise RuntimeError(f"{role} resolved to {Path(path).name}, expected {revision}")
        resolved[role] = path
    model_volume.commit()
    return resolved


@app.function(image=cpu_image, cpu=8, memory=32_768, timeout=21_600, volumes=DATA_AND_MODELS)
def prepare_dataset() -> dict[str, object]:
    """Run canonical curation, render with the student template, and select prompts."""

    data_volume.reload()
    model_volume.reload()
    Path(REMOTE_DATA_ROOT).mkdir(parents=True, exist_ok=True)
    if not Path(CANONICAL_DATA).exists():
        run(
            [
                "python", "data_curation/prepare_looptool_rl.py", "--output-dir", CURATION_DIR,
                "--max-prompt-tokens", str(RECIPE.max_prompt_tokens), "--bfcl-audit", "on", "--workers", "8",
            ]
        )
    if not Path(PROMPT_DATA).exists() or not Path(PROMPT_SUMMARY).exists():
        if Path(PROMPT_DATA).exists() != Path(PROMPT_SUMMARY).exists():
            raise RuntimeError("only one prompt artifact exists; use a new versioned REMOTE_DATA_ROOT")
        run(
            [
                "python", "data_curation/prepare_direct_opd_looptool.py",
                "--input", CANONICAL_DATA, "--output", PROMPT_DATA, "--summary", PROMPT_SUMMARY,
                "--tokenizer", snapshot_path(STUDENT_MODEL, STUDENT_REVISION),
                "--tokenizer-revision", STUDENT_REVISION,
                "--max-prompt-length", str(RECIPE.max_prompt_tokens), "--num-prompts", str(RECIPE.selected_prompts),
                "--selection-seed", str(RECIPE.data_seed), "--expected-input-rows", str(EXPECTED_CANONICAL_ROWS),
            ]
        )
    write_recipe_lock()
    summary = json.loads(Path(PROMPT_SUMMARY).read_text(encoding="utf-8"))
    if summary["selected_prompts"] != RECIPE.selected_prompts or summary["targets_in_output"] is not False:
        raise RuntimeError("prepared prompt summary violates the locked recipe")
    data_volume.commit()
    return summary


@app.function(image=cpu_image, cpu=4, memory=8_192, timeout=3_600, volumes=DATA_AND_MODELS)
def prepare_asset_lock() -> str:
    """Verify tokenizer/action-space compatibility for the three frozen roles."""

    data_volume.reload()
    model_volume.reload()
    require_paths(
        [
            PROMPT_DATA, RECIPE_LOCK, snapshot_path(STUDENT_MODEL, STUDENT_REVISION),
            snapshot_path(PRE_TEACHER_MODEL, PRE_TEACHER_REVISION),
            snapshot_path(POST_TEACHER_MODEL, POST_TEACHER_REVISION),
        ]
    )
    if not Path(ASSET_LOCK).exists():
        run(
            [
                "python", "data_curation/prepare_direct_opd_assets.py",
                "--student", snapshot_path(STUDENT_MODEL, STUDENT_REVISION), "--student-revision", STUDENT_REVISION,
                "--pre-teacher", snapshot_path(PRE_TEACHER_MODEL, PRE_TEACHER_REVISION),
                "--pre-teacher-revision", PRE_TEACHER_REVISION,
                "--post-teacher", snapshot_path(POST_TEACHER_MODEL, POST_TEACHER_REVISION),
                "--post-teacher-revision", POST_TEACHER_REVISION,
                "--output", ASSET_LOCK,
            ]
        )
    data_volume.commit()
    return ASSET_LOCK


@app.function(image=rollout_image, gpu="H100", cpu=8, memory=32_768, timeout=43_200, volumes=DATA_AND_MODELS)
def collect_rollout_shard(rank: int, world_size: int) -> str:
    """Generate one deterministic partition of the frozen Qwen3-4B cache."""

    data_volume.reload()
    model_volume.reload()
    require_paths([PROMPT_DATA, ASSET_LOCK, snapshot_path(STUDENT_MODEL, STUDENT_REVISION)])
    output_dir = Path(ROLLOUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob(f"rollouts-r{rank:05d}-*.parquet"))
    if existing:
        return f"rank {rank}: {len(existing)} existing shards"
    run(
        [
            "python", "data_curation/collect_direct_opd_rollouts.py",
            "--model", snapshot_path(STUDENT_MODEL, STUDENT_REVISION), "--model-revision", STUDENT_REVISION,
            "--asset-lock", ASSET_LOCK, "--input", PROMPT_DATA, "--output-dir", ROLLOUT_DIR,
            "--label-key", "label", "--prompt-id-key", "prompt_id",
            "--max-prompts", str(RECIPE.selected_prompts), "--responses-per-prompt", str(RECIPE.responses_per_prompt),
            "--max-prompt-length", str(RECIPE.max_prompt_tokens),
            "--max-response-length", str(RECIPE.max_response_tokens),
            "--top-k", str(RECIPE.top_k), "--temperature", str(RECIPE.temperature), "--top-p", str(RECIPE.top_p),
            "--seed", str(RECIPE.data_seed), "--batch-size", "64", "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", "0.90", "--dtype", "bfloat16", "--compilation-mode", "0",
            "--cudagraph-mode", "NONE", "--max-num-seqs", "32", "--shard-size", "800",
            "--rank", str(rank), "--world-size", str(world_size), "--enable-thinking",
        ]
    )
    data_volume.commit()
    return f"rank {rank}: complete"


SCORE_STAGES = {
    "post": (POST_TEACHER_MODEL, POST_TEACHER_REVISION, "post_teacher_log_probs", ROLLOUT_DIR, POST_DIR),
    "pre": (PRE_TEACHER_MODEL, PRE_TEACHER_REVISION, "pre_teacher_log_probs", POST_DIR, PRE_DIR),
    "reference": (STUDENT_MODEL, STUDENT_REVISION, "student_ref_sampled_log_probs", PRE_DIR, FINAL_DIR),
}


@app.function(image=runtime_image, gpu="H100", cpu=8, memory=32_768, timeout=43_200, volumes=DATA_AND_MODELS)
def score_anchor_shard(stage: str, rank: int, world_size: int) -> str:
    """Score one shard partition; stages are run post -> pre -> reference."""

    if stage not in SCORE_STAGES:
        raise ValueError(f"unknown score stage: {stage}")
    data_volume.reload()
    model_volume.reload()
    repo, revision, field, input_dir, output_dir = SCORE_STAGES[stage]
    require_paths([ASSET_LOCK, input_dir, snapshot_path(repo, revision)])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    run(
        [
            "python", "data_curation/precompute_direct_opd_scores.py",
            "--model", snapshot_path(repo, revision), "--model-revision", revision,
            "--asset-lock", ASSET_LOCK, "--score-field", field, "--input", input_dir, "--output-dir", output_dir,
            "--dtype", "bfloat16", "--device", "cuda:0", "--row-batch-size", "1",
            "--chunk-size", str(RECIPE.sequence_tokens), "--attn-implementation", "flash_attention_2",
            "--rank", str(rank), "--world-size", str(world_size),
        ]
    )
    data_volume.commit()
    return f"{stage} rank {rank}: complete"


@app.function(image=cpu_image, cpu=4, memory=16_384, timeout=3_600, volumes=DATA_AND_MODELS)
def seal_cache() -> dict[str, object]:
    """Seal shard hashes and reject an incomplete or wrong-sized cache."""

    import pyarrow.parquet as pq

    data_volume.reload()
    require_paths([FINAL_DIR, ASSET_LOCK, PROMPT_DATA])
    shards = sorted(Path(FINAL_DIR).glob("*.parquet"))
    rows = sum(pq.read_metadata(path).num_rows for path in shards)
    if rows != RECIPE.cached_trajectories:
        raise RuntimeError(f"final cache has {rows:,} rows; expected {RECIPE.cached_trajectories:,}")
    if not Path(MANIFEST).exists():
        run(
            [
                "python", "data_curation/prepare_direct_opd_manifest.py", "--input", FINAL_DIR,
                "--source-dataset", PROMPT_DATA, "--asset-lock", ASSET_LOCK, "--manifest-out", MANIFEST,
            ]
        )
    manifest = json.loads(Path(MANIFEST).read_text(encoding="utf-8"))
    if manifest["total_rows"] != RECIPE.cached_trajectories:
        raise RuntimeError("sealed manifest row count violates the recipe")
    data_volume.commit()
    return manifest


@app.function(image=runtime_image, gpu="H100:8", cpu=32, memory=131_072, timeout=14_400, volumes=ALL_VOLUMES)
def convert_student_checkpoint() -> str:
    """Convert only the trainable Qwen3-4B student to Megatron format."""

    model_volume.reload()
    checkpoint_volume.reload()
    student = snapshot_path(STUDENT_MODEL, STUDENT_REVISION)
    require_paths([student])
    output = Path(CONVERTED_CHECKPOINT)
    marker = output / "modal_conversion.json"
    if marker.exists():
        return str(output)
    if output.exists():
        raise RuntimeError(f"incomplete conversion directory exists: {output}")
    environment = os.environ.copy()
    environment.update(
        {
            "STUDENT_MODEL": student,
            "OUTPUT_DIR": str(output),
            "MODEL_TYPE": "qwen3-4B",
            "NUM_GPUS": "8",
            "MEGATRON_PATH": "/root/Megatron-LM",
        }
    )
    run(["bash", "scripts/convert_checkpoint.sh", "to-megatron"], env=environment)
    marker.write_text(
        json.dumps({"student": STUDENT_MODEL, "revision": STUDENT_REVISION}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_volume.commit()
    return str(output)


@app.function(
    image=runtime_image, gpu="H100:8", cpu=32, memory=131_072, timeout=86_400,
    # Volume writes are staged on local disk until commit; ten ~45 GB torch_dist
    # checkpoints (weights + Adam state) exceed the default container disk.
    ephemeral_disk=1_048_576, volumes=ALL_VOLUMES, secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_offline_dopd(resume: bool = False) -> str:
    """Run the 50-round actor-only Offline Direct-OPD training job."""

    data_volume.reload()
    model_volume.reload()
    checkpoint_volume.reload()
    student = snapshot_path(STUDENT_MODEL, STUDENT_REVISION)
    require_paths([student, FINAL_DIR, MANIFEST, CONVERTED_CHECKPOINT])
    save = Path(TRAINING_CHECKPOINT)
    if save.exists() and not resume:
        raise RuntimeError(f"training output already exists: {save}; relaunch with --resume only after inspection")
    load = str(save if resume else Path(CONVERTED_CHECKPOINT))
    run(
        [
            "python", "configs/lightning_weave/train.py", "--model-type", "qwen3-4B", "--student", student,
            "--data", FINAL_DIR, "--manifest", MANIFEST, "--load", load, "--save", str(save),
            "--num-gpus", str(RECIPE.training_gpus), "--alpha", str(RECIPE.alpha), "--lr", str(RECIPE.learning_rate),
            "--rollout-batch-size", str(RECIPE.rollout_batch_size),
            "--global-batch-size", str(RECIPE.global_batch_size),
            "--num-rollout", str(RECIPE.replay_rounds), "--save-interval", "5",
            "--max-tokens-per-gpu", str(RECIPE.sequence_tokens), "--top-k", str(RECIPE.top_k),
            "--responses-per-prompt", str(RECIPE.responses_per_prompt),
            "--max-prompt-length", str(RECIPE.max_prompt_tokens),
            "--max-response-length", str(RECIPE.max_response_tokens),
            "--temperature", str(RECIPE.temperature), "--top-p", str(RECIPE.top_p),
            "--seed", str(RECIPE.training_seed), "--rollout-seed", str(RECIPE.data_seed),
            "--loss-mode", "tilted_target", "--wandb-project", "looptool-qwen3-4b-opd",
            "--wandb-group", MODAL_APP_NAME,
        ]
    )
    checkpoint_volume.commit()
    return str(save)


@app.function(image=runtime_image, cpu=16, memory=65_536, timeout=14_400, volumes=ALL_VOLUMES)
def export_hf_checkpoint() -> str:
    """Export the latest completed Megatron iteration for downstream evaluation."""

    model_volume.reload()
    checkpoint_volume.reload()
    student = snapshot_path(STUDENT_MODEL, STUDENT_REVISION)
    require_paths([student, TRAINING_CHECKPOINT])
    iterations = sorted(Path(TRAINING_CHECKPOINT).glob("iter_[0-9]*"))
    if not iterations:
        raise FileNotFoundError(f"no completed iter_* checkpoint under {TRAINING_CHECKPOINT}")
    output = Path(EXPORTED_CHECKPOINT)
    if output.exists():
        raise RuntimeError(f"export output already exists: {output}")
    environment = os.environ.copy()
    environment.update(
        {
            "STUDENT_MODEL": student,
            "INPUT_DIR": str(iterations[-1]),
            "OUTPUT_DIR": str(output),
            "MODEL_TYPE": "qwen3-4B",
            "MEGATRON_PATH": "/root/Megatron-LM",
        }
    )
    run(["bash", "scripts/convert_checkpoint.sh", "to-hf"], env=environment)
    checkpoint_volume.commit()
    return str(output)


def run_parallel(function: modal.Function, items: list[tuple[object, ...]]) -> None:
    for result in function.starmap(items, order_outputs=False):
        print(result, flush=True)


def build_cache(workers: int) -> None:
    if workers != RECIPE.cache_workers:
        raise ValueError(
            f"this cache version is locked to {RECIPE.cache_workers} workers; "
            "change Recipe.cache_workers and REMOTE_DATA_ROOT together to create a new cache"
        )
    download_models.remote(full=True)
    prepare_dataset.remote()
    prepare_asset_lock.remote()
    ranks = [(rank, workers) for rank in range(workers)]
    run_parallel(collect_rollout_shard, ranks)
    for stage in ("post", "pre", "reference"):
        run_parallel(score_anchor_shard, [(stage, rank, workers) for rank in range(workers)])
    manifest = seal_cache.remote()
    print(json.dumps({"sealed_cache": MANIFEST, "rows": manifest["total_rows"]}, indent=2), flush=True)


@app.local_entrypoint()
def main(action: str = "plan", workers: int = RECIPE.cache_workers, resume: bool = False) -> None:
    """Actions: plan, prepare, cache, convert, train, export, or all."""

    if action == "plan":
        print(json.dumps(RECIPE.resolved(), indent=2, sort_keys=True))
    elif action == "prepare":
        download_models.remote(full=False)
        print(json.dumps(prepare_dataset.remote(), indent=2, sort_keys=True))
        print(prepare_asset_lock.remote())
    elif action == "cache":
        build_cache(workers)
    elif action == "convert":
        download_models.remote(full=True)
        print(convert_student_checkpoint.remote())
    elif action == "train":
        print(train_offline_dopd.remote(resume=resume))
    elif action == "export":
        print(export_hf_checkpoint.remote())
    elif action == "all":
        build_cache(workers)
        print(convert_student_checkpoint.remote())
        print(train_offline_dopd.remote(resume=resume))
        print(export_hf_checkpoint.remote())
    else:
        raise ValueError(f"unknown action {action!r}; choose plan, prepare, cache, convert, train, export, or all")
