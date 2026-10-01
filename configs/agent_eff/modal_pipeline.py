# /// script
# requires-python = "==3.12.*"
# dependencies = ["modal==1.5.5"]
# ///
"""Score donors, analyze shifts, and compose/train/export agent-efficiency students on Modal.

See docs/agent_efficiency_synthesis.md for commands and the experiment protocol.
Long-running stages execute server-side and reuse completed artifacts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

ENTRYPOINT_DIR = Path(__file__).resolve().parent
MOUNTED_PROJECT_ROOT = Path("/workspace/Lightning-Weave")
if (ENTRYPOINT_DIR / "config.py").is_file():
    PROJECT_ROOT = ENTRYPOINT_DIR.parents[1]
elif (MOUNTED_PROJECT_ROOT / "configs/agent_eff/config.py").is_file():
    PROJECT_ROOT = MOUNTED_PROJECT_ROOT
else:
    raise FileNotFoundError("cannot locate configs/agent_eff/config.py")
sys.path.insert(0, str(PROJECT_ROOT))

from configs.agent_eff import config as agent_eff  # noqa: E402

from configs.agent_eff.config import (  # noqa: E402
    AGENT_ACC,
    ANALYSIS_ROOT,
    BASE_CACHE,
    CANONICAL_DATA,
    CHECKPOINT_ROOT,
    DONORS,
    INITIAL_MEGATRON,
    MODAL_APP_NAME,
    MODAL_CHECKPOINT_VOLUME,
    MODAL_DATA_VOLUME,
    MODAL_MODEL_VOLUME,
    PRECISION_CHECK,
    PROBE_ROOT,
    PROMPT_DATA,
    PROVENANCE_FILE,
    RECIPE,
    REMOTE_MODEL_ROOT,
    REMOTE_REPO,
    ROLLOUT_DIR,
    ROLLOUT_IMAGE,
    RUNTIME_IMAGE,
    STUDENT_MODEL,
    STUDENT_REVISION,
    STUDENT_TOKENIZER_JSON,
    SYNTHETIC_ROOT,
    TURN_POSITION_ROOT,
    geometry_spec,
    variant_sources,
    variant_specs,
)

app = modal.App(MODAL_APP_NAME)
data_volume = modal.Volume.from_name(MODAL_DATA_VOLUME, create_if_missing=True)
model_volume = modal.Volume.from_name(MODAL_MODEL_VOLUME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(MODAL_CHECKPOINT_VOLUME, create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy==2.2.6",
        "pyarrow==20.0.0",
        "safetensors==0.6.2",
        "torch==2.8.0",
        "transformers==4.57.1",
        "huggingface_hub==0.35.3",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .env({"HF_HOME": REMOTE_MODEL_ROOT, "PYTHONPATH": REMOTE_REPO})
    .add_local_dir(str(PROJECT_ROOT), remote_path=REMOTE_REPO, copy=True)
)

rollout_image = (
    modal.Image.from_registry(ROLLOUT_IMAGE)
    .entrypoint([])
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


def require_paths(paths) -> None:
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("missing prerequisite artifacts:\n" + "\n".join(missing))


def pair(name: str):
    return AGENT_ACC if name == AGENT_ACC.name else DONORS[name]


def verified_donors(names) -> dict[str, str]:
    """Scored chains of donors that passed ``verify``."""
    donors = {name: DONORS[name].scores for name in names if name != AGENT_ACC.name}
    require_paths([Path(DONORS[name].directory) / "evidence.json" for name in donors])
    return donors


def merged_untrained(names) -> str:
    """One {pair: [student ids]} file for every audited pair among ``names``."""
    merged = {}
    for name in names:
        path = Path(pair(name).directory) / "untrained.json"
        require_paths([path])
        merged.update(json.loads(path.read_text(encoding="utf-8")))
    output = Path(ANALYSIS_ROOT) / f"untrained-{'-'.join(sorted(merged))}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, sort_keys=True) + "\n", encoding="utf-8")
    return str(output)


def donor_arguments(donors: dict[str, str]) -> list[str]:
    return [argument for name, path in donors.items() for argument in ("--donor", f"{name}={path}")]


def training_root(variant: str, seed: int) -> Path:
    return Path(CHECKPOINT_ROOT) / "joint" / variant / f"seed{seed}"


@app.function(image=cpu_image, cpu=4, memory=16_384, timeout=21_600, volumes={REMOTE_MODEL_ROOT: model_volume})
def download(names: list[str]) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    models = {model for name in names for model in (pair(name).pre, pair(name).post)}
    resolved = {}
    for model in sorted(models, key=lambda item: item.repo):
        path = snapshot_download(repo_id=model.repo, revision=model.revision, cache_dir=REMOTE_MODEL_ROOT)
        if Path(path).name != model.revision:
            raise RuntimeError(f"{model.repo} resolved to {Path(path).name}, expected {model.revision}")
        resolved[model.repo] = path
        model_volume.commit()
    return resolved


@app.function(image=cpu_image, cpu=8, memory=32_768, timeout=7_200, volumes=DATA_AND_MODELS)
def prepare(name: str) -> dict:
    """Asset lock, alias file, untrained-token rows, and the coverage audit for one pair."""
    data_volume.reload()
    model_volume.reload()
    donor = pair(name)
    student = agent_eff.Model(STUDENT_MODEL, STUDENT_REVISION).path
    require_paths([student, donor.pre.path, donor.post.path, BASE_CACHE])
    directory = Path(donor.directory)
    directory.mkdir(parents=True, exist_ok=True)
    if name != AGENT_ACC.name and not (directory / "assets.json").exists():
        run(
            [
                "python", "data_curation/prepare_direct_opd_assets.py",
                "--student", student, "--student-revision", STUDENT_REVISION,
                "--post-teacher", donor.post.path, "--post-teacher-revision", donor.post.revision,
                "--pre-teacher", donor.pre.path, "--pre-teacher-revision", donor.pre.revision,
                "--output", str(directory / "assets.json"),
            ]
        )
    aliases = directory / "aliases.json"
    if aliases.exists() and json.loads(aliases.read_text()) != donor.aliases and any((directory / "post").glob("*.parquet")):
        raise RuntimeError(f"{name}'s aliases changed after scoring; remove {directory}/post and /pre to rescore")
    aliases.write_text(json.dumps(donor.aliases, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    run(
        [
            "python", "data_curation/audit_donor_tokens.py",
            "--name", name, "--student", student, "--post", donor.post.path, "--pre", donor.pre.path,
            "--special-token-alias", str(aliases), "--cache", BASE_CACHE, "--output-dir", str(directory),
        ]
    )
    data_volume.commit()
    return json.loads((directory / "audit.json").read_text(encoding="utf-8"))


# 96 GB of RAM: a 14B checkpoint stored in float32 (DeepCoder-14B) is cast to bfloat16 while it loads. Scoring is
# idempotent (finished shards are skipped, writes are atomic), so failed inputs are retried.
@app.function(
    image=runtime_image, gpu="H100", cpu=8, memory=98_304, timeout=43_200, volumes=DATA_AND_MODELS, retries=2
)
def score_donor_shard(name: str, stage: str, rank: int, world_size: int) -> str:
    """Score one partition of the cached rollouts with one anchor of a donor pair."""
    data_volume.reload()
    model_volume.reload()
    donor = DONORS[name]
    directory = Path(donor.directory)
    model, field, source, output = {
        "post": (donor.post, "post_teacher_log_probs", ROLLOUT_DIR, directory / "post"),
        "pre": (donor.pre, "pre_teacher_log_probs", directory / "post", directory / "pre"),
    }[stage]
    require_paths([directory / "assets.json", model.path, source])
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "python", "data_curation/precompute_direct_opd_scores.py",
        "--model", model.path, "--model-revision", model.revision,
        "--asset-lock", str(directory / "assets.json"), "--score-field", field,
        "--input", str(source), "--output-dir", str(output),
        "--dtype", "bfloat16", "--device", "cuda:0", "--row-batch-size", "1",
        "--chunk-size", str(RECIPE.sequence_tokens), "--attn-implementation", "flash_attention_2",
        "--rank", str(rank), "--world-size", str(world_size), "--per-candidate-validity",
    ]
    if donor.aliases:
        command += ["--special-token-alias", str(directory / "aliases.json")]
    run(command)
    data_volume.commit()
    return f"{name} {stage} rank {rank}: complete"


@app.function(image=cpu_image, cpu=8, memory=32_768, timeout=14_400, volumes=DATA_AND_MODELS)
def verify(name: str) -> dict:
    data_volume.reload()
    model_volume.reload()
    donor = DONORS[name]
    directory = Path(donor.directory)
    require_paths([donor.scores, directory / "untrained.json"])
    run(
        [
            "python", "data_curation/verify_donor_scores.py",
            "--name", name, "--base", BASE_CACHE, "--scores", donor.scores,
            "--post-revision", donor.post.revision, "--pre-revision", donor.pre.revision,
            "--untrained", str(directory / "untrained.json"), "--tokenizer-json", STUDENT_TOKENIZER_JSON,
            "--output", str(directory / "evidence.json"),
        ]
    )
    data_volume.commit()
    return json.loads((directory / "evidence.json").read_text(encoding="utf-8"))["evidence"]


@app.function(image=cpu_image, cpu=2, memory=4_096, timeout=86_400, nonpreemptible=True)
def score_chain(names: list[str], workers: int) -> dict:
    """Server-side post -> pre -> verify, so no stage transition depends on the local client."""
    evidence, failed = score_pairs(names, workers)
    if failed:
        raise RuntimeError(f"donor scoring failed: {failed}")
    return evidence


@app.function(image=cpu_image, cpu=2, memory=4_096, timeout=86_400)
def onboard_chain(names: list[str], workers: int, tag: str) -> str:
    """Server-side download -> prepare -> score -> verify for new donors, then geometry with the core."""
    download.remote(names)
    list(prepare.map(names))  # each audit is written next to the donor's scores
    print(json.dumps(score_chain.remote(names, workers), indent=2), flush=True)
    core = [name for name, item in DONORS.items() if item.core]
    return analyze.remote(tag, core + [name for name in names if name not in core])


def precision_root(name: str) -> Path:
    return Path(DONORS[name].directory) / "precision"


# float32 needs room: a 14B model's weights alone take 59 GB, so the re-score runs on an H200.
@app.function(image=runtime_image, gpu="H200", cpu=8, memory=131_072, timeout=14_400, volumes=DATA_AND_MODELS)
def precision_score(name: str) -> dict:
    """Re-score the first rows of the first cache shard with other numerics (PRECISION_CHECK), post then pre.

    The weights are rounded to bfloat16 exactly as in the main run, then computed in float32 with SDPA attention,
    so the two scores differ only in arithmetic. Everything else matches ``score_donor_shard``: the asset lock,
    aliases, per-candidate validity, and chunk size. Finished stages are skipped, so a restarted call resumes.
    """
    import pyarrow.parquet as pq

    from data_curation.shift_geometry import shard_names

    data_volume.reload()
    model_volume.reload()
    donor = DONORS[name]
    directory = Path(donor.directory)
    root = precision_root(name)
    require_paths([directory / "assets.json", directory / "aliases.json", donor.pre.path, donor.post.path])
    shard = shard_names(Path(BASE_CACHE))[0]
    subset = root / "input" / shard
    if not subset.exists():
        subset.parent.mkdir(parents=True, exist_ok=True)
        partial = subset.with_name(subset.name + ".partial")
        pq.write_table(pq.read_table(Path(ROLLOUT_DIR) / shard).slice(0, PRECISION_CHECK["rows"]), partial)
        partial.rename(subset)
        data_volume.commit()
    stages = {
        "post": (donor.post, "post_teacher_log_probs", root / "input"),
        "pre": (donor.pre, "pre_teacher_log_probs", root / "post"),
    }
    for stage, (model, field, source) in stages.items():
        output = root / stage
        output.mkdir(parents=True, exist_ok=True)
        run(
            [
                "python", "data_curation/precompute_direct_opd_scores.py",
                "--model", model.path, "--model-revision", model.revision,
                "--asset-lock", str(directory / "assets.json"), "--score-field", field,
                "--input", str(source), "--output-dir", str(output),
                "--dtype", PRECISION_CHECK["weights_dtype"], "--compute-dtype", PRECISION_CHECK["compute_dtype"],
                "--device", "cuda:0", "--row-batch-size", "1", "--chunk-size", str(RECIPE.sequence_tokens),
                "--attn-implementation", PRECISION_CHECK["attn_implementation"],
                "--per-candidate-validity", "--special-token-alias", str(directory / "aliases.json"),
            ]
        )
        data_volume.commit()
    return precision_compare.remote(name)


@app.function(image=cpu_image, cpu=4, memory=16_384, timeout=3_600, volumes=DATA_AND_MODELS)
def precision_compare(name: str) -> dict:
    """The re-scored shift against the main scores of the same rows (``check_score_precision.py``)."""
    data_volume.reload()
    donor = DONORS[name]
    directory = Path(donor.directory)
    output = directory / "precision.json"
    require_paths([precision_root(name) / "pre", donor.scores, directory / "untrained.json"])
    settings = {key: PRECISION_CHECK[key] for key in ("rows", "weights_dtype", "compute_dtype", "attn_implementation")}
    run(
        [
            "python", "data_curation/check_score_precision.py",
            "--base", BASE_CACHE, "--main", donor.scores, "--check", str(precision_root(name) / "pre"),
            "--name", name, "--untrained", str(directory / "untrained.json"),
            "--settings", json.dumps(settings, sort_keys=True), "--output", str(output),
        ]
    )
    data_volume.commit()
    return json.loads(output.read_text(encoding="utf-8"))


def ensure_turn_positions() -> Path:
    """The turn-position report and weight files, regenerated unless their recorded provenance is current."""
    from data_curation.turn_positions import provenance

    root = Path(TURN_POSITION_ROOT)
    report = root / "positions.json"
    expected = provenance(Path(PROMPT_DATA), Path(CANONICAL_DATA))
    current = json.loads(report.read_text(encoding="utf-8")).get("provenance") if report.exists() else None
    if current != expected:
        run(
            [
                "python", "data_curation/turn_positions.py",
                "--prompts", PROMPT_DATA, "--canonical", CANONICAL_DATA, "--output-dir", str(root),
            ]
        )
        data_volume.commit()
    return root


def prompt_weight_path(name: str) -> Path:
    """The current weight file of a ``prompt_weights`` rule, regenerated first if its inputs changed."""
    from data_curation.turn_positions import RULES

    if name in RULES:
        return ensure_turn_positions() / f"{name}.json"
    raise KeyError(f"no source for prompt weights {name!r}")


def prompt_weight_arguments(names) -> list[str]:
    """``--prompt-weights NAME=PATH`` for each weight rule a spec uses (files made current first)."""
    paths = {name: prompt_weight_path(name) for name in sorted(set(names))}
    require_paths(paths.values())
    return [argument for name, path in paths.items() for argument in ("--prompt-weights", f"{name}={path}")]


def prompt_metadata() -> str:
    """{prompt_id: {conversation, target}} from the canonical LoopTool rows, for analysis slices."""
    from data_curation.looptool import cached_prompt_rows

    output = Path(ANALYSIS_ROOT) / "prompt_metadata.json"
    rows = cached_prompt_rows(PROMPT_DATA, CANONICAL_DATA)
    metadata = {
        p: {"conversation": row["metadata"]["conversation_kind"], "target": row["metadata"]["target_kind"]}
        for p, row in rows.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata) + "\n")
    return str(output)


@app.function(image=cpu_image, cpu=16, memory=131_072, timeout=43_200, volumes=DATA_AND_MODELS)
def analyze(tag: str, names: list[str], max_rows: int = 0) -> str:
    data_volume.reload()
    model_volume.reload()
    donors = verified_donors(names)
    spec_path = Path(ANALYSIS_ROOT) / f"{tag}.spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(geometry_spec(list(donors)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = Path(ANALYSIS_ROOT) / f"{tag}.json"
    command = [
        "python", "data_curation/shift_geometry.py",
        "--base", BASE_CACHE, "--spec", str(spec_path), "--tokenizer-json", STUDENT_TOKENIZER_JSON,
        "--untrained", merged_untrained([AGENT_ACC.name, *donors]), "--prompt-metadata", prompt_metadata(),
        "--output", str(output), *donor_arguments(donors),
    ]
    if max_rows:
        command += ["--max-rows", str(max_rows)]
    run(command)
    data_volume.commit()
    return str(output)


@app.function(image=cpu_image, cpu=8, memory=65_536, timeout=21_600, volumes=DATA_AND_MODELS)
def probe_select(tag: str) -> str:
    data_volume.reload()
    model_volume.reload()
    states = Path(PROBE_ROOT) / tag / "states.parquet"
    if not states.exists():
        states.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "python", "data_curation/mc_advantage_probes.py", "select",
                "--base", BASE_CACHE, "--tokenizer-json", STUDENT_TOKENIZER_JSON, "--output", str(states),
            ]
        )
        data_volume.commit()
    return str(states)


@app.function(image=rollout_image, gpu="H100", cpu=8, memory=65_536, timeout=43_200, volumes=DATA_AND_MODELS)
def probe_rollout_shard(tag: str, rank: int, world_size: int) -> str:
    data_volume.reload()
    model_volume.reload()
    root = Path(PROBE_ROOT) / tag
    output = root / f"records-r{rank:03d}-of-{world_size:03d}.parquet"
    if output.exists():
        return f"rank {rank}: existing"
    run(
        [
            "python", "data_curation/mc_advantage_probes.py", "rollout",
            "--states", str(root / "states.parquet"), "--prompts", PROMPT_DATA, "--canonical", CANONICAL_DATA,
            "--model", agent_eff.Model(STUDENT_MODEL, STUDENT_REVISION).path, "--output", str(output),
            "--max-response-tokens", str(RECIPE.max_response_tokens),
            "--max-model-len", str(RECIPE.sequence_tokens + 8),
            "--rank", str(rank), "--world-size", str(world_size),
        ]
    )
    data_volume.commit()
    return f"rank {rank}: complete"


@app.function(image=cpu_image, cpu=8, memory=65_536, timeout=21_600, volumes=DATA_AND_MODELS)
def probe_evaluate(tag: str, names: list[str], world_size: int) -> str:
    data_volume.reload()
    model_volume.reload()
    donors = verified_donors(names)
    root = Path(PROBE_ROOT) / tag
    records = [root / f"records-r{rank:03d}-of-{world_size:03d}.parquet" for rank in range(world_size)]
    require_paths(records)
    spec_path = root / "spec.json"
    spec_path.write_text(json.dumps(geometry_spec(list(donors)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run(
        [
            "python", "data_curation/mc_advantage_probes.py", "evaluate",
            "--states", str(root / "states.parquet"), "--records", *map(str, records),
            "--base", BASE_CACHE, "--spec", str(spec_path),
            "--untrained", merged_untrained([AGENT_ACC.name, *donors]),
            "--output", str(root / "report.json"), *donor_arguments(donors),
        ]
    )
    data_volume.commit()
    return str(root / "report.json")


@app.function(image=cpu_image, cpu=2, memory=4_096, timeout=86_400)
def probe_chain(tag: str, names: list[str], workers: int) -> str:
    """Server-side select -> rollouts -> evaluate; finished rollout shards are reused."""
    probe_select.remote(tag)
    for result in probe_rollout_shard.starmap([(tag, rank, workers) for rank in range(workers)], order_outputs=False):
        print(result, flush=True)
    return probe_evaluate.remote(tag, names, workers)


@app.function(image=cpu_image, cpu=8, memory=65_536, timeout=21_600, volumes=DATA_AND_MODELS)
def compose(variant: str) -> dict:
    """Seal one variant's target, or return the existing one if it was composed from the same spec."""
    data_volume.reload()
    model_volume.reload()
    spec = variant_specs()[variant]
    output = Path(SYNTHETIC_ROOT) / variant
    weights = [item["gate"]["prompt_weights"] for item in spec["terms"] if "prompt_weights" in (item.get("gate") or {})]
    if (output / "manifest.json").exists():
        composed = json.loads((output / "manifest.json").read_text(encoding="utf-8"))["post_teacher_model"]
        if composed["spec"] != spec:
            raise RuntimeError(f"{output} was composed from a different spec; remove it to recompose")
        if weights:
            from data_curation.common import file_sha256

            recorded = {name: item["sha256"] for name, item in composed.get("prompt_weights", {}).items()}
            current = {name: file_sha256(prompt_weight_path(name)) for name in weights}
            if recorded != current:
                raise RuntimeError(f"{output} was composed from other prompt weights; remove it to recompose")
        return {"path": str(output), "calibration": composed["calibration"]}
    donors = verified_donors(variant_sources(spec))
    spec_path = Path(SYNTHETIC_ROOT) / f"{variant}.spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run(
        [
            "python", "data_curation/build_synthetic_shift_target.py",
            "--base", BASE_CACHE, "--spec", str(spec_path), "--tokenizer-json", STUDENT_TOKENIZER_JSON,
            "--untrained", merged_untrained([AGENT_ACC.name, *donors]),
            "--output-dir", str(output), *donor_arguments(donors),
            *(prompt_weight_arguments(weights) if weights else []),
        ]
    )
    data_volume.commit()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    return {"path": str(output), "calibration": manifest["post_teacher_model"]["calibration"]}


@app.function(image=cpu_image, cpu=16, memory=131_072, timeout=21_600, volumes=DATA_AND_MODELS)
def review(variants: list[str], tag: str) -> str:
    """Geometry of the composed targets themselves: how each arm moves the student's forks."""
    data_volume.reload()
    model_volume.reload()
    targets = {variant: str(Path(SYNTHETIC_ROOT) / variant) for variant in variants}
    require_paths([Path(path) / "manifest.json" for path in targets.values()])
    spec = {"directions": {v: {"terms": [{"source": v, "coef": 1.0}], "keep_untrained": True} for v in variants}}
    spec_path = Path(ANALYSIS_ROOT) / f"{tag}.spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = Path(ANALYSIS_ROOT) / f"{tag}.json"
    run(
        [
            "python", "data_curation/shift_geometry.py",
            "--base", BASE_CACHE, "--spec", str(spec_path), "--tokenizer-json", STUDENT_TOKENIZER_JSON,
            "--prompt-metadata", prompt_metadata(), "--output", str(output),
            *donor_arguments(targets),
        ]
    )
    data_volume.commit()
    return str(output)


@app.function(
    image=runtime_image,
    gpu="H100:8",
    cpu=32,
    memory=131_072,
    timeout=86_400,
    ephemeral_disk=1_048_576,
    volumes=ALL_VOLUMES,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(variant: str, seed: int = RECIPE.training_seed) -> str:
    """The LoopTool recipe on one synthetic target; only the target, alpha, passes, seed, and save cadence differ.

    ``training_plan`` gives the variant's alpha (its spec's, 2.0 for every arm) and number of passes over the
    target: one, except the paper arms (two, on a repeated copy).
    The recipe seed reads the cache in its stored order, as V0 and the primary
    matrix did; any other seed also permutes the rows with that seed, so a
    replicate varies the data order and not only GPU nondeterminism (on a
    repeated copy the permutation spans both copies: each row is still read
    twice, but not once in each half of the run). Only the
    final checkpoint is saved (a run takes about 20 minutes per pass), and the run counts
    as complete only once that exact iteration exists. A completed run is
    returned as is; an interrupted one starts over.
    """
    data_volume.reload()
    model_volume.reload()
    checkpoint_volume.reload()
    student = agent_eff.Model(STUDENT_MODEL, STUDENT_REVISION).path
    plan = agent_eff.training_plan(variant)
    data = Path(plan["data"])
    require_paths([student, data / "manifest.json", INITIAL_MEGATRON])
    root = training_root(variant, seed)
    save = root / "train"
    manifest = json.loads((data / "manifest.json").read_text())
    if (root / "COMPLETE.json").exists():
        # Reuse only a run of this exact plan: provenance written before alpha and passes were recorded is a
        # one-pass run at the recipe's alpha.
        done = json.loads((root / "COMPLETE.json").read_text())
        recorded = (done["target_revision"], done.get("alpha", RECIPE.alpha), done.get("passes", 1))
        expected = (manifest["post_teacher_model"]["revision"], plan["alpha"], plan["passes"])
        if recorded != expected:
            raise RuntimeError(
                f"{root} holds a run of {recorded}, not the current plan {expected}; remove it to retrain"
            )
        return str(save)
    if save.exists():
        # An interrupted attempt, e.g. a preempted container restarted with the same input. Only the final
        # iteration is ever saved, so there is nothing to resume.
        print(f"removing the incomplete attempt at {save}", flush=True)
        shutil.rmtree(save)
        checkpoint_volume.commit()
    target_revision = manifest["post_teacher_model"]["revision"]
    if manifest["post_teacher_model"]["spec"]["alpha"] != plan["alpha"] or manifest["total_rows"] != plan["rows"]:
        raise RuntimeError(f"{data} does not match the training plan {plan}")
    shuffle = seed != RECIPE.training_seed
    order = ["--rollout-seed", str(seed), "--rollout-shuffle"] if shuffle else ["--rollout-seed", str(RECIPE.data_seed)]
    run(
        [
            "python", "configs/lightning_weave/train.py",
            "--model-type", "qwen3-4B", "--student", student,
            "--data", str(data), "--manifest", str(data / "manifest.json"),
            "--load", INITIAL_MEGATRON, "--save", str(save),
            "--num-gpus", str(RECIPE.training_gpus), "--alpha", str(plan["alpha"]), "--lr", str(RECIPE.learning_rate),
            "--rollout-batch-size", str(RECIPE.rollout_batch_size),
            "--global-batch-size", str(RECIPE.global_batch_size),
            "--num-rollout", str(plan["num_rollout"]), "--save-interval", str(plan["num_rollout"]),
            "--max-tokens-per-gpu", str(RECIPE.sequence_tokens), "--top-k", str(RECIPE.top_k),
            "--responses-per-prompt", str(RECIPE.responses_per_prompt),
            "--max-prompt-length", str(RECIPE.max_prompt_tokens),
            "--max-response-length", str(RECIPE.max_response_tokens),
            "--temperature", str(RECIPE.temperature), "--top-p", str(RECIPE.top_p),
            "--seed", str(seed), *order, "--loss-mode", "tilted_target",
            "--wandb-project", "agent-eff-synthesis", "--wandb-group", variant,
        ]
    )
    latest = (save / "latest_checkpointed_iteration.txt").read_text().strip()
    final_iteration = plan["final_iteration"]
    final = save / f"iter_{final_iteration:07d}"
    if latest != str(final_iteration) or not final.is_dir():
        raise RuntimeError(f"training ended at iteration {latest}, expected {final_iteration}")
    provenance = {
        "variant": variant,
        "seed": seed,
        "data_order": f"shuffled with seed {seed}" if shuffle else "stored",
        "target": str(data),
        "target_revision": target_revision,
        "final_iteration": final_iteration,
        "alpha": plan["alpha"],
        "passes": plan["passes"],
        "num_rollout": plan["num_rollout"],
        "student": {"repo": STUDENT_MODEL, "revision": STUDENT_REVISION},
    }
    (root / "COMPLETE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    checkpoint_volume.commit()
    return str(save)


@app.function(image=runtime_image, cpu=16, memory=65_536, timeout=14_400, volumes=ALL_VOLUMES)
def export(variant: str, seed: int = RECIPE.training_seed) -> str:
    """Export the completed final iteration and record its provenance next to the weights."""
    model_volume.reload()
    checkpoint_volume.reload()
    root = training_root(variant, seed)
    require_paths([root / "COMPLETE.json"])
    provenance = json.loads((root / "COMPLETE.json").read_text())
    output = root / "hf"
    if (output / PROVENANCE_FILE).exists():  # written last, after the weights
        return str(output)
    if output.exists():
        print(f"removing the incomplete export at {output}", flush=True)
        shutil.rmtree(output)
    environment = os.environ.copy()
    environment.update(
        {
            "STUDENT_MODEL": agent_eff.Model(STUDENT_MODEL, STUDENT_REVISION).path,
            "INPUT_DIR": str(root / "train" / f"iter_{provenance['final_iteration']:07d}"),
            "OUTPUT_DIR": str(output),
            "MODEL_TYPE": "qwen3-4B",
            "MEGATRON_PATH": "/root/Megatron-LM",
        }
    )
    run(["bash", "scripts/convert_checkpoint.sh", "to-hf"], env=environment)
    (output / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    checkpoint_volume.commit()
    return str(output)


@app.function(image=cpu_image, cpu=4, memory=16_384, timeout=7_200, volumes=DATA_AND_MODELS)
def repeat_target(variant: str) -> str:
    """The sealed target repeated for a multi-pass plan, as byte-identical shard copies; reused if current."""
    data_volume.reload()
    plan = agent_eff.training_plan(variant)
    source, output = Path(plan["target"]), Path(plan["data"])
    require_paths([source / "manifest.json"])
    if plan["passes"] == 1:
        return str(source)
    revision = json.loads((source / "manifest.json").read_text(encoding="utf-8"))["post_teacher_model"]["revision"]
    if (output / "manifest.json").exists():
        existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if existing["post_teacher_model"]["revision"] != revision or existing["total_rows"] != plan["rows"]:
            raise RuntimeError(f"{output} repeats another target or length; remove it to rebuild")
        return str(output)
    run(
        [
            "python", "data_curation/repeat_sealed_direct_opd.py", "--manifest", str(source / "manifest.json"),
            "--output-dir", str(output), "--total-rows", str(plan["rows"]), "--copy",
        ]
    )
    data_volume.commit()
    return str(output)


@app.function(image=cpu_image, cpu=1, memory=2_048, timeout=86_400, nonpreemptible=True)
def build_student(variant: str, seed: int) -> str:
    """compose -> (repeat, for a multi-pass plan) -> train -> export for one variant; finished stages are reused."""
    print(f"{variant}: composed {json.dumps(compose.remote(variant)['calibration'], sort_keys=True)}", flush=True)
    if agent_eff.training_passes(variant) > 1:
        print(f"{variant}: repeated for training at {repeat_target.remote(variant)}", flush=True)
    print(f"{variant}: trained {train.remote(variant, seed)}", flush=True)
    return export.remote(variant, seed)


@app.function(image=cpu_image, cpu=1, memory=2_048, timeout=86_400)
def build_chain(variants: list[str], seeds: list[int]) -> dict:
    """Compose (and, for a multi-pass plan, repeat) each distinct target once, then build every (variant, seed).

    Preparing targets first means seeds of one new variant never race to seal the same target or its repeated
    copy. One failure does not stop the others; each outcome is reported.
    """
    distinct = sorted(set(variants))
    composed = dict(zip(distinct, compose.map(distinct, return_exceptions=True)))
    failed = {
        variant: f"compose failed: {result}"
        for variant, result in composed.items()
        if isinstance(result, BaseException)
    }
    repeated = [variant for variant in distinct if variant not in failed and agent_eff.training_passes(variant) > 1]
    for variant, result in zip(repeated, repeat_target.map(repeated, return_exceptions=True), strict=True):
        if isinstance(result, BaseException):
            failed[variant] = f"repeat failed: {result}"
    pairs = [(variant, seed) for variant in distinct if variant not in failed for seed in seeds]
    report = dict(failed)
    results = build_student.starmap(pairs, return_exceptions=True)
    report.update({f"{variant} seed{seed}": str(result) for (variant, seed), result in zip(pairs, results)})
    return report


def score_pairs(names: list[str], workers: int) -> tuple[dict, dict]:
    """post -> pre -> verify for each pair; a failure stops only its own pair.

    Returns ({pair: evidence}, {pair: error}).
    """
    failed: dict[str, str] = {}
    for stage in ("post", "pre"):
        items = [(name, stage, rank, workers) for name in names if name not in failed for rank in range(workers)]
        results = score_donor_shard.starmap(items, return_exceptions=True)
        for (name, _, rank, _), result in zip(items, results, strict=True):
            if isinstance(result, BaseException) and name not in failed:
                failed[name] = f"{stage} rank {rank}: {type(result).__name__}: {result}"
    alive = [name for name in names if name not in failed]
    evidence = {}
    for name, result in zip(alive, verify.map(alive, return_exceptions=True), strict=True):
        if isinstance(result, BaseException):
            failed[name] = f"verify: {type(result).__name__}: {result}"
        else:
            evidence[name] = result
    return evidence, failed


@app.local_entrypoint()
def main(
    action: str = "plan",
    donors: str = "",
    variant: str = "",
    tag: str = "",
    workers: int = 8,
    max_rows: int = 0,
    seed: int = RECIPE.training_seed,
    seeds: str = "",
) -> None:
    names = [name for name in donors.split(",") if name] or [name for name, item in DONORS.items() if item.core]
    scored = [name for name in names if name != AGENT_ACC.name]
    if action == "plan":
        print(json.dumps(agent_eff.resolved(), indent=2, sort_keys=True))
    elif action == "download":
        print(json.dumps(download.remote(names), indent=2))
    elif action == "prepare":
        print(json.dumps(dict(zip(names, prepare.map(names))), indent=2, ensure_ascii=False))
    elif action == "score":
        call = score_chain.spawn(scored, workers)
        print(f"score chain running server-side: {call.object_id}")
    elif action == "onboard":
        call = onboard_chain.spawn(scored, workers, tag or "census")
        print(f"onboard chain running server-side: {call.object_id}")
    elif action == "precision":
        print(json.dumps(dict(zip(scored, precision_score.map(scored))), indent=2))
    elif action == "verify":
        print(json.dumps(dict(zip(scored, verify.map(scored))), indent=2))
    elif action == "analyze":
        print(analyze.remote(tag or "-".join(scored), scored, max_rows))
    elif action == "probe":
        call = probe_chain.spawn(tag or "core", scored, workers)
        print(f"probe chain running server-side: {call.object_id}")
    elif action == "compose":
        variants = [name for name in variant.split(",") if name]
        print(json.dumps(dict(zip(variants, compose.map(variants))), indent=2))
    elif action == "review":
        variants = [name for name in variant.split(",") if name]
        print(review.remote(variants, tag or "targets"))
    elif action == "train":
        for name in (item for item in variant.split(",") if item):
            print(f"{name}: training running server-side as {train.spawn(name, seed).object_id}", flush=True)
    elif action == "export":
        variants = [name for name in variant.split(",") if name]
        for result in export.starmap([(name, seed) for name in variants], order_outputs=False):
            print(result, flush=True)
    elif action == "build":
        variants = [name for name in variant.split(",") if name]
        unknown = sorted(set(variants) - set(variant_specs()))
        if unknown:
            raise ValueError(f"unknown variants {unknown}")
        seed_list = [int(item) for item in seeds.split(",") if item] or [seed]
        call = build_chain.spawn(variants, seed_list)
        print(f"build chain for seeds {seed_list} running server-side: {call.object_id}")
    else:
        raise ValueError(f"unknown action {action!r}")
