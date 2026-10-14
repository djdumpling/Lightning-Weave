# /// script
# requires-python = "==3.12.*"
# dependencies = ["modal==1.5.5"]
# ///
"""Qwen3.5-only stage smokes and transfer pipeline; dense Qwen3 paths stay pinned.

All stages require MODAL_ENVIRONMENT=alex-dev-2. Full stages require the smoke
report gate written only after collection, rollouts, scoring and training pass.
"""
from __future__ import annotations
import collections
import json
import os
import subprocess
import sys
from pathlib import Path
import modal

ENTRYPOINT = Path(__file__).resolve()
PROJECT = ENTRYPOINT.parents[2] if len(ENTRYPOINT.parents) > 2 else Path("/workspace/Lightning-Weave")
if not (PROJECT / "configs/qwen35_transfer/config.py").exists():
    PROJECT = Path("/workspace/Lightning-Weave")
sys.path.insert(0, str(PROJECT))
from configs.qwen35_transfer import config as c

if modal.is_local() and os.environ.get("MODAL_ENVIRONMENT") != "alex-dev-2":
    raise RuntimeError("MODAL_ENVIRONMENT=alex-dev-2 is required")
app = modal.App("lightning-weave-qwen35-transfer-20261012")
models = modal.Volume.from_name("lightning-weave-hf-models")
data = modal.Volume.from_name("lightning-weave-looptool-opd-data")
checkpoints = modal.Volume.from_name("lightning-weave-checkpoints")
tau = modal.Volume.from_name("lightning-weave-tau-eval")
MOUNTS = {"/models": models, "/opd": data, "/checkpoints": checkpoints, "/tau": tau.read_only()}
ENV = {"HF_HOME": "/models", "PYTHONPATH": "/workspace/Lightning-Weave:/root/Megatron-LM", "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_DEBUG": "WARN", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
# Add Python dependencies over the pinned compiled runtime. Never resolve torch,
# CUDA, Transformer Engine, flash-attn, or Megatron to new versions here.
overlay = (modal.Image.from_registry(c.RUNTIME_IMAGE).entrypoint([])
    .run_commands("python -m pip install --no-deps transformers==5.5.3 fla-core==0.4.2 flash-linear-attention==0.4.2",
                  "python -m pip install 'huggingface-hub==1.5.0' 'tokenizers==0.22.2' 'regex==2025.11.3' typer")
    .env(ENV).add_local_dir(str(PROJECT), remote_path="/workspace/Lightning-Weave", copy=True, ignore=[".git", ".venv", "outputs", "results", "scratchpad", "**/__pycache__", "**/*.pyc"]))
cpu = (modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "transformers==5.5.3", "pyarrow==20.0.0", "numpy==2.2.6", "accelerate==1.10.1", extra_index_url="https://download.pytorch.org/whl/cpu")
    .env(ENV).add_local_dir(str(PROJECT), remote_path="/workspace/Lightning-Weave", copy=True, ignore=[".git", ".venv", "outputs", "results", "scratchpad", "**/__pycache__", "**/*.pyc"]))
rollout = (modal.Image.from_registry(c.ROLLOUT_IMAGE).entrypoint([])
    .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python")
    .uv_pip_install("pyarrow==20.0.0")
    .env(ENV).add_local_dir(str(PROJECT), remote_path="/workspace/Lightning-Weave", copy=True, ignore=[".git", ".venv", "outputs", "results", "scratchpad", "**/__pycache__", "**/*.pyc"]))


def run(command, **kw):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd="/workspace/Lightning-Weave", check=True, **kw)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def reload():
    for volume in (models, data, checkpoints, tau):
        volume.reload()


def require_smokes(smoke):
    if not smoke:
        report = Path(c.DATA_ROOT) / "SMOKES_REPORTED.json"
        if not report.exists():
            raise RuntimeError("Full computation requires the completed, reported smoke gate")


@app.function(image=cpu, cpu=16, memory=131072, timeout=21600, volumes=MOUNTS)
def assets():
    """Extract immutable text-only student/anchors and verify token-ID equality."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoConfig
    from data_curation.common import canonical_hash
    reload()
    records, vocabularies = {}, {}
    for role, (repo, revision) in c.MODELS.items():
        source = snapshot_download(repo_id=repo, revision=revision, cache_dir="/models")
        if Path(source).name != revision:
            raise RuntimeError("unpinned model snapshot")
        target = Path(c.language_model(role))
        if not (target / "extraction-manifest.json").exists():
            run(["python", "tools/extract_qwen35_language_model.py", "--input-dir", source, "--output-dir", target, "--source-revision", revision])
        cfg = AutoConfig.from_pretrained(target)
        vocab = AutoTokenizer.from_pretrained(target).get_vocab()
        if cfg.model_type != "qwen3_5_text" or cfg.vocab_size != 248320:
            raise RuntimeError(f"wrong text-model config: {cfg.model_type}, {cfg.vocab_size}")
        vocabularies[role] = vocab
        records[role] = {"repo": repo, "revision": revision, "path": str(target), "model_vocab_size": cfg.vocab_size, "tokenizer_entries": len(vocab), "max_token_id": max(vocab.values()), "tokenizer_hash": canonical_hash(vocab)}
    if any(v != vocabularies["student"] for v in vocabularies.values()):
        raise RuntimeError("Qwen3.5 tokenizer mappings differ")
    if Path(c.DATA_ROOT + "/assets.json").exists():
        return json.loads(Path(c.DATA_ROOT + "/asset_audit.json").read_text())
    run(["python", "data_curation/prepare_direct_opd_assets.py", "--student", c.language_model("student"), "--student-revision", c.MODELS["student"][1], "--pre-teacher", c.language_model("pre_teacher"), "--pre-teacher-revision", c.MODELS["pre_teacher"][1], "--post-teacher", c.language_model("post_teacher"), "--post-teacher-revision", c.MODELS["post_teacher"][1], "--output", c.DATA_ROOT + "/assets.json"])
    write(c.DATA_ROOT + "/asset_audit.json", records)
    models.commit(); data.commit()
    return records


@app.function(image=overlay, gpu="H100", cpu=4, memory=16384, timeout=1800)
def runtime_probe():
    import importlib.metadata as md
    import torch, transformers, megatron.core, fla
    import mbridge, transformer_engine
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    versions = {name: md.version(name) for name in ("torch", "transformers", "megatron-core", "fla-core", "flash-linear-attention", "transformer-engine")}
    if versions["megatron-core"].split(".")[:2] != ["0", "14"]:
        raise RuntimeError(f"MCore 0.14 required: {versions}")
    q = torch.randn(1, 256, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True); v = torch.randn_like(q, requires_grad=True)
    g = torch.full((1, 256, 2), -.1, device="cuda", requires_grad=True)
    beta = torch.full_like(g, .5, requires_grad=True)
    output, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    output.float().square().mean().backward()
    if not torch.isfinite(output).all() or not all(torch.isfinite(x.grad).all() for x in (q, k, v, g, beta)):
        raise RuntimeError("FLA forward/backward produced non-finite values")
    versions["fla_forward_backward"] = "finite"
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    if hf.chunk_gated_delta_rule is None or hf.fused_recurrent_gated_delta_rule is None:
        raise RuntimeError("Transformers Qwen3.5 is not bound to the FLA kernels")
    versions["transformers_qwen35_fla_bound"] = True
    versions["causal_conv1d_extension"] = hf.causal_conv1d_fn is not None
    print(json.dumps(versions), flush=True)
    return versions


@app.function(image=cpu, cpu=8, memory=32768, timeout=3600, volumes=MOUNTS)
def collection_audit(smoke: bool = True):
    import numpy as np
    from transformers import AutoTokenizer
    from configs.tau_bench_eval.harness import unpack_tokens
    reload(); require_smokes(smoke)
    root = Path(c.data_root(smoke)); root.mkdir(parents=True, exist_ok=True)
    source = Path("/tau") / c.collection_run(smoke) / "qwen35" / "tau2_retail"
    records = [json.loads(p.read_text()) for p in sorted(source.glob("*.json")) if p.name != "failures.json"]
    expected = 4 if smoke else 563
    if len(records) != expected:
        raise RuntimeError(f"collection records {len(records)} != {expected}")
    tokenizer = AutoTokenizer.from_pretrained(c.snapshot("student"))
    count = collections.Counter(); lengths = []
    for record in records:
        count["episodes"] += 1
        if record.get("termination") in ("max_steps", "max_tokens", "too_many_errors"):
            count["episode_limit"] += 1
        for r in record["requests"]:
            count["requests"] += 1
            if r.get("capture_error"):
                raise RuntimeError(r["capture_error"])
            actual = unpack_tokens(r["actual_prompt_token_ids"])
            rendered = unpack_tokens(r["server_render"]["tokens"])
            if actual != rendered:
                raise RuntimeError("served IDs differ from independent /tokenize rendering")
            prompt = tokenizer.decode(actual, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if tokenizer.encode(prompt, add_special_tokens=False) != actual:
                raise RuntimeError("served prompt IDs fail tokenizer round trip")
            lengths.append(len(actual))
            response = r["response"]
            count["length_finished"] += response["finish_reason"] == "length"
            count["empty_turns"] += not response.get("content") and not response.get("tool_calls")
            messages = r["messages"]
            last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
            for i, m in enumerate(messages):
                reasoning = m.get("reasoning") or m.get("reasoning_content")
                if m["role"] == "assistant" and reasoning:
                    if i > last_user:
                        count["current_step_reasoning"] += 1
                        if reasoning not in prompt:
                            raise RuntimeError("current-step reasoning omitted from served IDs")
                    elif reasoning not in [x.get("reasoning") or x.get("reasoning_content") for x in messages[last_user + 1:]]:
                        count["older_step_reasoning"] += 1
                        if reasoning in prompt:
                            raise RuntimeError("older-step reasoning retained in served IDs")
            sampling = r["sampling"]; extra = sampling["extra_body"]
            observed = (sampling["temperature"], sampling["top_p"], sampling["presence_penalty"], extra["top_k"], extra["min_p"])
            if observed != (1.0, .95, 1.5, 20, 0.0):
                raise RuntimeError(f"wrong collection sampling: {observed}")
    report = {**count, "prompt_lengths": {"p50": float(np.percentile(lengths, 50)), "p95": float(np.percentile(lengths, 95)), "max": max(lengths)}, "collection_run": c.collection_run(smoke)}
    write(root / "collection_audit.json", report); data.commit()
    if smoke and (count["empty_turns"] or count["length_finished"] or count["episode_limit"] or not count["current_step_reasoning"]):
        raise RuntimeError(f"collection smoke failed: {report}")
    run(["python", "-m", "data_curation.fresh_states", "--collection", source.parent.parent, "--collectors", "qwen35", "--domains", "tau2_retail", "--tokenizer", c.snapshot("student"), "--max-prompt-tokens", str(c.MAX_PROMPT), "--states-per-episode", "8", "--generation-prompt", "<|im_start|>assistant\n<think>\n", "--num-prompts", "8" if smoke else str(c.NUM_PROMPTS), "--output", root / "prompts.parquet", "--summary", root / "prompts_summary.json"])
    data.commit()
    return report


@app.function(image=rollout, gpu="H100", cpu=8, memory=65536, timeout=43200, volumes=MOUNTS)
def rollouts(smoke: bool = True, rank: int = 0, workers: int = 1):
    reload(); require_smokes(smoke)
    root = Path(c.data_root(smoke))
    output = root / "rollouts"; marker = output / f"rank{rank}.COMPLETE.json"
    if marker.exists():
        done = json.loads(marker.read_text())
        if done["workers"] != workers or done["student_revision"] != c.MODELS["student"][1]:
            raise RuntimeError("rollout partition differs from completed cache")
        return str(output)
    existing = list(output.glob(f"rollouts-r{rank:05d}-*.parquet"))
    if existing:
        from datetime import datetime, timezone
        archive = root / (f"partial-rollouts-rank{rank}-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        archive.mkdir()
        for shard in existing: shard.rename(archive / shard.name)
    run(["python", "data_curation/collect_direct_opd_rollouts.py", "--model", c.snapshot("student"), "--model-revision", c.MODELS["student"][1], "--asset-lock", c.DATA_ROOT + "/assets.json", "--input", root / "prompts.parquet", "--output-dir", root / "rollouts", "--label-key", "label", "--prompt-id-key", "prompt_id", "--max-prompts", "8" if smoke else str(c.NUM_PROMPTS), "--responses-per-prompt", str(c.K), "--max-prompt-length", str(c.MAX_PROMPT), "--max-response-length", str(c.MAX_RESPONSE), "--top-k", "16", "--temperature", "1.0", "--top-p", "1.0", "--seed", "42", "--batch-size", "8", "--dtype", "bfloat16", "--gpu-memory-utilization", "0.90", "--compilation-mode", "0", "--cudagraph-mode", "NONE", "--max-num-seqs", "32", "--shard-size", "128", "--rank", str(rank), "--world-size", str(workers), "--language-model-only", "--gdn-prefill-backend", "triton", "--neutral-sampling"])
    import pyarrow.parquet as pq
    count = sum(pq.read_metadata(p).num_rows for p in output.glob(f"rollouts-r{rank:05d}-*.parquet"))
    prompts = 8 if smoke else c.NUM_PROMPTS
    per_rank = (prompts + workers - 1) // workers
    expected = max(0, min(per_rank, prompts - rank * per_rank)) * c.K
    if count != expected: raise RuntimeError(f"rollout rank {rank}: {count} rows != {expected}")
    write(marker, {"workers": workers, "student_revision": c.MODELS["student"][1], "rows": count})
    data.commit()
    return str(output)


@app.function(image=cpu, cpu=4, memory=16384, timeout=3600, volumes=MOUNTS)
def rollout_audit(smoke: bool = True):
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    reload(); require_smokes(smoke)
    root = Path(c.data_root(smoke)); tokenizer = AutoTokenizer.from_pretrained(c.snapshot("student"))
    rows = [r for p in sorted((root / "rollouts").glob("*.parquet")) for r in pq.read_table(p).to_pylist()]
    lengths, reasons, missing = [], collections.Counter(), 0
    for row in rows:
        m = row["metadata"]; lengths.append(m["response_length"]); reasons[m["finish_reason"]] += 1
        text = tokenizer.decode(m["response_tokens"], skip_special_tokens=False)
        missing += "</think>" not in text
        if m["generation_config"]["vllm_version"] != "0.28.0":
            raise RuntimeError("Qwen3.5 rollout engine must be vLLM 0.28.0")
    expected = (8 if smoke else c.NUM_PROMPTS) * c.K
    report = {"rows": len(rows), "finish_reasons": dict(reasons), "missing_closed_think": missing, "lengths": {"min": min(lengths), "p50": float(np.percentile(lengths, 50)), "p95": float(np.percentile(lengths, 95)), "max": max(lengths)}}
    write(root / "rollout_audit.json", report); data.commit()
    if len(rows) != expected or (smoke and (missing or reasons["length"] or not min(lengths))):
        raise RuntimeError(f"rollout smoke failed: {report}")
    return report


@app.function(image=overlay, gpu="H100", cpu=8, memory=65536, timeout=43200, volumes=MOUNTS)
def score(stage: str, smoke: bool = True, rank: int = 0, workers: int = 1):
    reload(); require_smokes(smoke)
    stages = {"post": ("post_teacher", "post_teacher_log_probs", "rollouts", "post"), "pre": ("pre_teacher", "pre_teacher_log_probs", "post", "pre"), "reference": ("student", "student_ref_sampled_log_probs", "pre", "final")}
    role, field, source, target = stages[stage]; root = Path(c.data_root(smoke))
    run(["python", "data_curation/precompute_direct_opd_scores.py", "--model", c.language_model(role), "--model-revision", c.MODELS[role][1], "--asset-lock", c.DATA_ROOT + "/assets.json", "--score-field", field, "--input", root / source, "--output-dir", root / target, "--dtype", "bfloat16", "--device", "cuda:0", "--row-batch-size", "1", "--chunk-size", "256", "--attn-implementation", "flash_attention_2", "--rank", str(rank), "--world-size", str(workers)])
    data.commit()
    return str(root / target)


@app.function(image=cpu, cpu=8, memory=32768, timeout=3600, volumes=MOUNTS)
def scoring_audit(smoke: bool = True):
    import numpy as np
    import pyarrow.parquet as pq
    from data_curation.shift_geometry import target_kl
    reload(); require_smokes(smoke)
    root = Path(c.data_root(smoke)); errors, kls = [], []; row_count = 0
    for p in sorted((root / "final").glob("*.parquet")):
        for row in pq.read_table(p).to_pylist():
            row_count += 1
            m = row["metadata"]
            errors.extend(np.abs(np.array(m["behavior_sampled_log_probs"]) - np.array(m["student_ref_sampled_log_probs"])).tolist())
            probs = np.exp(np.array(m["behavior_topk_log_probs"], dtype=np.float64))
            probs = np.concatenate([probs, np.clip(1 - probs.sum(-1, keepdims=True), 0, 1)], -1)
            delta = np.array(m["post_teacher_log_probs"]) - np.array(m["pre_teacher_log_probs"])
            delta = np.concatenate([delta, np.zeros((len(delta), 1))], -1)
            kls.extend(target_kl(probs, delta, np.zeros_like(delta), c.ALPHA).tolist())
    if row_count != (8 if smoke else c.NUM_PROMPTS) * c.K:
        raise RuntimeError(f"incomplete scored cache: {row_count} rows")
    report = {"rows": row_count, "tokens": len(errors), "student_logprob_abs_error": {"mean": float(np.mean(errors)), "p99": float(np.percentile(errors, 99)), "max": float(max(errors))}, "target_kl_nats_per_token": float(np.mean(kls)), "alpha": c.ALPHA, "coefficient": 1, "2507_reference_kl": .0697, "scorer": "qwen35_full_context_decoder_chunked_lm_head_v1"}
    write(root / "scoring_audit.json", report); data.commit()
    if not np.isfinite(kls).all() or not 0 < np.mean(kls) < 1 or np.mean(errors) > .02 or np.percentile(errors, 99) > .15:
        raise RuntimeError(f"scoring smoke failed: {report}")
    run(["python", "data_curation/prepare_direct_opd_manifest.py", "--input", root / "final", "--source-dataset", root / "prompts.parquet", "--asset-lock", c.DATA_ROOT + "/assets.json", "--manifest-out", root / "final/manifest.json"])
    data.commit()
    return report


@app.function(image=overlay, gpu="H100:8", cpu=32, memory=196608, timeout=14400, volumes=MOUNTS)
def convert():
    reload()
    root = Path(c.CHECKPOINT_ROOT); initial = root / "initial-megatron"; exported = root / "step0-hf"
    env = {**os.environ, "STUDENT_MODEL": c.language_model("student"), "MODEL_TYPE": "qwen3.5-4B", "NUM_GPUS": "8", "MEGATRON_PATH": "/root/Megatron-LM", "OUTPUT_DIR": str(initial)}
    if not (initial / "release/.metadata").exists():
        run(["bash", "scripts/convert_checkpoint.sh", "to-megatron"], env=env)
    if not (exported / "model.safetensors.index.json").exists():
        run(["bash", "scripts/convert_checkpoint.sh", "to-hf"], env={**env, "INPUT_DIR": str(initial / "release"), "OUTPUT_DIR": str(exported)})
    # Stronger than a sampled-logit comparison: every exported language weight
    # must be identical to the original language-only checkpoint.
    from safetensors import safe_open
    def index(folder):
        return json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    source = Path(c.language_model("student")); a, b = index(source), index(exported)
    if a.keys() != b.keys():
        raise RuntimeError(f"round-trip weight keys differ: {a.keys() - b.keys()}, {b.keys() - a.keys()}")
    import torch
    for name in a:
        with safe_open(source / a[name], framework="pt") as f, safe_open(exported / b[name], framework="pt") as g:
            if not torch.equal(f.get_tensor(name), g.get_tensor(name)):
                raise RuntimeError(f"step-0 round-trip changed {name}")
    report = {"weight_tensors": len(a), "all_weights_identical": True, "initial": str(initial), "export": str(exported)}
    write(root / "step0_weight_audit.json", report); checkpoints.commit()
    return report


@app.function(image=rollout, gpu="H100", cpu=8, memory=65536, timeout=7200, volumes=MOUNTS)
def roundtrip():
    import numpy as np
    reload()
    root = Path(c.CHECKPOINT_ROOT); wrapper = root / "step0-vllm"
    if not wrapper.exists():
        run(["python", "tools/build_qwen35_vllm_language_wrapper.py", "--language-model", root / "step0-hf", "--outer-checkpoint", c.snapshot("student"), "--output", wrapper])
    for label, model in (("base", c.snapshot("student")), ("step0", wrapper)):
        run(["python", "data_curation/qwen35_roundtrip_probe.py", "--model", model, "--input", Path(c.data_root(True)) / "rollouts", "--output", root / f"{label}_vllm_probe.json"])
    a, b = [json.loads((root / f"{label}_vllm_probe.json").read_text()) for label in ("base", "step0")]
    if len(a) != len(b) or any(x["generated_tokens"] != y["generated_tokens"] for x, y in zip(a, b)):
        raise RuntimeError("step-0 serving changes generated tokens")
    errors = [abs(lp - rp) for x, y in zip(a, b) for field in ("generation_logprobs", "forced_logprobs") for lp, rp in zip(x[field], y[field])]
    if max(errors) > 1e-6:
        raise RuntimeError(f"step-0 serving/scoring differs: {max(errors)}")
    report = {"rows": len(a), "generated_tokens_identical": True, "score_tokens": len(errors), "max_logprob_abs_error": max(errors)}
    write(root / "step0_serving_audit.json", report); checkpoints.commit()
    return report


@app.function(image=cpu, cpu=8, memory=32768, timeout=3600, volumes=MOUNTS)
def training_fixture():
    """Three repeated fixed batches, then one maximum-length memory stress batch.

    The synthetic stress row is infrastructure evidence only and is never used
    by the scientific experiment. Only the original response prefix has loss.
    """
    import copy
    import pyarrow as pa
    import pyarrow.parquet as pq
    reload()
    root = Path(c.data_root(True)); destination = root / "training-fixture"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        from slime.rollout.offline_direct_opd import validate_sealed_manifest
        manifest = validate_sealed_manifest(destination / "manifest.json", expected_top_k=16)
        if manifest["total_rows"] != 256: raise RuntimeError("wrong smoke fixture size")
        return {"rows": 256, "fixed_batch_updates": 3, "stress_sequence_length": c.MAX_PROMPT + c.MAX_RESPONSE, "existing_sealed_fixture": True}
    original = [r for f in sorted((root / "final").glob("*.parquet")) for r in pq.read_table(f).to_pylist()]
    assert len(original) == 32
    rows = [copy.deepcopy(original[i % 32]) for i in range(256)]
    stress = rows[-1]["metadata"]
    def repeat(values, size): return (values * ((size + len(values) - 1) // len(values)))[:size]
    n = len(stress["response_tokens"])
    stress["prompt_tokens"] = repeat(stress["prompt_tokens"], c.MAX_PROMPT)
    for field in ("response_tokens", "candidate_ids", "behavior_topk_log_probs", "behavior_sampled_log_probs", "student_ref_sampled_log_probs", "post_teacher_log_probs", "pre_teacher_log_probs"):
        stress[field] = repeat(stress[field], c.MAX_RESPONSE)
    stress["response_length"] = c.MAX_RESPONSE
    stress["loss_mask"] = [True] * min(n, c.MAX_RESPONSE) + [False] * max(0, c.MAX_RESPONSE - n)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(c.snapshot("student"))
    rows[-1]["prompt"] = tokenizer.decode(stress["prompt_tokens"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    from data_curation.precompute_direct_opd_scores import cast_offline_direct_opd_storage
    pq.write_table(cast_offline_direct_opd_storage(pa.Table.from_pylist(rows)), destination / "smoke-training.parquet", compression="zstd")
    run(["python", "data_curation/prepare_direct_opd_manifest.py", "--input", destination, "--source-dataset", root / "prompts.parquet", "--asset-lock", c.DATA_ROOT + "/assets.json", "--manifest-out", destination / "manifest.json"])
    data.commit()
    return {"rows": 256, "fixed_batch_updates": 3, "stress_sequence_length": c.MAX_PROMPT + c.MAX_RESPONSE, "stress_loss_tokens": min(n, c.MAX_RESPONSE)}


@app.function(image=overlay, gpu="H200:4", cpu=32, memory=196608, timeout=86400, volumes=MOUNTS)
def train(smoke: bool = True, seed: int = 1234):
    import ast
    import math
    import re
    import threading
    reload(); require_smokes(smoke)
    if seed not in c.SEEDS: raise ValueError(seed)
    root = Path(c.CHECKPOINT_ROOT) / ("training-smoke" if smoke else f"seed{seed}")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "train.log").exists():
        from datetime import datetime, timezone
        archive = root / ("attempt-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        archive.mkdir()
        for name in ("train.log", "train", "attempt_memory.json"):
            if (root / name).exists(): (root / name).rename(archive / name)
    target = Path(c.data_root(smoke)) / ("training-fixture" if smoke else "final")
    command = ["python", "configs/lightning_weave/train.py", "--model-type", "qwen3.5-4B", "--student", c.language_model("student"), "--data", target, "--manifest", target / "manifest.json", "--load", Path(c.CHECKPOINT_ROOT) / "initial-megatron", "--save", root / "train", "--num-gpus", str(c.TRAIN_GPUS), "--alpha", str(c.ALPHA), "--lr", str(c.LR), "--rollout-batch-size", "64" if smoke else "256", "--global-batch-size", "64", "--num-rollout", "4" if smoke else "50", "--save-interval", "4" if smoke else "50", "--max-tokens-per-gpu", str(c.MAX_PROMPT + c.MAX_RESPONSE), "--top-k", "16", "--responses-per-prompt", "4", "--max-prompt-length", str(c.MAX_PROMPT), "--max-response-length", str(c.MAX_RESPONSE), "--temperature", "1.0", "--top-p", "1.0", "--seed", str(seed), "--rollout-seed", "42" if smoke or seed == 1234 else str(seed), "--loss-mode", "tilted_target"]
    if not smoke and seed != 1234: command.append("--rollout-shuffle")
    command = list(map(str, command)); peaks = [0] * c.TRAIN_GPUS; capacities = [0] * c.TRAIN_GPUS
    stop = threading.Event(); steps = []; invalid_metric_lines = []
    def monitor():
        while not stop.wait(.5):
            query = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            for i, line in enumerate(query.stdout.splitlines()):
                used, capacity = map(int, line.split(",")); peaks[i] = max(peaks[i], used); capacities[i] = capacity
    watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
    try:
        with (root / "train.log").open("w") as log:
            process = subprocess.Popen(command, cwd="/workspace/Lightning-Weave", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, "MEGATRON_PATH": "/root/Megatron-LM"})
            for line in process.stdout:
                log.write(line); log.flush()
                if "train/loss" in line and "step " in line:
                    match = re.search(r"step \d+: (\{.*\})", line)
                    if match:
                        try:
                            # MCore builds may return a scalar Tensor for grad_norm.
                            literal = re.sub(r"tensor\(([-+0-9.eE]+)(?:,[^)]*)?\)", r"\1", match.group(1))
                            metrics = ast.literal_eval(literal); steps.append(metrics)
                            print(json.dumps(metrics), flush=True)
                        except (ValueError, SyntaxError):
                            invalid_metric_lines.append(line.rstrip())
                            print(line.rstrip(), flush=True)
                elif any(word in line for word in ("Error", "Traceback", "Budget:", "OOM", "saving checkpoint")):
                    print(line.rstrip(), flush=True)
            if process.wait(): raise RuntimeError(f"training failed; see {root / 'train.log'}")
    finally:
        subprocess.run(["ray", "stop", "--force"], capture_output=True)
        stop.set(); watcher.join()
        write(root / "attempt_memory.json", {"gpu_peak_observed_MiB": peaks, "gpu_capacity_MiB": capacities})
        checkpoints.commit()
    if invalid_metric_lines:
        raise RuntimeError(f"non-finite/unparseable training metrics: {invalid_metric_lines[:2]}")
    expected = 4 if smoke else c.UPDATES
    # Deduplicate Ray log repetitions by optimizer step.
    steps = list({m["train/step"]: m for m in steps}.values())
    if len(steps) != expected or not all(math.isfinite(m["train/loss"]) and math.isfinite(m["train/grad_norm"]) for m in steps):
        raise RuntimeError(f"finite loss/gradient evidence missing: {len(steps)} steps")
    if smoke and steps[2]["train/loss"] >= steps[0]["train/loss"]:
        raise RuntimeError("fixed-batch smoke loss failed to decrease")
    weighted = check_initial_weighted_metrics(steps[0]) if smoke else None
    report = {"seed": seed, "training_gpus": c.TRAIN_GPUS, "training_gpu_type": "H200", "optimizer_updates": len(steps), "losses": [m["train/loss"] for m in steps], "grad_norms": [m["train/grad_norm"] for m in steps], "step0_behavior_topk_abs_diff": steps[0].get("train/behavior_topk_abs_diff"), "step0_weighted_agreement": weighted, "gpu_peak_observed_MiB": peaks, "gpu_capacity_MiB": capacities, "vocab_size": 248320, "sequence_capacity": c.MAX_PROMPT + c.MAX_RESPONSE, "stress_sequence_length": c.MAX_PROMPT + c.MAX_RESPONSE if smoke else None, "synthetic_stress_excluded_from_full_run": True}
    write(root / "train_audit.json", report); checkpoints.commit()
    return report


@app.function(image=overlay, cpu=16, memory=131072, timeout=14400, volumes=MOUNTS)
def export(seed: int):
    reload(); require_smokes(False)
    root = Path(c.CHECKPOINT_ROOT) / f"seed{seed}"
    audit = json.loads((root / "train_audit.json").read_text())
    if audit["optimizer_updates"] != c.UPDATES: raise RuntimeError("incomplete training")
    env = {**os.environ, "STUDENT_MODEL": c.language_model("student"), "INPUT_DIR": str(root / "train/iter_0000049"), "OUTPUT_DIR": str(root / "hf"), "MODEL_TYPE": "qwen3.5-4B", "MEGATRON_PATH": "/root/Megatron-LM"}
    run(["bash", "scripts/convert_checkpoint.sh", "to-hf"], env=env)
    run(["python", "tools/build_qwen35_vllm_language_wrapper.py", "--language-model", root / "hf", "--outer-checkpoint", c.snapshot("student"), "--output", root / "vllm"])
    write(root / "vllm/transfer_provenance.json", {"student": c.MODELS["student"], "anchors": c.MODELS, "seed": seed, "alpha": c.ALPHA, "coefficient": 1, "updates": c.UPDATES, "data": c.data_root(False), "protocol": c.PROTOCOL.resolved()})
    checkpoints.commit()
    return str(root / "vllm")


@app.function(image=cpu, cpu=1, memory=2048, timeout=86400, nonpreemptible=True)
def score_chain(smoke: bool, workers: int):
    for step in ("post", "pre", "reference"):
        calls = [score.spawn(step, smoke, rank, workers) for rank in range(workers)]
        for call in calls: call.get()
    return scoring_audit.remote(smoke)


@app.function(image=cpu, cpu=1, memory=2048, timeout=86400, nonpreemptible=True)
def rollout_chain(smoke: bool, workers: int):
    calls = [rollouts.spawn(smoke, rank, workers) for rank in range(workers)]
    for call in calls: call.get()
    return rollout_audit.remote(smoke)


@app.function(image=cpu, cpu=4, memory=8192, timeout=3600, volumes=MOUNTS)
def record_reported_smokes(report: dict):
    """Call only after the completed smoke report has been delivered to the user."""
    reload()
    root = Path(c.data_root(True)); checkpoint = Path(c.CHECKPOINT_ROOT)
    observed = {name: json.loads((root / f"{name}_audit.json").read_text()) for name in ("collection", "rollout", "scoring")}
    observed["step0_weights"] = json.loads((checkpoint / "step0_weight_audit.json").read_text())
    observed["step0_serving"] = json.loads((checkpoint / "step0_serving_audit.json").read_text())
    observed["training"] = json.loads((checkpoint / "training-smoke/train_audit.json").read_text())
    if report.get("stage_smokes_passed") is not True or report.get("reported_to_user") is not True:
        raise RuntimeError("report must be delivered before full computation")
    if not observed["step0_weights"]["all_weights_identical"] or not observed["step0_serving"]["generated_tokens_identical"]:
        raise RuntimeError("step-0 evidence failed")
    if observed["training"]["optimizer_updates"] != 4 or observed["training"]["stress_sequence_length"] != c.MAX_PROMPT + c.MAX_RESPONSE:
        raise RuntimeError("training/memory smoke evidence incomplete")
    value = {**report, "smokes": observed}
    write(Path(c.DATA_ROOT) / "SMOKES_REPORTED.json", value); data.commit()
    return value


@app.function(image=cpu, cpu=1, memory=2048, timeout=86400, nonpreemptible=True)
def train_chain():
    calls = [(seed, train.spawn(False, seed)) for seed in c.SEEDS]
    return {str(seed): call.get() for seed, call in calls}


@app.function(image=cpu, cpu=1, memory=2048, timeout=86400, nonpreemptible=True)
def export_chain():
    # Full chain reaches this only after all three clean-initialized trainings.
    return {str(seed): export.remote(seed) for seed in c.SEEDS}


@app.function(image=cpu, cpu=8, memory=32768, timeout=3600, volumes=MOUNTS)
def analyze():
    from evaluation.qwen35_transfer import analyze as compare
    reload(); require_smokes(False)
    root = Path("/tau/tau-3e970a3828cb-full")
    output = Path(c.data_root(False)) / "evaluation"
    result = compare(root, output); data.commit()
    return result


@app.function(image=cpu, cpu=4, memory=8192, timeout=3600, volumes=MOUNTS)
def preserve_bad_streamed_scores():
    reload()
    root = Path(c.data_root(True)); archive = root / "unsupported_cached_context_chunks"
    archive.mkdir(exist_ok=True)
    for name in ("post", "pre", "final", "scoring_audit.json"):
        source = root / name
        if source.exists():
            if (archive / name).exists(): raise FileExistsError(archive / name)
            source.rename(archive / name)
    data.commit()
    return str(archive)


def expected_base_training_metrics():
    """Independent token-weighted reference with current policy equal to behavior."""
    import torch
    import pyarrow.parquet as pq
    from slime.rollout.offline_direct_opd import offline_direct_opd_tilted_target_terms
    totals = {}; count = 0
    for p in sorted((Path(c.data_root(True)) / "final").glob("*.parquet")):
        for row in pq.read_table(p).to_pylist():
            m = row["metadata"]; b = torch.tensor(m["behavior_topk_log_probs"], dtype=torch.float32)
            terms = offline_direct_opd_tilted_target_terms(b, b, torch.tensor(m["post_teacher_log_probs"], dtype=torch.float32), torch.tensor(m["pre_teacher_log_probs"], dtype=torch.float32), alpha=c.ALPHA)
            mask = torch.tensor(m["loss_mask"], dtype=torch.bool); count += int(mask.sum())
            for key, value in terms.items(): totals[key] = totals.get(key, 0.0) + float(value[mask].sum())
    return {key: value / count for key, value in totals.items()}


def check_initial_weighted_metrics(step):
    expected = expected_base_training_metrics()
    # Low-probability candidate log-errors receive equal weight in the legacy
    # diagnostic. Check the probability-weighted objective/distribution instead.
    limits = {"tilted_target_loss": .001, "target_forward_kl": .0005,
              "weighted_reward_token_mean": .001, "cached_support_mass": .0001,
              "target_entropy": .00001, "positive_reward_mass": .0001}
    errors = {key: abs(step["train/" + key] - expected[key]) for key in limits}
    if any(errors[key] > limit for key, limit in limits.items()):
        raise RuntimeError(f"initial probability-weighted metrics disagree: {errors}")
    return {"expected": expected, "absolute_errors": errors, "tolerances": limits}


@app.function(image=cpu, cpu=8, memory=32768, timeout=3600, volumes=MOUNTS)
def finish_training_smoke():
    """Audit completed optimizer/memory evidence; no additional training."""
    import ast, math, re
    import numpy as np
    import pyarrow.parquet as pq
    reload()
    root = Path(c.CHECKPOINT_ROOT) / "training-smoke"
    text = (root / "train.log").read_text()
    steps = []
    for line in text.splitlines():
        match = re.search(r"step \d+: (\{.*\})", line)
        if match and "train/loss" in match.group(1):
            literal = re.sub(r"tensor\(([-+0-9.eE]+)(?:,[^)]*)?\)", r"\1", match.group(1))
            steps.append(ast.literal_eval(literal))
    steps = list({m["train/step"]: m for m in steps}.values())
    if len(steps) != 4 or not all(math.isfinite(m["train/loss"]) and math.isfinite(m["train/grad_norm"]) for m in steps):
        raise RuntimeError("four finite optimizer updates required")
    if steps[2]["train/loss"] >= steps[0]["train/loss"]:
        raise RuntimeError("fixed-batch loss failed to decrease")
    weighted = check_initial_weighted_metrics(steps[0])
    if (root / "train/latest_checkpointed_iteration.txt").read_text().strip() != "3" or not (root / "train/iter_0000003").is_dir():
        raise RuntimeError("completed four-update checkpoint missing")
    rows = pq.read_table(Path(c.data_root(True)) / "training-fixture/smoke-training.parquet").to_pylist()
    lengths = [len(r["metadata"]["prompt_tokens"]) + len(r["metadata"]["response_tokens"]) for r in rows[-64:]]
    responses = [len(r["metadata"]["response_tokens"]) for r in rows[-64:]]
    if max(lengths) != c.MAX_PROMPT + c.MAX_RESPONSE:
        raise RuntimeError("maximum-sequence fixture missing")
    line = next(line for line in text.splitlines() if "rollout 3:" in line and "rollout/total_lengths" in line)
    actual = ast.literal_eval(line.split("rollout 3: ", 1)[1])
    if abs(actual["rollout/total_lengths"] - np.mean(lengths)) > 1e-8 or abs(actual["rollout/response_lengths"] - np.mean(responses)) > 1e-8:
        raise RuntimeError("stress row did not reach actual training input")
    memory = json.loads((root / "attempt_memory.json").read_text())
    # The first four-H200 monitor retained four unused zero slots from H100:8.
    active = [i for i, capacity in enumerate(memory["gpu_capacity_MiB"]) if capacity > 0]
    memory = {key: [values[i] for i in active] for key, values in memory.items()}
    if len(memory["gpu_capacity_MiB"]) != c.TRAIN_GPUS or any(peak >= capacity for peak, capacity in zip(memory["gpu_peak_observed_MiB"], memory["gpu_capacity_MiB"])):
        raise RuntimeError("memory evidence invalid")
    report = {"seed": 1234, "training_gpus": c.TRAIN_GPUS, "training_gpu_type": "H200", "optimizer_updates": 4,
              "losses": [m["train/loss"] for m in steps], "grad_norms": [m["train/grad_norm"] for m in steps],
              "step0_behavior_topk_abs_diff": steps[0]["train/behavior_topk_abs_diff"], "step0_weighted_agreement": weighted,
              **memory, "vocab_size": 248320, "sequence_capacity": c.MAX_PROMPT + c.MAX_RESPONSE,
              "stress_sequence_length": max(lengths), "actual_stress_batch_mean_total_tokens": actual["rollout/total_lengths"],
              "synthetic_stress_excluded_from_full_run": True}
    write(root / "train_audit.json", report); checkpoints.commit()
    return report


@app.local_entrypoint()
def main(stage: str = "runtime", smoke: bool = True, seed: int = 1234, workers: int = 1, report: str = ""):
    if stage == "assets": result = assets.remote()
    elif stage == "runtime": result = runtime_probe.remote()
    elif stage == "finish-training-smoke": result = finish_training_smoke.remote()
    elif stage == "preserve-bad-scores": result = preserve_bad_streamed_scores.remote()
    elif stage == "collection": result = collection_audit.remote(smoke)
    elif stage == "rollouts": result = rollout_chain.remote(smoke, workers)
    elif stage == "score": result = score_chain.remote(smoke, workers)
    elif stage == "convert": result = convert.remote()
    elif stage == "roundtrip": result = roundtrip.remote()
    elif stage == "train":
        if smoke: training_fixture.remote()
        result = train.remote(smoke, seed)
    elif stage == "export": result = export.remote(seed)
    elif stage == "train-seeds": result = train_chain.remote()
    elif stage == "export-seeds": result = export_chain.remote()
    elif stage == "analyze": result = analyze.remote()
    elif stage == "report-smokes": result = record_reported_smokes.remote(json.loads(Path(report).read_text()))
    else: raise ValueError(f"unknown stage {stage}")
    print(json.dumps(result, indent=2, sort_keys=True))
