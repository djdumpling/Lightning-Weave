import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONFIG = load("bfcl_eval_config", ROOT / "configs/bfcl_eval/config.py")
OPD = load("looptool_opd_config_for_bfcl", ROOT / "configs/looptool_opd/config.py")


def test_models_are_the_opd_student_and_its_exact_base():
    assert CONFIG.BASE_MODEL == OPD.STUDENT_MODEL
    assert CONFIG.BASE_REVISION == OPD.STUDENT_REVISION
    assert CONFIG.MODELS["opd"] == f"{OPD.REMOTE_CHECKPOINT_ROOT}/hf"
    assert CONFIG.MODAL_MODEL_VOLUME == OPD.MODAL_MODEL_VOLUME
    assert CONFIG.MODAL_CHECKPOINT_VOLUME == OPD.MODAL_CHECKPOINT_VOLUME


def test_serving_image_matches_the_rollout_image():
    assert CONFIG.SERVING_IMAGE == OPD.ROLLOUT_IMAGE


def test_protocol_is_v3_with_a_32k_response_in_a_64k_yarn_window():
    CONFIG.PROTOCOL.validate()
    assert CONFIG.PROTOCOL.bfcl_version == "v3"
    assert CONFIG.PROTOCOL.tokens_to_generate == 32_768
    assert CONFIG.PROTOCOL.max_model_len == 65_536
    assert CONFIG.PROTOCOL.rope_scaling == {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32_768}
    with pytest.raises(ValueError, match="YaRN"):
        replace(CONFIG.PROTOCOL, max_model_len=40_960).validate()
    resolved = CONFIG.PROTOCOL.resolved()
    assert resolved["recipe_tokens_to_generate"] == 131_072
    replace(CONFIG.PROTOCOL, bfcl_version="v4", tokens_to_generate=8_192).validate()
    with pytest.raises(ValueError, match="bfcl_version"):
        replace(CONFIG.PROTOCOL, bfcl_version="v5").validate()
    with pytest.raises(ValueError, match="max_model_len"):
        replace(CONFIG.PROTOCOL, tokens_to_generate=65_536).validate()


def test_recipe_budgets_match_agentic_eval():
    assert CONFIG.RECIPE_GEN_BUDGET == {"v3": 131_072, "v4": 8_192}
    assert CONFIG.CONTEXT_OVERFLOW_MARKERS


def test_request_sampling_matches_checkpoint_defaults():
    with pytest.raises(ValueError, match="temperature"):
        replace(CONFIG.PROTOCOL, temperature=1.0).validate()
    assert CONFIG.PROTOCOL.expected_generation_defaults["top_k"] == 20


def test_run_id_tracks_protocol_but_not_throughput():
    full, smoke = CONFIG.run_id(), CONFIG.run_id(3)
    assert full.startswith("bfcl-v3-")
    assert full != smoke and full.endswith("-full") and smoke.endswith("-smoke3")
    assert CONFIG.PROTOCOL.digest() in full
    changed = replace(CONFIG.PROTOCOL, temperature=0.7, expected_generation_defaults={"temperature": 0.7, "top_k": 20, "top_p": 0.95})
    assert changed.digest() != CONFIG.PROTOCOL.digest()
    resolved = CONFIG.PROTOCOL.resolved()
    assert "max_num_seqs" not in resolved and "concurrency" not in resolved


def test_vllm_command_enables_the_bfcl_serving_contract():
    command = CONFIG.SERVING.vllm_command("/models/x", "secret", CONFIG.PROTOCOL)
    joined = " ".join(command)
    for flag in (
        "--enable-auto-tool-choice",
        "--tool-call-parser hermes",
        "--reasoning-parser qwen3",
        "--generation-config auto",
        "--api-key secret",
        "--served-model-name model",
        "--max-model-len 65536",
        '"rope_type": "yarn"',
        "--enable-prefix-caching",
    ):
        assert flag in joined
    # The rollout stage disabled CUDA graphs; serving must not.
    assert "--enforce-eager" not in command and "--cudagraph-mode" not in command
    assert "--tensor-parallel-size" not in command


def test_driver_command_is_the_agentic_eval_recipe():
    command = CONFIG.driver_command(
        category="simple_python", input_file="in.jsonl", output_file="out/output.jsonl", base_url="https://x/v1", concurrency=64
    )
    assert command[:3] == ["python", "-m", "nemo_skills.inference.eval.bfcl"]
    for arg in (
        "++eval_type=bfcl",
        "++server.server_type=openai",
        "++inference.temperature=0.6",
        "++inference.top_p=0.95",
        "++use_client_parsing=False",
        "++skip_filled=True",
        "++max_concurrent_requests=64",
        "++inference.tokens_to_generate=32768",
        "++inference.timeout=3600",
        "++inference.random_seed=0",  # NeMo-Skills' default, now explicit
    ):
        assert arg in command
    assert not any(arg.startswith("++max_samples") for arg in command)
    # a decoding-seed replicate changes every request's sampling seed, not only vLLM's engine seed
    replicate = CONFIG.driver_command(
        category="simple_python", input_file="i", output_file="o", base_url="u", concurrency=64, random_seed=3
    )
    assert "++inference.random_seed=3" in replicate and "++inference.random_seed=0" not in replicate


def test_smoke_runs_are_partial_and_exclude_memory():
    command = CONFIG.driver_command(
        category="parallel", input_file="i", output_file="o", base_url="u", concurrency=8, smoke_samples=3
    )
    assert "++max_samples=3" in command and "++eval_config.partial_eval=True" in command
    with pytest.raises(ValueError, match="memory"):
        CONFIG.driver_command(category="memory_kv", input_file="i", output_file="o", base_url="u", concurrency=8, smoke_samples=3)


def test_every_multi_turn_task_runs_at_once_on_a_data_parallel_server():
    for category in ("multi_turn_base", "multi_turn_long_context", "multi_turn_miss_func", "multi_turn_miss_param"):
        assert CONFIG.LANES.concurrency_for(category) == 200
    assert CONFIG.LANES.concurrency_for("irrelevance") == CONFIG.LANES.default_concurrency
    assert CONFIG.SERVING.gpu == ["H100:4", "H200:4"]
    command = CONFIG.SERVING.vllm_command("/m", "k", CONFIG.PROTOCOL)
    assert command[command.index("--data-parallel-size") + 1] == "4"
    assert max(CONFIG.LANES.concurrency.values()) <= CONFIG.SERVING.max_num_seqs * CONFIG.SERVING.data_parallel_size


def test_decoding_seed_replicates_get_their_own_run_trees():
    seeded = replace(CONFIG.PROTOCOL, seed=1)
    assert CONFIG.run_id() != CONFIG.run_id(protocol=seeded)
    assert CONFIG.run_id(protocol=replace(CONFIG.PROTOCOL, seed=0)) == CONFIG.run_id()


def test_inference_time_baselines_only_change_serving():
    assert CONFIG.MODEL_SPECS["base-cap4k"].max_completion_tokens == 4_096
    assert CONFIG.MODEL_SPECS["base-concise"].system_prompt == CONFIG.CONCISE_SYSTEM_PROMPT
    assert CONFIG.MODEL_SPECS["base-cap4k"].path == CONFIG.MODELS["base"]


def test_agent_eff_students_take_the_concise_prompt_as_a_last_tag_part():
    root = CONFIG.AGENT_EFF_ROOT
    concise = CONFIG.resolve_model("ae.joint.acc-legacy+decs.concise")
    assert concise.path == f"{root}/joint/acc-legacy+decs/seed{CONFIG.AGENT_EFF_DEFAULT_SEED}/hf"
    assert concise.system_prompt == CONFIG.CONCISE_SYSTEM_PROMPT
    seeded = CONFIG.resolve_model("ae.joint.acc-legacy.s5678.concise")
    assert seeded.path == f"{root}/joint/acc-legacy/seed5678/hf" and seeded.system_prompt == CONFIG.CONCISE_SYSTEM_PROMPT
    assert CONFIG.resolve_model("ae.joint.acc-legacy.s5678").system_prompt is None
    assert CONFIG.resolve_model("ae.joint.concise").path.endswith("/joint/concise/seed1234/hf")  # a variant name, not a suffix
