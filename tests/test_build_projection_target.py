import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from data_curation import build_projection_target as builder
from data_curation import decision_projection as projection
from slime.rollout.offline_direct_opd import (
    SEQUENCE_WEIGHT_FIELD,
    hydrate_offline_direct_opd_sample,
    validate_sealed_manifest,
)
from slime.utils.types import Sample

GENERATION = {"temperature": 0.6, "top_p": 0.95, "top_k": 2, "responses_per_prompt": 2, "max_response_length": 8,
              "max_prompt_length": 8, "sampling_top_k": 20}


def rollout(prompt_id, response_id, response, tokens):
    """A row as ``collect_direct_opd_rollouts.py`` writes it: teacher fields still empty."""
    length = len(tokens)
    return {
        "prompt": f"prompt {prompt_id}",
        "label": "",
        "metadata": {
            "is_offline_direct_opd": False,
            "offline_direct_opd_stage": "rollout_collected",
            "schema_version": "offline_direct_opd_v1",
            "student_ref_sampled_log_probs": None,
            "post_teacher_log_probs": None,
            "pre_teacher_log_probs": None,
            "generation_seed": 42,
            "generation_config_hash": "g" * 64,
            "generation_config": GENERATION,
            "student_revision": "recipient-rev",
            "post_teacher_revision": None,
            "pre_teacher_revision": None,
            "tokenizer_hash": "t" * 64,
            "sample_id": f"{prompt_id}-{response_id}",
            "prompt_id": prompt_id,
            "group_id": prompt_id,
            "response_id": response_id,
            "prompt_tokens": [1, 2],
            "response_tokens": tokens,
            "loss_mask": [1] * length,
            "response": response,
            "response_length": length,
            "candidate_ids": [[token, token + 100] for token in tokens],
            "behavior_topk_log_probs": [[-0.1, -2.5]] * length,
            "behavior_sampled_log_probs": [-0.1] * length,
            "finish_reason": "stop",
        },
    }


@pytest.fixture
def inputs(tmp_path):
    rows = [
        rollout("p0", 0, "<think>long plan</think>A", [5, 6, 7, 8]),
        rollout("p0", 1, "<think>short</think>A", [5, 9]),
        rollout("p1", 0, "<think>x</think>B", [3]),
        rollout("p1", 1, "<think>y</think>C", [4, 4]),
    ]
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    pq.write_table(pa.Table.from_pylist(rows[:2]), rollouts / "rollouts-r00000-00000.parquet")
    pq.write_table(pa.Table.from_pylist(rows[2:]), rollouts / "rollouts-r00001-00000.parquet")
    pq.write_table(pa.table({"sample_id": [row["metadata"]["sample_id"] for row in rows],
                             "score": [-4.0, -1.0, -1.0, -2.0]}), tmp_path / "scores.parquet")
    projection.main(["weights", "--rollouts", str(rollouts), "--scores", str(tmp_path / "scores.parquet"),
                     "--alpha", "1.0", "--output", str(tmp_path / "weights.parquet")])
    asset_lock = {
        "tokenizer_hash": "t" * 64,
        "token_id_compatibility": {"mode": "official_input_tokenizer_null", "model_vocab_size": 300,
                                   "normalization_vocab_size": 300},
        "models": {"student": {"path": "/base", "revision": "base-rev", "model_vocab_size": 300,
                               "tokenizer_hash": "t" * 64}},
    }
    (tmp_path / "assets.json").write_text(json.dumps(asset_lock))
    (tmp_path / "prompts.parquet").write_bytes(b"prompts")
    return tmp_path


def build(inputs, arm, output_name=None):
    output = inputs / (output_name or arm)
    builder.main(["--rollouts", str(inputs / "rollouts"), "--weights", str(inputs / "weights.parquet"), "--arm", arm,
                  "--recipient", "/checkpoints/recipient/hf", "--recipient-revision", "recipient-rev",
                  "--asset-lock", str(inputs / "assets.json"), "--source-dataset", str(inputs / "prompts.parquet"),
                  "--output-dir", str(output)])
    return output


def test_each_arm_seals_the_same_rows_with_its_own_weights(inputs):
    weights = pq.read_table(inputs / "weights.parquet").to_pydict()
    for arm in projection.ARMS:
        output = build(inputs, arm)
        manifest = validate_sealed_manifest(output / "manifest.json", expected_top_k=2)
        assert manifest["total_rows"] == 4 and manifest["post_teacher_model"]["arm"] == arm
        assert manifest["student_model"]["path"] == "/checkpoints/recipient/hf"
        assert manifest["generation_config"] == GENERATION
        rows = [row for shard in manifest["shards"] for row in pq.read_table(output / shard["path"]).to_pylist()]
        by_id = {row["metadata"]["sample_id"]: row["metadata"] for row in rows}
        for sample_id, weight in zip(weights["sample_id"], weights[f"weight_{arm}"], strict=True):
            assert by_id[sample_id][SEQUENCE_WEIGHT_FIELD] == pytest.approx(weight)
        hydrated = hydrate_offline_direct_opd_sample(
            Sample(prompt=[1, 2], metadata=rows[0]["metadata"]), tokenizer=None, expected_top_k=2, trusted_sealed=True
        )
        torch.testing.assert_close(hydrated.sequence_weights, torch.full((4,), by_id["p0-0"][SEQUENCE_WEIGHT_FIELD],
                                                                         dtype=torch.float32))
        # A zero teacher shift: the loss reads none, and nothing else can move the target.
        assert rows[0]["metadata"]["post_teacher_log_probs"] == rows[0]["metadata"]["behavior_topk_log_probs"]
    revisions = {json.loads((inputs / arm / "manifest.json").read_text())["post_teacher_model"]["revision"]
                 for arm in projection.ARMS}
    assert len(revisions) == 3


def test_sealing_refuses_mismatched_inputs(inputs):
    with pytest.raises(ValueError, match="sampled by"):
        builder.main(["--rollouts", str(inputs / "rollouts"), "--weights", str(inputs / "weights.parquet"),
                      "--arm", "projected", "--recipient", "/r", "--recipient-revision", "another",
                      "--asset-lock", str(inputs / "assets.json"), "--source-dataset", str(inputs / "prompts.parquet"),
                      "--output-dir", str(inputs / "refused")])
    build(inputs, "uniform")
    with pytest.raises(FileExistsError):
        build(inputs, "uniform")
    table = pq.read_table(inputs / "weights.parquet")
    pq.write_table(table.slice(0, 3).replace_schema_metadata(table.schema.metadata), inputs / "weights.parquet")
    with pytest.raises(ValueError, match="no weight"):
        build(inputs, "projected", "missing")
    assert not (inputs / "missing").exists()  # nothing is published from a failed build
