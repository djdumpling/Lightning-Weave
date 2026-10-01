"""Donor audits: untrained-row detection, scored-shard integrity, and evidence coverage."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from data_curation import audit_donor_tokens as audit
from data_curation import verify_donor_scores as verify
from data_curation.shift_geometry import iter_joined_rows
from data_curation.shift_states import StateLabeler, TokenTable

K, LENGTH, ROWS = 3, 4, 4
SPECIAL = (10, 11)


def write_scores(root, *, revision="post", candidates=(1, 10, 11), tie=True, extra=False):
    rng = np.random.default_rng(0)
    root.mkdir(parents=True)
    fields = {
        name: []
        for name in (
            "sample_id",
            "prompt_id",
            "response_tokens",
            "candidate_ids",
            "behavior_topk_log_probs",
            "loss_mask",
            "pre_teacher_log_probs",
            "post_teacher_log_probs",
            "pre_teacher_revision",
            "post_teacher_revision",
        )
    }
    for index in range(ROWS):
        pre = rng.normal(size=(LENGTH, K)) - 3
        post = pre + rng.normal(size=(LENGTH, K))
        if tie:
            post[:, 2] = post[:, 1]  # both special candidates get identical scores, as untrained rows do
            pre[:, 2] = pre[:, 1]
        fields["sample_id"].append(f"s{index}")
        fields["prompt_id"].append(f"p{index}")
        fields["response_tokens"].append([1] * LENGTH)
        fields["candidate_ids"].append([list(candidates)] * LENGTH)
        fields["behavior_topk_log_probs"].append(np.log(np.tile([0.6, 0.2, 0.1], (LENGTH, 1))).tolist())
        fields["loss_mask"].append([True] * LENGTH)
        fields["pre_teacher_log_probs"].append(pre.tolist())
        fields["post_teacher_log_probs"].append(post.tolist())
        fields["pre_teacher_revision"].append("pre")
        fields["post_teacher_revision"].append(revision)
    floats = pa.list_(pa.list_(pa.float32()))
    types = {
        "candidate_ids": pa.list_(pa.list_(pa.int32())),
        "behavior_topk_log_probs": floats,
        "pre_teacher_log_probs": floats,
        "post_teacher_log_probs": floats,
        "loss_mask": pa.list_(pa.bool_()),
        "response_tokens": pa.list_(pa.int32()),
    }
    metadata = pa.StructArray.from_arrays(
        [pa.array(v, type=types.get(n)) for n, v in fields.items()], names=list(fields)
    )
    pq.write_table(pa.table({"metadata": metadata}), root / "rollouts-r00000-00000.parquet")
    if extra:
        pq.write_table(pa.table({"metadata": metadata}), root / "rollouts-r00009-00000.parquet")
    return root


def test_check_shards_accepts_aligned_scores_and_rejects_drift(tmp_path):
    base = write_scores(tmp_path / "base")
    donor = write_scores(tmp_path / "donor")
    assert verify.check_shards(base, donor, {"post": "post", "pre": "pre"}) == ROWS
    with pytest.raises(ValueError, match="revisions"):
        verify.check_shards(base, write_scores(tmp_path / "stale", revision="old"), {"post": "post", "pre": "pre"})
    with pytest.raises(ValueError, match="candidates"):
        verify.check_shards(
            base, write_scores(tmp_path / "other", candidates=(2, 10, 11)), {"post": "post", "pre": "pre"}
        )
    with pytest.raises(ValueError, match="lacks"):
        verify.check_shards(base, write_scores(tmp_path / "extra", extra=True), {"post": "post", "pre": "pre"})


def test_evidence_report_combines_mapping_and_training_and_flags_ties(tmp_path):
    base = write_scores(tmp_path / "base")
    donor = write_scores(tmp_path / "donor")
    labeler = StateLabeler(
        TokenTable.from_vocab(
            {"a": 1}, {"<think>": 5, "</think>": 6, "<tool_call>": 10, "</tool_call>": 7, "<|im_end|>": 11}
        )
    )
    rows = iter_joined_rows(base, {"donor": donor}, untrained={"donor": {10}})
    report = verify.evidence_report(rows, "donor", labeler, {10: "<tool_call>", 11: "<|im_end|>"})
    everything = report["by_state"]["all"]
    assert everything["whole_position_evidence"] == 0.0
    assert everything["behavior_mass_evidence"] == pytest.approx((0.6 + 0.1) / 0.9)
    assert report["structural_shift_ties"]["<tool_call>"]["tie_rate"] == 1.0


def test_row_diagnostics_flag_duplicate_and_zero_rows(tmp_path):
    from safetensors.torch import save_file

    rows = torch.randn(4, 8)
    rows[1] = rows[0]
    rows[3] = 0.0
    save_file({"model.embed_tokens.weight": rows}, str(tmp_path / "model.safetensors"))
    diagnostics = audit.row_diagnostics(tmp_path, [0, 1, 2, 3])
    assert [diagnostics[i]["untrained"] for i in range(4)] == [True, True, False, True]
    assert diagnostics[0]["exact_duplicates"] == [1]
    assert diagnostics[2]["max_cosine_to_other_special"] < 1.0


def test_row_diagnostics_flag_rows_one_rounding_step_apart(tmp_path):
    """7B/14B checkpoints keep untrained special rows that differ by about one bfloat16 step, not exactly."""
    from safetensors.torch import save_file

    torch.manual_seed(0)
    rows = torch.randn(5, 64)
    rows[1] = rows[0] + 1e-5 * torch.randn(64)  # an untrained row's shared initialization, perturbed
    rows[3] = 0.9 * rows[2] + 0.44 * torch.randn(64)  # correlated, as trained special rows can be (cosine ~0.9)
    save_file({"model.embed_tokens.weight": rows}, str(tmp_path / "model.safetensors"))
    diagnostics = audit.row_diagnostics(tmp_path, [0, 1, 2, 3, 4])
    assert [diagnostics[i]["untrained"] for i in range(5)] == [True, True, False, False, False]
    assert diagnostics[0]["exact_duplicates"] == [] and diagnostics[0]["near_duplicates"] == [1]
    assert 0.8 < diagnostics[2]["max_cosine_to_other_special"] < audit.NEAR_DUPLICATE_COSINE


@pytest.mark.parametrize("aliases", [{"</think>": "a"}, {"</think>": "<eos>", "<tool_call>": "<eos>"}])
def test_audit_and_scorer_both_reject_alias_collisions(monkeypatch, aliases):
    from transformers import AutoTokenizer
    from data_curation.precompute_direct_opd_scores import ExactTokenStringProjection

    student = SimpleNamespace(
        get_vocab=lambda: {"a": 0, "</think>": 1, "<tool_call>": 2},
        get_added_vocab=lambda: {"</think>": 1, "<tool_call>": 2},
    )
    teacher = SimpleNamespace(get_vocab=lambda: {"a": 4, "<eos>": 5})
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: teacher)
    with pytest.raises(ValueError, match="already mapped"):
        audit.pair_token_audit(student, {"pre": "unused"}, aliases)
    with pytest.raises(ValueError, match="already mapped"):
        ExactTokenStringProjection(student, teacher, aliases)
