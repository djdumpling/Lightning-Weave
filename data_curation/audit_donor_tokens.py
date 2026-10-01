#!/usr/bin/env python3
"""Audit a donor anchor pair on a cached student action space before trusting its shift.

Two failure modes make a pair's log-ratio meaningless at specific candidates:

1. **Untrained tokens.** Base checkpoints often ship special-token rows they
   never trained (Qwen3-4B-Base gives every chat/think/tool token the same
   log-probability). An output row that is exactly duplicated by another special
   row, or has near-zero norm, is positive evidence of an untrained token. A
   token is flagged for the pair if either anchor's row is: the log-ratio then
   contains an arbitrary term. The absence of a flag is not proof of training,
   so every structural token's row norm and closest special-row cosine are
   reported for manual review, and ``verify_donor_scores.py`` checks scored
   states for identical log-probabilities.
2. **Unmappable candidates.** Under cross-tokenizer projection a candidate
   whose exact string is absent from the teacher vocabulary cannot be scored.

The coverage audit reports, per token-state type, the fraction of positions
whose every candidate is usable and the behavior mass on usable candidates,
under three rules: exact string mapping, mapping with the alias map, and
aliased mapping restricted to tokens both anchors trained (the evidence the
composer actually uses).

Output: ``untrained.json`` (``{pair: [student token ids]}``) and ``audit.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import write_json
from data_curation.prepare_direct_opd_assets import candidate_token_mapping
from data_curation.shift_geometry import iter_joined_rows
from data_curation.shift_states import STATE_TYPES, StateLabeler, TokenTable

STRUCTURAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)


def _output_rows(model_path: Path, ids: list[int]) -> np.ndarray:
    """Rows of the output projection (tied embeddings when there is no lm_head) for ``ids``."""
    from safetensors import safe_open

    index = model_path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(model_path / "model.safetensors", framework="np") as handle:
            weight_map = {key: "model.safetensors" for key in handle.keys()}
    key = "lm_head.weight" if "lm_head.weight" in weight_map else "model.embed_tokens.weight"
    with safe_open(model_path / weight_map[key], framework="pt") as handle:
        tensor = handle.get_slice(key)
        rows, hidden = tensor.get_shape()
        return np.stack([tensor[i : i + 1].float().numpy()[0] if i < rows else np.zeros(hidden) for i in ids])


# Untrained special rows of one checkpoint keep their shared initialization: in the 7B and 14B R1-Distill
# checkpoints they differ by about one bfloat16 step, so they are not exact duplicates but have cosine within 1e-7
# of 1, while trained special rows stay below 0.9.
NEAR_DUPLICATE_COSINE = 1 - 1e-4


def row_diagnostics(
    model_path: Path, special_ids: list[int], *, rtol: float = 1e-6, near_cosine: float = NEAR_DUPLICATE_COSINE
) -> dict[int, dict]:
    """Per teacher special id: norm, closest other special row, (near-)duplicates, and the untrained flag."""
    rows = _output_rows(model_path, special_ids)
    norms = np.linalg.norm(rows, axis=-1)
    reference = float(np.median(norms[norms > 0])) if (norms > 0).any() else 1.0
    unit = rows / np.clip(norms[:, None], 1e-30, None)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    diagnostics = {}
    for i, token in enumerate(special_ids):
        duplicates = [
            special_ids[j]
            for j in range(len(special_ids))
            if j != i and np.allclose(rows[i], rows[j], rtol=rtol, atol=1e-8 * reference)
        ]
        near_zero = bool(norms[i] < 1e-6 * reference)
        near = [special_ids[j] for j in range(len(special_ids)) if j != i and similarity[i, j] >= near_cosine]
        diagnostics[token] = {
            "norm_over_median": float(norms[i] / reference),
            "max_cosine_to_other_special": float(similarity[i].max()) if len(special_ids) > 1 else None,
            "exact_duplicates": duplicates,
            "near_duplicates": near,
            "untrained": near_zero or bool(duplicates) or bool(near),
        }
    return diagnostics


def pair_token_audit(student, models: dict[str, Path], aliases: dict[str, str]) -> tuple[dict, dict]:
    """(untrained student ids per role, structural-token diagnostics per role)."""
    from transformers import AutoTokenizer

    student_special = sorted(set(student.get_added_vocab().values()))
    symbols = {token_id: symbol for symbol, token_id in student.get_vocab().items()}
    untrained, structural = {}, {}
    for role, path in models.items():
        teacher = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        mapping = candidate_token_mapping(student.get_vocab(), teacher.get_vocab(), aliases)
        teacher_special = sorted({mapping[i] for i in student_special if i in mapping})
        diagnostics = row_diagnostics(path, teacher_special)
        untrained[role] = sorted(i for i in student_special if i in mapping and diagnostics[mapping[i]]["untrained"])
        structural[role] = {
            symbols[i]: {"teacher_id": mapping.get(i), **(diagnostics.get(mapping.get(i)) or {"unmapped": True})}
            for i in student_special
            if symbols[i] in STRUCTURAL_TOKENS
        }
    return untrained, structural


def coverage_audit(rows, labeler: StateLabeler, rules: dict[str, set[int]]) -> dict:
    """Per rule and state type: positions whose every candidate is usable, and usable behavior mass."""
    stats = {rule: {state: np.zeros(4) for state in ("all", *STATE_TYPES)} for rule in rules}
    for row in rows:
        probs = np.exp(row.behavior_log_probs)
        codes = labeler.label(row.response_tokens, row.candidate_ids, probs)
        for rule, usable_ids in rules.items():
            usable = np.isin(row.candidate_ids, list(usable_ids))
            for state in ("all", *STATE_TYPES):
                selected = row.loss_mask if state == "all" else row.loss_mask & (codes == STATE_TYPES.index(state))
                stats[rule][state] += [
                    selected.sum(),
                    usable[selected].all(axis=-1).sum(),
                    (probs[selected] * usable[selected]).sum(),
                    probs[selected].sum(),
                ]
    return {
        rule: {
            state: {
                "positions": int(total),
                "whole_position_coverage": float(kept / total),
                "behavior_mass_coverage": float(mass / max(total_mass, 1e-12)),
            }
            for state, (total, kept, mass, total_mass) in by_state.items()
            if total
        }
        for rule, by_state in stats.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="pair name used as the untrained.json key")
    parser.add_argument("--student", required=True, type=Path)
    parser.add_argument("--post", required=True, type=Path)
    parser.add_argument("--pre", required=True, type=Path)
    parser.add_argument("--special-token-alias", type=Path)
    parser.add_argument("--cache", type=Path, help="sealed cache for the coverage audit")
    parser.add_argument("--audit-rows", type=int, default=400)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    aliases = json.loads(args.special_token_alias.read_text()) if args.special_token_alias else {}
    student = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    models = {"post": args.post, "pre": args.pre}
    untrained, structural = pair_token_audit(student, models, aliases)
    union = sorted(set(untrained["post"]) | set(untrained["pre"]))
    report = {
        "name": args.name,
        "untrained_by_role": untrained,
        "untrained": union,
        "structural_tokens": structural,
        "aliases": aliases,
    }
    if args.cache:
        teachers = {role: AutoTokenizer.from_pretrained(path, trust_remote_code=True) for role, path in models.items()}

        def mapped(alias_map):
            return set.intersection(*(
                set(candidate_token_mapping(student.get_vocab(), teacher.get_vocab(), alias_map))
                for teacher in teachers.values()
            ))

        labeler = StateLabeler(TokenTable.from_tokenizer(student))
        rows = list(iter_joined_rows(args.cache, {}, max_rows=args.audit_rows))
        rules = {"exact": mapped({}), "aliased": mapped(aliases), "aliased_and_trained": mapped(aliases) - set(union)}
        report["coverage"] = coverage_audit(rows, labeler, rules)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json({args.name: union}, args.output_dir / "untrained.json")
    write_json(report, args.output_dir / "audit.json")
    print(json.dumps({key: report[key] for key in ("name", "untrained_by_role")}, indent=2))


if __name__ == "__main__":
    main()
