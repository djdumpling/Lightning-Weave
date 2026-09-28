#!/usr/bin/env python3
"""Render and deterministically select LoopTool prompts for Offline Direct-OPD.

The canonical input is produced by ``prepare_looptool_rl.py``. Reference
targets are intentionally not copied: Offline Direct-OPD learns only from
fresh behavior-policy trajectories and cached anchor likelihoods.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import pyarrow as pa

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import file_sha256, write_json, write_parquet


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Canonical LoopTool JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Selected prompt Parquet")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-prompt-length", type=int, default=8_192)
    parser.add_argument("--num-prompts", type=int, default=3_200)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--expected-input-rows", type=int, default=23_000, help="0 disables this check")
    parser.add_argument("--tokenizer-batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.max_prompt_length <= 0 or args.num_prompts <= 0 or args.tokenizer_batch_size <= 0:
        parser.error("length, prompt count, and tokenizer batch size must be positive")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        row["messages"],
        tools=row["tools"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def token_lengths(tokenizer: Any, prompts: list[str], batch_size: int) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=True,
        )
        lengths.extend(int(value) for value in encoded["length"])
    return lengths


def selection_key(row_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{row_id}".encode()).hexdigest()


def percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "conversation_kind": dict(sorted(Counter(row["conversation_kind"] for row in rows).items())),
        "target_kind": dict(sorted(Counter(row["target_kind"] for row in rows).items())),
    }


def prepare_rows(
    source: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_prompt_length: int,
    num_prompts: int,
    seed: int,
    tokenizer_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = [render_prompt(tokenizer, row) for row in source]
    lengths = token_lengths(tokenizer, prompts, tokenizer_batch_size)
    seen_prompts: dict[str, str] = {}
    eligible: list[dict[str, Any]] = []
    duplicate_prompt_ids: list[dict[str, str]] = []
    over_limit = 0
    for row, prompt, length in zip(source, prompts, lengths, strict=True):
        if length > max_prompt_length:
            over_limit += 1
            continue
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_hash in seen_prompts:
            duplicate_prompt_ids.append({"kept": seen_prompts[prompt_hash], "removed": row["id"]})
            continue
        seen_prompts[prompt_hash] = row["id"]
        eligible.append(
            {
                "prompt": prompt,
                "label": "",
                "prompt_id": row["id"],
                "source_index": int(row["metadata"]["source_index"]),
                "prompt_token_length": length,
                "conversation_kind": row["metadata"]["conversation_kind"],
                # Kept only for distribution auditing; references themselves
                # never enter the DOPD prompt or objective.
                "target_kind": row["metadata"]["target_kind"],
            }
        )
    if len(eligible) < num_prompts:
        raise ValueError(f"only {len(eligible):,} eligible unique prompts; requested {num_prompts:,}")
    selected = sorted(eligible, key=lambda row: selection_key(row["prompt_id"], seed))[:num_prompts]
    output_rows = [
        {key: row[key] for key in ("prompt", "label", "prompt_id", "source_index", "prompt_token_length")}
        for row in selected
    ]
    summary = {
        "canonical_rows": len(source),
        "max_prompt_length": max_prompt_length,
        "over_limit_rows": over_limit,
        "eligible_unique_prompts": len(eligible),
        "exact_rendered_prompt_duplicates_removed": len(duplicate_prompt_ids),
        "duplicate_prompt_ids": duplicate_prompt_ids,
        "selection_seed": seed,
        "selected_prompts": len(selected),
        "eligible_distribution": distribution(eligible),
        "selected_distribution": distribution(selected),
        "student_prompt_token_lengths": {
            "p50": percentile([row["prompt_token_length"] for row in eligible], 50),
            "p90": percentile([row["prompt_token_length"] for row in eligible], 90),
            "p95": percentile([row["prompt_token_length"] for row in eligible], 95),
            "p99": percentile([row["prompt_token_length"] for row in eligible], 99),
            "max": max(row["prompt_token_length"] for row in eligible),
        },
    }
    return output_rows, summary


def main() -> None:
    args = parse_args()
    for path in (args.output, args.summary):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {path}")
    source = read_jsonl(args.input)
    if args.expected_input_rows and len(source) != args.expected_input_rows:
        raise ValueError(f"canonical input has {len(source):,} rows; expected {args.expected_input_rows:,}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
    )
    selected, summary = prepare_rows(
        source,
        tokenizer,
        max_prompt_length=args.max_prompt_length,
        num_prompts=args.num_prompts,
        seed=args.selection_seed,
        tokenizer_batch_size=args.tokenizer_batch_size,
    )
    write_parquet(pa.Table.from_pylist(selected), args.output, compression="zstd")
    summary.update(
        {
            "schema_version": "looptool_direct_opd_prompts_v1",
            "canonical_input": str(args.input),
            "canonical_input_sha256": file_sha256(args.input),
            "output": str(args.output),
            "output_sha256": file_sha256(args.output),
            "tokenizer": args.tokenizer,
            "tokenizer_revision": args.tokenizer_revision,
            "targets_in_output": False,
        }
    )
    write_json(summary, args.summary)
    print(
        f"Wrote {len(selected):,} prompts from {summary['eligible_unique_prompts']:,} eligible rows "
        f"to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
