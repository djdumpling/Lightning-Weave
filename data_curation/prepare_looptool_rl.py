#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets==4.1.1",
#   "transformers==4.57.1",
#   "jinja2==3.1.6",
#   "jsonschema==4.25.1",
#   "datasketch==1.6.5",
#   "bfcl-eval==2026.3.23",
# ]
# ///
"""Deterministic, CPU-only RL preprocessing for zhangkangning/LoopTool-23k.

Stages: parse -> structural validation -> conflicting references -> exact dedup
-> prompt rendering/measurement -> conservative near dedup -> optional BFCL
contamination audit -> prompt-length selection. No model weights, embeddings,
or LLM judgments are used; the pinned Qwen tokenizer only renders prompts.
See docs/looptool_rl_preprocessing.md.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import shlex
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import atomic_output, file_sha256
from data_curation.looptool import (
    Reject,
    SchemaChecker,
    canonical_hash,
    canonical_json,
    check_row,
    example_hash,
    parse_example,
    prompt_hash,
    repair_schema,
    target_key,
)

DATASET = "zhangkangning/LoopTool-23k"
DATASET_REVISION = "b6c572d442ed4f2177f23645d8e9a77522e712c3"
EXPECTED_ROWS = 23_040
TOKENIZER = "Qwen/Qwen3-4B-Thinking-2507"
TOKENIZER_REVISION = "768f209d9ea81521153ed38c47d515654e938aea"
LENGTH_CANDIDATES = (2_048, 4_096, 8_192, 10_240, 12_288, 16_384)
RECOMMENDED_MAX_RESPONSE_TOKENS = 32_768
MODEL_NATIVE_CONTEXT_TOKENS = 262_144
DATASET_CARD = {"total": 23_040, "non_function_call": 2_848, "single_turn": 5_586, "multi_turn": 14_606}
MAIN_STRATA = ("single_turn", "multi_turn", "single_call", "parallel_call", "text_no_call")
PP_WARNING = 0.5
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "config.json",
]
DEPENDENCIES = ("datasets", "transformers", "tokenizers", "jinja2", "jsonschema", "datasketch", "numpy", "scipy")
DEPENDENCIES += ("huggingface-hub", "bfcl-eval")

ARTIFACTS = {
    "canonical": "looptool_rl_canonical.jsonl",
    "train": "looptool_rl_train.jsonl",
    "rejections": "looptool_rejections.jsonl",
    "conflicts": "looptool_conflicts.jsonl",
    "exact_duplicates": "looptool_exact_duplicates.jsonl",
    "duplicate_clusters": "looptool_duplicate_clusters.jsonl",
    "length_filtered": "looptool_length_filtered.jsonl",
    "events": "looptool_events.jsonl",
    "bfcl_overlap": "looptool_bfcl_overlap.jsonl",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS, help="0 disables the source-size check")
    parser.add_argument("--source-file", type=Path, help="Local JSONL/Parquet with instruction/input/output columns")
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/looptool_rl"))
    parser.add_argument("--overwrite", action="store_true", help="Replace artifacts in an existing output directory")
    parser.add_argument("--limit", type=int, help="Process only the first N source rows (smoke runs)")
    parser.add_argument("--audit-only", action="store_true", help="Write audits and summaries but no RL datasets")
    parser.add_argument(
        "--no-schema-type-alias-repair",
        dest="schema_type_alias_repair",
        action="store_false",
        help="Reject Python-style JSON Schema type names instead of mapping them",
    )
    parser.add_argument("--near-dup-threshold", type=float, default=0.95, help="Exact Jaccard needed for removal")
    parser.add_argument("--minhash-permutations", type=int, default=256)
    parser.add_argument("--lsh-threshold", type=float, default=0.8, help="MinHash-LSH candidate threshold")
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="MinHash seed")
    parser.add_argument("--max-prompt-tokens", type=int, help="Override the automatically selected prompt limit")
    parser.add_argument(
        "--length-candidates",
        default=",".join(map(str, LENGTH_CANDIDATES)),
        help="Comma-separated prompt-limit candidates for automatic selection",
    )
    parser.add_argument("--bfcl-audit", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--bfcl-data-dir", type=Path, help="BFCL data directory (implies --bfcl-audit on)")
    parser.add_argument("--bfcl-remove-jaccard", type=float, default=0.99)
    parser.add_argument("--bfcl-report-jaccard", type=float, default=0.8)
    parser.add_argument("--review-clusters", type=int, default=50, help="Near-duplicate clusters shown in Markdown")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--skip-output-validation", action="store_true")
    parser.add_argument("--test-log", type=Path, help="Optional pytest log whose result line is recorded")
    args = parser.parse_args(argv)
    if args.bfcl_data_dir is not None:
        if args.bfcl_audit == "off":
            parser.error("--bfcl-data-dir conflicts with --bfcl-audit off")
        args.bfcl_audit = "on"
    args.length_candidates = tuple(sorted(int(value) for value in args.length_candidates.split(",")))
    return args


# ---------------------------------------------------------------------------
# Source and tokenizer


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_source_rows(count: int, args: argparse.Namespace) -> None:
    if args.expected_rows and count != args.expected_rows:
        raise SystemExit(f"{args.dataset}@{args.dataset_revision} has {count} rows; expected {args.expected_rows}.")


def load_source(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info: dict[str, Any] = {"dataset": args.dataset, "requested_revision": args.dataset_revision}
    if args.source_file is not None:
        from datasets import load_dataset

        kind = "parquet" if args.source_file.suffix == ".parquet" else "json"
        records = load_dataset(kind, data_files=str(args.source_file), split="train").to_list()
        info.update(resolved_revision=None, resolution="local file", source_file=str(args.source_file.resolve()))
        info["source_file_sha256"] = file_sha256(args.source_file)
    else:
        from datasets import load_dataset

        records = load_dataset(args.dataset, revision=args.dataset_revision, split="train").to_list()
        try:
            from huggingface_hub import HfApi

            info["resolved_revision"] = HfApi().dataset_info(args.dataset, revision=args.dataset_revision).sha
            info["resolution"] = "huggingface_hub.HfApi.dataset_info"
        except Exception as exc:  # noqa: BLE001 - offline cache use is allowed but recorded.
            info["resolved_revision"] = None
            info["resolution"] = f"unresolved: {type(exc).__name__}"
    info["source_rows"] = len(records)
    info["source_content_sha256"] = canonical_hash(
        [[row.get("instruction"), row.get("input"), row.get("output")] for row in records]
    )
    check_source_rows(len(records), args)
    if args.limit is not None:
        records = records[: args.limit]
    info["processed_rows"] = len(records)
    return records, info


def resolve_tokenizer(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """Download only tokenizer files and return their local snapshot directory."""
    from huggingface_hub import snapshot_download

    kwargs = {"repo_id": args.tokenizer, "revision": args.tokenizer_revision, "allow_patterns": TOKENIZER_FILES}
    try:
        path = snapshot_download(**kwargs)
        resolution = "huggingface_hub.snapshot_download"
    except Exception:  # noqa: BLE001 - fall back to an already cached snapshot.
        path = snapshot_download(**kwargs, local_files_only=True)
        resolution = "huggingface_hub.snapshot_download(local_files_only=True)"
    resolved = Path(path).name if Path(path).parent.name == "snapshots" else None
    return path, {
        "tokenizer": args.tokenizer,
        "requested_revision": args.tokenizer_revision,
        "resolved_revision": resolved,
        "resolution": resolution,
        "files": {name: file_sha256(Path(path) / name) for name in TOKENIZER_FILES if (Path(path) / name).exists()},
    }


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


# ---------------------------------------------------------------------------
# Parsing, conflicts, exact dedup


def parse_stage(records: list[dict[str, Any]], args: argparse.Namespace, revision: str):
    checker = SchemaChecker()
    examples, rejections, events = [], [], []
    for index, record in enumerate(records):
        try:
            example = parse_example(
                record,
                index=index,
                dataset=args.dataset,
                revision=revision,
                checker=checker,
                type_aliases=args.schema_type_alias_repair,
            )
        except Reject as reject:
            from data_curation.looptool import source_row_id

            rejections.append(
                {
                    "id": source_row_id(record, dataset=args.dataset, revision=revision, index=index),
                    "source_index": index,
                    "stage": reject.stage,
                    "reason": reject.reason,
                    "details": reject.details,
                }
            )
            continue
        examples.append(example.row)
        for event in example.events:
            events.append(
                {
                    "id": example.row["id"],
                    "source_index": index,
                    "kind": event.kind,
                    "code": event.code,
                    "details": event.details[:300],
                }
            )
    return examples, rejections, events


def find_conflicts(rows: list[dict[str, Any]]):
    """Identical tools+messages with materially different targets are all excluded."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[prompt_hash(row)].append(row)
    conflicted: set[str] = set()
    records, group_summaries = [], []
    for key in sorted(groups, key=lambda key: groups[key][0]["metadata"]["source_index"]):
        members = groups[key]
        targets = {target_key(row["target"]) for row in members}
        if len(targets) < 2:
            continue
        group_id = "conflict:" + key[:16]
        group_summaries.append(
            {
                "group_id": group_id,
                "rows": len(members),
                "distinct_targets": len(targets),
                "target_kinds": sorted({row["metadata"]["target_kind"] for row in members}),
                "call_names": sorted(
                    {call["function"]["name"] for row in members for call in row["target"]["tool_calls"]}
                ),
            }
        )
        for row in members:
            conflicted.add(row["id"])
            records.append(
                {"conflict_group": group_id, "prompt_hash": key, "target_key": target_key(row["target"]), "row": row}
            )
    return [row for row in rows if row["id"] not in conflicted], records, group_summaries


def exact_dedup(rows: list[dict[str, Any]]):
    representatives: dict[str, dict[str, Any]] = {}
    kept, removed = [], []
    for row in sorted(rows, key=lambda row: row["metadata"]["source_index"]):
        digest = example_hash(row)
        if digest in representatives:
            keeper = representatives[digest]
            removed.append(
                {
                    "removed_id": row["id"],
                    "removed_source_index": row["metadata"]["source_index"],
                    "representative_id": keeper["id"],
                    "representative_source_index": keeper["metadata"]["source_index"],
                    "canonical_hash": digest,
                }
            )
            continue
        representatives[digest] = row
        kept.append(row)
    return kept, removed


# ---------------------------------------------------------------------------
# Tokenizer rendering


def _ids(value: Any) -> list[int]:
    if isinstance(value, dict) or hasattr(value, "keys"):
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def target_message(target: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": target["content"], "tool_calls": target["tool_calls"]}


def measure_row(tokenizer, row: dict[str, Any]) -> dict[str, Any]:
    """Render the exact initial rollout prompt and check the target can be appended."""
    messages, tools = row["messages"], row["tools"]
    prompt = _ids(tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True))
    head = [message for message in messages[:1] if message["role"] == "system"]
    system_tool = (
        len(_ids(tokenizer.apply_chat_template(head, tools=tools, tokenize=True, add_generation_prompt=False)))
        if head or tools
        else 0
    )
    base = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
    full = tokenizer.apply_chat_template(
        messages + [target_message(row["target"])], tools=tools, tokenize=False, add_generation_prompt=False
    )
    error = None
    if not full.startswith(base):
        error = "target rendering changes the prompt prefix"
    else:
        suffix = full[len(base) :]
        names = [call["function"]["name"] for call in row["target"]["tool_calls"]]
        if not suffix.strip() or any(name not in suffix for name in names):
            error = "rendered target is missing content or call names"
    return {
        "prompt_tokens": len(prompt),
        "system_tool_tokens": system_tool,
        "history_tokens": len(prompt) - system_tool,
        "error": error,
    }


_WORKER_TOKENIZER = None


def _init_worker(path: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = load_tokenizer(path)


def _measure_worker(row: dict[str, Any]) -> dict[str, Any]:
    try:
        return measure_row(_WORKER_TOKENIZER, row)
    except Exception as exc:  # noqa: BLE001 - template failures become rejections.
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}


def measure_all(rows: list[dict[str, Any]], tokenizer, tokenizer_path: str | None, workers: int) -> list[dict]:
    if tokenizer is not None or workers <= 1 or len(rows) < 64:
        tokenizer = tokenizer if tokenizer is not None else load_tokenizer(tokenizer_path)
        results = []
        for row in rows:
            try:
                results.append(measure_row(tokenizer, row))
            except Exception as exc:  # noqa: BLE001
                results.append({"error": f"{type(exc).__name__}: {exc}"[:300]})
        return results
    import multiprocessing

    with multiprocessing.get_context("spawn").Pool(workers, _init_worker, (tokenizer_path,)) as pool:
        return list(pool.imap(_measure_worker, rows, chunksize=32))


def measure_stage(rows, tokenizer, tokenizer_path, workers):
    kept, rejections, lengths = [], [], {}
    for row, result in zip(rows, measure_all(rows, tokenizer, tokenizer_path, workers)):
        if result.get("error"):
            rejections.append(
                {
                    "id": row["id"],
                    "source_index": row["metadata"]["source_index"],
                    "stage": "render",
                    "reason": "chat_template_render_failed"
                    if "prompt_tokens" not in result
                    else "target_render_invalid",
                    "details": result["error"],
                }
            )
            continue
        row["metadata"]["prompt_tokens"] = result["prompt_tokens"]
        lengths[row["id"]] = result
        kept.append(row)
    return kept, rejections, lengths


# ---------------------------------------------------------------------------
# Near deduplication


def words(text: str) -> list[str]:
    return unicodedata.normalize("NFKC", text).casefold().split()


def shingles(tokens: list[str], size: int) -> set[str]:
    return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def dialogue_text(row: dict[str, Any]) -> str:
    """Role-labelled natural-language content, historical calls, and the target."""
    lines = []
    for message in row["messages"]:
        if message["role"] == "system":
            continue
        if message.get("content"):
            lines.append(f"{message['role']}: {message['content']}")
        for call in message.get("tool_calls", []):
            lines.append(f"assistant_call: {call['function']['name']} {canonical_json(call['function']['arguments'])}")
    if row["target"]["content"]:
        lines.append(f"target: {row['target']['content']}")
    for call in row["target"]["tool_calls"]:
        lines.append(f"target_call: {call['function']['name']} {canonical_json(call['function']['arguments'])}")
    return "\n".join(lines)


def stratum(row: dict[str, Any]) -> dict[str, Any]:
    system = next((message["content"] for message in row["messages"] if message["role"] == "system"), "")
    return {
        "tool_schema_fingerprint": canonical_hash(row["tools"])[:16],
        "system_text_fingerprint": canonical_hash(system)[:16],
        "role_sequence": "".join(message["role"][0] for message in row["messages"]),
        "history_call_names": [
            call["function"]["name"] for message in row["messages"] for call in message.get("tool_calls", [])
        ],
        "target_kind": row["metadata"]["target_kind"],
        "target_call_count": len(row["target"]["tool_calls"]),
        "target_call_names": sorted(call["function"]["name"] for call in row["target"]["tool_calls"]),
    }


def jaccard(left: set[str], right: set[str]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def _excerpt(row: dict[str, Any], width: int = 160) -> dict[str, str]:
    user = next((message["content"] for message in reversed(row["messages"]) if message["role"] == "user"), "")
    target = row["target"]["content"] or "; ".join(
        f"{call['function']['name']}({canonical_json(call['function']['arguments'])})"
        for call in row["target"]["tool_calls"]
    )
    clip = lambda text: " ".join(text.split())[:width]  # noqa: E731
    return {"last_user": clip(user), "target": clip(target)}


def near_dedup(rows: list[dict[str, Any]], *, threshold, num_perm, lsh_threshold, seed, shingle_size):
    """Greedy, representative-anchored clustering inside exact structural strata.

    LSH proposes candidates; a row is removed only if its exact shingle Jaccard
    against the retained representative meets ``threshold``. Removed rows never
    become representatives, so there is no transitive chain collapse.
    """
    from datasketch import MinHash, MinHashLSH

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stratum_of: dict[str, dict[str, Any]] = {}
    too_short = 0
    for row in rows:
        tokens = words(dialogue_text(row))
        if len(tokens) < shingle_size:
            too_short += 1
            continue
        key = stratum(row)
        stratum_of[row["id"]] = key
        strata[canonical_hash(key)].append(row)
    removed: set[str] = set()
    clusters = []
    candidate_pairs = verified_pairs = 0
    for key in sorted(strata):
        members = strata[key]
        if len(members) < 2:
            continue
        members.sort(
            key=lambda row: (
                row["metadata"]["prompt_tokens"],
                len(canonical_json(row)),
                row["metadata"]["source_index"],
            )
        )
        sets = {row["id"]: shingles(words(dialogue_text(row)), shingle_size) for row in members}
        lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
        hashes = {}
        for row in members:
            minhash = MinHash(num_perm=num_perm, seed=seed)
            minhash.update_batch([item.encode("utf-8") for item in sorted(sets[row["id"]])])
            hashes[row["id"]] = minhash
            lsh.insert(row["id"], minhash)
        order = {row["id"]: position for position, row in enumerate(members)}
        by_id = {row["id"]: row for row in members}
        for representative in members:
            if representative["id"] in removed:
                continue
            candidates = sorted(
                (
                    candidate
                    for candidate in lsh.query(hashes[representative["id"]])
                    if candidate != representative["id"]
                ),
                key=order.__getitem__,
            )
            cluster = []
            for candidate in candidates:
                if candidate in removed or order[candidate] < order[representative["id"]]:
                    continue
                candidate_pairs += 1
                score = jaccard(sets[representative["id"]], sets[candidate])
                if score >= threshold:
                    verified_pairs += 1
                    removed.add(candidate)
                    cluster.append((candidate, score))
            if cluster:
                member_ids = sorted(candidate for candidate, _ in cluster)
                clusters.append(
                    {
                        "cluster_id": "near:" + canonical_hash([representative["id"], member_ids])[:16],
                        "representative": {
                            "id": representative["id"],
                            "source_index": representative["metadata"]["source_index"],
                            "prompt_tokens": representative["metadata"]["prompt_tokens"],
                            "canonical_bytes": len(canonical_json(representative).encode("utf-8")),
                            "excerpt": _excerpt(representative),
                        },
                        "removed": [
                            {
                                "id": candidate,
                                "source_index": by_id[candidate]["metadata"]["source_index"],
                                "jaccard": round(score, 6),
                                "excerpt": _excerpt(by_id[candidate]),
                            }
                            for candidate, score in cluster
                        ],
                        "stratum": stratum_of[representative["id"]],
                    }
                )
    clusters.sort(key=lambda cluster: cluster["representative"]["source_index"])
    stats = {
        "strata_compared": sum(1 for members in strata.values() if len(members) >= 2),
        "rows_in_multi_member_strata": sum(len(members) for members in strata.values() if len(members) >= 2),
        "rows_below_min_words": too_short,
        "lsh_candidate_pairs_checked": candidate_pairs,
        "verified_pairs_removed": verified_pairs,
    }
    return [row for row in rows if row["id"] not in removed], clusters, stats


# ---------------------------------------------------------------------------
# BFCL contamination audit


def tool_signature(functions: list[dict[str, Any]]) -> str:
    items = []
    for function in functions:
        parameters = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
        items.append(canonical_json({"name": function.get("name"), "parameters": repair_schema(parameters)[0]}))
    return canonical_hash(sorted(items))


def _read_json_records(path: Path) -> list[dict[str, Any]] | None:
    text = path.read_text(encoding="utf-8")
    try:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, list) else None


def load_bfcl(data_dir: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info: dict[str, Any] = {"bfcl_eval_version": _version("bfcl-eval")}
    func_docs: dict[str, list[dict[str, Any]]] = {}
    mapping: dict[str, str] = {}
    if data_dir is None:
        import bfcl_eval

        data_dir = Path(bfcl_eval.__file__).parent / "data"
        info["source"] = "bfcl-eval package data"
    else:
        info["source"] = "--bfcl-data-dir"
    try:
        from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING as mapping
    except Exception:  # noqa: BLE001 - raw directories may not ship the package.
        mapping = {}
    info["data_dir"] = str(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"BFCL data directory does not exist: {data_dir}")
    entries, files, skipped = [], [], []
    for path in sorted(data_dir.glob("BFCL_*.json")):
        records = _read_json_records(path)
        if records is None or not all(isinstance(item, dict) and "question" in item for item in records):
            reason = (
                "no question entries (category -> id index)"
                if records
                and all(
                    isinstance(item, dict) and all(isinstance(value, list) for value in item.values())
                    for item in records
                )
                else "unrecognized entry format"
            )
            skipped.append({"file": path.name, "reason": reason})
            continue
        files.append({"file": path.name, "entries": len(records), "sha256": file_sha256(path)})
        category = path.stem.split("_", 2)[-1]
        for item in records:
            # A listed (possibly empty) inventory has a signature; one that cannot be
            # resolved from the func-doc mapping is unknown and never matches.
            functions = item.get("function")
            if functions is None and item.get("involved_classes"):
                classes = item["involved_classes"]
                if all(class_name in mapping for class_name in classes):
                    functions = []
                    for class_name in classes:
                        if class_name not in func_docs:
                            path = data_dir / "multi_turn_func_doc" / mapping[class_name]
                            func_docs[class_name] = _read_json_records(path) or []
                        functions.extend(
                            doc
                            for doc in func_docs[class_name]
                            if doc.get("name") not in item.get("excluded_function", [])
                        )
            known = isinstance(functions, list)
            functions = [function for function in functions or [] if isinstance(function, dict)]
            user_messages = [
                str(message.get("content", ""))
                for turn in item["question"]
                for message in (turn if isinstance(turn, list) else [turn])
                if isinstance(message, dict)
                and message.get("role") == "user"
                and str(message.get("content", "")).strip()
            ]
            entries.append(
                {
                    "bfcl_id": str(item.get("id")),
                    "file": path.name,
                    "category": category,
                    "user_messages": user_messages,
                    "signature": tool_signature(functions) if known else None,
                    "tool_names": sorted({str(function.get("name")) for function in functions}),
                }
            )
    if not entries:
        raise ValueError(f"no BFCL_*.json question files were loaded from {data_dir}")
    info.update(
        files=files,
        skipped_files=skipped,
        categories=sorted({entry["category"] for entry in entries}),
        splits="all BFCL_* question files at the data-directory root (possible_answer not read)",
        benchmark_prompts=len(entries),
    )
    return entries, info


def bfcl_audit(rows, entries, *, shingle_size, num_perm, seed, lsh_threshold, remove_jaccard, report_jaccard):
    from datasketch import MinHash, MinHashLSH

    def norm(text: str) -> str:
        return " ".join(words(text))

    def minhash(items: set[str]) -> MinHash:
        value = MinHash(num_perm=num_perm, seed=seed)
        value.update_batch([item.encode("utf-8") for item in sorted(items)])
        return value

    exact: dict[str, list[int]] = defaultdict(list)
    single: dict[str, list[int]] = defaultdict(list)
    bench_sets: dict[int, set[str]] = {}
    lsh = MinHashLSH(threshold=min(lsh_threshold, report_jaccard), num_perm=num_perm)
    for position, entry in enumerate(entries):
        prompt = norm("\n".join(entry["user_messages"]))
        if not prompt:
            continue
        exact[prompt].append(position)
        for message in entry["user_messages"]:
            if len(words(message)) >= shingle_size:
                single[norm(message)].append(position)
        items = shingles(prompt.split(), shingle_size)
        if items:
            bench_sets[position] = items
            lsh.insert(str(position), minhash(items))

    records, removed = [], set()
    for row in rows:
        users = [message["content"] for message in row["messages"] if message["role"] == "user"]
        prompt = norm("\n".join(users))
        functions = [tool["function"] for tool in row["tools"]]
        signature = tool_signature(functions)
        names = {function["name"] for function in functions}
        items = shingles(prompt.split(), shingle_size)
        matches: dict[int, dict[str, Any]] = {}
        for position in exact.get(prompt, []):
            matches[position] = {"prompt_exact": True, "prompt_jaccard": 1.0}
        if items:
            for key in lsh.query(minhash(items)):
                position = int(key)
                score = jaccard(items, bench_sets[position])
                if score >= report_jaccard and position not in matches:
                    matches[position] = {"prompt_exact": False, "prompt_jaccard": score}
        for message in users:
            for position in single.get(norm(message), []) if len(words(message)) >= shingle_size else []:
                matches.setdefault(
                    position,
                    {"prompt_exact": False, "prompt_jaccard": jaccard(items, bench_sets.get(position, set()))},
                )
                matches[position]["user_message_exact"] = True
        for position in sorted(matches):
            entry, match = entries[position], matches[position]
            schema_equal = signature is not None and signature == entry["signature"]
            shared = sorted(names & set(entry["tool_names"]))
            union = names | set(entry["tool_names"])
            if match["prompt_exact"] and schema_equal:
                disposition, reason = "removed", "exact_prompt_and_tool_schema"
            elif match["prompt_jaccard"] >= remove_jaccard and schema_equal:
                disposition, reason = "removed", f"prompt_jaccard>={remove_jaccard}_and_tool_schema"
            elif match["prompt_exact"]:
                disposition, reason = "audit_only", "exact_prompt_tool_schema_differs"
            elif match.get("user_message_exact"):
                disposition, reason = "audit_only", "user_message_exact_overlap"
            else:
                disposition, reason = "audit_only", f"prompt_jaccard>={report_jaccard}"
            if disposition == "removed":
                removed.add(row["id"])
            records.append(
                {
                    "id": row["id"],
                    "source_index": row["metadata"]["source_index"],
                    "bfcl_id": entry["bfcl_id"],
                    "bfcl_file": entry["file"],
                    "bfcl_category": entry["category"],
                    "prompt_exact": match["prompt_exact"],
                    "prompt_jaccard": round(match["prompt_jaccard"], 6),
                    "user_message_exact": bool(match.get("user_message_exact")),
                    "tool_schema_signature_equal": schema_equal,
                    "tool_name_jaccard": round(len(shared) / len(union), 6) if union else 0.0,
                    "shared_tool_names": shared[:20],
                    "disposition": disposition,
                    "reason": reason,
                }
            )
    return [row for row in rows if row["id"] not in removed], records


# ---------------------------------------------------------------------------
# Distributions and prompt lengths


def _bucket(value: int, edges: tuple[int, ...]) -> str:
    for low, high in zip(edges, edges[1:]):
        if low <= value < high:
            return str(low) if high == low + 1 else f"{low}-{high - 1}"
    return f"{edges[-1]}+"


USER_BUCKETS = (0, 1, 2, 3, 4, 6, 11)
HISTORY_BUCKETS = (0, 1, 2, 3, 6, 11)
TARGET_BUCKETS = (0, 1, 2, 3, 4, 6)


def history_bucket(metadata: dict[str, Any]) -> str:
    return _bucket(metadata["num_history_tool_calls"], HISTORY_BUCKETS)


def _bucket_labels(edges: tuple[int, ...]) -> list[str]:
    return [_bucket(low, edges) for low in edges]


def _share(counter: Counter, total: int, order: list[str] | None = None) -> dict[str, dict[str, float]]:
    keys = [key for key in order if key in counter] if order else sorted(counter)
    return {
        key: {"count": counter[key], "pct": round(100 * counter[key] / total, 4) if total else 0.0} for key in keys
    }


def distribution(metas: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(metas)
    conversation = Counter(meta["conversation_kind"] for meta in metas)
    target = Counter(meta["target_kind"] for meta in metas)
    for key in ("single_turn", "multi_turn"):
        conversation.setdefault(key, 0)
    for key in ("single_call", "parallel_call", "text_no_call"):
        target.setdefault(key, 0)
    crosstab = Counter((meta["conversation_kind"], meta["target_kind"]) for meta in metas)
    return {
        "total": total,
        "conversation_kind": _share(conversation, total),
        "target_kind": _share(target, total),
        "crosstab": {f"{left}|{right}": count for (left, right), count in sorted(crosstab.items())},
        "user_message_buckets": _share(
            Counter(_bucket(meta["num_user_messages"], USER_BUCKETS) for meta in metas),
            total,
            _bucket_labels(USER_BUCKETS),
        ),
        "history_call_buckets": _share(
            Counter(history_bucket(meta) for meta in metas), total, _bucket_labels(HISTORY_BUCKETS)
        ),
        "target_call_buckets": _share(
            Counter(_bucket(meta["num_target_tool_calls"], TARGET_BUCKETS) for meta in metas),
            total,
            _bucket_labels(TARGET_BUCKETS),
        ),
    }


def main_pcts(dist: dict[str, Any]) -> dict[str, float]:
    return {
        key: dist["conversation_kind" if key in ("single_turn", "multi_turn") else "target_kind"][key]["pct"]
        for key in MAIN_STRATA
    }


def raw_source_metas(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pre-parse view from raw markers only (no role/call parsing)."""
    metas = []
    for record in records:
        text, output = str(record.get("input", "")), str(record.get("output", ""))
        calls = output.count("<tool_call>")
        plain = "<|im_start|>" not in text and "<|im_end|>" not in text
        metas.append(
            {
                "conversation_kind": "single_turn" if plain else "multi_turn",
                "target_kind": "parallel_call" if calls >= 2 else "single_call" if calls == 1 else "text_no_call",
                "num_user_messages": 1
                + text.count("<|im_start|>user")
                - text.count("<|im_start|>user\n<tool_response>"),
                "num_history_tool_calls": text.count("<tool_call>"),
                "num_target_tool_calls": calls,
            }
        )
    return metas


def percentile(values: list[int], pct: float) -> int | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def length_stats(values: list[int], thresholds: tuple[int, ...]) -> dict[str, Any]:
    total = len(values)
    return {
        "count": total,
        **{f"p{pct}": percentile(values, pct) for pct in (50, 75, 90, 95, 99)},
        "max": max(values) if values else None,
        "mean": round(sum(values) / total, 2) if total else None,
        "at_or_below": {
            str(limit): {
                "count": sum(value <= limit for value in values),
                "pct": round(100 * sum(value <= limit for value in values) / total, 4) if total else 0.0,
                "exceeding": sum(value > limit for value in values),
            }
            for limit in thresholds
        },
    }


def length_report(rows: list[dict[str, Any]], lengths: dict[str, dict], thresholds: tuple[int, ...]) -> dict[str, Any]:
    tokens = [row["metadata"]["prompt_tokens"] for row in rows]
    groups: dict[str, dict[str, list[int]]] = {
        "conversation_kind": defaultdict(list),
        "target_kind": defaultdict(list),
    }
    groups["history_call_bucket"] = defaultdict(list)
    for row in rows:
        meta = row["metadata"]
        groups["conversation_kind"][meta["conversation_kind"]].append(meta["prompt_tokens"])
        groups["target_kind"][meta["target_kind"]].append(meta["prompt_tokens"])
        groups["history_call_bucket"][history_bucket(meta)].append(meta["prompt_tokens"])
    system = [lengths[row["id"]]["system_tool_tokens"] for row in rows]
    history = [lengths[row["id"]]["history_tokens"] for row in rows]
    contribution = {"all": {"system_tool": length_stats(system, ()), "dialogue_history": length_stats(history, ())}}
    for limit in thresholds:
        above = [row for row in rows if row["metadata"]["prompt_tokens"] > limit]
        contribution[f">{limit}"] = {
            "rows": len(above),
            "median_system_tool_tokens": percentile([lengths[row["id"]]["system_tool_tokens"] for row in above], 50),
            "median_history_tokens": percentile([lengths[row["id"]]["history_tokens"] for row in above], 50),
            "rows_where_system_tool_exceeds_history": sum(
                lengths[row["id"]]["system_tool_tokens"] > lengths[row["id"]]["history_tokens"] for row in above
            ),
        }
    total_tokens = sum(tokens) or 1
    contribution["token_share_pct"] = {
        "system_tool": round(100 * sum(system) / total_tokens, 3),
        "dialogue_history": round(100 * sum(history) / total_tokens, 3),
    }
    return {
        "overall": length_stats(tokens, thresholds),
        "by": {
            name: {
                key: length_stats(group[key], thresholds)
                for key in (_bucket_labels(HISTORY_BUCKETS) if name == "history_call_bucket" else sorted(group))
                if key in group
            }
            for name, group in groups.items()
        },
        "contribution": contribution,
    }


def evaluate_limits(rows: list[dict[str, Any]], candidates: tuple[int, ...]) -> list[dict[str, Any]]:
    metas = [row["metadata"] for row in rows]
    before_pct = main_pcts(distribution(metas))
    before_counts = {key: sum(meta[_axis(key)] == key for meta in metas) for key in MAIN_STRATA}
    results = []
    for limit in candidates:
        kept = [meta for meta in metas if meta["prompt_tokens"] <= limit]
        after_pct = main_pcts(distribution(kept))
        retention = {
            key: round(100 * sum(meta[_axis(key)] == key for meta in kept) / before_counts[key], 4)
            if before_counts[key]
            else 100.0
            for key in MAIN_STRATA
        }
        shift = {key: round(after_pct[key] - before_pct[key], 4) for key in MAIN_STRATA}
        overall = round(100 * len(kept) / len(rows), 4) if rows else 100.0
        checks = {
            "overall_retention>=95%": overall >= 95.0,
            "each_stratum_retention>=90%": all(value >= 90.0 for value in retention.values()),
            f"max_abs_pp_shift<={PP_WARNING}": all(abs(value) <= PP_WARNING for value in shift.values()),
        }
        results.append(
            {
                "limit": limit,
                "retained": len(kept),
                "excluded": len(rows) - len(kept),
                "overall_retention_pct": overall,
                "stratum_retention_pct": retention,
                "pp_shift": shift,
                "checks": checks,
                "satisfies_all": all(checks.values()),
            }
        )
    return results


def _axis(key: str) -> str:
    return "conversation_kind" if key in ("single_turn", "multi_turn") else "target_kind"


def select_limit(evaluations: list[dict[str, Any]], override: int | None) -> dict[str, Any]:
    passing = [item for item in evaluations if item["satisfies_all"]]
    fallback = max(item["limit"] for item in evaluations)
    automatic = passing[0]["limit"] if passing else fallback
    misses = []
    if not passing:
        misses = [
            name
            for name, ok in next(item for item in evaluations if item["limit"] == fallback)["checks"].items()
            if not ok
        ]
    return {
        "automatic": automatic,
        "automatic_reason": (
            f"smallest candidate satisfying all criteria: {automatic}"
            if passing
            else f"no candidate satisfies all criteria; fallback {fallback} misses {misses}"
        ),
        "fallback_misses": misses,
        "override": override,
        "effective": override if override is not None else automatic,
    }


# ---------------------------------------------------------------------------
# Output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with atomic_output(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(canonical_json(row))
                handle.write("\n")


def _event_summary(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        item = grouped.setdefault(event["code"], {"kind": event["kind"], "events": 0, "rows": set(), "sample_ids": []})
        item["events"] += 1
        if event["id"] not in item["rows"]:
            item["rows"].add(event["id"])
            if len(item["sample_ids"]) < 5:
                item["sample_ids"].append(f"{event['source_index']}:{event['id']}")
    return {
        code: {
            "kind": item["kind"],
            "events": item["events"],
            "rows": len(item["rows"]),
            "sample_ids": item["sample_ids"],
        }
        for code, item in sorted(grouped.items())
    }


def _reason_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        item = grouped.setdefault(f"{record['stage']}/{record['reason']}", {"rows": 0, "sample_ids": []})
        item["rows"] += 1
        if len(item["sample_ids"]) < 5:
            item["sample_ids"].append(f"{record['source_index']}:{record['id']}")
    return dict(sorted(grouped.items()))


def validate_outputs(paths: list[Path], tokenizer, tokenizer_path, workers) -> dict[str, Any]:
    """Re-read written JSONL and check the canonical contract plus rendering."""
    report: dict[str, Any] = {}
    for path in paths:
        rows, problems = [], Counter()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    problems["json_parse_error"] += 1
                    continue
                for problem in check_row(row):
                    problems[problem] += 1
                rows.append(row)
        indices = [row["metadata"]["source_index"] for row in rows]
        render_errors = mismatched = 0
        for row, result in zip(rows, measure_all(rows, tokenizer, tokenizer_path, workers)):
            if result.get("error"):
                render_errors += 1
            elif result["prompt_tokens"] != row["metadata"]["prompt_tokens"]:
                mismatched += 1
        report[path.name] = {
            "rows": len(rows),
            "contract_problems": dict(problems),
            "sorted_by_source_index": indices == sorted(indices) and len(set(indices)) == len(indices),
            "prompt_render_or_target_append_errors": render_errors,
            "prompt_token_mismatches_vs_metadata": mismatched,
            "passed": not problems and not render_errors and not mismatched and indices == sorted(indices),
        }
    return report


def pp_changes(stages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    changes, warnings = [], []
    baseline = next(stage for stage in stages if stage["stage"] == "structural_validation")
    # The first step compares parsed rows with the raw-marker view of the source; the
    # target-kind axis is defined identically in both, the conversation axis approximately.
    previous = None
    for stage in stages:
        pct = main_pcts(stage["distribution"])
        if previous is not None:
            step = {key: round(pct[key] - main_pcts(previous["distribution"])[key], 4) for key in MAIN_STRATA}
            cumulative = {key: round(pct[key] - main_pcts(baseline["distribution"])[key], 4) for key in MAIN_STRATA}
            changes.append({"stage": stage["stage"], "vs_previous_pp": step, "vs_structural_pp": cumulative})
            for key, value in step.items():
                if abs(value) > PP_WARNING:
                    warnings.append(f"{stage['stage']}: {key} changed {value:+.3f} pp vs previous stage")
        previous = stage
    return changes, warnings


def run(args: argparse.Namespace, *, records=None, source_info=None, tokenizer=None, tokenizer_info=None) -> dict:
    output_dir = args.output_dir
    if (output_dir / "looptool_summary.json").exists() and not args.overwrite:
        raise SystemExit(f"{output_dir} already has outputs; pass --overwrite to replace them.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if records is None:
        records, source_info = load_source(args)
    tokenizer_path = None
    if tokenizer is None:
        tokenizer_path, tokenizer_info = resolve_tokenizer(args)
    revision = source_info.get("resolved_revision") or args.dataset_revision

    examples, rejections, events = parse_stage(records, args, revision)
    stages = [
        {"stage": "source_raw_markers", "rows": len(records), "distribution": distribution(raw_source_metas(records))},
        {
            "stage": "structural_validation",
            "rows": len(examples),
            "distribution": distribution([row["metadata"] for row in examples]),
        },
    ]
    strict_losses = Counter()
    alias_rows = {event["id"] for event in events if event["code"] == "schema_type_alias"}
    for row in examples:
        if row["id"] in alias_rows:
            strict_losses[row["metadata"]["conversation_kind"]] += 1
            strict_losses[row["metadata"]["target_kind"]] += 1

    rows, conflicts, conflict_groups = find_conflicts(examples)
    stages.append(
        {"stage": "conflict_removal", "rows": len(rows), "distribution": distribution([r["metadata"] for r in rows])}
    )
    rows, exact_removed = exact_dedup(rows)
    stages.append(
        {"stage": "exact_dedup", "rows": len(rows), "distribution": distribution([r["metadata"] for r in rows])}
    )

    rows, render_rejections, lengths = measure_stage(rows, tokenizer, tokenizer_path, args.workers)
    rejections.extend(render_rejections)
    stages.append(
        {"stage": "tokenizer_render", "rows": len(rows), "distribution": distribution([r["metadata"] for r in rows])}
    )
    rows, clusters, near_stats = near_dedup(
        rows,
        threshold=args.near_dup_threshold,
        num_perm=args.minhash_permutations,
        lsh_threshold=args.lsh_threshold,
        seed=args.seed,
        shingle_size=args.shingle_size,
    )
    stages.append(
        {"stage": "near_dedup", "rows": len(rows), "distribution": distribution([r["metadata"] for r in rows])}
    )

    bfcl: dict[str, Any] = {"requested": args.bfcl_audit}
    overlap_records: list[dict[str, Any]] | None = None
    if args.bfcl_audit == "off":
        bfcl.update(status="skipped", reason="--bfcl-audit off")
    else:
        try:
            entries, info = load_bfcl(args.bfcl_data_dir)
        except Exception as exc:  # noqa: BLE001
            if args.bfcl_audit == "on":
                raise SystemExit(f"BFCL audit was requested but BFCL data is unavailable: {exc}") from exc
            entries, info = None, {"reason": f"BFCL data unavailable: {type(exc).__name__}: {exc}"}
        bfcl.update(info)
        if entries is None:
            bfcl["status"] = "skipped"
        else:
            before = len(rows)
            rows, overlap_records = bfcl_audit(
                rows,
                entries,
                shingle_size=args.shingle_size,
                num_perm=args.minhash_permutations,
                seed=args.seed,
                lsh_threshold=args.lsh_threshold,
                remove_jaccard=args.bfcl_remove_jaccard,
                report_jaccard=args.bfcl_report_jaccard,
            )
            bfcl.update(
                status="completed",
                removed_rows=before - len(rows),
                audit_records=len(overlap_records),
                dispositions=dict(Counter(f"{item['disposition']}/{item['reason']}" for item in overlap_records)),
                audit_only_rows=len({item["id"] for item in overlap_records if item["disposition"] == "audit_only"}),
                remove_rule=f"exact prompt+tool schema, or prompt Jaccard>={args.bfcl_remove_jaccard} with equal tool schema",
            )
    stages.append(
        {
            "stage": "contamination_filter",
            "rows": len(rows),
            "distribution": distribution([r["metadata"] for r in rows]),
        }
    )

    canonical = sorted(rows, key=lambda row: row["metadata"]["source_index"])
    lengths_report = length_report(canonical, lengths, args.length_candidates)
    evaluations = evaluate_limits(canonical, args.length_candidates)
    if args.max_prompt_tokens is not None and args.max_prompt_tokens not in args.length_candidates:
        evaluations += evaluate_limits(canonical, (args.max_prompt_tokens,))
    limit = select_limit(
        [item for item in evaluations if item["limit"] in args.length_candidates], args.max_prompt_tokens
    )
    train = [row for row in canonical if row["metadata"]["prompt_tokens"] <= limit["effective"]]
    length_filtered = [
        {
            "id": row["id"],
            "source_index": row["metadata"]["source_index"],
            "prompt_tokens": row["metadata"]["prompt_tokens"],
            "effective_limit": limit["effective"],
            "conversation_kind": row["metadata"]["conversation_kind"],
            "target_kind": row["metadata"]["target_kind"],
            "num_history_tool_calls": row["metadata"]["num_history_tool_calls"],
        }
        for row in canonical
        if row["metadata"]["prompt_tokens"] > limit["effective"]
    ]
    stages.append(
        {
            "stage": "prompt_length_filter",
            "rows": len(train),
            "distribution": distribution([r["metadata"] for r in train]),
        }
    )
    changes, distribution_warnings = pp_changes(stages)

    rejections.sort(key=lambda item: item["source_index"])
    events.sort(key=lambda item: item["source_index"])
    outputs = {
        "rejections": rejections,
        "conflicts": conflicts,
        "exact_duplicates": exact_removed,
        "duplicate_clusters": clusters,
        "events": events,
    }
    if not args.audit_only:
        outputs.update(canonical=canonical, train=train, length_filtered=length_filtered)
    if overlap_records is not None:
        outputs["bfcl_overlap"] = overlap_records
    for name, value in outputs.items():
        write_jsonl(output_dir / ARTIFACTS[name], value)
    for name, filename in ARTIFACTS.items():
        if name not in outputs and (output_dir / filename).exists():
            (output_dir / filename).unlink()

    validation = None
    if not args.audit_only and not args.skip_output_validation:
        validation = validate_outputs(
            [output_dir / ARTIFACTS["canonical"], output_dir / ARTIFACTS["train"]],
            tokenizer,
            tokenizer_path,
            args.workers,
        )
    test_result = None
    if args.test_log is not None and args.test_log.exists():
        lines = [line.strip() for line in args.test_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        test_result = next((line for line in reversed(lines) if " passed" in line or " failed" in line), None)

    options = {key: (str(value) if isinstance(value, Path) else value) for key, value in sorted(vars(args).items())}
    options["length_candidates"] = list(args.length_candidates)
    summary = {
        "pipeline": "data_curation/prepare_looptool_rl.py",
        "command": "python " + " ".join(shlex.quote(part) for part in [sys.argv[0], *sys.argv[1:]]),
        "arguments": options,
        "source": source_info,
        "tokenizer": tokenizer_info,
        "environment": {
            "python": platform.python_version(),
            "dependencies": {name: _version(name) for name in DEPENDENCIES},
        },
        "stage_counts": {stage["stage"]: stage["rows"] for stage in stages},
        "stages": stages,
        "distribution_changes": changes,
        "distribution_warnings": distribution_warnings,
        "dataset_card_comparison": _card_comparison(stages),
        "rejections": {"total": len(rejections), "by_reason": _reason_summary(rejections)},
        "repairs_and_warnings": _event_summary(events),
        "strict_schema_counterfactual": {
            "description": "Rows kept only because of the schema_type_alias repair (rejected under --no-schema-type-alias-repair)",
            "rows": len(alias_rows),
            "by_stratum": dict(sorted(strict_losses.items())),
        },
        "conflicts": {
            "groups": len(conflict_groups),
            "affected_rows": len(conflicts),
            "target_kind_sets": dict(Counter("+".join(group["target_kinds"]) for group in conflict_groups)),
            "call_names": dict(
                Counter(name for group in conflict_groups for name in group["call_names"]).most_common(30)
            ),
            "sample_groups": conflict_groups[:10],
        },
        "exact_dedup": {"removed": len(exact_removed), "sample": exact_removed[:10]},
        "near_dedup": {
            "threshold": args.near_dup_threshold,
            "minhash_permutations": args.minhash_permutations,
            "lsh_threshold": args.lsh_threshold,
            "seed": args.seed,
            "shingle_size": args.shingle_size,
            "clusters": len(clusters),
            "removed": sum(len(cluster["removed"]) for cluster in clusters),
            **near_stats,
            "review_clusters": sorted(clusters, key=lambda cluster: cluster["cluster_id"])[: args.review_clusters],
        },
        "bfcl_audit": bfcl,
        "prompt_lengths": lengths_report,
        "prompt_limit": {**limit, "candidates": evaluations},
        "rollout_budget": {
            "max_prompt_tokens": limit["effective"],
            "max_response_tokens": RECOMMENDED_MAX_RESPONSE_TOKENS,
            "minimum_combined_context_tokens": limit["effective"] + RECOMMENDED_MAX_RESPONSE_TOKENS,
            "model_native_context_tokens": MODEL_NATIVE_CONTEXT_TOKENS,
        },
        "output_validation": validation,
        "test_result": test_result,
        "artifacts": {
            name: {
                "path": str((output_dir / ARTIFACTS[name]).resolve()),
                "rows": len(value),
                "sha256": file_sha256(output_dir / ARTIFACTS[name]),
            }
            for name, value in outputs.items()
        },
    }
    from data_curation.common import write_json

    write_json(summary, output_dir / "looptool_summary.json")
    with atomic_output(output_dir / "looptool_summary.md") as temporary:
        temporary.write_text(render_markdown(summary), encoding="utf-8")
    return summary


def _card_comparison(stages: list[dict[str, Any]]) -> dict[str, Any]:
    raw = next(stage for stage in stages if stage["stage"] == "source_raw_markers")["distribution"]
    parsed = next(stage for stage in stages if stage["stage"] == "structural_validation")["distribution"]

    def function_call_split(dist: dict[str, Any]) -> dict[str, int]:
        cross = dist["crosstab"]
        return {
            "total": dist["total"],
            "non_function_call": dist["target_kind"]["text_no_call"]["count"],
            "single_turn": sum(cross.get(f"single_turn|{kind}", 0) for kind in ("single_call", "parallel_call")),
            "multi_turn": sum(cross.get(f"multi_turn|{kind}", 0) for kind in ("single_call", "parallel_call")),
        }

    return {
        "dataset_card": DATASET_CARD,
        "raw_markers": function_call_split(raw),
        "parsed": function_call_split(parsed),
        "note": (
            "Card categories are read as non-function-call / single-turn function-call / multi-turn function-call "
            "(they sum to 23,040). raw_markers: single_turn = input without <|im_start|>/<|im_end|>, call count = "
            "number of <tool_call> tags in output. parsed: single_turn = exactly one user message and no historical "
            "tool calls. No resampling is done to match the card."
        ),
    }


# ---------------------------------------------------------------------------
# Markdown


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(summary: dict[str, Any]) -> str:
    out = ["# LoopTool-23k RL preprocessing summary", ""]
    source, tokenizer = summary["source"], summary["tokenizer"]
    out += [
        "## Source and tokenizer",
        "",
        _table(
            ["item", "value"],
            [
                ["dataset", source["dataset"]],
                ["requested revision", source["requested_revision"]],
                ["resolved revision", source.get("resolved_revision")],
                ["source rows", source["source_rows"]],
                ["processed rows", source["processed_rows"]],
                ["source content sha256", source["source_content_sha256"]],
                ["tokenizer", tokenizer["tokenizer"]],
                ["tokenizer requested revision", tokenizer["requested_revision"]],
                ["tokenizer resolved revision", tokenizer.get("resolved_revision")],
                ["python", summary["environment"]["python"]],
                *[[f"`{name}`", version] for name, version in summary["environment"]["dependencies"].items()],
            ],
        ),
        "",
        f"Command: `{summary['command']}`",
        "",
        "Arguments: `" + canonical_json(summary["arguments"]) + "`",
        "",
        "## Stage counts and behavior distribution",
        "",
    ]
    rows = []
    for stage in summary["stages"]:
        dist = stage["distribution"]
        pct = main_pcts(dist)
        rows.append(
            [stage["stage"], stage["rows"], *[f"{dist[_axis(k)][k]['count']} ({pct[k]:.2f}%)" for k in MAIN_STRATA]]
        )
    out += [_table(["stage", "rows", *MAIN_STRATA], rows), ""]
    cells = sorted({key for stage in summary["stages"] for key in stage["distribution"]["crosstab"]})
    out += [
        "Cross-tab counts by stage (conversation|target):",
        "",
        _table(
            ["stage", *cells],
            [
                [stage["stage"], *[stage["distribution"]["crosstab"].get(key, 0) for key in cells]]
                for stage in summary["stages"]
            ],
        ),
        "",
    ]
    out += [
        "`source_raw_markers` classifies raw rows by markers only (see dataset-card comparison); later stages use parsed "
        "counts. The first pp step (structural_validation) is therefore measured against that raw-marker view.",
        "",
        "### Percentage-point changes",
        "",
        _table(
            ["stage", *[f"{key} (step / cumulative)" for key in MAIN_STRATA]],
            [
                [
                    change["stage"],
                    *[
                        f"{change['vs_previous_pp'][k]:+.3f} / {change['vs_structural_pp'][k]:+.3f}"
                        for k in MAIN_STRATA
                    ],
                ]
                for change in summary["distribution_changes"]
            ],
        ),
        "",
        "Distribution warnings (> 0.5 pp step change): "
        + ("; ".join(summary["distribution_warnings"]) if summary["distribution_warnings"] else "none"),
        "",
    ]
    final = summary["stages"][-1]["distribution"]
    canonical = next(stage for stage in summary["stages"] if stage["stage"] == "contamination_filter")["distribution"]
    for title, dist in (("canonical (pre-length)", canonical), ("train", final)):
        out += [f"### Cross-tabs: {title}", ""]
        out += [_table(["conversation|target", "rows"], [[key, value] for key, value in dist["crosstab"].items()]), ""]
        for name in ("user_message_buckets", "history_call_buckets", "target_call_buckets"):
            out += [
                _table(
                    [name, "rows", "pct"],
                    [[key, value["count"], f"{value['pct']:.2f}%"] for key, value in dist[name].items()],
                ),
                "",
            ]
    card = summary["dataset_card_comparison"]
    out += [
        "## Dataset-card comparison",
        "",
        _table(
            ["view", "total", "non_function_call", "single_turn (calls)", "multi_turn (calls)"],
            [
                [name, *[view[key] for key in ("total", "non_function_call", "single_turn", "multi_turn")]]
                for name, view in (
                    ("dataset card", card["dataset_card"]),
                    ("raw markers", card["raw_markers"]),
                    ("parsed", card["parsed"]),
                )
            ],
        ),
        "",
        card["note"],
        "",
        "## Rejections",
        "",
        _table(
            ["stage/reason", "rows", "sample source_index:id"],
            [
                [key, value["rows"], "<br>".join(value["sample_ids"][:3])]
                for key, value in summary["rejections"]["by_reason"].items()
            ],
        ),
        "",
        "## Repairs, warnings, and info events",
        "",
        _table(
            ["code", "kind", "rows", "events", "sample source_index:id"],
            [
                [code, value["kind"], value["rows"], value["events"], "<br>".join(value["sample_ids"][:2])]
                for code, value in summary["repairs_and_warnings"].items()
            ],
        ),
        "",
        f"Strict-schema counterfactual: {summary['strict_schema_counterfactual']['rows']} parsed rows depend on the "
        f"`schema_type_alias` repair (by stratum: `{canonical_json(summary['strict_schema_counterfactual']['by_stratum'])}`).",
        "",
    ]
    conflicts = summary["conflicts"]
    out += [
        "## Conflicting references",
        "",
        f"Groups: {conflicts['groups']}; affected rows: {conflicts['affected_rows']}.",
        "",
        f"Target-kind sets: `{canonical_json(conflicts['target_kind_sets'])}`",
        "",
        f"Call names (top 30): `{canonical_json(conflicts['call_names'])}`",
        "",
        "## Deduplication",
        "",
        f"Exact duplicates removed: {summary['exact_dedup']['removed']}.",
        "",
    ]
    near = summary["near_dedup"]
    out += [
        f"Near duplicates removed: {near['removed']} in {near['clusters']} clusters (exact Jaccard >= {near['threshold']}, "
        f"{near['minhash_permutations']} permutations, LSH threshold {near['lsh_threshold']}, seed {near['seed']}, "
        f"{near['shingle_size']}-word shingles). Strata compared: {near['strata_compared']} "
        f"({near['rows_in_multi_member_strata']} rows); LSH candidate pairs verified: {near['lsh_candidate_pairs_checked']}; "
        f"rows below the minimum word count: {near['rows_below_min_words']}.",
        "",
        f"### Review sample ({len(near['review_clusters'])} clusters, ordered by cluster id)",
        "",
    ]
    for cluster in near["review_clusters"]:
        rep = cluster["representative"]
        out.append(
            f"- `{cluster['cluster_id']}` keep #{rep['source_index']} ({rep['prompt_tokens']} tok): "
            f"user “{_md_escape(rep['excerpt']['last_user'])}” → `{_md_escape(rep['excerpt']['target'])}`"
        )
        for member in cluster["removed"]:
            out.append(
                f"  - drop #{member['source_index']} J={member['jaccard']:.4f}: user “{_md_escape(member['excerpt']['last_user'])}” "
                f"→ `{_md_escape(member['excerpt']['target'])}`"
            )
    bfcl = summary["bfcl_audit"]
    out += ["", "## BFCL contamination audit", ""]
    if bfcl.get("status") == "completed":
        out += [
            f"Status: completed (bfcl-eval {bfcl.get('bfcl_eval_version')}, {bfcl['source']}); benchmark prompts: "
            f"{bfcl['benchmark_prompts']}; categories: {', '.join(bfcl['categories'])}.",
            "",
            f"Files: {', '.join(item['file'] for item in bfcl['files'])}. Skipped: "
            f"{canonical_json(bfcl['skipped_files'])}.",
            "",
            f"Removed rows: {bfcl['removed_rows']}; audit-only rows: {bfcl['audit_only_rows']}; rule: {bfcl['remove_rule']}.",
            "",
            _table(
                ["disposition/reason", "records"],
                [[key, value] for key, value in sorted(bfcl["dispositions"].items())],
            ),
        ]
    else:
        out.append(f"Status: {bfcl.get('status')} ({bfcl.get('reason')}).")
    lengths = summary["prompt_lengths"]
    stats_headers = ["group", "n", "p50", "p75", "p90", "p95", "p99", "max", "mean", "≤2K", "≤4K", "≤8K", "≤16K"]

    def stats_row(name, stats):
        cells = [stats[key] for key in ("count", "p50", "p75", "p90", "p95", "p99", "max", "mean")]
        return [
            name,
            *cells,
            *[f"{value['pct']:.2f}% ({value['exceeding']} over)" for value in stats["at_or_below"].values()],
        ]

    table_rows = [stats_row("all", lengths["overall"])]
    for axis, groups in lengths["by"].items():
        table_rows += [stats_row(f"{axis}={key}", stats) for key, stats in groups.items()]
    contribution = lengths["contribution"]
    out += [
        "",
        "## Prompt tokens (canonical set, pinned tokenizer, add_generation_prompt=True)",
        "",
        _table(stats_headers[: 9 + len(lengths["overall"]["at_or_below"])], table_rows),
        "",
        f"Token share: system+tool schema {contribution['token_share_pct']['system_tool']}%, dialogue history "
        f"{contribution['token_share_pct']['dialogue_history']}%. System+tools p50/p95/max: "
        f"{contribution['all']['system_tool']['p50']}/{contribution['all']['system_tool']['p95']}/{contribution['all']['system_tool']['max']}; "
        f"history p50/p95/max: {contribution['all']['dialogue_history']['p50']}/{contribution['all']['dialogue_history']['p95']}/"
        f"{contribution['all']['dialogue_history']['max']}.",
        "",
        _table(
            ["prompts over", "rows", "median system+tools", "median history", "rows where tools > history"],
            [
                [
                    key,
                    value["rows"],
                    value["median_system_tool_tokens"],
                    value["median_history_tokens"],
                    value["rows_where_system_tool_exceeds_history"],
                ]
                for key, value in contribution.items()
                if key.startswith(">")
            ],
        ),
        "",
        "## Prompt-limit selection",
        "",
    ]
    limit = summary["prompt_limit"]
    out += [
        _table(
            [
                "limit",
                "retained",
                "overall %",
                *[f"{key} kept %" for key in MAIN_STRATA],
                "max |pp shift|",
                "all criteria",
            ],
            [
                [
                    item["limit"],
                    item["retained"],
                    f"{item['overall_retention_pct']:.2f}",
                    *[f"{item['stratum_retention_pct'][key]:.2f}" for key in MAIN_STRATA],
                    f"{max(abs(value) for value in item['pp_shift'].values()):.3f}",
                    "yes" if item["satisfies_all"] else "no",
                ]
                for item in limit["candidates"]
            ],
        ),
        "",
        f"Automatic: **{limit['automatic']}** ({limit['automatic_reason']}). Override: {limit['override']}. "
        f"Effective: **{limit['effective']}**.",
        "",
        "### Downstream rollout budget",
        "",
        _table(
            ["setting", "tokens"],
            [[key, value] for key, value in summary["rollout_budget"].items()],
        ),
        "",
        "The response limit is a generation ceiling, not the model context length. Prompt plus response requires at "
        "least `minimum_combined_context_tokens`; the model's native context is reported separately.",
        "",
        "## Output validation",
        "",
    ]
    if summary["output_validation"]:
        out.append(
            _table(
                ["file", "rows", "contract problems", "render errors", "token mismatches", "sorted", "passed"],
                [
                    [
                        name,
                        value["rows"],
                        canonical_json(value["contract_problems"]),
                        value["prompt_render_or_target_append_errors"],
                        value["prompt_token_mismatches_vs_metadata"],
                        value["sorted_by_source_index"],
                        value["passed"],
                    ]
                    for name, value in summary["output_validation"].items()
                ],
            )
        )
    else:
        out.append("Skipped.")
    out += [
        "",
        f"Tests: {summary['test_result'] or 'not recorded (pass --test-log)'}",
        "",
        "## Artifacts",
        "",
        _table(
            ["artifact", "rows", "sha256", "path"],
            [
                [name, value["rows"], f"`{value['sha256']}`", f"`{value['path']}`"]
                for name, value in summary["artifacts"].items()
            ],
        ),
        "",
        "## Caveats",
        "",
        "- Text/no-call targets (clarifications, refusals, summaries) are kept as one `text_no_call` class; exact string "
        "match against them is not a valid semantic reward.",
        "- Tool-call rewards should compare the parsed final action after the policy's own reasoning; no reference "
        "reasoning is stored.",
        "- Historical argument/schema disagreements are reported, never coerced; the source contains deliberate error/"
        "recovery turns.",
        "- Target argument/schema disagreements are rejected because the target is supervision, not recovery context.",
        "- The prompt limit is a dataset-selection choice, independent of the model's native context length.",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run(args)
    counts = summary["stage_counts"]
    print(
        json.dumps(
            {
                "stage_counts": counts,
                "prompt_limit": {k: summary["prompt_limit"][k] for k in ("automatic", "effective")},
            },
            indent=2,
        )
    )
    print(f"Wrote {args.output_dir.resolve()}/looptool_summary.md")


if __name__ == "__main__":
    main()
