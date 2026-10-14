# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract the language-only Qwen3.5 causal LM from a multimodal checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    for required in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        path = args.input_dir / required
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    model = AutoModelForCausalLM.from_pretrained(
        args.input_dir,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if type(model).__name__ != "Qwen3_5ForCausalLM":
        raise TypeError(f"expected Qwen3_5ForCausalLM, got {type(model).__name__}")
    if hasattr(model.config, "vision_config"):
        raise ValueError("language-only model unexpectedly retained vision_config")
    if int(model.config.vocab_size) != 248_320:
        raise ValueError(f"unexpected Qwen3.5 vocab size {model.config.vocab_size}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.input_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(args.output_dir)
    # Transformers 5 serializes the generic fast-tokenizer implementation as
    # ``TokenizersBackend``.  The training image intentionally remains on
    # Transformers 4.57, where that class name does not exist.  Preserve the
    # checkpoint author's portable Qwen2 tokenizer declaration instead; the
    # tokenizer.json payload itself is unchanged.
    source_tokenizer_config = args.input_dir / "tokenizer_config.json"
    source_tokenizer_metadata = json.loads(source_tokenizer_config.read_text())
    if source_tokenizer_metadata.get("tokenizer_class") != "Qwen2Tokenizer":
        raise ValueError(
            "expected official Qwen3.5 tokenizer_class=Qwen2Tokenizer, got "
            f"{source_tokenizer_metadata.get('tokenizer_class')!r}"
        )
    shutil.copyfile(
        source_tokenizer_config, args.output_dir / "tokenizer_config.json"
    )

    # Preserve model-specific non-weight assets not emitted by save_pretrained.
    for filename in (
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
        "LICENSE",
        "README.md",
    ):
        source = args.input_dir / filename
        destination = args.output_dir / filename
        if source.is_file() and not destination.exists():
            shutil.copyfile(source, destination)

    index_path = args.output_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            "expected a sharded language-only checkpoint, but no index was written"
        )
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("language-only weight index is empty")
    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        shard = args.output_dir / shard_name
        if not shard.is_file() or shard.stat().st_size == 0:
            raise FileNotFoundError(shard)

    manifest = {
        "schema_version": "qwen35_language_model_extraction_v1",
        "source_dir": str(args.input_dir.resolve()),
        "source_revision": args.source_revision,
        "source_config_sha256": sha256(args.input_dir / "config.json"),
        "output_config_sha256": sha256(args.output_dir / "config.json"),
        "model_class": type(model).__name__,
        "model_type": model.config.model_type,
        "dtype": str(model.dtype),
        "vocab_size": int(model.config.vocab_size),
        "parameter_count": parameter_count,
        "weight_tensors": len(weight_map),
        "weight_shards": shard_names,
        "weight_bytes": sum((args.output_dir / name).stat().st_size for name in shard_names),
        "tokenizer_vocab_size": len(tokenizer.get_vocab()),
        "tokenizer_class": source_tokenizer_metadata["tokenizer_class"],
        "tokenizer_config_sha256": sha256(
            args.output_dir / "tokenizer_config.json"
        ),
    }
    temporary = args.output_dir / "extraction-manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output_dir / "extraction-manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
