#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wrap trained Qwen3.5 language weights in vLLM's expected outer config.

Hugging Face can load the extracted ``Qwen3_5ForCausalLM`` text config
directly.  vLLM 0.25.1 instead registers Qwen3.5 through the outer
``Qwen3_5ForConditionalGeneration`` config, even in language-model-only mode.
The official checkpoint uses the same ``model.language_model.*`` names as the
text-only checkpoint, so a zero-copy wrapper is sufficient: outer non-weight
assets plus symlinks to the trained language-only index and shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language-model", required=True, type=Path)
    parser.add_argument("--outer-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    os.symlink(source.resolve(), destination)


def main() -> None:
    args = parse_args()
    language = args.language_model.resolve()
    outer = args.outer_checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing wrapper {output}")
    for directory in (language, outer):
        if not directory.is_dir():
            raise NotADirectoryError(directory)

    language_config_path = language / "config.json"
    outer_config_path = outer / "config.json"
    language_index_path = language / "model.safetensors.index.json"
    outer_index_path = outer / "model.safetensors.index.json"
    language_config = json.loads(language_config_path.read_text(encoding="utf-8"))
    outer_config = json.loads(outer_config_path.read_text(encoding="utf-8"))
    language_index = json.loads(language_index_path.read_text(encoding="utf-8"))
    outer_index = json.loads(outer_index_path.read_text(encoding="utf-8"))

    if language_config.get("model_type") != "qwen3_5_text":
        raise ValueError("language checkpoint is not qwen3_5_text")
    if language_config.get("architectures") != ["Qwen3_5ForCausalLM"]:
        raise ValueError("language checkpoint architecture is not Qwen3_5ForCausalLM")
    if outer_config.get("model_type") != "qwen3_5":
        raise ValueError("outer checkpoint is not qwen3_5")
    if outer_config.get("architectures") != [
        "Qwen3_5ForConditionalGeneration"
    ]:
        raise ValueError(
            "outer checkpoint architecture is not "
            "Qwen3_5ForConditionalGeneration"
        )
    if outer_config.get("text_config", {}).get("model_type") != "qwen3_5_text":
        raise ValueError("outer checkpoint has no qwen3_5_text config")
    for field in (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "linear_num_key_heads",
        "linear_num_value_heads",
    ):
        left = language_config.get(field)
        right = outer_config["text_config"].get(field)
        if left != right:
            raise ValueError(
                f"language/outer text config mismatch for {field}: {left} != {right}"
            )

    language_weights = language_index.get("weight_map", {})
    outer_weights = outer_index.get("weight_map", {})
    if len(language_weights) < 400:
        raise ValueError("language checkpoint weight index is unexpectedly small")
    if not all(
        name.startswith("model.language_model.") for name in language_weights
    ):
        raise ValueError(
            "language checkpoint contains weights outside model.language_model"
        )
    expected_language_names = {
        name for name in outer_weights if name.startswith("model.language_model.")
    }
    if set(language_weights) != expected_language_names:
        missing = sorted(expected_language_names - set(language_weights))
        unexpected = sorted(set(language_weights) - expected_language_names)
        raise ValueError(
            "language weights do not exactly match official outer names: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    shard_names = sorted(set(language_weights.values()))
    output.mkdir(parents=True)
    try:
        # Start from the official non-weight processor/tokenizer assets.  The
        # language checkpoint's tokenizer metadata is then preferred because
        # it has already passed the cross-version portability gate.
        linked_assets = []
        for source in sorted(outer.iterdir()):
            if (
                not source.is_file()
                or source.name == "config.json"
                or source.name == "model.safetensors.index.json"
                or source.name.endswith(".safetensors")
            ):
                continue
            link(source, output / source.name)
            linked_assets.append(source.name)
        for name in ("tokenizer.json", "tokenizer_config.json"):
            destination = output / name
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            link(language / name, destination)

        link(language_index_path, output / "model.safetensors.index.json")
        for shard_name in shard_names:
            link(language / shard_name, output / shard_name)

        # config.json is copied as an immutable JSON payload rather than
        # symlinked so the wrapper's hash remains explicit in its manifest.
        (output / "config.json").write_text(
            json.dumps(outer_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "qwen35_vllm_language_wrapper_v1",
            "language_model": str(language),
            "outer_checkpoint": str(outer),
            "architecture": "Qwen3_5ForConditionalGeneration",
            "language_model_only": True,
            "weight_tensors": len(language_weights),
            "weight_shards": shard_names,
            "weight_bytes": sum((language / name).stat().st_size for name in shard_names),
            "language_index_sha256": sha256(language_index_path),
            "outer_config_sha256": sha256(outer_config_path),
            "wrapper_config_sha256": sha256(output / "config.json"),
            "tokenizer_config_sha256": sha256(output / "tokenizer_config.json"),
            "linked_outer_assets": linked_assets,
        }
        manifest["lock_hash"] = canonical_hash(manifest)
        (output / "vllm-language-wrapper.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        for item in output.iterdir():
            item.unlink()
        output.rmdir()
        raise

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
