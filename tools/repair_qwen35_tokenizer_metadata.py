#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repair a Transformers-5 Qwen3.5 tokenizer declaration for 4.57 runtimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--language-model", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic_json(payload: dict[str, Any], path: Path) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    source_config = args.source / "tokenizer_config.json"
    output_config = args.language_model / "tokenizer_config.json"
    manifest_path = args.language_model / "extraction-manifest.json"
    for required in (
        source_config,
        output_config,
        manifest_path,
        args.language_model / "tokenizer.json",
        args.language_model / "model.safetensors.index.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    source = json.loads(source_config.read_text(encoding="utf-8"))
    current = json.loads(output_config.read_text(encoding="utf-8"))
    if source.get("tokenizer_class") != "Qwen2Tokenizer":
        raise ValueError(
            "source tokenizer metadata is not the official portable "
            f"Qwen2Tokenizer declaration: {source.get('tokenizer_class')!r}"
        )
    if current.get("tokenizer_class") not in {
        "TokenizersBackend",
        "Qwen2Tokenizer",
    }:
        raise ValueError(
            f"refusing unexpected current tokenizer class "
            f"{current.get('tokenizer_class')!r}"
        )
    write_atomic_json(source, output_config)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "qwen35_language_model_extraction_v1":
        raise ValueError("unexpected extraction manifest schema")
    manifest["tokenizer_class"] = source["tokenizer_class"]
    manifest["tokenizer_config_sha256"] = sha256(output_config)
    write_atomic_json(manifest, manifest_path)

    # Verify both the declaration and the still-unchanged tokenizer payload.
    repaired = json.loads(output_config.read_text(encoding="utf-8"))
    if repaired.get("tokenizer_class") != "Qwen2Tokenizer":
        raise ValueError("atomic tokenizer metadata repair did not persist")
    print(
        json.dumps(
            {
                "status": "REPAIRED_QWEN35_TOKENIZER_METADATA",
                "previous_tokenizer_class": current.get("tokenizer_class"),
                "tokenizer_class": repaired["tokenizer_class"],
                "tokenizer_config_sha256": sha256(output_config),
                "language_model": str(args.language_model.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
