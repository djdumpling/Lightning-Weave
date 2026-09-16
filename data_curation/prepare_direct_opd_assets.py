#!/usr/bin/env python3
"""Record model paths and tokenizer action spaces used by anchor scoring."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import canonical_hash, write_json

TOKEN_PROJECTION_MODE = "byte_level_exact_token_projection_v1"


def text_config_value(config, name):
    value = getattr(config, name, None)
    return value if value is not None else getattr(getattr(config, "text_config", None), name, None)


def resolve_model_vocab_size(config):
    return int(text_config_value(config, "vocab_size"))


def tokenizer_record(path, revision):
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    vocab = tokenizer.get_vocab()
    return {
        "path": path,
        "revision": revision,
        "model_type": config.model_type,
        "model_vocab_size": resolve_model_vocab_size(config),
        "max_position_embeddings": text_config_value(config, "max_position_embeddings"),
        "tokenizer_hash": canonical_hash(vocab),
    }, vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True)
    parser.add_argument("--post-teacher", required=True)
    parser.add_argument("--pre-teacher", required=True)
    parser.add_argument("--student-revision", default="unknown")
    parser.add_argument("--post-teacher-revision", default="unknown")
    parser.add_argument("--pre-teacher-revision", default="unknown")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    records, vocabularies = {}, {}
    for role in ("student", "post_teacher", "pre_teacher"):
        records[role], vocabularies[role] = tokenizer_record(getattr(args, role), getattr(args, f"{role}_revision"))
    sizes = {role: max(vocab.values()) + 1 for role, vocab in vocabularies.items()}
    # Mapping equality, rather than padded model width, selects token projection.
    projected = any(vocab != vocabularies["student"] for vocab in vocabularies.values())
    compatibility = (
        {"mode": TOKEN_PROJECTION_MODE, "normalization_vocab_sizes": sizes}
        if projected
        else {
            "mode": "official_input_tokenizer_null",
            "model_vocab_size": sizes["student"],
            "normalization_vocab_size": sizes["student"],
        }
    )
    write_json(
        {
            "schema_version": "offline_direct_opd_assets_v1",
            "tokenizer_hash": records["student"]["tokenizer_hash"],
            "token_id_compatibility": compatibility,
            "models": records,
        },
        args.output,
    )
    print(f"Wrote scoring metadata to {args.output}")


if __name__ == "__main__":
    main()
