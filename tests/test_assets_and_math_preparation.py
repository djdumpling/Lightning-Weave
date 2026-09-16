import json
import sys
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_curation import prepare_direct_opd_assets as assets
from data_curation import prepare_direct_opd_skywork_math as math_prep


def prepare_assets(tmp_path, monkeypatch, vocabularies):
    output = tmp_path / "assets.json"

    def record(role, revision):
        vocab = vocabularies[role]
        return {
            "model_vocab_size": 8,
            "tokenizer_hash": assets.canonical_hash(vocab),
            "revision": revision,
        }, vocab

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--student",
            "student",
            "--pre-teacher",
            "pre_teacher",
            "--post-teacher",
            "post_teacher",
            "--student-revision",
            "student-rev",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(assets, "tokenizer_record", record)
    assets.main()
    return json.loads(output.read_text())


def test_equal_token_mappings_use_tokenizer_action_space_not_padded_width(tmp_path, monkeypatch):
    vocabularies = {role: {"a": 0, "b": 2} for role in ("student", "pre_teacher", "post_teacher")}
    metadata = prepare_assets(tmp_path, monkeypatch, vocabularies)
    assert metadata["token_id_compatibility"] == {
        "mode": "official_input_tokenizer_null",
        "normalization_vocab_size": 3,
        "model_vocab_size": 3,
    }
    assert metadata["models"]["student"]["revision"] == "student-rev"


def test_equal_width_different_token_ids_require_projection(tmp_path, monkeypatch):
    vocabularies = {
        "student": {"a": 0, "b": 1, "student-only": 2},
        "pre_teacher": {"a": 1, "b": 0, "teacher-only": 2},
        "post_teacher": {"a": 1, "b": 0, "teacher-only": 2},
    }
    metadata = prepare_assets(tmp_path, monkeypatch, vocabularies)
    assert metadata["token_id_compatibility"] == {
        "mode": assets.TOKEN_PROJECTION_MODE,
        "normalization_vocab_sizes": {role: 3 for role in vocabularies},
    }


@pytest.mark.parametrize("field", ["vocab_size", "max_position_embeddings"])
def test_model_config_uses_top_level_then_nested_text_config(field):
    config = SimpleNamespace(**{field: 10, "text_config": SimpleNamespace(**{field: 20})})
    assert assets.text_config_value(config, field) == 10
    setattr(config, field, None)
    assert assets.text_config_value(config, field) == 20
    assert assets.text_config_value(SimpleNamespace(), field) is None


def test_model_vocab_size_reads_text_and_multimodal_configs():
    assert assets.resolve_model_vocab_size(SimpleNamespace(vocab_size=100)) == 100
    config = SimpleNamespace(vocab_size=None, text_config=SimpleNamespace(vocab_size=200))
    assert assets.resolve_model_vocab_size(config) == 200


def math_source():
    return pa.Table.from_pylist(
        [
            {
                "data_source": "source_math",
                "prompt": [{"role": "user", "content": "  What is 1 + 1?  "}],
                "ability": "math",
                "reward_model": {"ground_truth": '["2"]', "style": "rule"},
                "extra_info": {
                    "index": 9,
                    "model_difficulty": {
                        "DeepSeek-R1-Distill-Qwen-1.5B": 1,
                        "DeepSeek-R1-Distill-Qwen-32B": 2,
                        "DeepSeek-R1-Distill-Qwen-7B": 3,
                    },
                },
            }
        ]
    )


def test_math_conversion_preserves_prompt_label_and_source_metadata():
    converted = math_prep.convert_row(math_source().to_pylist()[0])
    assert converted == (
        {
            "data_source": "math_dapo",
            "prompt": [
                {
                    "role": "user",
                    "content": math_prep.PROMPT_PREFIX + "What is 1 + 1?" + math_prep.PROMPT_SUFFIX,
                }
            ],
            "ability": "MATH",
            "reward_model": {"ground_truth": "2", "style": "rule-lighteval/MATH_v2"},
            "extra_info": {
                "index": 9,
                "model_difficulty": {
                    "DeepSeek-R1-Distill-Qwen-1.5B": 1,
                    "DeepSeek-R1-Distill-Qwen-32B": 2,
                    "DeepSeek-R1-Distill-Qwen-7B": 3,
                },
                "original_data_source": "source_math",
                "prompt_style": "dapo_original",
            },
        }
    )


def test_math_main_converts_arbitrary_row_count_without_output_readback(tmp_path, monkeypatch):
    source = math_source()
    source_path = tmp_path / "source.parquet"
    output = tmp_path / "output.parquet"
    pq.write_table(source, source_path)
    monkeypatch.setattr(sys, "argv", ["prepare", "--input", str(source_path), "--output", str(output)])
    read_paths = []
    original_read_table = pq.read_table

    def read_table(path, **kwargs):
        read_paths.append(path)
        return original_read_table(path, **kwargs)

    monkeypatch.setattr(pq, "read_table", read_table)
    math_prep.main()
    assert read_paths == [source_path]
    assert original_read_table(output).to_pylist() == [math_prep.convert_row(row) for row in source.to_pylist()]
    with pytest.raises(FileExistsError, match="Output exists"):
        math_prep.main()
