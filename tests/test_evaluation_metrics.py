import json
import sys
from types import SimpleNamespace

import pytest

from evaluation.livecodebench_v6.lock_dataset import (
    load_locked_rows,
    release_files,
    sha256_file,
)
from evaluation.math_tasks.utils import identity_filter, process_results
from evaluation.summarize import summarize_code, summarize_math


def test_code_reports_sample_average_not_any_correct():
    rows = [
        {"question_id": "a", "graded_list": [False, True, False, False], "response_tokens": [10, 20, 30, 40]},
        {"question_id": "b", "graded_list": [False] * 4, "response_tokens": [10] * 4},
    ]
    result = summarize_code(rows)
    assert result["accuracy"] == 0.125
    assert result["mean_response_tokens"] == 17.5
    assert result["samples"] == 8


def test_code_rejects_missing_or_invalid_grades():
    with pytest.raises(ValueError):
        summarize_code([{"question_id": "a", "graded_list": [True], "response_tokens": [10]}])
    with pytest.raises(ValueError):
        summarize_code([{"question_id": "a", "graded_list": ["False"] * 4, "response_tokens": [10] * 4}])


def test_math_grades_every_identity_filtered_response(monkeypatch):
    monkeypatch.setitem(sys.modules, "math_verify", SimpleNamespace(
        parse=lambda text: text.strip("$"), verify=lambda target, answer: target == answer,
    ))
    assert process_results({"Answer": 42}, [["wrong", "42", "wrong", "wrong"]]) == {"accuracy": 0.25}
    with pytest.raises(ValueError):
        process_results({"Answer": 42}, ["42"])


def test_identity_filter_keeps_all_repeats():
    responses = [["answer"] * 64, ["other"] * 64]
    assert identity_filter(responses, [{}, {}]) is responses


def test_math_summary_preserves_all_responses():
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
    result = summarize_math([{
        "doc_id": 0, "accuracy": 0.5, "resps": [["a b", "c d e f"]],
    }], tokenizer, repeats=2)
    assert result["accuracy"] == 0.5
    assert result["mean_response_tokens"] == 3


def test_dataset_lock_checks_checksum_and_question_ids(tmp_path):
    raw = tmp_path / "test6.jsonl"
    raw.write_text(json.dumps({"question_id": "a", "contest_date": "2025-02-01"}) + "\n")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({
        "start_date": "2025-02-01", "question_ids": ["a"],
        "source_files": [{"path": "test6.jsonl", "sha256": sha256_file(raw)}],
    }))
    assert load_locked_rows(lock)[1][0]["question_id"] == "a"
    raw.write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        load_locked_rows(lock)


def test_release_files():
    assert release_files("release_v5")[-1] == "test5.jsonl"
    assert release_files("release_v6")[-1] == "test6.jsonl"
    with pytest.raises(ValueError):
        release_files("release_latest")
