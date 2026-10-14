"""Protect baseline reporting from missing trials and latency censoring."""
import hashlib
import json
from evaluation.tau_simulator_audit import audit, compare_existing_235b


def write_record(root, task, trial, *, success):
    filename = f"{task}-{hashlib.sha1(task.encode()).hexdigest()[:8]}.json"
    path = root / "agent" / "tau2_retail" / f"trial{trial}" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task_id": task, "trial": trial, "success": success,
                               "reward": float(success), "termination": "user_stop",
                               "user_calls": [], "result": {"messages": []}}))
    return path


def old_summary():
    return {"lanes": [{"domain": "tau2_retail", "complete": True, "trials": 24,
                        "per_task": {"0": 1.0, "1": 0.0}, "pass_hat_k": {"1": 0.5}}]}


def test_partial_baseline_withholds_score_and_comparison(tmp_path):
    for trial in range(24):
        write_record(tmp_path, "0", trial, success=True)
        if trial < 23:
            write_record(tmp_path, "1", trial, success=False)
    row = audit(tmp_path, "agent", tasks=2)
    assert row["missing"] == [["1", 23]]
    assert row["avg_at_24_percent"] is None
    comparison = compare_existing_235b(row, old_summary())
    assert not comparison["complete"]
    assert comparison["delta_pp"] is None
    assert comparison["task_bootstrap_95ci_pp"] is None


def test_complete_baseline_and_identity_corruption(tmp_path):
    for trial in range(24):
        write_record(tmp_path, "0", trial, success=True)
        path = write_record(tmp_path, "1", trial, success=False)
    row = audit(tmp_path, "agent", tasks=2)
    assert row["complete"] and row["avg_at_24_percent"] == 50.0
    comparison = compare_existing_235b(row, old_summary(), bootstrap=100)
    assert comparison["delta_pp"] == 0
    assert comparison["task_bootstrap_95ci_pp"] == [0, 0]
    record = json.loads(path.read_text())
    record["task_id"] = "0"
    path.write_text(json.dumps(record))
    corrupt = audit(tmp_path, "agent", tasks=2)
    assert not corrupt["complete"] and corrupt["avg_at_24_percent"] is None
    assert len(corrupt["corrupt"]) == 1


def test_latency_includes_calls_from_unfinished_conversations(tmp_path):
    write_record(tmp_path, "0", 0, success=True)
    stream = tmp_path / "agent" / "invocations" / "live.tau2_retail.user_latency.jsonl"
    stream.parent.mkdir(parents=True)
    stream.write_text(json.dumps({"task_id": "1", "trial": 0, "turn_index": 0,
                                 "seconds": 40, "started_at_unix_s": 1000}) + "\n" +
                      json.dumps({"task_id": "1", "trial": 0, "turn_index": 1,
                                 "seconds": 1, "started_at_unix_s": 1700}) + '\n{"partial":')
    row = audit(tmp_path, "agent", tasks=2)
    assert row["user_latency"]["source"] == "live_call_streams"
    assert row["user_latency"]["all"]["calls"] == 2
    assert row["user_latency"]["later_turn"]["p50_s"] == 1
    assert row["user_latency"]["warm_after_ten_minutes"]["calls"] == 1
    assert row["user_latency"]["warm_after_ten_minutes"]["p50_s"] == 1
    assert [b["minute_start"] for b in row["user_latency"]["five_minute_bins"]] == [0, 10]
    assert row["avg_at_24_percent"] is None
