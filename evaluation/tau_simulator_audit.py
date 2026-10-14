"""Audit repeated tau2 retail baselines, user latency, and task-5 confirmations."""
from __future__ import annotations
import collections
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def distribution(values):
    values = sorted(values)
    def quantile(q):
        return values[min(len(values) - 1, int(q * len(values)))] if values else None
    return {"calls": len(values), "mean_s": sum(values) / len(values) if values else None,
            "p50_s": quantile(.5), "p90_s": quantile(.9), "p99_s": quantile(.99),
            "max_s": max(values, default=None)}


def task5_dialog(record):
    messages = (record.get("result") or {}).get("messages", [])
    pairs, transcript = [], []
    pending = None
    for message in messages:
        role, content = message.get("role"), (message.get("content") or "").strip()
        if role in ("assistant", "user") and content:
            transcript.append({"role": role, "content": content})
        if role == "assistant" and content:
            pending = content if re.search(r"confirm|proceed|yes[/\s]+no|reply.*yes", content, re.I) else None
        elif role == "user" and pending is not None:
            pairs.append({"agent": pending, "user": content})
            pending = None
    return {"trial": record["trial"], "success": record["success"],
            "termination": record["termination"], "confirmations": pairs, "transcript": transcript}


def audit(root, tag, trials=24, tasks=114):
    root = Path(root) / tag
    records, missing, corrupt, events, dialogs, error_examples = [], [], [], [], [], []
    error_last_finish = collections.Counter()
    user_empty = user_reasoning = marker_leaks = inspected_user_messages = 0
    transcript_conversations = 0
    def read_record(spec):
        trial, task = spec
        filename = f"{task}-{hashlib.sha1(task.encode()).hexdigest()[:8]}.json"
        path = root / "tau2_retail" / f"trial{trial}" / filename
        try:
            record = json.loads(path.read_text())
            if record["task_id"] != task or record["trial"] != trial:
                raise ValueError("record identity mismatch")
            for key in ("success", "termination", "reward", "user_calls"):
                record[key]
        except FileNotFoundError:
            return trial, task, path, None, None
        except (ValueError, KeyError, TypeError) as error:
            return trial, task, path, None, str(error)
        return trial, task, path, record, None

    specs = [(trial, str(task)) for trial in range(trials) for task in range(tasks)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        for trial, task, path, record, error in pool.map(read_record, specs):
            if error is not None:
                corrupt.append({"path": str(path), "error": error}); continue
            if record is None:
                missing.append([task, trial]); continue
            records.append({k: record[k] for k in ("task_id", "trial", "success", "termination", "reward")})
            if record.get("error"):
                last_call = record.get("agent_calls", [])[-1:]
                error_last_finish.update(call.get("finish_reason") for call in last_call)
                if len(error_examples) < 12:
                    error_examples.append({"task_id": task, "trial": trial, "error": record["error"],
                                           "agent_last_calls": record.get("agent_calls", [])[-2:]})
            for index, call in enumerate(record["user_calls"]):
                if call.get("seconds") is not None:
                    events.append({**call, "turn_index": call.get("turn_index", index),
                                   "task_id": task, "trial": trial})
            transcript_conversations += record.get("result") is not None
            for message in (record.get("result") or {}).get("messages", []):
                if message.get("role") != "user": continue
                inspected_user_messages += 1
                content = (message.get("content") or "")
                user_empty += not content.strip() and not message.get("tool_calls")
                marker_leaks += any(marker in content for marker in ("<think>", "</think>", "[think]", "[end_think]"))
                native = (message.get("raw_data") or {}).get("message", {})
                user_reasoning += bool(native.get("reasoning") or native.get("reasoning_content"))
            if task == "5": dialogs.append(task5_dialog(record))
    grouped = collections.defaultdict(list)
    for record in records: grouped[record["task_id"]].append(record["success"])
    per_task = {task: {"trials": len(values), "successes": sum(values), "rate": sum(values) / len(values)}
                for task, values in sorted(grouped.items(), key=lambda item: int(item[0]))}
    record_events = len(events)
    stream_events = []
    stream_files = sorted((root / "invocations").glob("*.tau2_retail.user_latency.jsonl"))
    for stream_file in stream_files:
        for line in stream_file.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue  # an active writer may not have finished its last line
            if event.get("seconds") is not None:
                stream_events.append(event)
    latency_events = stream_events or events
    complete = not missing and not corrupt and len(records) == tasks * trials
    starts = [event["started_at_unix_s"] for event in latency_events if "started_at_unix_s" in event]
    first = min(starts, default=None)
    bins = collections.defaultdict(list)
    for event in latency_events:
        if first is not None and "started_at_unix_s" in event:
            bins[int((event["started_at_unix_s"] - first) // 300)].append(event["seconds"])
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    return {"tag": tag, "requested_trials": trials, "expected": tasks * trials, "conversations": len(records),
            "complete": complete,
            "missing": missing, "corrupt": corrupt, "manifest": manifest,
            "avg_at_24_percent": 100 * sum(record["success"] for record in records) / len(records) if complete and trials == 24 else None,
            "per_task": per_task, "terminations": dict(collections.Counter(record["termination"] for record in records)),
            "user_turns_in_saved_records": record_events, "inspected_user_messages": inspected_user_messages,
            "conversations_with_transcripts": transcript_conversations, "empty_user_turns": user_empty,
            "agent_error_last_finish_reasons": dict(error_last_finish), "error_examples": error_examples,
            "reasoning_user_turns": user_reasoning, "reasoning_marker_leaks": marker_leaks,
            "user_finish_reasons": dict(collections.Counter(event.get("finish_reason") for event in events)),
            "user_latency": {"source": "live_call_streams" if stream_events else "saved_conversations",
                             "stream_files": [str(p.name) for p in stream_files],
                             "all": distribution([e["seconds"] for e in latency_events]),
                             "first_turn": distribution([e["seconds"] for e in latency_events if e["turn_index"] == 0]),
                             "later_turn": distribution([e["seconds"] for e in latency_events if e["turn_index"] > 0]),
                             "warm_after_ten_minutes": distribution([e["seconds"] for e in latency_events
                                 if first is not None and e.get("started_at_unix_s", first) - first >= 600]),
                             "start_time_unix_s": first,
                             "five_minute_bins": [{"minute_start": k * 5, **distribution(v)} for k, v in sorted(bins.items())]},
            "task5_dialogs": dialogs}


def compare_existing_235b(new, summary, *, seed=20261012, bootstrap=10000):
    lane = next(lane for lane in summary["lanes"] if lane["domain"] == "tau2_retail")
    if not lane["complete"] or lane["trials"] != 24:
        raise ValueError("existing 235B comparator must be complete avg@24")
    old = lane["per_task"]
    rows = [{"task_id": task, "deepseek": info["rate"], "old235b": old[task],
             "delta_pp": 100 * (info["rate"] - old[task]), "new_trials": info["trials"], "old_trials": lane["trials"]}
            for task, info in new["per_task"].items() if task in old]
    complete = new["complete"] and new.get("requested_trials", 24) == 24 and len(rows) == len(old)
    deltas = [row["delta_pp"] for row in rows] if complete else []
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(bootstrap)) if deltas else []
    return {"old235b_avg24_percent": 100 * lane["pass_hat_k"]["1"],
            "new_deepseek_avg24_percent": new["avg_at_24_percent"],
            "matched_tasks": len(rows), "complete": complete,
            "delta_pp": sum(deltas) / len(deltas) if deltas else None,
            "task_bootstrap_95ci_pp": [draws[int(.025 * len(draws))], draws[int(.975 * len(draws))]] if draws else None,
            "per_task": rows, "bootstrap_seed": seed, "bootstrap_replicates": bootstrap}
