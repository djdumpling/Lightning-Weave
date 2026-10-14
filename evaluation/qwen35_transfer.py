"""Strict retail avg@24 comparison with a paired task bootstrap."""
from __future__ import annotations
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np


def read_model(root: Path, tag: str, protocol: dict):
    folder = root / tag
    manifest = json.loads((folder / "manifest.json").read_text())
    # Dataclass tuples become lists in JSON; retain every field in the comparison.
    if manifest["protocol"] != json.loads(json.dumps(protocol)):
        raise ValueError(f"{tag}: evaluation protocol mismatch")
    rows = {}; sources = []
    for trial in range(24):
        directory = folder / "tau2_retail" / f"trial{trial}"
        files = [f for f in directory.glob("*.json") if f.name != "failures.json"]
        if len(files) != 114:
            raise ValueError(f"{tag}, trial {trial}: {len(files)} records, expected 114")
        sources.extend((trial, f) for f in files)
    def read_record(source):
        trial, path = source
        return trial, path, json.loads(path.read_text())
    # Remote-volume reads are independent; retain deterministic validation order.
    with ThreadPoolExecutor(max_workers=32) as reader:
        for trial, f, r in reader.map(read_record, sources):
            key = (str(r["task_id"]), int(r["trial"]))
            if key[1] != trial or key in rows or r["success"] not in (0, 1, False, True):
                raise ValueError(f"invalid or repeated record: {f}")
            rows[key] = bool(r["success"])
    print(f"Validated {tag}: {len(rows)} task/trial records", flush=True)
    tasks = [str(i) for i in range(114)]
    if {key[0] for key in rows} != set(tasks): raise ValueError("task set mismatch")
    rates = np.array([np.mean([rows[task, trial] for trial in range(24)]) for task in tasks])
    return rates, {"conversations": len(rows), "successes": sum(rows.values()), "avg_at_24_percent": float(rates.mean() * 100)}


def analyze(root: Path, output: Path):
    from configs.qwen35_transfer.config import PROTOCOL, SEEDS
    protocol = PROTOCOL.resolved()
    base, base_summary = read_model(root, "qwen35", protocol)
    rng = np.random.default_rng(20261012)
    indices = rng.integers(0, 114, size=(10000, 114))
    results = {"baseline": base_summary, "bootstrap": {"unit": "task", "paired": True, "seed": 20261012, "resamples": 10000, "interval": "95% percentile"}, "seeds": {}}
    rates = {}
    for seed in SEEDS:
        trained, summary = read_model(root, f"qwen35-retail-s{seed}", protocol)
        rates[seed] = trained
        difference = (trained - base) * 100
        interval = np.percentile(difference[indices].mean(axis=1), [2.5, 97.5])
        results["seeds"][str(seed)] = {**summary, "delta_points": float(difference.mean()), "task_bootstrap_95_percent_interval": interval.tolist()}
    average = (np.mean(list(rates.values()), axis=0) - base) * 100
    results["mean_of_three_seeds"] = {"delta_points": float(average.mean()), "task_bootstrap_95_percent_interval": np.percentile(average[indices].mean(axis=1), [2.5, 97.5]).tolist()}
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    with (output / "task_rates.csv").open("w") as f:
        writer = csv.writer(f); writer.writerow(["task_id", "base", *map(str, SEEDS)])
        for i in range(114): writer.writerow([i, base[i], *[rates[s][i] for s in SEEDS]])
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path); p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(); print(json.dumps(analyze(args.run, args.output), indent=2))
