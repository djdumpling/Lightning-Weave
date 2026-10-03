#!/usr/bin/env python3
"""Probe the projection students on LoopTool prompts: how much shortening they realized, and whether decisions moved.

Fixed before launch (``PROJECTION_PROBE`` in configs/agent_eff/config.py):

- ``prompts`` (CPU): 400 training prompts, chosen by a fixed hash of their ids, and the 600 held-out prompts of the
  reasoning-value probe (never trained on; each has a tool-call reference). For the training prompts it also records
  each arm's target on the cached responses, and the cached responses themselves for teacher forcing.
- ``sample`` (GPU, vLLM): one model gives 8 responses per prompt with the collection settings (T 0.6, top-p 0.95,
  top-k 20, at most 2,048 tokens, no YaRN), each reduced to its decision as BFCL's server shows it; with
  ``--teacher-force`` it also scores the cached responses of the training prompts (raw log-likelihood).
- ``compare`` (CPU): per student and prompt set, with a bootstrap over prompts:
  - savings: 1 − mean response tokens / the recipient's (its two draws pooled);
  - realization (training prompts): savings / the arm's target-implied savings on the same 400 prompts;
  - generalization: held-out savings against training-prompt savings, undefined unless the latter are shown positive;
  - decision movement: an unbiased estimate of D² = Σ_a (p_a − q_a)² between the student's and the recipient's
    decision distributions per prompt (within-policy pairs exclude self-pairs, so equal distributions give 0 in
    expectation and differences cannot cancel), call-level and byte-exact; the null is the recipient's two draws;
  - fit (descriptive): within-prompt slope of the change in log-likelihood of cached responses on the target's
    log-weight; held-out exact call accuracy against the reference (descriptive).

Readings are heuristics fixed in advance, each with an inconclusive branch. High realization shows the shortening was
substantially learned, not that the target was fitted completely; low realization motivates looking at optimization
without proving that more passes are the cure. Decision movement gets two separate flags: detectable movement (the
interval excludes 0) and equivalence within a margin (the interval lies under it); both can hold, and a non-significant
distance alone is neither.

    python evaluation/projection_probe.py prompts --rollouts DIR --weights W.parquet --heldout prompts.jsonl --output DIR
    python evaluation/projection_probe.py sample --probe DIR --model PATH --seed S --output samples.jsonl [--teacher-force]
    python evaluation/projection_probe.py compare --probe DIR --samples DIR --output report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.decision_projection import ARMS, parse_response
from evaluation.decision_drift import call_signature

SETS = ("train", "heldout")
VIEWS = {"calls": call_signature, "exact": lambda decision: decision}


def selection_key(prompt_id: str) -> str:
    return hashlib.sha256(f"projection-probe\0{prompt_id}".encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.rename(path)


# --- estimators -----------------------------------------------------------------------------------------------------


def unbiased_squared_distance(first: list[str], second: list[str]) -> float:
    """An unbiased estimate of Σ_a (p_a − q_a)² from independent samples of p (first) and q (second)."""
    n, m = len(first), len(second)
    if n < 2 or m < 2:
        raise ValueError("each policy needs at least two samples")
    left, right = Counter(first), Counter(second)
    within_left = sum(count * (count - 1) for count in left.values()) / (n * (n - 1))
    within_right = sum(count * (count - 1) for count in right.values()) / (m * (m - 1))
    cross = sum(left[key] * right[key] for key in left) / (n * m)
    return within_left + within_right - 2 * cross


def target_squared_distance(decisions: list[str], weights: list[float]) -> float:
    """Σ_a (q_a − p̂_a)² between a target's weights and the uniform empirical distribution over cached responses."""
    total = sum(weights)
    moved = defaultdict(float)
    for decision, weight in zip(decisions, weights, strict=True):
        moved[decision] += weight / total - 1 / len(decisions)
    return sum(value * value for value in moved.values())


def bootstrap(statistic, prompts: int, *, draws: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = [statistic(rng.integers(0, prompts, prompts)) for _ in range(draws)]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def reading(interval: list[float], *, high: float, low: float, names: tuple[str, str, str]) -> str:
    """names = (lower bound at least ``high``, upper bound at most ``low``, otherwise)."""
    if interval[0] >= high:
        return names[0]
    if interval[1] <= low:
        return names[1]
    return names[2]


# --- prompts --------------------------------------------------------------------------------------------------------


def build_prompts(rollouts: Path, weights_path: Path, heldout: Path, output: Path, *, train_prompts: int) -> dict:
    """Probe prompts, the arms' targets on the chosen training prompts, and their cached responses."""
    import pyarrow.parquet as pq

    weights = pq.read_table(weights_path).to_pydict()
    by_sample = {
        sample: {arm: weights[f"weight_{arm}"][index] for arm in ARMS}
        for index, sample in enumerate(weights["sample_id"])
    }
    columns = ["prompt_id", "sample_id", "prompt_tokens", "response_tokens", "response", "finish_reason"]
    cached = defaultdict(list)
    rollouts = Path(rollouts)
    files = [rollouts] if rollouts.is_file() else sorted(rollouts.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no rollout shards under {rollouts}")
    for file in files:
        table = pq.read_table(file, columns=[f"metadata.{name}" for name in columns])
        for row in zip(*(table.column(index).to_pylist() for index in range(len(columns))), strict=True):
            cached[str(row[0])].append(dict(zip(columns, row, strict=True)))
    chosen = sorted(cached, key=selection_key)[:train_prompts]
    prompts, fit_rows, targets = [], [], {}
    for prompt_id in chosen:
        rows = cached[prompt_id]
        prompts.append({"set": "train", "prompt_id": prompt_id, "prompt_token_ids": rows[0]["prompt_tokens"]})
        targets[prompt_id] = {
            "tokens": [len(row["response_tokens"]) for row in rows],
            "decisions": [parse_response(row["response"], row["finish_reason"]).decision for row in rows],
            "weights": {arm: [by_sample[str(row["sample_id"])][arm] for row in rows] for arm in ARMS},
        }
        fit_rows.extend(
            {"prompt_id": prompt_id, "sample_id": str(row["sample_id"]), "prompt_tokens": row["prompt_tokens"],
             "response_tokens": row["response_tokens"]}
            for row in rows
        )
    for row in read_jsonl(heldout):
        if row["prompt_id"] in cached:
            raise ValueError(f"held-out prompt {row['prompt_id']} is a training prompt")
        prompts.append({"set": "heldout", "prompt_id": row["prompt_id"], "prompt_token_ids": row["prompt_token_ids"],
                        "target": row["target"], "conversation_kind": row.get("conversation_kind")})
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(prompts, output / "prompts.jsonl")
    write_jsonl(fit_rows, output / "fit_rows.jsonl")
    (output / "targets.json").write_text(json.dumps(targets, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"train": len(chosen), "heldout": sum(p["set"] == "heldout" for p in prompts), "fit_rows": len(fit_rows)}
    (output / "prompts.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


# --- sampling -------------------------------------------------------------------------------------------------------


def exact_call_verdict(text: str, target: dict) -> str:
    """``match`` / ``miss`` against a tool-call reference (calls compared as a set); ``unverifiable`` otherwise."""
    from data_curation.looptool import Reject, canonical_json, parse_assistant

    if not target or not target.get("tool_calls"):
        return "unverifiable"
    try:
        _, calls = parse_assistant(text, "verify", [])
    except Reject:
        return "miss"
    produced = sorted(canonical_json({"name": call["name"], "arguments": call["arguments"]}) for call in calls)
    expected = sorted(canonical_json(call["function"]) for call in target["tool_calls"])
    return "match" if produced == expected else "miss"


def sample_rows(llm, sampling, prompts: list[dict]) -> list[dict]:
    outputs = llm.generate([{"prompt_token_ids": prompt["prompt_token_ids"]} for prompt in prompts], sampling,
                           use_tqdm=False)
    rows = []
    for prompt, output in zip(prompts, outputs, strict=True):
        responses = []
        for item in output.outputs:
            response = {"tokens": len(item.token_ids), "finish_reason": item.finish_reason,
                        "decision": parse_response(item.text, item.finish_reason).decision}
            if prompt["set"] == "heldout":
                response["verdict"] = exact_call_verdict(item.text, prompt.get("target"))
            responses.append(response)
        rows.append({"set": prompt["set"], "prompt_id": prompt["prompt_id"], "responses": responses})
    return rows


def teacher_force(llm, sampling, fit_rows: list[dict], *, chunk: int = 256) -> list[dict]:
    """Summed raw (T = 1) log-likelihood of each cached response after its prompt.

    Training fits temperature-scaled (T = 0.6) log-probabilities, so this is a descriptive proxy for the fit: a weak
    slope alone does not establish poor optimization of the actual objective. Requests go in chunks, since vLLM
    returns a log-probability for every prompt position.
    """
    scored = []
    for start in range(0, len(fit_rows), chunk):
        scored.extend(_teacher_force_chunk(llm, sampling, fit_rows[start : start + chunk]))
    return scored


def _teacher_force_chunk(llm, sampling, fit_rows: list[dict]) -> list[dict]:
    outputs = llm.generate(
        [{"prompt_token_ids": row["prompt_tokens"] + row["response_tokens"]} for row in fit_rows], sampling,
        use_tqdm=False,
    )
    scored = []
    for row, output in zip(fit_rows, outputs, strict=True):
        start = len(row["prompt_tokens"])
        ids = row["prompt_tokens"] + row["response_tokens"]
        total = sum(output.prompt_logprobs[position][ids[position]].logprob for position in range(start, len(ids)))
        scored.append({"prompt_id": row["prompt_id"], "sample_id": row["sample_id"], "logprob": float(total)})
    return scored


# --- comparison -----------------------------------------------------------------------------------------------------


def _mean_tokens(rows: dict[str, dict], ids: list[str]) -> np.ndarray:
    return np.array([np.mean([r["tokens"] for r in rows[prompt]["responses"]]) for prompt in ids])


def generalization(train: tuple, heldout: tuple, train_savings: dict, thresholds: dict, *, draws: int,
                   seed: int) -> dict:
    """Held-out savings relative to training-prompt savings, read without dividing by a possibly nonpositive saving.

    Undefined unless the training-prompt savings are demonstrably positive (lower bound above 0). The readings test the
    contrasts held-out − k · training for k at the two thresholds, resampling both prompt sets: "generalizes" if
    held-out ≥ substantial · training with confidence, "training-specific" if held-out ≤ weak · training with
    confidence. The ratio itself is descriptive.
    """
    if train_savings["ci95"][0] <= 0:
        return {"point": None, "reading": "undefined: training-prompt savings not shown positive"}
    (train_tokens, train_base), (held_tokens, held_base) = train, heldout
    rng = np.random.default_rng(seed)
    high, low = [], []
    for _ in range(draws):
        ti = rng.integers(0, len(train_tokens), len(train_tokens))
        hi = rng.integers(0, len(held_tokens), len(held_tokens))
        train_saving = 1 - train_tokens[ti].mean() / train_base[ti].mean()
        held_saving = 1 - held_tokens[hi].mean() / held_base[hi].mean()
        high.append(held_saving - thresholds["substantial"] * train_saving)
        low.append(held_saving - thresholds["weak"] * train_saving)
    high_lower, low_upper = float(np.percentile(high, 2.5)), float(np.percentile(low, 97.5))
    point = (1 - held_tokens.mean() / held_base.mean()) / (1 - train_tokens.mean() / train_base.mean())
    reading_ = "generalizes" if high_lower >= 0 else "training-specific" if low_upper <= 0 else "inconclusive"
    return {"point": float(point), "contrast_high_lower": high_lower, "contrast_low_upper": low_upper,
            "reading": reading_}


def compare(probe: Path, samples: Path, settings: dict, *, draws: int = 2_000, seed: int = 0) -> dict:
    """The probe report: savings, realization, generalization, decision movement and fit, with readings."""
    targets = json.loads((probe / "targets.json").read_text(encoding="utf-8"))
    load = {path.stem: {row["prompt_id"]: row for row in read_jsonl(path) if "responses" in row}
            for path in sorted(samples.glob("*.jsonl")) if not path.stem.endswith(".fit")}
    fits = {path.stem.removesuffix(".fit"): {row["sample_id"]: row["logprob"] for row in read_jsonl(path)}
            for path in sorted(samples.glob("*.fit.jsonl"))}
    reference, null = load.pop("recipient"), load.pop("recipient-null")
    sets = {name: sorted(p for p, row in reference.items() if row["set"] == name) for name in SETS}
    thresholds = settings["thresholds"]
    report: dict = {"settings": settings, "prompts": {name: len(ids) for name, ids in sets.items()}, "students": {}}

    pooled = {p: {"responses": reference[p]["responses"] + null[p]["responses"]} for p in reference}
    recipient_tokens = {name: _mean_tokens(pooled, ids) for name, ids in sets.items()}
    report["recipient_mean_tokens"] = {name: float(values.mean()) for name, values in recipient_tokens.items()}

    def decisions(rows, prompt, view):
        return [VIEWS[view](r["decision"]) for r in rows[prompt]["responses"]]

    train_ids = sets["train"]
    fit_order = defaultdict(list)
    if (probe / "fit_rows.jsonl").exists():
        for row in read_jsonl(probe / "fit_rows.jsonl"):
            fit_order[row["prompt_id"]].append(row["sample_id"])
    implied = {}
    for arm in ARMS:
        cached_tokens = sum(sum(targets[p]["tokens"]) for p in train_ids)
        weighted = sum(sum(w * t for w, t in zip(targets[p]["weights"][arm], targets[p]["tokens"], strict=True))
                       for p in train_ids)
        implied[arm] = {
            "savings": 1 - weighted / cached_tokens,
            **{f"squared_distance_{view}": float(np.mean([
                target_squared_distance([VIEWS[view](d) for d in targets[p]["decisions"]], targets[p]["weights"][arm])
                for p in train_ids])) for view in VIEWS},
        }
    report["target_implied_on_train_prompts"] = implied
    bound = settings["equivalence_fraction"] * implied["ordinary"]["squared_distance_calls"]
    report["equivalence_bound_calls"] = bound

    report["null"] = {}
    for name, ids in sets.items():
        report["null"][name] = {
            view: float(np.mean([unbiased_squared_distance(decisions(null, p, view), decisions(reference, p, view))
                                 for p in ids]))
            for view in VIEWS
        }

    for student, rows in sorted(load.items()):
        arm = student.split("-seed")[0]
        entry: dict = {"arm": arm}
        savings = {}
        for name, ids in sets.items():
            tokens, base = _mean_tokens(rows, ids), recipient_tokens[name]
            savings[name] = (tokens, base)
            point = 1 - tokens.mean() / base.mean()
            interval = bootstrap(lambda idx, t=tokens, b=base: 1 - t[idx].mean() / b[idx].mean(), len(ids),
                                 draws=draws, seed=seed)
            entry[f"savings_{name}"] = {"point": float(point), "ci95": interval}
            for view in VIEWS:
                values = np.array([unbiased_squared_distance(decisions(rows, p, view), decisions(pooled, p, view))
                                   for p in ids])
                interval = bootstrap(lambda idx, v=values: v[idx].mean(), len(ids), draws=draws, seed=seed)
                movement = {"point": float(values.mean()), "ci95": interval}
                if view == "calls":
                    # Two separate flags: a small but detectable movement can also lie within the margin. The margin is
                    # a mechanism diagnostic, not a guarantee that accuracy is preserved.
                    movement["detectable_movement"] = bool(interval[0] > 0)
                    movement["within_margin"] = bool(interval[1] < bound)
                entry[f"squared_distance_{view}_{name}"] = movement
        if implied[arm]["savings"] > 0:
            tokens, base = savings["train"]
            ratio = (1 - tokens.mean() / base.mean()) / implied[arm]["savings"]
            target = implied[arm]["savings"]
            interval = bootstrap(lambda idx, t=tokens, b=base, s=target: (1 - t[idx].mean() / b[idx].mean()) / s,
                                 len(train_ids), draws=draws, seed=seed)
            entry["realization"] = {"point": float(ratio), "ci95": interval, "reading": reading(
                interval, high=thresholds["substantial"], low=thresholds["weak"],
                names=("substantially realized", "weakly realized", "inconclusive"))}
            entry["generalization"] = generalization(savings["train"], savings["heldout"], entry["savings_train"],
                                                     thresholds, draws=draws, seed=seed)
        if student in fits and "recipient" in fits:
            # Cached responses were written in the same per-prompt order as the targets' weights.
            student_fit, base_fit = fits[student], fits["recipient"]
            xs, ys, changes = [], [], []
            for p in train_ids:
                delta = np.array([student_fit[s] - base_fit[s] for s in fit_order[p]])
                log_w = np.log(np.maximum(np.array(targets[p]["weights"][arm]), 1e-12))
                changes.append(delta.mean())
                if np.ptp(log_w) > 0:
                    xs.append(log_w - log_w.mean())
                    ys.append(delta - delta.mean())
            entry["fit"] = {"mean_logprob_change": float(np.mean(changes))}
            if xs:
                x, y = np.concatenate(xs), np.concatenate(ys)
                entry["fit"]["slope_on_log_weight"] = float((x * y).sum() / (x * x).sum())
        verdicts = [r["verdict"] for p in sets["heldout"] for r in rows[p]["responses"] if r.get("verdict") != "unverifiable"]
        base_verdicts = [r["verdict"] for p in sets["heldout"] for r in pooled[p]["responses"]
                         if r.get("verdict") != "unverifiable"]
        if verdicts and base_verdicts:
            entry["heldout_call_accuracy"] = {"student": verdicts.count("match") / len(verdicts),
                                              "recipient": base_verdicts.count("match") / len(base_verdicts)}
        report["students"][student] = entry
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prompts = commands.add_parser("prompts")
    prompts.add_argument("--rollouts", type=Path, required=True)
    prompts.add_argument("--weights", type=Path, required=True)
    prompts.add_argument("--heldout", type=Path, required=True)
    prompts.add_argument("--train-prompts", type=int, default=400)
    prompts.add_argument("--output", type=Path, required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--probe", type=Path, required=True)
    sample.add_argument("--model", required=True)
    sample.add_argument("--seed", type=int, required=True, help="first of the n per-sample seeds s, s+1, ...")
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--responses", type=int, default=8)
    sample.add_argument("--temperature", type=float, default=0.6)
    sample.add_argument("--top-p", type=float, default=0.95)
    sample.add_argument("--top-k", type=int, default=20)
    sample.add_argument("--max-tokens", type=int, default=2_048)
    sample.add_argument("--max-model-len", type=int, default=8_192 + 2_048 + 8)
    fit = commands.add_parser("fit", help="teacher forcing of the cached responses (fit_rows.jsonl)")
    fit.add_argument("--probe", type=Path, required=True)
    fit.add_argument("--model", required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--max-num-batched-tokens", type=int, default=4_096)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--probe", type=Path, required=True)
    compare_parser.add_argument("--samples", type=Path, required=True)
    compare_parser.add_argument("--settings", type=json.loads, required=True, help="PROJECTION_PROBE as JSON")
    compare_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "prompts":
        print(json.dumps(build_prompts(args.rollouts, args.weights, args.heldout, args.output,
                                       train_prompts=args.train_prompts)))
    elif args.command == "sample":
        from vllm import LLM, SamplingParams

        prompts = read_jsonl(args.probe / "prompts.jsonl")
        llm = LLM(model=args.model, max_model_len=args.max_model_len, seed=args.seed, dtype="bfloat16",
                  gpu_memory_utilization=0.9, enable_prefix_caching=False)
        sampling = SamplingParams(n=args.responses, temperature=args.temperature, top_p=args.top_p,
                                  top_k=args.top_k, max_tokens=args.max_tokens, seed=args.seed)
        write_jsonl(sample_rows(llm, sampling, prompts), args.output)
        print(f"sampled {len(prompts)} prompts -> {args.output}")
    elif args.command == "fit":
        from vllm import LLM, SamplingParams

        # A separate engine: vLLM materializes full-vocabulary float32 log-probabilities for every scheduled prompt
        # token, which its memory profile does not reserve. Small steps and a lower memory fraction leave room.
        rows = read_jsonl(args.probe / "fit_rows.jsonl")
        longest = max(len(row["prompt_tokens"]) + len(row["response_tokens"]) for row in rows)
        llm = LLM(model=args.model, max_model_len=longest + 1, seed=0, dtype="bfloat16", gpu_memory_utilization=0.7,
                  max_num_batched_tokens=args.max_num_batched_tokens, enable_prefix_caching=False)
        scoring = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0)
        write_jsonl(teacher_force(llm, scoring, rows), args.output)
        print(f"teacher-forced {len(rows)} responses -> {args.output}")
    else:
        report = compare(args.probe, args.samples, args.settings)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
