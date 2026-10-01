#!/usr/bin/env python3
"""What is a reflection worth? Force "Wait" vs "So" at cached reflection forks and compare the outcomes.

At a reflection fork, the student is choosing between a reflection marker ("Wait", "Hmm", "But", ...) and
a conclusion marker ("So", "Therefore", ...). For each selected fork, a policy continues the cached prefix
after each forced marker, a fixed number of times, to the end of the assistant message. Each finished
message is scored against the LoopTool reference by :func:`mc_advantage_probes.score_message`: its tool
calls must equal the reference's as an unordered multiset (for a text reference, no call). That is a
message-level call-structure outcome, not multi-turn task success.

The question is whether a response's first reflection fork, which is the first check after the new tool
result or user message it follows, is worth more than later ones. The gate of
``acc-legacy+decs-protect-first`` assumes so. Forks are therefore stratified by conversation kind and by
order (first vs later) and drawn from a subset of prompt folds (default two of five), keeping the rest held
out for any later gate tuning. For each stratum the report gives the
reflect-minus-conclude change in call-match rate and in generated tokens, with prompt-clustered bootstrap
intervals, plus first-minus-later differences. Messages that hit the token budget count as non-matching
and are reported as censored.

Decoding follows the BFCL recipe (temperature 0.6, top-p 0.95, top-k 20), so values refer to the policy
as it is evaluated.

Stages: ``select`` (CPU), ``rollout`` (GPU, vLLM), ``evaluate`` (CPU).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.mc_advantage_probes import reference_targets, score_message
from data_curation.shift_geometry import iter_joined_rows, prompt_unit, read_shard_fields, shard_names
from data_curation.shift_states import STATE_CODE, StateLabeler, TokenTable

ALTERNATIVES = ("reflect", "conclude")
GROUPS = {"reflect": "reflection", "conclude": "conclusion"}
DEFAULT_QUOTAS = "multi_turn:first=300,multi_turn:later=300,single_turn:first=150,single_turn:later=150"
DECODING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def parse_quotas(text: str) -> dict[tuple[str, str], int]:
    quotas = {}
    for item in text.split(","):
        key, count = item.split("=")
        kind, order = key.split(":")
        if order not in ("first", "later"):
            raise ValueError(f"order must be first or later, got {order!r}")
        quotas[(kind, order)] = int(count)
    return quotas


def best_member(labeler: StateLabeler, group: str, candidates: np.ndarray, probs: np.ndarray) -> int | None:
    """The slot of the most likely candidate in ``group``, if any."""
    slots = np.nonzero(labeler.member(group, candidates))[0]
    return int(slots[np.argmax(probs[slots])]) if slots.size else None


def select_states(
    base: Path,
    labeler: StateLabeler,
    metadata: dict,
    *,
    folds: set[int],
    quotas: dict[tuple[str, str], int],
    min_mass: float,
    seed: int,
) -> list[dict]:
    """Reflection forks of the given prompt folds where both a reflection and a conclusion marker are live."""
    pool: dict[tuple[str, str], list[tuple[float, dict]]] = {key: [] for key in quotas}
    for shard in shard_names(base):
        fields = read_shard_fields(base / shard, ("sample_id", "prompt_tokens"))
        for index, row in enumerate(iter_joined_rows(base, {}, shards=[shard])):
            if fields["sample_id"][index] != row.sample_id:
                raise ValueError(f"{shard}: prompt tokens are not row-aligned")
            if row.fold not in folds:
                continue
            kind = metadata.get(row.prompt_id, {}).get("conversation", "unknown")
            probs = np.exp(row.behavior_log_probs)
            codes = labeler.label(row.response_tokens, row.candidate_ids, probs)
            forks = np.nonzero(codes == STATE_CODE["think_reflection_fork"])[0]
            for order, position in enumerate(forks):
                key = (kind, "first" if order == 0 else "later")
                if key not in pool or not row.loss_mask[position]:
                    continue
                candidates, p = row.candidate_ids[position], probs[position]
                slots = {name: best_member(labeler, GROUPS[name], candidates, p) for name in ALTERNATIVES}
                if any(slot is None or p[slot] < min_mass for slot in slots.values()):
                    continue
                prompt = [int(token) for token in fields["prompt_tokens"][index]]
                pool[key].append(
                    (
                        prompt_unit(f"{row.sample_id}:{position}", f"reflection-probe-{seed}"),
                        {
                            "sample_id": row.sample_id,
                            "prompt_id": row.prompt_id,
                            "kind": kind,
                            "order": key[1],
                            "fork_index": int(order),
                            "position": int(position),
                            "prefix_tokens": prompt + [int(t) for t in row.response_tokens[:position]],
                            "response_prefix_length": int(position),
                            "forced": {name: int(candidates[slot]) for name, slot in slots.items()},
                            "behavior_probs": {name: float(p[slot]) for name, slot in slots.items()},
                        },
                    )
                )
    selected = []
    for key, quota in quotas.items():
        selected += [state for _, state in sorted(pool[key], key=lambda pair: pair[0])[:quota]]
    return selected


def rollout(states: list[dict], targets: dict[str, dict], args) -> list[dict]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    think_close = tokenizer.convert_tokens_to_ids("</think>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    missing = {state["prompt_id"] for state in states} - set(targets)
    if missing:
        raise ValueError(f"{len(missing)} probed prompts have no LoopTool reference")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        seed=args.seed,
    )
    requests, params, meta = [], [], []
    for state in states:
        budget = max(args.max_response_tokens - state["response_prefix_length"] - 1, 1)
        for name in ALTERNATIVES:
            requests.append({"prompt_token_ids": state["prefix_tokens"] + [state["forced"][name]]})
            unit = prompt_unit(f"{state['sample_id']}:{state['position']}:{name}", f"reflection-probe-{args.seed}")
            seed = int(unit * 2**31)
            options = {"n": args.samples, "max_tokens": budget, "stop_token_ids": [im_end], "seed": seed, **DECODING}
            params.append(SamplingParams(**options))
            meta.append((state, name))
    records = []
    for (state, name), output in zip(meta, llm.generate(requests, params, use_tqdm=True)):
        response_prefix = state["prefix_tokens"][len(state["prefix_tokens"]) - state["response_prefix_length"] :]
        for sample, completion in enumerate(output.outputs):
            generated = [state["forced"][name], *completion.token_ids]
            finished = completion.finish_reason == "stop"
            think = generated.index(think_close) if think_close in generated else len(generated)
            message = tokenizer.decode(response_prefix + generated, skip_special_tokens=False).replace("<|im_end|>", "")
            if finished:
                calls, match, malformed = score_message(message, targets[state["prompt_id"]])
            else:  # a message cut at the budget cannot match; it is censored, not malformed
                calls, match, malformed = message.count("<tool_call>"), False, False
            records.append(
                {
                    "sample_id": state["sample_id"],
                    "position": state["position"],
                    "alternative": name,
                    "sample": sample,
                    "finished": finished,
                    "think_tokens": think,
                    "total_tokens": len(generated),
                    "tool_calls": calls,
                    "call_match": bool(match),
                    "malformed": bool(malformed),
                }
            )
    return records


def state_values(states: list[dict], records: list[dict]) -> list[dict]:
    """Per state and alternative: match rate (censored count as misses), mean tokens, censoring rate."""
    grouped: dict[tuple[str, int, str], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["sample_id"], record["position"], record["alternative"]), []).append(record)
    values = []
    for state in states:
        item = {key: state[key] for key in ("sample_id", "prompt_id", "kind", "order", "position")}
        for name in ALTERNATIVES:
            samples = grouped.get((state["sample_id"], state["position"], name), [])
            if not samples:
                raise ValueError(f"no records for {state['sample_id']}:{state['position']} {name}")
            item[name] = {
                "samples": len(samples),
                "match": float(np.mean([s["call_match"] for s in samples])),
                "tokens": float(np.mean([s["total_tokens"] for s in samples])),
                "think_tokens": float(np.mean([s["think_tokens"] for s in samples])),
                "censored": float(np.mean([not s["finished"] for s in samples])),
                "malformed": float(np.mean([s["malformed"] for s in samples])),
            }
        values.append(item)
    return values


def clustered_mean(items: list[dict], value, *, draws: int, rng) -> dict:
    """Mean of ``value(item)`` with a bootstrap interval that resamples prompts."""
    by_prompt: dict[str, list[float]] = {}
    for item in items:
        by_prompt.setdefault(item["prompt_id"], []).append(value(item))
    prompts = list(by_prompt)
    point = float(np.mean([v for vs in by_prompt.values() for v in vs]))
    boot = []
    for _ in range(draws):
        chosen = rng.choice(len(prompts), size=len(prompts), replace=True)
        boot.append(np.mean([v for i in chosen for v in by_prompt[prompts[i]]]))
    interval = np.percentile(boot, [2.5, 97.5]).tolist()
    return {"mean": point, "ci95": interval, "states": len(items), "prompts": len(prompts)}


def clustered_difference(first: list[dict], later: list[dict], value, *, draws: int, rng) -> dict:
    """mean(first) − mean(later), resampling prompts jointly (a prompt can contribute to both)."""
    prompts = sorted({item["prompt_id"] for item in first + later})
    index = {prompt: i for i, prompt in enumerate(prompts)}
    groups = [[[] for _ in prompts], [[] for _ in prompts]]
    for side, items in enumerate((first, later)):
        for item in items:
            groups[side][index[item["prompt_id"]]].append(value(item))

    def difference(chosen) -> float:
        a = [v for i in chosen for v in groups[0][i]]
        b = [v for i in chosen for v in groups[1][i]]
        return float(np.mean(a) - np.mean(b)) if a and b else np.nan

    boot = [difference(rng.choice(len(prompts), size=len(prompts), replace=True)) for _ in range(draws)]
    return {"mean": difference(range(len(prompts))), "ci95": np.nanpercentile(boot, [2.5, 97.5]).tolist()}


def evaluate(states: list[dict], records: list[dict], *, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values = state_values(states, records)
    measures = {
        "match_reflect": lambda v: v["reflect"]["match"],
        "match_conclude": lambda v: v["conclude"]["match"],
        "delta_match": lambda v: v["reflect"]["match"] - v["conclude"]["match"],
        "delta_tokens": lambda v: v["reflect"]["tokens"] - v["conclude"]["tokens"],
        "censored_reflect": lambda v: v["reflect"]["censored"],
        "censored_conclude": lambda v: v["conclude"]["censored"],
    }
    strata = {}
    for kind in sorted({v["kind"] for v in values}):
        for order in ("first", "later"):
            items = [v for v in values if v["kind"] == kind and v["order"] == order]
            if items:
                strata[f"{kind}:{order}"] = {
                    name: clustered_mean(items, measure, draws=draws, rng=rng) for name, measure in measures.items()
                }
        first = [v for v in values if v["kind"] == kind and v["order"] == "first"]
        later = [v for v in values if v["kind"] == kind and v["order"] == "later"]
        if first and later:
            strata[f"{kind}:first_minus_later"] = {
                name: clustered_difference(first, later, measures[name], draws=draws, rng=rng)
                for name in ("delta_match", "delta_tokens")
            }
    return {"states": len(values), "records": len(records), "decoding": DECODING, "strata": strata}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    select = stages.add_parser("select")
    select.add_argument("--base", required=True, type=Path)
    select.add_argument("--tokenizer-json", required=True, type=Path)
    select.add_argument("--prompt-metadata", required=True, type=Path)
    select.add_argument("--output", required=True, type=Path)
    select.add_argument("--folds", default="0,1", help="comma-separated prompt folds (of 5) to draw forks from")
    select.add_argument("--quotas", default=DEFAULT_QUOTAS)
    select.add_argument("--min-mass", type=float, default=0.02, help="minimum behavior prob of each forced marker")
    select.add_argument("--seed", type=int, default=0)
    run = stages.add_parser("rollout")
    run.add_argument("--states", required=True, type=Path)
    run.add_argument("--prompts", required=True, type=Path)
    run.add_argument("--canonical", required=True, type=Path)
    run.add_argument("--model", required=True)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--samples", type=int, default=8)
    run.add_argument("--max-response-tokens", type=int, default=8_192)
    run.add_argument("--max-model-len", type=int, default=8_192 + 8_192 + 8)
    run.add_argument("--rank", type=int, default=0)
    run.add_argument("--world-size", type=int, default=1)
    run.add_argument("--seed", type=int, default=0)
    report = stages.add_parser("evaluate")
    report.add_argument("--states", required=True, type=Path)
    report.add_argument("--records", required=True, type=Path, nargs="+")
    report.add_argument("--output", required=True, type=Path)
    report.add_argument("--bootstrap", type=int, default=2_000)
    report.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "select":
        labeler = StateLabeler(TokenTable.from_tokenizer_json(args.tokenizer_json))
        metadata = json.loads(args.prompt_metadata.read_text(encoding="utf-8"))
        states = select_states(
            args.base, labeler, metadata, folds={int(f) for f in args.folds.split(",")},
            quotas=parse_quotas(args.quotas), min_mass=args.min_mass, seed=args.seed,
        )
        write_jsonl(states, args.output)
        counts = {}
        for state in states:
            counts[f"{state['kind']}:{state['order']}"] = counts.get(f"{state['kind']}:{state['order']}", 0) + 1
        print(f"SELECTED {len(states)} states {json.dumps(counts, sort_keys=True)}")
    elif args.stage == "rollout":
        states = [s for i, s in enumerate(read_jsonl(args.states)) if i % args.world_size == args.rank]
        records = rollout(states, reference_targets(args.prompts, args.canonical), args)
        write_jsonl(records, args.output)
        print(f"ROLLOUT rank {args.rank}: {len(states)} states, {len(records)} records")
    else:
        records = [record for path in args.records for record in read_jsonl(path)]
        report = evaluate(read_jsonl(args.states), records, draws=args.bootstrap, seed=args.seed)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report["strata"], indent=2))


if __name__ == "__main__":
    main()
