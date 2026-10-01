#!/usr/bin/env python3
"""What is the reasoning DECS removes worth, prompt by prompt, and can a cheap label predict it?

A difficulty gate assumes that where the behavior policy already succeeds, shortening reasoning is safe. That
is a claim about P(success | full reasoning) − P(success | shortened reasoning), which success with full
reasoning alone cannot establish. This probe measures that difference directly, with a concrete shortening
intervention, on held-out LoopTool prompts:

- ``select`` (CPU): canonical LoopTool rows never used in training (neither their id nor their rendered prompt
  is in the training prompts), rendered exactly as the training prompts were, with tool-call references only.
  Text references cannot be verified automatically. They are stratified by conversation kind and drawn in a
  fixed hash order.
- ``rollout`` (GPU): each policy samples complete assistant messages. ``base``, the cache's behavior policy
  with the cache's decoding, gives the label a gate would use. ``full`` (acc-legacy) and ``short``
  (acc-legacy+decs), with BFCL decoding, give success with and without the reasoning DECS removes. Every
  finished message is scored by :func:`prompt_difficulty.verify`: exact tool calls against the reference. A
  message cut at the budget is ``unfinished`` and counts as a miss.
- ``evaluate`` (CPU): per prompt, success and tokens under each policy, and the shortening effect
  Δ = success(short) − success(full). Effects are reported by the base label (``verified_solved``: every base
  sample matched), by base pass rate, and by conversation kind. The key contrast is Δ(solved) − Δ(not solved),
  with prompt-bootstrap intervals.

This is still a single-message outcome, not multi-turn task success. ``evaluation/bfcl_difficulty_strata.py``
is the full-interaction counterpart on BFCL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.prompt_difficulty import verify

DECODING = {
    # The cache's behavior-policy recipe (LoopTool Offline Direct-OPD rollouts).
    "cache": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 2_048},
    # The BFCL evaluation recipe, with the budget used by the reflection probe.
    "bfcl": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 8_192},
}
DEFAULT_QUOTAS = "multi_turn=400,single_turn=200"
MAX_PROMPT_TOKENS = 8_192


def selection_key(prompt_id: str, seed: int) -> str:
    return hashlib.sha256(f"value-probe-{seed}\0{prompt_id}".encode()).hexdigest()


def select_prompts(canonical: Path, training: Path, tokenizer, *, quotas: dict[str, int], seed: int) -> list[dict]:
    """Held-out, call-referenced prompts in hash order until every conversation-kind quota is filled."""
    from data_curation.prepare_direct_opd_looptool import render_prompt

    trained = pq.read_table(training, columns=["prompt_id", "prompt"]).to_pylist()
    trained_ids = {row["prompt_id"] for row in trained}
    trained_prompts = {hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest() for row in trained}
    with canonical.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [row for row in rows if row["id"] not in trained_ids and row["target"]["tool_calls"]]
    rows.sort(key=lambda row: selection_key(row["id"], seed))
    selected, counts, seen = [], dict.fromkeys(quotas, 0), set()
    for row in rows:
        kind = row["metadata"]["conversation_kind"]
        if kind not in quotas or counts[kind] >= quotas[kind]:
            continue
        prompt = render_prompt(tokenizer, row)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if digest in trained_prompts or digest in seen:
            continue
        tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(tokens) > MAX_PROMPT_TOKENS:
            continue
        seen.add(digest)
        counts[kind] += 1
        selected.append(
            {
                "prompt_id": row["id"],
                "conversation_kind": kind,
                "target_kind": row["metadata"]["target_kind"],
                "target": row["target"],
                "prompt_token_ids": tokens,
            }
        )
        if all(counts[k] >= quotas[k] for k in quotas):
            break
    short = {k: quotas[k] - counts[k] for k in quotas if counts[k] < quotas[k]}
    if short:
        raise ValueError(f"not enough held-out prompts for quotas: {short}")
    return selected


def rollout(prompts: list[dict], args) -> list[dict]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    decoding = dict(DECODING[args.decoding])
    budget = decoding.pop("max_tokens")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    think_close = tokenizer.convert_tokens_to_ids("</think>")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=MAX_PROMPT_TOKENS + budget + 8,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        seed=args.seed,
    )
    requests = [{"prompt_token_ids": item["prompt_token_ids"]} for item in prompts]
    params = [
        SamplingParams(
            n=args.samples,
            max_tokens=budget,
            stop_token_ids=[im_end],
            seed=int(selection_key(f"{item['prompt_id']}:{args.policy}", args.seed)[:8], 16),
            **decoding,
        )
        for item in prompts
    ]
    records = []
    for item, output in zip(prompts, llm.generate(requests, params, use_tqdm=True)):
        for sample, completion in enumerate(output.outputs):
            tokens = list(completion.token_ids)
            finished = completion.finish_reason == "stop"
            text = tokenizer.decode(tokens, skip_special_tokens=False).replace("<|im_end|>", "")
            records.append(
                {
                    "prompt_id": item["prompt_id"],
                    "policy": args.policy,
                    "sample": sample,
                    "finished": finished,
                    "outcome": verify(text, item["target"]) if finished else "unfinished",
                    "tokens": len(tokens),
                    "think_tokens": tokens.index(think_close) if think_close in tokens else len(tokens),
                }
            )
    return records


def per_prompt(prompts: list[dict], records: list[dict], policies: tuple[str, ...]) -> list[dict]:
    """Per prompt and policy: success rate (unfinished counted as misses), mean tokens, unfinished share."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["prompt_id"], record["policy"]), []).append(record)
    values = []
    for item in prompts:
        row = {"prompt_id": item["prompt_id"], "conversation_kind": item["conversation_kind"]}
        for policy in policies:
            samples = grouped.get((item["prompt_id"], policy))
            if not samples:
                raise ValueError(f"no {policy} samples for {item['prompt_id']}")
            row[policy] = {
                "samples": len(samples),
                "success": float(np.mean([s["outcome"] == "match" for s in samples])),
                "tokens": float(np.mean([s["tokens"] for s in samples])),
                "unfinished": float(np.mean([s["outcome"] == "unfinished" for s in samples])),
            }
        values.append(row)
    return values


def bootstrap_mean(values: np.ndarray, *, draws: int, rng) -> dict:
    boot = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)]
    return {
        "mean": float(values.mean()),
        "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
        "prompts": len(values),
    }


def evaluate(
    prompts: list[dict],
    records: list[dict],
    *,
    label_policy: str = "base",
    full: str = "full",
    short: str = "short",
    draws: int = 2_000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    values = per_prompt(prompts, records, (label_policy, full, short))
    solved = np.array([v[label_policy]["success"] == 1.0 for v in values])
    base_rate = np.array([v[label_policy]["success"] for v in values])
    effect = np.array([v[short]["success"] - v[full]["success"] for v in values])
    tokens_full = np.array([v[full]["tokens"] for v in values])
    tokens_short = np.array([v[short]["tokens"] for v in values])
    kinds = np.array([v["conversation_kind"] for v in values])

    def describe(keep: np.ndarray) -> dict:
        if not keep.any():
            return {"prompts": 0}
        return {
            "prompts": int(keep.sum()),
            "success_full": float(np.mean([v[full]["success"] for v, k in zip(values, keep) if k])),
            "success_short": float(np.mean([v[short]["success"] for v, k in zip(values, keep) if k])),
            "shortening_effect": bootstrap_mean(effect[keep], draws=draws, rng=rng),
            "total_token_ratio": float(tokens_short[keep].sum() / tokens_full[keep].sum() - 1),
            "unfinished_full": float(np.mean([v[full]["unfinished"] for v, k in zip(values, keep) if k])),
            "unfinished_short": float(np.mean([v[short]["unfinished"] for v, k in zip(values, keep) if k])),
        }

    groups = {"all": np.ones(len(values), dtype=bool), "solved": solved, "not_solved": ~solved}
    groups |= {f"kind={kind}": kinds == kind for kind in sorted(set(kinds))}
    groups |= {f"kind={kind}:solved": (kinds == kind) & solved for kind in sorted(set(kinds))}
    groups |= {f"kind={kind}:not_solved": (kinds == kind) & ~solved for kind in sorted(set(kinds))}
    for rate in sorted(set(base_rate.tolist())):
        groups[f"base_pass_rate={rate:.2f}"] = base_rate == rate

    boot = []
    for _ in range(draws):
        index = rng.integers(0, len(values), len(values))
        a, b = effect[index][solved[index]], effect[index][~solved[index]]
        boot.append(a.mean() - b.mean() if len(a) and len(b) else np.nan)
    contrast = float(effect[solved].mean() - effect[~solved].mean()) if solved.any() and (~solved).any() else None
    return {
        "prompts": len(values),
        "policies": {"label": label_policy, "full": full, "short": short},
        "groups": {name: describe(keep) for name, keep in groups.items()},
        "solved_minus_not_solved_effect": {
            "mean": contrast,
            "ci95": np.nanpercentile(boot, [2.5, 97.5]).tolist() if contrast is not None else [None, None],
        },
        # Does the label predict the full policy's own success (the precondition for it predicting safety)?
        "label_predicts_full_success": {
            name: float(np.mean([v[full]["success"] for v, k in zip(values, keep) if k])) if keep.any() else None
            for name, keep in (("solved", solved), ("not_solved", ~solved))
        },
    }


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
    select.add_argument("--canonical", required=True, type=Path)
    select.add_argument("--training-prompts", required=True, type=Path, help="prompts.parquet of the cache")
    select.add_argument("--tokenizer", required=True, help="model directory with the chat template")
    select.add_argument("--output", required=True, type=Path)
    select.add_argument("--quotas", default=DEFAULT_QUOTAS)
    select.add_argument("--seed", type=int, default=0)
    run = stages.add_parser("rollout")
    run.add_argument("--prompts", required=True, type=Path)
    run.add_argument("--policy", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--decoding", required=True, choices=sorted(DECODING))
    run.add_argument("--samples", type=int, required=True)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--rank", type=int, default=0)
    run.add_argument("--world-size", type=int, default=1)
    run.add_argument("--seed", type=int, default=0)
    report = stages.add_parser("evaluate")
    report.add_argument("--prompts", required=True, type=Path)
    report.add_argument("--records", required=True, type=Path, nargs="+")
    report.add_argument("--output", required=True, type=Path)
    report.add_argument("--bootstrap", type=int, default=2_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "select":
        from transformers import AutoTokenizer

        quotas = {kind: int(count) for kind, count in (item.split("=") for item in args.quotas.split(","))}
        prompts = select_prompts(
            args.canonical,
            args.training_prompts,
            AutoTokenizer.from_pretrained(args.tokenizer),
            quotas=quotas,
            seed=args.seed,
        )
        write_jsonl(prompts, args.output)
        print(f"SELECTED {len(prompts)} held-out prompts")
    elif args.stage == "rollout":
        prompts = [p for i, p in enumerate(read_jsonl(args.prompts)) if i % args.world_size == args.rank]
        records = rollout(prompts, args)
        write_jsonl(records, args.output)
        print(f"ROLLOUT {args.policy} rank {args.rank}: {len(prompts)} prompts, {len(records)} samples")
    else:
        records = [record for path in args.records for record in read_jsonl(path)]
        result = evaluate(read_jsonl(args.prompts), records, draws=args.bootstrap)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        keys = ("solved_minus_not_solved_effect", "label_predicts_full_success")
        print(json.dumps({key: result[key] for key in keys}, indent=2))
        for name in ("all", "solved", "not_solved"):
            group = result["groups"][name]
            if group["prompts"]:
                effect = group["shortening_effect"]
                low, high = effect["ci95"]
                print(
                    f"{name}: n={group['prompts']} full {group['success_full']:.3f} short {group['success_short']:.3f} "
                    f"effect {effect['mean']:+.3f} [{low:+.3f}, {high:+.3f}] "
                    f"tokens {100 * group['total_token_ratio']:+.1f}%"
                )


if __name__ == "__main__":
    main()
