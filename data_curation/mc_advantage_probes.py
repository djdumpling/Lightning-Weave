#!/usr/bin/env python3
"""Monte Carlo advantage probes: the student's own message-level cost ground truth at cached forks.

Motivation: under KL-regularized RL a converged policy's log-ratio to its
reference is a scaled advantage, so a transferred efficiency shift should
resemble the frozen student's cost advantage of each candidate. These probes
estimate that advantage without any anchor. For each selected state s (a
response prefix from the sealed cache) and each probed candidate a, the frozen
student continues ``prefix + a`` to the end of the assistant message a fixed
number of times, recording:

- ``think_tokens``: reasoning tokens still to come (0 once ``</think>`` is past);
- ``total_tokens``: generated tokens including a;
- ``tool_calls``: calls in the finished message;
- ``call_match``: the message's tool calls equal the LoopTool reference as an
  unordered multiset. For a text-only reference this checks only that no call
  was made, not the text, so it is a call-structure outcome, not correctness;
- ``malformed``: the message does not parse.

Scope: continuations stop at the end of the assistant message; there is no
environment, tool result, or later turn. The probes measure message-level
token and call cost, not interaction-level efficiency. A continuation that hits
the response cap is censored: token outcomes are used only for states where
every continuation finished, and censoring rates are reported per state type
and candidate.

Candidates: each state's defining alternatives are always probed (stop vs.
continue, reflection vs. not, call vs. text, end vs. call again), then the most
likely remaining candidates up to ``--candidates``. The probed behavior mass is
recorded and must reach ``--min-probed-mass``. Advantages are centered under
the behavior distribution renormalized over the probed candidates.

Inference: the split-half noise ceiling, correlation intervals from a
prompt-level bootstrap, and a coefficient fit on one half of the prompts,
evaluated on the other.

Stages: ``select`` (CPU), ``rollout`` (GPU, vLLM), ``evaluate`` (CPU).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import write_json, write_parquet
from data_curation.looptool import cached_prompt_rows
from data_curation.shift_geometry import (
    bucket_probs,
    direction_vectors,
    entropy,
    iter_joined_rows,
    load_spec,
    load_untrained,
    prompt_unit,
    read_shard_fields,
    shard_names,
)
from data_curation.shift_states import STATE_TYPES, StateLabeler, TokenTable

DEFAULT_QUOTAS = {
    "think_stop_fork": 400,
    "think_reflection_fork": 300,
    "think_body": 300,
    "act_vs_talk": 150,
    "tool_name": 100,
    "tool_args": 100,
    "call_boundary": 100,
    "end_of_message": 100,
    "post_text": 50,
}
# The candidate groups whose best member must be probed at each state type (None: any other token).
DEFINING_ALTERNATIVES = {
    "think_stop_fork": ("single_newline", "paragraph_break"),
    "think_reflection_fork": ("reflection", None),
    "act_vs_talk": ("tool_call", None),
    "call_boundary": ("im_end", None),
    "end_of_message": ("im_end", None),
}
TOKEN_METRICS = ("think_tokens", "total_tokens")
OUTCOME_METRICS = ("tool_calls", "call_match", "malformed")
METRICS = TOKEN_METRICS + OUTCOME_METRICS


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def probe_candidates(labeler: StateLabeler, state: str, candidates: np.ndarray, topk: np.ndarray, width: int) -> list[int]:
    """Slots to probe: the best member of each defining group, then the most likely others."""
    order = [int(k) for k in np.argsort(-topk)]
    chosen: list[int] = []
    for group in DEFINING_ALTERNATIVES.get(state, ()):
        if group is None:
            members = [k for k in order if k not in chosen]
        else:
            members = [k for k in order if labeler.member(group, candidates[k : k + 1])[0]]
        if members and members[0] not in chosen:
            chosen.append(members[0])
    chosen += [k for k in order if k not in chosen][: max(width - len(chosen), 0)]
    return chosen


def select_states(
    base: Path,
    labeler: StateLabeler,
    *,
    fold: int,
    quotas: dict[str, int],
    candidates: int,
    min_candidate_mass: float,
    min_probed_mass: float,
    min_entropy: float,
    seed: int,
) -> list[dict]:
    """Deterministically sample forking states of held-out prompts, stratified by state type."""
    pool: dict[str, list[tuple[float, dict]]] = {name: [] for name in quotas}
    table = labeler.table
    for shard in shard_names(base):
        prompts = read_shard_fields(base / shard, ("prompt_tokens",))["prompt_tokens"]
        for index, row in enumerate(iter_joined_rows(base, {}, shards=[shard])):
            if row.fold != fold:
                continue
            probs = bucket_probs(row.behavior_log_probs)
            codes = labeler.label(row.response_tokens, row.candidate_ids, np.exp(row.behavior_log_probs))
            response = row.response_tokens.tolist()
            for position in np.nonzero(row.loss_mask & (entropy(probs) >= min_entropy))[0]:
                state = STATE_TYPES[codes[position]]
                if state not in pool:
                    continue
                topk = np.exp(row.behavior_log_probs[position])
                slots = [
                    k
                    for k in probe_candidates(labeler, state, row.candidate_ids[position], topk, candidates)
                    if topk[k] >= min_candidate_mass
                ]
                probed_mass = float(topk[slots].sum())
                if len(slots) < 2 or probed_mass < min_probed_mass:
                    continue
                prefix = response[:position]
                pool[state].append(
                    (
                        prompt_unit(f"{row.sample_id}:{position}", f"probe-select-{seed}"),
                        {
                            "sample_id": row.sample_id,
                            "prompt_id": row.prompt_id,
                            "position": int(position),
                            "state_type": state,
                            "prefix_tokens": [int(t) for t in prompts[index]] + [int(t) for t in prefix],
                            "response_prefix_length": int(position),
                            "candidate_slots": [int(k) for k in slots],
                            "candidate_ids": [int(row.candidate_ids[position][k]) for k in slots],
                            "behavior_probs": [float(topk[k]) for k in slots],
                            "probed_mass": probed_mass,
                            "in_think": table.think in prefix and table.think_close not in prefix,
                        },
                    )
                )
    selected = []
    for state, quota in quotas.items():
        selected += [item for _, item in sorted(pool[state], key=lambda pair: pair[0])[:quota]]
    return selected


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------


def reference_targets(prompts_path: Path, canonical_path: Path) -> dict[str, dict]:
    """prompt_id -> the LoopTool reference assistant turn."""
    return {p: row["target"] for p, row in cached_prompt_rows(prompts_path, canonical_path).items()}


def score_message(text: str, target: dict) -> tuple[int, bool, bool]:
    """(tool calls, call-structure match, malformed) of a finished message against the reference."""
    from data_curation.looptool import Reject, canonical_json, parse_assistant

    try:
        _, calls = parse_assistant(text, "probe", [])
    except Reject:
        return text.count("<tool_call>"), False, True
    produced = sorted(canonical_json({"name": call["name"], "arguments": call["arguments"]}) for call in calls)
    expected = sorted(canonical_json(call["function"]) for call in target["tool_calls"])
    return len(calls), produced == expected, False


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
    requests, meta = [], []
    for state_index, state in enumerate(states):
        remaining = args.max_response_tokens - state["response_prefix_length"] - 1
        for candidate_index, token in enumerate(state["candidate_ids"]):
            requests.append({"prompt_token_ids": state["prefix_tokens"] + [token]})
            meta.append((state_index, candidate_index, max(remaining, 1)))
    params = [
        SamplingParams(
            n=args.samples,
            temperature=1.0,
            top_p=1.0,
            max_tokens=remaining,
            stop_token_ids=[im_end],
            seed=int(prompt_unit(f"{states[s]['sample_id']}:{states[s]['position']}:{c}", f"probe-{args.seed}") * 2**31),
        )
        for s, c, remaining in meta
    ]
    records = []
    for (state_index, candidate_index, _), output in zip(meta, llm.generate(requests, params, use_tqdm=True)):
        state = states[state_index]
        length = state["response_prefix_length"]
        prefix_response = state["prefix_tokens"][len(state["prefix_tokens"]) - length :]
        for sample_index, completion in enumerate(output.outputs):
            generated = [state["candidate_ids"][candidate_index]] + list(completion.token_ids)
            finished = completion.finish_reason == "stop"
            think_tokens = (generated.index(think_close) if think_close in generated else len(generated)) if state["in_think"] else 0
            message = tokenizer.decode(prefix_response + generated, skip_special_tokens=False).replace("<|im_end|>", "")
            # A truncated message cannot match; it is censored, not malformed.
            if finished:
                calls, match, malformed = score_message(message, targets[state["prompt_id"]])
            else:
                calls, match, malformed = message.count("<tool_call>"), False, False
            records.append(
                {
                    "sample_id": state["sample_id"],
                    "position": state["position"],
                    "candidate_index": candidate_index,
                    "sample_index": sample_index,
                    "think_tokens": think_tokens,
                    "total_tokens": len(generated),
                    "tool_calls": calls,
                    "call_match": bool(match),
                    "malformed": bool(malformed),
                    "finished": finished,
                }
            )
    return records


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def advantages(states: list[dict], records: list[dict], samples: int) -> dict[tuple[str, int], dict]:
    """Per state: probe weights and centered advantages (all samples, and two halves) per metric.

    Every probed candidate must have exactly ``samples`` continuations. Token
    metrics are defined only when all of a state's continuations finished.
    """
    by_state: dict[tuple[str, int], dict[int, list[dict]]] = {}
    for record in records:
        key = (record["sample_id"], record["position"])
        by_state.setdefault(key, {}).setdefault(record["candidate_index"], []).append(record)
    out = {}
    for state in states:
        key = (state["sample_id"], state["position"])
        groups = by_state.get(key, {})
        counts = [len(groups.get(c, [])) for c in range(len(state["candidate_ids"]))]
        if counts != [samples] * len(counts):
            raise ValueError(f"state {key}: continuation counts {counts}, expected {samples} each")
        ordered = [sorted(groups[c], key=lambda r: r["sample_index"]) for c in range(len(counts))]
        weights = np.asarray(state["behavior_probs"], dtype=np.float64)
        weights /= weights.sum()
        finished = np.asarray([[r["finished"] for r in samples_] for samples_ in ordered])
        entry = {
            "state": state,
            "weights": weights,
            "finish_rate": finished.mean(axis=1).tolist(),
            "uncensored": bool(finished.all()),
        }
        for metric in METRICS:
            if metric in TOKEN_METRICS and not entry["uncensored"]:
                continue
            values = np.asarray([[float(r[metric]) for r in samples_] for samples_ in ordered])
            for label, columns in (("full", slice(None)), ("half_a", slice(0, None, 2)), ("half_b", slice(1, None, 2))):
                means = values[:, columns].mean(axis=1)
                entry[f"{metric}:{label}"] = means - weights @ means
        out[key] = entry
    return out


def fisher_terms(pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Per-state (Σ w u v, Σ w u², Σ w v²), with u and v already centered under w."""
    return np.asarray([[(w * u * v).sum(), (w * u * u).sum(), (w * v * v).sum()] for w, u, v in pairs]).reshape(-1, 3)


def fisher_correlation(terms: np.ndarray) -> float:
    """Pooled Fisher correlation Σ_s Σ_a w u v / sqrt(Σ w u² · Σ w v²) from :func:`fisher_terms`."""
    dot, uu, vv = terms.sum(axis=0)
    return float(dot / np.sqrt(uu * vv)) if uu > 0 and vv > 0 else float("nan")


def clustered_interval(terms: np.ndarray, prompts: list[str], draws: int, rng) -> list[float]:
    """95% interval of the pooled correlation under a prompt-level bootstrap."""
    labels, index = np.unique(prompts, return_inverse=True)
    per_prompt = np.zeros((len(labels), 3))
    np.add.at(per_prompt, index, terms)
    sums = per_prompt[rng.integers(0, len(labels), size=(draws, len(labels)))].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = sums[:, 0] / np.sqrt(sums[:, 1] * sums[:, 2])
    return [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]


def fit_and_hold_out(entries, keys, directions, names, metric) -> dict:
    """Fisher least squares of the advantage on the directions (fit prompts), scored on held-out prompts."""
    fit = [k for k in keys if prompt_unit(entries[k]["state"]["prompt_id"], "probe-fit") < 0.5]
    held = [k for k in keys if k not in set(fit)]
    if len(fit) < 10 or len(held) < 10:
        return {}
    design = sum(np.einsum("a,aj,ak->jk", entries[k]["weights"], directions[k], directions[k]) for k in fit)
    target = sum(np.einsum("a,aj,a->j", entries[k]["weights"], directions[k], entries[k][f"{metric}:full"]) for k in fit)
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    pairs = [(entries[k]["weights"], directions[k] @ coefficients, entries[k][f"{metric}:full"]) for k in held]
    return {"coefficients": dict(zip(names, coefficients.tolist())), "held_out_r": fisher_correlation(fisher_terms(pairs)), "held_out_states": len(held)}


def evaluate(entries: dict, directions: dict, names: list[str], *, bootstrap: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    report = {"states": len(entries), "by_type": {}}
    groups = {"all": list(entries)} | {
        name: [key for key, entry in entries.items() if entry["state"]["state_type"] == name] for name in STATE_TYPES
    }
    for group, keys in groups.items():
        keys = [key for key in keys if key in directions]
        if len(keys) < 10:
            continue
        section = {
            "states": len(keys),
            "uncensored_fraction": float(np.mean([entries[k]["uncensored"] for k in keys])),
            "min_candidate_finish_rate": float(np.min([min(entries[k]["finish_rate"]) for k in keys])),
            "noise_ceiling": {},
            "correlation": {},
            "held_out_fit": {},
        }
        for metric in METRICS:
            usable = [k for k in keys if f"{metric}:full" in entries[k]]
            if len(usable) < 10:
                continue
            prompts = [entries[k]["state"]["prompt_id"] for k in usable]
            split = fisher_correlation(
                fisher_terms([(entries[k]["weights"], entries[k][f"{metric}:half_a"], entries[k][f"{metric}:half_b"]) for k in usable])
            )
            section["noise_ceiling"][metric] = {
                "states": len(usable),
                "split_half": split,
                "spearman_brown": 2 * split / (1 + split) if split > -1 else float("nan"),
            }
            for j, name in enumerate(names):
                terms = fisher_terms([(entries[k]["weights"], directions[k][:, j], entries[k][f"{metric}:full"]) for k in usable])
                section["correlation"].setdefault(name, {})[metric] = {
                    "r": fisher_correlation(terms),
                    "ci95_prompt_bootstrap": clustered_interval(terms, prompts, bootstrap, rng),
                }
            section["held_out_fit"][metric] = fit_and_hold_out(entries, usable, directions, names, metric)
        report["by_type"][group] = section
    return report


def probed_directions(base: Path, donors: dict, spec: Path, states: list[dict], untrained=None) -> tuple[dict, list[str]]:
    """Direction values on each state's probed candidates, centered under the probe weights."""
    directions, _ = load_spec(spec)
    wanted: dict[str, list[dict]] = {}
    for state in states:
        wanted.setdefault(state["sample_id"], []).append(state)
    out = {}
    for row in iter_joined_rows(base, donors, untrained=untrained):
        if row.sample_id not in wanted:
            continue
        probs = bucket_probs(row.behavior_log_probs)
        vectors = direction_vectors(row, directions, probs)
        for state in wanted[row.sample_id]:
            weights = np.asarray(state["behavior_probs"]) / sum(state["behavior_probs"])
            values = vectors[state["position"], state["candidate_slots"], :]
            out[(state["sample_id"], state["position"])] = values - weights @ values
    return out, [direction.name for direction in directions]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    select = sub.add_parser("select")
    select.add_argument("--base", required=True, type=Path)
    select.add_argument("--tokenizer-json", required=True, type=Path)
    select.add_argument("--output", required=True, type=Path)
    select.add_argument("--fold", type=int, default=0)
    select.add_argument("--candidates", type=int, default=4)
    select.add_argument("--min-candidate-mass", type=float, default=0.02)
    select.add_argument("--min-probed-mass", type=float, default=0.7)
    select.add_argument("--min-entropy", type=float, default=0.3)
    select.add_argument("--quotas", type=json.loads, default=DEFAULT_QUOTAS)
    select.add_argument("--seed", type=int, default=0)
    run = sub.add_parser("rollout")
    run.add_argument("--states", required=True, type=Path)
    run.add_argument("--prompts", required=True, type=Path)
    run.add_argument("--canonical", required=True, type=Path)
    run.add_argument("--model", required=True)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--samples", type=int, default=8)
    run.add_argument("--max-response-tokens", type=int, default=2_048)
    run.add_argument("--max-model-len", type=int, default=10_240 + 8)
    run.add_argument("--rank", type=int, default=0)
    run.add_argument("--world-size", type=int, default=1)
    run.add_argument("--seed", type=int, default=0)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--states", required=True, type=Path)
    evaluation.add_argument("--records", required=True, type=Path, nargs="+")
    evaluation.add_argument("--base", required=True, type=Path)
    evaluation.add_argument("--donor", action="append", default=[])
    evaluation.add_argument("--spec", required=True, type=Path)
    evaluation.add_argument("--untrained", type=Path)
    evaluation.add_argument("--samples", type=int, default=8)
    evaluation.add_argument("--output", required=True, type=Path)
    evaluation.add_argument("--bootstrap", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "select":
        labeler = StateLabeler(TokenTable.from_tokenizer_json(args.tokenizer_json))
        states = select_states(
            args.base,
            labeler,
            fold=args.fold,
            quotas=args.quotas,
            candidates=args.candidates,
            min_candidate_mass=args.min_candidate_mass,
            min_probed_mass=args.min_probed_mass,
            min_entropy=args.min_entropy,
            seed=args.seed,
        )
        write_parquet(pa.Table.from_pylist(states), args.output, compression="zstd")
        counts = {name: sum(state["state_type"] == name for state in states) for name in args.quotas}
        print(json.dumps({"states": len(states), "by_type": counts}, indent=2))
    elif args.stage == "rollout":
        states = pq.read_table(args.states).to_pylist()[args.rank :: args.world_size]
        records = rollout(states, reference_targets(args.prompts, args.canonical), args)
        write_parquet(pa.Table.from_pylist(records), args.output, compression="zstd")
        print(f"WROTE {len(records)} continuations for {len(states)} states to {args.output}")
    else:
        states = pq.read_table(args.states).to_pylist()
        records = [record for path in args.records for record in pq.read_table(path).to_pylist()]
        donors = dict(item.split("=", 1) for item in args.donor)
        entries = advantages(states, records, args.samples)
        directions, names = probed_directions(args.base, donors, args.spec, states, load_untrained(args.untrained))
        report = evaluate(entries, directions, names, bootstrap=args.bootstrap, seed=0)
        report["finished_fraction"] = float(np.mean([record["finished"] for record in records]))
        report["malformed_fraction"] = float(np.mean([record["malformed"] for record in records]))
        report["probed_mass"] = {
            name: float(np.mean([s["probed_mass"] for s in states if s["state_type"] == name]))
            for name in {s["state_type"] for s in states}
        }
        write_json(report, args.output)
        print(json.dumps({group: section["states"] for group, section in report["by_type"].items()}, indent=2))


if __name__ == "__main__":
    main()
