#!/usr/bin/env python3
"""Decision drift on fixed histories: how far a trained student's decisions move from the recipient's (exploratory).

The projection keeps the recipient's decision frequencies on its training samples; whether the trained student keeps
them on BFCL's histories is measured here, the same way for every arm.

- ``histories`` (CPU): fixed multi-turn requests from one logged BFCL run (``--log-requests``), exactly as the model
  saw them, split into turn starts (the last message is the user's) and later steps (the last is a tool result).
  Chosen by a fixed hash of each request, from the recipient's run alone, before any arm is replayed.
- ``sample`` (GPU, vLLM): ``--responses`` decoded responses per history from one model, with BFCL's serving settings
  (decoder, 32,768 new tokens, the YaRN context), each reduced to its decision as the server shows it
  (``data_curation/decision_projection.py``). vLLM gives the n samples of a request the seeds s, s + 1, ..., so
  draws meant to be independent need disjoint seed blocks.
- ``compare`` (CPU): two views of a decision, byte-exact and call-level (the calls with their arguments; any text
  reply counts as "reply"). Per view, arm and history: the total variation from the recipient's draw, minus that of
  a second, independent recipient draw; and the rate at which an arm's samples match the recipient's, against the
  second draw's rate. Each is averaged over histories, with a bootstrap over histories, overall and per kind.

With a handful of samples, the empirical total variation saturates: when nearly every response is unique, two draws
of the same policy and draws of two different policies are all at distance 1, and the excess is 0 either way. Each
view therefore reports how often a model repeats a decision and how often the recipient's own distance saturates;
a small excess where both are high is no evidence that decisions were preserved.

    python evaluation/decision_drift.py histories --requests RESULTS/run/tag/requests --output histories.jsonl
    python evaluation/decision_drift.py sample --histories histories.jsonl --model PATH --output samples.jsonl --seed S
    python evaluation/decision_drift.py compare --reference A.jsonl --null B.jsonl --arm projected=P.jsonl ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.bfcl_eval.config import PROTOCOL
from data_curation.decision_projection import parse_response

MULTI_TURN_PREFIX = "multi_turn"
KINDS = ("turn_start", "after_tool")
TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def call_signature(decision: str) -> str:
    """The call-level view of a decision: its calls with parsed arguments, or "reply" for text alone."""
    item = json.loads(decision)
    if item["finish_reason"] == "length":
        return "truncated"
    visible = item["visible"] or ""
    calls = []
    for raw in TOOL_CALL.findall(visible):
        try:
            call = json.loads(raw)
            calls.append({"name": call.get("name"), "arguments": call.get("arguments")})
        except (json.JSONDecodeError, AttributeError):
            calls.append({"invalid": raw})
    return json.dumps(calls, sort_keys=True) if calls else "reply"


def history_kind(messages: list[dict]) -> str | None:
    role = messages[-1].get("role") if messages else None
    return {"user": "turn_start", "tool": "after_tool"}.get(role)


def select_histories(records: list[dict], per_kind: int) -> list[dict]:
    """Distinct multi-turn requests, at most ``per_kind`` of each kind, chosen by a fixed hash of the request."""
    distinct: dict[str, dict] = {}
    for record in records:
        if not record["lane"].startswith(MULTI_TURN_PREFIX):
            continue
        messages = record["request"]["messages"]
        kind = history_kind(messages)
        if kind is None or record["request"].get("tools") is None:
            continue
        distinct.setdefault(record["body_digest"], {
            "history_id": record["body_digest"],
            "lane": record["lane"],
            "kind": kind,
            "messages": messages,
            "tools": record["request"]["tools"],
        })
    chosen = []
    for kind in KINDS:
        pool = sorted((item for item in distinct.values() if item["kind"] == kind),
                      key=lambda item: hashlib.sha256(item["history_id"].encode()).hexdigest())
        chosen.extend(pool[:per_kind])
    return chosen


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.rename(path)


def sample_decisions(llm, sampling, histories: list[dict]) -> list[dict]:
    """Decode every history (grouped by tool list, which ``LLM.chat`` takes once per call) and parse its decisions."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for history in histories:
        groups[json.dumps(history["tools"], sort_keys=True)].append(history)
    rows = []
    for tools, members in groups.items():
        outputs = llm.chat([history["messages"] for history in members], sampling, tools=json.loads(tools),
                           use_tqdm=False)
        for history, output in zip(members, outputs, strict=True):
            parsed = [parse_response(item.text, item.finish_reason) for item in output.outputs]
            rows.append({
                "history_id": history["history_id"],
                "kind": history["kind"],
                "lane": history["lane"],
                "decisions": [item.decision for item in parsed],
                "tokens": [len(item.token_ids) for item in output.outputs],
            })
    return rows


def total_variation(first: list[str], second: list[str]) -> float:
    left, right = Counter(first), Counter(second)
    return 0.5 * sum(abs(left[key] / len(first) - right[key] / len(second)) for key in set(left) | set(right))


def match_rate(first: list[str], second: list[str]) -> float:
    """The fraction of (one sample of each) pairs that are the same decision."""
    right = Counter(second)
    return sum(right[key] for key in first) / (len(first) * len(second))


def repeats(decisions: list[str]) -> bool:
    return len(set(decisions)) < len(decisions)


def _bootstrap(values: np.ndarray, mask: np.ndarray, resamples: np.ndarray) -> list[float]:
    boot = np.where(mask, values, 0.0)[resamples].sum(axis=1) / np.maximum(mask[resamples].sum(axis=1), 1)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def drift_report(reference: list[dict], null: list[dict], arms: dict[str, list[dict]], *, draws: int = 2_000,
                 seed: int = 0) -> dict:
    by_id = {name: {row["history_id"]: row for row in rows} for name, rows in {"null": null, **arms}.items()}
    histories = [row for row in reference if all(row["history_id"] in rows for rows in by_id.values())]
    if not histories:
        raise ValueError("no history was sampled by every model")
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(histories), (draws, len(histories)))
    kinds = np.array([row["kind"] for row in histories])
    masks = (("all", np.ones(len(histories), bool)), *((kind, kinds == kind) for kind in KINDS))
    report: dict = {
        "exploratory": True,
        "histories": len(histories),
        "by_kind": dict(Counter(kinds.tolist())),
        "reference_mean_tokens": float(np.mean([np.mean(row["tokens"]) for row in histories])),
        "views": {},
    }
    for view, transform in (("exact", lambda decision: decision), ("calls", call_signature)):

        def decisions(name, history, transform=transform):
            row = history if name == "reference" else by_id[name][history["history_id"]]
            return [transform(decision) for decision in row["decisions"]]

        floor_tv = np.array([total_variation(decisions("null", h), decisions("reference", h)) for h in histories])
        floor_match = np.array([match_rate(decisions("null", h), decisions("reference", h)) for h in histories])
        entry: dict = {
            "null": {
                "total_variation": float(floor_tv.mean()),
                "match_rate": float(floor_match.mean()),
                "saturated_fraction": float((floor_tv == 1.0).mean()),  # the recipient's own draws never agree
            },
            "repeating_fraction": {
                name: float(np.mean([repeats(decisions(name, h)) for h in histories]))
                for name in ("reference", "null", *arms)
            },
            "arms": {},
        }
        for name in arms:
            distance = np.array([total_variation(decisions(name, h), decisions("reference", h)) for h in histories])
            matches = np.array([match_rate(decisions(name, h), decisions("reference", h)) for h in histories])
            entry["arms"][name] = {
                kind: {
                    "total_variation": float(distance[mask].mean()),
                    "excess_over_null": float((distance - floor_tv)[mask].mean()),
                    "excess_ci95": _bootstrap(distance - floor_tv, mask, resamples),
                    "match_rate": float(matches[mask].mean()),
                    "match_deficit": float((floor_match - matches)[mask].mean()),
                    "match_deficit_ci95": _bootstrap(floor_match - matches, mask, resamples),
                }
                for kind, mask in masks
                if mask.any()
            }
        report["views"][view] = entry
    report["mean_tokens"] = {
        name: float(np.mean([np.mean(by_id[name][h["history_id"]]["tokens"]) for h in histories])) for name in arms
    }
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    histories = commands.add_parser("histories")
    histories.add_argument("--requests", type=Path, required=True, help="a logged run's requests/ directory")
    histories.add_argument("--per-kind", type=int, default=500)
    histories.add_argument("--output", type=Path, required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--histories", type=Path, required=True)
    sample.add_argument("--model", required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--responses", type=int, default=4)
    # BFCL's serving settings (configs/bfcl_eval/config.py), so the replay decodes the evaluated policy.
    sample.add_argument("--temperature", type=float, default=PROTOCOL.temperature)
    sample.add_argument("--top-p", type=float, default=PROTOCOL.top_p)
    sample.add_argument("--top-k", type=int, default=PROTOCOL.expected_generation_defaults["top_k"])
    sample.add_argument("--max-tokens", type=int, default=PROTOCOL.tokens_to_generate)
    sample.add_argument("--max-model-len", type=int, default=PROTOCOL.max_model_len)
    sample.add_argument("--rope-scaling", type=json.loads, default=PROTOCOL.rope_scaling,
                        help="the YaRN override BFCL serves with (JSON); 'null' for none")
    sample.add_argument("--seed", type=int, required=True, help="first of the n per-sample seeds s, s+1, ...")
    sample.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    compare = commands.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True, help="the recipient's samples")
    compare.add_argument("--null", type=Path, required=True, help="the recipient again, with another seed")
    compare.add_argument("--arm", action="append", required=True, help="NAME=SAMPLES.jsonl")
    compare.add_argument("--histories", type=Path, help="the frozen histories, to record and check their hash")
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "histories":
        from configs.bfcl_eval.config import read_request_logs

        if args.output.exists():
            raise FileExistsError(f"{args.output}: histories are frozen once chosen")
        chosen = select_histories(read_request_logs(args.requests), args.per_kind)
        write_jsonl(chosen, args.output)
        frozen = {"requests": str(args.requests), "per_kind": args.per_kind, "histories": len(chosen),
                  "by_kind": dict(Counter(item["kind"] for item in chosen)),
                  "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
        args.output.with_suffix(".frozen.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
        print(f"{len(chosen)} histories ({frozen['by_kind']}) -> {args.output}")
    elif args.command == "sample":
        from vllm import LLM, SamplingParams

        overrides = {"rope_scaling": args.rope_scaling} if args.rope_scaling else None
        llm = LLM(model=args.model, max_model_len=args.max_model_len, seed=args.seed, dtype=PROTOCOL.dtype,
                  gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=True, hf_overrides=overrides)
        sampling = SamplingParams(n=args.responses, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                  max_tokens=args.max_tokens, seed=args.seed)
        rows = sample_decisions(llm, sampling, read_jsonl(args.histories))
        write_jsonl([{**row, "model": args.model, "seed": args.seed} for row in rows], args.output)
        print(f"sampled {len(rows)} histories x {args.responses} -> {args.output}")
    else:
        arms = dict(item.split("=", 1) for item in args.arm)
        report = drift_report(read_jsonl(args.reference), read_jsonl(args.null),
                              {name: read_jsonl(Path(path)) for name, path in arms.items()})
        if args.histories is not None:  # the frozen selection every model was replayed on
            frozen = json.loads(args.histories.with_suffix(".frozen.json").read_text(encoding="utf-8"))
            if hashlib.sha256(args.histories.read_bytes()).hexdigest() != frozen["sha256"]:
                raise ValueError(f"{args.histories} changed after it was frozen")
            report["histories_frozen"] = frozen
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
