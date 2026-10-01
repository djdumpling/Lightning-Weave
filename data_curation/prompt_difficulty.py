#!/usr/bin/env python3
"""Per-prompt difficulty from a sealed cache: how often the behavior policy's responses are verifiably correct.

Each cached prompt has several responses sampled from the behavior policy. :func:`verify` scores one finished
response against the LoopTool reference:

- tool-call references are verifiable: the response's calls must equal the reference's (names and arguments,
  as an unordered multiset);
- text references are not: whether a reply asks for the right missing parameter cannot be checked
  automatically, so those responses are ``unverifiable`` rather than correct;
- a response cut at the cache's length cap is ``unfinished``, which is not the same as a verified failure.

A prompt is ``verified_solved`` only when its reference is a tool call and every response finished and
matched. That is performance *with* the policy's full reasoning; it does not measure what shorter reasoning
would do (four successes happen about 41% of the time at a true success rate of 0.8). So the label is a
hypothesis about where shortening is safe, to be tested against a direct intervention
(``reasoning_value_probe.py``), not a measurement of it.

Weights written for ``prompt_weights`` gates: ``verified_solved`` (0/1) and ``call_pass_rate`` (the share of
matching responses for tool-call references, unfinished counted as misses; 0 for text references). The report
splits coverage by conversation kind and target kind, and records provenance (hashes of every input and of
the scorer code) so stale files can be detected.

    python data_curation/prompt_difficulty.py --base CACHE --canonical canonical.jsonl --prompts prompts.parquet \\
        --tokenizer-json tokenizer.json --output-dir DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import file_sha256
from data_curation.shift_geometry import read_shard_fields, shard_names

OUTCOMES = ("match", "miss", "unverifiable", "unfinished")
RULES = ("verified_solved", "call_pass_rate")
SCORER_SOURCES = ("data_curation/prompt_difficulty.py", "data_curation/looptool.py")
REPO = Path(__file__).resolve().parents[1]


def verify(text: str, target: dict) -> str:
    """``match`` / ``miss`` for a finished response to a tool-call reference; ``unverifiable`` for a text one."""
    from data_curation.looptool import Reject, canonical_json, parse_assistant

    if not target["tool_calls"]:
        return "unverifiable"
    try:
        _, calls = parse_assistant(text, "verify", [])
    except Reject:
        return "miss"
    produced = sorted(canonical_json({"name": call["name"], "arguments": call["arguments"]}) for call in calls)
    expected = sorted(canonical_json(call["function"]) for call in target["tool_calls"])
    return "match" if produced == expected else "miss"


def references(prompts: Path, canonical: Path) -> dict[str, dict]:
    """prompt_id -> {target, conversation_kind, target_kind} for every prompt of ``prompts``."""
    wanted = {
        row["source_index"]: row["prompt_id"]
        for row in pq.read_table(prompts, columns=["prompt_id", "source_index"]).to_pylist()
    }
    found = {}
    with canonical.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prompt_id = wanted.get(row["metadata"]["source_index"])
            if prompt_id is not None:
                found[prompt_id] = {
                    "target": row["target"],
                    "conversation_kind": row["metadata"]["conversation_kind"],
                    "target_kind": row["metadata"]["target_kind"],
                }
    missing = set(wanted.values()) - set(found)
    if missing:
        raise KeyError(f"{len(missing)} prompts have no canonical reference")
    return found


def cached_responses(base: Path) -> Iterable[tuple[str, list[int]]]:
    """(prompt_id, response tokens) of every cached row."""
    for shard in shard_names(base):
        fields = read_shard_fields(base / shard, ("prompt_id", "response_tokens"))
        yield from zip(fields["prompt_id"], ([int(t) for t in tokens] for tokens in fields["response_tokens"]))


def prompt_statistics(responses: Iterable[tuple[str, list[int]]], outcome: Callable[[str, list[int]], str], kinds):
    """{prompt_id: counts per outcome, tokens, and the prompt's kinds}; ``outcome`` returns one of OUTCOMES."""
    stats: dict[str, dict] = {}
    for prompt_id, tokens in responses:
        result = outcome(prompt_id, tokens)
        if result not in OUTCOMES:
            raise ValueError(f"unknown outcome {result!r}")
        item = stats.setdefault(
            prompt_id, {"responses": 0, "tokens": 0, **dict.fromkeys(OUTCOMES, 0), **kinds[prompt_id]}
        )
        item["responses"] += 1
        item["tokens"] += len(tokens)
        item[result] += 1
    for item in stats.values():
        callable_target = item["target_kind"] != "text_no_call"
        item["call_pass_rate"] = item["match"] / item["responses"] if callable_target else None
        item["verified_solved"] = callable_target and item["match"] == item["responses"]
        item["mean_tokens"] = item.pop("tokens") / item["responses"]
    return stats


def rule_weights(stats: dict[str, dict]) -> dict[str, dict[str, float]]:
    return {
        "verified_solved": {p: float(item["verified_solved"]) for p, item in stats.items()},
        "call_pass_rate": {p: float(item["call_pass_rate"] or 0.0) for p, item in stats.items()},
    }


def summary(stats: dict[str, dict]) -> dict:
    """Coverage overall and per (conversation kind, target kind)."""

    def describe(items: list[dict]) -> dict:
        responses = sum(i["responses"] for i in items)
        tokens = sum(i["mean_tokens"] * i["responses"] for i in items)
        calls = [i for i in items if i["call_pass_rate"] is not None]
        solved = [i for i in items if i["verified_solved"]]
        return {
            "prompts": len(items),
            "responses": responses,
            **{f"{name}_share": sum(i[name] for i in items) / max(responses, 1) for name in OUTCOMES},
            "mean_call_pass_rate": float(np.mean([i["call_pass_rate"] for i in calls])) if calls else None,
            "verified_solved_prompts": len(solved),
            "verified_solved_prompt_share": len(solved) / max(len(items), 1),
            "verified_solved_token_share": sum(i["mean_tokens"] * i["responses"] for i in solved) / max(tokens, 1e-9),
            "mean_response_tokens": tokens / max(responses, 1),
        }

    groups: dict[str, list[dict]] = {}
    for item in stats.values():
        groups.setdefault(f"{item['conversation_kind']}:{item['target_kind']}", []).append(item)
    return {"all": describe(list(stats.values())), **{key: describe(items) for key, items in sorted(groups.items())}}


def provenance(base: Path, canonical: Path, prompts: Path, tokenizer_json: Path) -> dict:
    """Hashes of every input and of the scorer's code; a changed hash means the files are stale."""
    return {
        "cache_manifest": file_sha256(base / "manifest.json"),
        "canonical": file_sha256(canonical),
        "prompts": file_sha256(prompts),
        "tokenizer": file_sha256(tokenizer_json),
        "code": {source: file_sha256(REPO / source) for source in SCORER_SOURCES},
    }


def looptool_outcome(refs: dict[str, dict], tokenizer_json: Path) -> Callable[[str, list[int]], str]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    im_end = tokenizer.token_to_id("<|im_end|>")
    think_close = tokenizer.token_to_id("</think>")

    def outcome(prompt_id: str, tokens: list[int]) -> str:
        if im_end not in tokens or think_close not in tokens:
            return "unfinished"  # cut at the cache's length cap
        text = tokenizer.decode(tokens[: tokens.index(im_end)], skip_special_tokens=False)
        return verify(text, refs[prompt_id]["target"])

    return outcome


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path, help="sealed cache (anchor/final)")
    parser.add_argument("--canonical", required=True, type=Path, help="canonical LoopTool rows with targets")
    parser.add_argument("--prompts", required=True, type=Path, help="prompts.parquet (prompt_id, source_index)")
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refs = references(args.prompts, args.canonical)
    kinds = {
        p: {"conversation_kind": r["conversation_kind"], "target_kind": r["target_kind"]} for p, r in refs.items()
    }
    stats = prompt_statistics(cached_responses(args.base), looptool_outcome(refs, args.tokenizer_json), kinds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for rule, values in rule_weights(stats).items():
        (args.output_dir / f"{rule}.json").write_text(json.dumps(values, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "provenance": provenance(args.base, args.canonical, args.prompts, args.tokenizer_json),
        "summary": summary(stats),
        "prompts": stats,
    }
    # Written last: its provenance marks the weight files as complete and current.
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output_dir / "difficulty.json").write_text(text, encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
