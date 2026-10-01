#!/usr/bin/env python3
"""Where each cached prompt's target message sits in its conversation, and prompt weights that gate a term by it.

Each LoopTool prompt is a conversation history whose last message the target answers. From the canonical
message roles:

- ``first_turn``: the target answers the conversation's first user message (no earlier assistant message);
- ``turn_start``: the target answers a new user message after earlier assistant turns. The model must re-derive
  the task state from the visible history here, since the Qwen3 chat template drops reasoning from earlier turns;
- ``after_tool``: the target answers tool results within a turn.

For each rule in ``RULES`` a weight file ``{prompt_id: weight}`` is written for ``prompt_weights`` gates:

- ``protect_turn_starts`` is 0 at turn starts and 1 elsewhere, so a term gated by it is removed from the whole
  response (reasoning, calls, and text) on every message that answers a new user message mid-conversation;
- ``random_multi_turn_control`` is 0 on as many multi-turn prompts chosen at random (a fixed hash of the prompt id):
  the same allocation to multi-turn prompts, without choosing turn starts.

``positions.json`` (written last) holds every prompt's position, the counts per position and conversation kind, how
many prompts each rule gates (and their overlap), and provenance hashes of the inputs and this code.

    python data_curation/turn_positions.py --prompts prompts.parquet --canonical canonical.jsonl --output-dir DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.common import file_sha256
from data_curation.looptool import cached_prompt_rows

POSITIONS = ("first_turn", "turn_start", "after_tool")
RULES = ("protect_turn_starts", "random_multi_turn_control")
CONTROL_SALT = "random-multi-turn-control"
REPO = Path(__file__).resolve().parents[1]
CODE = "data_curation/turn_positions.py"


def position(messages: list[dict]) -> str:
    """The position of the message that answers ``messages`` (system messages are ignored)."""
    roles = [message["role"] for message in messages if message["role"] != "system"]
    if not roles or roles[-1] not in {"user", "tool"}:
        raise ValueError(f"a target must answer a user or tool message, not {roles[-1:] or 'nothing'}")
    if roles[-1] == "tool":
        return "after_tool"
    return "turn_start" if "assistant" in roles[:-1] else "first_turn"


def positions(prompts: Path, canonical: Path) -> tuple[dict[str, str], dict[str, str]]:
    """({prompt_id: position}, {prompt_id: conversation_kind}) for every prompt of ``prompts``."""
    rows = cached_prompt_rows(prompts, canonical)
    return (
        {p: position(row["messages"]) for p, row in rows.items()},
        {p: row["metadata"]["conversation_kind"] for p, row in rows.items()},
    )


def control_rank(prompt_id: str) -> str:
    return hashlib.sha256(f"{CONTROL_SALT}:{prompt_id}".encode()).hexdigest()


def rule_weights(found: dict[str, str], kinds: dict[str, str]) -> dict[str, dict[str, float]]:
    """Both rules' weights: turn starts, and as many multi-turn prompts chosen at random (by a fixed hash)."""
    protect = {prompt_id: 0.0 if where == "turn_start" else 1.0 for prompt_id, where in found.items()}
    gated = sum(weight == 0.0 for weight in protect.values())
    multi_turn = sorted((prompt_id for prompt_id in found if kinds[prompt_id] == "multi_turn"), key=control_rank)
    if gated > len(multi_turn):
        raise ValueError(f"{gated} turn starts but only {len(multi_turn)} multi-turn prompts")
    chosen = set(multi_turn[:gated])
    control = {prompt_id: 0.0 if prompt_id in chosen else 1.0 for prompt_id in found}
    return {"protect_turn_starts": protect, "random_multi_turn_control": control}


def summary(found: dict[str, str], kinds: dict[str, str], weights: dict[str, dict[str, float]]) -> dict:
    counts = Counter(found.values())
    by_kind = Counter(f"{kinds[prompt_id]}:{where}" for prompt_id, where in found.items())
    gated = {rule: {p for p, weight in values.items() if weight == 0.0} for rule, values in weights.items()}
    return {
        "prompts": len(found),
        "positions": {p: counts[p] for p in POSITIONS},
        "by_kind": dict(sorted(by_kind.items())),
        "gated": {
            rule: {"prompts": len(ids), **dict(sorted(Counter(f"{kinds[p]}:{found[p]}" for p in ids).items()))}
            for rule, ids in gated.items()
        },
        "gated_by_both_rules": len(set.intersection(*gated.values())),
    }


def provenance(prompts: Path, canonical: Path) -> dict:
    """Hashes of the inputs and of this code; a changed hash means the files are stale."""
    return {
        "prompts": file_sha256(prompts),
        "canonical": file_sha256(canonical),
        "code": {source: file_sha256(REPO / source) for source in (CODE, "data_curation/looptool.py")},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", required=True, type=Path, help="prompts.parquet (prompt_id, source_index)")
    parser.add_argument("--canonical", required=True, type=Path, help="canonical LoopTool rows")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    found, kinds = positions(args.prompts, args.canonical)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = rule_weights(found, kinds)
    for rule, values in weights.items():
        (args.output_dir / f"{rule}.json").write_text(json.dumps(values, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "provenance": provenance(args.prompts, args.canonical),
        "summary": summary(found, kinds, weights),
        "positions": found,
    }
    # Written last: its provenance marks the weight files as complete and current.
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output_dir / "positions.json").write_text(text, encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
