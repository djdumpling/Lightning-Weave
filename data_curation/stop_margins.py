#!/usr/bin/env python3
"""Where could a selective "stop thinking" target act, and what would it change? Stop states in a sealed cache.

Every gate is restricted to loss-mask thinking positions, so a gate means the same thing everywhere:

- ``<scope>:close_top``: ``</think>`` is already the top cached candidate.
- ``<scope>:near@tau``: ``</think>`` is a candidate, not the top one, within ``tau`` nats of the top one
  (ThinkBrake's margin, arXiv 2510.00546).
- ``<scope>:far@tau``: ``</think>`` is a candidate more than ``tau`` below the top one.
- ``<scope>:absent_certified@tau``: ``</think>`` is not cached and the K-th candidate is already more than
  ``tau`` below the top one, so the true margin also exceeds ``tau``.
- ``<scope>:absent_unresolved@tau``: ``</think>`` is not cached but the K-th candidate is within ``tau``, so
  the true margin is unknown without rescoring.
- ``stop_fork``: the newline choice before ``</think>`` (a single newline, which can close the thought, vs a
  paragraph break, which continues it), as labeled by :class:`StateLabeler`; a distinct intervention.

``<scope>`` is ``all`` (every thinking position) or ``boundary`` (the previous token ends a sentence or a line).

For each gate the report gives positions and the efficiency direction's Fisher energy share (explanatory).
When the composition is given (``--coef``), it also gives the stop probability under the behavior policy b,
the accuracy-only target q_acc ∝ b·exp(r_acc/α), and the composed target q_new ∝ b·exp((r_acc + coef·e)/α),
where r_acc is the raw (legacy) accuracy shift and e the efficiency direction's evidence shift. It also
gives the added KL(q_new ‖ q_acc) at the gate, as a mean and a maximum. The stop probability is b(</think>)
for ``</think>`` gates and the single-newline mass for ``stop_fork``. The overall mean added KL should
reproduce the composition's calibrated budget.

``near`` gates also report the removable reasoning suffix: on the recorded trajectory, the thinking tokens
after a closed thought's first eligible position. This is not a net saving, because stopping there changes
the answer, later tool calls, and observations. Thoughts that never close are censored and reported
separately (count and observed thinking length). Energy is further split by state type and by
reflection-fork order within a response. A cached response follows one new observation, so "first" is the
first reflection after it; the order says nothing by itself about whether a reflection is necessary.

    python data_curation/stop_margins.py --base CACHE --donor decs=DIR --direction decs --coef 3.11 \\
        --tokenizer-json tokenizer.json --untrained untrained.json --prompt-metadata meta.json --output out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.shift_geometry import (
    CachedRow,
    Direction,
    bucket_probs,
    bucket_shift,
    direction_vectors,
    evidence_shift,
    iter_joined_rows,
    load_untrained,
    log_tilted_target,
    target_kl,
)
from data_curation.shift_states import STATE_CODE, STATE_TYPES, StateLabeler, TokenTable

DEFAULT_TAUS = (0.25, 0.5, 1.0, 2.0)
SCOPES = ("all", "boundary")
THINKING_STATES = ("think_body", "think_stop_fork", "think_reflection_fork", "think_close")
THINKING_CODES = np.asarray([STATE_CODE[name] for name in THINKING_STATES])
ORDINALS = ("first", "second", "later")
GATE_SUMS = ("positions", "energy", "stop_b", "stop_acc", "stop_new", "kl", "close_shift")


def gate_names(taus) -> list[str]:
    names = []
    for scope in SCOPES:
        names.append(f"{scope}:close_top")
        for tau in taus:
            names += [f"{scope}:{kind}@{tau}" for kind in ("near", "far", "absent_certified", "absent_unresolved")]
    return [*names, "stop_fork"]


def sentence_boundaries(tokens, table: TokenTable) -> np.ndarray:
    """Positions whose previous generated token ends a sentence or a line."""
    texts = [table.text(int(token)) for token in tokens]
    boundary = np.zeros(len(texts), dtype=bool)
    for position in range(1, len(texts)):
        previous = texts[position - 1]
        boundary[position] = previous.endswith("\n") or previous.rstrip().endswith((".", "!", "?"))
    return boundary


def row_statistics(
    row: CachedRow,
    labeler: StateLabeler,
    source: str,
    taus,
    *,
    accuracy_source: str | None = None,
    coef: float | None = None,
    alpha: float = 2.0,
) -> dict:
    log_probs = np.asarray(row.behavior_log_probs, dtype=np.float64)
    probs = bucket_probs(log_probs)
    behavior = np.exp(log_probs)
    tokens = np.asarray(row.response_tokens)
    candidates = np.asarray(row.candidate_ids)
    codes = labeler.label(tokens, candidates, behavior)
    mask = np.asarray(row.loss_mask, dtype=bool)
    in_think = np.isin(codes, THINKING_CODES)
    thinking = in_think & mask
    scopes = {"all": thinking, "boundary": thinking & sentence_boundaries(tokens, labeler.table)}

    vector = direction_vectors(row, [Direction("efficiency", ((source, 1.0),))], probs)[..., 0]
    energy = (probs * vector**2).sum(-1)
    close = candidates == labeler.table.think_close  # [T, K]
    in_support = close.any(-1)
    close_top = close[np.arange(len(close)), log_probs.argmax(-1)]
    margin = log_probs.max(-1) - np.where(close, log_probs, -np.inf).max(-1)  # inf where not cached
    tail_margin = log_probs.max(-1) - log_probs.min(-1)
    close_shift = np.where(close, vector[:, :-1], 0.0).sum(-1)
    newline = labeler.member("single_newline", candidates)

    stops = {"close": {"b": (behavior * close).sum(-1)}, "newline": {"b": (behavior * newline).sum(-1)}}
    added_kl = np.zeros(len(tokens))
    if coef is not None:
        accuracy = bucket_shift(row.shifts[accuracy_source])  # raw: the legacy accuracy target
        composed = accuracy + float(coef) * evidence_shift(row, source)
        q_acc = np.exp(log_tilted_target(probs, accuracy, alpha))[:, :-1]
        q_new = np.exp(log_tilted_target(probs, composed, alpha))[:, :-1]
        for name, members in (("close", close), ("newline", newline)):
            stops[name]["acc"] = (q_acc * members).sum(-1)
            stops[name]["new"] = (q_new * members).sum(-1)
        added_kl = target_kl(probs, composed, accuracy, alpha)

    gates = {}
    for scope, base in scopes.items():
        gates[f"{scope}:close_top"] = base & close_top
        candidate = base & in_support & ~close_top
        absent = base & ~in_support
        for tau in taus:
            gates[f"{scope}:near@{tau}"] = candidate & (margin <= tau)
            gates[f"{scope}:far@{tau}"] = candidate & (margin > tau)
            gates[f"{scope}:absent_certified@{tau}"] = absent & (tail_margin > tau)
            gates[f"{scope}:absent_unresolved@{tau}"] = absent & (tail_margin <= tau)
    gates["stop_fork"] = thinking & (codes == STATE_CODE["think_stop_fork"])

    ends = np.nonzero(tokens == labeler.table.think_close)[0]
    think_end = int(ends[0]) if ends.size else None
    gate_stats, suffix = {}, {}
    for name, selected in gates.items():
        stop = stops["newline" if name == "stop_fork" else "close"]
        gate_stats[name] = {
            "positions": int(selected.sum()),
            "energy": float(energy[selected].sum()),
            "stop_b": float(stop["b"][selected].sum()),
            "stop_acc": float(stop["acc"][selected].sum()) if coef is not None else 0.0,
            "stop_new": float(stop["new"][selected].sum()) if coef is not None else 0.0,
            "kl": float(added_kl[selected].sum()),
            "kl_max": float(added_kl[selected].max()) if selected.any() else 0.0,
            "close_shift": float(close_shift[selected].sum()),
        }
        if ":near@" in name and think_end is not None:
            first = np.nonzero(selected[: think_end + 1])[0]
            suffix[name] = think_end - int(first[0]) if first.size else 0

    reflection = {ordinal: [0, 0.0] for ordinal in ORDINALS}
    for order, position in enumerate(np.nonzero(in_think & (codes == STATE_CODE["think_reflection_fork"]))[0]):
        if mask[position]:
            slot = reflection[ORDINALS[min(order, len(ORDINALS) - 1)]]
            slot[0] += 1
            slot[1] += float(energy[position])

    return {
        "positions": int(mask.sum()),
        "thinking_positions": int(thinking.sum()),
        "energy": float(energy[mask].sum()),
        "kl": float(added_kl[mask].sum()),
        "state_positions": {name: int((mask & (codes == STATE_CODE[name])).sum()) for name in STATE_TYPES},
        "state_energy": {name: float(energy[mask & (codes == STATE_CODE[name])].sum()) for name in STATE_TYPES},
        "reflection": reflection,
        "thinking_tokens": think_end,  # None: the thought never closed (censored)
        "unclosed_thinking": 0 if think_end is not None else int(in_think.sum()),
        "gates": gate_stats,
        "suffix": suffix,
    }


class Totals:
    """Sums of :func:`row_statistics` per conversation kind, and over all rows as ``all``."""

    def __init__(self, taus, *, has_target: bool):
        self.taus = tuple(taus)
        self.names = gate_names(self.taus)
        self.has_target = has_target
        self.kinds: dict[str, dict] = {}

    def _empty(self) -> dict:
        return {
            "rows": 0,
            "positions": 0,
            "thinking_positions": 0,
            "energy": 0.0,
            "kl": 0.0,
            "state_positions": dict.fromkeys(STATE_TYPES, 0),
            "state_energy": dict.fromkeys(STATE_TYPES, 0.0),
            "reflection": {ordinal: [0, 0.0] for ordinal in ORDINALS},
            "closed_rows": 0,
            "thinking_tokens": 0,
            "unclosed_rows": 0,
            "unclosed_thinking": 0,
            "gates": {
                name: {**dict.fromkeys(GATE_SUMS, 0.0), "kl_max": 0.0, "rows_with_any": 0} for name in self.names
            },
            "suffix": {name: 0 for name in self.names if ":near@" in name},
        }

    def add(self, kind: str, stats: dict) -> None:
        for name in (kind, "all"):
            total = self.kinds.setdefault(name, self._empty())
            total["rows"] += 1
            for key in ("positions", "thinking_positions", "energy", "kl"):
                total[key] += stats[key]
            for state in STATE_TYPES:
                total["state_positions"][state] += stats["state_positions"][state]
                total["state_energy"][state] += stats["state_energy"][state]
            for ordinal in ORDINALS:
                total["reflection"][ordinal][0] += stats["reflection"][ordinal][0]
                total["reflection"][ordinal][1] += stats["reflection"][ordinal][1]
            if stats["thinking_tokens"] is None:
                total["unclosed_rows"] += 1
                total["unclosed_thinking"] += stats["unclosed_thinking"]
            else:
                total["closed_rows"] += 1
                total["thinking_tokens"] += stats["thinking_tokens"]
            for gate, values in stats["gates"].items():
                sums = total["gates"][gate]
                for key in GATE_SUMS:
                    sums[key] += values[key]
                sums["kl_max"] = max(sums["kl_max"], values["kl_max"])
                sums["rows_with_any"] += values["positions"] > 0
            for gate, tokens in stats["suffix"].items():
                total["suffix"][gate] += tokens

    def summary(self) -> dict:
        def ratio(numerator, denominator):
            return numerator / denominator if denominator else None

        out = {}
        for kind, total in sorted(self.kinds.items()):
            gates = {}
            for name, sums in total["gates"].items():
                count = sums["positions"]
                item = {
                    "positions": int(count),
                    "share_of_thinking_positions": ratio(count, total["thinking_positions"]),
                    "rows_with_any_share": ratio(sums["rows_with_any"], total["rows"]),
                    "energy_share": ratio(sums["energy"], total["energy"]),
                    "mean_stop_behavior": ratio(sums["stop_b"], count),
                    "mean_efficiency_shift_on_close": (
                        None if name == "stop_fork" else ratio(sums["close_shift"], count)
                    ),
                }
                if self.has_target:
                    item.update(
                        {
                            "mean_stop_accuracy_target": ratio(sums["stop_acc"], count),
                            "mean_stop_composed_target": ratio(sums["stop_new"], count),
                            "mean_added_kl": ratio(sums["kl"], count),
                            "max_added_kl": sums["kl_max"],
                            "share_of_added_kl": ratio(sums["kl"], total["kl"]),
                        }
                    )
                if name in total["suffix"]:
                    share = ratio(total["suffix"][name], total["thinking_tokens"])
                    item["removable_suffix_share_of_closed_thinking"] = share
                gates[name] = item
            out[kind] = {
                "rows": total["rows"],
                "thinking_share_of_positions": ratio(total["thinking_positions"], total["positions"]),
                "mean_added_kl": ratio(total["kl"], total["positions"]) if self.has_target else None,
                "closed_rows": total["closed_rows"],
                "unclosed_rows": total["unclosed_rows"],
                "mean_unclosed_thinking_tokens": ratio(total["unclosed_thinking"], total["unclosed_rows"]),
                "states": {
                    state: {
                        "position_share": ratio(total["state_positions"][state], total["positions"]),
                        "energy_share": ratio(total["state_energy"][state], total["energy"]),
                    }
                    for state in STATE_TYPES
                },
                "reflection_forks": {
                    ordinal: {"positions": count, "energy_share": ratio(energy, total["energy"])}
                    for ordinal, (count, energy) in total["reflection"].items()
                },
                "gates": gates,
            }
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path, help="sealed agent-acc cache (anchor/final)")
    parser.add_argument("--base-name", default="agent_acc", help="the base cache's source, also the accuracy shift")
    parser.add_argument("--donor", action="append", default=[], help="NAME=DIR of a post+pre scored donor chain")
    parser.add_argument("--direction", default="decs", help="the efficiency source")
    parser.add_argument("--coef", type=float, help="its coefficient in the composed target (enables target stats)")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--untrained", type=Path, help="JSON {source: [token ids]}")
    parser.add_argument("--prompt-metadata", type=Path, help="JSON {prompt_id: {conversation: kind, ...}}")
    parser.add_argument("--taus", default=",".join(map(str, DEFAULT_TAUS)))
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    donors = {name: Path(path) for name, path in (item.split("=", 1) for item in args.donor)}
    if args.direction not in {args.base_name, *donors}:
        raise ValueError(f"--direction {args.direction!r} is neither --base-name nor a --donor")
    taus = tuple(float(value) for value in args.taus.split(","))
    labeler = StateLabeler(TokenTable.from_tokenizer_json(args.tokenizer_json))
    metadata = json.loads(args.prompt_metadata.read_text()) if args.prompt_metadata else {}
    totals = Totals(taus, has_target=args.coef is not None)
    rows = iter_joined_rows(
        args.base, donors, base_name=args.base_name, untrained=load_untrained(args.untrained), max_rows=args.max_rows
    )
    for row in rows:
        kind = metadata.get(row.prompt_id, {}).get("conversation", "unknown")
        stats = row_statistics(
            row, labeler, args.direction, taus, accuracy_source=args.base_name, coef=args.coef, alpha=args.alpha
        )
        totals.add(kind, stats)
    report = {
        "direction": args.direction,
        "coef": args.coef,
        "alpha": args.alpha,
        "taus": list(taus),
        "kinds": totals.summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
