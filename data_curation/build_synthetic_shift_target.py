#!/usr/bin/env python3
"""Seal a Direct-OPD cache whose shift is synthesized from several anchor pairs.

The spec is a list of terms. Each term is a direction (a linear combination of
source shifts, see :mod:`data_curation.shift_geometry`), a transform, a
coefficient, and an optional gate:

- ``raw``: the legacy log-ratio Σ w (log p_post − log p_pre) on the cached
  candidates with the remaining-vocabulary bucket at 0, including tokens an
  anchor never trained. A spec of raw terms reproduces
  ``build_direct_opd_composed_target.py`` exactly (except that this builder
  always keeps the base cache's loss mask). Raw terms refuse sources with
  unmapped candidates, whose stored scores are placeholders.
- ``evidence``: δ = 0 wherever a source has no evidence (unmapped, or a token
  either anchor never trained), so such candidates keep their odds against the
  remaining vocabulary.
- ``heuristic``: no source; fixed ``pushes`` add a shift to one candidate group
  (e.g. reflection markers) at one state type (e.g. reflection forks), a
  hand-coded control for what a donor does at those forks.

An evidence term may set ``equalize``: each of its sources is first rescaled to
unit Fisher RMS per position (behavior-weighted, over the loss mask), so a sum
of donors weights each donor's direction equally rather than by its magnitude.
``equalize: "auto"`` measures the same RMS sizes but keeps the raw weights when
the largest is within ``equalize_max_ratio`` (default 2) of the smallest, and
equalizes otherwise; the sizes, their ratio, and the rule applied are recorded.

Gates remove a term at excluded state types or the first reflection fork, or
scale it by named per-prompt weights. Removing a term at an entire state keeps
that decision unchanged; masking one candidate still lets its competitors move.

One term may set ``kl_budget``: its coefficient is scaled so the mean per-token
KL(q_with ‖ q_without) over a prompt-stratified calibration sample equals the
budget, within ``max_multiplier``. The global and active-state KL
distributions are recorded so a budget reached through a few extreme states is
visible.

The summed shift is re-leveled so the remaining-vocabulary bucket is exactly 0
(the target is invariant to per-state constants over all K+1 buckets) and
stored as the difference of two non-positive synthetic score columns, the
encoding the tilted-target loss already reads.

The manifest records the spec, the source directories, the untrained-token
lists, the prompt-weight files (path and sha256), the equalization scales, and
the calibration. Like every sealed cache here, an output directory is
never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.build_direct_opd_composed_target import replace_score_fields
from data_curation.common import (
    canonical_hash,
    file_sha256,
    read_manifest,
    staged_directory,
    write_json,
    write_parquet,
)
from data_curation.composition import trainable_tokens
from data_curation.shift_geometry import (
    CachedRow,
    Direction,
    bucket_probs,
    bucket_shift,
    direction_vectors,
    iter_joined_rows,
    load_untrained,
    prompt_unit,
    relevel,
    shard_names,
    target_kl,
)
from data_curation.shift_states import STATE_CODE, STATE_TYPES, StateLabeler, TokenTable

SCHEMA_VERSION = "offline_direct_opd_synthetic_shift_v2"
TRANSFORMS = {"raw", "evidence", "heuristic"}
KL_QUANTILES = (50, 90, 95, 99, 100)
GATE_KEYS = {"exclude_state_types", "exclude_first_reflection", "prompt_weights"}


class Term:
    def __init__(self, index: int, spec: dict):
        self.index = index
        self.spec = spec
        self.transform = spec.get("transform", "evidence")
        if self.transform not in TRANSFORMS:
            raise ValueError(f"term {index}: unknown transform {self.transform!r}")
        heuristic = self.transform == "heuristic"
        pushes = spec.get("pushes", [])
        if heuristic != bool(pushes) or heuristic == ("direction" in spec):
            raise ValueError(f"term {index}: heuristic terms take pushes and no direction; other terms the reverse")
        unknown = {item["state"] for item in pushes} - set(STATE_TYPES)
        if unknown:
            raise ValueError(f"term {index}: unknown push states {sorted(unknown)}")
        self.pushes = [(STATE_CODE[item["state"]], item["group"], float(item["shift"])) for item in pushes]
        self.direction = None if heuristic else Direction.from_spec(f"term{index}", spec["direction"])
        self.coef = float(spec.get("coef", 1.0))
        self.gate = spec.get("gate") or {}
        unknown_keys = set(self.gate) - GATE_KEYS
        if unknown_keys:
            raise ValueError(f"term {index}: unknown gate keys {sorted(unknown_keys)}")
        self.kl_budget = spec.get("kl_budget")
        self.max_multiplier = float(spec.get("max_multiplier", 10.0))
        excluded = self.gate.get("exclude_state_types")
        unknown = set(excluded or ()) - set(STATE_TYPES)
        if unknown:
            raise ValueError(f"term {index}: unknown excluded state types {sorted(unknown)}")
        self.excluded_codes = None if not excluded else np.asarray([STATE_CODE[name] for name in excluded])
        if self.transform in {"raw", "heuristic"} and self.gate:
            raise ValueError(f"term {index}: {self.transform} terms are controls and take no gates")
        self.equalize = spec.get("equalize") or False  # a null equalize means off, as it did before "auto"
        if self.equalize not in (False, True, "auto"):
            raise ValueError(f"term {index}: equalize is true, false, or 'auto', not {self.equalize!r}")
        ratio = spec.get("equalize_max_ratio", 2.0)
        number = isinstance(ratio, (int, float)) and not isinstance(ratio, bool)
        if "equalize_max_ratio" in spec and (self.equalize != "auto" or not number or not 1.0 <= ratio < np.inf):
            raise ValueError(
                f"term {index}: equalize_max_ratio (a finite number, at least 1) belongs to equalize 'auto'"
            )
        self.equalize_max_ratio = float(ratio)
        distinct = self.direction is not None and len(set(self.direction.sources)) == len(self.direction.sources) >= 2
        if self.equalize and (self.transform != "evidence" or not distinct):
            raise ValueError(f"term {index}: equalize weights two or more distinct sources of an evidence term")

    @property
    def sources(self) -> set[str]:
        return set(self.direction.sources) if self.direction is not None else set()

    @property
    def needs_labels(self) -> bool:
        return (
            self.excluded_codes is not None
            or bool(self.gate.get("exclude_first_reflection"))
            or self.transform == "heuristic"
        )


def heuristic_vector(row: CachedRow, pushes, labeler: StateLabeler, codes: np.ndarray) -> np.ndarray:
    """Σ shift · [state matches] · [candidate in group] on K+1 buckets; the remaining bucket stays 0."""
    delta = np.zeros(row.behavior_log_probs.shape, dtype=np.float64)
    for code, group, shift in pushes:
        delta += shift * ((codes == code)[:, None] & labeler.member(group, row.candidate_ids))
    return bucket_shift(delta)


def raw_vector(row: CachedRow, direction: Direction) -> np.ndarray:
    """Σ w (post − pre) on K+1 buckets in the reference composer's float64 order."""
    delta = np.zeros(row.behavior_log_probs.shape, dtype=np.float64)
    for source, coef in direction.terms:
        if not row.mapped[source][row.loss_mask].all():
            raise ValueError(f"raw term reads placeholder scores of unmapped {source} candidates ({row.sample_id})")
        delta += float(coef) * row.shifts[source]
    return bucket_shift(delta)


class SyntheticComposer:
    def __init__(
        self,
        spec: dict,
        *,
        labeler: StateLabeler | None,
        prompt_weights: dict[str, dict[str, float]] | None = None,
    ):
        self.spec = spec
        self.alpha = float(spec["alpha"])
        self.terms = [Term(index, item) for index, item in enumerate(spec["terms"])]
        if sum(term.kl_budget is not None for term in self.terms) > 1:
            raise ValueError("at most one term may set kl_budget; joint calibration is order-dependent")
        if any(term.needs_labels for term in self.terms) and labeler is None:
            raise ValueError("state-type gates and heuristics need --tokenizer-json")
        unknown = {group for term in self.terms for _, group, _ in term.pushes} - set(
            labeler.groups if labeler else ()
        )
        if unknown:
            raise ValueError(f"heuristic pushes name unknown candidate groups {sorted(unknown)}")
        needed = {term.gate["prompt_weights"] for term in self.terms if "prompt_weights" in term.gate}
        self.prompt_weights = prompt_weights or {}
        missing = needed - set(self.prompt_weights)
        if missing:
            raise ValueError(f"prompt_weights gates name weights without --prompt-weights: {sorted(missing)}")
        for name in needed:
            values = np.asarray(list(self.prompt_weights[name].values()), dtype=np.float64)
            if not (np.isfinite(values).all() and (values >= 0).all()):
                raise ValueError(f"prompt weights {name!r} must be finite and non-negative")
        self.labeler = labeler
        self.multipliers = [1.0 for _ in self.terms]
        self.equalization: dict[int, dict[str, float]] = {}
        self.equalization_rules: dict[int, dict] = {}
        self.needs_codes = any(term.needs_labels for term in self.terms)

    @property
    def sources(self) -> set[str]:
        return set().union(*(term.sources for term in self.terms))

    # Fit source scales before KL calibration.
    def needs_fit(self) -> bool:
        return any(term.equalize for term in self.terms)

    def fit(self, rows) -> None:
        components = {
            term.index: [
                Direction(f"term{term.index}_{source}", ((source, 1.0),)) for source in term.direction.sources
            ]
            for term in self.terms
            if term.equalize
        }
        energy = {index: np.zeros(len(directions)) for index, directions in components.items()}
        positions = 0
        for row in rows:
            probs = bucket_probs(row.behavior_log_probs)
            positions += int(row.loss_mask.sum())
            for index, directions in components.items():
                stacked = direction_vectors(row, directions, probs)[row.loss_mask]
                energy[index] += np.einsum("tk,tkj->j", probs[row.loss_mask], stacked**2)
        for index, total in energy.items():
            term = self.terms[index]
            rms = np.sqrt(total / max(positions, 1))
            if not (rms > 0).all():
                raise ValueError(f"term {index}: a source has no evidence to equalize")
            scales = dict(zip(term.direction.sources, (1.0 / rms).tolist()))
            if term.equalize == "auto":
                ratio = float(rms.max() / rms.min())
                rule = "raw" if ratio <= term.equalize_max_ratio else "equalized"
                if rule == "raw":
                    scales = dict.fromkeys(term.direction.sources, 1.0)
                self.equalization_rules[index] = {
                    "rms": dict(zip(term.direction.sources, rms.tolist())),
                    "ratio": ratio,
                    "max_ratio": term.equalize_max_ratio,
                    "rule": rule,
                }
            self.equalization[index] = scales
            term.direction = Direction(
                term.direction.name,
                tuple((source, coef * scales[source]) for source, coef in term.direction.terms),
                term.direction.keep_untrained,
            )

    # -- per-row term vectors ----------------------------------------------------------------------
    def _codes(self, row: CachedRow) -> np.ndarray | None:
        if not self.needs_codes:
            return None
        return self.labeler.label(row.response_tokens, row.candidate_ids, np.exp(row.behavior_log_probs))

    def term_vector(self, term: Term, row: CachedRow, probs: np.ndarray, codes) -> np.ndarray:
        if term.transform == "raw":
            return raw_vector(row, term.direction)
        if term.transform == "heuristic":
            return heuristic_vector(row, term.pushes, self.labeler, codes)
        vector = direction_vectors(row, [term.direction], probs)[..., 0]
        return vector * self.gate(term, row, probs, codes)[:, None]

    def gate(self, term: Term, row: CachedRow, probs: np.ndarray, codes) -> np.ndarray:
        gate = np.ones(len(probs), dtype=bool)
        if term.excluded_codes is not None:
            gate &= ~np.isin(codes, term.excluded_codes)
        if term.gate.get("exclude_first_reflection"):
            forks = np.nonzero(codes == STATE_CODE["think_reflection_fork"])[0]
            if forks.size:
                gate[forks[0]] = False
        weight = 1.0
        if "prompt_weights" in term.gate:
            weights = self.prompt_weights[term.gate["prompt_weights"]]
            if row.prompt_id not in weights:
                raise KeyError(f"prompt weights {term.gate['prompt_weights']!r} have no entry for {row.prompt_id}")
            weight = float(weights[row.prompt_id])
        return gate.astype(np.float64) * weight

    def row_terms(self, row: CachedRow) -> tuple[np.ndarray, list[np.ndarray]]:
        probs = bucket_probs(row.behavior_log_probs)
        codes = self._codes(row)
        return probs, [term.coef * self.term_vector(term, row, probs, codes) for term in self.terms]

    def combine(self, vectors: list[np.ndarray], multipliers=None) -> np.ndarray:
        multipliers = self.multipliers if multipliers is None else multipliers
        total = np.zeros_like(vectors[0])
        for multiplier, vector in zip(multipliers, vectors):
            total += multiplier * vector
        return relevel(total)

    # -- KL-budget calibration ---------------------------------------------------------------------
    def calibrate(self, rows) -> dict:
        budgeted = [term for term in self.terms if term.kl_budget is not None]
        if not budgeted:
            return {}
        term = budgeted[0]
        cached, codes = [], []
        for row in rows:
            cached.append((row.loss_mask, *self.row_terms(row)))
            if self.labeler is not None:  # state labels only describe where the KL lands
                labels = self.labeler.label(row.response_tokens, row.candidate_ids, np.exp(row.behavior_log_probs))
                codes.append(labels[row.loss_mask])
        if not cached:
            raise ValueError("the calibration sample is empty")

        def kl_values(multiplier: float) -> tuple[np.ndarray, np.ndarray]:
            with_term, without = list(self.multipliers), list(self.multipliers)
            with_term[term.index], without[term.index] = multiplier, 0.0
            values, active = [], []
            for mask, probs, vectors in cached:
                kl = target_kl(probs, self.combine(vectors, with_term), self.combine(vectors, without), self.alpha)
                values.append(kl[mask])
                active.append((np.abs(vectors[term.index]).max(axis=-1) > 0)[mask])
            return np.concatenate(values), np.concatenate(active)

        budget = float(term.kl_budget)
        reachable = kl_values(term.max_multiplier)[0].mean()
        if reachable < budget:
            raise ValueError(
                f"term {term.index}: KL budget {budget} is unreachable (mean KL {reachable:.5f} at the "
                f"max_multiplier {term.max_multiplier}); the term has too little evidence"
            )
        low, high = 0.0, term.max_multiplier
        for _ in range(50):
            middle = 0.5 * (low + high)
            low, high = (middle, high) if kl_values(middle)[0].mean() < budget else (low, middle)
        multiplier = 0.5 * (low + high)
        values, active = kl_values(multiplier)
        quantiles = lambda x: (
            dict(zip(map(str, KL_QUANTILES), np.percentile(x, KL_QUANTILES).tolist())) if x.size else {}
        )  # noqa: E731
        report = {
            "term": term.index,
            "budget": budget,
            "multiplier": multiplier,
            "effective_coef": multiplier * term.coef,
            "achieved_kl": float(values.mean()),
            "calibration_rows": len(cached),
            "active_fraction": float(active.mean()),
            "kl_quantiles": quantiles(values),
            "active_kl_quantiles": quantiles(values[active]),
            "kl_share_of_top_positions": kl_concentration(values),
        }
        if codes:
            report.update(kl_by_state(values, np.concatenate(codes)))
        self.multipliers[term.index] = multiplier
        return report


def kl_concentration(values: np.ndarray, fractions=(0.001, 0.01, 0.1)) -> dict[str, float]:
    """The share of the summed KL carried by the top ``fraction`` of positions."""
    total = float(values.sum())
    ordered = np.sort(values)[::-1]
    return {
        f"{100 * fraction:g}%": float(ordered[: max(1, int(round(fraction * len(values))))].sum() / total)
        if total > 0
        else 0.0
        for fraction in fractions
    }


def kl_by_state(values: np.ndarray, codes: np.ndarray) -> dict[str, dict[str, float]]:
    """Per state type: its share of positions, its share of the summed KL, and its mean per-position KL."""
    total = float(values.sum())
    shares, positions, means = {}, {}, {}
    for code, name in enumerate(STATE_TYPES):
        selected = codes == code
        if selected.any():
            positions[name] = float(selected.mean())
            shares[name] = float(values[selected].sum() / total) if total > 0 else 0.0
            means[name] = float(values[selected].mean())
    return {"position_share_by_state": positions, "kl_share_by_state": shares, "mean_kl_by_state": means}


def encode(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop the remaining-vocabulary bucket (0 after re-leveling) and split into non-positive scores."""
    candidates = delta[..., :-1]
    return np.minimum(-candidates, 0.0).astype(np.float32), np.minimum(candidates, 0.0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path, help="sealed base cache (the agent-acc anchor/final)")
    parser.add_argument("--base-name", default="agent_acc")
    parser.add_argument("--donor", action="append", default=[], help="NAME=DIR of a post+pre scored donor chain")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--untrained", type=Path, help="JSON {source: [token ids]}")
    parser.add_argument(
        "--prompt-weights", action="append", default=[], help="NAME=PATH of a JSON {prompt_id: weight} file"
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.25, help="prompt-stratified sample share")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {args.output_dir}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    donors = {name: Path(path) for name, path in (item.split("=", 1) for item in args.donor)}
    untrained = load_untrained(args.untrained)
    table = TokenTable.from_tokenizer_json(args.tokenizer_json) if args.tokenizer_json else None
    weight_files = {name: Path(path) for name, path in (item.split("=", 1) for item in args.prompt_weights)}
    composer = SyntheticComposer(
        spec,
        labeler=StateLabeler(table) if table else None,
        prompt_weights={name: json.loads(path.read_text(encoding="utf-8")) for name, path in weight_files.items()},
    )
    missing = composer.sources - {args.base_name} - set(donors)
    if missing:
        raise ValueError(f"spec uses sources without --donor: {sorted(missing)}")
    donors = {name: donors[name] for name in sorted(composer.sources - {args.base_name})}
    compose(args, composer, donors, untrained, weight_files)


def compose(args, composer, donors, untrained, weight_files=None) -> None:
    def rows(**kwargs):
        return iter_joined_rows(args.base, donors, base_name=args.base_name, untrained=untrained, **kwargs)

    if composer.needs_fit():
        composer.fit(rows())
    sample = (row for row in rows() if prompt_unit(row.prompt_id, "kl-calibration") < args.calibration_fraction)
    calibration = composer.calibrate(sample)
    model = {
        "composition_schema_version": SCHEMA_VERSION,
        "spec": composer.spec,
        "sources": {name: str(path) for name, path in donors.items()},
        "untrained": {name: sorted(ids) for name, ids in sorted((untrained or {}).items())},
        "calibration": calibration,
    }
    if composer.equalization:
        model["equalization"] = {str(index): scales for index, scales in composer.equalization.items()}
    if composer.equalization_rules:
        model["equalization_rules"] = {str(index): rule for index, rule in composer.equalization_rules.items()}
    if weight_files:
        model["prompt_weights"] = {
            name: {"path": str(path), "sha256": file_sha256(path)} for name, path in sorted(weight_files.items())
        }
    revision = canonical_hash(model)
    model.update({"model_type": "synthetic_shift_composition", "revision": revision})
    shards, total_rows, total_tokens = [], 0, 0
    kl_sum, kl_count = 0.0, 0
    output = args.output_dir
    with staged_directory(output) as temporary:
        for name in shard_names(args.base):
            table = pq.read_table(args.base / name)
            pre_rows, post_rows, masks = [], [], []
            for row in rows(shards=[name]):
                probs, vectors = composer.row_terms(row)
                delta = composer.combine(vectors)
                pre, post = encode(delta)
                pre_rows.append(pre)
                post_rows.append(post)
                masks.append(row.loss_mask.tolist())
                kl = target_kl(probs, delta, np.zeros_like(delta), composer.alpha)
                kl_sum += float(kl[row.loss_mask].sum())
                kl_count += int(row.loss_mask.sum())
            if len(pre_rows) != table.num_rows:
                raise RuntimeError(f"{name}: composed {len(pre_rows)} rows for {table.num_rows}")
            composed = replace_score_fields(
                table,
                pre_rows=pre_rows,
                post_rows=post_rows,
                loss_masks=masks,
                projection_masks=None,
                revision=revision,
            )
            write_parquet(composed, temporary / name, compression="zstd", row_group_size=128)
            shards.append({"path": name, "rows": composed.num_rows, "sha256": file_sha256(temporary / name)})
            total_rows += composed.num_rows
            total_tokens += trainable_tokens(composed)
        manifest = {
            **read_manifest(args.base / "manifest.json"),
            "post_teacher_model": model,
            "pre_teacher_model": {"model_type": "synthetic_shift_composition_reference", "revision": revision},
            "total_rows": total_rows,
            "total_trainable_tokens": total_tokens,
            "target_kl_to_behavior_per_token": kl_sum / max(kl_count, 1),
            "shards": shards,
        }
        write_json(manifest, temporary / "manifest.json")
    print(
        f"SEALED_SYNTHETIC_SHIFT {output} rows={total_rows} tokens={total_tokens} "
        f"kl_to_behavior={manifest['target_kl_to_behavior_per_token']:.4f} revision={revision}"
    )
    if calibration:
        print(json.dumps(calibration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
