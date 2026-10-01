#!/usr/bin/env python3
"""Fisher geometry of cached policy shifts on the student's own token states.

The tilted-target loss (``slime/rollout/offline_direct_opd.py``) works over
K+1 buckets: the cached Top-K candidates plus one remaining-vocabulary bucket
whose shift is fixed at 0. On that bucket space the target

    q(a|s) ∝ b(a|s) · exp(δ(a|s) / α)

is exactly invariant to adding a per-state constant to all K+1 entries, so a
shift is an equivalence class modulo state baselines. Every direction here is
therefore a K+1 vector ``[δ_1..δ_K, 0]`` centered under the behavior bucket
distribution b, and directions are compared with the Fisher inner product of
the exponential tilt,

    ⟨u, v⟩_s = Cov_b(u, v),   KL(q_u ‖ q_v) ≈ Var_b(u − v) / (2α²),

aggregated over states. Its normalized form is the "Fisher cosine": a local
correlation under the student.

**Evidence.** A source has no evidence at a candidate that it cannot score
(unmapped across tokenizers) or whose row one of its anchors never trained.
Such a candidate gets δ = 0, the same "no change" value the loss assigns to
every unscored token in the remaining-vocabulary bucket, so its odds against
that bucket are untouched. ``coverage`` is the student's behavior mass on the
candidates a source can score; ``support`` is the smaller of the two anchors'
own probability mass on them (whether the donor finds the state plausible).

"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.shift_states import FORK_MASS, STATE_CODE, STATE_TYPES, StateLabeler, TokenTable, position_bins

OTHER_TOKEN = -1
OTHER_FLOOR = 1e-8
FOLDS = 5
POSITION_SLICES = 10
BASE_FIELDS = (
    "sample_id",
    "prompt_id",
    "response_tokens",
    "candidate_ids",
    "behavior_topk_log_probs",
    "loss_mask",
    "post_teacher_log_probs",
    "pre_teacher_log_probs",
)
DONOR_FIELDS = ("sample_id", "loss_mask", "post_teacher_log_probs", "pre_teacher_log_probs")
OPTIONAL_DONOR_FIELDS = (
    "candidate_projection_valid_mask",
    "token_projection_valid_mask",
    "loss_mask_before_token_projection",
)


# ---------------------------------------------------------------------------
# Bucket-space math (all arrays are [..., K] or [..., K+1] along the last axis)
# ---------------------------------------------------------------------------


def bucket_probs(topk_log_probs: np.ndarray) -> np.ndarray:
    """K+1 behavior distribution, identical to ``_topk_plus_other_distribution``."""
    topk = np.exp(np.asarray(topk_log_probs, dtype=np.float64))
    other = np.clip(1.0 - topk.sum(axis=-1, keepdims=True), OTHER_FLOOR, None)
    buckets = np.concatenate([topk, other], axis=-1)
    return buckets / buckets.sum(axis=-1, keepdims=True)


def bucket_shift(delta: np.ndarray) -> np.ndarray:
    """A K-candidate shift on K+1 buckets; the remaining-vocabulary bucket is 0 as in the loss."""
    delta = np.asarray(delta, dtype=np.float64)
    return np.concatenate([delta, np.zeros(delta.shape[:-1] + (1,))], axis=-1)


def center(delta: np.ndarray, probs: np.ndarray) -> np.ndarray:
    return delta - (probs * delta).sum(axis=-1, keepdims=True)


def relevel(delta: np.ndarray) -> np.ndarray:
    """Shift each state's vector so the remaining-vocabulary bucket is exactly 0."""
    return delta - delta[..., -1:]


def log_tilted_target(probs: np.ndarray, delta: np.ndarray, alpha: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-30, None)) + delta / float(alpha)
    logits -= logits.max(axis=-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


def target_kl(probs: np.ndarray, delta_p: np.ndarray, delta_q: np.ndarray, alpha: float) -> np.ndarray:
    """Per-state KL(q_p ‖ q_q) between two tilted targets of the same behavior distribution."""
    log_p = log_tilted_target(probs, delta_p, alpha)
    log_q = log_tilted_target(probs, delta_q, alpha)
    return (np.exp(log_p) * (log_p - log_q)).sum(axis=-1)


def fisher_products(directions: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Per-state Fisher Gram entries Σ_a b(a|s) u_i(a) u_j(a) for centered directions [T, K+1, J]."""
    return np.einsum("tk,tki,tkj->tij", probs, directions, directions)


def entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(np.clip(probs, 1e-30, None))).sum(axis=-1)


def cosines(gram: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.clip(np.diagonal(gram, axis1=-2, axis2=-1), 1e-30, None))
    return gram / (scale[..., :, None] * scale[..., None, :])


# ---------------------------------------------------------------------------
# Cache loading (row-aligned shards joined and checked on sample_id)
# ---------------------------------------------------------------------------


def _ragged(array: pa.Array) -> list[np.ndarray]:
    """list<list<x>> or list<x> Arrow column to one numpy array per row, without Python floats."""
    array = array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array
    lengths = np.diff(array.offsets.to_numpy())
    child = array.flatten()
    if pa.types.is_list(child.type):
        widths = np.diff(child.offsets.to_numpy())
        width = int(widths[0]) if widths.size else 0
        if widths.size and not np.all(widths == width):
            raise ValueError("ragged inner lists are not supported")
        flat = child.flatten().to_numpy(zero_copy_only=False).reshape(-1, width)
    else:
        flat = child.to_numpy(zero_copy_only=False)
    return np.split(flat, np.cumsum(lengths)[:-1]) if len(lengths) else []


def read_shard_fields(path: Path, fields: tuple[str, ...]) -> dict[str, list]:
    available = {child.name for child in pq.read_schema(path).field("metadata").type}
    wanted = [name for name in fields if name in available]
    table = pq.read_table(path, columns=[f"metadata.{name}" for name in wanted])
    columns = {}
    for name in wanted:
        column = table[name].combine_chunks()
        columns[name] = (
            column.to_pylist()
            if pa.types.is_string(column.type) or pa.types.is_large_string(column.type)
            else _ragged(column)
        )
    return columns


@dataclass
class CachedRow:
    """One cached trajectory with every source's shift and evidence masks ([T, K] unless noted)."""

    sample_id: str
    prompt_id: str
    response_tokens: np.ndarray
    candidate_ids: np.ndarray
    behavior_log_probs: np.ndarray
    loss_mask: np.ndarray
    shifts: dict[str, np.ndarray] = field(default_factory=dict)
    mapped: dict[str, np.ndarray] = field(default_factory=dict)
    trained: dict[str, np.ndarray] = field(default_factory=dict)
    support: dict[str, np.ndarray] = field(default_factory=dict)  # [T]

    @property
    def fold(self) -> int:
        return prompt_fold(self.prompt_id)

    def valid(self, source: str) -> np.ndarray:
        return self.mapped[source] & self.trained[source]

    def coverage(self, source: str) -> np.ndarray:
        """[T] behavior mass of the cached candidates on which ``source`` has evidence."""
        topk = np.exp(self.behavior_log_probs)
        return (topk * self.valid(source)).sum(axis=-1) / np.clip(topk.sum(axis=-1), 1e-30, None)


def prompt_fold(prompt_id: str, folds: int = FOLDS) -> int:
    return int(hashlib.sha256(prompt_id.encode("utf-8")).hexdigest(), 16) % folds


def prompt_unit(prompt_id: str, salt: str) -> float:
    """A deterministic uniform draw per prompt, independent of the fold assignment."""
    return int(hashlib.sha256(f"{salt}\0{prompt_id}".encode("utf-8")).hexdigest()[:15], 16) / float(16**15)


def shard_names(base_dir: Path) -> list[str]:
    """Shard order of a cache: its sealed manifest if present, else sorted file names."""
    manifest = Path(base_dir) / "manifest.json"
    if manifest.exists():
        return [shard["path"] for shard in json.loads(manifest.read_text(encoding="utf-8"))["shards"]]
    return sorted(path.name for path in Path(base_dir).glob("*.parquet"))


def source_masks(columns: dict, position: int, candidates: np.ndarray, untrained: set[int]):
    """(mapped, trained) candidate masks of one scored row.

    With per-candidate validity the scorer's position-level AND is undone: one
    unmappable candidate removes only that candidate's evidence.
    """
    per_candidate = "candidate_projection_valid_mask" in columns
    positions = columns.get("loss_mask_before_token_projection") if per_candidate else None
    if positions is None:
        positions = columns["loss_mask"]
    mapped = np.repeat(np.asarray(positions[position], dtype=bool)[:, None], candidates.shape[1], 1)
    if per_candidate:
        mapped &= np.asarray(columns["candidate_projection_valid_mask"][position], dtype=bool)
    elif "token_projection_valid_mask" in columns:
        mapped &= np.asarray(columns["token_projection_valid_mask"][position], dtype=bool)[:, None]
    trained = ~np.isin(candidates, list(untrained)) if untrained else np.ones_like(mapped)
    return mapped, trained


def iter_joined_rows(
    base_dir: Path,
    donors: Mapping[str, Path],
    *,
    base_name: str = "agent_acc",
    untrained: Mapping[str, set[int]] | None = None,
    max_rows: int | None = None,
    shards: list[str] | None = None,
) -> Iterator[CachedRow]:
    """Yield base rows with every source's raw K-candidate shift and evidence masks.

    Donor directories must hold shards with the base's file names and row order
    (the scorer preserves both); ``sample_id`` equality is checked on every row.
    ``untrained`` lists, per source, the student token ids that either anchor of
    the pair never trained.
    """
    untrained = untrained or {}
    emitted = 0
    for shard_name in shards or shard_names(base_dir):
        shard = Path(base_dir) / shard_name
        base = read_shard_fields(shard, BASE_FIELDS)
        donor_columns = {}
        for name, directory in donors.items():
            path = Path(directory) / shard.name
            if not path.exists():
                raise FileNotFoundError(f"donor {name} lacks shard {shard.name}")
            donor_columns[name] = read_shard_fields(path, DONOR_FIELDS + OPTIONAL_DONOR_FIELDS)
            if donor_columns[name]["sample_id"] != base["sample_id"]:
                raise ValueError(f"donor {name} shard {shard.name} is not row-aligned with the base")
        for index, sample_id in enumerate(base["sample_id"]):
            candidates = np.asarray(base["candidate_ids"][index], dtype=np.int64)
            row = CachedRow(
                sample_id=sample_id,
                prompt_id=base["prompt_id"][index],
                response_tokens=np.asarray(base["response_tokens"][index], dtype=np.int64),
                candidate_ids=candidates,
                behavior_log_probs=np.asarray(base["behavior_topk_log_probs"][index], dtype=np.float64),
                loss_mask=np.asarray(base["loss_mask"][index], dtype=bool),
            )
            sources = {base_name: base} | donor_columns
            for name, columns in sources.items():
                post = np.asarray(columns["post_teacher_log_probs"][index], dtype=np.float64)
                pre = np.asarray(columns["pre_teacher_log_probs"][index], dtype=np.float64)
                mapped, trained = source_masks(columns, index, candidates, untrained.get(name, set()))
                valid = mapped & trained
                row.shifts[name] = post - pre
                row.mapped[name] = mapped
                row.trained[name] = trained
                # Unmapped entries hold placeholder scores, so only valid candidates count.
                # Aliases are injective (the scorer enforces it), so no teacher token is counted twice.
                row.support[name] = np.minimum((np.exp(post) * valid).sum(-1), (np.exp(pre) * valid).sum(-1))
            yield row
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                return


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Direction:
    """A named linear combination of source shifts.

    Unmapped candidates never carry evidence. ``keep_untrained`` retains the
    raw log-ratio at tokens an anchor never trained, which reproduces the legacy
    agent-accuracy target; it is only for that control.
    """

    name: str
    terms: tuple[tuple[str, float], ...]
    keep_untrained: bool = False

    @classmethod
    def from_spec(cls, name: str, spec: Mapping) -> "Direction":
        return cls(
            name=name,
            terms=tuple((term["source"], float(term["coef"])) for term in spec["terms"]),
            keep_untrained=bool(spec.get("keep_untrained", False)),
        )

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(source for source, _ in self.terms)


def evidence_shift(row: CachedRow, source: str, *, keep_untrained: bool = False) -> np.ndarray:
    """K+1 shift of one source with δ = 0 wherever it has no evidence (not centered)."""
    evidence = row.mapped[source] if keep_untrained else row.valid(source)
    return bucket_shift(np.where(evidence, row.shifts[source], 0.0))


def direction_vectors(row: CachedRow, directions: list[Direction], probs: np.ndarray) -> np.ndarray:
    """[T, K+1, J] centered direction vectors."""
    cache: dict[tuple[str, bool], np.ndarray] = {}
    out = np.zeros(probs.shape + (len(directions),))
    for j, direction in enumerate(directions):
        for source, coef in direction.terms:
            key = (source, direction.keep_untrained)
            if key not in cache:
                cache[key] = center(evidence_shift(row, source, keep_untrained=direction.keep_untrained), probs)
            out[..., j] += coef * cache[key]
    return out


# ---------------------------------------------------------------------------
# Streaming analysis with prompt-clustered uncertainty
# ---------------------------------------------------------------------------


PROBES = {
    "stop_minus_continue": ("think_stop_fork", "single_newline", "paragraph_break"),
    "reflect_minus_conclude": ("think_reflection_fork", "reflection", "conclusion"),
    "call_minus_rest": ("act_vs_talk", "tool_call", None),
    "end_minus_rest": ("end_of_message", "im_end", None),
    "end_minus_recall": ("call_boundary", "im_end", None),
}


@dataclass
class GeometryAccumulator:
    """Per-slice, per-prompt Fisher Gram matrices and behavioral-probe sums."""

    names: list[str]
    grams: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    positions: dict[str, int] = field(default_factory=dict)
    probe_sums: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    def add(self, prompt_id: str, products: np.ndarray, slices: dict[str, np.ndarray]) -> None:
        for name, selected in slices.items():
            if selected.any():
                prompts = self.grams.setdefault(name, {})
                prompts[prompt_id] = prompts.get(prompt_id, 0) + products[selected].sum(axis=0)
                self.positions[name] = self.positions.get(name, 0) + int(selected.sum())

    def add_probe(self, prompt_id: str, probe: str, contrasts: np.ndarray) -> None:
        """Accumulate (Σ contrast per direction, count) per prompt."""
        prompts = self.probe_sums.setdefault(probe, {})
        total = prompts.get(prompt_id, np.zeros(len(self.names) + 1))
        total[:-1] += contrasts.sum(axis=0)
        total[-1] += len(contrasts)
        prompts[prompt_id] = total

    def slice_gram(self, name: str) -> np.ndarray:
        return sum(self.grams[name].values())

    def summary(self, *, bootstrap: int = 1_000, placebos: int = 200, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        out = {"directions": self.names, "slices": {}, "probes": {}}
        energy_all = np.diag(self.slice_gram("all")) if "all" in self.grams else None
        for name, prompts in self.grams.items():
            stacked = np.stack(list(prompts.values()))
            gram = stacked.sum(axis=0)
            draws = rng.integers(0, len(stacked), size=(bootstrap, len(stacked)))
            boot = cosines(np.stack([stacked[d].sum(axis=0) for d in draws]))
            # Placebo: an independent sign per prompt and direction destroys cross-direction
            # alignment while keeping each direction's own clustered variance.
            signs = rng.choice([-1.0, 1.0], size=(placebos, len(stacked), len(self.names)))
            placebo = cosines(np.einsum("npi,npj,pij->nij", signs, signs, stacked))
            out["slices"][name] = {
                "positions": self.positions[name],
                "prompts": len(stacked),
                "gram": gram.tolist(),
                "cosine": cosines(gram).tolist(),
                "cosine_ci95": np.percentile(boot, [2.5, 97.5], axis=0).tolist(),
                "placebo_cosine_ci95": np.percentile(placebo, [2.5, 97.5], axis=0).tolist(),
                "energy_share": (np.diag(gram) / energy_all).tolist() if energy_all is not None else None,
                "fisher_norm_per_position": (np.diag(gram) / max(self.positions[name], 1)).tolist(),
            }
        for probe, prompts in self.probe_sums.items():
            stacked = np.stack(list(prompts.values()))
            draws = rng.integers(0, len(stacked), size=(bootstrap, len(stacked)))
            boot = np.stack([stacked[d].sum(axis=0) for d in draws])
            means = boot[:, :-1] / boot[:, -1:]
            total = stacked.sum(axis=0)
            out["probes"][probe] = {
                name: {
                    "n": int(total[-1]),
                    "prompts": len(stacked),
                    "mean": float(total[j] / total[-1]),
                    "ci95": np.percentile(means[:, j], [2.5, 97.5]).tolist(),
                }
                for j, name in enumerate(self.names)
            }
        return out


def behavioral_probes(labeler: StateLabeler, row: CachedRow, vectors: np.ndarray, probs: np.ndarray, codes: np.ndarray):
    """Yield (probe, [n, J] contrasts) at stopping, reflection, and interaction forks.

    Each contrast is the behavior-weighted mean centered shift on one candidate
    group minus that on another (or on all remaining buckets): positive means
    the direction pushes toward the first group at that state.
    """
    slots = np.concatenate([row.candidate_ids, np.full((len(row.candidate_ids), 1), OTHER_TOKEN)], axis=-1)
    for probe, (state, positive, negative) in PROBES.items():
        positions = np.nonzero(row.loss_mask & (codes == STATE_CODE[state]))[0]
        if not positions.size:
            continue
        p, s, u = probs[positions], slots[positions], vectors[positions]
        is_positive = labeler.member(positive, s)
        is_negative = labeler.member(negative, s) if negative is not None else ~is_positive
        mass_positive, mass_negative = (p * is_positive).sum(-1), (p * is_negative).sum(-1)
        usable = (mass_positive > 1e-6) & (mass_negative > 1e-6)
        if usable.any():
            contrast = np.einsum("tk,tkj->tj", p * is_positive, u) / mass_positive[:, None].clip(1e-12)
            contrast -= np.einsum("tk,tkj->tj", p * is_negative, u) / mass_negative[:, None].clip(1e-12)
            yield probe, contrast[usable]


@dataclass(frozen=True)
class SliceOptions:
    """Which state slices to report beyond ``all`` and the token-state types."""

    min_coverage: float = 0.9
    min_support: float = 0.5
    prompt_metadata: Mapping[str, Mapping[str, str]] = field(default_factory=dict)


def row_slices(row: CachedRow, codes: np.ndarray, sources: tuple[str, ...], options: SliceOptions) -> dict[str, np.ndarray]:
    mask = row.loss_mask
    slices = {"all": mask}
    slices.update({name: mask & (codes == STATE_CODE[name]) for name in STATE_TYPES})
    bins = position_bins(len(mask), POSITION_SLICES)
    slices.update({f"position:{b}": mask & (bins == b) for b in range(POSITION_SLICES)})
    supported = np.ones_like(mask)
    for source in sources:
        supported &= (row.coverage(source) >= options.min_coverage) & (row.support[source] >= options.min_support)
    slices["supported"] = mask & supported
    slices["unsupported"] = mask & ~supported
    for key, value in options.prompt_metadata.get(row.prompt_id, {}).items():
        slices[f"{key}={value}"] = mask
    return slices


def analyze(
    rows: Iterator[CachedRow] | list[CachedRow],
    directions: list[Direction],
    labeler: StateLabeler,
    *,
    options: SliceOptions = SliceOptions(),
) -> dict[str, GeometryAccumulator]:
    """Accumulate Fisher geometry and behavioral probes in one pass."""
    accumulator = GeometryAccumulator([direction.name for direction in directions])
    sources = tuple(sorted({source for direction in directions for source in direction.sources}))
    for row in rows:
        probs = bucket_probs(row.behavior_log_probs)
        vectors = direction_vectors(row, directions, probs)
        codes = labeler.label(row.response_tokens, row.candidate_ids, np.exp(row.behavior_log_probs))
        slices = row_slices(row, codes, sources, options)
        accumulator.add(row.prompt_id, fisher_products(vectors, probs), slices)
        for probe, contrasts in behavioral_probes(labeler, row, vectors, probs, codes):
            accumulator.add_probe(row.prompt_id, probe, contrasts)
    return {"raw": accumulator}


def load_spec(path: Path) -> tuple[list[Direction], dict]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    directions = [Direction.from_spec(name, item) for name, item in spec["directions"].items()]
    return directions, spec


def load_untrained(path: Path | None) -> dict[str, set[int]] | None:
    return {name: set(ids) for name, ids in json.loads(Path(path).read_text()).items()} if path else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path, help="sealed agent-acc cache (anchor/final)")
    parser.add_argument("--donor", action="append", default=[], help="NAME=DIR of a post+pre scored donor chain")
    parser.add_argument("--spec", required=True, type=Path, help="JSON {directions: {...}}")
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--untrained", type=Path, help="JSON {source: [token ids]} of untrained tokens")
    parser.add_argument("--prompt-metadata", type=Path, help="JSON {prompt_id: {slice key: value}}")
    parser.add_argument("--fork-mass", type=float, default=FORK_MASS, help="state-labeler fork threshold")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--bootstrap", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directions, spec = load_spec(args.spec)
    donors = dict(item.split("=", 1) for item in args.donor)
    untrained = load_untrained(args.untrained)
    table = TokenTable.from_tokenizer_json(args.tokenizer_json)
    labeler = StateLabeler(table, fork_mass=args.fork_mass)
    options = SliceOptions(
        prompt_metadata=json.loads(args.prompt_metadata.read_text()) if args.prompt_metadata else {}
    )

    def rows():
        return iter_joined_rows(args.base, donors, untrained=untrained, max_rows=args.max_rows)

    report = {"spec": spec, "donors": donors, "fork_mass": args.fork_mass}
    runs = analyze(rows(), directions, labeler, options=options)
    for name, accumulator in runs.items():
        report[name] = accumulator.summary(bootstrap=args.bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
