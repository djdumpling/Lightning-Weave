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

    ⟨u, v⟩_s = Cov_b(u, v),   KL(q_u ‖ q_v) ≈ Var_b(u − v) / (2α²).

**Evidence.** A source has no evidence at a candidate that it cannot score
(unmapped across tokenizers) or whose row one of its anchors never trained.
Such a candidate gets δ = 0, the same "no change" value the loss assigns to
every unscored token in the remaining-vocabulary bucket, so its odds against
that bucket are untouched. ``coverage`` is the student's behavior mass on the
candidates a source can score; ``support`` is the smaller of the two anchors'
own probability mass on them (whether the donor finds the state plausible).

"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OTHER_FLOOR = 1e-8
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

    def valid(self, source: str) -> np.ndarray:
        return self.mapped[source] & self.trained[source]

    def coverage(self, source: str) -> np.ndarray:
        """[T] behavior mass of the cached candidates on which ``source`` has evidence."""
        topk = np.exp(self.behavior_log_probs)
        return (topk * self.valid(source)).sum(axis=-1) / np.clip(topk.sum(axis=-1), 1e-30, None)


def prompt_unit(prompt_id: str, salt: str) -> float:
    """A deterministic uniform draw per prompt."""
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


def load_untrained(path: Path | None) -> dict[str, set[int]] | None:
    return {name: set(ids) for name, ids in json.loads(Path(path).read_text()).items()} if path else None
