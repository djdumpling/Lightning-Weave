"""Synthetic shift targets: legacy parity, evidence handling, gates, KL calibration, and content addressing."""

import json
from argparse import Namespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_curation import build_synthetic_shift_target as synthetic
from data_curation.build_direct_opd_composed_target import compose_anchor_deltas
from data_curation.common import file_sha256, write_json
from data_curation.composition import flatten_metadata
from data_curation.shift_geometry import (
    CachedRow,
    Direction,
    behavioral_probes,
    bucket_probs,
    direction_vectors,
    iter_joined_rows,
    log_tilted_target,
)
from data_curation.shift_states import STATE_CODE, StateLabeler, TokenTable
from slime.rollout.offline_direct_opd import SCHEMA_VERSION, validate_sealed_manifest

K, LENGTH, ROWS = 3, 5, 8


def cache(root, *, seed, sealed=True, loss_mask=None, candidate_valid=None, shift_scale=1.0):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    names = (
        "sample_id",
        "prompt_id",
        "response_tokens",
        "candidate_ids",
        "behavior_topk_log_probs",
        "loss_mask",
        "pre_teacher_log_probs",
        "post_teacher_log_probs",
        "pre_teacher_revision",
        "post_teacher_revision",
        "candidate_projection_valid_mask",
    )
    fields = {name: [] for name in names}
    for index in range(ROWS):
        behavior = np.random.default_rng(100 + index).normal(size=(LENGTH, 6))
        behavior = behavior - np.log(np.exp(behavior).sum(-1, keepdims=True))
        pre = rng.normal(size=(LENGTH, K)) - 3
        fields["sample_id"].append(f"s{index}")
        fields["prompt_id"].append(f"p{index // 2}")
        fields["response_tokens"].append(list(range(1, LENGTH + 1)))
        fields["candidate_ids"].append([[1, 2, 3]] * LENGTH)
        fields["behavior_topk_log_probs"].append((-np.sort(-behavior, axis=-1)[:, :K]).tolist())
        fields["loss_mask"].append(loss_mask or [True] * LENGTH)
        fields["pre_teacher_log_probs"].append(pre.tolist())
        fields["post_teacher_log_probs"].append((pre + shift_scale * rng.normal(size=(LENGTH, K))).tolist())
        fields["pre_teacher_revision"].append("pre")
        fields["post_teacher_revision"].append("post")
        fields["candidate_projection_valid_mask"].append(candidate_valid or [[True] * K] * LENGTH)
    floats = pa.list_(pa.list_(pa.float32()))
    types = {
        "response_tokens": pa.list_(pa.int32()),
        "candidate_ids": pa.list_(pa.list_(pa.int32())),
        "behavior_topk_log_probs": floats,
        "pre_teacher_log_probs": floats,
        "post_teacher_log_probs": floats,
        "loss_mask": pa.list_(pa.bool_()),
        "candidate_projection_valid_mask": pa.list_(pa.list_(pa.bool_())),
    }
    metadata = pa.StructArray.from_arrays(
        [pa.array(v, type=types.get(n)) for n, v in fields.items()], names=list(fields)
    )
    table = pa.table({"prompt": [f"prompt-{i // 2}" for i in range(ROWS)], "label": [""] * ROWS, "metadata": metadata})
    shards = []
    for index, offset in enumerate(range(0, ROWS, 4)):
        path = root / f"rollouts-r00000-{index:05d}.parquet"
        pq.write_table(table.slice(offset, 4), path)
        shards.append({"path": path.name, "rows": 4, "sha256": file_sha256(path)})
    if sealed:
        write_json(
            {
                "schema_version": SCHEMA_VERSION,
                "sealed": True,
                "top_k": K,
                "student_model": {"revision": "student", "path": "/models/student"},
                "pre_teacher_model": {"revision": "pre"},
                "post_teacher_model": {"revision": "post"},
                "tokenizer_hash": "tokenizer",
                "model_vocab_size": 32,
                "generation_config": {"responses_per_prompt": 2},
                "loss_mask_storage_dtype": "bool",
                "score_storage_dtype": "float32",
                "token_storage_dtype": "int32",
                "source_dataset_sha256": "s" * 64,
                "total_rows": ROWS,
                "shards": shards,
            },
            root / "manifest.json",
        )
    return root


def run_main(tmp_path, spec, donors=(), *, name="variant", untrained=None, prompt_weights=None):
    spec_path = tmp_path / f"{name}.json"
    spec_path.write_text(json.dumps(spec))
    untrained_path = None
    if untrained:
        untrained_path = tmp_path / f"{name}-untrained.json"
        untrained_path.write_text(json.dumps(untrained))
    output = tmp_path / "synthetic" / name
    args = Namespace(
        base=tmp_path / "base",
        base_name="agent_acc",
        donor=[f"{donor}={tmp_path / donor}" for donor in donors],
        spec=spec_path,
        tokenizer_json=None,
        untrained=untrained_path,
        calibration_fraction=1.0,
        output_dir=output,
        prompt_weights=[],
    )
    for weight_name, weights in (prompt_weights or {}).items():
        path = tmp_path / f"{name}-{weight_name}-weights.json"
        path.write_text(json.dumps(weights))
        args.prompt_weights.append(f"{weight_name}={path}")
    original = synthetic.parse_args
    synthetic.parse_args = lambda: args
    try:
        synthetic.main()
    finally:
        synthetic.parse_args = original
    manifest = validate_sealed_manifest(output / "manifest.json")
    tables = [pq.read_table(output / shard["path"]) for shard in manifest["shards"]]
    return output, manifest, flatten_metadata(pa.concat_tables(tables))


def scores(root, field):
    tables = [pq.read_table(path) for path in sorted(root.glob("*.parquet"))]
    return [
        np.asarray(item, dtype=np.float32)
        for item in flatten_metadata(pa.concat_tables(tables)).field(field).to_pylist()
    ]


def term(*sources, transform="evidence", **extra):
    return {"transform": transform, "direction": {"terms": [{"source": s, "coef": c} for s, c in sources]}, **extra}


def delta(metadata, row):
    post = np.asarray(metadata.field("post_teacher_log_probs")[row].as_py(), dtype=np.float64)
    pre = np.asarray(metadata.field("pre_teacher_log_probs")[row].as_py(), dtype=np.float64)
    return np.concatenate([post - pre, np.zeros((LENGTH, 1))], axis=-1)


def test_raw_terms_reproduce_the_reference_composer_bit_for_bit(tmp_path):
    cache(tmp_path / "base", seed=0)
    cache(tmp_path / "donor", seed=1, sealed=False)
    spec = {"alpha": 2.0, "terms": [term(("agent_acc", 1.0), transform="raw"), term(("donor", 0.75), transform="raw")]}
    _, manifest, metadata = run_main(tmp_path, spec, donors=["donor"])
    pres = [scores(tmp_path / name, "pre_teacher_log_probs") for name in ("base", "donor")]
    posts = [scores(tmp_path / name, "post_teacher_log_probs") for name in ("base", "donor")]
    for row in range(ROWS):
        expected_pre, expected_post, _ = compose_anchor_deltas(
            [pres[0][row], pres[1][row]], [posts[0][row], posts[1][row]], [1.0, 0.75]
        )
        np.testing.assert_array_equal(np.asarray(metadata.field("pre_teacher_log_probs")[row].as_py()), expected_pre)
        np.testing.assert_array_equal(np.asarray(metadata.field("post_teacher_log_probs")[row].as_py()), expected_post)
    assert manifest["post_teacher_model"]["model_type"] == "synthetic_shift_composition"


def test_raw_terms_refuse_placeholder_scores(tmp_path):
    cache(tmp_path / "base", seed=0)
    cache(tmp_path / "donor", seed=1, sealed=False, candidate_valid=[[True, False, True]] * LENGTH)
    with pytest.raises(ValueError, match="placeholder"):
        run_main(tmp_path, {"alpha": 2.0, "terms": [term(("donor", 1.0), transform="raw")]}, donors=["donor"])


def test_evidence_terms_leave_unscorable_candidates_at_their_behavior_odds(tmp_path):
    cache(tmp_path / "base", seed=0)
    cache(tmp_path / "donor", seed=1, sealed=False, candidate_valid=[[True, False, True]] * LENGTH)
    _, _, metadata = run_main(tmp_path, {"alpha": 2.0, "terms": [term(("donor", 1.0))]}, donors=["donor"])
    behavior = [np.asarray(x) for x in metadata.field("behavior_topk_log_probs").to_pylist()]
    for row in range(ROWS):
        probs = bucket_probs(behavior[row])
        target = log_tilted_target(probs, delta(metadata, row), 2.0)
        np.testing.assert_allclose(target[:, 1] - target[:, -1], np.log(probs[:, 1] / probs[:, -1]), atol=1e-5)


def test_untrained_tokens_are_dropped_only_from_evidence_terms(tmp_path):
    cache(tmp_path / "base", seed=0)
    untrained = {"agent_acc": [2]}
    _, _, clean = run_main(
        tmp_path, {"alpha": 2.0, "terms": [term(("agent_acc", 1.0))]}, name="clean", untrained=untrained
    )
    _, _, legacy = run_main(
        tmp_path,
        {"alpha": 2.0, "terms": [term(("agent_acc", 1.0), transform="raw")]},
        name="legacy",
        untrained=untrained,
    )
    raw = (
        scores(tmp_path / "base", "post_teacher_log_probs")[0] - scores(tmp_path / "base", "pre_teacher_log_probs")[0]
    )
    np.testing.assert_array_equal(delta(clean, 0)[:, 1], 0.0)
    np.testing.assert_allclose(delta(legacy, 0)[:, 1], raw[:, 1], atol=1e-6)


def test_donor_positions_without_evidence_keep_the_base_mask(tmp_path):
    cache(tmp_path / "base", seed=0, loss_mask=[True, True, False, True, True])
    cache(tmp_path / "donor", seed=1, sealed=False, loss_mask=[False, True, True, True, True])
    _, _, metadata = run_main(tmp_path, {"alpha": 2.0, "terms": [term(("donor", 1.0))]}, donors=["donor"])
    for row in range(ROWS):
        assert metadata.field("loss_mask")[row].as_py() == [True, True, False, True, True]
        np.testing.assert_array_equal(delta(metadata, row)[0], 0.0)
        assert np.abs(delta(metadata, row)[1]).max() > 0


def test_agreement_gates_require_evidence_on_both_sides(tmp_path):
    cache(tmp_path / "base", seed=0)
    cache(tmp_path / "donor", seed=1, sealed=False)
    cache(tmp_path / "silent", seed=2, sealed=False, loss_mask=[False] * LENGTH)
    gate = {"agree_with": {"terms": [{"source": "silent", "coef": 1.0}]}, "min_cosine": 0.0}
    _, _, metadata = run_main(
        tmp_path, {"alpha": 2.0, "terms": [term(("donor", 1.0), gate=gate)]}, donors=["donor", "silent"]
    )
    for row in range(ROWS):
        np.testing.assert_array_equal(delta(metadata, row), 0.0)


def test_kl_budget_is_met_bounded_and_reported(tmp_path):
    cache(tmp_path / "base", seed=0)
    cache(tmp_path / "donor", seed=1, sealed=False)
    spec = {"alpha": 2.0, "terms": [term(("agent_acc", 1.0), transform="raw"), term(("donor", 1.0), kl_budget=0.01)]}
    _, manifest, _ = run_main(tmp_path, spec, donors=["donor"])
    calibration = manifest["post_teacher_model"]["calibration"]
    assert calibration["achieved_kl"] == pytest.approx(0.01, rel=1e-3)
    assert set(calibration["active_kl_quantiles"]) == {"50", "90", "95", "99", "100"}
    unreachable = {"alpha": 2.0, "terms": [term(("donor", 1.0), kl_budget=50.0, max_multiplier=2.0)]}
    with pytest.raises(ValueError, match="unreachable"):
        run_main(tmp_path, unreachable, donors=["donor"], name="unreachable")
    with pytest.raises(ValueError, match="at most one"):
        synthetic.SyntheticComposer(
            {"alpha": 2.0, "terms": [term(("donor", 1.0), kl_budget=0.01), term(("agent_acc", 1.0), kl_budget=0.01)]},
            labeler=None,
            vocab_size=10,
        )


def test_outputs_record_the_spec_and_are_never_overwritten(tmp_path):
    cache(tmp_path / "base", seed=0)
    spec = {"alpha": 2.0, "terms": [term(("agent_acc", 1.0))]}
    _, manifest, _ = run_main(tmp_path, spec)
    assert manifest["post_teacher_model"]["spec"] == spec
    with pytest.raises(FileExistsError):
        run_main(tmp_path, spec)


def test_state_gates_require_a_tokenizer():
    with pytest.raises(ValueError, match="tokenizer"):
        synthetic.SyntheticComposer(
            {"alpha": 2.0, "terms": [term(("agent_acc", 1.0), gate={"state_types": ["think_body"]})]},
            labeler=None,
            vocab_size=10,
        )


def test_heuristic_pushes_move_only_their_groups_at_their_forks():
    vocab = {"Ġa": 0, ".Ċ": 1, ".ĊĊ": 2, "Wait": 3, "So": 4}
    added = {"<think>": 20, "</think>": 21, "<tool_call>": 22, "</tool_call>": 23, "<|im_end|>": 24}
    labeler = StateLabeler(TokenTable.from_vocab(vocab, added))
    tokens = np.asarray([20, 0, 1, 2, 3, 0, 21])
    candidates = np.asarray([[20, 0, 1], [0, 1, 2], [1, 2, 0], [2, 1, 0], [3, 4, 0], [0, 1, 2], [21, 0, 1]])
    probs = np.full((len(tokens), 3), [0.97, 0.01, 0.01])
    probs[2], probs[4] = [0.6, 0.3, 0.05], [0.5, 0.4, 0.05]
    row = CachedRow("s0", "p0", tokens, candidates, np.log(probs), np.ones(len(tokens), dtype=bool))
    pushes = [
        {"state": "think_stop_fork", "group": "single_newline", "shift": 0.5},
        {"state": "think_stop_fork", "group": "paragraph_break", "shift": -0.5},
        {"state": "think_reflection_fork", "group": "reflection", "shift": -2.0},
        {"state": "think_reflection_fork", "group": "conclusion", "shift": 2.0},
    ]
    composer = synthetic.SyntheticComposer(
        {"alpha": 2.0, "terms": [{"transform": "heuristic", "pushes": pushes}]}, labeler=labeler, vocab_size=32
    )
    assert composer.sources == set()
    bucket, (vector,) = composer.row_terms(row)
    codes = labeler.label(tokens, candidates, probs)
    assert (codes[2], codes[4]) == (STATE_CODE["think_stop_fork"], STATE_CODE["think_reflection_fork"])
    expected = np.zeros((len(tokens), 4))
    expected[2, :2], expected[4, :2] = [0.5, -0.5], [-2.0, 2.0]
    np.testing.assert_array_equal(vector, expected)
    # The fork probes read exactly twice each push, which is how the control is matched to a donor arm.
    probes = {
        name: contrast[:, 0] for name, contrast in behavioral_probes(labeler, row, vector[..., None], bucket, codes)
    }
    np.testing.assert_allclose(probes["stop_minus_continue"], [1.0])
    np.testing.assert_allclose(probes["reflect_minus_conclude"], [-4.0])


def test_heuristic_terms_are_validated():
    push = {"state": "think_stop_fork", "group": "single_newline", "shift": 1.0}
    with pytest.raises(ValueError, match="tokenizer"):
        synthetic.SyntheticComposer(
            {"alpha": 2.0, "terms": [{"transform": "heuristic", "pushes": [push]}]}, labeler=None, vocab_size=10
        )
    with pytest.raises(ValueError, match="no direction"):
        synthetic.Term(0, {"transform": "heuristic", "pushes": [push], "direction": {"terms": []}})
    with pytest.raises(ValueError, match="no direction"):
        synthetic.Term(0, term(("agent_acc", 1.0), pushes=[push]))
    with pytest.raises(ValueError, match="push states"):
        synthetic.Term(0, {"transform": "heuristic", "pushes": [push | {"state": "stop"}]})


def test_equalize_weights_each_source_to_unit_fisher_rms(tmp_path):
    cache(tmp_path / "base", seed=0)
    cache(tmp_path / "small", seed=1, sealed=False)
    cache(tmp_path / "large", seed=2, sealed=False, shift_scale=5.0)
    spec = {"alpha": 2.0, "terms": [term(("small", 1.0), ("large", 1.0), equalize=True)]}
    _, manifest, metadata = run_main(tmp_path, spec, donors=["small", "large"])
    scales = manifest["post_teacher_model"]["equalization"]["0"]
    assert 3.0 < scales["small"] / scales["large"] < 8.0
    donors = {"small": tmp_path / "small", "large": tmp_path / "large"}
    energy, positions, expected = {"small": 0.0, "large": 0.0}, 0, []
    for row in iter_joined_rows(tmp_path / "base", donors):
        probs = bucket_probs(row.behavior_log_probs)
        components = [Direction(name, ((name, scales[name]),)) for name in ("small", "large")]
        vectors = direction_vectors(row, components, probs)
        for j, name in enumerate(("small", "large")):
            energy[name] += float((probs * vectors[..., j] ** 2)[row.loss_mask].sum())
        positions += int(row.loss_mask.sum())
        expected.append(vectors.sum(-1))
    for name in energy:
        assert energy[name] / positions == pytest.approx(1.0)
    # The sealed target is the sum of the unit-RMS components (up to the per-state constant re-leveling removes).
    behavior = [np.asarray(x) for x in metadata.field("behavior_topk_log_probs").to_pylist()]
    for row, target in enumerate(expected):
        probs = bucket_probs(behavior[row])
        composed = delta(metadata, row)
        np.testing.assert_allclose(composed - (probs * composed).sum(-1, keepdims=True), target, atol=1e-4)


def test_equalize_auto_keeps_raw_weights_for_comparable_sources(tmp_path):
    cache(tmp_path / "base", seed=0)
    cache(tmp_path / "one", seed=1, sealed=False)
    cache(tmp_path / "two", seed=2, sealed=False)
    auto = {"alpha": 2.0, "terms": [term(("one", 1.0), ("two", 1.0), equalize="auto")]}
    raw = {"alpha": 2.0, "terms": [term(("one", 1.0), ("two", 1.0))]}
    _, manifest, metadata = run_main(tmp_path, auto, donors=["one", "two"], name="auto")
    _, plain, reference = run_main(tmp_path, raw, donors=["one", "two"], name="raw")
    model = manifest["post_teacher_model"]
    rule = model["equalization_rules"]["0"]
    assert rule["rule"] == "raw" and 1.0 <= rule["ratio"] <= 2.0 and rule["max_ratio"] == 2.0
    assert rule["ratio"] == pytest.approx(max(rule["rms"].values()) / min(rule["rms"].values()))
    assert model["equalization"]["0"] == {"one": 1.0, "two": 1.0}
    assert "equalization_rules" not in plain["post_teacher_model"]
    for field in ("post_teacher_log_probs", "pre_teacher_log_probs", "loss_mask"):  # exactly the plain raw sum
        assert metadata.field(field).to_pylist() == reference.field(field).to_pylist()
    # the rule is inclusive: at exactly the measured ratio the weights stay raw
    boundary = {
        "alpha": 2.0,
        "terms": [term(("one", 1.0), ("two", 1.0), equalize="auto", equalize_max_ratio=rule["ratio"])],
    }
    _, boundary_manifest, _ = run_main(tmp_path, boundary, donors=["one", "two"], name="boundary")
    assert boundary_manifest["post_teacher_model"]["equalization_rules"]["0"]["rule"] == "raw"


def test_equalize_auto_equalizes_a_dominant_source_like_equalize_true(tmp_path):
    cache(tmp_path / "base", seed=0)
    cache(tmp_path / "small", seed=1, sealed=False)
    cache(tmp_path / "large", seed=2, sealed=False, shift_scale=5.0)
    sources = (("small", 1.0), ("large", 1.0))
    auto = {"alpha": 2.0, "terms": [term(*sources, equalize="auto")]}
    equal = {"alpha": 2.0, "terms": [term(*sources, equalize=True)]}
    lenient = {"alpha": 2.0, "terms": [term(*sources, equalize="auto", equalize_max_ratio=100.0)]}
    _, manifest, metadata = run_main(tmp_path, auto, donors=["small", "large"], name="auto")
    _, reference_manifest, reference = run_main(tmp_path, equal, donors=["small", "large"], name="equal")
    _, lenient_manifest, _ = run_main(tmp_path, lenient, donors=["small", "large"], name="lenient")
    rule = manifest["post_teacher_model"]["equalization_rules"]["0"]
    assert rule["rule"] == "equalized" and rule["ratio"] > 2.0
    assert manifest["post_teacher_model"]["equalization"] == reference_manifest["post_teacher_model"]["equalization"]
    for row in range(ROWS):
        np.testing.assert_allclose(delta(metadata, row), delta(reference, row), atol=1e-6)
    assert lenient_manifest["post_teacher_model"]["equalization_rules"]["0"]["rule"] == "raw"
    # equalize: true targets predate "auto": they record no rule, so their revisions are unchanged
    assert "equalization_rules" not in reference_manifest["post_teacher_model"]
    assert lenient_manifest["post_teacher_model"]["equalization"]["0"] == {"small": 1.0, "large": 1.0}


def test_equalize_options_are_validated():
    pair = (("one", 1.0), ("two", 1.0))
    with pytest.raises(ValueError, match="equalize is"):
        synthetic.Term(0, term(*pair, equalize="yes"))
    assert synthetic.Term(0, term(*pair, equalize=None)).equalize is False
    for ratio in (None, True, float("inf"), "2"):
        with pytest.raises(ValueError, match="equalize_max_ratio"):
            synthetic.Term(0, term(*pair, equalize="auto", equalize_max_ratio=ratio))
    for bad in (
        term(*pair, equalize=True, equalize_max_ratio=2.0),
        term(*pair, equalize="auto", equalize_max_ratio=0.5),
    ):
        with pytest.raises(ValueError, match="equalize_max_ratio"):
            synthetic.Term(0, bad)
    with pytest.raises(ValueError, match="equalize weights"):
        synthetic.Term(0, term(("one", 1.0), equalize="auto"))
    assert synthetic.Term(0, term(*pair, equalize="auto")).equalize_max_ratio == 2.0


def test_calibration_reports_where_the_kl_lands_by_state():
    vocab = {"Ġa": 0, ".Ċ": 1, ".ĊĊ": 2, "Wait": 3, "So": 4}
    added = {"<think>": 20, "</think>": 21, "<tool_call>": 22, "</tool_call>": 23, "<|im_end|>": 24}
    labeler = StateLabeler(TokenTable.from_vocab(vocab, added))
    tokens = np.asarray([20, 0, 2, 3, 0, 2, 3, 0, 21])
    fork, plain = [3, 4, 0], [0, 1, 2]
    candidates = np.asarray([[20, 0, 1], plain, [2, 1, 0], fork, plain, [2, 1, 0], fork, plain, [21, 0, 1]])
    probs = np.full((len(tokens), 3), [0.97, 0.01, 0.01])
    probs[3] = probs[6] = [0.5, 0.4, 0.05]
    ones = np.ones(candidates.shape, dtype=bool)
    row = CachedRow(
        "s0",
        "p0",
        tokens,
        candidates,
        np.log(probs),
        np.ones(len(tokens), dtype=bool),
        shifts={"d": np.where(candidates == 3, 1.0, 0.0)},  # a push on "Wait", a candidate only at the two forks
        mapped={"d": ones},
        trained={"d": ones},
        support={"d": np.ones(len(tokens))},
    )
    spec = {"alpha": 2.0, "terms": [term(("d", 1.0), kl_budget=0.01, max_multiplier=100.0)]}
    report = synthetic.SyntheticComposer(spec, labeler=labeler, vocab_size=32).calibrate([row])
    fork = "think_reflection_fork"
    assert report["position_share_by_state"][fork] == pytest.approx(2 / 9)
    assert report["kl_share_by_state"][fork] == pytest.approx(1.0)
    assert sum(report["kl_share_by_state"].values()) == pytest.approx(1.0)
    assert report["mean_kl_by_state"][fork] == pytest.approx(0.01 * 9 / 2, rel=1e-3)
    assert report["kl_share_of_top_positions"]["10%"] == pytest.approx(0.5)  # one of the two forks
    unlabeled = synthetic.SyntheticComposer(spec, labeler=None, vocab_size=32).calibrate([row])
    assert "kl_share_by_state" not in unlabeled and unlabeled["effective_coef"] == report["effective_coef"]


def test_masking_a_call_does_not_preserve_the_decision_but_excluding_the_state_does():
    """A zero shift on an unscorable <tool_call> still lets pushes on competing text move its probability; only
    removing the term at the whole state keeps the decision where the other terms put it."""
    vocab = {"Ġa": 0, ".Ċ": 1, ".ĊĊ": 2}
    added = {"<think>": 20, "</think>": 21, "<tool_call>": 22, "</tool_call>": 23, "<|im_end|>": 24}
    labeler = StateLabeler(TokenTable.from_vocab(vocab, added))
    tokens = np.asarray([20, 0, 21, 0])
    candidates = np.asarray([[20, 0, 1], [0, 1, 2], [21, 0, 1], [22, 0, 1]])  # position 3: call or reply
    probs = np.asarray([[0.97, 0.01, 0.01], [0.6, 0.3, 0.05], [0.97, 0.01, 0.01], [0.49, 0.49, 0.01]])
    mapped = np.ones(candidates.shape, dtype=bool)
    mapped[3, 0] = False  # the donor cannot score <tool_call>
    shift = np.zeros(candidates.shape)
    shift[1, 1] = shift[3, 1] = 1.0  # a push on text, in the reasoning and at the decision
    row = CachedRow(
        "s0",
        "p0",
        tokens,
        candidates,
        np.log(probs),
        np.ones(len(tokens), dtype=bool),
        shifts={"d": shift},
        mapped={"d": mapped},
        trained={"d": np.ones_like(mapped)},
        support={"d": np.ones(len(tokens))},
    )
    codes = labeler.label(tokens, candidates, probs)
    assert codes[3] == STATE_CODE["act_vs_talk"]

    def call_probability(spec):
        buckets, (vector,) = synthetic.SyntheticComposer(spec, labeler=labeler, vocab_size=32).row_terms(row)
        return np.exp(log_tilted_target(buckets, vector, 2.0)), vector

    masked, masked_vector = call_probability({"alpha": 2.0, "terms": [term(("d", 1.0))]})
    assert masked_vector[3, 0] == masked_vector[3, -1]  # no relative shift on the call itself...
    assert masked[3, 0] < 0.49 - 0.1  # ...yet its probability falls (0.49 -> about 0.37)
    gated, gated_vector = call_probability(
        {"alpha": 2.0, "terms": [term(("d", 1.0), gate={"exclude_state_types": ["act_vs_talk"]})]}
    )
    np.testing.assert_allclose(gated[3, :3], probs[3])  # the decision is exactly the behavior's
    np.testing.assert_allclose(gated_vector[1], masked_vector[1])  # reasoning states keep the push
    with pytest.raises(ValueError, match="excluded state types"):
        synthetic.Term(0, term(("d", 1.0), gate={"exclude_state_types": ["tool_decision"]}))


def test_equalize_needs_distinct_sources_of_an_evidence_term():
    for bad in (
        term(("donor", 1.0), equalize=True),
        term(("donor", 1.0), ("donor", 1.0), equalize=True),
        term(("donor", 1.0), ("other", 1.0), transform="raw", equalize=True),
    ):
        with pytest.raises(ValueError, match="equalize"):
            synthetic.Term(0, bad)


def test_exclude_first_reflection_removes_the_term_only_at_the_first_reflection_fork():
    vocab = {"Ġa": 0, ".Ċ": 1, ".ĊĊ": 2, "Wait": 3, "So": 4}
    added = {"<think>": 20, "</think>": 21, "<tool_call>": 22, "</tool_call>": 23, "<|im_end|>": 24}
    labeler = StateLabeler(TokenTable.from_vocab(vocab, added))
    tokens = np.asarray([20, 0, 2, 3, 0, 2, 3, 0, 21])
    fork, plain = [3, 4, 0], [0, 1, 2]
    candidates = np.asarray([[20, 0, 1], plain, [2, 1, 0], fork, plain, [2, 1, 0], fork, plain, [21, 0, 1]])
    probs = np.full((len(tokens), 3), [0.97, 0.01, 0.01])
    probs[3] = probs[6] = [0.5, 0.4, 0.05]
    ones = np.ones(candidates.shape, dtype=bool)
    row = CachedRow(
        "s0",
        "p0",
        tokens,
        candidates,
        np.log(probs),
        np.ones(len(tokens), dtype=bool),
        shifts={"d": np.where(candidates == 3, 1.0, 0.0)},
        mapped={"d": ones},
        trained={"d": ones},
        support={"d": np.ones(len(tokens))},
    )
    spec = {"alpha": 2.0, "terms": [term(("d", 1.0), gate={"exclude_first_reflection": True})]}
    composer = synthetic.SyntheticComposer(spec, labeler=labeler, vocab_size=32)
    codes = labeler.label(tokens, candidates, probs)
    assert list(np.nonzero(codes == STATE_CODE["think_reflection_fork"])[0]) == [3, 6]
    _, (gated,) = composer.row_terms(row)
    _, (ungated,) = synthetic.SyntheticComposer(
        {"alpha": 2.0, "terms": [term(("d", 1.0))]}, labeler=labeler, vocab_size=32
    ).row_terms(row)
    assert np.abs(ungated[3]).max() > 0 and np.abs(ungated[6]).max() > 0
    np.testing.assert_array_equal(gated[3], 0.0)
    np.testing.assert_array_equal(gated[6], ungated[6])
    np.testing.assert_array_equal(np.delete(gated, 3, axis=0), np.delete(ungated, 3, axis=0))


def test_gates_reject_unknown_keys_and_the_reflection_gate_needs_labels():
    with pytest.raises(ValueError, match="unknown gate keys"):
        synthetic.Term(0, term(("d", 1.0), gate={"exclude_first_reflections": True}))
    spec = {"alpha": 2.0, "terms": [term(("d", 1.0), gate={"exclude_first_reflection": True})]}
    with pytest.raises(ValueError, match="tokenizer"):
        synthetic.SyntheticComposer(spec, labeler=None, vocab_size=8)


def test_prompt_weights_scale_the_term_per_prompt_and_are_recorded(tmp_path):
    cache(tmp_path / "base", seed=0)
    cache(tmp_path / "donor", seed=1, sealed=False)
    weights = {"p0": 1.0, "p1": 0.0, "p2": 0.5, "p3": 1.0}
    gated = {"alpha": 2.0, "terms": [term(("donor", 1.0), gate={"prompt_weights": "w"})]}
    _, manifest, weighted = run_main(tmp_path, gated, donors=["donor"], name="weighted", prompt_weights={"w": weights})
    _, _, plain = run_main(tmp_path, {"alpha": 2.0, "terms": [term(("donor", 1.0))]}, donors=["donor"], name="plain")
    prompts = weighted.field("prompt_id").to_pylist()
    for row, prompt in enumerate(prompts):
        # One term, no budget: the composed shift is the weight times the unweighted one.
        np.testing.assert_allclose(delta(weighted, row), weights[prompt] * delta(plain, row), atol=1e-6)
    recorded = manifest["post_teacher_model"]["prompt_weights"]["w"]
    assert recorded["sha256"] == file_sha256(tmp_path / "weighted-w-weights.json")


def test_prompt_weights_must_cover_every_prompt_and_be_named():
    spec = {"alpha": 2.0, "terms": [term(("d", 1.0), gate={"prompt_weights": "w"})]}
    with pytest.raises(ValueError, match="without --prompt-weights"):
        synthetic.SyntheticComposer(spec, labeler=None, vocab_size=8)
    with pytest.raises(ValueError, match="non-negative"):
        synthetic.SyntheticComposer(spec, labeler=None, vocab_size=8, prompt_weights={"w": {"p": -1.0}})
    composer = synthetic.SyntheticComposer(spec, labeler=None, vocab_size=8, prompt_weights={"w": {"other": 1.0}})
    row = CachedRow("s", "p", np.zeros(2), np.zeros((2, 3)), np.log(np.full((2, 3), 0.3)), np.ones(2, dtype=bool))
    row.shifts["d"], row.mapped["d"], row.trained["d"] = np.zeros((2, 3)), np.ones((2, 3), bool), np.ones((2, 3), bool)
    row.support["d"] = np.ones(2)
    with pytest.raises(KeyError, match="no entry for p"):
        composer.row_terms(row)
