"""Multi-turn failure classes, the loss decomposition, turn-level behavior, and ground-truth validation."""

import json

import numpy as np
import pytest

from evaluation.bfcl_multiturn_failures import CLASSES, analyze, behavior, classify, load_multiturn


def act(*names):
    """One turn with a single step calling ``names``."""
    return [[{name: json.dumps({"x": 1})} for name in names]]


def state(turn):
    return {"error_type": "multi_turn:instance_state_mismatch", "execution_result": [{}] * (turn + 1)}


def empty(turn):
    return {"error_type": "multi_turn:empty_turn_model_response", "error_message": f"... is empty for turn {turn}"}


REFERENCE = [["cd(folder='a')", "mv(source='f', destination='b')"], [], ["sort('f')"]]


def test_classify_reads_the_failing_turn_and_assigns_one_class():
    ok = act("cd", "mv")
    assert classify([ok, ["Which file?"], act("sort")], REFERENCE, state(2), overflow=False) == (
        "wrong_arguments",
        2,
    )
    assert classify([act("cd"), [], []], REFERENCE, state(0), overflow=False) == ("missing_calls", 0)
    assert classify([act("cd", "ls"), [], []], REFERENCE, state(0), overflow=False) == ("different_calls", 0)
    assert classify([act("cd", "mv", "rm"), [], []], REFERENCE, state(0), overflow=False) == ("extra_calls", 0)
    assert classify([ok, ["?"], ["Done."]], REFERENCE, empty(2), overflow=False) == ("no_action", 2)
    # sorting on the unanswerable turn is acting early, even if the failing turn then does nothing
    assert classify([ok, act("sort"), ["Already sorted."]], REFERENCE, empty(2), overflow=False) == (
        "acted_early",
        2,
    )
    # an unrelated call on the unanswerable turn is not; sorting twice is a repeat
    assert classify([ok, act("ls"), act("sort", "sort")], REFERENCE, state(2), overflow=False) == (
        "repeated_calls",
        2,
    )
    assert classify([act("cd", "mv", "mv", "rm"), [], []], REFERENCE, state(0), overflow=False) == ("extra_calls", 0)
    forced = {"error_type": "multi_turn:force_terminated"}
    assert classify([ok, act("cd")], REFERENCE, forced, overflow=False) == ("step_limit", 1)
    assert classify([ok], REFERENCE, forced, overflow=True) == ("overflow", 0)
    with pytest.raises(ValueError, match="unexpected"):
        classify([ok], REFERENCE, {"error_type": "multi_turn:other"}, overflow=False)


def test_behavior_counts_correctness_until_the_failure_and_cost_over_all_turns():
    turns = [act("cd", "cd") + ["Moved."], act("sort"), ["Sorted."]]
    counts = behavior(turns, [[100, 10], [50], [70]], REFERENCE, last=1)
    assert counts["action_turns"] == 1 and counts["missing"] == 1 and counts["called_action_turns"] == 1
    assert counts["repeated"] == 1 and counts["identical_repeats"] == 1 and counts["on_path_calls"] == 3
    assert counts["unanswerable_turns"] == 1 and counts["early"] == 1
    assert counts["turns"] == 3 and counts["first_step_tokens"] == 220
    assert counts["later_steps"] == 1 and counts["later_step_tokens"] == 10
    assert counts["first_turns"] == 1 and counts["first_step_tokens_first_turn"] == 100
    assert counts["later_turns"] == 2 and counts["first_step_tokens_later_turns"] == 120


def record(correct, label=None, turn=None, no_call=0):
    counts = {
        name: 1
        for name in (
            "turns",
            "steps",
            "calls",
            "on_path_calls",
            "first_step_tokens",
            "first_turns",
            "first_step_tokens_first_turn",
            "later_turns",
            "first_step_tokens_later_turns",
            "later_steps",
            "later_step_tokens",
        )
    }
    counts.update(action_turns=1, no_call=no_call, called_action_turns=1, unanswerable_turns=1)
    return {"correct": correct, "overflow": False, "class": label, "failure_turn": turn, "behavior": counts}


def test_decomposition_sums_to_the_accuracy_change_and_pools_pairs():
    base = {f"e{i}": record(True) for i in range(6)}
    arm = {**base, "e0": record(False, "no_action", 0, no_call=1), "e1": record(False, "missing_calls", 3)}
    ref2 = {**base, "e2": record(False, "missing_calls", 1)}
    arm2 = {**base, "e3": record(False, "no_action", 2, no_call=1)}
    runs = {"arm": arm, "ref": base, "arm2": arm2, "ref2": ref2}
    report = analyze(runs, {"pooled": [("arm", "ref"), ("arm2", "ref2")]}, draws=200)["comparisons"]["pooled"]
    by_class = {label: item["delta"] for label, item in report["by_class"].items()}
    assert set(by_class) == set(CLASSES)
    # pair 1: +1 no_action, +1 missing; pair 2: +1 no_action, −1 missing; averaged over 6 entries
    assert by_class["no_action"] == pytest.approx(100 * 2 / 2 / 6)
    assert by_class["missing_calls"] == pytest.approx(0.0)
    assert sum(by_class.values()) == pytest.approx(-report["accuracy_change"]["delta"])
    assert sum(item["delta"] for item in report["by_failure_turn"].values()) == pytest.approx(
        -report["accuracy_change"]["delta"]
    )
    assert report["by_failure_turn"]["3+"]["delta"] == pytest.approx(100 / 2 / 6)
    assert report["behavior"]["no_call"]["delta"] == pytest.approx(1 / 6)
    low, high = report["accuracy_change"]["ci95"]
    assert low <= report["accuracy_change"]["delta"] <= high
    assert np.isfinite(report["behavior"]["first_step_tokens"]["relative"])


def write_run(root, correct_state, graded_reference):
    categories = ("multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context")
    (root / "scores").mkdir(parents=True)
    for category in categories:
        directory = root / f"bfcl_v3.{category}"
        directory.mkdir()
        rows, failures = [], []
        if category == "multi_turn_base":
            generation = [act("cd") + act("mv"), ["Which file?"], act("sort")]
            rows.append(
                {
                    "id": "multi_turn_base_0",
                    "generation": generation,
                    "num_generated_tokens_list": [5, 5, 5, 5],
                    "num_generated_tokens": 20,
                    "question": [[{"role": "user", "content": "q"}]] * 3,
                    "is_correct": correct_state,
                }
            )
            if not correct_state:
                failures.append({"id": "multi_turn_base_0", "possible_answer": graded_reference, "error": state(2)})
        (directory / "output.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        header = json.dumps({"accuracy": 0.0})
        body = "".join("\n" + json.dumps(f) for f in failures)
        (root / "scores" / f"BFCL_v4_{category}_score.json").write_text(header + body)


def test_load_validates_the_graded_reference_and_correctness(tmp_path):
    truth = {"multi_turn_base_0": REFERENCE}
    write_run(tmp_path / "wrong", False, REFERENCE)
    loaded = load_multiturn(tmp_path / "wrong", truth)["multi_turn_base_0"]
    assert (loaded["class"], loaded["failure_turn"]) == ("wrong_arguments", 2)
    assert loaded["behavior"]["action_turns"] == 2 and loaded["behavior"]["later_steps"] == 1
    write_run(tmp_path / "stale", False, [["cd(folder='a')"], [], ["sort('f')"]])
    with pytest.raises(ValueError, match="gorilla pin"):
        load_multiturn(tmp_path / "stale", truth)
    write_run(tmp_path / "right", True, REFERENCE)
    assert load_multiturn(tmp_path / "right", truth)["multi_turn_base_0"]["class"] is None
