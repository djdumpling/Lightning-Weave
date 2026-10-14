"""Teacher rescue: at the student's own states, does a teacher's next turn lead to success more often than the
student's? (CPU analysis of executed branches.)

Every branch continues with the student and the same user simulator, so only the next turn differs. Per state, a
source's per-sample success rate is the mean outcome of its branches there. A teacher's turns are compared with the
student's own cached turns (sampled identically) at equal per-sample budget, within strata fixed by the student's
own outcomes at that state: every student sample failed (can the teacher rescue it?), mixed, or every student
sample succeeded (does the teacher break it?). Intervals bootstrap states.

    python data_curation/teacher_rescue.py --strata strata.json --student RUN-s*/thinking2507 \\
        --teacher agentforge=TEACHER_RUN/thinking2507 --rollouts agentforge=TEACHER_ROLLOUTS ... --output rescue.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_curation.branch_outcomes import branch_outcome, read_rollouts


def state_rates(branch_roots: list[Path], rollouts: list[dict]) -> dict[str, list[float]]:
    """Per state, the outcomes of a source's branches (branch id = sample id prefix)."""
    state_of = {row["sample_id"][:24]: row["prompt_id"] for row in rollouts}
    out: dict[str, list[float]] = collections.defaultdict(list)
    for root in branch_roots:
        for path in glob.glob(str(root / "tau2_*" / "*.json")):
            record = json.loads(Path(path).read_text(encoding="utf-8"))
            outcome = branch_outcome(record)
            if outcome is not None and record["task_id"] in state_of:
                out[state_of[record["task_id"]]].append(outcome)
    return out


def compare(teacher: dict[str, list[float]], student: dict[str, list[float]], states: list[str], draws: int, seed: int) -> dict:
    """Mean per-sample success of teacher and student over the states both have, and their paired difference."""
    keys = [state for state in states if teacher.get(state) and student.get(state)]
    t = {k: sum(teacher[k]) / len(teacher[k]) for k in keys}
    s = {k: sum(student[k]) / len(student[k]) for k in keys}
    if not keys:
        return {"states": 0}
    rng = random.Random(seed)
    diffs = sorted(
        sum(t[k] - s[k] for k in sample) / len(sample)
        for sample in ([rng.choice(keys) for _ in keys] for _ in range(draws))
    )
    return {
        "states": len(keys),
        "teacher": sum(t.values()) / len(keys),
        "student": sum(s.values()) / len(keys),
        "difference": sum(t[k] - s[k] for k in keys) / len(keys),
        "lo": diffs[int(0.025 * draws)],
        "hi": diffs[int(0.975 * draws) - 1],
        "states_with_a_teacher_success": sum(max(teacher[k]) > 0 for k in keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strata", type=Path, required=True, help="JSON {stratum: [prompt ids]}")
    parser.add_argument("--student", type=Path, nargs="+", required=True, help="the student's branch runs")
    parser.add_argument("--student-rollouts", type=Path, required=True)
    parser.add_argument("--teacher", action="append", default=[], help="NAME=BRANCH_ROOT[,BRANCH_ROOT...]")
    parser.add_argument("--teacher-rollouts", action="append", default=[], help="NAME=ROLLOUT_DIR")
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    strata = json.loads(args.strata.read_text(encoding="utf-8"))
    student = state_rates(args.student, read_rollouts(args.student_rollouts))
    rollout_dirs = dict(item.split("=", 1) for item in args.teacher_rollouts)
    report = {}
    for item in args.teacher:
        name, roots = item.split("=", 1)
        teacher = state_rates([Path(root) for root in roots.split(",")], read_rollouts(Path(rollout_dirs[name])))
        report[name] = {stratum: compare(teacher, student, states, args.draws, 0) for stratum, states in strata.items()}
        report[name]["all"] = compare(teacher, student, [s for states in strata.values() for s in states], args.draws, 0)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, rows in report.items():
        for stratum, row in rows.items():
            if row.get("states"):
                print(f"{name:12s} {stratum:8s} states {row['states']:3d}  teacher {row['teacher']:.3f}  student {row['student']:.3f}  "
                      f"diff {row['difference']:+.3f} [{row['lo']:+.3f}, {row['hi']:+.3f}]  rescued {row['states_with_a_teacher_success']}")


if __name__ == "__main__":
    main()
