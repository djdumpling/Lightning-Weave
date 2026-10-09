"""AReaL tau2 training tasks for fresh-state collection, audited against the tau2-bench evaluation tasks.

The fresh-state pilot (docs/agent_efficiency_synthesis.md) runs Qwen3-4B students on
inclusionAI/AReaL-tau2-data training tasks to collect multi-turn decision states. A training
task is excluded when it could rehearse an evaluation task: it names an evaluation task's
reservation/order (the object the evaluation acts on) or user, or its instructions overlap an
evaluation task's text. Airline tasks share the official database (or a near copy of it) with
the evaluation, so shared users are excluded too; there are spare airline tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.tau_bench_eval.config import (  # noqa: E402
    AREAL_REPO as DATASET_REPO,
    AREAL_REVISION as DATASET_REVISION,
    AREAL_TASK_COUNT as TASK_COUNT,
    AREAL_TASKS_FILE as TASKS_FILE,
    AREAL_TASKS_SHA256 as TASKS_SHA256,
)

DOMAINS = ("airline", "retail")
EVAL_TASKS = {"airline": 50, "retail": 114}
TEXT_OVERLAP = 0.2  # word 5-gram Jaccard with any evaluation task

USER_ID = re.compile(r"\b[a-z]+_[a-z]+_\d{3,5}\b")
OBJECT_ID = {
    # Six-character reservation codes with a letter and a digit; flight numbers (HAT123) are shared catalog entries.
    "airline": re.compile(r"\b(?=[A-Z0-9]{6}\b)(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{6}\b"),
    "retail": re.compile(r"#W\d{7}"),
}
FLIGHT_NUMBER = re.compile(r"^HAT\d{3}$")


def domain_of(task: dict) -> str:
    return task["user_scenario"]["instructions"]["domain"]


def instructions_text(task: dict) -> str:
    instructions = task["user_scenario"]["instructions"]
    if isinstance(instructions, str):
        return instructions
    return " ".join(str(value) for key, value in instructions.items() if value and key != "domain")


def entity_ids(domain: str, text: str) -> tuple[set[str], set[str]]:
    objects = {item for item in OBJECT_ID[domain].findall(text) if not FLIGHT_NUMBER.match(item)}
    return objects, set(USER_ID.findall(text))


def shingles(text: str, n: int = 5) -> set[str]:
    words = re.findall(r"[a-z0-9#]+", text.lower())
    return {" ".join(words[index : index + n]) for index in range(max(0, len(words) - n + 1))}


def load_training_tasks(path: Path) -> list[dict]:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != TASKS_SHA256:
        raise ValueError(f"{path} is not {DATASET_REPO}@{DATASET_REVISION}/{TASKS_FILE}")
    tasks = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len(tasks) != TASK_COUNT or len({task["id"] for task in tasks}) != TASK_COUNT:
        raise ValueError(f"expected {TASK_COUNT} unique tasks, found {len(tasks)}")
    return tasks


def audit(training: list[dict], evaluation: dict[str, list[dict]]) -> dict:
    """Eligible task ids per domain and the reason for every exclusion."""
    eligible: dict[str, list[str]] = {domain: [] for domain in DOMAINS}
    excluded: list[dict] = []
    for domain in DOMAINS:
        tasks = evaluation[domain]
        if len(tasks) != EVAL_TASKS[domain]:
            raise ValueError(f"tau2 {domain} has {len(tasks)} evaluation tasks, expected {EVAL_TASKS[domain]}")
        objects, users, texts = set(), set(), []
        for task in tasks:
            actions = json.dumps((task.get("evaluation_criteria") or {}).get("actions") or [])
            found = entity_ids(domain, instructions_text(task) + " " + actions)
            objects |= found[0]
            users |= found[1]
            texts.append(shingles(instructions_text(task)))
        for task in (item for item in training if domain_of(item) == domain):
            text = instructions_text(task)
            # Entities come from the instructions and the task's own gold actions (AReaL stores those as a JSON string).
            criteria = task.get("evaluation_criteria") or {}
            actions = json.dumps(json.loads(criteria) if isinstance(criteria, str) else criteria)
            own_objects, own_users = entity_ids(domain, text + " " + actions)
            own = shingles(text)
            overlap = max((len(own & other) / len(own | other) for other in texts if own | other), default=0.0)
            reasons = {
                "shared_object": sorted(own_objects & objects),
                "shared_user": sorted(own_users & users),
                "text_overlap": round(overlap, 3) if overlap >= TEXT_OVERLAP else None,
            }
            if any(reasons.values()):
                excluded.append({"id": task["id"], "domain": domain, "db": task["db_path"], **reasons})
            else:
                eligible[domain].append(task["id"])
    return {
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "file": TASKS_FILE, "sha256": TASKS_SHA256},
        "rule": "exclude a training task naming an evaluation task's reservation/order or user, "
        f"or with word 5-gram Jaccard >= {TEXT_OVERLAP} to an evaluation task's instructions",
        "counts": {
            domain: {"eligible": len(eligible[domain]), "excluded": sum(item["domain"] == domain for item in excluded)}
            for domain in DOMAINS
        },
        "eligible": eligible,
        "excluded": excluded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--areal-tasks", type=Path, required=True, help=f"{TASKS_FILE} at {DATASET_REVISION}")
    parser.add_argument("--tau2-domains", type=Path, required=True, help="tau2-bench data/tau2/domains at the eval pin")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation = {domain: json.loads((args.tau2_domains / domain / "tasks.json").read_text()) for domain in DOMAINS}
    report = audit(load_training_tasks(args.areal_tasks), evaluation)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
