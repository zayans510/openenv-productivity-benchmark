from __future__ import annotations

from typing import Any, Dict

from env.tasks import GRADERS, TASK_SPECS, TaskSpec, get_task, schema_json, task_names


def easy_grader(candidate: Dict[str, Any]) -> float:
    return GRADERS["easy"](candidate)


def medium_grader(candidate: Dict[str, Any]) -> float:
    return GRADERS["medium"](candidate)


def hard_grader(candidate: Dict[str, Any]) -> float:
    return GRADERS["hard"](candidate)


# Canonical checker-friendly task registry: each entry explicitly exposes a grader.
TASKS = [
    {"name": "easy", "difficulty": "easy", "grader": easy_grader, "spec": get_task("easy")},
    {"name": "medium", "difficulty": "medium", "grader": medium_grader, "spec": get_task("medium")},
    {"name": "hard", "difficulty": "hard", "grader": hard_grader, "spec": get_task("hard")},
]

# Backward-compatible alias.
TASKS_WITH_GRADERS = TASKS


__all__ = [
    "TaskSpec",
    "get_task",
    "schema_json",
    "task_names",
    "TASK_SPECS",
    "easy_grader",
    "medium_grader",
    "hard_grader",
    "TASKS",
    "TASKS_WITH_GRADERS",
]
