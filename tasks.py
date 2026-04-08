from __future__ import annotations

from typing import Any, Dict

from env.tasks import GRADERS, TASKS, TaskSpec, get_task, schema_json, task_names


def easy_grader(candidate: Dict[str, Any]) -> float:
    return GRADERS["easy"](candidate)


def medium_grader(candidate: Dict[str, Any]) -> float:
    return GRADERS["medium"](candidate)


def hard_grader(candidate: Dict[str, Any]) -> float:
    return GRADERS["hard"](candidate)


TASKS_WITH_GRADERS = [
    {"name": "easy", "grader": easy_grader},
    {"name": "medium", "grader": medium_grader},
    {"name": "hard", "grader": hard_grader},
]


__all__ = [
    "TASKS",
    "TaskSpec",
    "get_task",
    "schema_json",
    "task_names",
    "easy_grader",
    "medium_grader",
    "hard_grader",
    "TASKS_WITH_GRADERS",
]
