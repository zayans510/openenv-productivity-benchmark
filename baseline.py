from __future__ import annotations

import json

from env.environment import ProductivityEnvironment
from env.tasks import get_task, task_names


def run_baseline() -> None:
    env = ProductivityEnvironment(max_steps=5)
    summary: dict[str, dict[str, float | int | bool]] = {}

    for task_name in task_names():
        task = get_task(task_name)
        env.reset(task_name)
        action = f"final:{json.dumps(task.expected, separators=(',', ':'), sort_keys=True)}"
        _, reward, done, info = env.step(action)
        summary[task_name] = {
            "done": bool(done),
            "score": float(info["best_score"]),
            "reward": float(reward.value),
            "steps": 1,
        }

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    run_baseline()
