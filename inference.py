from __future__ import annotations

import argparse
import json
import os
import re
from typing import Optional, Tuple

from openai import OpenAI

from env.environment import ProductivityEnvironment


MAX_STEPS = 5


def _compact(text: Optional[str]) -> str:
    if text is None:
        return "null"
    return re.sub(r"\s+", " ", str(text)).strip() or "null"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _print_start(task_name: str, env_name: str, model_name: str) -> None:
    print(f"[START] task={task_name} env={env_name} model={model_name}")


def _print_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={_compact(action)} reward={reward:.2f} "
        f"done={_bool_text(done)} error={_compact(error)}"
    )


def _print_end(success: bool, steps: int, rewards: list[float]) -> None:
    reward_text = ",".join(f"{value:.2f}" for value in rewards)
    print(f"[END] success={_bool_text(success)} steps={steps} rewards={reward_text}")


def _build_client() -> Tuple[Optional[OpenAI], Optional[str], Optional[str]]:
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    token = os.getenv("HF_TOKEN")

    if not api_base_url:
        return None, model_name, "missing API_BASE_URL"
    if not model_name:
        return None, None, "missing MODEL_NAME"
    if not token:
        return None, model_name, "missing HF_TOKEN"

    try:
        client = OpenAI(base_url=api_base_url, api_key=token)
    except Exception as exc:
        return None, model_name, f"client_initialization_failed:{_compact(exc)}"
    return client, model_name, None


def _extract_action(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text.splitlines()[0].strip() if text else ""


def _query_model(client: OpenAI, model_name: str, observation_json: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are solving a deterministic RL benchmark. "
                        "Reply with exactly one line and no explanation. "
                        "Allowed formats are inspect, propose:{...}, or final:{...}. "
                        "Use compact JSON. Prefer final:{...} once confident."
                    ),
                },
                {"role": "user", "content": observation_json},
            ],
        )
    except Exception as exc:
        return None, f"api_error:{_compact(exc)}"

    try:
        content = response.choices[0].message.content
    except Exception as exc:
        return None, f"malformed_response:{_compact(exc)}"

    if not content or not str(content).strip():
        return None, "empty_response"

    action = _extract_action(str(content))
    if not action:
        return None, "empty_action"
    return action, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=os.getenv("TASK_NAME", "easy"))
    args = parser.parse_args()

    env = ProductivityEnvironment(max_steps=MAX_STEPS)
    task_name = args.task
    model_name_for_log = os.getenv("MODEL_NAME") or "unknown"
    rewards: list[float] = []
    success = False
    steps_taken = 0

    _print_start(task_name, env.benchmark_name, model_name_for_log)

    try:
        observation = env.reset(task_name=task_name)
    except Exception as exc:
        _print_step(1, "inspect", 0.00, True, f"reset_failed:{_compact(exc)}")
        _print_end(False, 1, [0.00])
        return

    client, model_name, init_error = _build_client()
    if init_error is not None or client is None or model_name is None:
        rewards.append(0.00)
        _print_step(1, "inspect", 0.00, True, init_error)
        _print_end(False, 1, rewards)
        return

    done = False
    last_error: Optional[str] = None

    for step_number in range(1, MAX_STEPS + 1):
        steps_taken = step_number
        action, model_error = _query_model(
            client,
            model_name,
            json.dumps(observation.model_dump(), separators=(",", ":"), sort_keys=True),
        )

        if model_error is not None or action is None:
            rewards.append(0.00)
            _print_step(step_number, "inspect", 0.00, True, model_error)
            done = True
            last_error = model_error
            break

        try:
            observation, reward, done, info = env.step(action)
            error = info.get("error")
            rewards.append(reward.value)
            _print_step(step_number, action, reward.value, done, error)
            last_error = error
        except Exception as exc:
            rewards.append(0.00)
            _print_step(step_number, action, 0.00, True, f"step_failed:{_compact(exc)}")
            done = True
            last_error = str(exc)
            break

        if done:
            break

    if rewards:
        success = done and env.state().best_score >= 1.0 and last_error in (None, "")
    _print_end(success, max(steps_taken, 1), rewards if rewards else [0.00])


if __name__ == "__main__":
    main()
