from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional, Tuple

from env.models import Action, Observation, Reward, StepInfo
from env.tasks import get_task, task_names


class ProductivityEnvironment:
    benchmark_name = "openenv-productivity"

    def __init__(self, max_steps: int = 5) -> None:
        self._default_max_steps = max_steps
        self._task_name = "easy"
        self._task = get_task(self._task_name)
        self._step_count = 0
        self._done = False
        self._best_score = 0.0
        self._last_action: Optional[str] = None
        self._prior_action: Optional[str] = None
        self._last_feedback: Optional[str] = None
        self._last_reward = Reward(
            value=0.0,
            score=0.0,
            delta=0.0,
            step_penalty=0.0,
            wrong_answer_penalty=0.0,
            loop_penalty=0.0,
            malformed_penalty=0.0,
            explanation="Environment initialized.",
        )

    @property
    def max_steps(self) -> int:
        return min(self._default_max_steps, self._task.max_steps)

    def available_tasks(self) -> list[str]:
        return task_names()

    def reset(self, task_name: str = "easy") -> Observation:
        self._task_name = task_name
        self._task = get_task(task_name)
        self._step_count = 0
        self._done = False
        self._best_score = 0.0
        self._last_action = None
        self._prior_action = None
        self._last_feedback = "Environment reset."
        self._last_reward = Reward(
            value=0.0,
            score=0.0,
            delta=0.0,
            step_penalty=0.0,
            wrong_answer_penalty=0.0,
            loop_penalty=0.0,
            malformed_penalty=0.0,
            explanation="Environment reset.",
        )
        return self.state()

    def state(self) -> Observation:
        return Observation(
            benchmark=self.benchmark_name,
            task_name=self._task.name,
            instruction=self._task.instruction,
            payload={
                "data": self._task.public_payload(),
                "schema": self._task.public_schema(),
            },
            action_format=[
                "inspect",
                'propose:{"field":"value"}',
                'final:{"field":"value"}',
            ],
            step_count=self._step_count,
            max_steps=self.max_steps,
            best_score=self._best_score,
            last_action=self._last_action,
            last_feedback=self._last_feedback,
            done=self._done,
        )

    def step(self, action: Any) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._done:
            info = StepInfo(
                current_score=self._best_score,
                best_score=self._best_score,
                terminated_by="already_done",
                error="step called after completion",
            )
            reward = Reward(
                value=-0.1,
                score=self._best_score,
                delta=0.0,
                step_penalty=-0.1,
                wrong_answer_penalty=0.0,
                loop_penalty=0.0,
                malformed_penalty=0.0,
                explanation="Step rejected because the episode is already done.",
            )
            self._last_reward = reward
            self._last_feedback = reward.explanation
            return self.state(), reward, True, info.model_dump()

        parsed_action, action_error = self._coerce_action(action)
        self._step_count += 1

        if action_error is not None or parsed_action is None:
            reward = self._build_reward(
                current_score=self._best_score,
                previous_best=self._best_score,
                malformed_penalty=-0.25,
                explanation=f"Malformed action: {action_error}",
            )
            self._last_feedback = reward.explanation
            self._maybe_finish()
            info = StepInfo(
                current_score=self._best_score,
                best_score=self._best_score,
                terminated_by="max_steps" if self._done else None,
                error=action_error,
            )
            return self.state(), reward, self._done, info.model_dump()

        if parsed_action.raw == "inspect":
            self._prior_action = self._last_action
            self._last_action = parsed_action.raw
            reward = self._build_reward(
                current_score=self._best_score,
                previous_best=self._best_score,
                explanation="Inspection used. No new answer submitted.",
            )
            self._last_feedback = reward.explanation
            self._maybe_finish()
            info = StepInfo(
                parsed_action={"type": "inspect"},
                current_score=self._best_score,
                best_score=self._best_score,
                terminated_by="max_steps" if self._done else None,
            )
            return self.state(), reward, self._done, info.model_dump()

        action_type, candidate = self._parse_payload_action(parsed_action.raw)
        self._prior_action = self._last_action
        self._last_action = parsed_action.raw

        if candidate is None:
            reward = self._build_reward(
                current_score=self._best_score,
                previous_best=self._best_score,
                malformed_penalty=-0.25,
                explanation="Malformed JSON payload in action.",
            )
            self._last_feedback = reward.explanation
            self._maybe_finish()
            info = StepInfo(
                parsed_action={"type": action_type},
                current_score=self._best_score,
                best_score=self._best_score,
                terminated_by="max_steps" if self._done else None,
                error="invalid_json_payload",
            )
            return self.state(), reward, self._done, info.model_dump()

        current_score, components = self._task.grade_submission(candidate)
        previous_best = self._best_score
        if current_score > self._best_score:
            self._best_score = current_score

        wrong_answer_penalty = 0.0
        if current_score < previous_best:
            wrong_answer_penalty = -0.15
        elif current_score == 0.0:
            wrong_answer_penalty = -0.05

        explanation = (
            f"Submitted {action_type} with score {current_score:.2f}. "
            f"Components: {json.dumps(components, sort_keys=True)}."
        )
        reward = self._build_reward(
            current_score=current_score,
            previous_best=previous_best,
            wrong_answer_penalty=wrong_answer_penalty,
            explanation=explanation,
        )
        self._last_feedback = explanation

        terminated_by = None
        if action_type == "final":
            self._done = True
            terminated_by = "final_action"
        elif self._best_score >= 1.0:
            self._done = True
            terminated_by = "perfect_score"
        else:
            self._maybe_finish()
            if self._done:
                terminated_by = "max_steps"

        info = StepInfo(
            parsed_action={"type": action_type, "candidate": candidate, "components": components},
            current_score=current_score,
            best_score=self._best_score,
            terminated_by=terminated_by,
        )
        return self.state(), reward, self._done, info.model_dump()

    def _coerce_action(self, action: Any) -> Tuple[Optional[Action], Optional[str]]:
        try:
            if isinstance(action, Action):
                return action, None
            if isinstance(action, dict) and "raw" in action:
                return Action.model_validate(action), None
            return Action(raw=str(action)), None
        except Exception as exc:
            return None, str(exc)

    def _parse_payload_action(self, raw: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        if raw.startswith("propose:"):
            action_type = "propose"
            payload_text = raw[len("propose:") :]
        else:
            action_type = "final"
            payload_text = raw[len("final:") :]

        try:
            parsed = json.loads(payload_text)
        except json.JSONDecodeError:
            return action_type, None

        if not isinstance(parsed, dict):
            return action_type, None
        return action_type, parsed

    def _build_reward(
        self,
        current_score: float,
        previous_best: float,
        explanation: str,
        wrong_answer_penalty: float = 0.0,
        malformed_penalty: float = 0.0,
    ) -> Reward:
        delta = max(current_score - previous_best, 0.0)
        step_penalty = -0.02
        loop_penalty = -0.05 if self._last_action is not None and self._last_action == self._prior_action else 0.0
        value = delta + step_penalty + wrong_answer_penalty + loop_penalty + malformed_penalty
        value = float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        reward = Reward(
            value=max(-1.0, min(1.0, value)),
            score=current_score,
            delta=float(Decimal(str(delta)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
            step_penalty=step_penalty,
            wrong_answer_penalty=wrong_answer_penalty,
            loop_penalty=loop_penalty,
            malformed_penalty=malformed_penalty,
            explanation=explanation,
        )
        self._last_reward = reward
        return reward

    def _maybe_finish(self) -> None:
        if self._step_count >= self.max_steps:
            self._done = True
