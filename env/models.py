from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    raw: str = Field(..., min_length=1, max_length=4000)

    @field_validator("raw")
    @classmethod
    def validate_action_prefix(cls, value: str) -> str:
        allowed_prefixes = ("inspect", "propose:", "final:")
        if not value.startswith(allowed_prefixes):
            raise ValueError(
                "action must start with 'inspect', 'propose:', or 'final:'"
            )
        return value


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(..., ge=-1.0, le=1.0)
    score: float = Field(..., ge=0.0, le=1.0)
    delta: float = Field(..., ge=-1.0, le=1.0)
    step_penalty: float = Field(..., ge=-1.0, le=0.0)
    wrong_answer_penalty: float = Field(..., ge=-1.0, le=0.0)
    loop_penalty: float = Field(..., ge=-1.0, le=0.0)
    malformed_penalty: float = Field(..., ge=-1.0, le=0.0)
    explanation: str = Field(..., min_length=1, max_length=500)


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark: str
    task_name: Literal["easy", "medium", "hard"]
    instruction: str
    payload: Dict[str, Any]
    action_format: List[str]
    step_count: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=1)
    best_score: float = Field(..., ge=0.0, le=1.0)
    last_action: Optional[str] = None
    last_feedback: Optional[str] = None
    done: bool = False


class StepInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parsed_action: Optional[Dict[str, Any]] = None
    current_score: float = Field(..., ge=0.0, le=1.0)
    best_score: float = Field(..., ge=0.0, le=1.0)
    terminated_by: Optional[str] = None
    error: Optional[str] = None
