from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class ProductivityClient:
    """Minimal OpenEnv client wrapper for remote action generation."""

    def __init__(self) -> None:
        api_base_url = os.getenv("API_BASE_URL")
        model_name = os.getenv("MODEL_NAME")
        token = os.getenv("HF_TOKEN")

        if not api_base_url:
            raise ValueError("missing API_BASE_URL")
        if not model_name:
            raise ValueError("missing MODEL_NAME")
        if not token:
            raise ValueError("missing HF_TOKEN")

        self.model_name = model_name
        self.client = OpenAI(base_url=api_base_url, api_key=token)

    def act(self, observation: Any) -> str:
        observation_json = json.dumps(observation, sort_keys=True, separators=(",", ":"))
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Reply with exactly one line and no explanation. "
                        "Allowed formats: inspect, propose:{...}, final:{...}."
                    ),
                },
                {"role": "user", "content": observation_json},
            ],
        )
        content = response.choices[0].message.content or ""
        action = content.splitlines()[0].strip() if content else ""
        return _compact(action) if action else "inspect"
