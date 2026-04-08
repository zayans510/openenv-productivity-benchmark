from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from env.environment import ProductivityEnvironment

app = FastAPI(title="openenv-productivity", version="1.0.0")
_ENV = ProductivityEnvironment(max_steps=5)


class StepRequest(BaseModel):
    action: str


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "openenv-productivity",
        "message": "Server is running. Use /health for health checks.",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(task: str = "easy") -> dict:
    return _ENV.reset(task_name=task).model_dump()


@app.get("/state")
def state() -> dict:
    return _ENV.state().model_dump()


@app.post("/step")
def step(payload: StepRequest) -> dict:
    obs, reward, done, info = _ENV.step(payload.action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/{path:path}")
def fallback(path: str) -> dict[str, str]:
    return {
        "status": "ok",
        "service": "openenv-productivity",
        "path": f"/{path}",
        "message": "Server is running.",
    }


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
