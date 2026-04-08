from __future__ import annotations

import os

from fastapi import FastAPI
import uvicorn


app = FastAPI(title="openenv-productivity", version="1.0.0")


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
