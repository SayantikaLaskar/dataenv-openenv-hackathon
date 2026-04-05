"""FastAPI server exposing the DataEnv environment."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from dataenv.env import DataEnv
from dataenv.models import DataAction

app = FastAPI(title="DataEnv", version="1.0.0", description="Data Pipeline RL Environment")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = DataEnv()


@app.get("/")
def root() -> dict:
    """Return root metadata."""

    return {
        "env": "dataenv",
        "version": "1.0.0",
        "status": "running",
        "tasks": ["schema_fix", "clean_pipeline", "join_repair"],
    }


@app.get("/health")
def health() -> dict:
    """Liveness endpoint."""

    return {"status": "ok"}


@app.post("/reset")
def reset() -> dict:
    """Reset the active environment."""

    return env.reset().model_dump()


@app.post("/step")
def step(action: DataAction) -> dict:
    """Step the environment with a structured action."""

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict:
    """Return the full environment state."""

    return env.state().model_dump()


@app.get("/tasks")
def list_tasks() -> dict:
    """List available tasks."""

    return {
        "tasks": [
            {"id": "schema_fix", "difficulty": "easy", "max_steps": 10},
            {"id": "clean_pipeline", "difficulty": "medium", "max_steps": 15},
            {"id": "join_repair", "difficulty": "hard", "max_steps": 20},
        ]
    }
