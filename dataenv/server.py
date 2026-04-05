"""FastAPI server exposing the DataEnv environment."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse

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


@app.get("/web")
def web() -> RedirectResponse:
    """Compatibility route for Hugging Face web probes."""

    return RedirectResponse(url="/", status_code=307)


@app.get("/docs-ui", response_class=HTMLResponse)
def docs_ui() -> str:
    """Lightweight human-facing page for browser visits."""

    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>DataEnv</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 2rem; line-height: 1.5; }
          code { background: #f4f4f4; padding: 0.15rem 0.3rem; border-radius: 4px; }
        </style>
      </head>
      <body>
        <h1>DataEnv</h1>
        <p>Data pipeline debugging environment for OpenEnv.</p>
        <p>Available endpoints: <code>/health</code>, <code>/reset</code>, <code>/step</code>, <code>/state</code>, <code>/tasks</code>.</p>
      </body>
    </html>
    """


@app.get("/health")
def health() -> dict:
    """Liveness endpoint."""

    return {"status": "ok"}


@app.post("/reset")
def reset(task_id: str | None = Query(default=None)) -> dict:
    """Reset the active environment."""

    global env
    if task_id is not None:
        env = DataEnv(task_id=task_id)
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
