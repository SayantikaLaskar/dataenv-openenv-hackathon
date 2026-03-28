from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from baseline import run_baseline_agent
from env import MedicalTriageEnv
from graders import run_grader
from models import TriageAction
from tasks import TASKS

app = FastAPI(
    title="MedicalTriageEnv",
    description="Emergency room triage environment following an OpenEnv-style API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[str, MedicalTriageEnv] = {}


def _dump(model: Any) -> Any:
    return model.model_dump(mode="json") if hasattr(model, "model_dump") else model


def _get_session(session_key: str) -> MedicalTriageEnv:
    env = _sessions.get(session_key)
    if env is None:
        raise HTTPException(status_code=400, detail=f"Unknown session_key: {session_key}")
    return env


@app.post("/reset")
async def reset(
    task_id: str = Query("task_1_critical_detection"),
    seed: int = Query(42),
) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    env = MedicalTriageEnv(task_id=task_id, seed=seed)
    observation = env.reset()
    session_key = f"{task_id}_{seed}_{env.episode_id}"
    _sessions[session_key] = env
    return {
        "observation": _dump(observation),
        "session_key": session_key,
        "task_id": task_id,
        "max_steps": TASKS[task_id]["max_steps"],
    }


@app.post("/step")
async def step(
    action: TriageAction,
    session_key: Optional[str] = Query(None),
    task_id: str = Query("task_1_critical_detection"),
    seed: int = Query(42),
) -> Dict[str, Any]:
    if session_key is not None:
        env = _get_session(session_key)
    else:
        if task_id not in TASKS:
            raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
        shared_key = f"{task_id}_{seed}_shared"
        env = _sessions.get(shared_key)
        if env is None:
            env = MedicalTriageEnv(task_id=task_id, seed=seed)
            env.reset()
            _sessions[shared_key] = env
        session_key = shared_key

    try:
        observation, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "observation": _dump(observation),
        "reward": _dump(reward),
        "done": done,
        "info": info,
        "session_key": session_key,
    }


@app.get("/state")
async def state(
    session_key: Optional[str] = Query(None),
    task_id: str = Query("task_1_critical_detection"),
    seed: int = Query(42),
) -> Dict[str, Any]:
    if session_key is not None:
        return _dump(_get_session(session_key).state())
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    shared_key = f"{task_id}_{seed}_shared"
    env = _sessions.get(shared_key)
    if env is None:
        env = MedicalTriageEnv(task_id=task_id, seed=seed)
        _sessions[shared_key] = env
    return _dump(env.state())


@app.get("/tasks")
async def get_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "id": task_id,
                "name": task["name"],
                "description": task["description"],
                "difficulty": task["difficulty"],
                "max_steps": task["max_steps"],
                "success_threshold": task["success_threshold"],
                "action_schema": task["action_schema"],
                "scoring_metric": task["scoring_metric"],
                "clinical_context": task["clinical_context"],
            }
            for task_id, task in TASKS.items()
        ]
    }


@app.post("/grader")
async def grader(
    task_id: str = Query(...),
    episode_actions: List[Dict[str, Any]] = Body(...),
) -> Dict[str, Any]:
    try:
        score = run_grader(task_id, episode_actions)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    threshold = TASKS[task_id]["success_threshold"]
    return {
        "task_id": task_id,
        "score": score,
        "n_actions": len(episode_actions),
        "threshold": threshold,
        "passed": score >= threshold,
    }


@app.post("/baseline")
async def baseline() -> Dict[str, Any]:
    results = {}
    for task_id, config in TASKS.items():
        score = run_baseline_agent(task_id, seed=42)
        results[task_id] = {
            "score": score,
            "threshold": config["success_threshold"],
            "passed": score >= config["success_threshold"],
        }
    return {
        "agent": "clinical_heuristic_v1",
        "description": "Rule-based ESI decision tree. No LLM required.",
        "scores": results,
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "env": "MedicalTriageEnv", "version": "1.0.0"}


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "name": "MedicalTriageEnv",
        "version": "1.0.0",
        "description": "Real-world emergency room triage RL environment.",
        "domain": "medical_triage",
        "real_world_task": "ER Triage Nurse",
        "protocol": "Emergency Severity Index (ESI) v4",
        "endpoints": [
            "/reset",
            "/step",
            "/state",
            "/tasks",
            "/grader",
            "/baseline",
            "/health",
        ],
        "openenv_spec": "1.0",
        "tasks": list(TASKS.keys()),
    }
