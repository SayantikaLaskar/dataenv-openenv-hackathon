"""Environment smoke tests."""

from __future__ import annotations

from dataenv.env import DataEnv
from dataenv.models import DataAction


def run_episode(task_id: str) -> None:
    """Run a one-step submit episode for a task."""

    env = DataEnv(task_id=task_id, seed=42)
    obs = env.reset()
    assert obs.task_id == task_id
    assert len(obs.columns) > 0

    obs, reward, done, info = env.step(DataAction(action_type="submit", reasoning="test submit"))
    assert 0.0 <= reward.reward <= 1.0
    assert done is True
    assert info["step"] == 1
    env.close()


def test_easy_episode() -> None:
    run_episode("schema_fix")


def test_medium_episode() -> None:
    run_episode("clean_pipeline")


def test_hard_episode() -> None:
    run_episode("join_repair")


def test_state_endpoint_shape() -> None:
    env = DataEnv(task_id="schema_fix", seed=42)
    env.reset()
    state = env.state()
    assert state.task_id == "schema_fix"
    assert state.step == 0
    assert state.current_observation.shape[1] == 6

