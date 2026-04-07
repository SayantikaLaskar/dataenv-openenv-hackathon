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
    assert 0.0 < reward.reward < 1.0
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


def test_completed_episodes_keep_final_score_strictly_below_one() -> None:
    cases = {
        "schema_fix": [
            DataAction(action_type="fix_schema", column="age", target_dtype="int64"),
            DataAction(action_type="fix_schema", column="salary", target_dtype="float64"),
            DataAction(action_type="fix_schema", column="hire_date", target_dtype="datetime64"),
            DataAction(action_type="fix_schema", column="is_active", target_dtype="bool"),
            DataAction(action_type="submit"),
        ],
        "clean_pipeline": [
            DataAction(action_type="drop_duplicates"),
            DataAction(action_type="fill_missing", column="customer_name", fill_strategy="drop"),
            DataAction(action_type="fill_missing", column="amount", fill_strategy="median"),
            DataAction(action_type="fill_missing", column="category", fill_strategy="mode"),
            DataAction(action_type="fill_missing", column="timestamp", fill_strategy="ffill"),
            DataAction(action_type="submit"),
        ],
        "join_repair": [
            DataAction(action_type="fix_join_key", column="customer_ref"),
            DataAction(action_type="filter_rows", condition="amount < 0"),
            DataAction(action_type="rename_column", column="customers.created_at", new_name="customer_created_at"),
            DataAction(action_type="submit"),
        ],
    }

    for task_id, actions in cases.items():
        env = DataEnv(task_id=task_id, seed=42)
        env.reset()
        reward = None
        for action in actions:
            _, reward, done, _ = env.step(action)
        assert reward is not None
        assert done is True
        assert 0.0 < reward.reward < 1.0
