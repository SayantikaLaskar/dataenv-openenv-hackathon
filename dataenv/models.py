"""Pydantic models for DataEnv."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DataAction(BaseModel):
    """Action the agent takes on the data pipeline."""

    action_type: Literal[
        "fix_schema",
        "fill_missing",
        "drop_duplicates",
        "rename_column",
        "drop_column",
        "fix_join_key",
        "filter_rows",
        "submit",
    ]
    column: Optional[str] = None
    target_dtype: Optional[str] = None
    fill_strategy: Optional[str] = None
    fill_value: Optional[Any] = None
    new_name: Optional[str] = None
    condition: Optional[str] = None
    join_key_left: Optional[str] = None
    join_key_right: Optional[str] = None
    reasoning: Optional[str] = None


class DataObservation(BaseModel):
    """What the agent observes at each step."""

    task_id: str
    task_description: str
    step: int
    max_steps: int
    columns: List[str]
    dtypes: Dict[str, str]
    shape: List[int]
    null_counts: Dict[str, int]
    duplicate_rows: int
    sample_rows: List[Dict[str, Any]]
    detected_issues: List[str]
    last_action_result: Optional[str] = None
    last_action_error: Optional[str] = None
    done: bool = False
    score_so_far: float = 0.0


class DataReward(BaseModel):
    """Reward signal returned after each step."""

    reward: float = Field(ge=0.0, le=1.0)
    partial_scores: Dict[str, float]
    feedback: str
    done: bool
    success: bool


class EpisodeState(BaseModel):
    """Full episode state returned by state() endpoint."""

    task_id: str
    step: int
    current_observation: DataObservation
    actions_taken: List[DataAction]
    cumulative_reward: float
    issues_resolved: List[str]
    issues_remaining: List[str]
    done: bool

