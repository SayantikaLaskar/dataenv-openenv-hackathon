"""Deterministic grader for the schema fix task."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_float_dtype, is_integer_dtype

from dataenv.graders.common import clamp, format_progress_feedback
from dataenv.models import DataAction, DataReward
from dataenv.tasks import task_easy

PRIMARY_KEYS = ["age_dtype", "salary_dtype", "hire_date_dtype", "is_active_dtype"]


def _score_components(data: Dict) -> Dict[str, float]:
    df = data["df"]
    original_df = data["original_df"]
    scores = {
        "age_dtype": 0.20 if is_integer_dtype(df["age"]) else 0.0,
        "salary_dtype": 0.20 if is_float_dtype(df["salary"]) else 0.0,
        "hire_date_dtype": 0.20 if is_datetime64_any_dtype(df["hire_date"]) else 0.0,
        "is_active_dtype": 0.20 if is_bool_dtype(df["is_active"]) else 0.0,
        "row_loss": 0.10 if len(df) >= len(original_df) * 0.95 else 0.0,
        "non_target_preserved": 0.10
        if str(df["employee_id"].dtype) == str(original_df["employee_id"].dtype)
        and str(df["department"].dtype) == str(original_df["department"].dtype)
        else 0.0,
    }
    return scores


def get_resolved_issues(data: Dict) -> List[str]:
    """Return the user-facing issue labels resolved so far."""

    scores = _score_components(data)
    resolved: List[str] = []
    if scores["age_dtype"] > 0:
        resolved.append(task_easy.ISSUE_LABELS[0])
    if scores["salary_dtype"] > 0:
        resolved.append(task_easy.ISSUE_LABELS[1])
    if scores["hire_date_dtype"] > 0:
        resolved.append(task_easy.ISSUE_LABELS[2])
    if scores["is_active_dtype"] > 0:
        resolved.append(task_easy.ISSUE_LABELS[3])
    return resolved


def compute_step_reward(
    data: Dict,
    action: DataAction,
    actions_taken: List[DataAction],
    issues_resolved: List[str],
) -> DataReward:
    """Compute dense reward after a single action."""

    previous_scores = data.setdefault("progress_cache", {}).copy()
    current_scores = _score_components(data)
    data["progress_cache"] = current_scores.copy()

    improved = [
        key
        for key in current_scores
        if current_scores[key] > previous_scores.get(key, 0.0)
    ]
    progress = sum(
        max(0.0, current_scores[key] - previous_scores.get(key, 0.0))
        for key in current_scores
    )

    penalties: List[str] = []
    metrics = data.get("last_action_metrics", {})
    if metrics.get("introduced_null_fraction", 0.0) > 0.30:
        penalties.append("Wrong dtype cast introduced >30% null-like values.")
        progress = max(0.0, progress - 0.10)
    if len(actions_taken) >= 2 and actions_taken[-1].model_dump() == actions_taken[-2].model_dump():
        penalties.append("Repeated identical action.")
        progress = max(0.0, progress - 0.05)

    feedback = format_progress_feedback(improved, penalties, "No new schema issue resolved.")
    return DataReward(
        reward=round(clamp(progress), 4),
        partial_scores={key: round(value, 4) for key, value in current_scores.items()},
        feedback=feedback,
        done=False,
        success=sum(current_scores.values()) >= task_easy.SUCCESS_THRESHOLD,
    )


def compute_final_reward(data: Dict) -> DataReward:
    """Compute the final deterministic reward."""

    scores = _score_components(data)
    total = sum(scores.values())
    resolved_ratio = len(get_resolved_issues(data)) / len(task_easy.ISSUE_LABELS)
    episode_metrics = data.get("episode_metrics", {})
    submitted = bool(episode_metrics.get("submitted"))
    if submitted and resolved_ratio < 0.5:
        total *= 0.5
    if total >= task_easy.SUCCESS_THRESHOLD and episode_metrics.get("steps_used", task_easy.MAX_STEPS) < episode_metrics.get("max_steps", task_easy.MAX_STEPS):
        total += 0.10
        scores["efficiency_bonus"] = 0.10
    if episode_metrics.get("zero_data_loss", False):
        total += 0.05
        scores["zero_data_loss_bonus"] = 0.05
    total = clamp(total)
    return DataReward(
        reward=round(total, 4),
        partial_scores={key: round(clamp(value), 4) for key, value in scores.items()},
        feedback="Final schema grading completed.",
        done=True,
        success=total >= task_easy.SUCCESS_THRESHOLD,
    )

