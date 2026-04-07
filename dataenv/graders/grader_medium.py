"""Deterministic grader for the clean pipeline task."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from dataenv.graders.common import clamp, clamp_strict, format_progress_feedback
from dataenv.models import DataAction, DataReward
from dataenv.tasks import task_medium


def grade_clean_pipeline(
    original_df: pd.DataFrame,
    current_df: pd.DataFrame,
    expected_duplicate_count: int,
    expected_null_drops: int,
    original_categories: list[str],
) -> dict:
    """Standalone grading function for direct evaluation."""

    original_unique = original_df.drop_duplicates(subset=["transaction_id"], keep="first")
    original_median = float(original_unique["amount"].median())
    removed_duplicates = len(original_df) - len(current_df)
    customer_name_drop_count = int(original_unique["customer_name"].isna().sum())
    expected_clean_rows = len(original_unique) - customer_name_drop_count

    scores = {
        "duplicates_removed": 0.25
        if abs((len(original_df) - len(current_df)) - expected_duplicate_count) <= 2
        else 0.0,
        "amount_filled": 0.15
        if current_df["amount"].isna().sum() == 0 and abs(float(current_df["amount"].median()) - original_median) < 0.01
        else 0.0,
        "category_filled": 0.15
        if current_df["category"].isna().sum() == 0 and current_df["category"].isin(original_categories).all()
        else 0.0,
        "customer_name_dropped": 0.15
        if abs((len(original_unique) - len(current_df[current_df["customer_name"].notna()])) - expected_null_drops) <= 2
        else 0.0,
        "timestamp_filled": 0.15 if current_df["timestamp"].isna().sum() == 0 else 0.0,
        "final_shape": 0.15 if abs(len(current_df) - expected_clean_rows) <= max(1, int(expected_clean_rows * 0.05)) else 0.0,
    }
    total = clamp_strict(sum(scores.values()))
    return {
        "reward": total,
        "partial_scores": scores,
        "feedback": "Standalone clean_pipeline grade computed.",
        "done": True,
        "success": total >= task_medium.SUCCESS_THRESHOLD,
    }


def _score_components(data: Dict) -> Dict[str, float]:
    original_df = data["original_df"]
    df = data["df"]
    ground_truth = data["ground_truth"]
    unique_original = original_df.drop_duplicates(subset=["transaction_id"], keep="first")
    expected_clean_rows = ground_truth["expected_clean_rows"]
    customer_nulls_now = int(df["customer_name"].isna().sum())
    duplicates_now = int(df.duplicated(subset=["transaction_id"]).sum())
    timestamp_nulls = int(df["timestamp"].isna().sum())
    amount_ok = df["amount"].isna().sum() == 0 and abs(float(df["amount"].median()) - ground_truth["original_median"]) < 0.01
    category_ok = df["category"].isna().sum() == 0 and df["category"].isin(ground_truth["original_categories"]).all()
    rows_with_customer_name_removed = len(unique_original) - len(df)

    return {
        "duplicates_removed": 0.25 if abs(duplicates_now - 0) <= 2 and len(original_df) - len(df) >= ground_truth["expected_duplicate_count"] - 2 else 0.0,
        "amount_filled": 0.15 if amount_ok else 0.0,
        "category_filled": 0.15 if category_ok else 0.0,
        "customer_name_dropped": 0.15 if abs(rows_with_customer_name_removed - ground_truth["expected_null_drops"]) <= 2 and customer_nulls_now == 0 else 0.0,
        "timestamp_filled": 0.15 if timestamp_nulls == 0 else 0.0,
        "final_shape": 0.15 if abs(len(df) - expected_clean_rows) <= max(1, int(expected_clean_rows * 0.05)) else 0.0,
    }


def get_resolved_issues(data: Dict) -> List[str]:
    """Return the user-facing issue labels resolved so far."""

    scores = _score_components(data)
    resolved: List[str] = []
    if scores["duplicates_removed"] > 0:
        resolved.append(task_medium.ISSUE_LABELS[0])
    if scores["amount_filled"] > 0:
        resolved.append(task_medium.ISSUE_LABELS[1])
    if scores["category_filled"] > 0:
        resolved.append(task_medium.ISSUE_LABELS[2])
    if scores["customer_name_dropped"] > 0:
        resolved.append(task_medium.ISSUE_LABELS[3])
    if scores["timestamp_filled"] > 0:
        resolved.append(task_medium.ISSUE_LABELS[4])
    return resolved


def compute_step_reward(
    data: Dict,
    action: DataAction,
    actions_taken: List[DataAction],
    issues_resolved: List[str],
) -> DataReward:
    """Compute dense reward after a cleaning step."""

    previous_scores = data.setdefault("progress_cache", {}).copy()
    current_scores = _score_components(data)
    data["progress_cache"] = current_scores.copy()
    improved = [key for key in current_scores if current_scores[key] > previous_scores.get(key, 0.0)]
    progress = sum(max(0.0, current_scores[key] - previous_scores.get(key, 0.0)) for key in current_scores)

    penalties: List[str] = []
    metrics = data.get("last_action_metrics", {})
    if metrics.get("wrong_strategy"):
        penalties.append("Fill strategy mismatched the column type.")
        progress = max(0.0, progress - 0.10)
    if metrics.get("row_drop_ratio", 0.0) > 0.30:
        penalties.append("Single action dropped >30% of rows.")
        progress = max(0.0, progress - 0.15)
    if len(actions_taken) >= 2 and actions_taken[-1].model_dump() == actions_taken[-2].model_dump():
        penalties.append("Repeated identical action.")
        progress = max(0.0, progress - 0.05)

    feedback = format_progress_feedback(improved, penalties, "No new cleaning issue resolved.")
    return DataReward(
        reward=round(clamp_strict(progress), 4),
        partial_scores={key: round(value, 4) for key, value in current_scores.items()},
        feedback=feedback,
        done=False,
        success=sum(current_scores.values()) >= task_medium.SUCCESS_THRESHOLD,
    )


def compute_final_reward(data: Dict) -> DataReward:
    """Compute the final deterministic reward."""

    scores = _score_components(data)
    total = sum(scores.values())
    resolved_ratio = len(get_resolved_issues(data)) / len(task_medium.ISSUE_LABELS)
    episode_metrics = data.get("episode_metrics", {})
    if episode_metrics.get("submitted") and resolved_ratio < 0.5:
        total *= 0.5
    if total >= task_medium.SUCCESS_THRESHOLD and episode_metrics.get("steps_used", task_medium.MAX_STEPS) < episode_metrics.get("max_steps", task_medium.MAX_STEPS):
        total += 0.10
        scores["efficiency_bonus"] = 0.10
    if episode_metrics.get("zero_data_loss", False):
        total += 0.05
        scores["zero_data_loss_bonus"] = 0.05
    total = clamp_strict(total)
    return DataReward(
        reward=round(total, 4),
        partial_scores={key: round(clamp(value), 4) for key, value in scores.items()},
        feedback="Final clean_pipeline grading completed.",
        done=True,
        success=total >= task_medium.SUCCESS_THRESHOLD,
    )
