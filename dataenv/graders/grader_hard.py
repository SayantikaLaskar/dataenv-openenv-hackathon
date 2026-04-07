"""Deterministic grader for the join repair task."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from dataenv.graders.common import export_score, export_scores, format_progress_feedback
from dataenv.models import DataAction, DataReward
from dataenv.tasks import task_hard


def grade_join_repair(
    orders_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    current_orders: pd.DataFrame,
    current_customers: pd.DataFrame,
    submitted: bool,
) -> dict:
    """Standalone grading function for direct evaluation."""

    pattern_rate = float(current_orders["customer_ref"].astype(str).str.match(r"^CUST_\d{3}$", na=False).mean())
    clash_resolved = not ("created_at" in current_orders.columns and "created_at" in current_customers.columns)
    scores = {
        "join_key_normalized": 0.30 if pattern_rate > 0.90 else 0.15 if pattern_rate > 0.70 else 0.0,
        "negative_amounts_filtered": 0.20 if int((current_orders["amount"] < 0).sum()) == 0 else 0.0,
        "column_clash_resolved": 0.15 if clash_resolved else 0.0,
        "join_success": 0.0,
    }
    if submitted:
        merged = pd.merge(
            current_orders,
            current_customers,
            left_on="customer_ref",
            right_on="customer_id",
            how="inner",
            suffixes=("_orders", "_customers"),
        )
        match_rate = len(merged) / len(current_orders) if len(current_orders) else 0.0
        if match_rate > 0.95:
            scores["join_success"] = 0.35
        elif match_rate > 0.80:
            scores["join_success"] = 0.25
        elif match_rate > 0.50:
            scores["join_success"] = 0.10
    total = export_score(sum(scores.values()))
    return {
        "reward": total,
        "partial_scores": export_scores(scores),
        "feedback": "Standalone join_repair grade computed.",
        "done": True,
        "success": total >= task_hard.SUCCESS_THRESHOLD,
    }


def _score_components(data: Dict, *, include_join: bool) -> Dict[str, float]:
    orders = data["orders"]
    customers = data["customers"]
    pattern_rate = float(orders["customer_ref"].astype(str).str.match(r"^CUST_\d{3}$", na=False).mean())
    scores = {
        "join_key_normalized": 0.30 if pattern_rate > 0.90 else 0.15 if pattern_rate > 0.70 else 0.0,
        "negative_amounts_filtered": 0.20 if int((orders["amount"] < 0).sum()) == 0 else 0.0,
        "column_clash_resolved": 0.15
        if not ("created_at" in orders.columns and "created_at" in customers.columns)
        else 0.0,
        "join_success": 0.0,
    }
    if include_join:
        merged = pd.merge(
            orders,
            customers,
            left_on="customer_ref",
            right_on="customer_id",
            how="inner",
            suffixes=("_orders", "_customers"),
        )
        match_rate = len(merged) / len(orders) if len(orders) else 0.0
        if match_rate > 0.95:
            scores["join_success"] = 0.35
        elif match_rate > 0.80:
            scores["join_success"] = 0.25
        elif match_rate > 0.50:
            scores["join_success"] = 0.10
    return scores


def get_resolved_issues(data: Dict) -> List[str]:
    """Return the user-facing issue labels resolved so far."""

    scores = _score_components(data, include_join=False)
    resolved: List[str] = []
    if scores["join_key_normalized"] >= 0.30:
        resolved.append(task_hard.ISSUE_LABELS[0])
    if scores["negative_amounts_filtered"] > 0:
        resolved.append(task_hard.ISSUE_LABELS[1])
    if scores["column_clash_resolved"] > 0:
        resolved.append(task_hard.ISSUE_LABELS[2])
    return resolved


def compute_step_reward(
    data: Dict,
    action: DataAction,
    actions_taken: List[DataAction],
    issues_resolved: List[str],
) -> DataReward:
    """Compute dense reward after a repair step."""

    previous_scores = data.setdefault("progress_cache", {}).copy()
    current_scores = _score_components(data, include_join=False)
    data["progress_cache"] = current_scores.copy()
    improved = [key for key in current_scores if current_scores[key] > previous_scores.get(key, 0.0)]
    progress = sum(max(0.0, current_scores[key] - previous_scores.get(key, 0.0)) for key in current_scores)

    penalties: List[str] = []
    metrics = data.get("last_action_metrics", {})
    if metrics.get("row_drop_ratio", 0.0) > 0.30:
        penalties.append("Single action dropped >30% of order rows.")
        progress = max(0.0, progress - 0.15)
    if len(actions_taken) >= 2 and actions_taken[-1].model_dump() == actions_taken[-2].model_dump():
        penalties.append("Repeated identical action.")
        progress = max(0.0, progress - 0.05)

    feedback = format_progress_feedback(improved, penalties, "No new join repair issue resolved.")
    return DataReward(
        reward=export_score(progress),
        partial_scores=export_scores(current_scores),
        feedback=feedback,
        done=False,
        success=sum(current_scores.values()) >= task_hard.SUCCESS_THRESHOLD,
    )


def compute_final_reward(data: Dict) -> DataReward:
    """Compute the final deterministic reward."""

    episode_metrics = data.get("episode_metrics", {})
    scores = _score_components(data, include_join=bool(episode_metrics.get("submitted")))
    total = sum(scores.values())
    resolved_ratio = len(get_resolved_issues(data)) / len(task_hard.ISSUE_LABELS)
    if episode_metrics.get("submitted") and resolved_ratio < 0.5:
        total *= 0.5
    if total >= task_hard.SUCCESS_THRESHOLD and episode_metrics.get("steps_used", task_hard.MAX_STEPS) < episode_metrics.get("max_steps", task_hard.MAX_STEPS):
        total += 0.10
        scores["efficiency_bonus"] = 0.10
    if episode_metrics.get("zero_data_loss", False):
        total += 0.05
        scores["zero_data_loss_bonus"] = 0.05
    total = export_score(total)
    return DataReward(
        reward=total,
        partial_scores=export_scores(scores),
        feedback="Final join_repair grading completed.",
        done=True,
        success=total >= task_hard.SUCCESS_THRESHOLD,
    )
