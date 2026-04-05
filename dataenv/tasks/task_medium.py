"""Task 2: clean pipeline."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from dataenv.models import DataAction
from dataenv.tasks.common import update_action_metrics

TASK_ID = "clean_pipeline"
DIFFICULTY = "medium"
MAX_STEPS = 15
SUCCESS_THRESHOLD = 0.75
DESCRIPTION = (
    "A customer transaction dataset suffers from two bugs: a double-ingestion "
    "event created ~80 duplicate rows, and four columns have missing values "
    "requiring different fill strategies. Remove duplicates (by transaction_id) "
    "and handle nulls: fill 'amount' with median, fill 'category' with mode, "
    "drop rows where 'customer_name' is null, and forward-fill 'timestamp'."
)
ISSUE_LABELS = [
    "Remove duplicate transaction rows",
    "Fill amount nulls with median",
    "Fill category nulls with mode",
    "Drop rows with null customer_name",
    "Forward-fill timestamp nulls",
]


def get_initial_issues(data: Dict) -> List[str]:
    """Return the canonical issue list for the task."""

    return ISSUE_LABELS.copy()


def detect_issues(data: Dict) -> List[str]:
    """Detect data quality issues in the current dataset."""

    df = data["df"]
    issues: List[str] = []
    duplicate_count = int(df.duplicated(subset=["transaction_id"]).sum())
    if duplicate_count:
        issues.append(f"{duplicate_count} duplicate rows detected (same transaction_id)")
    for column in ["amount", "category", "customer_name", "timestamp"]:
        null_count = int(df[column].isna().sum())
        if null_count:
            pct = round((null_count / max(len(df), 1)) * 100, 1)
            issues.append(f"Column '{column}' has {null_count} null values ({pct}%)")
    return issues


def apply_action(data: Dict, action: DataAction) -> str:
    """Apply an action for the medium task."""

    df = data["df"]
    previous_rows = len(df)
    wrong_strategy = False
    note = ""

    if action.action_type == "submit":
        data.setdefault("episode_metrics", {})["submitted"] = True
        update_action_metrics(data, previous_rows=previous_rows, current_rows=len(df), notes="Submitted dataset.")
        return "Submission requested."

    if action.action_type == "drop_duplicates":
        data["df"] = df.drop_duplicates(subset=["transaction_id"], keep="first").reset_index(drop=True)
        update_action_metrics(
            data,
            previous_rows=previous_rows,
            current_rows=len(data["df"]),
            notes="Dropped duplicate transaction_id rows.",
        )
        return "Duplicate transaction rows removed."

    if action.action_type != "fill_missing":
        raise ValueError("clean_pipeline supports only fill_missing, drop_duplicates, and submit")
    if action.column is None or action.fill_strategy is None:
        raise ValueError("fill_missing requires both 'column' and 'fill_strategy'")

    column = action.column
    strategy = action.fill_strategy
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")

    if strategy == "median":
        if not pd.api.types.is_numeric_dtype(df[column]):
            wrong_strategy = True
        fill_value = float(df[column].median()) if df[column].notna().any() else 0.0
        df[column] = df[column].fillna(fill_value)
        note = f"Filled '{column}' nulls with median."
    elif strategy == "mode":
        if pd.api.types.is_numeric_dtype(df[column]) and column != "category":
            wrong_strategy = True
        fill_value = df[column].mode(dropna=True).iloc[0] if df[column].notna().any() else "unknown"
        df[column] = df[column].fillna(fill_value)
        note = f"Filled '{column}' nulls with mode."
    elif strategy == "drop":
        data["df"] = df[df[column].notna()].reset_index(drop=True)
        note = f"Dropped rows where '{column}' is null."
    elif strategy == "ffill":
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            wrong_strategy = True
        data["df"][column] = data["df"][column].ffill()
        note = f"Forward-filled '{column}' nulls."
    else:
        raise ValueError(f"Unsupported fill strategy '{strategy}'")

    update_action_metrics(
        data,
        previous_rows=previous_rows,
        current_rows=len(data["df"]),
        wrong_strategy=wrong_strategy,
        notes=note,
    )
    return note

