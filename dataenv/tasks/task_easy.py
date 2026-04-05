"""Task 1: schema fix."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_float_dtype, is_integer_dtype

from dataenv.models import DataAction
from dataenv.tasks.common import update_action_metrics

TASK_ID = "schema_fix"
DIFFICULTY = "easy"
MAX_STEPS = 10
SUCCESS_THRESHOLD = 0.80
DESCRIPTION = (
    "A CSV dataset was loaded without dtype inference. Four columns have wrong "
    "types: 'age' (should be int64), 'salary' (should be float64, has '$' prefix), "
    "'hire_date' (should be datetime64), 'is_active' (should be bool, stored as "
    "Yes/No). Fix all schema issues without dropping more than 5% of rows."
)
ISSUE_LABELS = [
    "Fix age dtype",
    "Fix salary dtype",
    "Fix hire_date dtype",
    "Fix is_active dtype",
]


def get_initial_issues(data: Dict) -> List[str]:
    """Return the canonical issue list for the task."""

    return ISSUE_LABELS.copy()


def detect_issues(data: Dict) -> List[str]:
    """Detect schema issues in the current dataset."""

    df = data["df"]
    issues: List[str] = []
    if not is_integer_dtype(df["age"]):
        issues.append("Column 'age' has dtype object but should be int64")
    if not is_float_dtype(df["salary"]):
        issues.append("Column 'salary' has dtype object but should be float64 (contains '$' prefix)")
    if not is_datetime64_any_dtype(df["hire_date"]):
        issues.append("Column 'hire_date' has dtype object but should be datetime64")
    if not is_bool_dtype(df["is_active"]):
        issues.append("Column 'is_active' has dtype object but should be bool (contains Yes/No strings)")
    return issues


def _fix_age(series: pd.Series) -> tuple[pd.Series, float]:
    parsed = pd.to_numeric(series.astype(str).str.strip(), errors="coerce")
    null_fraction = float(parsed.isna().mean())
    fill_value = int(round(parsed.dropna().median())) if parsed.notna().any() else 30
    return parsed.fillna(fill_value).round().astype("int64"), null_fraction


def _fix_salary(series: pd.Series) -> tuple[pd.Series, float]:
    cleaned = series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
    parsed = pd.to_numeric(cleaned, errors="coerce")
    null_fraction = float(parsed.isna().mean())
    fill_value = float(parsed.dropna().median()) if parsed.notna().any() else 0.0
    return parsed.fillna(fill_value).astype("float64"), null_fraction


def _fix_hire_date(series: pd.Series) -> tuple[pd.Series, float]:
    parsed = pd.to_datetime(series, errors="coerce")
    null_fraction = float(parsed.isna().mean())
    fill_value = parsed.dropna().mode().iloc[0] if parsed.notna().any() else pd.Timestamp("2020-01-01")
    return parsed.fillna(fill_value), null_fraction


def _fix_is_active(series: pd.Series) -> tuple[pd.Series, float]:
    mapping = {"YES": True, "NO": False, "TRUE": True, "FALSE": False, "1": True, "0": False}
    normalized = series.astype(str).str.strip().str.upper().map(mapping).astype("boolean")
    null_fraction = float(normalized.isna().mean())
    mode = bool(normalized.dropna().mode().iloc[0]) if normalized.notna().any() else True
    return normalized.fillna(mode).astype(bool), null_fraction


def apply_action(data: Dict, action: DataAction) -> str:
    """Apply an action for the easy task."""

    if action.action_type == "submit":
        data.setdefault("episode_metrics", {})["submitted"] = True
        update_action_metrics(
            data,
            previous_rows=len(data["df"]),
            current_rows=len(data["df"]),
            notes="Submitted current dataframe for grading.",
        )
        return "Submission requested."

    if action.action_type != "fix_schema":
        raise ValueError("schema_fix only supports 'fix_schema' and 'submit' actions")
    if action.column is None or action.target_dtype is None:
        raise ValueError("fix_schema requires both 'column' and 'target_dtype'")

    df = data["df"]
    previous_rows = len(df)
    column = action.column
    target = action.target_dtype
    introduced_null_fraction = 0.0

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist")

    if column == "age" and target == "int64":
        df[column], introduced_null_fraction = _fix_age(df[column])
    elif column == "salary" and target == "float64":
        df[column], introduced_null_fraction = _fix_salary(df[column])
    elif column == "hire_date" and target == "datetime64":
        df[column], introduced_null_fraction = _fix_hire_date(df[column])
    elif column == "is_active" and target == "bool":
        df[column], introduced_null_fraction = _fix_is_active(df[column])
    else:
        raise ValueError(f"Unsupported schema fix for column '{column}' -> '{target}'")

    update_action_metrics(
        data,
        previous_rows=previous_rows,
        current_rows=len(df),
        introduced_null_fraction=introduced_null_fraction,
        notes=f"Normalized '{column}' to {target}.",
    )
    return f"Column '{column}' converted toward {target}."
