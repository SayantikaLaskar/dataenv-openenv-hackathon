"""Shared helpers for task modules."""

from __future__ import annotations

import re
from typing import Dict, Tuple

import pandas as pd


def update_action_metrics(
    data: Dict,
    *,
    previous_rows: int,
    current_rows: int,
    introduced_null_fraction: float = 0.0,
    wrong_strategy: bool = False,
    notes: str = "",
) -> None:
    """Track action-level metrics used by reward shaping."""

    row_drop_ratio = 0.0
    if previous_rows > 0 and current_rows < previous_rows:
        row_drop_ratio = (previous_rows - current_rows) / previous_rows

    episode_metrics = data.setdefault("episode_metrics", {})
    if previous_rows != current_rows:
        episode_metrics["zero_data_loss"] = False

    data["last_action_metrics"] = {
        "previous_rows": previous_rows,
        "current_rows": current_rows,
        "row_drop_ratio": row_drop_ratio,
        "introduced_null_fraction": introduced_null_fraction,
        "wrong_strategy": wrong_strategy,
        "notes": notes,
    }


def parse_table_column(raw_column: str | None) -> Tuple[str | None, str | None]:
    """Parse an optional table-qualified column name."""

    if raw_column is None:
        return None, None
    if "." not in raw_column:
        return None, raw_column
    table_name, column_name = raw_column.split(".", 1)
    return table_name.strip().lower(), column_name.strip()


def normalize_customer_ref(value: object) -> object:
    """Normalize mixed customer reference formats to CUST_XXX."""

    if pd.isna(value):
        return value
    match = re.search(r"(\d+)", str(value).strip().upper())
    if not match:
        return value
    return f"CUST_{int(match.group(1)):03d}"

