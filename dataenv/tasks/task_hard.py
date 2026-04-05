"""Task 3: join repair."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from dataenv.models import DataAction
from dataenv.tasks.common import normalize_customer_ref, parse_table_column, update_action_metrics

TASK_ID = "join_repair"
DIFFICULTY = "hard"
MAX_STEPS = 20
SUCCESS_THRESHOLD = 0.70
DESCRIPTION = (
    "Two tables — orders and customers — need to be joined, but the pipeline "
    "is broken. The 'customer_ref' column in orders has mixed ID formats that "
    "don't match the clean 'CUST_XXX' format in customers. Additionally, ~30 "
    "orders have corrupted negative amounts that must be filtered, and both "
    "tables have a 'created_at' column causing a name clash. Fix all issues "
    "then submit to trigger the join evaluation."
)
ISSUE_LABELS = [
    "Normalize customer_ref join key",
    "Filter negative amounts",
    "Resolve created_at column clash",
]


def get_initial_issues(data: Dict) -> List[str]:
    """Return the canonical issue list for the task."""

    return ISSUE_LABELS.copy()


def _join_match_rate(orders: pd.DataFrame, customers: pd.DataFrame) -> float:
    customer_ids = set(customers["customer_id"].astype(str))
    if len(orders) == 0:
        return 0.0
    matches = orders["customer_ref"].astype(str).isin(customer_ids).sum()
    return float(matches / len(orders))


def detect_issues(data: Dict) -> List[str]:
    """Detect join repair issues."""

    orders = data["orders"]
    customers = data["customers"]
    issues: List[str] = []
    normalized_mask = orders["customer_ref"].astype(str).str.match(r"^CUST_\d{3}$", na=False)
    if normalized_mask.mean() < 0.9:
        issues.append("Column 'customer_ref' has mixed formats: CUST_XXX, plain int, C-format")
    negative_count = int((orders["amount"] < 0).sum())
    if negative_count:
        issues.append(f"Column 'amount' has {negative_count} negative values (corrupted data)")
    if "created_at" in orders.columns and "created_at" in customers.columns:
        issues.append("Both orders and customers tables have 'created_at' column (name clash)")
    match_rate = round(_join_match_rate(orders, customers) * 100, 1)
    issues.append(f"Join key match rate currently: {match_rate}% (need normalization)")
    return issues


def apply_action(data: Dict, action: DataAction) -> str:
    """Apply an action for the hard task."""

    orders = data["orders"]
    customers = data["customers"]
    previous_rows = len(orders)

    if action.action_type == "submit":
        data.setdefault("episode_metrics", {})["submitted"] = True
        update_action_metrics(data, previous_rows=previous_rows, current_rows=len(orders), notes="Submitted tables.")
        return "Submission requested."

    if action.action_type == "fix_join_key":
        column = action.column or "customer_ref"
        if column != "customer_ref":
            raise ValueError("fix_join_key currently supports only the 'customer_ref' column")
        data["orders"]["customer_ref"] = data["orders"]["customer_ref"].map(normalize_customer_ref)
        update_action_metrics(
            data,
            previous_rows=previous_rows,
            current_rows=len(data["orders"]),
            notes="Normalized mixed customer_ref formats to CUST_XXX.",
        )
        return "Normalized customer_ref values."

    if action.action_type == "filter_rows":
        if not action.condition:
            raise ValueError("filter_rows requires a condition")
        data["orders"] = orders.query(f"not ({action.condition})").reset_index(drop=True)
        update_action_metrics(
            data,
            previous_rows=previous_rows,
            current_rows=len(data["orders"]),
            notes=f"Filtered rows matching '{action.condition}'.",
        )
        return f"Filtered rows using condition '{action.condition}'."

    if action.action_type in {"rename_column", "drop_column"}:
        table_name, column_name = parse_table_column(action.column)
        if column_name is None:
            raise ValueError(f"{action.action_type} requires a column")

        target_df = None
        target_label = None
        if table_name == "orders":
            target_df = data["orders"]
            target_label = "orders"
        elif table_name == "customers":
            target_df = data["customers"]
            target_label = "customers"
        elif column_name in data["orders"].columns and column_name not in data["customers"].columns:
            target_df = data["orders"]
            target_label = "orders"
        elif column_name in data["customers"].columns and column_name not in data["orders"].columns:
            target_df = data["customers"]
            target_label = "customers"
        else:
            target_df = data["customers"]
            target_label = "customers"

        if action.action_type == "rename_column":
            if not action.new_name:
                raise ValueError("rename_column requires new_name")
            if column_name not in target_df.columns:
                raise ValueError(f"Column '{column_name}' does not exist in {target_label}")
            renamed_df = target_df.rename(columns={column_name: action.new_name})
            data[target_label] = renamed_df
            update_action_metrics(
                data,
                previous_rows=previous_rows,
                current_rows=len(data["orders"]),
                notes=f"Renamed {target_label}.{column_name} to {action.new_name}.",
            )
            return f"Renamed {target_label}.{column_name} to {action.new_name}."

        if column_name not in target_df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in {target_label}")
        data[target_label] = target_df.drop(columns=[column_name])
        update_action_metrics(
            data,
            previous_rows=previous_rows,
            current_rows=len(data["orders"]),
            notes=f"Dropped {target_label}.{column_name}.",
        )
        return f"Dropped {target_label}.{column_name}."

    raise ValueError("join_repair supports fix_join_key, filter_rows, rename_column, drop_column, and submit")
