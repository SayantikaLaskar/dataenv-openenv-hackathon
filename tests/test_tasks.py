"""Task action tests."""

from __future__ import annotations

from dataenv.data_generators import generate_easy, generate_hard, generate_medium
from dataenv.models import DataAction
from dataenv.tasks import task_easy, task_hard, task_medium


def test_easy_schema_fix_actions_resolve_all_detected_issues() -> None:
    data = generate_easy.generate(seed=42)
    task_easy.apply_action(data, DataAction(action_type="fix_schema", column="age", target_dtype="int64"))
    task_easy.apply_action(data, DataAction(action_type="fix_schema", column="salary", target_dtype="float64"))
    task_easy.apply_action(data, DataAction(action_type="fix_schema", column="hire_date", target_dtype="datetime64"))
    task_easy.apply_action(data, DataAction(action_type="fix_schema", column="is_active", target_dtype="bool"))
    assert task_easy.detect_issues(data) == []


def test_medium_pipeline_cleaning_removes_duplicates_and_nulls() -> None:
    data = generate_medium.generate(seed=42)
    task_medium.apply_action(data, DataAction(action_type="drop_duplicates"))
    task_medium.apply_action(data, DataAction(action_type="fill_missing", column="customer_name", fill_strategy="drop"))
    task_medium.apply_action(data, DataAction(action_type="fill_missing", column="amount", fill_strategy="median"))
    task_medium.apply_action(data, DataAction(action_type="fill_missing", column="category", fill_strategy="mode"))
    task_medium.apply_action(data, DataAction(action_type="fill_missing", column="timestamp", fill_strategy="ffill"))
    assert data["df"].duplicated(subset=["transaction_id"]).sum() == 0
    assert data["df"][["customer_name", "amount", "category", "timestamp"]].isna().sum().sum() == 0


def test_hard_join_repair_normalizes_keys_and_resolves_clash() -> None:
    data = generate_hard.generate(seed=42)
    task_hard.apply_action(data, DataAction(action_type="fix_join_key", column="customer_ref"))
    task_hard.apply_action(data, DataAction(action_type="filter_rows", condition="amount < 0"))
    task_hard.apply_action(
        data,
        DataAction(action_type="rename_column", column="customers.created_at", new_name="customer_created_at"),
    )
    assert data["orders"]["customer_ref"].astype(str).str.match(r"^CUST_\d{3}$").all()
    assert (data["orders"]["amount"] < 0).sum() == 0
    assert "created_at" in data["orders"].columns
    assert "customer_created_at" in data["customers"].columns

