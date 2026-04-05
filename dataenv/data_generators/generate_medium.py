"""Medium task data generator."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _base_payload() -> Dict[str, Any]:
    return {
        "progress_cache": {},
        "last_action_metrics": {},
        "episode_metrics": {
            "submitted": False,
            "zero_data_loss": True,
            "steps_used": 0,
            "max_steps": 15,
        },
    }


def generate(seed: int) -> Dict[str, Any]:
    """Generate the medium cleaning dataset."""

    rng = np.random.default_rng(seed)
    row_count = 500
    categories = ["electronics", "books", "home", "fashion", "grocery"]
    regions = ["north", "south", "east", "west"]
    statuses = ["pending", "complete", "cancelled"]
    first_names = [
        "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Parker", "Quinn", "Harper"
    ]
    last_names = ["Smith", "Johnson", "Brown", "Lee", "Patel", "Garcia", "Khan", "Miller"]

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(1, row_count + 1, dtype=np.int64),
            "customer_name": [
                f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(row_count)
            ],
            "amount": rng.uniform(10.0, 900.0, size=row_count).round(2),
            "category": rng.choice(categories, size=row_count),
            "timestamp": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(row_count), unit="h"),
            "product_id": rng.integers(1000, 9999, size=row_count, dtype=np.int64),
            "region": rng.choice(regions, size=row_count),
            "status": rng.choice(statuses, size=row_count, p=[0.2, 0.7, 0.1]),
        }
    )

    name_null_idx = rng.choice(row_count, size=50, replace=False)
    amount_null_idx = rng.choice(row_count, size=75, replace=False)
    category_null_idx = rng.choice(row_count, size=100, replace=False)
    timestamp_candidates = np.arange(1, row_count)
    timestamp_null_idx = rng.choice(timestamp_candidates, size=50, replace=False)

    df.loc[name_null_idx, "customer_name"] = pd.NA
    df.loc[amount_null_idx, "amount"] = np.nan
    df.loc[category_null_idx, "category"] = pd.NA
    df.loc[timestamp_null_idx, "timestamp"] = pd.NaT

    duplicate_idx = rng.choice(row_count, size=80, replace=False)
    duplicated_rows = df.iloc[duplicate_idx].copy(deep=True)
    duplicated_rows.index = np.arange(row_count, row_count + 80)
    full_df = pd.concat([df, duplicated_rows], ignore_index=True)

    payload = {
        "df": full_df.copy(deep=True),
        "original_df": full_df.copy(deep=True),
        "ground_truth": {
            "expected_duplicate_count": 80,
            "expected_null_drops": 50,
            "original_categories": categories,
            "original_median": float(df["amount"].median()),
            "expected_clean_rows": 450,
            "expected_clean_shape": [450, 8],
        },
    }
    payload.update(_base_payload())
    return payload

