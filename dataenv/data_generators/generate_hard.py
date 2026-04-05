"""Hard task data generator."""

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
            "max_steps": 20,
        },
    }


def generate(seed: int) -> Dict[str, Any]:
    """Generate the hard join-repair dataset."""

    rng = np.random.default_rng(seed)
    customer_count = 150
    order_count = 300
    tiers = ["bronze", "silver", "gold", "platinum"]
    statuses = ["processing", "shipped", "delivered", "returned"]
    products = ["laptop", "chair", "monitor", "desk", "phone", "printer"]

    customer_numbers = np.arange(1, customer_count + 1, dtype=np.int64)
    customers_df = pd.DataFrame(
        {
            "customer_id": [f"CUST_{value:03d}" for value in customer_numbers],
            "name": [f"Customer {value:03d}" for value in customer_numbers],
            "email": [f"customer{value:03d}@example.com" for value in customer_numbers],
            "created_at": pd.to_datetime("2022-01-01") + pd.to_timedelta(customer_numbers * 2, unit="D"),
            "tier": rng.choice(tiers, size=customer_count, p=[0.35, 0.3, 0.25, 0.1]),
        }
    )

    raw_customer_refs = rng.choice(customer_numbers, size=order_count, replace=True)
    format_selector = rng.choice(["cust", "plain", "c_short"], size=order_count, p=[0.4, 0.35, 0.25])
    customer_ref = []
    for customer_num, fmt in zip(raw_customer_refs, format_selector):
        if fmt == "cust":
            customer_ref.append(f"CUST_{customer_num:03d}")
        elif fmt == "plain":
            customer_ref.append(str(customer_num))
        else:
            customer_ref.append(f"C{customer_num:03d}")

    amounts = rng.uniform(30.0, 2500.0, size=order_count).round(2)
    negative_idx = rng.choice(order_count, size=30, replace=False)
    amounts[negative_idx] = -np.abs(amounts[negative_idx])

    orders_df = pd.DataFrame(
        {
            "order_id": np.arange(1, order_count + 1, dtype=np.int64),
            "customer_ref": customer_ref,
            "amount": amounts,
            "created_at": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(order_count), unit="h"),
            "status": rng.choice(statuses, size=order_count),
            "product": rng.choice(products, size=order_count),
        }
    )

    payload = {
        "orders": orders_df.copy(deep=True),
        "customers": customers_df.copy(deep=True),
        "original_orders": orders_df.copy(deep=True),
        "original_customers": customers_df.copy(deep=True),
        "ground_truth": {
            "expected_negative_count": 30,
            "expected_join_rows": 270,
            "expected_join_match_rate": 1.0,
            "expected_customer_pattern": r"^CUST_\d{3}$",
        },
    }
    payload.update(_base_payload())
    return payload
