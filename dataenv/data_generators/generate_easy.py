"""Easy task data generator."""

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
            "max_steps": 10,
        },
    }


def generate(seed: int) -> Dict[str, Any]:
    """Generate the easy schema-fix dataset."""

    rng = np.random.default_rng(seed)
    row_count = 200
    departments = ["Engineering", "Finance", "HR", "Sales", "Marketing", "Operations"]
    employee_id = np.arange(1001, 1001 + row_count, dtype=np.int64)
    age_clean = rng.integers(21, 66, size=row_count, dtype=np.int64)
    salary_clean = rng.uniform(45_000, 160_000, size=row_count).round(2)
    hire_clean = pd.to_datetime("2017-01-01") + pd.to_timedelta(
        rng.integers(0, 2500, size=row_count), unit="D"
    )
    active_clean = rng.choice([True, False], size=row_count, p=[0.78, 0.22])

    age_broken = np.array(
        [f"{value}" if idx % 3 else f"{value} " for idx, value in enumerate(age_clean)],
        dtype=object,
    )
    salary_broken = np.array([f"${value:.2f}" for value in salary_clean], dtype=object)
    hire_broken = np.array([value.strftime("%Y-%m-%d") for value in hire_clean], dtype=object)
    active_broken = np.array(["Yes" if value else "No" for value in active_clean], dtype=object)

    invalid_count = int(row_count * 0.05)
    age_bad_idx = rng.choice(row_count, size=invalid_count, replace=False)
    salary_bad_idx = rng.choice(row_count, size=invalid_count, replace=False)
    hire_bad_idx = rng.choice(row_count, size=invalid_count, replace=False)
    active_bad_idx = rng.choice(row_count, size=invalid_count, replace=False)

    age_broken[age_bad_idx] = rng.choice(["unknown", "n/a", "forty"], size=invalid_count)
    salary_broken[salary_bad_idx] = rng.choice(["$??", "missing", "USD_nan"], size=invalid_count)
    hire_broken[hire_bad_idx] = rng.choice(["2023-13-99", "not_a_date", "0000-00-00"], size=invalid_count)
    active_broken[active_bad_idx] = rng.choice(["Maybe", "TBD", "unknown"], size=invalid_count)

    df = pd.DataFrame(
        {
            "employee_id": employee_id,
            "department": rng.choice(departments, size=row_count),
            "age": age_broken,
            "salary": salary_broken,
            "hire_date": hire_broken,
            "is_active": active_broken,
        }
    )

    payload = {
        "df": df.copy(deep=True),
        "original_df": df.copy(deep=True),
        "ground_truth": {
            "expected_dtypes": {
                "employee_id": "int64",
                "department": "object",
                "age": "int64",
                "salary": "float64",
                "hire_date": "datetime64[ns]",
                "is_active": "bool",
            }
        },
    }
    payload.update(_base_payload())
    return payload

