---
title: DataEnv
emoji: 🔧
colorFrom: blue
colorTo: teal
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - data-engineering
  - agent
license: apache-2.0
---

# DataEnv

DataEnv is an OpenEnv-compatible reinforcement learning environment where an agent acts as a data engineering co-pilot. It receives broken data pipelines, applies structured remediation actions, and earns dense deterministic rewards for fixing schema issues, null handling bugs, duplicate ingestion, and multi-table join failures.

## Environment Overview

| Property | Value |
| --- | --- |
| Tasks | 3 |
| Difficulties | easy, medium, hard |
| Reward range | 0.0 to 1.0 |
| Action space | Structured `DataAction` |
| Observation space | Structured `DataObservation` |
| Runtime target | under 20 minutes on 2 vCPU / 8 GB |

## Action Space

`DataAction` supports:
- `fix_schema`
- `fill_missing`
- `drop_duplicates`
- `rename_column`
- `drop_column`
- `fix_join_key`
- `filter_rows`
- `submit`

Key parameters include `column`, `target_dtype`, `fill_strategy`, `fill_value`, `new_name`, `condition`, and `reasoning`.

## Observation Space

`DataObservation` exposes:
- task metadata: `task_id`, `task_description`, `step`, `max_steps`
- dataframe profile: `columns`, `dtypes`, `shape`, `null_counts`, `duplicate_rows`
- execution trace: `sample_rows`, `detected_issues`, `last_action_result`, `last_action_error`
- episode status: `done`, `score_so_far`

## Tasks

### `schema_fix`

Fix wrong dtypes in an employee dataset:
- `age` string to `int64`
- `salary` `"$"`-prefixed string to `float64`
- `hire_date` string to `datetime64`
- `is_active` `"Yes"`/`"No"` to `bool`

### `clean_pipeline`

Repair a customer transaction table:
- Remove duplicate rows created by double ingestion
- Fill `amount` with median
- Fill `category` with mode
- Drop rows where `customer_name` is null
- Forward-fill `timestamp`

### `join_repair`

Repair a broken orders/customers join:
- Normalize mixed customer IDs to `CUST_XXX`
- Filter corrupted negative order amounts
- Resolve `created_at` column clashes
- Submit to trigger deterministic join grading

## Project Layout

```text
dataenv/
├── Dockerfile
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── requirements.txt
├── validate-submission.sh
├── dataenv/
│   ├── env.py
│   ├── models.py
│   ├── server.py
│   ├── data_generators/
│   ├── graders/
│   └── tasks/
└── tests/
```

## Local Setup

```bash
pip install -r requirements.txt
pip install -e .
uvicorn dataenv.server:app --host 0.0.0.0 --port 7860
```

## Docker

```bash
docker build -t dataenv .
docker run -p 7860:7860 dataenv
```

## Inference

`inference.py` supports two modes:
- Remote model inference with `HF_TOKEN`, `OPENAI_API_KEY`, or `API_KEY`
- Local deterministic fallback heuristic when no API key is provided

Expected environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` or `OPENAI_API_KEY`
- `LOCAL_IMAGE_NAME` is accepted for compatibility with hackathon evaluator configs

```bash
python inference.py
```

The script writes only the required `[START]`, `[STEP]`, and `[END]` lines to stdout.

## Baseline

The included heuristic baseline deterministically solves all three tasks in the local environment and provides a reproducible floor even when no remote model credentials are set.

## Validation

```bash
openenv validate .
pytest tests/ -v
./validate-submission.sh http://127.0.0.1:7860 .
```

## Notes

- All graders are deterministic.
- All reward values are clamped to `[0.0, 1.0]`.
- The hard task simulates a realistic ETL failure mode where mixed-format keys silently break joins.
