---
title: MedicalTriageEnv
emoji: "🏥"
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# MedicalTriageEnv

A production-ready OpenEnv-style reinforcement learning environment for emergency room triage. The agent receives one patient at a time and must assign an Emergency Severity Index level, disposition, and expected resource use.

## Why this environment

Emergency triage is a real, high-stakes workflow with immediate practical value for evaluation and agent training. The environment is deterministic, uses no external APIs, and supports dense partial-credit rewards instead of binary grading.

## Tasks

| Task | Difficulty | Steps | Metric |
| --- | --- | --- | --- |
| `task_1_critical_detection` | Easy | 15 | F-beta on critical patient detection |
| `task_2_full_esi_classification` | Medium | 20 | Weighted ESI accuracy |
| `task_3_boundary_cases` | Hard | 25 | Mean reward with consistency penalty |

## API

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `POST /baseline`
- `GET /health`

## Local run

```bash
cd medi_triage
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Validation

```bash
cd medi_triage
python validate.py
```

## Hugging Face Spaces

This folder is Docker-ready for Hugging Face Spaces on port `7860`.
