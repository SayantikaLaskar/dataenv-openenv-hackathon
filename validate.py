from __future__ import annotations

import sys

import httpx

BASE_URL = "http://localhost:7860"


def check(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
    return passed


def main() -> int:
    print("\nMedicalTriageEnv Pre-Submission Validator\n")
    all_passed = True

    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=5)
        payload = response.json()
        all_passed &= check("GET /health returns 200", response.status_code == 200)
        all_passed &= check("Health body has status=ok", payload.get("status") == "ok")
    except Exception as exc:
        all_passed &= check("Health endpoint reachable", False, str(exc))

    session_key = None
    try:
        response = httpx.post(f"{BASE_URL}/reset?task_id=task_1_critical_detection&seed=42", timeout=5)
        payload = response.json()
        observation = payload.get("observation", {})
        session_key = payload.get("session_key")
        all_passed &= check("POST /reset returns observation", "observation" in payload)
        all_passed &= check("Observation has patient_id", "patient_id" in observation)
        all_passed &= check("Observation has chief_complaint", "chief_complaint" in observation)
        all_passed &= check("reset returns session_key", bool(session_key))
    except Exception as exc:
        all_passed &= check("reset works", False, str(exc))

    try:
        action = {
            "esi_level": 3,
            "disposition": "treatment_room",
            "estimated_resources": 2,
            "requires_immediate_physician": False,
            "requires_monitoring": False,
            "suspected_diagnosis": "Abdominal pain rule-out",
        }
        suffix = f"?session_key={session_key}" if session_key else ""
        response = httpx.post(f"{BASE_URL}/step{suffix}", json=action, timeout=5)
        payload = response.json()
        reward = payload.get("reward", {})
        total = reward.get("total", -1)
        all_passed &= check("POST /step returns reward", "reward" in payload)
        all_passed &= check("Reward total in [0,1]", 0.0 <= total <= 1.0, f"got {total}")
        all_passed &= check("step returns next observation", "observation" in payload)
        all_passed &= check("step returns done flag", "done" in payload)
    except Exception as exc:
        all_passed &= check("step works", False, str(exc))

    try:
        suffix = f"?session_key={session_key}" if session_key else "?task_id=task_1_critical_detection"
        response = httpx.get(f"{BASE_URL}/state{suffix}", timeout=5)
        payload = response.json()
        all_passed &= check("GET /state returns episode_id", "episode_id" in payload)
        all_passed &= check("state returns cumulative_reward", "cumulative_reward" in payload)
    except Exception as exc:
        all_passed &= check("state works", False, str(exc))

    try:
        response = httpx.get(f"{BASE_URL}/tasks", timeout=5)
        payload = response.json()
        tasks = payload.get("tasks", [])
        all_passed &= check("GET /tasks returns 3+ tasks", len(tasks) >= 3, f"got {len(tasks)}")
    except Exception as exc:
        all_passed &= check("tasks works", False, str(exc))

    try:
        dummy_log = [
            {
                "action": {
                    "esi_level": 2,
                    "disposition": "immediate_room",
                    "estimated_resources": 3,
                    "requires_immediate_physician": True,
                    "requires_monitoring": True,
                    "suspected_diagnosis": "test",
                },
                "ground_truth": {
                    "esi_level": 2,
                    "disposition": "immediate_room",
                    "estimated_resources": 3,
                    "requires_immediate_physician": True,
                    "requires_monitoring": True,
                    "diagnosis": "test",
                },
                "reward": 0.9,
            }
        ]
        response = httpx.post(f"{BASE_URL}/grader?task_id=task_1_critical_detection", json=dummy_log, timeout=5)
        payload = response.json()
        score = payload.get("score", -1)
        all_passed &= check("POST /grader returns score in [0,1]", 0.0 <= score <= 1.0, f"{score:.4f}")
    except Exception as exc:
        all_passed &= check("grader works", False, str(exc))

    try:
        response = httpx.post(f"{BASE_URL}/baseline", timeout=30)
        payload = response.json()
        scores = payload.get("scores", {})
        all_passed &= check("POST /baseline returns 3 task scores", len(scores) >= 3)
        for task_id, result in scores.items():
            score = result.get("score", -1)
            all_passed &= check(f"{task_id} baseline score in [0,1]", 0.0 <= score <= 1.0, f"{score:.4f}")
    except Exception as exc:
        all_passed &= check("baseline works", False, str(exc))

    print("\nALL CHECKS PASSED\n" if all_passed else "\nSOME CHECKS FAILED\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
