from __future__ import annotations

from typing import Dict

from env import MedicalTriageEnv
from graders import run_grader
from models import DispositionDecision, ESILevel, ResourceEstimate, TriageAction
from tasks import TASKS


def _vitals_critical(obs: Dict) -> bool:
    return (
        obs["systolic_bp"] < 90
        or obs["heart_rate"] > 130
        or obs["heart_rate"] < 40
        or obs["oxygen_saturation"] < 90
        or obs["respiratory_rate"] < 8
        or obs["respiratory_rate"] > 30
        or obs["glasgow_coma_scale"] < 8
    )


def _vitals_abnormal(obs: Dict) -> bool:
    return (
        obs["systolic_bp"] < 100
        or obs["heart_rate"] > 110
        or obs["oxygen_saturation"] < 94
        or obs["respiratory_rate"] > 24
        or obs["temperature_celsius"] > 38.5
        or obs["glasgow_coma_scale"] < 14
    )


def clinical_heuristic_triage(obs: Dict) -> TriageAction:
    complaint = obs["chief_complaint"].lower()
    symptoms = [s.lower() for s in obs.get("primary_symptoms", [])]
    red_flags = [s.lower() for s in obs.get("red_flag_symptoms", [])]
    pain = obs["pain_scale"]
    gcs = obs["glasgow_coma_scale"]
    symptom_text = " ".join(symptoms + red_flags)

    esi1_triggers = {
        "apnea",
        "not breathing",
        "unresponsive",
        "pulseless",
        "cardiac arrest",
        "respiratory arrest",
        "status epilepticus",
        "no pulse",
    }
    if (
        any(trigger in complaint for trigger in esi1_triggers)
        or gcs <= 6
        or obs["oxygen_saturation"] < 80
        or obs["systolic_bp"] < 70
        or any(trigger in symptom_text for trigger in ("apnea", "cardiac_arrest", "respiratory_arrest"))
    ):
        return TriageAction(
            esi_level=ESILevel.IMMEDIATE,
            disposition=DispositionDecision.RESUSCITATION_BAY,
            estimated_resources=ResourceEstimate.FOUR_PLUS,
            requires_immediate_physician=True,
            requires_monitoring=True,
            suspected_diagnosis="Life threat requiring immediate intervention",
        )

    esi2_triggers = {
        "chest pain",
        "stroke",
        "facial droop",
        "weakness",
        "anaphylaxis",
        "throat tightening",
        "seizure",
        "sepsis",
        "hypoglycemia",
        "subarachnoid",
        "thunderclap headache",
        "overdose",
    }
    if (
        any(trigger in complaint for trigger in esi2_triggers)
        or _vitals_critical(obs)
        or (pain >= 8 and bool(red_flags))
        or gcs <= 12
        or obs["oxygen_saturation"] < 92
        or any(trigger in symptom_text for trigger in ("anaphylaxis", "hypotension", "altered"))
    ):
        return TriageAction(
            esi_level=ESILevel.EMERGENT,
            disposition=DispositionDecision.IMMEDIATE_ROOM,
            estimated_resources=ResourceEstimate.FOUR_PLUS,
            requires_immediate_physician=True,
            requires_monitoring=True,
            suspected_diagnosis="High acuity complaint requiring prompt evaluation",
        )

    esi3_triggers = {
        "abdominal pain",
        "back pain",
        "shortness of breath",
        "dyspnea",
        "fever",
        "pneumonia",
        "kidney stone",
        "appendicitis",
        "cellulitis",
        "fracture",
        "dizziness",
        "vomiting",
        "urinary",
    }
    if (
        any(trigger in complaint for trigger in esi3_triggers)
        or _vitals_abnormal(obs)
        or (pain >= 6 and obs["symptom_duration_hours"] < 48)
        or obs["age"] >= 65
        or len(obs.get("relevant_history", [])) >= 2
    ):
        return TriageAction(
            esi_level=ESILevel.URGENT,
            disposition=DispositionDecision.TREATMENT_ROOM,
            estimated_resources=ResourceEstimate.THREE,
            requires_immediate_physician=False,
            requires_monitoring=obs["age"] >= 65 or _vitals_abnormal(obs),
            suspected_diagnosis="Acute complaint requiring workup",
        )

    esi4_triggers = {"sprain", "laceration", "cut", "ear pain", "sore throat", "rash", "twisted", "wound"}
    if any(trigger in complaint for trigger in esi4_triggers) or pain <= 5:
        return TriageAction(
            esi_level=ESILevel.LESS_URGENT,
            disposition=DispositionDecision.FAST_TRACK,
            estimated_resources=ResourceEstimate.ONE,
            requires_immediate_physician=False,
            requires_monitoring=False,
            suspected_diagnosis="Minor complaint with one expected resource",
        )

    return TriageAction(
        esi_level=ESILevel.NON_URGENT,
        disposition=DispositionDecision.WAITING_ROOM,
        estimated_resources=ResourceEstimate.NONE,
        requires_immediate_physician=False,
        requires_monitoring=False,
        suspected_diagnosis="Non-urgent complaint",
    )


def run_baseline_agent(task_id: str, seed: int = 42) -> float:
    env = MedicalTriageEnv(task_id=task_id, seed=seed)
    observation = env.reset()
    episode_log = []
    done = False

    while not done:
        action = clinical_heuristic_triage(observation.model_dump(mode="json"))
        observation, reward, done, _ = env.step(action)
        episode_log.append(
            {
                "action": {
                    "esi_level": action.esi_level.value,
                    "disposition": action.disposition.value,
                    "estimated_resources": action.estimated_resources.value,
                    "requires_immediate_physician": action.requires_immediate_physician,
                    "requires_monitoring": action.requires_monitoring,
                    "suspected_diagnosis": action.suspected_diagnosis,
                },
                "ground_truth": env.get_episode_log()[-1]["ground_truth"],
                "reward": reward.total,
            }
        )

    return run_grader(task_id, episode_log)


if __name__ == "__main__":
    print("\nMedicalTriageEnv - Baseline Agent Evaluation\n")
    print("-" * 50)
    for task_id in TASKS:
        score = run_baseline_agent(task_id, seed=42)
        threshold = TASKS[task_id]["success_threshold"]
        status = "PASS" if score >= threshold else "INFO"
        print(f"{status} {task_id}: score={score:.4f} threshold={threshold:.2f}")
    print("-" * 50)
    print("Baseline complete. No LLM API key required.\n")
