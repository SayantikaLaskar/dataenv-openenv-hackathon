from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from data_generator import generate_patient_batch
from models import ERState, PatientObservation, TriageAction, TriageReward
from reward import compute_reward
from tasks import TASKS


class MedicalTriageEnv:
    def __init__(self, task_id: str = "task_1_critical_detection", seed: int = 42):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}")

        self.task_id = task_id
        self.task_config = TASKS[task_id]
        self.seed = seed
        self.episode_id: Optional[str] = None
        self._patients: List[Dict[str, Any]] = []
        self._index = 0
        self._log: List[Dict[str, Any]] = []
        self._cumulative_reward = 0.0
        self._done = False
        self._er_capacity = 0.65
        self._reset_count = 0

    def reset(self) -> PatientObservation:
        self.episode_id = str(uuid.uuid4())
        self._index = 0
        self._log = []
        self._cumulative_reward = 0.0
        self._done = False
        self._reset_count += 1

        rng = random.Random(self.seed + self._reset_count * 4099)
        self._er_capacity = round(rng.uniform(0.5, 0.9), 2)
        episode_seed = self.seed + (self._reset_count - 1) * 9973
        self._patients = generate_patient_batch(
            task_id=self.task_id,
            n_patients=self.task_config["max_steps"],
            difficulty=self.task_config["difficulty"],
            seed=episode_seed,
        )
        return self._build_observation()

    def step(self, action: TriageAction) -> Tuple[PatientObservation, TriageReward, bool, Dict[str, Any]]:
        if self.episode_id is None:
            raise RuntimeError("Episode not started. Call reset() first.")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        current_patient = self._patients[self._index]
        reward = compute_reward(action, current_patient["ground_truth"])
        self._log.append(
            {
                "step": self._index,
                "patient_id": current_patient["patient_id"],
                "action": {
                    "esi_level": action.esi_level.value,
                    "disposition": action.disposition.value,
                    "estimated_resources": action.estimated_resources.value,
                    "requires_immediate_physician": action.requires_immediate_physician,
                    "requires_monitoring": action.requires_monitoring,
                    "suspected_diagnosis": action.suspected_diagnosis,
                },
                "ground_truth": current_patient["ground_truth"],
                "reward": reward.total,
                "clinical_notes": reward.clinical_notes,
            }
        )

        self._cumulative_reward += reward.total
        self._index += 1
        self._done = self._index >= self.task_config["max_steps"]

        info = {
            "episode_id": self.episode_id,
            "step": self._index - 1,
            "clinical_notes": reward.clinical_notes,
            "reward_breakdown": {
                "esi_accuracy": reward.esi_accuracy,
                "disposition_accuracy": reward.disposition_accuracy,
                "resource_accuracy": reward.resource_accuracy,
                "safety_score": reward.safety_score,
                "penalty": reward.penalty,
            },
        }
        next_observation = self._build_terminal_observation() if self._done else self._build_observation()
        return next_observation, reward, self._done, info

    def state(self) -> ERState:
        if self.episode_id is None:
            return ERState(
                episode_id="",
                task_id=self.task_id,
                step=0,
                max_steps=self.task_config["max_steps"],
                patients_triaged=0,
                correct_triages=0,
                near_miss_triages=0,
                undertriage_count=0,
                overtriage_count=0,
                critical_errors=0,
                cumulative_reward=0.0,
                er_capacity_pct=self._er_capacity,
                done=False,
                info={},
            )

        correct = sum(1 for entry in self._log if entry["reward"] >= 0.85)
        near_miss = sum(1 for entry in self._log if 0.50 <= entry["reward"] < 0.85)
        undertriage = sum(
            1 for entry in self._log if entry["action"]["esi_level"] > entry["ground_truth"]["esi_level"]
        )
        overtriage = sum(
            1 for entry in self._log if entry["action"]["esi_level"] < entry["ground_truth"]["esi_level"]
        )
        critical_errors = sum(
            1
            for entry in self._log
            if entry["ground_truth"]["esi_level"] <= 2
            and entry["action"]["esi_level"] > entry["ground_truth"]["esi_level"] + 1
        )

        return ERState(
            episode_id=self.episode_id,
            task_id=self.task_id,
            step=self._index,
            max_steps=self.task_config["max_steps"],
            patients_triaged=len(self._log),
            correct_triages=correct,
            near_miss_triages=near_miss,
            undertriage_count=undertriage,
            overtriage_count=overtriage,
            critical_errors=critical_errors,
            cumulative_reward=round(self._cumulative_reward, 4),
            er_capacity_pct=self._er_capacity,
            done=self._done,
            info={
                "log_entries": len(self._log),
                "avg_reward": round(self._cumulative_reward / max(1, len(self._log)), 4),
            },
        )

    def get_episode_log(self) -> List[Dict[str, Any]]:
        return list(self._log)

    def _build_observation(self) -> PatientObservation:
        patient = self._patients[self._index]
        return PatientObservation(
            patient_id=patient["patient_id"],
            chief_complaint=patient["chief_complaint"],
            symptom_duration_hours=patient["symptom_duration_hours"],
            pain_scale=patient["pain_scale"],
            onset=patient["onset"],
            systolic_bp=patient["systolic_bp"],
            diastolic_bp=patient["diastolic_bp"],
            heart_rate=patient["heart_rate"],
            respiratory_rate=patient["respiratory_rate"],
            oxygen_saturation=patient["oxygen_saturation"],
            temperature_celsius=patient["temperature_celsius"],
            glasgow_coma_scale=patient["glasgow_coma_scale"],
            age=patient["age"],
            is_pregnant=patient.get("is_pregnant", False),
            allergies=patient.get("allergies", []),
            current_medications=patient.get("current_medications", []),
            relevant_history=patient.get("relevant_history", []),
            primary_symptoms=patient["primary_symptoms"],
            red_flag_symptoms=patient["red_flag_symptoms"],
            arrival_mode=patient["arrival_mode"],
            accompanied_by=patient["accompanied_by"],
            step_number=self._index,
            patients_seen=len(self._log),
            task_id=self.task_id,
            ward_capacity_pct=self._er_capacity,
        )

    def _build_terminal_observation(self) -> PatientObservation:
        return PatientObservation(
            patient_id="TERMINAL",
            chief_complaint="[Episode Complete - call reset()]",
            symptom_duration_hours=0,
            pain_scale=0,
            onset="sudden",
            systolic_bp=120,
            diastolic_bp=80,
            heart_rate=70,
            respiratory_rate=16,
            oxygen_saturation=99.0,
            temperature_celsius=37.0,
            glasgow_coma_scale=15,
            age=0,
            is_pregnant=False,
            allergies=[],
            current_medications=[],
            relevant_history=[],
            primary_symptoms=[],
            red_flag_symptoms=[],
            arrival_mode="walk-in",
            accompanied_by="alone",
            step_number=self._index,
            patients_seen=len(self._log),
            task_id=self.task_id,
            ward_capacity_pct=self._er_capacity,
        )
