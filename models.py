from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ESILevel(int, Enum):
    IMMEDIATE = 1
    EMERGENT = 2
    URGENT = 3
    LESS_URGENT = 4
    NON_URGENT = 5


class DispositionDecision(str, Enum):
    RESUSCITATION_BAY = "resuscitation_bay"
    IMMEDIATE_ROOM = "immediate_room"
    TREATMENT_ROOM = "treatment_room"
    FAST_TRACK = "fast_track"
    WAITING_ROOM = "waiting_room"


class ResourceEstimate(int, Enum):
    NONE = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR_PLUS = 4


class PatientObservation(BaseModel):
    patient_id: str = Field(..., description="Anonymized patient identifier")
    chief_complaint: str = Field(..., description="Primary reason for visit")
    symptom_duration_hours: float = Field(..., description="How long symptoms have been present")
    pain_scale: int = Field(..., ge=0, le=10, description="Self-reported pain 0-10")
    onset: str = Field(..., description="sudden | gradual | worsening | improving")
    systolic_bp: int = Field(..., description="Systolic blood pressure mmHg")
    diastolic_bp: int = Field(..., description="Diastolic blood pressure mmHg")
    heart_rate: int = Field(..., description="Heart rate beats per minute")
    respiratory_rate: int = Field(..., description="Respiratory rate breaths per minute")
    oxygen_saturation: float = Field(..., description="SpO2 percentage")
    temperature_celsius: float = Field(..., description="Body temperature celsius")
    glasgow_coma_scale: int = Field(..., ge=3, le=15, description="GCS score")
    age: int = Field(..., description="Patient age in years")
    is_pregnant: bool = Field(False, description="Known or suspected pregnancy")
    allergies: List[str] = Field(default_factory=list, description="Known allergies")
    current_medications: List[str] = Field(default_factory=list, description="Active medications")
    relevant_history: List[str] = Field(default_factory=list, description="Relevant past medical history")
    primary_symptoms: List[str] = Field(..., description="Structured symptom list")
    red_flag_symptoms: List[str] = Field(default_factory=list, description="High-acuity warning symptoms present")
    arrival_mode: str = Field(..., description="walk-in | ambulance | helicopter | police | wheelchair")
    accompanied_by: str = Field(..., description="alone | family | paramedics | police")
    step_number: int = Field(..., ge=0, description="Current step in episode")
    patients_seen: int = Field(..., ge=0, description="Total patients triaged so far")
    task_id: str = Field(..., description="Active task identifier")
    ward_capacity_pct: float = Field(..., ge=0.0, le=1.0, description="Current ER capacity 0.0-1.0")


class TriageAction(BaseModel):
    esi_level: ESILevel = Field(..., description="Assigned ESI triage level 1-5")
    disposition: DispositionDecision = Field(..., description="Where to send the patient")
    estimated_resources: ResourceEstimate = Field(..., description="Estimated hospital resources needed")
    requires_immediate_physician: bool = Field(..., description="Flag patient for immediate doctor review")
    requires_monitoring: bool = Field(..., description="Patient needs continuous vital sign monitoring")
    suspected_diagnosis: str = Field(..., description="Primary suspected diagnosis")


class TriageReward(BaseModel):
    total: float = Field(..., ge=0.0, le=1.0, description="Total reward 0.0-1.0")
    esi_accuracy: float = Field(..., description="ESI level accuracy component")
    disposition_accuracy: float = Field(..., description="Disposition accuracy")
    resource_accuracy: float = Field(..., description="Resource estimate accuracy")
    safety_score: float = Field(..., description="Safety component")
    penalty: float = Field(0.0, description="Critical error penalty")
    clinical_notes: str = Field(..., description="Human-readable explanation of scoring")


class ERState(BaseModel):
    episode_id: str
    task_id: str
    step: int
    max_steps: int
    patients_triaged: int
    correct_triages: int
    near_miss_triages: int
    undertriage_count: int
    overtriage_count: int
    critical_errors: int
    cumulative_reward: float
    er_capacity_pct: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
