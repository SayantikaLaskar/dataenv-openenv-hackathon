"""
data_generator.py — Clinically realistic synthetic patient generator.

All data is procedurally generated — NO external API keys required.
Cases are based on real ESI triage decision trees and ACEP clinical guidelines.
Each generated patient has a deterministic ground truth ESI level.
"""

import random
import hashlib
from typing import Dict, Any, List, Tuple

# ─── ESI-1 Cases: Immediate life threat ──────────────────────────────────────
ESI1_CASES = [
    {
        "chief_complaint": "Not breathing, found unresponsive by family",
        "symptom_duration_hours": 0.1,
        "pain_scale": 0,
        "onset": "sudden",
        "systolic_bp": 60, "diastolic_bp": 30,
        "heart_rate": 140, "respiratory_rate": 4,
        "oxygen_saturation": 72.0, "temperature_celsius": 36.5,
        "glasgow_coma_scale": 3,
        "age": 58,
        "primary_symptoms": ["apnea", "unresponsive", "cyanosis"],
        "red_flag_symptoms": ["apnea", "loss_of_consciousness", "cyanosis"],
        "arrival_mode": "ambulance",
        "accompanied_by": "paramedics",
        "relevant_history": ["COPD", "hypertension"],
        "ground_truth": {
            "esi_level": 1, "disposition": "resuscitation_bay",
            "estimated_resources": 4, "requires_immediate_physician": True,
            "requires_monitoring": True,
            "diagnosis": "Respiratory arrest / COPD exacerbation"
        }
    },
    {
        "chief_complaint": "Sudden severe chest pain, arm numb, sweating",
        "symptom_duration_hours": 0.5,
        "pain_scale": 10,
        "onset": "sudden",
        "systolic_bp": 88, "diastolic_bp": 55,
        "heart_rate": 118, "respiratory_rate": 24,
        "oxygen_saturation": 91.0, "temperature_celsius": 36.8,
        "glasgow_coma_scale": 14,
        "age": 62,
        "primary_symptoms": ["chest_pain", "diaphoresis", "left_arm_numbness", "nausea"],
        "red_flag_symptoms": ["chest_pain_radiating", "diaphoresis", "hypotension"],
        "arrival_mode": "ambulance",
        "accompanied_by": "paramedics",
        "relevant_history": ["diabetes", "smoker", "hypertension"],
        "ground_truth": {
            "esi_level": 1, "disposition": "resuscitation_bay",
            "estimated_resources": 4, "requires_immediate_physician": True,
            "requires_monitoring": True,
            "diagnosis": "STEMI / acute myocardial infarction"
        }
    },
    {
        "chief_complaint": "Severe headache, worst of my life, stiff neck, confused",
        "symptom_duration_hours": 1.0,
        "pain_scale": 10,
        "onset": "sudden",
        "systolic_bp": 195, "diastolic_bp": 115,
        "heart_rate": 96, "respiratory_rate": 20,
        "oxygen_saturation": 97.0, "temperature_celsius": 38.9,
        "glasgow_coma_scale": 12,
        "age": 34,
        "primary_symptoms": ["thunderclap_headache", "neck_stiffness", "photophobia", "confusion"],
        "red_flag_symptoms": ["thunderclap_headache", "meningismus", "altered_consciousness"],
        "arrival_mode": "ambulance",
        "accompanied_by": "family",
        "relevant_history": [],
        "ground_truth": {
            "esi_level": 1, "disposition": "resuscitation_bay",
            "estimated_resources": 4, "requires_immediate_physician": True,
            "requires_monitoring": True,
            "diagnosis": "Subarachnoid hemorrhage / bacterial meningitis"
        }
    },
    {
        "chief_complaint": "Seizure lasting 10 minutes, still not waking up",
        "symptom_duration_hours": 0.2,
        "pain_scale": 0,
        "onset": "sudden",
        "systolic_bp": 155, "diastolic_bp": 90,
        "heart_rate": 130, "respiratory_rate": 8,
        "oxygen_saturation": 88.0, "temperature_celsius": 37.2,
        "glasgow_coma_scale": 6,
        "age": 27,
        "primary_symptoms": ["status_epilepticus", "post_ictal_unresponsive"],
        "red_flag_symptoms": ["prolonged_seizure", "unresponsive", "hypoxia"],
        "arrival_mode": "ambulance",
        "accompanied_by": "paramedics",
        "relevant_history": ["epilepsy"],
        "ground_truth": {
            "esi_level": 1, "disposition": "resuscitation_bay",
            "estimated_resources": 4, "requires_immediate_physician": True,
            "requires_monitoring": True,
            "diagnosis": "Status epilepticus"
        }
    },
]

# ─── ESI-2 Cases: High risk, should not wait ─────────────────────────────────
ESI2_CASES = [
    {
        "chief_complaint": "Severe chest pain, 8/10, started 2 hours ago, feels like pressure",
        "symptom_duration_hours": 2.0,
        "pain_scale": 8,
        "onset": "sudden",
        "systolic_bp": 142, "diastolic_bp": 88,
        "heart_rate": 102, "respiratory_rate": 20,
        "oxygen_saturation": 96.0, "temperature_celsius": 37.1,
        "glasgow_coma_scale": 15,
        "age": 55,
        "primary_symptoms": ["chest_pain", "dyspnea", "palpitations"],
        "red_flag_symptoms": ["chest_pain_radiating"],
        "arrival_mode": "walk-in",
        "accompanied_by": "family",
        "relevant_history": ["hypertension", "high_cholesterol"],
        "ground_truth": {
            "esi_level": 2, "disposition": "immediate_room",
            "estimated_resources": 4, "requires_immediate_physician": True,
            "requires_monitoring": True,
            "diagnosis": "Unstable angina / NSTEMI rule-out"
        }
    },
    {
        "chief_complaint": "Sudden right-sided weakness, can't speak properly, face drooping",
        "symptom_duration_hours": 0.75,
        "pain_scale": 2,
        "onset": "sudden",
        "systolic_bp": 178, "diastolic_bp": 102,
        "heart_rate": 88, "respiratory_rate": 18,
        "oxygen_saturation": 98.0, "temperature_celsius": 37.0,
        "glasgow_coma_scale": 14,
        "age": 71,
        "primary_symptoms": ["facial_droop", "arm_weakness", "speech_difficulty"],
        "red_flag_symptoms": ["acute_focal_neuro_deficit", "facial_droop", "aphasia"],
        "arrival_mode": "ambulance",
        "accompanied_by": "family",
        "relevant_history": ["atrial_fibrillation", "hypertension"],
        "ground_truth": {
            "esi_level": 2, "disposition": "immediate_room",
            "estimated_resources": 4, "requires_immediate_physician": True,
            "requires_monitoring": True,
            "diagnosis": "Acute ischemic stroke (FAST positive)"
        }
    },
    {
        "chief_complaint": "Allergic reaction, throat tightening, hives all over",
        "symptom_duration_hours": 0.3,
        "pain_scale": 6,
        "onset": "sudden",
        "systolic_bp": 95, "diastolic_bp": 60,
        "heart_rate": 125, "respiratory_rate": 28,
        "oxygen_saturation": 94.0, "temperature_celsius": 37.4,
        "glasgow_coma_scale": 14,
        "age": 22,
        "primary_symptoms": ["urticaria", "throat_tightness", "stridor", "hypotension"],
        "red_flag_symptoms": ["anaphylaxis", "stridor", "hypotension"],
        "arrival_mode": "walk-in",
        "accompanied_by": "family",
        "relevant_history": ["peanut_allergy"],
        "ground_truth": {
            "esi_level": 2, "disposition": "immediate_room",
            "estimated_resources": 3, "requires_immediate_physician": True,
            "requires_monitoring": True,
            "diagnosis": "Anaphylaxis"
        }
    },
    {
        "chief_complaint": "Blood sugar 35, shaking, sweating, barely coherent",
        "symptom_duration_hours": 0.5,
        "pain_scale": 2,
        "onset": "gradual",
        "systolic_bp": 100, "diastolic_bp": 65,
        "heart_rate": 108, "respiratory_rate": 18,
        "oxygen_saturation": 99.0, "temperature_celsius": 36.9,
        "glasgow_coma_scale": 13,
        "age": 45,
        "primary_symptoms": ["diaphoresis", "tremor", "confusion", "hypoglycemia"],
        "red_flag_symptoms": ["altered_mental_status", "severe_hypoglycemia"],
        "arrival_mode": "ambulance",
        "accompanied_by": "paramedics",
        "relevant_history": ["type_1_diabetes", "insulin_dependent"],
        "ground_truth": {
            "esi_level": 2, "disposition": "immediate_room",
            "estimated_resources": 2, "requires_immediate_physician": True,
            "requires_monitoring": True,
            "diagnosis": "Severe hypoglycemia"
        }
    },
]

# ─── ESI-3 Cases: Stable but needs multiple resources ────────────────────────
ESI3_CASES = [
    {
        "chief_complaint": "Abdominal pain right lower quadrant, worse with movement, nausea",
        "symptom_duration_hours": 18.0,
        "pain_scale": 7,
        "onset": "gradual",
        "systolic_bp": 122, "diastolic_bp": 78,
        "heart_rate": 96, "respiratory_rate": 18,
        "oxygen_saturation": 99.0, "temperature_celsius": 37.9,
        "glasgow_coma_scale": 15,
        "age": 24,
        "primary_symptoms": ["RLQ_pain", "rebound_tenderness", "nausea", "anorexia"],
        "red_flag_symptoms": [],
        "arrival_mode": "walk-in",
        "accompanied_by": "family",
        "relevant_history": [],
        "ground_truth": {
            "esi_level": 3, "disposition": "treatment_room",
            "estimated_resources": 3, "requires_immediate_physician": False,
            "requires_monitoring": False,
            "diagnosis": "Acute appendicitis rule-out"
        }
    },
    {
        "chief_complaint": "Severe back pain radiating to groin, blood in urine",
        "symptom_duration_hours": 4.0,
        "pain_scale": 9,
        "onset": "sudden",
        "systolic_bp": 135, "diastolic_bp": 85,
        "heart_rate": 105, "respiratory_rate": 20,
        "oxygen_saturation": 98.0, "temperature_celsius": 37.2,
        "glasgow_coma_scale": 15,
        "age": 38,
        "primary_symptoms": ["flank_pain", "hematuria", "colicky_pain", "nausea"],
        "red_flag_symptoms": [],
        "arrival_mode": "walk-in",
        "accompanied_by": "alone",
        "relevant_history": ["prior_kidney_stones"],
        "ground_truth": {
            "esi_level": 3, "disposition": "treatment_room",
            "estimated_resources": 3, "requires_immediate_physician": False,
            "requires_monitoring": False,
            "diagnosis": "Renal colic / ureteral calculus"
        }
    },
    {
        "chief_complaint": "Shortness of breath, fever 5 days, coughing yellow sputum",
        "symptom_duration_hours": 120.0,
        "pain_scale": 4,
        "onset": "gradual",
        "systolic_bp": 118, "diastolic_bp": 74,
        "heart_rate": 104, "respiratory_rate": 24,
        "oxygen_saturation": 93.0, "temperature_celsius": 38.8,
        "glasgow_coma_scale": 15,
        "age": 67,
        "primary_symptoms": ["dyspnea", "productive_cough", "fever", "fatigue"],
        "red_flag_symptoms": ["low_oxygen_saturation"],
        "arrival_mode": "walk-in",
        "accompanied_by": "family",
        "relevant_history": ["COPD", "smoker_40_pack_years"],
        "ground_truth": {
            "esi_level": 3, "disposition": "treatment_room",
            "estimated_resources": 4, "requires_immediate_physician": False,
            "requires_monitoring": True,
            "diagnosis": "Community-acquired pneumonia / COPD exacerbation"
        }
    },
]

# ─── ESI-4 Cases: Stable, one resource ───────────────────────────────────────
ESI4_CASES = [
    {
        "chief_complaint": "Sprained ankle after fall, swollen, can barely walk",
        "symptom_duration_hours": 3.0,
        "pain_scale": 6,
        "onset": "sudden",
        "systolic_bp": 128, "diastolic_bp": 80,
        "heart_rate": 82, "respiratory_rate": 16,
        "oxygen_saturation": 99.0, "temperature_celsius": 36.9,
        "glasgow_coma_scale": 15,
        "age": 29,
        "primary_symptoms": ["ankle_swelling", "ankle_pain", "limited_mobility"],
        "red_flag_symptoms": [],
        "arrival_mode": "walk-in",
        "accompanied_by": "family",
        "relevant_history": [],
        "ground_truth": {
            "esi_level": 4, "disposition": "fast_track",
            "estimated_resources": 1, "requires_immediate_physician": False,
            "requires_monitoring": False,
            "diagnosis": "Ankle sprain / rule-out fracture (X-ray needed)"
        }
    },
    {
        "chief_complaint": "Ear pain for 2 days, pulling at ear, mild fever",
        "symptom_duration_hours": 48.0,
        "pain_scale": 5,
        "onset": "gradual",
        "systolic_bp": 100, "diastolic_bp": 65,
        "heart_rate": 95, "respiratory_rate": 22,
        "oxygen_saturation": 99.0, "temperature_celsius": 38.1,
        "glasgow_coma_scale": 15,
        "age": 4,
        "primary_symptoms": ["otalgia", "fever", "irritability"],
        "red_flag_symptoms": [],
        "arrival_mode": "walk-in",
        "accompanied_by": "family",
        "relevant_history": ["recurrent_ear_infections"],
        "ground_truth": {
            "esi_level": 4, "disposition": "fast_track",
            "estimated_resources": 1, "requires_immediate_physician": False,
            "requires_monitoring": False,
            "diagnosis": "Otitis media"
        }
    },
    {
        "chief_complaint": "Cut on hand from kitchen knife, needs stitches maybe",
        "symptom_duration_hours": 1.0,
        "pain_scale": 4,
        "onset": "sudden",
        "systolic_bp": 120, "diastolic_bp": 78,
        "heart_rate": 78, "respiratory_rate": 16,
        "oxygen_saturation": 99.0, "temperature_celsius": 36.8,
        "glasgow_coma_scale": 15,
        "age": 35,
        "primary_symptoms": ["laceration", "bleeding_controlled", "hand_pain"],
        "red_flag_symptoms": [],
        "arrival_mode": "walk-in",
        "accompanied_by": "alone",
        "relevant_history": [],
        "ground_truth": {
            "esi_level": 4, "disposition": "fast_track",
            "estimated_resources": 1, "requires_immediate_physician": False,
            "requires_monitoring": False,
            "diagnosis": "Hand laceration requiring suture repair"
        }
    },
]

# ─── ESI-5 Cases: Minor, no resources ────────────────────────────────────────
ESI5_CASES = [
    {
        "chief_complaint": "Prescription refill for blood pressure medicine, ran out 2 days ago",
        "symptom_duration_hours": 48.0,
        "pain_scale": 0,
        "onset": "gradual",
        "systolic_bp": 132, "diastolic_bp": 84,
        "heart_rate": 72, "respiratory_rate": 16,
        "oxygen_saturation": 99.0, "temperature_celsius": 36.7,
        "glasgow_coma_scale": 15,
        "age": 52,
        "primary_symptoms": ["medication_request"],
        "red_flag_symptoms": [],
        "arrival_mode": "walk-in",
        "accompanied_by": "alone",
        "relevant_history": ["hypertension_controlled"],
        "ground_truth": {
            "esi_level": 5, "disposition": "waiting_room",
            "estimated_resources": 0, "requires_immediate_physician": False,
            "requires_monitoring": False,
            "diagnosis": "Medication refill — non-urgent"
        }
    },
    {
        "chief_complaint": "Sore throat for 2 days, mild, no fever, eating fine",
        "symptom_duration_hours": 48.0,
        "pain_scale": 2,
        "onset": "gradual",
        "systolic_bp": 118, "diastolic_bp": 76,
        "heart_rate": 70, "respiratory_rate": 16,
        "oxygen_saturation": 99.0, "temperature_celsius": 37.0,
        "glasgow_coma_scale": 15,
        "age": 19,
        "primary_symptoms": ["sore_throat", "mild_pharyngitis"],
        "red_flag_symptoms": [],
        "arrival_mode": "walk-in",
        "accompanied_by": "alone",
        "relevant_history": [],
        "ground_truth": {
            "esi_level": 5, "disposition": "waiting_room",
            "estimated_resources": 0, "requires_immediate_physician": False,
            "requires_monitoring": False,
            "diagnosis": "Viral pharyngitis"
        }
    },
    {
        "chief_complaint": "Cold symptoms for 3 days, runny nose, mild cough, no fever",
        "symptom_duration_hours": 72.0,
        "pain_scale": 1,
        "onset": "gradual",
        "systolic_bp": 115, "diastolic_bp": 73,
        "heart_rate": 68, "respiratory_rate": 15,
        "oxygen_saturation": 100.0, "temperature_celsius": 36.9,
        "glasgow_coma_scale": 15,
        "age": 31,
        "primary_symptoms": ["rhinorrhea", "mild_cough", "congestion"],
        "red_flag_symptoms": [],
        "arrival_mode": "walk-in",
        "accompanied_by": "alone",
        "relevant_history": [],
        "ground_truth": {
            "esi_level": 5, "disposition": "waiting_room",
            "estimated_resources": 0, "requires_immediate_physician": False,
            "requires_monitoring": False,
            "diagnosis": "Upper respiratory infection (common cold)"
        }
    },
]

ALL_CASES = {
    1: ESI1_CASES,
    2: ESI2_CASES,
    3: ESI3_CASES,
    4: ESI4_CASES,
    5: ESI5_CASES,
}


def _stable_int(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)


def generate_patient_id(seed: int, index: int) -> str:
    raw = f"PT-{seed}-{index}"
    return "PT-" + hashlib.md5(raw.encode()).hexdigest()[:8].upper()


def _add_noise(case: Dict, rng: random.Random, difficulty: str) -> Dict:
    """
    Add realistic vital-sign noise and variation.
    Harder difficulties introduce subtler presentations.
    """
    c = dict(case)
    noise = 1 if difficulty == "easy" else (3 if difficulty == "medium" else 6)

    c["systolic_bp"]  = max(50, c["systolic_bp"] + rng.randint(-noise, noise))
    c["diastolic_bp"] = max(30, c["diastolic_bp"] + rng.randint(-noise, noise))
    c["heart_rate"]   = max(30, c["heart_rate"] + rng.randint(-noise, noise))
    c["temperature_celsius"] = round(
        c["temperature_celsius"] + rng.uniform(-0.2, 0.2), 1
    )
    c["oxygen_saturation"] = round(
        max(70.0, min(100.0, c["oxygen_saturation"] + rng.uniform(-0.5, 0.5))), 1
    )

    if difficulty == "hard":
        # In hard mode, chief complaint is more vague
        vague_prefixes = [
            "I don't feel well, ",
            "Something's wrong, ",
            "Not sure what's happening but ",
        ]
        c["chief_complaint"] = rng.choice(vague_prefixes) + c["chief_complaint"].lower()
        # Also obscure some red flag symptoms
        if c["red_flag_symptoms"] and rng.random() < 0.4:
            c["red_flag_symptoms"] = c["red_flag_symptoms"][:1]

    return c


def generate_patient_batch(
    task_id: str,
    n_patients: int,
    difficulty: str,
    seed: int,
) -> List[Dict]:
    """
    Generate a batch of patients with ground truth ESI levels.
    Difficulty affects ESI distribution and presentation clarity.
    """
    rng = random.Random(seed + _stable_int(task_id))

    if difficulty == "easy":
        # More ESI-5/4 cases (clearer decisions)
        distribution = [1, 2, 3, 4, 4, 5, 5, 5]
    elif difficulty == "medium":
        # Balanced mix
        distribution = [1, 2, 2, 3, 3, 4, 5]
    else:  # hard
        # More ESI 2/3 boundary cases (hardest triage decisions)
        distribution = [1, 2, 2, 2, 3, 3, 3, 4]

    patients = []
    for i in range(n_patients):
        esi = rng.choice(distribution)
        case_list = ALL_CASES[esi]
        base_case = rng.choice(case_list)
        case = _add_noise(base_case, rng, difficulty)

        patients.append({
            "patient_id": generate_patient_id(seed, i),
            "chief_complaint": case["chief_complaint"],
            "symptom_duration_hours": case["symptom_duration_hours"],
            "pain_scale": case["pain_scale"],
            "onset": case["onset"],
            "systolic_bp": case["systolic_bp"],
            "diastolic_bp": case["diastolic_bp"],
            "heart_rate": case["heart_rate"],
            "respiratory_rate": case["respiratory_rate"],
            "oxygen_saturation": case["oxygen_saturation"],
            "temperature_celsius": case["temperature_celsius"],
            "glasgow_coma_scale": case["glasgow_coma_scale"],
            "age": case.get("age", 40),
            "is_pregnant": False,
            "allergies": case.get("allergies", []),
            "current_medications": case.get("current_medications", []),
            "relevant_history": case.get("relevant_history", []),
            "primary_symptoms": case["primary_symptoms"],
            "red_flag_symptoms": case["red_flag_symptoms"],
            "arrival_mode": case["arrival_mode"],
            "accompanied_by": case["accompanied_by"],
            "ground_truth": case["ground_truth"],
        })

    return patients
