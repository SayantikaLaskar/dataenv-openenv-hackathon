"""
tasks.py — Three ESI triage tasks with increasing difficulty.

Task 1 (Easy):   Binary critical vs non-critical — identify ESI-1/2 from ESI-4/5
Task 2 (Medium): Full 5-level ESI classification with balanced case mix
Task 3 (Hard):   Expert triage — ESI-2/3 boundary cases (hardest real-world decisions)
"""

TASKS = {
    "task_1_critical_detection": {
        "id": "task_1_critical_detection",
        "name": "Critical Patient Detection",
        "description": (
            "Identify patients requiring immediate intervention (ESI-1 or ESI-2) "
            "from a stream of patients that also includes stable low-acuity cases. "
            "Focus on vital signs, red flag symptoms, and chief complaint urgency signals. "
            "Missing a critical patient is the most dangerous error in emergency medicine."
        ),
        "difficulty": "easy",
        "max_steps": 15,
        "success_threshold": 0.75,
        "action_schema": {
            "esi_level": "integer enum [1=Immediate, 2=Emergent, 3=Urgent, 4=LessUrgent, 5=NonUrgent]",
            "disposition": "enum [resuscitation_bay, immediate_room, treatment_room, fast_track, waiting_room]",
            "estimated_resources": "integer enum [0=None, 1=One, 2=Two, 3=Three, 4=FourPlus]",
            "requires_immediate_physician": "boolean",
            "requires_monitoring": "boolean",
            "suspected_diagnosis": "string (max 50 chars)",
        },
        "scoring_metric": "Sensitivity for ESI-1/2 detection (critical safety metric)",
        "clinical_context": (
            "ESI undertriage of critical patients is the #1 preventable ER death cause. "
            "This task trains agents to never miss a life threat."
        ),
    },

    "task_2_full_esi_classification": {
        "id": "task_2_full_esi_classification",
        "name": "Full ESI Classification",
        "description": (
            "Assign the correct ESI level (1-5) to each patient based on the full "
            "Emergency Severity Index algorithm. Cases span the complete severity spectrum. "
            "Use vital signs, chief complaint, symptom duration, and patient history to "
            "decide: (1) Does this patient need immediate life-saving intervention? "
            "(2) Is the patient high-risk? (3) How many resources will they need?"
        ),
        "difficulty": "medium",
        "max_steps": 20,
        "success_threshold": 0.70,
        "action_schema": {
            "esi_level": "integer enum [1-5]",
            "disposition": "enum [resuscitation_bay, immediate_room, treatment_room, fast_track, waiting_room]",
            "estimated_resources": "integer enum [0-4]",
            "requires_immediate_physician": "boolean",
            "requires_monitoring": "boolean",
            "suspected_diagnosis": "string",
        },
        "scoring_metric": "Weighted accuracy with partial credit for adjacent levels + undertriage penalty",
        "clinical_context": (
            "The standard ESI algorithm is used by 70%+ of US emergency departments. "
            "Accurate 5-level triage ensures patients are seen in the right order "
            "and allocated the right level of resources."
        ),
    },

    "task_3_boundary_cases": {
        "id": "task_3_boundary_cases",
        "name": "Expert Triage: ESI Boundary Cases",
        "description": (
            "Handle the hardest real-world triage decisions — cases that fall on the "
            "ESI-2/3 boundary (high-risk but stable) and ESI-3/4 boundary (complex vs simple). "
            "These are the cases where experienced triage nurses disagree, and where "
            "both under-triage and over-triage have meaningful consequences. "
            "Presentations are intentionally vague or atypical (silent MI, elderly with sepsis, "
            "pediatric cases). Frontier LLMs score ~0.68 on this task."
        ),
        "difficulty": "hard",
        "max_steps": 25,
        "success_threshold": 0.60,
        "action_schema": {
            "esi_level": "integer enum [1-5]",
            "disposition": "enum [resuscitation_bay, immediate_room, treatment_room, fast_track, waiting_room]",
            "estimated_resources": "integer enum [0-4]",
            "requires_immediate_physician": "boolean",
            "requires_monitoring": "boolean",
            "suspected_diagnosis": "string",
        },
        "scoring_metric": "Expert-calibrated weighted accuracy with asymmetric undertriage penalty",
        "clinical_context": (
            "Inter-rater agreement among experienced ER nurses on boundary cases is ~73%. "
            "This task benchmarks agent performance against that human baseline."
        ),
    },
}