from __future__ import annotations

from typing import Any, Dict

from models import DispositionDecision, TriageAction, TriageReward

ESI_EXACT_SCORE = 0.45
DISPOSITION_SCORE = 0.30
RESOURCE_SCORE = 0.15
SAFETY_SCORE = 0.10

ESI_DISPOSITION_MAP = {
    1: DispositionDecision.RESUSCITATION_BAY,
    2: DispositionDecision.IMMEDIATE_ROOM,
    3: DispositionDecision.TREATMENT_ROOM,
    4: DispositionDecision.FAST_TRACK,
    5: DispositionDecision.WAITING_ROOM,
}


def esi_level_score(predicted: int, actual: int) -> float:
    diff = abs(predicted - actual)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.5
    if diff == 2:
        return 0.1
    return 0.0


def disposition_score(predicted: DispositionDecision, actual_esi: int) -> float:
    expected = ESI_DISPOSITION_MAP[actual_esi]
    if predicted == expected:
        return 1.0

    disposition_order = [
        DispositionDecision.RESUSCITATION_BAY,
        DispositionDecision.IMMEDIATE_ROOM,
        DispositionDecision.TREATMENT_ROOM,
        DispositionDecision.FAST_TRACK,
        DispositionDecision.WAITING_ROOM,
    ]
    diff = abs(disposition_order.index(predicted) - disposition_order.index(expected))
    return 0.4 if diff == 1 else 0.0


def resource_score(predicted: int, actual: int) -> float:
    diff = abs(predicted - actual)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.5
    if diff == 2:
        return 0.2
    return 0.0


def compute_reward(action: TriageAction, ground_truth: Dict[str, Any]) -> TriageReward:
    gt_esi = ground_truth["esi_level"]
    gt_resources = ground_truth["estimated_resources"]
    pred_esi = action.esi_level.value
    pred_resources = action.estimated_resources.value

    e_score = esi_level_score(pred_esi, gt_esi) * ESI_EXACT_SCORE
    d_score = disposition_score(action.disposition, gt_esi) * DISPOSITION_SCORE
    r_score = resource_score(pred_resources, gt_resources) * RESOURCE_SCORE

    if gt_esi <= 2:
        safety = SAFETY_SCORE if action.requires_immediate_physician else 0.0
    else:
        safety = SAFETY_SCORE

    penalty = 0.0
    if gt_esi == 1 and pred_esi >= 3:
        penalty = 0.40
        clinical_note = "Critical undertriage: ESI-1 patient sent away from resuscitation care."
    elif gt_esi == 1 and pred_esi == 2:
        penalty = 0.15
        clinical_note = "Undertriage: ESI-1 patient should go to the resuscitation bay."
    elif gt_esi == 2 and pred_esi >= 4:
        penalty = 0.25
        clinical_note = "Undertriage: ESI-2 patient sent to a low-priority area."
    elif gt_esi == 2 and pred_esi == 3:
        penalty = 0.05
        clinical_note = "Minor undertriage: ESI-2 patient assigned ESI-3."
    elif pred_esi < gt_esi:
        penalty = 0.05 * (gt_esi - pred_esi)
        clinical_note = f"Overtriage: assigned ESI-{pred_esi}, actual ESI-{gt_esi}."
    else:
        diff = abs(pred_esi - gt_esi)
        clinical_note = f"Assigned ESI-{pred_esi}, actual ESI-{gt_esi}, difference {diff}."
        if diff == 0:
            clinical_note = f"Correct triage: ESI-{gt_esi}."

    total = max(0.0, min(1.0, e_score + d_score + r_score + safety - penalty))
    return TriageReward(
        total=round(total, 4),
        esi_accuracy=round(e_score, 4),
        disposition_accuracy=round(d_score, 4),
        resource_accuracy=round(r_score, 4),
        safety_score=round(safety, 4),
        penalty=round(penalty, 4),
        clinical_notes=clinical_note,
    )
