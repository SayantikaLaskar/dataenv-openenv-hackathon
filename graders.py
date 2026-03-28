"""
graders.py — Deterministic episode graders for each task.
All graders return scores in [0.0, 1.0]. No LLM required.
"""

from typing import List, Dict, Any


def grade_task_1(episode_actions: List[Dict]) -> float:
    """
    Task 1: Critical Patient Detection.
    Metric: F1 score for correctly identifying ESI-1 and ESI-2 patients.
    This is a safety-critical metric — recall (sensitivity) is more important
    than precision (we'd rather over-triage than miss a critical patient).
    """
    if not episode_actions:
        return 0.0

    tp = fp = fn = tn = 0
    for entry in episode_actions:
        action = entry["action"]
        gt     = entry["ground_truth"]

        is_critical_gt   = gt["esi_level"] <= 2
        is_critical_pred = action["esi_level"] <= 2

        if is_critical_gt and is_critical_pred:
            tp += 1
        elif not is_critical_gt and is_critical_pred:
            fp += 1
        elif is_critical_gt and not is_critical_pred:
            fn += 1  # DANGEROUS: missed a critical patient
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Weighted F-score: recall weighted 2x (missing critical = worse than over-calling)
    beta = 2.0
    if (beta**2 * precision + recall) == 0:
        return 0.0
    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

    # Extra penalty for any missed ESI-1 patients
    missed_esi1 = sum(
        1 for e in episode_actions
        if e["ground_truth"]["esi_level"] == 1 and e["action"]["esi_level"] > 2
    )
    esi1_penalty = missed_esi1 * 0.1

    return round(max(0.0, min(1.0, f_beta - esi1_penalty)), 4)


def grade_task_2(episode_actions: List[Dict]) -> float:
    """
    Task 2: Full ESI Classification.
    Metric: Weighted accuracy with partial credit and asymmetric undertriage penalty.
    """
    if not episode_actions:
        return 0.0

    total_score = 0.0
    n = len(episode_actions)

    for entry in episode_actions:
        action = entry["action"]
        gt     = entry["ground_truth"]

        pred_esi = action["esi_level"]
        true_esi = gt["esi_level"]
        diff     = abs(pred_esi - true_esi)

        if diff == 0:
            total_score += 1.0
        elif diff == 1:
            total_score += 0.5
        elif diff == 2:
            total_score += 0.1
        # diff >= 3: 0 points

        # Asymmetric undertriage penalty (undertriage of high-acuity is worse)
        if true_esi <= 2 and pred_esi > true_esi:
            undertriage_severity = (pred_esi - true_esi)
            total_score -= 0.2 * undertriage_severity

    return round(max(0.0, min(1.0, total_score / n)), 4)


def grade_task_3(episode_actions: List[Dict]) -> float:
    """
    Task 3: Boundary Cases.
    Metric: Mean reward from multi-dimensional scoring (reward.py already computed).
    Also penalizes inconsistent performance (high variance = unreliable triage).
    """
    if not episode_actions:
        return 0.0

    rewards = [entry.get("reward", 0.0) for entry in episode_actions]
    mean_reward = sum(rewards) / len(rewards)

    # Penalize high variance (inconsistent triage is clinically dangerous)
    if len(rewards) > 1:
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        consistency_penalty = min(0.1, variance * 0.5)
    else:
        consistency_penalty = 0.0

    final = max(0.0, min(1.0, mean_reward - consistency_penalty))
    return round(final, 4)


GRADER_REGISTRY = {
    "task_1_critical_detection":       grade_task_1,
    "task_2_full_esi_classification":  grade_task_2,
    "task_3_boundary_cases":           grade_task_3,
}


def run_grader(task_id: str, episode_actions: List[Dict]) -> float:
    grader = GRADER_REGISTRY.get(task_id)
    if grader is None:
        raise ValueError(f"No grader for task_id '{task_id}'. Valid: {list(GRADER_REGISTRY.keys())}")
    return grader(episode_actions)