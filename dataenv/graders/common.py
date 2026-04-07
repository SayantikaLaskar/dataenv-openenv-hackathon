"""Common grading helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List

from dataenv.models import DataReward

STRICT_SCORE_EPSILON = 1e-4
# Minimum and maximum values after rounding to 4 decimal places
# These ensure scores are strictly between 0 and 1 (not 0.0 and not 1.0)
MIN_SCORE = 0.0001
MAX_SCORE = 0.9999


def clamp(value: float) -> float:
    """Clamp a reward value to the OpenEnv range."""

    return max(0.0, min(1.0, float(value)))


def clamp_strict(value: float) -> float:
    """Clamp a task reward to the validator-safe open interval (0, 1)."""

    return max(MIN_SCORE, min(MAX_SCORE, float(value)))


def export_score(value: float) -> float:
    """Normalize any externally visible score into the open interval (0, 1).
    
    Ensures the score is strictly between 0 and 1 after rounding to 4 decimal places.
    This satisfies the validator requirement that scores must not be exactly 0.0 or 1.0.
    """

    clamped = clamp_strict(value)
    rounded = round(clamped, 4)
    # Double-check after rounding to ensure we never return exactly 0.0 or 1.0
    if rounded <= 0.0:
        return MIN_SCORE
    if rounded >= 1.0:
        return MAX_SCORE
    return rounded


def export_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize all externally visible score components."""

    return {key: export_score(value) for key, value in scores.items()}


def reward_from_scores(
    partial_scores: Dict[str, float],
    *,
    feedback: str,
    success_threshold: float,
    done: bool,
) -> DataReward:
    """Build a DataReward from partial scores."""

    total = export_score(sum(partial_scores.values()))
    return DataReward(
        reward=total,
        partial_scores=export_scores(partial_scores),
        feedback=feedback,
        done=done,
        success=total >= success_threshold,
    )


def format_progress_feedback(improved: List[str], penalties: Iterable[str], fallback: str) -> str:
    """Compose concise reward feedback."""

    parts: List[str] = []
    if improved:
        parts.append("Resolved: " + ", ".join(improved))
    penalty_list = [penalty for penalty in penalties if penalty]
    if penalty_list:
        parts.append("Warnings: " + "; ".join(penalty_list))
    if not parts:
        parts.append(fallback)
    return " ".join(parts)
