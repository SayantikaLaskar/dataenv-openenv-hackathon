"""Common grading helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List

from dataenv.models import DataReward

STRICT_SCORE_EPSILON = 1e-4


def clamp(value: float) -> float:
    """Clamp a reward value to the OpenEnv range."""

    return max(0.0, min(1.0, float(value)))


def clamp_strict(value: float) -> float:
    """Clamp a task reward to the validator-safe open interval (0, 1)."""

    return max(STRICT_SCORE_EPSILON, min(1.0 - STRICT_SCORE_EPSILON, float(value)))


def reward_from_scores(
    partial_scores: Dict[str, float],
    *,
    feedback: str,
    success_threshold: float,
    done: bool,
) -> DataReward:
    """Build a DataReward from partial scores."""

    total = clamp_strict(sum(partial_scores.values()))
    return DataReward(
        reward=total,
        partial_scores={key: round(clamp(value), 4) for key, value in partial_scores.items()},
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
