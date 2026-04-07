"""Grader range and determinism tests."""

from __future__ import annotations

from dataenv.data_generators import generate_easy, generate_hard, generate_medium
from dataenv.graders import grader_easy, grader_hard, grader_medium


def test_easy_grader_range() -> None:
    data = generate_easy.generate(seed=42)
    result = grader_easy.compute_final_reward(data)
    assert 0.0 < result.reward < 1.0


def test_medium_grader_range() -> None:
    data = generate_medium.generate(seed=42)
    result = grader_medium.compute_final_reward(data)
    assert 0.0 < result.reward < 1.0


def test_hard_grader_range() -> None:
    data = generate_hard.generate(seed=42)
    result = grader_hard.compute_final_reward(data)
    assert 0.0 < result.reward < 1.0


def test_graders_are_deterministic() -> None:
    data1 = generate_easy.generate(seed=99)
    data2 = generate_easy.generate(seed=99)
    r1 = grader_easy.compute_final_reward(data1)
    r2 = grader_easy.compute_final_reward(data2)
    assert r1.reward == r2.reward
