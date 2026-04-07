"""Core OpenEnv-compatible environment implementation."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from dataenv.data_generators import generate_easy, generate_hard, generate_medium
from dataenv.graders.common import export_score
from dataenv.graders import grader_easy, grader_hard, grader_medium
from dataenv.models import DataAction, DataObservation, DataReward, EpisodeState
from dataenv.tasks import task_easy, task_hard, task_medium

logger = logging.getLogger(__name__)

TASK_MAP = {
    "schema_fix": (generate_easy, grader_easy, task_easy, task_easy.MAX_STEPS),
    "clean_pipeline": (generate_medium, grader_medium, task_medium, task_medium.MAX_STEPS),
    "join_repair": (generate_hard, grader_hard, task_hard, task_hard.MAX_STEPS),
}


class DataEnv:
    """OpenEnv-compatible environment for data pipeline debugging."""

    def __init__(self, task_id: str | None = None, seed: int | None = None):
        self.task_id = task_id or os.getenv("DATAENV_TASK", "schema_fix")
        if self.task_id not in TASK_MAP:
            raise ValueError(f"Unknown DATAENV_TASK '{self.task_id}'. Expected one of {list(TASK_MAP)}.")
        env_seed = seed if seed is not None else os.getenv("DATAENV_SEED")
        self.seed = int(env_seed) if env_seed is not None else int(np.random.randint(0, 10000))
        self._reset_state()

    def _reset_state(self) -> None:
        generator, grader, task, max_steps = TASK_MAP[self.task_id]
        self.generator = generator
        self.grader = grader
        self.task = task
        self.max_steps = max_steps
        self.data = self.generator.generate(seed=self.seed)
        self.current_step = 0
        self.actions_taken: list[DataAction] = []
        self.issues_resolved = []
        self.issues_remaining = self.task.get_initial_issues(self.data)
        self.cumulative_reward = export_score(0.0)
        self.done = False
        self.data.setdefault("episode_metrics", {})["max_steps"] = self.max_steps
        self.data.setdefault("progress_cache", {})
        logger.info("Environment reset for task=%s seed=%s", self.task_id, self.seed)

    def reset(self) -> DataObservation:
        """Reset the environment and return the initial observation."""

        self._reset_state()
        return self._build_observation()

    def step(self, action: DataAction) -> Tuple[DataObservation, DataReward, bool, Dict[str, Any]]:
        """Apply one structured action."""

        if self.done:
            raise RuntimeError("Episode already done. Call reset() first.")

        self.current_step += 1
        self.data.setdefault("episode_metrics", {})["steps_used"] = self.current_step
        error_msg = None
        result_msg = None

        try:
            result_msg = self.task.apply_action(self.data, action)
            self.actions_taken.append(action)
            reward_obj = self.grader.compute_step_reward(
                self.data,
                action,
                self.actions_taken,
                self.issues_resolved,
            )
            self.issues_resolved = self.grader.get_resolved_issues(self.data)
            self.issues_remaining = [issue for issue in self.task.get_initial_issues(self.data) if issue not in self.issues_resolved]
            self.cumulative_reward = export_score(self.cumulative_reward + reward_obj.reward)
        except Exception as exc:
            error_msg = str(exc)
            logger.exception("Action failed for task=%s step=%s", self.task_id, self.current_step)
            reward_obj = DataReward(
                reward=export_score(0.0),
                partial_scores={},
                feedback=f"Action failed: {error_msg}",
                done=False,
                success=False,
            )

        if action.action_type == "submit" or self.current_step >= self.max_steps:
            final_reward = self.grader.compute_final_reward(self.data)
            reward_obj = final_reward
            self.cumulative_reward = final_reward.reward
            self.done = True

        obs = self._build_observation(last_action_result=result_msg, last_action_error=error_msg)
        obs.done = self.done
        obs.score_so_far = self.cumulative_reward
        return obs, reward_obj, self.done, {"step": self.current_step}

    def state(self) -> EpisodeState:
        """Return the full episode state."""

        return EpisodeState(
            task_id=self.task_id,
            step=self.current_step,
            current_observation=self._build_observation(),
            actions_taken=self.actions_taken,
            cumulative_reward=self.cumulative_reward,
            issues_resolved=self.issues_resolved,
            issues_remaining=self.issues_remaining,
            done=self.done,
        )

    def close(self) -> None:
        """Close the environment."""

        logger.info("Environment closed for task=%s", self.task_id)

    def _primary_df(self) -> pd.DataFrame:
        if "df" in self.data:
            return self.data["df"]
        return self.data["orders"]

    def _build_observation(
        self,
        last_action_result: str | None = None,
        last_action_error: str | None = None,
    ) -> DataObservation:
        """Build a structured observation from the current state."""

        df = self._primary_df().copy(deep=False)
        duplicate_count = int(df.duplicated().sum())
        if self.task_id == "clean_pipeline":
            duplicate_count = int(df.duplicated(subset=["transaction_id"]).sum())

        sample_df = df.head(5).copy()
        for column in sample_df.columns:
            if pd.api.types.is_datetime64_any_dtype(sample_df[column]):
                sample_df[column] = sample_df[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
        sample_df = sample_df.where(pd.notnull(sample_df), "NULL")

        return DataObservation(
            task_id=self.task_id,
            task_description=self.task.DESCRIPTION,
            step=self.current_step,
            max_steps=self.max_steps,
            columns=list(df.columns),
            dtypes={column: str(df[column].dtype) for column in df.columns},
            shape=[int(df.shape[0]), int(df.shape[1])],
            null_counts={column: int(df[column].isna().sum()) for column in df.columns},
            duplicate_rows=duplicate_count,
            sample_rows=sample_df.to_dict(orient="records"),
            detected_issues=self.task.detect_issues(self.data),
            last_action_result=last_action_result,
            last_action_error=last_action_error,
            done=self.done,
            score_so_far=self.cumulative_reward,
        )
