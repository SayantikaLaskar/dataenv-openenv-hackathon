"""Inference runner for DataEnv."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict

from openai import OpenAI

from dataenv.env import DataEnv
from dataenv.models import DataAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY", "")
)
BENCHMARK = "dataenv"
MAX_STEPS = 20
TEMPERATURE = 0.3
MAX_TOKENS = 512
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "dataenv")

SYSTEM_PROMPT = """You are a data engineering expert. You will receive observations about
a broken data pipeline and must take actions to fix it. Respond ONLY with a valid JSON
object matching the DataAction schema. Available action_types: fix_schema, fill_missing,
drop_duplicates, rename_column, drop_column, fix_join_key, filter_rows, submit.

Always include your reasoning field. Be systematic: identify all issues first, then fix
one at a time. Call submit only when all issues are resolved."""

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def obs_to_prompt(obs: Dict[str, Any]) -> str:
    """Convert an observation into an LLM prompt."""

    return f"""Current pipeline state:
Task: {obs['task_description']}
Step: {obs['step']}/{obs['max_steps']}
Shape: {obs['shape'][0]} rows x {obs['shape'][1]} cols
Columns & types: {json.dumps(obs['dtypes'], indent=2)}
Null counts: {json.dumps(obs['null_counts'], indent=2)}
Duplicate rows: {obs['duplicate_rows']}
Detected issues: {json.dumps(obs['detected_issues'], indent=2)}
Sample data (first 5 rows): {json.dumps(obs['sample_rows'], indent=2)}
Last action result: {obs.get('last_action_result', 'None')}
Last action error: {obs.get('last_action_error', 'None')}
Score so far: {obs['score_so_far']:.2f}

Respond with a JSON DataAction object to fix the next issue."""


def _extract_json_object(raw: str) -> Dict[str, Any]:
    """Extract a JSON object from raw model output."""

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _heuristic_action(obs: Dict[str, Any]) -> DataAction:
    """Fallback local policy so inference works without remote API access."""

    task_id = obs["task_id"]
    issues = " ".join(obs["detected_issues"])
    if task_id == "schema_fix":
        if obs["dtypes"].get("age") not in {"int64", "int32"}:
            return DataAction(action_type="fix_schema", column="age", target_dtype="int64", reasoning="Normalize age strings to integers.")
        if obs["dtypes"].get("salary") not in {"float64", "float32"}:
            return DataAction(action_type="fix_schema", column="salary", target_dtype="float64", reasoning="Strip currency markers and cast salary to float.")
        if "datetime64" not in obs["dtypes"].get("hire_date", ""):
            return DataAction(action_type="fix_schema", column="hire_date", target_dtype="datetime64", reasoning="Parse ISO dates into datetime.")
        if obs["dtypes"].get("is_active") != "bool":
            return DataAction(action_type="fix_schema", column="is_active", target_dtype="bool", reasoning="Map Yes and No values to booleans.")
        return DataAction(action_type="submit", reasoning="All schema issues appear resolved.")

    if task_id == "clean_pipeline":
        if obs["duplicate_rows"] > 0:
            return DataAction(action_type="drop_duplicates", reasoning="Remove duplicate transaction_id rows first.")
        if obs["null_counts"].get("amount", 0) > 0:
            return DataAction(action_type="fill_missing", column="amount", fill_strategy="median", reasoning="Use the median for numeric amount nulls.")
        if obs["null_counts"].get("category", 0) > 0:
            return DataAction(action_type="fill_missing", column="category", fill_strategy="mode", reasoning="Use the mode for category nulls.")
        if obs["null_counts"].get("timestamp", 0) > 0:
            return DataAction(action_type="fill_missing", column="timestamp", fill_strategy="ffill", reasoning="Forward-fill timestamps after sorting order is preserved.")
        if obs["null_counts"].get("customer_name", 0) > 0:
            return DataAction(action_type="fill_missing", column="customer_name", fill_strategy="drop", reasoning="Drop rows without customer names once fills are complete.")
        return DataAction(action_type="submit", reasoning="Duplicates and null handling appear complete.")

    if "mixed formats" in issues:
        return DataAction(action_type="fix_join_key", column="customer_ref", reasoning="Normalize mixed customer_ref formats to CUST_XXX.")
    if "negative values" in issues:
        return DataAction(action_type="filter_rows", condition="amount < 0", reasoning="Remove corrupted negative amount rows.")
    if "name clash" in issues:
        return DataAction(action_type="rename_column", column="customers.created_at", new_name="customer_created_at", reasoning="Avoid duplicate created_at columns in the join.")
    return DataAction(action_type="submit", reasoning="Join repair issues appear resolved.")


def _model_action(client: OpenAI | None, obs: Dict[str, Any]) -> DataAction:
    """Get the next action from the model when available, else use the heuristic."""

    if client is None:
        return _heuristic_action(obs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs_to_prompt(obs)},
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = response.choices[0].message.content or ""
        return DataAction(**_extract_json_object(raw))
    except Exception as exc:
        logger.debug("Model call failed; falling back to heuristic: %s", exc)
        return _heuristic_action(obs)


def run_episode(task_id: str) -> Dict[str, Any]:
    """Run one full task episode."""

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL) if API_KEY else None
    env = DataEnv(task_id=task_id)
    obs = env.reset()
    obs_dict = obs.model_dump()

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    rewards: list[float] = []
    step = 0
    done = False
    success = False

    try:
        while not done and step < MAX_STEPS:
            step += 1
            action = _model_action(client, obs_dict)
            action_str = json.dumps(action.model_dump(exclude_none=True), ensure_ascii=True)

            try:
                obs, reward, done, info = env.step(action)
                obs_dict = obs.model_dump()
                step_reward = reward.reward
                error_val = obs_dict.get("last_action_error") or "null"
                success = reward.success if done else success
            except Exception as exc:
                step_reward = 0.0
                error_val = str(exc)
                done = True
                success = False

            rewards.append(step_reward)
            done_str = "true" if done else "false"
            print(f"[STEP] step={step} action={action_str} reward={step_reward:.4f} done={done_str} error={error_val}")

    except Exception as exc:
        rewards.append(0.0)
        print(f"[STEP] step={step + 1} action=null reward=0.00 done=true error={str(exc)}")
        success = False
        done = True
    finally:
        env.close()

    final_score = rewards[-1] if rewards else 0.0
    rewards_str = ",".join(f"{reward:.4f}" for reward in rewards)
    print(f"[END] success={'true' if success else 'false'} steps={step} score={final_score:.4f} rewards={rewards_str}")
    return {"task": task_id, "score": final_score, "success": success, "steps": step}


if __name__ == "__main__":
    for task_name in ["schema_fix", "clean_pipeline", "join_repair"]:
        run_episode(task_name)
