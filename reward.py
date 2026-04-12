"""Dense reward shaping for the RAG pipeline debugger."""

from __future__ import annotations

from typing import Any

from graders import progress_report

MIN_REWARD = 0.01
MAX_REWARD = 0.99


def _clamp(value: float) -> float:
    return max(MIN_REWARD, min(MAX_REWARD, round(float(value), 4)))


def step_reward(
    task_id: str,
    final_config: dict[str, Any],
    action_type: str,
    loop_penalty: bool,
) -> tuple[float, dict[str, Any]]:
    report = progress_report(task_id, final_config)
    progress = float(report["objective_progress"])

    action_bonus = {
        "configure": 0.08,
        "reindex": 0.10,
        "request_hint": -0.05,
    }.get(action_type, 0.03)

    reward = 0.08 + (progress * 0.60) + action_bonus
    breakdown: dict[str, Any] = {
        "objective_progress": report["objective_progress"],
        "config_progress": report["config_progress"],
        "retrieval_progress": report["retrieval_progress"],
        "reindex_progress": report["reindex_progress"],
        "overflow_progress": report["overflow_progress"],
        "action_bonus": round(action_bonus, 4),
    }

    if loop_penalty:
        reward -= 0.08
        breakdown["loop_penalty"] = -0.08

    if action_type == "request_hint":
        breakdown["hint_penalty"] = -0.05

    return _clamp(reward), breakdown


def terminal_reward_from_grader(grader_score: float) -> float:
    return _clamp(grader_score)
