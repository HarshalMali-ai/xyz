"""Programmatic graders for the RAG pipeline debugger."""

from __future__ import annotations

from typing import Any

from tasks import get_task_spec

MIN_SCORE = 0.01
MAX_SCORE = 0.99


def _strict_score(value: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, round(float(value), 4)))


def _bounded_ratio(actual: Any, target: Any) -> float:
    if isinstance(target, bool):
        return 1.0 if bool(actual) is bool(target) else 0.0
    if isinstance(target, str):
        return 1.0 if str(actual) == target else 0.0
    if isinstance(target, int):
        try:
            actual_int = int(actual)
        except (TypeError, ValueError):
            return 0.0
        tolerance = max(1, abs(target))
        return max(0.0, 1.0 - (abs(actual_int - target) / tolerance))
    return 0.0


def _config_progress(spec: dict[str, Any], final_config: dict[str, Any]) -> tuple[float, dict[str, float]]:
    target_config = spec["target_config"]
    breakdown: dict[str, float] = {}
    if not target_config:
        return 1.0, breakdown

    total = 0.0
    for key, target_value in target_config.items():
        progress = _bounded_ratio(final_config.get(key), target_value)
        breakdown[key] = round(progress, 4)
        total += progress
    return total / len(target_config), breakdown


def _retrieval_progress(spec: dict[str, Any], final_config: dict[str, Any]) -> float:
    ideal_ids = spec.get("ideal_retrieval_ids", [])
    current_ids = list(final_config.get("retrieved_preview_ids", []) or [])
    if not ideal_ids:
        return 1.0
    if not current_ids:
        return 0.0
    overlap = len(set(ideal_ids) & set(current_ids))
    return overlap / len(ideal_ids)


def progress_report(task_id: str, final_config: dict[str, Any], episode: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = get_task_spec(task_id)
    cfg = final_config or {}
    _ = episode

    config_score, config_breakdown = _config_progress(spec, cfg)
    retrieval_score = _retrieval_progress(spec, cfg)
    reindex_score = 1.0 if (not spec["reindex_required"] or bool(cfg.get("reindex_completed"))) else 0.0
    overflow_score = 1.0 if (not spec["overflow_sensitive"] or not bool(cfg.get("context_overflow_detected"))) else 0.0

    weights = {
        "config": 0.45,
        "retrieval": 0.30,
        "reindex": 0.15,
        "overflow": 0.10,
    }

    total_progress = (
        (config_score * weights["config"])
        + (retrieval_score * weights["retrieval"])
        + (reindex_score * weights["reindex"])
        + (overflow_score * weights["overflow"])
    )

    return {
        "task_id": spec["id"],
        "config_progress": round(config_score, 4),
        "config_breakdown": config_breakdown,
        "retrieval_progress": round(retrieval_score, 4),
        "reindex_progress": round(reindex_score, 4),
        "overflow_progress": round(overflow_score, 4),
        "objective_progress": round(total_progress, 4),
    }


def grade_episode(task_id: str, final_config: dict[str, Any], episode: dict[str, Any] | None = None) -> float:
    report = progress_report(task_id, final_config, episode)
    return _strict_score(report["objective_progress"])


def grade_action_dummy(task_id: str, payload: dict[str, Any] | None) -> float:
    if payload is None:
        return MIN_SCORE
    return grade_episode(task_id, payload)
