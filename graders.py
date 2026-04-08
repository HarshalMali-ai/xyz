"""Deterministic programmatic graders (no LLM). Returns score in (0.0, 1.0)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
MIN_SCORE = 0.001
MAX_SCORE = 0.999


def _strict_score(value: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, float(value)))


def _load_cases(name: str) -> dict[str, Any]:
    p = _ROOT / "dataset" / name
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _get_ground(task_id: str) -> dict[str, Any]:
    mapping = {
        "task_easy": "easy_cases.json",
        "task_medium": "medium_cases.json",
        "task_hard": "hard_cases.json",
    }
    data = _load_cases(mapping[task_id])
    return data["ground_truth"]


def grade_episode(task_id: str, final_config: dict[str, Any], episode: dict[str, Any] | None = None) -> float:
    """
    Grade final pipeline config. episode is optional (for future partial credit hooks).
    """
    if episode is None:
        episode = {}
    _ = episode
    gt = _get_ground(task_id)
    cfg = final_config or {}

    if task_id == "task_easy":
        return _strict_score(_grade_easy(cfg, gt))
    if task_id == "task_medium":
        return _strict_score(_grade_medium(cfg, gt))
    if task_id == "task_hard":
        return _strict_score(_grade_hard(cfg, gt))
    return MIN_SCORE


def _grade_easy(cfg: dict[str, Any], gt: dict[str, Any]) -> float:
    score = 0.0
    expected_fp = gt.get("retrieved_fingerprint")
    fp = str(cfg.get("retrieved_fingerprint", ""))
    if expected_fp:
        if fp == expected_fp:
            score += 0.7
        elif not fp:
            # If callers didn't provide retrieval artifacts, fall back to config-based checks.
            if int(cfg.get("chunk_size", -1)) == int(gt["chunk_size"]):
                score += 0.7
    else:
        # Backward-compatible fallback (should rarely be used)
        if int(cfg.get("chunk_size", -1)) == int(gt["chunk_size"]):
            score += 0.7
    if bool(cfg.get("reindex_completed")):
        score += 0.3
    return score


def _grade_medium(cfg: dict[str, Any], gt: dict[str, Any]) -> float:
    score = 0.0
    expected_fp = gt.get("retrieved_fingerprint")
    fp = str(cfg.get("retrieved_fingerprint", ""))
    if expected_fp:
        if fp == expected_fp:
            score += 0.7
        elif not fp:
            # If retrieval artifacts are missing, fall back to config-based checks.
            m = str(cfg.get("embedding_model", ""))
            qm = str(cfg.get("query_embedding_model", ""))
            t = str(gt["embedding_model"])
            if m == t:
                score += 0.35
            if qm == t:
                score += 0.35
    else:
        # Backward-compatible fallback (should rarely be used)
        m = str(cfg.get("embedding_model", ""))
        qm = str(cfg.get("query_embedding_model", ""))
        t = str(gt["embedding_model"])
        if m == t:
            score += 0.35
        if qm == t:
            score += 0.35
    if bool(cfg.get("reindex_completed")):
        score += 0.3
    return score


def _grade_hard(cfg: dict[str, Any], gt: dict[str, Any]) -> float:
    score = 0.0

    # Prefer retrieval fingerprint match (it reflects top_k + reranking effects)
    expected_fp = gt.get("retrieved_fingerprint")
    fp = str(cfg.get("retrieved_fingerprint", ""))
    if expected_fp:
        if fp:
            if fp == expected_fp:
                score += 0.45
        else:
            # Backward-compatible fallback when retrieval artifacts aren't provided.
            tk = int(cfg.get("top_k", 999))
            if tk <= int(gt["top_k"]):
                score += 0.45
            if bool(cfg.get("rerank_enabled")) == bool(gt["rerank_enabled"]):
                score += 0.35
            if not bool(cfg.get("context_overflow_detected")):
                score += 0.2
            return score

    tk = int(cfg.get("top_k", 999))
    if tk <= int(gt["top_k"]):
        score += 0.25

    if bool(cfg.get("rerank_enabled")) == bool(gt["rerank_enabled"]):
        score += 0.15

    # context_overflow_detected is produced by retrieval simulation
    if not bool(cfg.get("context_overflow_detected")):
        score += 0.15

    return score


def grade_action_dummy(task_id: str, payload: dict[str, Any] | None) -> float:
    """For /grader tests: never crash on malformed input."""
    if payload is None:
        return MIN_SCORE
    return grade_episode(task_id, payload)
