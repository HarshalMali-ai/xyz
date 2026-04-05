"""Dense reward shaping (-1.0 .. 1.0) for RAG pipeline debugging."""

from __future__ import annotations

from typing import Any


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


def step_reward(
    task_id: str,
    ground_truth: dict[str, Any],
    config: dict[str, Any],
    action_type: str,
    loop_penalty: bool,
) -> tuple[float, dict[str, Any]]:
    """Intermediate reward after an action (dense)."""
    breakdown: dict[str, Any] = {"action": action_type}
    r = 0.0

    if action_type == "request_hint":
        r -= 0.05
        breakdown["hint_penalty"] = -0.05
        return _clamp(r), breakdown

    if loop_penalty:
        r -= 0.05
        breakdown["loop_penalty"] = -0.05

    if task_id == "task_easy":
        expected_fp = str(ground_truth.get("retrieved_fingerprint", ""))
        fp = str(config.get("retrieved_fingerprint", ""))
        if expected_fp and fp == expected_fp:
            r += 0.35
            breakdown["retrieval_fingerprint_match"] = 0.35
        else:
            target_cs = int(ground_truth.get("chunk_size", 0))
            cs = int(config.get("chunk_size", 0))
            if cs == target_cs:
                r += 0.25
                breakdown["chunk_size_match_fallback"] = 0.25

    elif task_id == "task_medium":
        expected_fp = str(ground_truth.get("retrieved_fingerprint", ""))
        fp = str(config.get("retrieved_fingerprint", ""))
        if expected_fp and fp == expected_fp:
            r += 0.35
            breakdown["retrieval_fingerprint_match"] = 0.35
        else:
            m = str(config.get("embedding_model", ""))
            qm = str(config.get("query_embedding_model", ""))
            t = str(ground_truth.get("embedding_model", ""))
            if m == t and qm == t:
                r += 0.25
                breakdown["models_aligned_fallback"] = 0.25
        if bool(config.get("reindex_completed")):
            r += 0.35
            breakdown["reindex_ok"] = 0.35

    elif task_id == "task_hard":
        tk = int(config.get("top_k", 0))
        max_k = int(ground_truth["top_k"])
        if 0 < tk <= max_k:
            r += 0.2
            breakdown["top_k_ok"] = 0.2
        if bool(config.get("rerank_enabled")) == bool(ground_truth["rerank_enabled"]):
            r += 0.2
            breakdown["rerank_ok"] = 0.2
        if not bool(config.get("context_overflow_detected")):
            r += 0.25
            breakdown["no_overflow"] = 0.25

    feedback = "Intermediate progress" if r > 0 else "Keep adjusting configuration"
    return _clamp(r), breakdown


def terminal_reward_from_grader(grader_score: float) -> float:
    """Map grader [0,1] to [-1,1] for final step: strong success -> high positive."""
    return _clamp(-1.0 + 2.0 * float(grader_score))
