"""Scenario-driven RAG pipeline debugging environment."""

from __future__ import annotations

import json
from copy import deepcopy
from math import ceil
from typing import Any

from pydantic import ValidationError

from graders import MIN_SCORE, grade_episode
from models import Action, Observation
from reward import step_reward, terminal_reward_from_grader
from tasks import FLAGSHIP_TASKS, get_task_spec, list_task_specs, resolve_task_id


def _default_state() -> dict[str, Any]:
    return {
        "initialized": False,
        "task_id": None,
        "config": {},
        "step_count": 0,
        "done": False,
        "episode_actions": [],
        "hints_used": 0,
        "previous_actions": [],
    }


def _ratio(actual: int, target: int) -> float:
    if target <= 0:
        return 1.0
    return max(0.0, 1.0 - (abs(actual - target) / max(target, 1)))


class RAGPipelineEnv:
    """Simulates realistic retrieval failures across multiple production RAG scenarios."""

    def __init__(self) -> None:
        self._task: dict[str, Any] | None = None
        self._config: dict[str, Any] = {}
        self._step_count = 0
        self._done = False
        self._initialized = False
        self._last_actions: list[str] = []
        self._episode_actions: list[dict[str, Any]] = []
        self._hints_used = 0
        self._last_hint = ""

    @property
    def max_steps(self) -> int:
        if self._task is None:
            return 20
        return int(self._task.get("max_steps", 20))

    def reset(self, task_id: str | None = None) -> Observation:
        self._task = get_task_spec(task_id or FLAGSHIP_TASKS[0])
        self._config = deepcopy(self._task["default_config"])
        self._step_count = 0
        self._done = False
        self._initialized = True
        self._last_actions = []
        self._episode_actions = []
        self._hints_used = 0
        self._last_hint = ""
        return self._observe()

    def _simulate(self) -> dict[str, Any]:
        assert self._task is not None
        cfg = self._config
        target = self._task["target_config"]
        docs = self._task["corpus"]

        target_chunk = int(target.get("chunk_size", cfg.get("chunk_size", 500)))
        target_overlap = int(target.get("chunk_overlap", cfg.get("chunk_overlap", 0)))
        target_top_k = int(target.get("top_k", cfg.get("top_k", 5)))
        target_embed = str(target.get("embedding_model", cfg.get("embedding_model", "")))
        target_query = str(target.get("query_embedding_model", target_embed))
        target_rerank = bool(target.get("rerank_enabled", cfg.get("rerank_enabled", False)))

        chunk_fit = _ratio(int(cfg.get("chunk_size", target_chunk)), target_chunk)
        overlap_fit = _ratio(int(cfg.get("chunk_overlap", target_overlap)), max(target_overlap, 1)) if target_overlap else 1.0
        topk_penalty = max(0.0, (int(cfg.get("top_k", target_top_k)) - target_top_k) / max(target_top_k, 1))
        embed_fit = 1.0 if (
            str(cfg.get("embedding_model", "")) == target_embed
            and str(cfg.get("query_embedding_model", "")) == target_query
        ) else 0.0
        rerank_fit = 1.0 if bool(cfg.get("rerank_enabled", False)) == target_rerank else 0.0
        reindex_fit = 1.0 if (not self._task["reindex_required"] or bool(cfg.get("reindex_completed", False))) else 0.0

        scored_docs: list[tuple[float, dict[str, Any]]] = []
        for doc in docs:
            role = doc["role"]
            base = {"gold": 0.64, "support": 0.46, "noise": 0.14}[role]
            if role in {"gold", "support"}:
                score = (
                    base
                    + (0.20 * chunk_fit)
                    + (0.10 * overlap_fit)
                    + (0.20 * embed_fit)
                    + (0.12 * rerank_fit)
                    + (0.08 * reindex_fit)
                )
            else:
                score = (
                    base
                    + (0.18 * (1.0 - chunk_fit))
                    + (0.10 * (1.0 - overlap_fit))
                    + (0.22 * (1.0 - embed_fit))
                    + (0.10 * (1.0 - rerank_fit))
                    + (0.08 * (1.0 - reindex_fit))
                )
            if topk_penalty > 0 and role == "noise":
                score += min(0.25, topk_penalty * 0.20)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        top_k = max(1, int(cfg.get("top_k", target_top_k)))
        retrieved_docs = [doc for _, doc in scored_docs[:top_k]]

        chunk_size = max(1, int(cfg.get("chunk_size", 500)))
        chunk_overlap = max(0, int(cfg.get("chunk_overlap", 0)))
        stride = max(1, chunk_size - chunk_overlap)

        preview: list[dict[str, Any]] = []
        total_context_tokens = 0
        for rank, doc in enumerate(retrieved_docs, start=1):
            chunk_count = max(1, ceil(doc["tokens"] / stride))
            effective_tokens = min(doc["tokens"], chunk_size + chunk_overlap)
            if bool(cfg.get("rerank_enabled", False)):
                effective_tokens = int(effective_tokens * 0.88)
            total_context_tokens += effective_tokens
            preview.append(
                {
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "role": doc["role"],
                    "rank": rank,
                    "estimated_tokens": effective_tokens,
                    "summary": doc["summary"],
                    "chunk_count": chunk_count,
                }
            )

        max_context = int(cfg.get("max_context_tokens", 4096))
        overflow = total_context_tokens > max_context
        retrieved_ids = [doc["id"] for doc in retrieved_docs]
        ideal_ids = self._task.get("ideal_retrieval_ids", [])
        overlap_count = len(set(retrieved_ids) & set(ideal_ids))
        retrieval_precision = overlap_count / max(1, len(ideal_ids))

        return {
            "retrieved_preview": preview,
            "retrieved_preview_ids": retrieved_ids,
            "retrieved_fingerprint": "|".join(retrieved_ids),
            "retrieval_precision": round(retrieval_precision, 4),
            "estimated_context_tokens": total_context_tokens,
            "context_overflow_detected": overflow,
            "ideal_retrieval_ids": ideal_ids,
            "config_fit": {
                "chunk_fit": round(chunk_fit, 4),
                "overlap_fit": round(overlap_fit, 4),
                "embedding_fit": round(embed_fit, 4),
                "rerank_fit": round(rerank_fit, 4),
                "reindex_fit": round(reindex_fit, 4),
            },
        }

    def _refresh_sim_into_config(self, sim: dict[str, Any] | None = None) -> None:
        if sim is None:
            sim = self._simulate()
        self._config["context_overflow_detected"] = bool(sim["context_overflow_detected"])
        self._config["retrieved_preview_ids"] = list(sim["retrieved_preview_ids"])
        self._config["retrieved_fingerprint"] = str(sim["retrieved_fingerprint"])
        self._config["retrieval_precision"] = float(sim["retrieval_precision"])
        self._config["estimated_context_tokens"] = int(sim["estimated_context_tokens"])

    def observe(self) -> Observation:
        return self._observe()

    def _observe(self) -> Observation:
        if not self._initialized or self._task is None:
            return Observation(
                task_id="uninitialized",
                task_description="Call reset() before interacting with the environment.",
                current_context={"error": "reset required"},
                step_count=0,
                difficulty="easy",
                max_steps=20,
                hints_used=0,
                previous_actions=[],
                metadata={},
            )

        sim = self._simulate()
        self._refresh_sim_into_config(sim)
        return Observation(
            task_id=self._task["id"],
            task_description=self._task["description"],
            current_context={
                "operator_story": self._task["operator_story"],
                "business_impact": self._task["business_impact"],
                "user_query": self._task["user_query"],
                "acceptance_criteria": list(self._task["acceptance_criteria"]),
                "pipeline_config": deepcopy(self._config),
                "simulation": sim,
                "corpus": {
                    "documents": [
                        {
                            "id": doc["id"],
                            "title": doc["title"],
                            "estimated_tokens": doc["tokens"],
                            "summary": doc["summary"],
                        }
                        for doc in self._task["corpus"]
                    ]
                },
            },
            step_count=self._step_count,
            difficulty=self._task["difficulty"],
            max_steps=self.max_steps,
            hints_used=self._hints_used,
            previous_actions=list(self._last_actions),
            metadata={
                "title": self._task["title"],
                "reindex_required": self._task["reindex_required"],
                "overflow_sensitive": self._task["overflow_sensitive"],
                "last_hint": self._last_hint,
            },
        )

    def _loop_penalty(self, action_key: str) -> bool:
        self._last_actions.append(action_key)
        if len(self._last_actions) < 3:
            return False
        return self._last_actions[-1] == self._last_actions[-2] == self._last_actions[-3]

    def step(self, action: Action | None) -> tuple[Observation, float, bool, dict[str, Any]]:
        if not self._initialized or self._task is None:
            obs = self.reset()
            return obs, MIN_SCORE, False, {"warning": "environment auto-reset"}

        if action is None:
            return self._observe(), MIN_SCORE, False, {"error": "null action"}

        try:
            Action.model_validate(action.model_dump())
        except ValidationError as exc:
            return self._observe(), MIN_SCORE, False, {"error": "invalid action", "detail": str(exc)}

        self._step_count += 1
        if self._step_count > self.max_steps:
            self._done = True
            return self._observe(), MIN_SCORE, True, {"error": "max_steps"}

        payload = deepcopy(action.payload or {})
        action_key = f"{action.action_type}:{json.dumps(payload, sort_keys=True)[:200]}"
        loop = self._loop_penalty(action_key)
        info: dict[str, Any] = {}

        if action.action_type == "request_hint":
            hint_index = min(self._hints_used, len(self._task["hints"]) - 1)
            self._last_hint = self._task["hints"][hint_index]
            self._hints_used += 1
            reward, breakdown = step_reward(self._task["id"], self._config, "request_hint", loop)
            info = {"breakdown": breakdown, "hint": self._last_hint}
            self._episode_actions.append({"action": "request_hint", "payload": {}, "reward": reward})
            return self._observe(), reward, False, info

        if action.action_type == "configure":
            reindex_sensitive = {"chunk_size", "chunk_overlap", "embedding_model", "query_embedding_model"}
            for key, value in payload.items():
                if value is None:
                    continue
                if key in {"chunk_size", "chunk_overlap", "top_k", "max_context_tokens"}:
                    self._config[key] = int(value)
                elif key == "rerank_enabled":
                    self._config[key] = bool(value)
                else:
                    self._config[key] = str(value)
                if key in reindex_sensitive:
                    self._config["reindex_completed"] = False

            self._refresh_sim_into_config()
            reward, breakdown = step_reward(self._task["id"], self._config, "configure", loop)
            info = {"breakdown": breakdown}

        elif action.action_type == "reindex":
            self._config["reindex_completed"] = True
            self._refresh_sim_into_config()
            reward, breakdown = step_reward(self._task["id"], self._config, "reindex", loop)
            info = {"breakdown": breakdown}

        elif action.action_type == "submit":
            self._refresh_sim_into_config(self._simulate())
            grader_score = grade_episode(self._task["id"], self._config, {"actions": self._episode_actions})
            reward = terminal_reward_from_grader(grader_score)
            self._done = True
            info = {"grader_score": grader_score, "terminal": True}
            self._episode_actions.append({"action": "submit", "payload": {}, "reward": reward})
            return self._observe(), reward, True, info

        else:
            reward = MIN_SCORE
            info = {"error": f"unknown action_type: {action.action_type}"}

        self._episode_actions.append({"action": action.action_type, "payload": payload, "reward": reward})
        return self._observe(), reward, self._done, info

    def state(self) -> dict[str, Any]:
        if not self._initialized or self._task is None:
            return _default_state()
        return {
            "initialized": True,
            "task_id": self._task["id"],
            "task_title": self._task["title"],
            "available_tasks": [task["id"] for task in list_task_specs()],
            "config": deepcopy(self._config),
            "step_count": self._step_count,
            "done": self._done,
            "episode_actions": deepcopy(self._episode_actions),
            "hints_used": self._hints_used,
            "previous_actions": list(self._last_actions),
        }


def default_state() -> dict[str, Any]:
    return _default_state()
