"""Core environment: reset, step, state — RAG pipeline debugging simulation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from graders import grade_episode
from models import Action, Observation
from reward import step_reward, terminal_reward_from_grader

_ROOT = Path(__file__).resolve().parent.parent

# Tiny fixed corpus (deterministic)
_CORPUS: list[dict[str, Any]] = [
    {"id": "doc_a", "text": "OpenEnv standardizes agent environments. " * 80},
    {"id": "doc_b", "text": "Retrieval quality depends on chunking and embeddings. " * 80},
]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _load_ground(task_id: str) -> dict[str, Any]:
    name = {
        "task_easy": "easy_cases.json",
        "task_medium": "medium_cases.json",
        "task_hard": "hard_cases.json",
    }[task_id]
    with open(_ROOT / "dataset" / name, encoding="utf-8") as f:
        return json.load(f)["ground_truth"]


def _default_config(task_id: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "chunk_size": 500,
        "embedding_model": "text-embedding-3-small",
        "query_embedding_model": "text-embedding-3-small",
        "top_k": 5,
        "rerank_enabled": True,
        "reindex_completed": False,
        "max_context_tokens": 12000,
        "context_overflow_detected": False,
    }
    if task_id == "task_easy":
        base["chunk_size"] = 2000
        base["reindex_completed"] = False
    elif task_id == "task_medium":
        base["embedding_model"] = "text-embedding-ada-002"
        base["query_embedding_model"] = "text-embedding-3-small"
        base["reindex_completed"] = True  # index exists but wrong model
    elif task_id == "task_hard":
        base["top_k"] = 20
        base["rerank_enabled"] = False
        base["max_context_tokens"] = 2000
    return base


class RAGPipelineEnv:
    max_steps: int = 24

    def __init__(self) -> None:
        self._task_id: str = "task_easy"
        self._config: dict[str, Any] = {}
        self._step_count: int = 0
        self._done: bool = False
        self._last_actions: list[str] = []
        self._episode_actions: list[dict[str, Any]] = []
        self._initialized: bool = False

    def reset(self, task_id: str | None = None) -> Observation:
        tid = task_id or "task_easy"
        if tid not in ("task_easy", "task_medium", "task_hard"):
            tid = "task_easy"
        self._task_id = tid
        self._config = _default_config(tid)
        self._step_count = 0
        self._done = False
        self._last_actions = []
        self._episode_actions = []
        self._initialized = True
        return self._observe()

    def _simulate(self) -> dict[str, Any]:
        cfg = self._config
        chunk_size = int(cfg.get("chunk_size", 500))
        chunks: list[dict[str, Any]] = []
        for doc in _CORPUS:
            text = doc["text"]
            est_doc_tokens = _estimate_tokens(text)
            n = max(1, len(text) // max(1, chunk_size // 4))
            for i in range(n):
                seg = text[i * len(text) // n : (i + 1) * len(text) // n]
                chunks.append(
                    {
                        "doc_id": doc["id"],
                        "chunk_index": i,
                        "est_tokens": _estimate_tokens(seg),
                    }
                )

        index_m = str(cfg.get("embedding_model", ""))
        query_m = str(cfg.get("query_embedding_model", ""))
        mismatch = index_m != query_m
        top_k = int(cfg.get("top_k", 5))
        retrieved = chunks[: min(top_k, len(chunks))]
        if mismatch:
            retrieved = list(reversed(retrieved))[:top_k]

        # Deterministic "reranking": prefer chunks with fewer estimated tokens.
        # This simulates an LLM-based reranker reducing context flood.
        if bool(cfg.get("rerank_enabled", False)):
            retrieved = sorted(retrieved, key=lambda c: (int(c["est_tokens"]), int(c["chunk_index"])))[:top_k]

        total_ctx = sum(c["est_tokens"] for c in retrieved)
        max_ctx = int(cfg.get("max_context_tokens", 12000))
        overflow = total_ctx > max_ctx

        retrieved_ids = [f"{c['doc_id']}:{c['chunk_index']}" for c in retrieved]
        retrieved_fingerprint = "|".join(retrieved_ids)

        return {
            "num_chunks_indexed": len(chunks),
            "retrieved_preview": retrieved[:5],
            "retrieved_preview_ids": retrieved_ids[:5],
            "retrieved_fingerprint": retrieved_fingerprint,
            "retrieved_count": len(retrieved),
            "embedding_index_mismatch": mismatch,
            "estimated_context_tokens": total_ctx,
            "context_overflow_detected": overflow,
            "doc_token_estimate": est_doc_tokens,
        }

    def _refresh_sim_into_config(self, sim: dict[str, Any] | None = None) -> None:
        if sim is None:
            sim = self._simulate()
        self._config["context_overflow_detected"] = bool(sim["context_overflow_detected"])
        self._config["embedding_index_mismatch"] = bool(sim["embedding_index_mismatch"])
        self._config["retrieved_count"] = int(sim["retrieved_count"])
        self._config["estimated_context_tokens"] = int(sim["estimated_context_tokens"])
        self._config["retrieved_preview_ids"] = list(sim.get("retrieved_preview_ids", []))
        self._config["retrieved_fingerprint"] = str(sim.get("retrieved_fingerprint", ""))

    def observe(self) -> Observation:
        """Public observation (used by API on validation errors)."""
        return self._observe()

    def _observe(self) -> Observation:
        sim = self._simulate()
        self._refresh_sim_into_config(sim)
        desc = {
            "task_easy": "Fix chunk_size so chunks align with ~500-token documents.",
            "task_medium": "Align embedding_model with query_embedding_model and reindex.",
            "task_hard": "Lower top_k and enable reranking to avoid context overflow.",
        }[self._task_id]
        ctx = {
            "pipeline_config": deepcopy(self._config),
            "simulation": sim,
            "corpus": {"documents": len(_CORPUS), "hint": "Use configure action, then reindex when required."},
        }
        return Observation(
            task_id=self._task_id,
            task_description=desc,
            current_context=ctx,
            step_count=self._step_count,
        )

    def _loop_penalty(self, action_key: str) -> bool:
        self._last_actions.append(action_key)
        if len(self._last_actions) < 3:
            return False
        return self._last_actions[-1] == self._last_actions[-2] == self._last_actions[-3]

    def step(self, action: Action | None) -> tuple[Observation, float, bool, dict[str, Any]]:
        if not self._initialized:
            obs = Observation(
                task_id="none",
                task_description="",
                current_context={"error": "call reset first"},
                step_count=0,
            )
            return obs, 0.001, False, {"error": "reset required"}

        if action is None:
            return self._observe(), 0.001, False, {"error": "null action"}

        try:
            Action.model_validate(action.model_dump())
        except ValidationError as e:
            return self._observe(), 0.001, False, {"error": "invalid action", "detail": str(e)}

        self._step_count += 1
        if self._step_count > self.max_steps:
            self._done = True
            return self._observe(), 0.001, True, {"error": "max_steps"}

        a = action.action_type
        payload = action.payload or {}
        key = f"{a}:{json.dumps(payload, sort_keys=True)[:200]}"
        loop = self._loop_penalty(key)

        info: dict[str, Any] = {}
        reward = 0.0

        if a == "request_hint":
            if len(json.dumps(payload)) > 10000:
                payload = {"truncated": True}
            self._refresh_sim_into_config()
            reward, br = step_reward(self._task_id, _load_ground(self._task_id), self._config, "request_hint", loop)
            info["breakdown"] = br

        elif a == "configure":
            for k in ("chunk_size", "top_k", "embedding_model", "query_embedding_model", "rerank_enabled"):
                if k in payload and payload[k] is not None:
                    if k in ("chunk_size", "top_k"):
                        self._config[k] = int(payload[k])
                    elif k == "rerank_enabled":
                        self._config[k] = bool(payload[k])
                    else:
                        self._config[k] = str(payload[k])
            if self._task_id == "task_easy":
                self._config["reindex_completed"] = False
            elif self._task_id == "task_medium":
                if any(k in payload for k in ("embedding_model", "query_embedding_model")):
                    self._config["reindex_completed"] = False
            self._refresh_sim_into_config()
            reward, br = step_reward(self._task_id, _load_ground(self._task_id), self._config, "configure", loop)
            info["breakdown"] = br

        elif a == "reindex":
            self._config["reindex_completed"] = True
            self._refresh_sim_into_config()
            reward, br = step_reward(self._task_id, _load_ground(self._task_id), self._config, "configure", loop)
            info["breakdown"] = br

        elif a == "submit":
            self._refresh_sim_into_config(self._simulate())
            g = grade_episode(self._task_id, self._config, {"actions": self._episode_actions})
            reward = terminal_reward_from_grader(g)
            self._done = True
            info["grader_score"] = g
            info["terminal"] = True
            return self._observe(), reward, True, info

        else:
            reward = 0.001
            info["error"] = f"unknown action_type: {a}"

        self._episode_actions.append({"action": a, "payload": payload, "reward": reward})
        return self._observe(), reward, self._done, info

    def state(self) -> dict[str, Any]:
        if not self._initialized:
            return {
                "initialized": False,
                "task_id": None,
                "config": {},
                "step_count": 0,
                "done": False,
            }
        return {
            "initialized": True,
            "task_id": self._task_id,
            "config": deepcopy(self._config),
            "step_count": self._step_count,
            "done": self._done,
            "episode_actions": self._episode_actions,
        }


def default_state() -> dict[str, Any]:
    return {
        "initialized": False,
        "task_id": None,
        "config": {},
        "step_count": 0,
        "done": False,
    }
