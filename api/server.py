"""FastAPI server: all endpoints required by the hackathon guide."""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from environment.rag_environment import RAGPipelineEnv
from graders import MIN_SCORE, grade_episode
from models import Action
from tasks import list_tasks_payload

_env: RAGPipelineEnv | None = None
_tasks_payload_cache: list[dict[str, Any]] | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm common state eagerly so the service is ready before the first request.
    global _tasks_payload_cache
    get_env()
    if _tasks_payload_cache is None:
        _tasks_payload_cache = list_tasks_payload()
    yield


app = FastAPI(title="OpenEnv RAG Pipeline Debugger", version="1.0.0", lifespan=lifespan)


def get_env() -> RAGPipelineEnv:
    global _env
    if _env is None:
        _env = RAGPipelineEnv()
    return _env


class ResetBody(BaseModel):
    task_id: str | None = None


class StepBody(BaseModel):
    action: dict[str, Any] | None = None


class GraderBody(BaseModel):
    task_id: str = "task_easy"
    final_config: dict[str, Any] = Field(default_factory=dict)
    episode: dict[str, Any] | None = None


def _score_for_task(task_id: str, final_config: dict[str, Any] | None = None, episode: dict[str, Any] | None = None) -> float:
    cfg = final_config or {}
    if not cfg:
        st = get_env().state()
        if st.get("task_id") == task_id:
            cfg = st.get("config", {}) or {}
            if episode is None:
                episode = {"actions": st.get("episode_actions", [])}
    return float(grade_episode(task_id, cfg, episode))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "OpenEnv RAG Pipeline Debugger",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "tasks": "/tasks",
        "state": "/state",
    }


@app.post("/reset")
def reset(body: ResetBody | None = None) -> dict[str, Any]:
    tid = body.task_id if body else None
    obs = get_env().reset(task_id=tid)
    return json.loads(obs.model_dump_json())


@app.post("/step")
def step_route(body: StepBody | None = None) -> dict[str, Any]:
    env = get_env()
    if body is None or body.action is None:
        obs, r, done, info = env.step(None)
    else:
        try:
            act = Action.model_validate(body.action)
        except ValidationError as e:
            obs = env.observe()
            return {
                "observation": json.loads(obs.model_dump_json()),
                "reward": -0.1,
                "done": False,
                "info": {"error": "validation", "detail": str(e)},
            }
        obs, r, done, info = env.step(act)
    return {
        "observation": json.loads(obs.model_dump_json()),
        "reward": float(r),
        "done": bool(done),
        "info": info,
    }


@app.get("/state")
def state_route() -> dict[str, Any]:
    return get_env().state()


@app.get("/tasks")
def tasks_route() -> dict[str, Any]:
    global _tasks_payload_cache
    if _tasks_payload_cache is None:
        _tasks_payload_cache = list_tasks_payload()
    return {"tasks": _tasks_payload_cache}


@app.post("/grader")
def grader_route(body: GraderBody | None = None) -> dict[str, float]:
    if body is None:
        return {"score": MIN_SCORE}
    try:
        return {"score": _score_for_task(body.task_id, body.final_config, body.episode)}
    except Exception:
        return {"score": MIN_SCORE}


@app.get("/grade/{task_id}")
@app.post("/grade/{task_id}")
async def grade_task_route(task_id: str, request: Request) -> dict[str, Any]:
    try:
        payload: dict[str, Any] = {}
        if request.method == "POST":
            try:
                payload = await request.json()
            except Exception:
                payload = {}
        final_config = payload.get("final_config") or payload.get("config") or {}
        episode = payload.get("episode")
        score = _score_for_task(task_id, final_config, episode)
        return {"task_id": task_id, "score": score}
    except Exception:
        return {"task_id": task_id, "score": MIN_SCORE}


@app.post("/baseline")
def baseline_route() -> dict[str, Any]:
    """
    Runs a fast deterministic heuristic (no network) so OPENAI_API_KEY is optional.
    If OPENAI_API_KEY is set, still uses the same heuristic for reproducibility & speed.
    """
    t0 = time.time()
    if not os.environ.get("OPENAI_API_KEY"):
        pass  # optional per guide — script may warn; we succeed without it

    scores: dict[str, float] = {}
    for tid in ("task_easy", "task_medium", "task_hard"):
        e = RAGPipelineEnv()
        e.reset(task_id=tid)
        if tid == "task_easy":
            e.step(Action(action_type="configure", payload={"chunk_size": 500}))
            e.step(Action(action_type="reindex", payload={}))
            e.step(Action(action_type="submit", payload={}))
        elif tid == "task_medium":
            e.step(
                Action(
                    action_type="configure",
                    payload={
                        "embedding_model": "text-embedding-3-small",
                        "query_embedding_model": "text-embedding-3-small",
                    },
                )
            )
            e.step(Action(action_type="reindex", payload={}))
            e.step(Action(action_type="submit", payload={}))
        else:
            e.step(Action(action_type="configure", payload={"top_k": 3, "rerank_enabled": True}))
            e.step(Action(action_type="submit", payload={}))
        st = e.state()
        g = grade_episode(tid, st.get("config", {}), {"actions": st.get("episode_actions", [])})
        scores[tid] = float(g)

    elapsed = time.time() - t0
    if elapsed > 60:
        raise HTTPException(status_code=504, detail="baseline timeout")
    return {"scores": scores, "elapsed_sec": round(elapsed, 3)}
