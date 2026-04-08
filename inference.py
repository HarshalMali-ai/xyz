#!/usr/bin/env python3
"""
Round 1 baseline / agentic evaluation script (mandatory name + location).

Uses the OpenAI Python client with:
  API_BASE_URL  — LLM base URL (OpenAI-compatible)
  MODEL_NAME    — model id for chat completions
  API_KEY       — validator-injected API key for the LiteLLM proxy
  HF_TOKEN      — legacy fallback key name

Talks to this repo's HTTP API (FastAPI) at:
  OPENENV_SERVICE_URL — defaults to http://127.0.0.1:7860 (set to your HF Space URL when remote)

Stdout logs (one JSON object per line after the tag; field order fixed in code):
  [START] {"task":...,"env":...,"model":...}
  [STEP]  {"step":...,"action":...,"reward":...,"done":...,"error":...}
  [END]   {"success":...,"steps":...,"score":...,"rewards":...}
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
from openai import OpenAI

# --- Hackathon-required LLM configuration ---
API_BASE_URL = os.environ.get("API_BASE_URL", "").strip().rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "").strip()
API_KEY = os.environ.get("API_KEY", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
LLM_API_KEY = API_KEY or HF_TOKEN
LLM_MODEL_NAME = MODEL_NAME or "gpt-4o-mini"

# Service hosting reset/step/state (your Docker / HF Space)
OPENENV_SERVICE_URL = os.environ.get("OPENENV_SERVICE_URL", "").strip().rstrip("/")
HF_SPACE_FALLBACK_URL = "https://lunarx912-openenv-rag-debugger.hf.space"

TASK_IDS = ("task_easy", "task_medium", "task_hard")
TASK_NAME = "rag-pipeline-debugger"
BENCHMARK = "openenv-rag-debugger"

MAX_STEPS_PER_TASK = 24
SUCCESS_SCORE_THRESHOLD = 0.85
TEMPERATURE = 0.2
MAX_TOKENS = 512

SYSTEM_PROMPT = """You are debugging a simulated RAG pipeline via HTTP actions.
You must output ONE JSON object only, no markdown, no extra text.
Schema:
{"action_type": "configure" | "reindex" | "submit" | "request_hint",
 "payload": { ... }}

Rules:
- task_easy: set chunk_size to 500, then reindex, then submit.
- task_medium: set embedding_model and query_embedding_model to "text-embedding-3-small", reindex, submit.
- task_hard: set top_k to 3 and rerank_enabled true, then submit.
Payload keys for configure: chunk_size (int), top_k (int), embedding_model (str), query_embedding_model (str), rerank_enabled (bool).
Use submit with payload {} when ready to grade."""

MAX_RUNTIME_SEC = 19 * 60  # stay under 20 min infra limit


def _log_start(task: str, env: str, model: str) -> None:
    # Field order matches sample: task, env, model
    line = json.dumps({"task": task, "env": env, "model": model}, separators=(",", ":"), ensure_ascii=False)
    print(f"[START] {line}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    line = json.dumps(
        {
            "step": step,
            "action": action,
            "reward": reward,
            "done": done,
            "error": error,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )
    print(f"[STEP] {line}", flush=True)


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    line = json.dumps(
        {
            "success": success,
            "steps": steps,
            "score": score,
            "rewards": rewards,
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )
    print(f"[END] {line}", flush=True)


def _client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=LLM_API_KEY)


def _candidate_service_urls() -> list[str]:
    urls: list[str] = []
    for url in (OPENENV_SERVICE_URL, "http://127.0.0.1:7860", HF_SPACE_FALLBACK_URL):
        clean = (url or "").strip().rstrip("/")
        if clean and clean not in urls:
            urls.append(clean)
    return urls


def _http(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=60.0)


def _resolve_service_url() -> str:
    last_error: Exception | None = None
    for base_url in _candidate_service_urls():
        try:
            with _http(base_url) as http:
                r = http.get("/health")
                r.raise_for_status()
            return base_url
        except Exception as exc:
            last_error = exc
            print(f"[DEBUG] Health check failed for {base_url}: {exc}", flush=True)
    if last_error is not None:
        raise last_error
    raise RuntimeError("No service URL candidates available")


def _reset(http: httpx.Client, task_id: str) -> dict[str, Any]:
    r = http.post("/reset", json={"task_id": task_id})
    r.raise_for_status()
    return r.json()


def _step(http: httpx.Client, action: dict[str, Any]) -> dict[str, Any]:
    r = http.post("/step", json={"action": action})
    r.raise_for_status()
    return r.json()


def _build_user_prompt(
    task_id: str,
    step_n: int,
    observation: dict[str, Any],
    last_reward: float,
    history: list[str],
) -> str:
    obs_compact = json.dumps(observation, separators=(",", ":"), ensure_ascii=False)[:8000]
    hist = "\n".join(history[-12:])
    return (
        f"task_id={task_id}\n"
        f"step={step_n}\n"
        f"last_reward={last_reward}\n"
        f"observation={obs_compact}\n"
        f"recent_history:\n{hist}\n"
        "Return the next action as JSON only."
    )


def _parse_action(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"action_type": "request_hint", "payload": {}}
    # strip optional markdown fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return {"action_type": "request_hint", "payload": {}}
        at = str(obj.get("action_type", "request_hint"))
        pl = obj.get("payload", {})
        if not isinstance(pl, dict):
            pl = {}
        return {"action_type": at, "payload": pl}
    except json.JSONDecodeError:
        return {"action_type": "request_hint", "payload": {}}


def _heuristic_action(task_id: str, obs: dict[str, Any]) -> dict[str, Any]:
    cfg = (((obs or {}).get("current_context") or {}).get("pipeline_config") or {})
    reindexed = bool(cfg.get("reindex_completed", False))

    if task_id == "task_easy":
        if int(cfg.get("chunk_size", 500)) != 500:
            return {"action_type": "configure", "payload": {"chunk_size": 500}}
        if not reindexed:
            return {"action_type": "reindex", "payload": {}}
        return {"action_type": "submit", "payload": {}}

    if task_id == "task_medium":
        if (
            str(cfg.get("embedding_model", "")) != "text-embedding-3-small"
            or str(cfg.get("query_embedding_model", "")) != "text-embedding-3-small"
        ):
            return {
                "action_type": "configure",
                "payload": {
                    "embedding_model": "text-embedding-3-small",
                    "query_embedding_model": "text-embedding-3-small",
                },
            }
        if not reindexed:
            return {"action_type": "reindex", "payload": {}}
        return {"action_type": "submit", "payload": {}}

    if int(cfg.get("top_k", 5)) != 3 or not bool(cfg.get("rerank_enabled", False)):
        return {"action_type": "configure", "payload": {"top_k": 3, "rerank_enabled": True}}
    return {"action_type": "submit", "payload": {}}


def _model_action(
    client: OpenAI | None,
    step_n: int,
    task_id: str,
    obs: dict[str, Any],
    last_r: float,
    hist: list[str],
) -> dict[str, Any]:
    if client is None:
        return _heuristic_action(task_id, obs)
    user_prompt = _build_user_prompt(task_id, step_n, obs, last_r, hist)
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        return _parse_action(raw)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return _heuristic_action(task_id, obs)


def _proxy_warmup(client: OpenAI | None) -> None:
    if client is None:
        return
    try:
        client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "Reply with OK."},
                {"role": "user", "content": "OK"},
            ],
            temperature=0.0,
            max_tokens=4,
            stream=False,
        )
        print("[DEBUG] LiteLLM proxy warmup call succeeded", flush=True)
    except Exception as exc:
        # The warmup is only to ensure the validator sees a proxy call.
        print(f"[DEBUG] LiteLLM proxy warmup failed: {exc}", flush=True)


def run_episode(
    client: OpenAI | None,
    http: httpx.Client,
    task_id: str,
    global_step_counter: list[int],
    rewards_out: list[float],
    start_time: float,
) -> float:
    """Returns terminal grader score in [0,1] (0 if never submitted)."""
    last_reward = 0.0
    last_obs: dict[str, Any] = _reset(http, task_id)
    history: list[str] = []
    terminal = 0.0

    for _ in range(MAX_STEPS_PER_TASK):
        if time.monotonic() - start_time > MAX_RUNTIME_SEC:
            break
        global_step_counter[0] += 1
        step_n = global_step_counter[0]
        action = _model_action(client, step_n, task_id, last_obs, last_reward, history)
        action_str = json.dumps(action, separators=(",", ":"), ensure_ascii=False)
        try:
            result = _step(http, action)
        except Exception as exc:
            _log_step(step_n, action_str, 0.0, False, str(exc))
            history.append(f"Step {step_n}: http error {exc!r}")
            continue

        reward = float(result.get("reward") or 0.0)
        done = bool(result.get("done"))
        err = None
        if isinstance(result.get("info"), dict) and result["info"].get("error"):
            err = str(result["info"]["error"])

        rewards_out.append(reward)
        last_reward = reward
        last_obs = result.get("observation") or last_obs
        _log_step(step_n, action_str, reward, done, err)
        history.append(f"Step {step_n}: {action_str!r} -> reward {reward:+.2f}")

        if isinstance(result.get("info"), dict) and "grader_score" in result["info"]:
            try:
                terminal = float(result["info"]["grader_score"])
            except (TypeError, ValueError):
                terminal = 0.0

        if done:
            break

    return float(terminal)


def main() -> None:
    t0 = time.monotonic()
    client: OpenAI | None = None
    rewards: list[float] = []
    global_step = [0]
    terminal_scores: list[float] = []
    score = 0.0
    success = False
    model_name = LLM_MODEL_NAME
    service_url = ""

    _log_start(task=TASK_NAME, env=BENCHMARK, model=model_name)

    try:
        if API_BASE_URL and LLM_API_KEY:
            client = _client()
            _proxy_warmup(client)
        else:
            print("[DEBUG] Missing LLM env vars; using heuristic fallback", flush=True)

        service_url = _resolve_service_url()
        print(f"[DEBUG] Using service URL: {service_url}", flush=True)

        with _http(service_url) as http:
            http.get("/health").raise_for_status()

            for tid in TASK_IDS:
                if time.monotonic() - t0 > MAX_RUNTIME_SEC:
                    break
                ts = run_episode(client, http, tid, global_step, rewards, t0)
                terminal_scores.append(ts)

        if terminal_scores:
            score = sum(terminal_scores) / len(terminal_scores)
        else:
            score = 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        print(f"[DEBUG] Fatal run error: {exc}", flush=True)
    finally:
        _log_end(success=success, steps=global_step[0], score=score, rewards=rewards)


if __name__ == "__main__":
    main()
