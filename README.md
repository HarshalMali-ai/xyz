---
title: OpenEnv RAG Pipeline Debugger
emoji: "🧪"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# OpenEnv — RAG Pipeline Debugger

Hackathon Round 1 environment: an agent debugs a **simulated** RAG stack (chunking, embedding alignment, retrieval flooding). **Graders are programmatic** (no LLM-as-judge). Final scoring uses **retrieval fingerprints** and overflow signals from the simulator, not only raw config equality.

## What this satisfies (Round 1 checklist)

| Requirement | This repo |
|-------------|-----------|
| Real-world task (not a game) | RAG / retrieval pipeline debugging |
| `openenv.yaml` + metadata | Root `openenv.yaml` |
| ≥ 3 tasks easy → hard | `task_easy`, `task_medium`, `task_hard` |
| Programmatic graders in `[0, 1]` | `graders.py` |
| Meaningful dense rewards | `reward.py` + step penalties in `environment/rag_environment.py` |
| FastAPI + Docker + HF Space | `api/server.py`, `Dockerfile`, port **7860** |
| **Mandatory `inference.py` in repo root** | Uses **OpenAI** client + `[START]` / `[STEP]` / `[END]` logs |
| Baseline / reproducibility | `scripts/baseline.py` (heuristic oracle) + `inference.py` (LLM-driven) |

**Pre-validation (organizer script):** install the framework they specify, e.g. `pip install openenv-core`, then run `openenv validate` from the repo root. If `openenv` is not on PATH, use the exact install command from the hackathon materials.

**Submission:** only the **team lead** can submit; Space must be tagged **`openenv`**.

## Layout

- `inference.py` — **required** LLM baseline (env vars below; structured stdout)
- `models.py` — `Observation`, `Action`, `Reward` (Pydantic)
- `tasks.py` — task metadata + action schemas for `GET /tasks`
- `graders.py` — scores in `[0, 1]`
- `reward.py` — dense step rewards in `[-1, 1]` (API `step` returns float reward)
- `dataset/*.json` — ground truth (including expected `retrieved_fingerprint` where used)
- `environment/rag_environment.py` — env logic; syncs simulation artifacts into config for grading
- `api/server.py` — FastAPI: `/health`, `/reset`, `/step`, `/state`, `/tasks`, `/baseline`, `/grader`
- `scripts/baseline.py` — fast deterministic oracle (expects `OPENAI_API_KEY` for checklist-style smoke)
- `scripts/local_validate.py` — local structural checks if `openenv validate` is unavailable
- `openenv.yaml` — environment metadata
- `Dockerfile` — `python:3.11-slim`, `uvicorn` on `0.0.0.0:7860`

## Observation & action space

**Observation** (`Observation`): `task_id`, `task_description`, `current_context` (dict), `step_count`.

`current_context` includes:

- `pipeline_config` — chunking / embedding / top_k / rerank / reindex flags
- `simulation` — retrieval preview, `retrieved_fingerprint`, overflow flags, token estimates

**Action** (`Action`): `action_type` ∈ `configure` | `reindex` | `submit` | `request_hint`, plus `payload` (dict).

Typical payloads:

- `configure`: `chunk_size`, `top_k`, `embedding_model`, `query_embedding_model`, `rerank_enabled`
- `reindex`: `{}`
- `submit`: `{}`

## Mandatory environment variables (inference / LLM)

Used by **`inference.py`** (OpenAI-compatible client):

| Variable | Purpose |
|----------|---------|
| `API_BASE_URL` | LLM base URL (e.g. `https://api.openai.com/v1` or your router) |
| `MODEL_NAME` | Chat model id (e.g. `gpt-4o-mini`) |
| `HF_TOKEN` | API key passed to `OpenAI(api_key=...)` (per hackathon naming) |

**Service URL** for this environment’s HTTP API (local or HF Space):

| Variable | Default |
|----------|---------|
| `OPENENV_SERVICE_URL` | `http://127.0.0.1:7860` |

Example (PowerShell, local server running):

```powershell
$env:API_BASE_URL = "https://api.openai.com/v1"
$env:MODEL_NAME = "gpt-4o-mini"
$env:HF_TOKEN = "sk-..."   # your key
$env:OPENENV_SERVICE_URL = "http://127.0.0.1:7860"
python inference.py
```

**Stdout format** (one JSON object per line after the tag; stable key order in code):

- `[START] {"task":"...","env":"...","model":"..."}`
- `[STEP] {"step":n,"action":"...","reward":r,"done":bool,"error":null}`
- `[END] {"success":bool,"steps":n,"score":0-1,"rewards":[...]}`

Aggregate `score` is the mean of terminal **grader** scores over the three tasks. Runtime is capped under **20 minutes** for infra limits.

## Install (local)

```powershell
cd C:\Users\himma\Documents\openenv-rag-debugger
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `Activate.ps1` is blocked, call the venv Python by full path (as above).

## Run the API

```powershell
cd C:\Users\himma\Documents\openenv-rag-debugger
.\.venv\Scripts\python.exe -m uvicorn api.server:app --host 0.0.0.0 --port 7860
```

Smoke checks:

- `GET http://127.0.0.1:7860/health`
- `POST http://127.0.0.1:7860/baseline`

## Deterministic oracle (no LLM)

```powershell
$env:OPENAI_API_KEY = "sk-local-test"
python scripts/baseline.py
```

## Tests & local validation

```powershell
python -m pytest
python scripts/local_validate.py .
```

## Docker

```powershell
cd C:\Users\himma\Documents\openenv-rag-debugger
docker build -t rag-openenv .
docker run --rm -p 7860:7860 rag-openenv
```

## Hugging Face Space (Docker)

1. Push this repo to GitHub.
2. New Space → **Docker** SDK → connect repo; tag **`openenv`**.
3. After build: `https://<your-space>.hf.space/health` and `/baseline`.

For `inference.py` against the Space, set `OPENENV_SERVICE_URL` to your Space URL (no trailing slash path issues).

## Oracle policy (for agents / grading)

| Task | Fix sequence |
|------|----------------|
| `task_easy` | `configure` `chunk_size: 500` → `reindex` → `submit` |
| `task_medium` | both embedding fields `text-embedding-3-small` → `reindex` → `submit` |
| `task_hard` | `configure` `top_k: 3`, `rerank_enabled: true` → `submit` |

## Motivation

Developers misconfigure RAG daily: chunking that does not match content, index/query embedding skew, and retrieval that floods the context window. This environment is a **deterministic** testbed for agent evaluation with **typed** actions, **HTTP** deployment, and **non-LLM** grading—aligned with production debugging workflows.
