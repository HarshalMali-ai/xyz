# OpenEnv — RAG Pipeline Debugger

Hackathon-style environment: an agent debugs a **simulated** RAG stack (chunking, embedding alignment, retrieval flooding) with **deterministic** graders.

## Layout (matches team guide)

- `models.py` — `Observation`, `Action`, `Reward`
- `tasks.py` — task metadata + action schemas for `/tasks`
- `graders.py` — programmatic scores in `[0, 1]`
- `reward.py` — dense step rewards in `[-1, 1]`
- `dataset/*.json` — ground truth per difficulty
- `environment/rag_environment.py` — `reset`, `step`, `state`
- `api/server.py` — FastAPI: `/health`, `/reset`, `/step`, `/state`, `/tasks`, `/baseline`, `/grader`
- `scripts/baseline.py` — CLI baseline (checks `OPENAI_API_KEY`)
- `openenv.yaml` — environment metadata
- `Dockerfile` — port **7860**, `uvicorn` on `0.0.0.0`

## Prerequisites

- **Python 3.11** (recommended) on PATH as `python`
- Optional: **Docker Desktop** for container runs
- Optional: `pip install openenv` (note: the pip package may not include a `validate` CLI)
- Local validation (works everywhere): `python scripts/local_validate.py .`

## Install (local)

```powershell
cd C:\Users\himma\Documents\openenv-rag-debugger
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the API

```powershell
cd C:\Users\himma\Documents\openenv-rag-debugger
.\.venv\Scripts\Activate.ps1
uvicorn api.server:app --host 0.0.0.0 --port 7860
```

Try:

- `GET http://127.0.0.1:7860/health`
- `POST http://127.0.0.1:7860/reset` with body `{"task_id":"task_easy"}`
- `POST http://127.0.0.1:7860/baseline`

## Baseline script (hackathon checklist)

The script **requires** `OPENAI_API_KEY` to be set (value may be a placeholder for local smoke tests):

```powershell
$env:OPENAI_API_KEY = "sk-local-test"
python scripts/baseline.py
```

## Tests

```powershell
python -m pytest
```

## Local validation

```powershell
python scripts/local_validate.py .
```

## Docker

```powershell
cd C:\Users\himma\Documents\openenv-rag-debugger
docker build -t rag-openenv .
docker run --rm -p 7860:7860 rag-openenv
```

Then open `http://127.0.0.1:7860/health`.

## Task solutions (oracle / baseline)

| Task        | Fix |
|------------|-----|
| `task_easy`   | `configure` `chunk_size: 500` → `reindex` → `submit` |
| `task_medium` | set both embedding models to `text-embedding-3-small` → `reindex` → `submit` |
| `task_hard`   | `configure` `top_k: 3`, `rerank_enabled: true` → `submit` |

## Motivation (README requirement)

Developers routinely misconfigure RAG: chunk sizes that don’t match documents, index/query embedding skew, and retrieval that overfills the context window. This environment gives agents a **reproducible** sandbox with **typed** actions and **non-LLM** grading—suitable for evaluation and training loops.
