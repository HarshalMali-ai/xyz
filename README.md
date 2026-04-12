---
title: OpenEnv RAG Pipeline Debugger
emoji: "🧪"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# OpenEnv RAG Pipeline Debugger

An OpenEnv-compliant benchmark where an agent debugs production-style retrieval-augmented generation pipelines instead of playing a toy game. The environment focuses on the kinds of failures real LLM platform teams see in practice: bad chunking, partial embedding migrations, disabled rerankers, stale indexes, and context-window overflows during customer-facing incidents.

## Why this benchmark matters

RAG systems fail quietly. A support bot still returns an answer, but the answer is pulled from the wrong runbook, the wrong language, or a giant wall of irrelevant context. Teams usually discover the problem only after a quality incident.

This environment turns that real operational pain into a deterministic training benchmark:

- It models tasks platform teams already perform: inspecting retrieval quality, aligning embeddings, rebuilding indexes, and tuning context budgets.
- It gives agents partial credit for meaningful progress instead of binary pass/fail grading.
- It captures realistic tradeoffs between retrieval quality and context efficiency.
- It stays fully programmatic. No LLM-as-judge logic is needed to score success.

## What is inside

| Property | Value |
|---|---|
| Domain | Production RAG debugging |
| Tasks | 9 total |
| Difficulty ladder | 3 easy, 3 medium, 3 hard |
| Episode style | Dense-reward configuration and incident debugging |
| API | FastAPI on port `7860` |
| Core actions | `configure`, `reindex`, `submit`, `request_hint` |

## Task set

### Easy

- `easy_chunk_alignment` - fix oversized chunks so support runbooks retrieve procedure-sized evidence
- `easy_chunk_overlap` - restore chunk overlap so instructions do not break across chunk boundaries
- `easy_top_k_budget` - trim top-k for short FAQ answers to prevent low-value context flooding

### Medium

- `medium_embedding_migration` - complete a partial embedding migration by aligning index/query models and rebuilding
- `medium_multilingual_embeddings` - repair bilingual retrieval after switching only the query side to a multilingual model
- `medium_rerank_precision` - re-enable reranking so operational runbooks outrank generic policy pages

### Hard

- `hard_context_overflow` - recover a long-context overflow regression after top-k was increased
- `hard_release_migration` - fix release-note retrieval drift caused by index drift and over-retrieval
- `hard_multiknob_repair` - repair a postmortem search stack that needs chunking, overlap, reranking, and context-budget fixes together

## Observation space

Each observation is typed and includes both the system state and the human stakes of the task.

```json
{
  "task_id": "hard_context_overflow",
  "task_description": "A long-context answer chain is flooding the model with too many chunks...",
  "current_context": {
    "operator_story": "The assistant must answer a customer escalation...",
    "business_impact": "Context overflow causes slow, contradictory answers...",
    "user_query": "What customer-safe workaround should we communicate...",
    "acceptance_criteria": ["Context overflow clears", "..."],
    "pipeline_config": {},
    "simulation": {},
    "corpus": {"documents": []}
  },
  "step_count": 0,
  "difficulty": "hard",
  "max_steps": 18,
  "hints_used": 0,
  "previous_actions": [],
  "metadata": {
    "title": "Recover A Long-Context Overflow Regression",
    "reindex_required": false,
    "overflow_sensitive": true
  }
}
```

Ground truth never appears in the observation.

## Action space

The environment intentionally keeps the action set small and operational:

- `configure`
  Update retrieval/index settings such as `chunk_size`, `chunk_overlap`, `top_k`, `embedding_model`, `query_embedding_model`, `rerank_enabled`, or `max_context_tokens`.
- `reindex`
  Rebuild the index after chunking or embedding changes.
- `submit`
  Ask the grader to score the current pipeline state.
- `request_hint`
  Receive the next progressive hint at a small reward penalty.

## Reward design

Rewards are dense and intentionally shaped around operational progress rather than only the terminal grade.

- Progress toward the target configuration increases reward.
- Better retrieval quality increases reward.
- Clearing context overflow increases reward on overflow-sensitive tasks.
- Reindex-required tasks reward the agent for actually rebuilding the index.
- Looping and hint usage reduce reward slightly.
- Final submit rewards are derived from the deterministic grader and clamped inside `(0.01, 0.99)`.

This gives agents a smoother learning signal than a flat 0/1 evaluator.

## Grading logic

Each task is scored programmatically using four components:

1. configuration progress
2. retrieval quality against ideal documents
3. reindex completion when required
4. context-overflow recovery when relevant

All final scores are deterministic and clamped into the open interval `(0.01, 0.99)`.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | liveness check |
| `/reset` | POST | start a fresh episode |
| `/step` | POST | apply one action |
| `/state` | GET | inspect current state |
| `/tasks` | GET | list the full task catalog and action schema |
| `/grader` | POST | score a pipeline state |
| `/grade/{task_id}` | GET/POST | direct task-specific grader endpoint |
| `/baseline` | POST | run the deterministic baseline over flagship tasks |
| `/validate` | GET | return benchmark self-check metadata |

## Baseline behavior

The repo includes two evaluation entrypoints:

- `scripts/baseline.py`
  A deterministic heuristic baseline over the flagship easy/medium/hard scenarios.
- `inference.py`
  The hackathon-required OpenAI-client script with structured `[START]`, `[STEP]`, and `[END]` logs.

The baseline intentionally follows the minimum successful fix sequence for each flagship task:

- easy: correct chunk size, reindex, submit
- medium: align embeddings, reindex, submit
- hard: reduce top-k, enable reranking, submit

## Local setup

```powershell
cd C:\Users\himma\Documents\openenv-rag-debugger
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If Windows blocks activation scripts, call the venv Python directly instead of activating the shell.

## Run locally

```powershell
cd C:\Users\himma\Documents\openenv-rag-debugger
.\.venv\Scripts\python.exe -m uvicorn api.server:app --host 0.0.0.0 --port 7860
```

Fallback without using `.venv`:

```powershell
cd C:\Users\himma\Documents\openenv-rag-debugger
py -3.14 -m uvicorn api.server:app --host 127.0.0.1 --port 7860
```

Useful checks:

```powershell
irm http://127.0.0.1:7860/health
irm http://127.0.0.1:7860/tasks
irm http://127.0.0.1:7860/validate
irm http://127.0.0.1:7860/baseline -Method Post
```

## Run the baseline

```powershell
$env:OPENAI_API_KEY = "sk-local-test"
python scripts/baseline.py
```

## Run the hackathon inference script

```powershell
$env:API_BASE_URL = "https://api.openai.com/v1"
$env:MODEL_NAME = "gpt-4o-mini"
$env:HF_TOKEN = "sk-..."
$env:OPENENV_SERVICE_URL = "http://127.0.0.1:7860"
python inference.py
```

## Docker

```powershell
cd C:\Users\himma\Documents\openenv-rag-debugger
docker build -t rag-pipeline-debugger .
docker run --rm -p 7860:7860 rag-pipeline-debugger
```

## Validation

```powershell
python -m pytest
python scripts/local_validate.py .
```

If `openenv` is installed but not on `PATH`, run:

```powershell
C:\Users\himma\AppData\Local\Python\pythoncore-3.14-64\Scripts\openenv.exe validate
```

## Why this is stronger now

Compared with a single-config toy benchmark, this environment now gives reviewers:

- a broader task catalog
- clearer difficulty progression
- richer observation context with real business pressure
- deterministic graders with meaningful partial credit
- a domain that maps directly to real LLM platform work

That makes it much closer to something a platform team could use for regression testing or RL-style evaluation in practice.
