"""Task definitions and metadata for openenv.yaml alignment."""

from __future__ import annotations

from typing import Any

TASK_SPECS: dict[str, dict[str, Any]] = {
    "task_easy": {
        "difficulty": "easy",
        "description": (
            "Fix chunking: documents are ~500 tokens each but chunk_size is set to 2000, "
            "so retrieval returns wrong spans. Set chunk_size to the correct value."
        ),
        "grader": {
            "type": "programmatic",
            "endpoint": "/grader",
            "method": "POST",
        },
        "action_schema": {
            "configure": {
                "payload": {
                    "chunk_size": "int | null",
                    "embedding_model": "str | null",
                    "query_embedding_model": "str | null",
                    "top_k": "int | null",
                    "rerank_enabled": "bool | null",
                }
            },
            "reindex": {"payload": {}},
            "submit": {"payload": {}},
            "request_hint": {"payload": {}},
        },
    },
    "task_medium": {
        "difficulty": "medium",
        "description": (
            "Fix embedding mismatch: the vector index was built with one embedding model "
            "but queries use another. Align models and re-index."
        ),
        "grader": {
            "type": "programmatic",
            "endpoint": "/grader",
            "method": "POST",
        },
        "action_schema": {
            "configure": {
                "payload": {
                    "chunk_size": "int | null",
                    "embedding_model": "str | null",
                    "query_embedding_model": "str | null",
                    "top_k": "int | null",
                    "rerank_enabled": "bool | null",
                }
            },
            "reindex": {"payload": {}},
            "submit": {"payload": {}},
            "request_hint": {"payload": {}},
        },
    },
    "task_hard": {
        "difficulty": "hard",
        "description": (
            "Reduce context overflow: top_k is too high and reranking is off, so the LLM "
            "context is flooded and answers degrade. Lower top_k and enable reranking."
        ),
        "grader": {
            "type": "programmatic",
            "endpoint": "/grader",
            "method": "POST",
        },
        "action_schema": {
            "configure": {
                "payload": {
                    "chunk_size": "int | null",
                    "embedding_model": "str | null",
                    "query_embedding_model": "str | null",
                    "top_k": "int | null",
                    "rerank_enabled": "bool | null",
                }
            },
            "reindex": {"payload": {}},
            "submit": {"payload": {}},
            "request_hint": {"payload": {}},
        },
    },
}


def list_tasks_payload() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tid, spec in TASK_SPECS.items():
        out.append(
            {
                "id": tid,
                "difficulty": spec["difficulty"],
                "description": spec["description"],
                "grader": spec["grader"],
                "action_schema": spec["action_schema"],
            }
        )
    return out
