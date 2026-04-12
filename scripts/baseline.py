"""
Baseline inference script (deterministic heuristic, no network).
Hackathon checklist: read OPENAI_API_KEY from the environment (required to be present).
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from environment.rag_environment import RAGPipelineEnv
from graders import grade_episode
from models import Action
from tasks import flagship_task_ids


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is not set. Export a key (e.g. from https://platform.openai.com/api-keys) "
            "or set a placeholder for local smoke tests: set OPENAI_API_KEY=sk-local-test"
        )

    scores: dict[str, float] = {}
    for tid in flagship_task_ids():
        e = RAGPipelineEnv()
        e.reset(task_id=tid)
        if tid == "easy_chunk_alignment":
            e.step(Action(action_type="configure", payload={"chunk_size": 450}))
            e.step(Action(action_type="reindex", payload={}))
            e.step(Action(action_type="submit", payload={}))
        elif tid == "medium_embedding_migration":
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

    average_score = sum(scores.values()) / max(1, len(scores))
    print("Baseline scores (0.0 - 1.0):")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")
    print(f"  average: {average_score:.3f}")


if __name__ == "__main__":
    main()
