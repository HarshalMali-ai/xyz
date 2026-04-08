import pytest

from environment.rag_environment import RAGPipelineEnv
from models import Action


def test_reset_and_state_before_observe() -> None:
    e = RAGPipelineEnv()
    s = e.state()
    assert s["initialized"] is False


def test_episode_easy_success() -> None:
    e = RAGPipelineEnv()
    e.reset(task_id="task_easy")
    e.step(Action(action_type="configure", payload={"chunk_size": 500}))
    e.step(Action(action_type="reindex", payload={}))
    o, r, done, info = e.step(Action(action_type="submit", payload={}))
    assert done is True
    assert "grader_score" in info
    assert 0.0 < info["grader_score"] < 1.0


def test_null_action() -> None:
    e = RAGPipelineEnv()
    e.reset(task_id="task_easy")
    o, r, done, info = e.step(None)
    assert r == -0.1
    assert done is False


def test_max_steps() -> None:
    e = RAGPipelineEnv()
    e.reset(task_id="task_easy")
    last_done = False
    for _ in range(30):
        o, r, done, info = e.step(Action(action_type="request_hint", payload={}))
        last_done = done
        if done:
            break
    assert last_done is True
