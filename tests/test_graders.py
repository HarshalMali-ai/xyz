from graders import grade_action_dummy, grade_episode


def test_grader_malformed() -> None:
    assert grade_action_dummy("task_easy", None) == 0.0


def test_easy_perfect() -> None:
    s = grade_episode(
        "task_easy",
        {"chunk_size": 500, "reindex_completed": True},
    )
    assert s == 1.0


def test_medium_partial() -> None:
    s = grade_episode(
        "task_medium",
        {"embedding_model": "text-embedding-3-small", "query_embedding_model": "x", "reindex_completed": True},
    )
    assert 0.0 < s < 1.0


def test_hard_deterministic() -> None:
    cfg = {"top_k": 3, "rerank_enabled": True, "context_overflow_detected": False}
    a = grade_episode("task_hard", cfg)
    b = grade_episode("task_hard", cfg)
    assert a == b == 1.0
