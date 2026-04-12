from graders import grade_action_dummy, grade_episode


def test_grader_malformed() -> None:
    assert 0.0 < grade_action_dummy("task_easy", None) < 1.0


def test_easy_perfect() -> None:
    s = grade_episode(
        "easy_chunk_alignment",
        {
            "chunk_size": 450,
            "reindex_completed": True,
            "retrieved_preview_ids": [
                "doc_runbook_api_rotation",
                "doc_webhook_replay_guardrails",
                "doc_release_notes",
            ],
        },
    )
    assert 0.0 < s < 1.0


def test_medium_partial() -> None:
    s = grade_episode(
        "medium_embedding_migration",
        {
            "embedding_model": "text-embedding-3-small",
            "query_embedding_model": "x",
            "reindex_completed": True,
            "retrieved_preview_ids": ["doc_signature_rotation"],
        },
    )
    assert 0.0 < s < 1.0


def test_hard_deterministic() -> None:
    cfg = {
        "top_k": 3,
        "rerank_enabled": True,
        "context_overflow_detected": False,
        "retrieved_preview_ids": [
            "doc_write_path_workaround",
            "doc_customer_comms_template",
            "doc_failover_postmortem",
        ],
    }
    a = grade_episode("hard_context_overflow", cfg)
    b = grade_episode("hard_context_overflow", cfg)
    assert a == b
    assert 0.0 < a < 1.0
