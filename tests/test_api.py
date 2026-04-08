from fastapi.testclient import TestClient

from api.server import app

client = TestClient(app)


def test_root() -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_tasks_schema() -> None:
    r = client.get("/tasks")
    assert r.status_code == 200
    tasks = r.json()
    assert len(tasks) >= 3
    assert "action_schema" in tasks[0]
    assert "grader" in tasks[0]


def test_reset_step_submit() -> None:
    client.post("/reset", json={"task_id": "task_easy"})
    r = client.post("/step", json={"action": {"action_type": "configure", "payload": {"chunk_size": 500}}})
    assert r.status_code == 200
    assert 0.0 < r.json()["reward"]["score"] < 1.0


def test_baseline_fast() -> None:
    r = client.post("/baseline")
    assert r.status_code == 200
    body = r.json()
    assert "scores" in body
    assert set(body["scores"].keys()) == {"task_easy", "task_medium", "task_hard"}
    assert all(0.0 < v < 1.0 for v in body["scores"].values())
