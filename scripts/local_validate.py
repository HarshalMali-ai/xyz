from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path


REQUIRED_FILES = [
    "openenv.yaml",
    "Dockerfile",
    "api/server.py",
    "environment/rag_environment.py",
    "models.py",
    "tasks.py",
    "graders.py",
    "reward.py",
]

REQUIRED_ENDPOINTS = [
    ("GET", "/health"),
    ("POST", "/reset"),
    ("POST", "/step"),
    ("GET", "/state"),
    ("GET", "/tasks"),
    ("POST", "/baseline"),
    ("POST", "/grader"),
]


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"OK: {msg}")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def validate_openenv_yaml(project_dir: Path) -> None:
    p = project_dir / "openenv.yaml"
    if not p.exists():
        _fail("openenv.yaml missing")
    text = _read_text(p)

    if not re.search(r"^name:\s*.+$", text, flags=re.M):
        _fail("openenv.yaml missing `name:`")
    if not re.search(r"^tasks:\s*$", text, flags=re.M):
        _fail("openenv.yaml missing `tasks:`")

    task_ids = re.findall(r"^\s*-\s*id:\s*(task_[a-z]+)\s*$", text, flags=re.M)
    if len(task_ids) < 3:
        _fail("openenv.yaml must list at least 3 tasks")

    want = {"task_easy", "task_medium", "task_hard"}
    found = set(task_ids)
    missing = want - found
    if missing:
        _fail(f"openenv.yaml missing task ids: {sorted(missing)}")

    # quick sanity: ensure each task has a difficulty key near it
    if not re.search(r"difficulty:\s*(easy|medium|hard)", text):
        _fail("openenv.yaml missing `difficulty:` values")

    _ok("openenv.yaml structure looks correct")


def validate_files(project_dir: Path) -> None:
    for rel in REQUIRED_FILES:
        p = project_dir / rel
        if not p.exists():
            _fail(f"Required file missing: {rel}")
    _ok("required files exist")


def validate_server_endpoints(project_dir: Path) -> None:
    server_path = project_dir / "api" / "server.py"
    text = _read_text(server_path)

    # Regex checks for typical FastAPI patterns.
    for method, route in REQUIRED_ENDPOINTS:
        route_pat = re.escape(f'@app.{method.lower()}("{route}")')
        if re.search(route_pat, text):
            continue
        # be a little more flexible if formatting differs
        if not re.search(rf"@app\.(get|post)\(.*{re.escape(route)}.*\)", text):
            _fail(f"server.py missing route decorator for {method} {route}")

    _ok("FastAPI routes appear present")


def validate_imports(project_dir: Path) -> None:
    sys.path.insert(0, str(project_dir))

    # Import modules to catch syntax/import errors.
    importlib.import_module("models")
    importlib.import_module("tasks")
    importlib.import_module("graders")
    importlib.import_module("reward")
    importlib.import_module("environment.rag_environment")
    importlib.import_module("api.server")

    _ok("Python imports succeed")


def validate_graders_deterministic(project_dir: Path) -> None:
    sys.path.insert(0, str(project_dir))
    from graders import grade_episode

    cfg_easy = {"chunk_size": 500, "reindex_completed": True}
    a = grade_episode("task_easy", cfg_easy)
    b = grade_episode("task_easy", cfg_easy)
    if a != b:
        _fail("graders are not deterministic (task_easy mismatch)")
    if not (0.0 <= a <= 1.0):
        _fail("grader returned out-of-range score")

    _ok("graders deterministic + in range")


def main() -> None:
    project_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
    if not project_dir.exists():
        _fail(f"Directory not found: {project_dir}")

    validate_openenv_yaml(project_dir)
    validate_files(project_dir)
    validate_server_endpoints(project_dir)
    validate_imports(project_dir)
    validate_graders_deterministic(project_dir)

    print("\nLocal validate: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

