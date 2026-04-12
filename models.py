"""Pydantic models required by the OpenEnv hackathon spec."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Observation(BaseModel):
    task_id: str
    task_description: str
    current_context: dict[str, Any]
    step_count: int
    difficulty: str = "easy"
    max_steps: int = 20
    hints_used: int = 0
    previous_actions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    action_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    score: float
    breakdown: dict[str, Any]
    feedback: str


class EpisodeRecord(BaseModel):
    task_id: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    final_config: dict[str, Any] = Field(default_factory=dict)
    actions: list[dict[str, Any]] = Field(default_factory=list)
