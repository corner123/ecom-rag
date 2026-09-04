"""Typed business constraints accepted by the reviewed schema registry."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QueryConstraints(BaseModel):
    """Business fields requested by a future SQL planner, never raw SQL."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    metrics: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
