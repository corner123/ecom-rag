"""Checkpoint-safe contracts for the bounded SQL and RAG graph."""
from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator


class EvidenceRef(BaseModel):
    """Content-addressed pointer to an Evidence payload outside the checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    evidence_id: StrictStr = Field(pattern=r"^(?:sql|rag)_[0-9a-f]{64}$")
    payload_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["stored", "validated", "excluded"] = "stored"


class RoutePlan(BaseModel):
    """Deterministic execution channels selected from a validated intent."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    need_trade_data: StrictBool
    need_external_intel: StrictBool
    refuse: StrictBool = False
    refusal_reason: StrictStr | None = None
    required_branches: tuple[Literal["sql", "rag"], ...] = ()

    @model_validator(mode="after")
    def coherent(self) -> "RoutePlan":
        expected = tuple(
            branch
            for branch, enabled in (
                ("rag", self.need_external_intel),
                ("sql", self.need_trade_data),
            )
            if enabled
        )
        if self.required_branches != expected:
            raise ValueError("required branches must be sorted and match route flags")
        if self.refuse != (self.refusal_reason is not None):
            raise ValueError("refusal route and reason must agree")
        if self.refuse and expected:
            raise ValueError("a refusal route cannot execute evidence branches")
        return self


def merge_status(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    return {**left, **right}


def merge_step(left: int, right: int) -> int:
    return max(left, right)


class TradeIntelState(TypedDict, total=False):
    """Minimal durable graph state; Evidence bodies are intentionally absent."""

    question: str
    current_question: str
    explicit_filters: dict[str, object]
    intent: dict[str, object]
    route_plan: dict[str, object]
    sql_evidence_refs: list[dict[str, object]]
    rag_evidence_refs: list[dict[str, object]]
    evidence_refs: list[dict[str, object]]
    sql_report: dict[str, object]
    rag_report: dict[str, object]
    branch_reports: list[dict[str, object]]
    conflicts: list[dict[str, object]]
    validation: dict[str, object]
    draft: dict[str, object]
    guard: dict[str, object]
    claims: list[dict[str, object]]
    answer: str | None
    refusal_reason: str | None
    rewrite_count: int
    retry_count: int
    llm_calls: int
    step_count: Annotated[int, merge_step]
    errors: Annotated[list[dict[str, object]], add]
    node_status: Annotated[dict[str, str], merge_status]
    terminal: bool
