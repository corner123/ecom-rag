"""Checkpoint-safe contracts for the bounded SQL and RAG graph."""
from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from trade_agent.evidence.models import Claim


NodeName = Literal[
    "policy_gate",
    "router",
    "sql_node",
    "rag_node",
    "normalize",
    "entity_dedup_conflict",
    "evidence_validator",
    "query_rewrite",
    "answer_draft",
    "claim_guard",
    "finalizer",
]
NodeExecutionStatus = Literal["completed", "failed", "retryable_failed", "skipped"]
_NODE_NAMES = {
    "policy_gate", "router", "sql_node", "rag_node", "normalize",
    "entity_dedup_conflict", "evidence_validator", "query_rewrite",
    "answer_draft", "claim_guard", "finalizer",
}
_NODE_EXECUTION_STATUSES = {"completed", "failed", "retryable_failed", "skipped"}


class EvidenceRef(BaseModel):
    """Content-addressed pointer to an Evidence payload outside the checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    evidence_id: StrictStr = Field(pattern=r"^(?:sql|rag)_[0-9a-f]{64}$")
    payload_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["stored", "validated", "excluded"] = "stored"


class DraftRef(BaseModel):
    """Content-addressed pointer to an untrusted draft outside checkpoint state."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    draft_id: StrictStr = Field(pattern=r"^draft_[0-9a-f]{64}$")
    payload_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["stored"] = "stored"

    @model_validator(mode="after")
    def content_address_matches(self) -> "DraftRef":
        if self.draft_id != f"draft_{self.payload_sha256}":
            raise ValueError("draft identity must match its payload hash")
        return self


class RequestRef(BaseModel):
    """Content-addressed request pointer stored outside Redis checkpoints."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    request_id: StrictStr = Field(pattern=r"^request_[0-9a-f]{64}$")
    payload_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class GuardProjection(BaseModel):
    """Checkpoint-safe result of guarding; untrusted answer text is omitted."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    accepted: StrictBool
    claims: tuple[Claim, ...]
    refusal_reason: StrictStr | None
    error_codes: tuple[StrictStr, ...] = ()

    @model_validator(mode="after")
    def coherent(self) -> "GuardProjection":
        if self.accepted:
            if not self.claims or self.refusal_reason is not None:
                raise ValueError("accepted guard projection requires only retained claims")
        elif self.claims or self.refusal_reason is None or not self.refusal_reason.strip():
            raise ValueError("refused guard projection requires only a refusal reason")
        if self.error_codes != tuple(sorted(set(self.error_codes))):
            raise ValueError("guard error codes must be sorted and unique")
        if any(not value.strip() for value in self.error_codes):
            raise ValueError("guard error codes must not be blank")
        return self


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


def merge_status(
    left: dict[NodeName, NodeExecutionStatus],
    right: dict[NodeName, NodeExecutionStatus],
) -> dict[NodeName, NodeExecutionStatus]:
    merged = {**left, **right}
    if any(name not in _NODE_NAMES for name in merged):
        raise ValueError("node status contains an unknown graph node")
    if any(status not in _NODE_EXECUTION_STATUSES for status in merged.values()):
        raise ValueError("node status contains an unknown execution status")
    return merged


def merge_step(left: int, right: int) -> int:
    return max(left, right)


class TradeIntelInput(TypedDict, total=False):
    """Only caller-controlled fields admitted at the graph boundary."""

    question: str
    explicit_filters: dict[str, object]
    idempotency_key: str


class TradeIntelState(TypedDict, total=False):
    """Minimal durable graph state; Evidence bodies are intentionally absent."""

    question: str
    idempotency_key: str
    current_question: str
    explicit_filters: dict[str, object]
    intent: dict[str, object]
    request_ref: dict[str, object]
    route_plan: dict[str, object]
    sql_evidence_refs: list[dict[str, object]]
    rag_evidence_refs: list[dict[str, object]]
    evidence_refs: list[dict[str, object]]
    sql_report: dict[str, object]
    rag_report: dict[str, object]
    branch_reports: list[dict[str, object]]
    conflicts: list[dict[str, object]]
    validation: dict[str, object]
    draft_ref: dict[str, object]
    guard: dict[str, object]
    claims: list[dict[str, object]]
    answer: str | None
    refusal_reason: str | None
    rewrite_count: int
    retry_count: int
    llm_calls: int
    step_count: Annotated[int, merge_step]
    errors: Annotated[list[dict[str, object]], add]
    node_status: Annotated[dict[NodeName, NodeExecutionStatus], merge_status]
    terminal: bool
