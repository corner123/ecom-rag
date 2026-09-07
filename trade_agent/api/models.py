"""Public request and response contracts for the v1 HTTP API."""
from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    StringConstraints,
    model_validator,
)

from trade_agent.evidence.models import Claim, Conflict, Evidence
from trade_agent.retrieval.filters import RetrievalFilter
from trade_agent.retrieval.planner import RetrievalPlan
from trade_agent.schemas.source import ChunkRecord


Question = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=2_000),
]
RunId = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"),
]
IdempotencyKey = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"),
]
class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class QueryRequest(_StrictModel):
    question: Question
    top_k: StrictInt = Field(default=10, ge=1, le=100)
    explicit_filters: RetrievalFilter = Field(default_factory=RetrievalFilter)
    idempotency_key: IdempotencyKey | None = None


class RetrieveRequest(_StrictModel):
    question: Question
    top_k: StrictInt = Field(default=10, ge=1, le=100)
    explicit_filters: RetrievalFilter = Field(default_factory=RetrievalFilter)


class ResumeRequest(_StrictModel):
    """Reserved strict body; resume scope comes only from the stored request."""


class ReadinessCheck(_StrictModel):
    ok: StrictBool
    code: StrictStr | None = None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.ok and self.code is not None:
            raise ValueError("successful readiness checks cannot carry an error code")
        if not self.ok and (self.code is None or not self.code.strip()):
            raise ValueError("failed readiness checks require an error code")
        return self


class ReadinessResponse(_StrictModel):
    ready: StrictBool
    build_id: StrictStr | None
    schema_fingerprint: StrictStr | None
    checks: dict[StrictStr, ReadinessCheck]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if not self.checks:
            raise ValueError("readiness requires dependency checks")
        expected = all(item.ok for item in self.checks.values())
        if self.ready is not expected:
            raise ValueError("ready must equal the conjunction of dependency checks")
        if self.ready and (self.build_id is None or self.schema_fingerprint is None):
            raise ValueError("ready requires live schema and build identities")
        return self


class PublicError(_StrictModel):
    code: StrictStr = Field(min_length=1, max_length=128)
    node: StrictStr = Field(min_length=1, max_length=128)
    retryable: StrictBool


class QueryResponse(_StrictModel):
    run_id: RunId
    status: Literal["completed", "refused", "failed"]
    build_id: StrictStr
    answer: StrictStr | None
    claims: tuple[Claim, ...]
    evidence: tuple[Evidence, ...]
    conflicts: tuple[Conflict, ...]
    refusal_reason: StrictStr | None
    errors: tuple[PublicError, ...]
    degraded_components: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def guarded_public_projection(self) -> Self:
        if not self.build_id.startswith("build_") or len(self.build_id) != 38:
            raise ValueError("query response requires a published build identity")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if evidence_ids != tuple(sorted(set(evidence_ids))):
            raise ValueError("public Evidence must be sorted and unique")
        known = set(evidence_ids)
        cited = {item for claim in self.claims for item in claim.evidence_ids}
        cited.update(item for conflict in self.conflicts for item in conflict.evidence_ids)
        if cited != known:
            raise ValueError("public Evidence must be exactly the cited set")
        if self.status == "completed":
            if not self.claims or any(claim.status != "supported" for claim in self.claims):
                raise ValueError("completed answers require only guarded supported claims")
            reconstructed = "\n".join(claim.text for claim in self.claims)
            if self.answer != reconstructed or self.refusal_reason is not None:
                raise ValueError("answer must be reconstructed from guarded claims")
        elif self.answer is not None or self.claims:
            raise ValueError("refused and failed responses cannot expose answer claims")
        elif self.refusal_reason is None or not self.refusal_reason.strip():
            raise ValueError("refused and failed responses require a reason")
        if self.degraded_components != tuple(sorted(set(self.degraded_components))):
            raise ValueError("degraded components must be sorted and unique")
        return self


class RetrievalTrace(_StrictModel):
    build_id: StrictStr
    profile_id: StrictStr
    profile_version: StrictStr
    planner_version: StrictStr
    filter_expression_version: StrictStr
    component_ranks: dict[StrictStr, StrictInt]
    fusion_score: float
    source_prior: float
    rerank_score: float | None
    degradation: tuple[StrictStr, ...]


class RetrievalHitResponse(_StrictModel):
    chunk_id: StrictStr
    rank: StrictInt = Field(ge=1)
    record: ChunkRecord
    trace: RetrievalTrace


class RetrievalResponse(_StrictModel):
    build_id: StrictStr
    query: StrictStr
    plan: RetrievalPlan
    hits: tuple[RetrievalHitResponse, ...]
    degradation: tuple[StrictStr, ...]
    ranking_only: Literal[True] = True


class ErrorDetail(_StrictModel):
    code: StrictStr


class ErrorResponse(_StrictModel):
    detail: ErrorDetail
