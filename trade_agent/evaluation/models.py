"""Strict, immutable contracts for leakage-controlled trade evaluation."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from enum import Enum
from typing import Annotated, Any, Literal, Mapping, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictStr, TypeAdapter, field_validator, model_validator


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_EVIDENCE_ID_PATTERN = r"^(?:rag|sql)_[0-9a-f]{64}$"
_CLAIM_ID_PATTERN = r"^claim_[0-9a-f]{64}$"

Identifier = Annotated[StrictStr, Field(min_length=1, max_length=128, pattern=_ID_PATTERN)]
Hash = Annotated[StrictStr, Field(min_length=64, max_length=64, pattern=_SHA256_PATTERN)]
Question = Annotated[StrictStr, Field(min_length=1, max_length=2_000, pattern=r"\S")]
LabelText = Annotated[StrictStr, Field(min_length=1, max_length=4_000, pattern=r"\S")]
EvidenceId = Annotated[StrictStr, Field(min_length=68, max_length=68, pattern=_EVIDENCE_ID_PATTERN)]
ClaimId = Annotated[StrictStr, Field(min_length=70, max_length=70, pattern=_CLAIM_ID_PATTERN)]


class TaskType(str, Enum):
    """The task families represented in the controlled evaluation corpus."""

    EXACT_COMPANY_LOOKUP = "exact_company_lookup"
    HS_CODE_LOOKUP = "hs_code_lookup"
    SEMANTIC_LEAD_DISCOVERY = "semantic_lead_discovery"
    OPERATING_STATUS = "operating_status"
    PRODUCT_COMPETITOR = "product_competitor"
    SQL_AGGREGATE = "sql_aggregate"
    MIXED_SOURCE = "mixed_source"
    TEMPORAL_CONFLICT = "temporal_conflict"
    DUPLICATE_SOURCE = "duplicate_source"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSAFE_OR_OUT_OF_SCOPE = "unsafe_or_out_of_scope"


BackendStatus = Literal["available", "degraded", "unavailable", "not_run", "failed"]
BackendStatuses = Annotated[
    dict[Identifier, BackendStatus],
    Field(
        json_schema_extra={
            "additionalProperties": False,
            "propertyNames": {"pattern": _ID_PATTERN},
        }
    ),
]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Keep derived immutable contracts validated instead of bypassing their guards."""
        if update is None:
            return super().model_copy(deep=deep)
        values = self.model_dump(mode="python")
        if deep:
            values = deepcopy(values)
        values.update(update)
        return type(self).model_validate(values)


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


def _unique(values: tuple[str, ...], field: str) -> tuple[str, ...]:
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must be unique")
    return values


class EvaluationCase(_Contract):
    """A product-facing query that only points to private label records by ID."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"dataset_role": {"const": "holdout"}}},
                    "then": {"properties": {"visibility": {"const": "private"}}},
                },
                {
                    "if": {"properties": {"answerable": {"const": True}}},
                    "then": {"properties": {"key_claim_ids": {"minItems": 1}}},
                },
            ]
        },
    )

    case_id: Identifier
    question: Question
    task_type: TaskType
    answerable: StrictBool
    dataset_role: Literal["development", "holdout"]
    visibility: Literal["public", "private"]
    as_of_date: date
    reference_evidence_set_id: Identifier
    key_claim_ids: tuple[ClaimId, ...] = Field(json_schema_extra={"uniqueItems": True})
    business_decision_id: Identifier | None = None
    label_version: Literal["trade-intel-eval/v1"] = "trade-intel-eval/v1"

    @field_validator("case_id", "question", "reference_evidence_set_id", "business_decision_id")
    @classmethod
    def nonblank_values(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value)

    @model_validator(mode="after")
    def protect_holdout_and_claim_refs(self) -> Self:
        _unique(self.key_claim_ids, "key_claim_ids")
        if self.dataset_role == "holdout" and self.visibility != "private":
            raise ValueError("holdout cases must be private")
        if self.answerable and not self.key_claim_ids:
            raise ValueError("answerable cases require key claim IDs")
        return self

    @classmethod
    def validated_fixture(cls, **updates: object) -> Self:
        """A valid, label-free fixture for contract tests and examples."""
        values: dict[str, object] = {
            "case_id": "case-development-001",
            "question": "What is the verified operating status of Example Exporter?",
            "task_type": TaskType.OPERATING_STATUS,
            "answerable": True,
            "dataset_role": "development",
            "visibility": "public",
            "as_of_date": date(2026, 8, 30),
            "reference_evidence_set_id": "reference-set-development-001",
            "key_claim_ids": ("claim_" + "1" * 64,),
            "business_decision_id": "decision-development-001",
        }
        values.update(updates)
        return cls.model_validate(values)


class ReferenceEvidence(_Contract):
    """A private reference mapping from an evaluation evidence set to evidence identity."""

    reference_evidence_id: Identifier
    reference_evidence_set_id: Identifier
    evidence_id: EvidenceId
    required: StrictBool

    @field_validator("reference_evidence_id", "reference_evidence_set_id")
    @classmethod
    def nonblank_values(cls, value: str) -> str:
        return _nonblank(value)


class ReferenceClaim(_Contract):
    """A labeled claim stored separately from the product-facing evaluation case."""

    claim_id: ClaimId
    reference_evidence_set_id: Identifier
    evidence_ids: tuple[EvidenceId, ...] = Field(json_schema_extra={"uniqueItems": True})
    claim_text: LabelText

    @field_validator("reference_evidence_set_id", "claim_text")
    @classmethod
    def nonblank_values(cls, value: str) -> str:
        return _nonblank(value)

    @model_validator(mode="after")
    def evidence_ids_are_unique(self) -> Self:
        _unique(self.evidence_ids, "evidence_ids")
        return self


class BusinessDecision(_Contract):
    """A separate decision label used by business evaluation metrics."""

    business_decision_id: Identifier
    key_claim_ids: tuple[ClaimId, ...] = Field(json_schema_extra={"uniqueItems": True})
    decision_text: LabelText
    decision_type: Literal["recommend", "escalate", "refuse"] = "recommend"

    @field_validator("business_decision_id", "decision_text")
    @classmethod
    def nonblank_values(cls, value: str) -> str:
        return _nonblank(value)

    @model_validator(mode="after")
    def claims_are_unique(self) -> Self:
        _unique(self.key_claim_ids, "key_claim_ids")
        return self


class EvaluationSnapshot(_Contract):
    """The hashes that freeze a reproducible dataset and execution candidate."""

    snapshot_id: Identifier
    dataset_hash: Hash
    reference_hash: Hash
    corpus_hash: Hash
    index_hash: Hash
    profile_hash: Hash
    model_hash: Hash
    prompt_hash: Hash
    evaluator_hash: Hash
    code_hash: Hash

    @field_validator("snapshot_id")
    @classmethod
    def nonblank_value(cls, value: str) -> str:
        return _nonblank(value)


class RunManifest(_Contract):
    """The reproducibility record for one evaluation run.

    ``snapshot`` is the single source of every frozen input hash. Backend
    statuses encode both available and degraded components without a second,
    contradictory component list.
    """

    run_id: Identifier
    snapshot: EvaluationSnapshot
    backend_statuses: BackendStatuses

    @field_validator("run_id")
    @classmethod
    def nonblank_value(cls, value: str) -> str:
        return _nonblank(value)

class PerQueryResult(_Contract):
    """A label-free trace retaining IDs and statuses needed to recompute metrics."""

    run_id: Identifier
    case_id: Identifier
    status: Literal["completed", "refused", "failed"]
    retrieved_evidence_ids: tuple[EvidenceId, ...] = Field(json_schema_extra={"uniqueItems": True})
    produced_claim_ids: tuple[ClaimId, ...] = Field(json_schema_extra={"uniqueItems": True})
    backend_statuses: BackendStatuses
    latency_ms: StrictFloat = Field(ge=0)

    @field_validator("run_id", "case_id")
    @classmethod
    def nonblank_values(cls, value: str) -> str:
        return _nonblank(value)

    @model_validator(mode="after")
    def trace_ids_and_degradation_are_consistent(self) -> Self:
        _unique(self.retrieved_evidence_ids, "retrieved_evidence_ids")
        _unique(self.produced_claim_ids, "produced_claim_ids")
        return self


EvaluationArtifact = (
    EvaluationCase
    | ReferenceEvidence
    | ReferenceClaim
    | BusinessDecision
    | EvaluationSnapshot
    | RunManifest
    | PerQueryResult
)


def evaluation_json_schema() -> dict[str, Any]:
    """Return the generated Draft 2020-12 schema for every evaluation artifact."""
    schema = TypeAdapter(EvaluationArtifact).json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://trade-agent.local/schemas/evaluation-v1.schema.json"
    return schema
