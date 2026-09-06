"""Typed, public-safe failures emitted by the trade workflow."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr


GraphErrorCode = Literal[
    "invalid_request",
    "policy_denied",
    "missing_entity_binding",
    "missing_sql_hmac_key",
    "retrieval_timeout_unsupported",
    "sql_timeout",
    "sql_transport",
    "sql_unavailable",
    "rag_timeout",
    "rag_transport",
    "rag_unavailable",
    "generation_timeout",
    "guard_timeout",
    "evidence_repository_error",
    "invalid_contract",
    "step_limit_exceeded",
    "retry_limit_exceeded",
    "llm_limit_exceeded",
    "internal_error",
]


class GraphError(BaseModel):
    """Stable error metadata safe to persist in a graph checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    code: GraphErrorCode
    node: StrictStr
    retryable: StrictBool
    detail: StrictStr


class GraphWorkflowError(RuntimeError):
    """Base exception for dependency configuration failures."""

    code: GraphErrorCode = "internal_error"


class EvidenceRepositoryError(GraphWorkflowError):
    code: GraphErrorCode = "evidence_repository_error"
