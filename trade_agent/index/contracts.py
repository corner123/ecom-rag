"""Immutable contracts for vectors written to a retrieval build."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr, field_validator, model_validator

_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_BGE_M3_MODEL = "BAAI/bge-m3"
_BGE_M3_DIMENSION = 1024


class EmbeddingContract(BaseModel):
    """The exact embedding identity and output invariant for one index build."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider: StrictStr
    model_name: StrictStr
    requested_revision: StrictStr
    resolved_revision: StrictStr
    dimension: StrictInt
    normalized: StrictBool
    dtype: Literal["float32"]
    library_version: StrictStr

    @field_validator("provider", "model_name", "library_version")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("requested_revision", "resolved_revision")
    @classmethod
    def immutable_revision(cls, value: str) -> str:
        if not _FULL_SHA.fullmatch(value):
            raise ValueError("revision must be a 40-character lowercase SHA")
        return value

    @field_validator("dimension")
    @classmethod
    def positive_dimension(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("dimension must be positive")
        return value

    @model_validator(mode="after")
    def verify_identity(self) -> "EmbeddingContract":
        if self.requested_revision != self.resolved_revision:
            raise ValueError("resolved revision must exactly match requested revision")
        if self.model_name == _BGE_M3_MODEL and self.dimension != _BGE_M3_DIMENSION:
            raise ValueError("BAAI/bge-m3 must have observed dimension 1024")
        if not self.normalized:
            raise ValueError("embedding vectors must be normalized")
        return self

    @property
    def revision(self) -> str:
        """Compatibility alias for consumers that need the requested revision."""

        return self.requested_revision
