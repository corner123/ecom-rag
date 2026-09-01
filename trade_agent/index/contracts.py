"""Immutable contracts for vectors written to a retrieval build."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_BGE_M3_MODEL = "BAAI/bge-m3"
_BGE_M3_DIMENSION = 1024
_BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_BGE_M3_ARTIFACT_MANIFEST_SHA256 = (
    "3a862f1d0a8543acc13e9faa5e6d6d1f916ee609b264960be8337e6ba509856b"
)
_BGE_M3_ARTIFACT_COUNT = 10
_TEST_REVISION = "0" * 40
_TEST_MANIFEST_SHA256 = "0" * 64
_TEST_LIBRARY_VERSION = "deterministic-test-v1"


class EmbeddingContract(BaseModel):
    """The exact embedding identity and output invariant for one index build."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider: Literal["sentence-transformers", "deterministic-test"]
    model_name: StrictStr
    requested_revision: StrictStr
    resolved_revision: StrictStr
    dimension: StrictInt
    normalized: StrictBool
    dtype: Literal["float32"]
    library_version: StrictStr
    artifact_manifest_sha256: StrictStr
    verified_artifact_count: StrictInt

    @field_validator("model_name", "library_version")
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

    @field_validator("artifact_manifest_sha256")
    @classmethod
    def manifest_sha(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("artifact manifest hash must be a SHA-256 digest")
        return value

    @field_validator("verified_artifact_count")
    @classmethod
    def nonnegative_verified_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("verified artifact count must not be negative")
        return value

    @model_validator(mode="after")
    def verify_identity(self) -> "EmbeddingContract":
        if self.provider == "sentence-transformers":
            if (self.model_name, self.requested_revision, self.resolved_revision) != (
                _BGE_M3_MODEL,
                _BGE_M3_REVISION,
                _BGE_M3_REVISION,
            ):
                raise ValueError(
                    "production contract must use the exact pinned BAAI/bge-m3 identity"
                )
            if self.dimension != _BGE_M3_DIMENSION or not self.normalized:
                raise ValueError(
                    "production BAAI/bge-m3 vectors must be normalized 1024-dimensional"
                )
            if (
                self.artifact_manifest_sha256 != _BGE_M3_ARTIFACT_MANIFEST_SHA256
                or self.verified_artifact_count != _BGE_M3_ARTIFACT_COUNT
            ):
                raise ValueError(
                    "production contract requires the exact trusted artifact manifest"
                )
        elif (
            self.model_name,
            self.requested_revision,
            self.resolved_revision,
            self.dimension,
            self.normalized,
            self.library_version,
            self.artifact_manifest_sha256,
            self.verified_artifact_count,
        ) != (
            "deterministic-test",
            _TEST_REVISION,
            _TEST_REVISION,
            _BGE_M3_DIMENSION,
            True,
            _TEST_LIBRARY_VERSION,
            _TEST_MANIFEST_SHA256,
            0,
        ):
            raise ValueError(
                "deterministic-test contract must use its explicit test-only identity"
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.provider == "sentence-transformers"

    def require_production(self) -> "EmbeddingContract":
        if not self.is_production:
            raise ValueError("a production embedding contract is required")
        return self

    @property
    def revision(self) -> str:
        """Compatibility alias for consumers that need the requested revision."""

        return self.requested_revision
