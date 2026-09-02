"""Immutable embedding identity contracts for retrieval index builds."""

from __future__ import annotations

import hmac
from importlib.metadata import version
import json
import re
import secrets
from typing import Literal
import weakref

from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from .provenance import VerifiedEmbeddingSnapshot, _require_verified_snapshot

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


class _ContractToken:
    __slots__ = ("mac", "__weakref__")

    def __init__(self, mac: bytes) -> None:
        self.mac = mac


class _ContractAttestor:
    """Issue copy-preserving tokens whose HMAC binds every public field."""

    def __init__(self) -> None:
        self._key = secrets.token_bytes(32)
        self._tokens: dict[int, weakref.ReferenceType[_ContractToken]] = {}

    def issue(self, payload: bytes) -> _ContractToken:
        token = _ContractToken(hmac.digest(self._key, payload, "sha256"))
        identity = id(token)

        def discard(reference: weakref.ReferenceType[_ContractToken]) -> None:
            if self._tokens.get(identity) is reference:
                self._tokens.pop(identity, None)

        self._tokens[identity] = weakref.ref(token, discard)
        return token

    def verifies(self, token: object | None, payload: bytes) -> bool:
        if type(token) is not _ContractToken:
            return False
        reference = self._tokens.get(id(token))
        if reference is None or reference() is not token:
            return False
        expected = hmac.digest(self._key, payload, "sha256")
        return hmac.compare_digest(token.mac, expected)


class EmbeddingContract(BaseModel):
    """The exact embedding identity and output invariant for one index build."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )

    _production_attestation: object | None = PrivateAttr(default=None)

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
    def verify_identity(self) -> EmbeddingContract:
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
        return _is_live_production_contract(self)

    def require_production(self) -> EmbeddingContract:
        if not self.is_production:
            raise ValueError("a live verified production embedding contract is required")
        return self

    @property
    def revision(self) -> str:
        """Compatibility alias for consumers that need the requested revision."""

        return self.requested_revision


def _canonical_contract_payload(contract: EmbeddingContract) -> bytes:
    public_schema = set(EmbeddingContract.model_fields)
    if set(contract.__dict__) != public_schema:
        raise ValueError("embedding contract instance does not match the complete public schema")
    public_fields = {field_name: getattr(contract, field_name) for field_name in public_schema}
    validated = EmbeddingContract.model_validate(public_fields)
    canonical = json.dumps(
        validated.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return canonical.encode("utf-8")


def _build_contract_authority() -> tuple[object, object]:
    attestor = _ContractAttestor()
    contract_payload = _canonical_contract_payload
    installed_version = version
    require_snapshot = _require_verified_snapshot

    def issue_contract(
        *,
        receipt: VerifiedEmbeddingSnapshot,
    ) -> EmbeddingContract:
        verified = require_snapshot(receipt)
        contract = EmbeddingContract(
            provider="sentence-transformers",
            model_name=verified.model_name,
            requested_revision=verified.requested_revision,
            resolved_revision=verified.resolved_revision,
            dimension=_BGE_M3_DIMENSION,
            normalized=True,
            dtype="float32",
            library_version=installed_version("sentence-transformers"),
            artifact_manifest_sha256=verified.artifact_manifest_sha256,
            verified_artifact_count=verified.artifact_count,
        )
        contract._production_attestation = attestor.issue(
            contract_payload(contract)
        )
        return contract

    def is_live_contract(contract: EmbeddingContract) -> bool:
        if contract.provider != "sentence-transformers":
            return False
        try:
            payload = contract_payload(contract)
        except (AttributeError, TypeError, ValueError):
            return False
        return attestor.verifies(contract._production_attestation, payload)

    return issue_contract, is_live_contract


(
    _issue_verified_production_contract,
    _is_live_production_contract,
) = _build_contract_authority()
del _build_contract_authority
