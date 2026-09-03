"""Versioned source/fact ranking profiles for weighted retrieval."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, field_validator, model_validator


PROFILE_VERSION = "trade-source-profiles-v1"
_PRIOR_KEY = re.compile(r"(?:\*|[a-z0-9_]+):(?:\*|[a-z0-9_]+)")


class RetrievalProfile(BaseModel):
    """Bounded, auditable ranking controls; priors never decide truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: StrictStr
    version: StrictStr
    rrf_k: StrictInt = Field(ge=1, le=1000)
    retriever_weights: dict[StrictStr, StrictFloat]
    source_fact_priors: dict[StrictStr, StrictFloat]
    default_prior: StrictFloat
    candidate_limits: dict[StrictStr, StrictInt] = Field(default_factory=dict)

    @field_validator("profile_id", "version")
    @classmethod
    def nonblank_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("profile identity must not be blank")
        return value

    @field_validator("retriever_weights")
    @classmethod
    def bounded_weights(cls, value: dict[str, float]) -> dict[str, float]:
        if not value or any(item <= 0 or item > 1 for item in value.values()):
            raise ValueError("retriever weights must be in (0, 1]")
        return dict(sorted(value.items()))

    @field_validator("source_fact_priors")
    @classmethod
    def bounded_priors(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not _PRIOR_KEY.fullmatch(key) for key in value):
            raise ValueError("source/fact priors must use source:fact keys")
        if any(item < 0 or item > 1 for item in value.values()):
            raise ValueError("source/fact priors must be in [0, 1]")
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def required_retrievers(self) -> "RetrievalProfile":
        if not {"dense", "bm25"}.issubset(self.retriever_weights):
            raise ValueError("profile requires dense and bm25 weights")
        return self

    def prior(self, source_type: str, fact_type: str | None) -> float:
        source = getattr(source_type, "value", source_type)
        fact = getattr(fact_type, "value", fact_type)
        if fact is not None:
            exact = self.source_fact_priors.get(f"{source}:{fact}")
            if exact is not None:
                return float(exact)
            wildcard_fact = self.source_fact_priors.get(f"{source}:*")
            if wildcard_fact is not None:
                return float(wildcard_fact)
            wildcard_source = self.source_fact_priors.get(f"*:{fact}")
            if wildcard_source is not None:
                return float(wildcard_source)
        wildcard_all = self.source_fact_priors.get("*:*")
        return float(self.default_prior if wildcard_all is None else wildcard_all)


class _ProfileFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: StrictStr
    default_profile: StrictStr
    profiles: dict[StrictStr, RetrievalProfile]

    @model_validator(mode="after")
    def validate_file(self) -> "_ProfileFile":
        if self.version != PROFILE_VERSION:
            raise ValueError("unsupported retrieval profile version")
        if self.default_profile not in self.profiles:
            raise ValueError("default retrieval profile is absent")
        return self


def load_retrieval_profile(path: str | Path | None = None) -> RetrievalProfile:
    source = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "trade_agent/config/retrieval_profiles.yaml"
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        profile_file = _ProfileFile.model_validate(raw)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid retrieval profile file: {type(error).__name__}") from error
    return profile_file.profiles[profile_file.default_profile]
