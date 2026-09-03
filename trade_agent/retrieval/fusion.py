"""Auditable Weighted Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

from trade_agent.retrieval.profiles import RetrievalProfile


_ALLOWED_RETRIEVERS = {"dense", "bm25"}


class _RetrievalHit(Protocol):
    @property
    def chunk_id(self) -> str: ...

    @property
    def record(self) -> Any: ...


class ComponentRank(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: StrictInt = Field(ge=1)
    raw_score: float | None
    retriever_weight: StrictFloat
    relevance_contribution: StrictFloat


class FusedHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: StrictStr
    record: Any
    score: StrictFloat
    rank: StrictInt = Field(ge=1)
    relevance_subtotal: StrictFloat
    source_prior: StrictFloat
    prior_contribution: StrictFloat
    profile_id: StrictStr
    profile_version: StrictStr
    components: dict[StrictStr, ComponentRank]


def weighted_rrf(
    rankings: Mapping[str, Sequence[_RetrievalHit]],
    profile: RetrievalProfile,
) -> list[FusedHit]:
    """Fuse rank-only retrievals with bounded source/fact priors."""

    if type(rankings) is not dict and not isinstance(rankings, Mapping):
        raise TypeError("rankings must be a mapping")
    unknown = set(rankings) - _ALLOWED_RETRIEVERS
    if unknown:
        raise ValueError(f"unknown retriever(s): {', '.join(sorted(unknown))}")
    missing = set(profile.retriever_weights) - set(rankings)
    if missing:
        raise ValueError(f"missing required retriever(s): {', '.join(sorted(missing))}")

    accumulators: dict[str, dict[str, Any]] = {}
    for retriever, hits in rankings.items():
        weight = profile.retriever_weights[retriever]
        if isinstance(hits, (str, bytes)):
            raise TypeError("retriever rankings must be sequences of hits")
        limit = profile.candidate_limits.get(retriever)
        materialized = tuple(hits)
        if limit is not None:
            materialized = materialized[:limit]
        seen: set[str] = set()
        for rank, hit in enumerate(materialized, start=1):
            if hit.chunk_id in seen:
                raise ValueError(f"duplicate chunk_id {hit.chunk_id} in {retriever} ranking")
            seen.add(hit.chunk_id)
            relevance = weight / (profile.rrf_k + rank)
            prior = profile.prior(
                getattr(hit.record.metadata.source_type, "value", hit.record.metadata.source_type),
                getattr(hit.record.metadata.fact_type, "value", hit.record.metadata.fact_type),
            )
            contribution = prior * relevance
            accumulator = accumulators.setdefault(
                hit.chunk_id,
                {
                    "record": hit.record,
                    "relevance": 0.0,
                    "prior": prior,
                    "prior_contribution": 0.0,
                    "components": {},
                },
            )
            accumulator["relevance"] += relevance
            accumulator["prior_contribution"] += contribution
            accumulator["components"][retriever] = ComponentRank(
                rank=rank,
                raw_score=float(getattr(hit, "score")) if hasattr(hit, "score") else None,
                retriever_weight=weight,
                relevance_contribution=contribution,
            )

    ordered = sorted(
        accumulators.items(),
        key=lambda item: (-item[1]["prior_contribution"], item[0]),
    )
    output_limit = profile.candidate_limits.get("output")
    if output_limit is not None:
        ordered = ordered[:output_limit]
    return [
        FusedHit(
            chunk_id=chunk_id,
            record=accumulator["record"],
            score=accumulator["prior_contribution"],
            rank=rank,
            relevance_subtotal=accumulator["relevance"],
            source_prior=accumulator["prior"],
            prior_contribution=accumulator["prior_contribution"],
            profile_id=profile.profile_id,
            profile_version=profile.version,
            components=accumulator["components"],
        )
        for rank, (chunk_id, accumulator) in enumerate(ordered, start=1)
    ]
