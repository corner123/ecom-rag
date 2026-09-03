"""Pinned BGE cross-encoder reranking with explicit degradation."""

from __future__ import annotations

from importlib.metadata import version
import math
import time
from collections.abc import Sequence
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr

from trade_agent.retrieval.fusion import FusedHit


RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


class RerankerContract(BaseModel):
    """The model identity used for one rerank operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["sentence-transformers", "test"] = "sentence-transformers"
    model_name: Literal["BAAI/bge-reranker-v2-m3"]
    revision: Literal["953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"]
    max_length: StrictInt = Field(ge=32, le=8192)
    library_version: StrictStr


class RerankedHit(FusedHit):
    rerank_score: StrictFloat | None
    pre_rerank_rank: StrictInt = Field(ge=1)


class RerankOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hits: tuple[RerankedHit, ...]
    model_contract: RerankerContract
    degraded: StrictBool
    error_code: StrictStr | None
    latency_ms: StrictFloat = Field(ge=0)


class _CrossEncoder(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> Any: ...


class _ContractModel(Protocol):
    @property
    def contract(self) -> RerankerContract: ...


class BgeReranker:
    """Lazy pinned reranker; call failures are never silently hidden."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        contract: RerankerContract | None = None,
        max_length: int = 8192,
        batch_size: int = 16,
        device: str = "cpu",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        strict_mode: bool = True,
    ) -> None:
        if max_length < 32 or max_length > 8192:
            raise ValueError("max_length must be between 32 and 8192")
        self._injected_model = model
        self._injected_contract = contract
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.strict_mode = strict_mode
        self._model: Any | None = None

    @property
    def default_contract(self) -> RerankerContract:
        return RerankerContract(
            model_name=RERANKER_MODEL,
            revision=RERANKER_REVISION,
            max_length=self.max_length,
            library_version=version("sentence-transformers"),
        )

    def rerank(
        self,
        query: str,
        hits: Sequence[FusedHit],
        top_k: int,
    ) -> RerankOutcome:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonblank string")
        if type(top_k) is not int or not 1 <= top_k <= 512:
            raise ValueError("top_k must be an integer from 1 through 512")
        if isinstance(hits, (str, bytes)) or not isinstance(hits, Sequence):
            raise TypeError("hits must be a sequence of FusedHit values")
        materialized = tuple(hits)
        if any(type(hit) is not FusedHit for hit in materialized):
            raise TypeError("hits must be exact FusedHit values")
        chunk_ids = [hit.chunk_id for hit in materialized]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("hits contain duplicate chunk IDs")

        started = time.perf_counter()
        model_contract: RerankerContract
        try:
            model = self._load()
            model_contract = (
                self._injected_contract
                if self._injected_contract is not None
                else getattr(model, "contract", None)
                or self.default_contract
            )
            pairs = [
                (query, str(hit.record.get("content", "")) if isinstance(hit.record, dict) else str(hit.record))
                for hit in materialized
            ]
            raw_scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            scores = np.asarray(raw_scores).reshape(-1)
            if len(scores) != len(materialized) or not np.isfinite(scores).all():
                raise ValueError("reranker returned invalid scores")
            scored = sorted(
                (
                    (float(score), position, hit)
                    for position, (score, hit) in enumerate(zip(scores, materialized))
                ),
                key=lambda item: (-item[0], item[1]),
            )
            reranked = [
                RerankedHit.model_validate(
                    {
                        **hit.model_dump(mode="python"),
                        "rerank_score": score,
                        "pre_rerank_rank": position + 1,
                        "rank": rank,
                    }
                )
                for rank, (score, position, hit) in enumerate(scored[:top_k], start=1)
            ]
            return RerankOutcome(
                hits=reranked,
                model_contract=model_contract,
                degraded=False,
                error_code=None,
                latency_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
            )
        except Exception as error:
            if self.strict_mode:
                raise RerankerUnavailable("reranker_unavailable") from error
            degraded_hits = tuple(
                RerankedHit.model_validate(
                    {
                        **hit.model_dump(mode="python"),
                        "rerank_score": None,
                        "pre_rerank_rank": hit.rank,
                        "rank": rank,
                    }
                )
                for rank, hit in enumerate(materialized[:top_k], start=1)
            )
            return RerankOutcome(
                hits=degraded_hits,
                model_contract=self._injected_contract or self.default_contract,
                degraded=True,
                error_code="reranker_unavailable",
                latency_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
            )

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if self._injected_model is not None:
            self._model = self._injected_model
            return self._model
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            RERANKER_MODEL,
            max_length=self.max_length,
            device=self.device,
            cache_dir=self.cache_dir,
            revision=RERANKER_REVISION,
            local_files_only=self.local_files_only,
            trust_remote_code=False,
        )
        return self._model


class RerankerUnavailable(RuntimeError):
    """Raised in strict mode when the cross-encoder cannot produce evidence."""


def smoke() -> dict[str, Any]:
    def hit(chunk_id: str, content: str) -> FusedHit:
        return FusedHit(
            chunk_id=chunk_id,
            record={"content": content},
            score=1.0,
            rank=1,
            relevance_subtotal=1.0,
            source_prior=1.0,
            prior_contribution=1.0,
            profile_id="balanced-v1",
            profile_version="trade-source-profiles-v1",
            components={},
        )

    outcome = BgeReranker().rerank(
        "HS 850440 charger procurement growth",
        [
            hit("irrelevant", "Unrelated synthetic scanned policy"),
            hit("relevant", "HS 850440 USB-C charger procurement increased"),
        ],
        2,
    )
    if outcome.degraded:
        raise RuntimeError("real BGE reranker smoke degraded")
    return {
        "status": "ok",
        "model": outcome.model_contract.model_name,
        "revision": outcome.model_contract.revision,
        "top_chunk_id": outcome.hits[0].chunk_id,
        "latency_ms": outcome.latency_ms,
    }
