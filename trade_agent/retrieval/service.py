"""Filter-first hybrid recall, bounded reranking and auditable final evidence."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from trade_agent.data.manifest import BuildManifest
from trade_agent.entities.dedup import EvidenceDeduplicator
from trade_agent.entities.models import EntityMention, EntityRecord, EvidenceCandidate
from trade_agent.entities.resolver import EntityResolver, ResolutionOutcome
from trade_agent.index.builder import validate_sparse_build
from trade_agent.index.milvus_store import DenseHit, chunk_ids_sha256
from trade_agent.retrieval.bm25 import BM25Index
from trade_agent.retrieval.filters import compile_filter_binding
from trade_agent.retrieval.fusion import ComponentRank, FusedHit, weighted_rrf
from trade_agent.retrieval.models import SparseHit
from trade_agent.retrieval.planner import QueryIntent, RetrievalPlan, RetrievalPlanner, candidate_chunk_ids
from trade_agent.retrieval.profiles import RetrievalProfile
from trade_agent.retrieval.reranker import RerankedHit, RerankOutcome, RerankerContract
from trade_agent.schemas.source import ChunkMetadata, ChunkRecord


class HitTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    build_id: str
    profile_id: str
    profile_version: str
    planner_version: str
    filter_expression_version: str
    components: dict[str, ComponentRank]
    fusion_score: float
    source_prior: float
    rerank_score: float | None
    pre_rerank_rank: int
    entity_resolution: ResolutionOutcome
    dedupe_cluster_id: str
    duplicate_chunk_ids: tuple[str, ...]
    dedupe_reasons: frozenset[str]
    degradation: tuple[str, ...]


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    chunk_id: str
    record: ChunkRecord
    rank: int
    trace: HitTrace

    @property
    def metadata(self) -> ChunkMetadata:
        return self.record.metadata


class RetrievalOutcome(BaseModel):
    """Final selected evidence plus the complete intermediate audit trail."""
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    build_id: str
    query: str
    plan: RetrievalPlan
    filter_expression: str
    profile: RetrievalProfile
    dense_hits: tuple[DenseHit, ...]
    sparse_hits: tuple[SparseHit, ...]
    fused_hits: tuple[FusedHit, ...]
    reranked_hits: tuple[RerankedHit, ...] | None
    rerank_contract: RerankerContract | None
    rerank_degraded: bool
    rerank_error_code: str | None
    rerank_latency_ms: float = Field(ge=0)
    hits: tuple[RetrievalHit, ...]
    degradation: tuple[str, ...]
    diversity_dropped_chunk_ids: tuple[str, ...]
    source_diversity_cap: int


# Compatibility with the inherited search() result name.
RetrievalResult = RetrievalOutcome


class RetrievalService:
    def __init__(self, *, build: BuildManifest, bm25: BM25Index, milvus: Any,
                 embedding_manager: Any, profile: RetrievalProfile, reranker: Any = None,
                 source_diversity_cap: int = 3, entity_registry: Sequence[EntityRecord] = ()):
        if not isinstance(build, BuildManifest) or not isinstance(bm25, BM25Index):
            raise TypeError("build and bm25 must be typed build artifacts")
        if not isinstance(profile, RetrievalProfile):
            raise TypeError("profile must be a RetrievalProfile")
        if type(source_diversity_cap) is not int or source_diversity_cap < 1:
            raise ValueError("source_diversity_cap must be a positive integer")
        effective_limits = {"dense": 100, "bm25": 100, "output": 100, **profile.candidate_limits}
        for component in ("dense", "bm25", "output"):
            limit = effective_limits[component]
            if not 1 <= limit <= 512:
                raise ValueError("candidate limits must be from 1 through 512")
        validate_sparse_build(build, bm25)
        contract = milvus.contract
        if (contract.build_id != build.build_id or contract.build_fingerprint != build.fingerprint
            or contract.build_chunk_count != len(build.chunks)
            or contract.build_chunk_ids_sha256 != chunk_ids_sha256(build.chunk_ids)):
            raise ValueError("Milvus collection contract does not match the frozen build")
        self.build = build
        self.build_id = build.build_id
        self._bm25, self._milvus = bm25, milvus
        self._embedding_manager, self._reranker = embedding_manager, reranker
        self.profile = profile.model_copy(update={"candidate_limits": effective_limits}, deep=True)
        self.source_diversity_cap = source_diversity_cap
        self._planner = RetrievalPlanner()
        self._records = {s.chunk_id: s.restore() for s in build.chunks}
        self._metadata_universe = {}
        for chunk_id, record in self._records.items():
            values = record.metadata.model_dump(mode="python")
            if isinstance(values.get("publish_time"), datetime):
                values["publish_time_epoch"] = int(values["publish_time"].timestamp())
            self._metadata_universe[chunk_id] = values
        self._resolver = EntityResolver(entity_registry)

    def search(self, intent: QueryIntent, *, top_k: int, rerank: bool = True) -> RetrievalOutcome:
        plan = self._planner.plan(intent)
        return self.retrieve(plan.query, plan, top_k=top_k, rerank=rerank)

    def retrieve(self, query: str, plan: RetrievalPlan, *, top_k: int = 10,
                 rerank: bool = True) -> RetrievalOutcome:
        if type(plan) is not RetrievalPlan:
            raise TypeError("plan must be an exact RetrievalPlan")
        if not isinstance(query, str) or not query.strip() or query != plan.query:
            raise ValueError("query must be nonblank and match the retrieval plan")
        if type(top_k) is not int or not 1 <= top_k <= 512:
            raise ValueError("top_k must be an integer from 1 through 512")
        compiled = compile_filter_binding(plan.filter)
        allowed = candidate_chunk_ids(self._metadata_universe, plan.filter)
        dense, sparse = (), ()
        if allowed:
            def dense_recall():
                vector = self._embedding_manager.embed_query(query)
                return tuple(self._milvus.search(vector,
                    top_k=self.profile.candidate_limits["dense"], filter_=plan.filter))
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="trade-recall") as pool:
                dense_future = pool.submit(dense_recall)
                sparse_future = pool.submit(self._bm25.search, query,
                    top_k=self.profile.candidate_limits["bm25"], allowed_chunk_ids=allowed)
                dense, sparse = dense_future.result(), tuple(sparse_future.result())
            self._validate_hits(dense, allowed, DenseHit)
            self._validate_hits(sparse, allowed, SparseHit)
        fused = tuple(weighted_rrf({"dense": dense, "bm25": sparse}, self.profile))
        reranked = None
        contract = None
        degraded, error, latency = False, None, 0.0
        degradation = []
        ordered = fused
        if not rerank:
            degradation.append("reranker_disabled")
        elif self._reranker is None:
            degradation.append("reranker_unavailable")
        elif fused:
            outcome = self._reranker.rerank(query, fused, len(fused))
            if not isinstance(outcome, RerankOutcome):
                raise TypeError("reranker must return RerankOutcome")
            reranked = tuple(hit.model_copy(update={"record": ChunkRecord.model_validate(hit.record)})
                             for hit in outcome.hits)
            seen = set()
            for hit in reranked:
                if hit.chunk_id not in {h.chunk_id for h in fused} or hit.chunk_id in seen:
                    raise ValueError("reranker returned unknown or duplicate candidates")
                if hit.record != self._records[hit.chunk_id]:
                    raise ValueError("reranker changed the frozen payload")
                seen.add(hit.chunk_id)
            ordered = reranked
            contract, degraded = outcome.model_contract, outcome.degraded
            error, latency = outcome.error_code, outcome.latency_ms
            if degraded:
                degradation.append("reranker:" + (error or "unknown_error"))
        else:
            contract = getattr(self._reranker, "default_contract", None)
        hits, dropped = self._select(ordered, plan, top_k, tuple(degradation), compiled.version)
        return RetrievalOutcome(build_id=self.build_id, query=query, plan=plan,
            filter_expression=compiled.expression, profile=self.profile,
            dense_hits=dense, sparse_hits=sparse, fused_hits=fused, reranked_hits=reranked,
            rerank_contract=contract, rerank_degraded=degraded, rerank_error_code=error,
            rerank_latency_ms=latency, hits=hits, degradation=tuple(degradation),
            diversity_dropped_chunk_ids=dropped, source_diversity_cap=self.source_diversity_cap).model_copy(deep=True)

    def _validate_hits(self, hits, allowed, expected_type):
        seen = set()
        for hit in hits:
            if type(hit) is not expected_type:
                raise TypeError("retriever returned an invalid hit type")
            if hit.build_id != self.build_id or hit.chunk_id not in allowed:
                raise ValueError("retriever returned a foreign build or violated the filter")
            if hit.chunk_id in seen:
                raise ValueError("retriever returned duplicate chunk IDs")
            if hit.record != self._records[hit.chunk_id]:
                raise ValueError("retriever payload does not match frozen build")
            seen.add(hit.chunk_id)

    def _select(self, ordered, plan, top_k, degradation, filter_version):
        resolutions, groups = {}, defaultdict(list)
        for hit in ordered:
            metadata = hit.record.metadata
            if metadata.entity_id:
                resolution = ResolutionOutcome(status="resolved", entity_id=metadata.entity_id,
                    reason="frozen_metadata", confidence=1.0)
            elif metadata.company_name or metadata.normalized_name:
                resolution = self._resolver.resolve(EntityMention(
                    name=metadata.company_name or metadata.normalized_name,
                    country_code=metadata.country_code))
            else:
                resolution = ResolutionOutcome(status="unresolved", entity_id=None,
                    reason="no_entity_mention", confidence=0.0)
            resolutions[hit.chunk_id] = resolution
            entity = resolution.entity_id or metadata.document_id
            fact = metadata.fact_type.value if metadata.fact_type else "unknown"
            candidate = EvidenceCandidate(evidence_id=hit.chunk_id, entity_id=entity,
                fact_type=fact, fact_value=hit.record.content, source_type=metadata.source_type.value,
                source_weight=metadata.source_weight, content=hit.record.content,
                content_hash=metadata.content_hash, parent_document_hash=metadata.parent_document_hash,
                source_url=str(metadata.source_url) if metadata.source_url else None,
                canonical_url=str(metadata.canonical_url) if metadata.canonical_url else None,
                publish_time=metadata.publish_time or metadata.ingested_at,
                valid_from=metadata.valid_from or metadata.ingested_at, valid_to=metadata.valid_to,
                syndication_group_id=metadata.dedupe_cluster_id)
            # Unrelated facts/entities must not collapse merely for sharing a URL.
            groups[(entity, fact)].append(candidate)
        clusters = {}
        for candidates in groups.values():
            for cluster in EvidenceDeduplicator().cluster(candidates):
                for candidate in cluster.candidates:
                    clusters[candidate.evidence_id] = cluster
        selected, dropped, used_clusters, counts = [], [], set(), Counter()
        for hit in ordered:
            cluster = clusters[hit.chunk_id]
            if cluster.cluster_id in used_clusters:
                continue
            source = hit.record.metadata.source_type.value
            if counts[source] >= self.source_diversity_cap:
                dropped.append(hit.chunk_id)
                continue
            if len(selected) >= top_k:
                continue
            used_clusters.add(cluster.cluster_id)
            counts[source] += 1
            selected.append(RetrievalHit(chunk_id=hit.chunk_id, record=hit.record,
                rank=len(selected)+1, trace=HitTrace(build_id=self.build_id,
                    profile_id=hit.profile_id, profile_version=hit.profile_version,
                    planner_version=plan.planner_version, filter_expression_version=filter_version,
                    components=hit.components, fusion_score=hit.score, source_prior=hit.source_prior,
                    rerank_score=getattr(hit, "rerank_score", None),
                    pre_rerank_rank=getattr(hit, "pre_rerank_rank", hit.rank),
                    entity_resolution=resolutions[hit.chunk_id], dedupe_cluster_id=cluster.cluster_id,
                    duplicate_chunk_ids=tuple(c.evidence_id for c in cluster.candidates),
                    dedupe_reasons=cluster.reasons, degradation=degradation)))
        return tuple(selected), tuple(dropped)
