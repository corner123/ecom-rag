"""Adapters that run real BM25/dense/hybrid engineering-RAG ablations."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Mapping

from rag_core.engineering import EngineeringIndex, EngineeringRAGService
from rag_core.engineering.index import HybridPartitionRetriever
from rag_core.engineering.target import (
    resolve_target_repository,
    resolve_target_source_id,
)
from rag_core.ingestion import BuildManifest
from rag_core.sources import GitRepositorySource
from rag_core.retrieval.engineering import (
    FederatedRetriever,
    EngineeringCrossEncoderReranker,
    LiveASTRetriever,
    LiveCodeRetriever,
    LiveGitVerifier,
)

from .engineering import (
    EvaluationPrediction,
    EvaluationSample,
    EngineeringPredictor,
    expected_route,
)
from .retrieval_profiles import (
    RetrievalExperimentProfile,
    validate_unique_profiles,
)


@dataclass(slots=True)
class ServiceEvaluationPredictor:
    """Expose an :class:`EngineeringRAGService` through the evaluator contract."""

    service: EngineeringRAGService
    evaluation_metadata: dict | None = None

    def __call__(self, sample: EvaluationSample, top_k: int) -> EvaluationPrediction:
        retrieval = self.service.retrieve(sample.question, top_k=top_k)
        answer = self.service.answerer.answer(retrieval)
        return EvaluationPrediction(
            predicted_intent=retrieval.intent.value,
            results=tuple(retrieval.results),
            refused=answer.refused,
            metadata={
                "sufficient_evidence": retrieval.sufficient_evidence,
                "warnings": list(retrieval.warnings),
                "matched_rule": retrieval.matched_rule,
                "refusal_reason": retrieval.refusal_reason,
                "answer_text": answer.answer,
                "citation_sources": [citation.source for citation in answer.citations],
                "live_verification_attempted": retrieval.live_verification_attempted,
                "live_revision": dict(retrieval.live_revision),
            },
        )


@dataclass(slots=True)
class OracleIndexPredictor:
    """Pure indexed retrieval under the sample's explicit v2 source route."""

    retriever: FederatedRetriever
    evaluation_metadata: dict
    prediction_metadata: dict | None = None
    candidate_budget_multiplier: int = 3

    def __post_init__(self) -> None:
        if (
            type(self.candidate_budget_multiplier) is not int
            or self.candidate_budget_multiplier < 1
        ):
            raise ValueError("candidate_budget_multiplier must be a positive integer")

    def __call__(self, sample: EvaluationSample, top_k: int) -> EvaluationPrediction:
        route = expected_route(sample)
        filters = {
            "design": (("internal",), ("design", "history", "code", "test")),
            "implementation": (("internal",), ("code", "test", "design", "history")),
            "official": (("official",), ("official",)),
            "comparison": (
                ("internal", "official"),
                ("code", "test", "design", "history", "official"),
            ),
        }
        corpora, authorities = filters[route]
        results = self.retriever.search(
            sample.question,
            top_k=max(top_k, top_k * self.candidate_budget_multiplier),
            corpora=corpora,
            authorities=authorities,
        )
        degraded = sorted(
            {
                str(component)
                for result in results
                for component in result.metadata.get(
                    "retrieval_degraded_components", []
                )
            }
        )
        if degraded:
            raise RuntimeError(
                "formal index benchmark refuses degraded retrieval: "
                + ", ".join(degraded)
            )
        return EvaluationPrediction(
            predicted_intent=route,
            results=tuple(results),
            refused=False,
            metadata={
                "oracle_route": True,
                "live_verification": False,
                **dict(self.prediction_metadata or {}),
            },
        )


def _retriever_view(
    index: EngineeringIndex,
    mode: str,
    *,
    candidate_multiplier: int | None = None,
    profile: RetrievalExperimentProfile | None = None,
) -> FederatedRetriever:
    """Reuse one loaded index while replacing only the per-partition retriever."""

    if mode not in {"bm25", "dense", "hybrid"}:
        raise ValueError(f"unsupported ablation mode: {mode}")
    if profile is not None and mode != "hybrid":
        raise ValueError("retrieval profiles are only valid for hybrid mode")
    if profile is not None and candidate_multiplier is not None:
        raise ValueError(
            "candidate_multiplier and retrieval profile cannot be supplied together"
        )
    source = index.federated
    federated_multiplier = (
        profile.federated_candidate_multiplier
        if profile is not None
        else (
            source.candidate_multiplier
            if candidate_multiplier is None
            else candidate_multiplier
        )
    )
    federated = FederatedRetriever(
        rrf_k=source.rrf_k,
        candidate_multiplier=federated_multiplier,
        fail_open=(
            False
            if candidate_multiplier is not None or profile is not None
            else source.fail_open
        ),
    )
    for partition in source.partitions:
        hybrid = partition.retriever
        if mode == "hybrid":
            retriever = (
                hybrid
                if candidate_multiplier is None and profile is None
                else HybridPartitionRetriever(
                    hybrid.dense,
                    hybrid.bm25,
                    rrf_k=hybrid.rrf_k,
                    candidate_multiplier=(
                        profile.partition_candidate_multiplier
                        if profile is not None
                        else candidate_multiplier
                    ),
                    dense_weight=(
                        profile.dense_weight
                        if profile is not None
                        else hybrid.dense_weight
                    ),
                    bm25_weight=(
                        profile.bm25_weight
                        if profile is not None
                        else hybrid.bm25_weight
                    ),
                    fail_open=False,
                )
            )
        else:
            retriever = getattr(hybrid, mode, None)
            if retriever is None:
                raise TypeError(
                    f"partition {partition.corpus}/{partition.authority} "
                    f"does not expose a {mode} retriever"
                )
        federated.add_partition(
            partition.corpus,
            partition.authority,
            retriever,
            weight=partition.weight,
        )
    return federated


def create_retrieval_profile_predictors(
    *,
    index_root: str | Path,
    profiles: Iterable[RetrievalExperimentProfile],
    embedding_manager=None,
    candidate_budget_multiplier: int = 1,
) -> Mapping[str, EngineeringPredictor]:
    """Create a development-only hybrid grid against one frozen FAISS build.

    Varying candidate multipliers changes both computation and latency.  This
    factory records those budgets but intentionally does not call the formal
    equal-budget suite contract.  A selected candidate must later be compared
    with the baseline under an explicitly declared common budget, or its budget
    difference must be reported as an engineering trade-off.
    """

    if (
        type(candidate_budget_multiplier) is not int
        or candidate_budget_multiplier < 1
    ):
        raise ValueError("candidate_budget_multiplier must be a positive integer")
    selected_profiles = validate_unique_profiles(profiles)
    index = EngineeringIndex.load(
        index_root,
        embedding_manager=embedding_manager,
        runtime_backend="faiss",
    )
    _assert_current_build(index)
    common_metadata = {
        **_public_index_metadata(index),
        "oracle_route": True,
        "live_verification": False,
        "answer_generation": False,
        "development_grid": True,
        "fail_closed": True,
        "candidate_budget_multiplier": candidate_budget_multiplier,
    }
    predictors: dict[str, EngineeringPredictor] = {}
    for profile in selected_profiles:
        profile_metadata = profile.as_metadata()
        predictors[profile.name] = OracleIndexPredictor(
            _retriever_view(index, "hybrid", profile=profile),
            evaluation_metadata={
                **common_metadata,
                **profile_metadata,
                "retrieval_strategy": profile.name,
            },
            prediction_metadata=profile_metadata,
            candidate_budget_multiplier=candidate_budget_multiplier,
        )
    return predictors


def create_predictors(
    *,
    index_root: str | Path,
    mini_nanobot_repo: str | Path | None = None,
    embedding_manager=None,
    enable_live_code: bool = True,
    enable_reranker: bool = False,
    reranker_model: str = "BAAI/bge-reranker-base",
) -> Mapping[str, EngineeringPredictor]:
    """Create fair BM25/dense/hybrid predictors from one loaded build."""

    # Formal comparisons are frozen to the portable FAISS artifact. Runtime
    # ENGINEERING_VECTOR_BACKEND must never alter published benchmark metrics.
    index = EngineeringIndex.load(
        index_root,
        embedding_manager=embedding_manager,
        runtime_backend="faiss",
    )
    repo = Path(mini_nanobot_repo).expanduser() if mini_nanobot_repo else None
    live = (
        LiveCodeRetriever(
            repo,
            corpus="internal",
            authority="code",
            case_sensitive=True,
        )
        if enable_live_code and repo is not None and repo.is_dir()
        else None
    )
    live_ast = LiveASTRetriever(repo) if live is not None else None
    live_git = LiveGitVerifier(repo) if live is not None else None
    predictors: dict[str, EngineeringPredictor] = {}
    for mode in ("bm25", "dense", "hybrid"):
        service = EngineeringRAGService(
            _retriever_view(index, mode),
            live_code=live,
            live_ast=live_ast,
            live_git=live_git,
            index_stats=index.stats(),
        )
        predictors[mode] = ServiceEvaluationPredictor(
            service,
            evaluation_metadata={
                **_public_index_metadata(index),
                "live_verification": bool(live),
                "reranker": False,
            },
        )
    if enable_reranker:
        service = EngineeringRAGService(
            _retriever_view(index, "hybrid"),
            live_code=live,
            live_ast=live_ast,
            live_git=live_git,
            reranker=EngineeringCrossEncoderReranker(reranker_model),
            index_stats=index.stats(),
        )
        predictors["hybrid_rerank"] = ServiceEvaluationPredictor(
            service,
            evaluation_metadata={
                **_public_index_metadata(index),
                "live_verification": bool(live),
                "reranker": reranker_model,
            },
        )
    return predictors


def create_index_ablation_predictors(
    *,
    index_root: str | Path,
    embedding_manager=None,
) -> Mapping[str, EngineeringPredictor]:
    index = EngineeringIndex.load(
        index_root,
        embedding_manager=embedding_manager,
        runtime_backend="faiss",
    )
    _assert_current_build(index)
    metadata = {
        **_public_index_metadata(index),
        "oracle_route": True,
        "live_verification": False,
        "answer_generation": False,
    }
    return {
        mode: OracleIndexPredictor(
            _retriever_view(index, mode, candidate_multiplier=1),
            evaluation_metadata={
                **metadata,
                "retrieval_strategy": mode,
                "candidate_budget_multiplier": 3,
                "partition_candidate_multiplier": 1,
                "federated_candidate_multiplier": 1,
                "fail_closed": True,
            },
        )
        for mode in ("bm25", "dense", "hybrid")
    }


def create_e2e_predictors(
    *,
    index_root: str | Path,
    mini_nanobot_repo: str | Path,
    target_source_id: str | None = None,
    embedding_manager=None,
) -> Mapping[str, EngineeringPredictor]:
    index = EngineeringIndex.load(
        index_root,
        embedding_manager=embedding_manager,
        runtime_backend="faiss",
    )
    manifest = _assert_current_build(index)
    repo = Path(mini_nanobot_repo).expanduser().resolve()
    _assert_live_repo_matches_manifest(
        repo,
        manifest,
        source_id=resolve_target_source_id(target_source_id),
    )
    live = LiveCodeRetriever(
        repo,
        corpus="internal",
        authority="code",
        case_sensitive=True,
    )
    service = EngineeringRAGService(
        _retriever_view(index, "hybrid", candidate_multiplier=3),
        live_code=live,
        live_ast=LiveASTRetriever(repo),
        live_git=LiveGitVerifier(repo),
        index_stats=index.stats(),
    )
    return {
        "hybrid_live": ServiceEvaluationPredictor(
            service,
            evaluation_metadata={
                **_public_index_metadata(index),
                "retrieval_strategy": "hybrid",
                "oracle_route": False,
                "live_verification": True,
                "answer_generation": True,
                "reranker": False,
                "candidate_budget_multiplier": 3,
                "partition_candidate_multiplier": 3,
                "federated_candidate_multiplier": 3,
                "fail_closed": True,
            },
        )
    }


def create_default_predictors() -> Mapping[str, EngineeringPredictor]:
    """Environment-driven pure-index ablation factory."""

    index_root = os.getenv(
        "ENGINEERING_INDEX_DIR", "data/indexes/engineering"
    )
    return create_index_ablation_predictors(
        index_root=index_root,
    )


def create_default_e2e_predictors() -> Mapping[str, EngineeringPredictor]:
    index_root = os.getenv("ENGINEERING_INDEX_DIR", "data/indexes/engineering")
    target_repo = resolve_target_repository()
    if not target_repo:
        raise ValueError(
            "KNOWLEDGE_TARGET_REPO is required for end-to-end evaluation "
            "(legacy MINI_NANOBOT_REPO also supported)"
        )
    return create_e2e_predictors(
        index_root=index_root,
        mini_nanobot_repo=target_repo,
        target_source_id=resolve_target_source_id(),
    )


def _public_index_metadata(index: EngineeringIndex) -> dict:
    stats = index.stats()
    return {
        "build_id": stats["build_id"],
        "embedding_model": stats["embedding_model"],
        "document_count": stats["document_count"],
        "partitions": stats["partitions"],
    }


def _assert_current_build(index: EngineeringIndex) -> BuildManifest:
    manifest_path = os.getenv(
        "ENGINEERING_MANIFEST_PATH", "data/manifests/builds/current.json"
    )
    manifest = BuildManifest.read(manifest_path)
    if manifest.build_id != index.build_id:
        raise RuntimeError(
            f"evaluation refused stale index: manifest={manifest.build_id}, index={index.build_id}"
        )
    return manifest


def _assert_live_repo_matches_manifest(
    repo: Path, manifest: BuildManifest, *, source_id: str = "mini_nanobot"
) -> None:
    expected = next(
        (record for record in manifest.sources if record.source_id == source_id),
        None,
    )
    if expected is None:
        raise RuntimeError(f"evaluation manifest is missing source: {source_id}")
    options = expected.metadata
    collected = GitRepositorySource(
        source_id,
        repo,
        include=options.get("include"),
        exclude=options.get("exclude"),
        include_python_symbols=bool(options.get("python_symbol_cards")),
        git_history_limit=int(options.get("git_history_limit") or 0),
        max_file_size_bytes=int(options.get("max_file_size_bytes") or 2_000_000),
        version=expected.version,
        license=expected.license,
        metadata={"corpus": "internal", "authority": "project"},
    ).collect().record
    mismatches = {
        key: (getattr(expected, key), getattr(collected, key))
        for key in ("commit_sha", "dirty", "content_hash")
        if getattr(expected, key) != getattr(collected, key)
    }
    if mismatches:
        raise RuntimeError(
            "end-to-end evaluation refused a live repository that differs "
            f"from the indexed snapshot: {mismatches}"
        )
