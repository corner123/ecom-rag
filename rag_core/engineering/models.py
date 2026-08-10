"""Public response models for engineering retrieval and grounded answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rag_core.retrieval.engineering import EngineeringSearchResult, SourceIntent


@dataclass(frozen=True, slots=True)
class EvidenceCitation:
    """A structured, user-visible citation independent of vector-store details."""

    citation_id: str
    source: str
    corpus: str
    authority: str
    evidence_role: str
    line_start: int | None = None
    line_end: int | None = None
    symbol: str | None = None
    revision: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    url: str | None = None
    live_verified: bool = False
    source_version: str | None = None
    fetched_at: str | None = None
    content_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalOutcome:
    query: str
    intent: SourceIntent
    results: list[EngineeringSearchResult]
    citations: list[EvidenceCitation]
    sufficient_evidence: bool
    refusal_reason: str | None = None
    live_verification_attempted: bool = False
    live_verification_terms: tuple[str, ...] = ()
    live_revision: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    matched_rule: str = ""

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        return {
            "schema_version": "engineering-retrieval/v1",
            "query": self.query,
            "intent": self.intent.value,
            "sufficient_evidence": self.sufficient_evidence,
            "refusal_reason": self.refusal_reason,
            "live_verification_attempted": self.live_verification_attempted,
            "live_verification_terms": list(self.live_verification_terms),
            "live_revision": dict(self.live_revision),
            "warnings": list(self.warnings),
            "matched_rule": self.matched_rule,
            "citations": [citation.to_dict() for citation in self.citations],
            "results": [
                _public_result(result, include_content=include_content)
                for result in self.results
            ],
        }


_PUBLIC_METADATA_KEYS = frozenset(
    {
        "record_kind",
        "source_id",
        "source_type",
        "source_version",
        "source_fetched_at",
        "source_content_hash",
        "document_content_hash",
        "source_dirty",
        "source_commit_sha",
        "document_path",
        "document_title",
        "parent_path",
        "relative_path",
        "symbol_kind",
        "evidence_role",
        "verification_method",
        "verification_term",
        "git_commit_sha",
        "git_branch",
        "git_dirty",
        "federated_partition",
        "rrf_retrievers",
        "first_stage_score",
        "rerank_score",
        "retrieval_degraded_components",
    }
)


def _public_metadata(metadata: dict[str, Any] | Any) -> dict[str, Any]:
    """Return only stable evidence fields; never expose host or HTTP internals."""

    if not isinstance(metadata, dict):
        return {}
    return {
        key: metadata[key]
        for key in _PUBLIC_METADATA_KEYS
        if key in metadata and metadata[key] is not None
    }


def _public_source(result: EngineeringSearchResult) -> str:
    if result.metadata.get("live_verification"):
        relative = result.metadata.get("relative_path")
        if relative:
            return str(relative).replace("\\", "/")
    return result.source


def _public_result(
    result: EngineeringSearchResult, *, include_content: bool = True
) -> dict[str, Any]:
    return {
        "content": result.content if include_content else "",
        "citation": _public_source(result),
        "source": _public_source(result),
        "score": result.score,
        "corpus": result.corpus,
        "authority": result.authority,
        "line_start": result.line_start,
        "line_end": result.line_end,
        "symbol": result.symbol,
        "retriever": result.retriever,
        "evidence_role": result.metadata.get("evidence_role", ""),
        "live_verified": bool(result.metadata.get("live_verification")),
        "metadata": _public_metadata(result.metadata),
    }


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    """One public evidence block with its stable retrieval citation.

    ``retrieved_evidence`` contains the complete Top-K list, while
    ``generation_context`` contains the evidence entries selected for
    generation. Its public payload keeps each full chunk for review, while
    prompt construction may truncate chunk text to the configured character
    budget. Both lists retain the original retrieval ID so the prompt, answer
    and public response agree on the meaning of ``[E#]``.
    """

    citation: EvidenceCitation
    result: EngineeringSearchResult

    def to_dict(self) -> dict[str, Any]:
        payload = _public_result(self.result)
        payload["citation_id"] = self.citation.citation_id
        payload["citation_detail"] = self.citation.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class GenerationStatus:
    """Machine-readable answer-generation state without provider secrets."""

    status: str
    attempted: bool
    succeeded: bool
    provider: str
    model: str | None = None
    failure_code: str | None = None
    retrieved_count: int = 0
    context_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnswerOutcome:
    query: str
    intent: SourceIntent
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[EvidenceCitation]
    warnings: list[str] = field(default_factory=list)
    generation_mode: str = "deterministic"
    generation_provider: str = "deterministic"
    generation_model: str | None = None
    retrieved_evidence: list[RetrievedEvidence] = field(default_factory=list)
    generation_context: list[RetrievedEvidence] = field(default_factory=list)
    answer_citations: list[EvidenceCitation] = field(default_factory=list)
    generation: GenerationStatus | None = None

    def to_dict(self) -> dict[str, Any]:
        generation = self.generation or GenerationStatus(
            status=(
                "model"
                if self.generation_mode == "model"
                else "fallback"
                if self.generation_mode == "deterministic_fallback"
                else "refusal"
                if self.generation_mode == "refusal"
                else "evidence_only"
            ),
            attempted=self.generation_mode in {"model", "deterministic_fallback"},
            succeeded=self.generation_mode == "model",
            provider=self.generation_provider,
            model=self.generation_model,
            retrieved_count=len(self.retrieved_evidence),
            context_count=len(self.generation_context),
        )
        return {
            "schema_version": "engineering-answer/v2",
            "query": self.query,
            "intent": self.intent.value,
            "answer": self.answer,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "warnings": list(self.warnings),
            "generation_mode": self.generation_mode,
            "generation_provider": self.generation_provider,
            "generation_model": self.generation_model,
            "generation": generation.to_dict(),
            "retrieved_evidence": [
                evidence.to_dict() for evidence in self.retrieved_evidence
            ],
            "generation_context": [
                evidence.to_dict() for evidence in self.generation_context
            ],
            "answer_citations": [
                citation.to_dict() for citation in self.answer_citations
            ],
            # ``citations`` is the v1 compatibility field.  It continues to
            # describe all evidence returned with the answer; v2 consumers
            # should use ``answer_citations`` for claims made by the model.
            "citations": [citation.to_dict() for citation in self.citations],
        }
