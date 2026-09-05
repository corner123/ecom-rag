"""Project selected retrieval hits into minimal, replayable Evidence."""
from __future__ import annotations

import json
from typing import Any

from trade_agent.evidence.models import Evidence, RetrievalComponentProvenance, RetrievalEvidenceLocator, RetrievalProvenance, retrieval_evidence_id
from trade_agent.index.milvus_store import collection_name_for_build_id
from trade_agent.retrieval.service import RetrievalOutcome


_SAFE_RECORD_ID_KEYS = ("product_id", "story_id", "post_id")


def _dimension_values(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else tuple(value) if isinstance(value, (list, tuple)) else None
    if values is None or any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"retrieval aggregation {field} must contain nonblank strings")
    return tuple(sorted(set(values)))


def _source_record_id(metadata: Any) -> str | None:
    if metadata.raw_record_id:
        return metadata.raw_record_id
    raw = metadata.source_locator.raw
    if not isinstance(raw, dict):
        return metadata.source_locator.post_id
    for key in _SAFE_RECORD_ID_KEYS:
        value = raw.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip():
            return str(value)
    return metadata.source_locator.post_id


def _selection_key(evidence: Evidence) -> tuple[int, int, str]:
    provenance = evidence.retrieval_provenance
    if provenance is None:  # Defensive: this helper is retrieval-only.
        raise ValueError("retrieval Evidence is missing retrieval provenance")
    canonical = json.dumps(provenance.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return provenance.rank, provenance.pre_rerank_rank, canonical


def normalize_retrieval(outcome: RetrievalOutcome) -> list[Evidence]:
    """Normalize final hits while keeping raw documents and arbitrary metadata private."""
    if type(outcome) is not RetrievalOutcome:
        raise TypeError("retrieval normalization requires an exact RetrievalOutcome")
    collection_name = collection_name_for_build_id(outcome.build_id)
    by_id: dict[str, Evidence] = {}
    for hit in outcome.hits:
        metadata = hit.record.metadata
        if hit.trace.build_id != outcome.build_id:
            raise ValueError("retrieval hit trace belongs to a foreign build")
        if hit.chunk_id != metadata.chunk_id:
            raise ValueError("retrieval hit chunk identity does not match its record")
        source_url = str(metadata.source_url) if metadata.source_url else None
        canonical_url = str(metadata.canonical_url) if metadata.canonical_url else None
        source_identity = source_url or canonical_url or f"urn:trade-agent:document:{metadata.document_id}"
        record_id = _source_record_id(metadata)
        fact_type = metadata.fact_type.value if metadata.fact_type is not None else "unknown"
        source_type = metadata.source_type.value
        evidence_id = retrieval_evidence_id(
            build_id=outcome.build_id,
            collection_name=collection_name,
            source_identity=source_identity,
            source_type=source_type,
            document_id=metadata.document_id,
            chunk_id=metadata.chunk_id,
            content_hash=metadata.content_hash,
            fact_type=fact_type,
        )
        locator = metadata.source_locator
        replay = RetrievalEvidenceLocator(
            manifest_id=outcome.build_id,
            build_id=outcome.build_id,
            collection_name=collection_name,
            source_identity=source_identity,
            document_id=metadata.document_id,
            chunk_id=metadata.chunk_id,
            chunk_index=metadata.chunk_index,
            content_hash=metadata.content_hash,
            source_record_id=record_id,
            page=locator.page,
            page_end=locator.page_end,
            block=locator.block,
            table=locator.table,
            section=locator.section,
            post_id=locator.post_id,
            row=locator.row,
            profile=locator.profile,
        )
        components = tuple(
            RetrievalComponentProvenance(
                retriever=name,
                rank=value.rank,
                raw_score=value.raw_score,
                retriever_weight=value.retriever_weight,
                relevance_contribution=value.relevance_contribution,
            )
            for name, value in sorted(hit.trace.components.items())
        )
        provenance = RetrievalProvenance(
            build_id=outcome.build_id,
            collection_name=collection_name,
            profile_id=hit.trace.profile_id,
            profile_version=hit.trace.profile_version,
            planner_version=hit.trace.planner_version,
            filter_expression_version=hit.trace.filter_expression_version,
            rank=hit.rank,
            pre_rerank_rank=hit.trace.pre_rerank_rank,
            components=components,
            fusion_score=hit.trace.fusion_score,
            source_prior=hit.trace.source_prior,
            rerank_score=hit.trace.rerank_score,
            entity_resolution_status=hit.trace.entity_resolution.status,
            entity_resolution_id=hit.trace.entity_resolution.entity_id,
            entity_resolution_reason=hit.trace.entity_resolution.reason,
            entity_resolution_confidence=hit.trace.entity_resolution.confidence,
            dedupe_cluster_id=hit.trace.dedupe_cluster_id,
            duplicate_chunk_ids=tuple(sorted(set(hit.trace.duplicate_chunk_ids))),
            dedupe_reasons=tuple(sorted(hit.trace.dedupe_reasons)),
            degraded_components=tuple(sorted(set((*outcome.degradation, *hit.trace.degradation)))),
        )
        aggregation = metadata.aggregation_info or {}
        grain = _dimension_values(aggregation.get("aggregation_grain"), "aggregation_grain")
        confidence = metadata.ocr_confidence if metadata.ocr_confidence is not None else 1.0
        evidence = Evidence(
            evidence_id=evidence_id,
            entity_id=hit.trace.entity_resolution.entity_id or metadata.entity_id,
            company_name=metadata.company_name,
            country_code=metadata.country_code,
            hs_code=metadata.hs_code,
            fact_type=fact_type,
            source_type=source_type,
            source_id=source_identity,
            source_weight=metadata.source_weight,
            content=hit.record.content,
            source_url=source_url,
            canonical_url=canonical_url,
            locator=replay,
            raw_record_id=record_id,
            publish_time=metadata.publish_time,
            valid_from=metadata.valid_from,
            valid_to=metadata.valid_to,
            time_grain="month" if aggregation.get("calendar_month") else "aggregate" if grain else "document",
            currencies=_dimension_values(aggregation.get("currencies", aggregation.get("currency")), "currency"),
            units=_dimension_values(aggregation.get("units", aggregation.get("unit")), "unit"),
            aggregation_grain=grain,
            confidence=confidence,
            confidence_basis="ocr_extraction" if metadata.ocr_confidence is not None else "content_hash_verified",
            is_synthetic=metadata.is_synthetic,
            retrieval_provenance=provenance,
            sql_provenance=None,
        )
        existing = by_id.get(evidence_id)
        if existing is None or _selection_key(evidence) < _selection_key(existing):
            by_id[evidence_id] = evidence
    return sorted(by_id.values(), key=lambda item: (_selection_key(item), item.evidence_id))
