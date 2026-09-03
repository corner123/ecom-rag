from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError


def _one_chunk_build(tmp_path: Path):
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    root = tmp_path / "corpus"
    (root / "manifests").mkdir(parents=True)
    source = root / "site.html"
    source.write_text(
        "<h1>Example exporter</h1><p>Verified synthetic trade evidence.</p>",
        encoding="utf-8",
    )
    (root / "manifests" / "corpus_manifest.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "path": "site.html",
                        "content_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "expected_entity": "Example exporter",
                        "fact_type": "company_status",
                        "file_type": "html",
                        "source_type": "official_website",
                        "ingested_at": "2026-08-30T00:00:00+00:00",
                        "publish_time": "2026-08-30T00:00:00+00:00",
                        "valid_from": "2026-08-30T00:00:00+00:00",
                        "is_synthetic": True,
                        "locator": {"section": "overview"},
                        "reference_claim_ids": ["CLAIM-1"],
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        "\n".join(
            [
                "catalog_version: 1",
                f"root: {root.as_posix()}",
                "synthetic_notice: test-only",
                "sources:",
                "  - source_type: official_website",
                "    file_types: [html]",
                "    paths: [site.html]",
                "    url_pattern: https://company-*.example",
            ]
        ),
        encoding="utf-8",
    )
    catalog = SourceCatalog.from_yaml(catalog_path)
    return IngestionPipeline().run(catalog, tmp_path / "build.json")


def test_collection_name_uses_only_the_first_twenty_build_hex_characters() -> None:
    from trade_agent.index.milvus_store import collection_name_for_build_id

    assert collection_name_for_build_id("build_0123456789abcdef0123456789abcdef") == (
        "trade_intel_chunks_0123456789abcdef0123"
    )
    for invalid in (
        "0123456789abcdef0123456789abcdef",
        "build_0123456789ABCDEF0123456789abcdef",
        "build_0123",
        "build_0123456789abcdef0123456789abcdef00",
    ):
        with pytest.raises(ValueError):
            collection_name_for_build_id(invalid)


def test_retrieval_filter_compiles_every_allowlisted_field_deterministically() -> None:
    from trade_agent.retrieval.filters import RetrievalFilter, compile_filter

    filter_ = RetrievalFilter(
        region="North America",
        country_codes=["us", "CA", "us"],
        hs_codes=["850440", "730890", "850440"],
        entity_ids=["entity_02", "entity_01"],
        source_types=["official_website", "customs_profile"],
        fact_types=["company_status", "trade_activity"],
        published_after=datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        published_before=datetime(2026, 2, 3, 4, 5, 6, tzinfo=timezone.utc),
        is_synthetic=True,
    )

    assert compile_filter(filter_) == (
        'region == "North America"'
        ' and country_code in ["CA","US"]'
        ' and hs_code in ["730890","850440"]'
        ' and entity_id in ["entity_01","entity_02"]'
        ' and source_type in ["customs_profile","official_website"]'
        ' and fact_type in ["company_status","trade_activity"]'
        " and publish_time_epoch >= 1735787045"
        " and publish_time_epoch <= 1770091506"
        " and is_synthetic == true"
    )


def test_filter_rejects_raw_expressions_unknown_fields_and_injection_values() -> None:
    from trade_agent.retrieval.filters import RetrievalFilter, compile_filter

    with pytest.raises(ValidationError):
        RetrievalFilter.model_validate({"expr": 'region == "Europe"'})
    with pytest.raises(ValidationError):
        RetrievalFilter(entity_ids=['entity_1" or is_synthetic == false'])
    with pytest.raises(ValidationError):
        RetrievalFilter(region="Europe\x00")
    with pytest.raises(TypeError):
        compile_filter('region == "Europe"')  # type: ignore[arg-type]


def test_filter_execution_uses_typed_template_parameters_not_literal_syntax() -> None:
    from trade_agent.retrieval.filters import RetrievalFilter, compile_filter_binding

    compiled = compile_filter_binding(
        RetrievalFilter(
            region="North America",
            hs_codes=["850440"],
            source_types=["customs_profile"],
        )
    )
    assert compiled.expression == (
        "region == {region_0} and hs_code in {hs_code_0} "
        "and source_type in {source_type_0}"
    )
    assert compiled.parameters == {
        "region_0": "North America",
        "hs_code_0": ["850440"],
        "source_type_0": ["customs_profile"],
    }
    assert compiled.audit_expression == (
        'region == "North America" and hs_code in ["850440"] '
        'and source_type in ["customs_profile"]'
    )


def test_filter_deduplicates_mixed_source_type_strings_and_enums() -> None:
    from trade_agent.retrieval.filters import RetrievalFilter
    from trade_agent.schemas.source import FactType, SourceType

    filter_ = RetrievalFilter(
        source_types=["customs_profile", SourceType.CUSTOMS_PROFILE],
        fact_types=["trade_activity", FactType.TRADE_ACTIVITY],
    )
    assert filter_.source_types == (SourceType.CUSTOMS_PROFILE,)
    assert filter_.fact_types == (FactType.TRADE_ACTIVITY,)


def test_filter_rejects_naive_or_inverted_publication_ranges() -> None:
    from trade_agent.retrieval.filters import RetrievalFilter

    with pytest.raises(ValidationError):
        RetrievalFilter(published_after=datetime(2026, 1, 1))
    with pytest.raises(ValidationError):
        RetrievalFilter(
            published_after=datetime(2026, 2, 1, tzinfo=timezone.utc),
            published_before=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


def test_filter_allows_cjk_regions_and_turns_bad_sequence_items_into_validation_errors() -> None:
    from trade_agent.retrieval.filters import RetrievalFilter, compile_filter_binding

    compiled = compile_filter_binding(RetrievalFilter(region="东亚"))
    assert compiled.parameters == {"region_0": "东亚"}
    with pytest.raises(ValidationError):
        RetrievalFilter.model_validate({"country_codes": [{"not": "a string"}]})


def test_hnsw_parameters_are_typed_bounded_and_use_reviewed_defaults() -> None:
    from trade_agent.index.milvus_store import HnswConfig

    defaults = HnswConfig()
    assert defaults.model_dump() == {
        "m": 32,
        "ef_construction": 256,
        "ef_search": 128,
    }
    for payload in (
        {"m": 3},
        {"m": 65},
        {"ef_construction": 7},
        {"ef_construction": 513},
        {"ef_search": 0},
        {"ef_search": 1025},
        {"m": "32"},
    ):
        with pytest.raises(ValidationError):
            HnswConfig.model_validate(payload)


def test_canonical_url_hash_has_documented_deterministic_fallback_order() -> None:
    from trade_agent.index.milvus_store import canonical_url_hash

    assert canonical_url_hash(
        canonical_url="https://company.example/canonical",
        source_url="https://company.example/source",
        document_id="doc_1",
    ) == "7be8c03cc3b1575d26bf525eff3c8c44c193c3b6923a851d9153937f0db5fe1b"
    assert canonical_url_hash(
        canonical_url=None,
        source_url="https://company.example/source",
        document_id="doc_1",
    ) == "eeec9f51e556c806c65078c7644e411f3397a8241ab0f1c467044faee11e2eec"
    assert canonical_url_hash(
        canonical_url=None,
        source_url=None,
        document_id="doc_1",
    ) == "495cbcc5553d9516954fd1aa31585184eb95deff84f0b01281e90009da49c334"


def test_collection_contract_hash_covers_build_model_index_and_schema_identity() -> None:
    from trade_agent.index.milvus_store import CollectionContract

    payload = {
        "owner": "trade-agent",
        "schema_version": "trade-milvus-v1",
        "collection_name": "trade_intel_chunks_0123456789abcdef0123",
        "build_id": "build_0123456789abcdef0123456789abcdef",
        "build_fingerprint": "a" * 64,
        "build_chunk_count": 120,
        "build_chunk_ids_sha256": "b" * 64,
        "metadata_schema_version": "task7-source-metadata-v2",
        "embedding_provider": "sentence-transformers",
        "embedding_model": "BAAI/bge-m3",
        "embedding_requested_revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "embedding_resolved_revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "embedding_dimension": 1024,
        "embedding_normalized": True,
        "embedding_dtype": "float32",
        "embedding_library_version": "3.4.1",
        "embedding_artifact_manifest_sha256": "3a862f1d0a8543acc13e9faa5e6d6d1f916ee609b264960be8337e6ba509856b",
        "embedding_verified_artifact_count": 10,
        "vector_field": "dense_vector",
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "hnsw_m": 32,
        "hnsw_ef_construction": 256,
        "default_ef_search": 128,
        "filter_expression_version": "trade-filter-v1",
        "null_string_sentinel": "__trade_agent_null_v1__",
        "null_epoch_sentinel": -(2**63),
        "materialized_fields": (
            "chunk_id",
            "dense_vector",
            "text",
            "document_id",
            "entity_id",
            "country_code",
            "region",
            "hs_code",
            "source_type",
            "source_weight",
            "fact_type",
            "publish_time_epoch",
            "valid_to_epoch",
            "file_type",
            "is_synthetic",
            "canonical_url_hash",
            "dedupe_cluster_id",
            "metadata_json",
        ),
    }
    contract = CollectionContract.model_validate(payload)
    first = contract.contract_sha256
    assert len(first) == 64
    assert CollectionContract.model_validate_json(contract.canonical_json()).contract_sha256 == first
    changed = contract.model_copy(update={"default_ef_search": 256})
    assert changed.contract_sha256 != first


def test_deterministic_embedding_contract_cannot_reach_milvus_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_agent.index.embeddings import BgeEmbeddingManager
    from trade_agent.index.milvus_store import TradeMilvusStore

    class Encoder:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 1024), dtype=np.float32)

    class ClientMustNotBeCalled:
        def __getattr__(self, name):
            raise AssertionError(f"Milvus client was called: {name}")

    monkeypatch.setenv("TEST_EMBEDDING_PROVIDER", "deterministic")
    manager = BgeEmbeddingManager(test_mode=True, test_encoder=Encoder())
    manager.embed_query("contract probe")
    store = TradeMilvusStore(client=ClientMustNotBeCalled(), embedding_manager=manager)
    with pytest.raises(ValueError, match="live verified production"):
        store.create(object(), manager.contract)  # type: ignore[arg-type]


def test_build_chunk_validation_requires_the_exact_complete_frozen_set(tmp_path: Path) -> None:
    from trade_agent.index.milvus_store import validate_build_chunks

    build = _one_chunk_build(tmp_path)
    canonical = [snapshot.restore() for snapshot in build.chunks]
    assert validate_build_chunks(build, canonical) == tuple(canonical)
    with pytest.raises(ValueError, match="exactly"):
        validate_build_chunks(build, [])
    with pytest.raises(ValueError, match="duplicate"):
        validate_build_chunks(build, canonical + canonical)
    from trade_agent.schemas.source import ChunkRecord, content_sha256

    changed_content = canonical[0].content + " tampered"
    changed_metadata = canonical[0].metadata.model_copy(
        update={"content_hash": content_sha256(changed_content)}
    )
    changed = ChunkRecord(content=changed_content, metadata=changed_metadata)
    with pytest.raises(ValueError, match="frozen manifest"):
        validate_build_chunks(build, [changed])


def test_materialized_row_has_full_canonical_metadata_and_null_sentinels(tmp_path: Path) -> None:
    from trade_agent.data.manifest import canonical_json
    from trade_agent.index.milvus_store import (
        NULL_EPOCH_SENTINEL,
        NULL_STRING_SENTINEL,
        materialize_chunk,
    )

    chunk = _one_chunk_build(tmp_path).chunks[0].restore()
    vector = np.zeros(1024, dtype=np.float32)
    vector[0] = 1.0
    row = materialize_chunk(chunk, vector)
    assert row["chunk_id"] == chunk.metadata.chunk_id
    assert row["dense_vector"] == vector.tolist()
    assert row["metadata_json"] == canonical_json(chunk.metadata.model_dump(mode="json"))
    assert json.loads(row["metadata_json"])["source_locator"] == chunk.metadata.source_locator.model_dump(mode="json")
    for field in ("entity_id", "country_code", "region", "hs_code", "fact_type", "dedupe_cluster_id"):
        if getattr(chunk.metadata, field) is None:
            assert row[field] == NULL_STRING_SENTINEL
    if chunk.metadata.valid_to is None:
        assert row["valid_to_epoch"] == NULL_EPOCH_SENTINEL


@pytest.mark.parametrize(
    "vector",
    [
        np.zeros((1, 1024), dtype=np.float32),
        np.zeros(1023, dtype=np.float32),
        np.zeros(1024, dtype=np.float64),
        np.full(1024, np.nan, dtype=np.float32),
        np.zeros(1024, dtype=np.float32),
        np.full(1024, 2.0 / np.sqrt(1024), dtype=np.float32),
    ],
)
def test_query_vector_validation_rejects_shape_dtype_finite_and_norm_errors(vector) -> None:
    from trade_agent.index.milvus_store import validate_query_vector

    with pytest.raises((TypeError, ValueError)):
        validate_query_vector(vector, dimension=1024)


def test_query_vector_validation_accepts_only_normalized_float32() -> None:
    from trade_agent.index.milvus_store import validate_query_vector

    vector = np.zeros(1024, dtype=np.float32)
    vector[7] = 1.0
    validated = validate_query_vector(vector, dimension=1024)
    assert validated.dtype == np.float32
    assert validated.shape == (1024,)
