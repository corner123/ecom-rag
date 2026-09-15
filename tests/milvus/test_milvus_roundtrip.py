from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from uuid import uuid4

import pytest
import yaml


pytestmark = [pytest.mark.milvus, pytest.mark.integration]


@pytest.fixture(scope="module")
def isolated_canonical_build(tmp_path_factory: pytest.TempPathFactory):
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog

    root = Path(__file__).resolve().parents[2]
    source_catalog = yaml.safe_load(
        (root / "data/sources/trade_intel_demo.yaml").read_text(encoding="utf-8")
    )
    source_catalog["root"] = (root / "demo/trade_intel_seed").as_posix()
    source_catalog["synthetic_notice"] += f" ISOLATED-MILVUS-{uuid4().hex}"
    directory = tmp_path_factory.mktemp("milvus-build")
    catalog_path = directory / "catalog.yaml"
    catalog_path.write_text(
        yaml.safe_dump(source_catalog, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    build = IngestionPipeline().run(
        SourceCatalog.from_yaml(catalog_path),
        directory / "build.json",
    )
    assert len(build.chunks) == 120
    assert build.metadata_complete is True
    assert all(chunk.restore().metadata.is_synthetic for chunk in build.chunks)
    return build


def _manager():
    from scripts.smoke_embeddings import model_settings_from_environment
    from trade_agent.index.embeddings import BgeEmbeddingManager

    models = model_settings_from_environment()
    return BgeEmbeddingManager(
        cache_folder=models.embedding_cache_dir,
        local_files_only=True,
        batch_size=models.embedding_batch_size,
        device=models.embedding_device,
    )


@pytest.mark.milvus
def test_real_milvus_phase_a_contract_create_reconnect_and_cleanup(
    isolated_canonical_build,
) -> None:
    from pymilvus import MilvusClient

    from trade_agent.index.milvus_store import (
        CONTRACT_COLLECTION_NAME,
        CollectionContractMismatch,
        MATERIALIZED_FIELDS,
        TradeMilvusStore,
    )

    uri = os.environ.get("MILVUS_TEST_URI", "http://127.0.0.1:19530")
    build = isolated_canonical_build
    manager = _manager()
    manager.embed_query("phase A production contract probe")
    manager.contract.require_production()
    client = MilvusClient(uri=uri)
    store = TradeMilvusStore(client=client, embedding_manager=manager)
    expected = None
    try:
        expected = store.create(build, manager.contract)
        names = set(client.list_collections())
        assert expected.collection_name in names
        assert CONTRACT_COLLECTION_NAME in names

        stats = store.validate(expected, require_complete=False)
        assert stats.row_count == 0
        assert stats.contract_state == "created"
        assert stats.contract_source == "companion_collection"
        assert stats.contract_sha256 == expected.contract_sha256
        assert stats.contract_row_count == 1
        assert stats.field_names == MATERIALIZED_FIELDS
        assert stats.index_type == "HNSW"
        assert stats.metric_type == "COSINE"
        assert stats.hnsw_m == 32
        assert stats.hnsw_ef_construction == 256
        assert stats.loaded is True

        persisted = client.query(
            collection_name=CONTRACT_COLLECTION_NAME,
            ids=[expected.collection_name],
            output_fields=[
                "collection_name",
                "owner",
                "schema_version",
                "contract_sha256",
                "contract_json",
                "state",
            ],
            consistency_level="Strong",
        )
        assert len(persisted) == 1
        assert persisted[0]["owner"] == "trade-agent"
        assert persisted[0]["state"] == "created"
        assert persisted[0]["contract_sha256"] == expected.contract_sha256
        assert (
            persisted[0]["contract_json"]
            == expected.canonical_json()
        )

        client.close()
        reconnect_client = MilvusClient(uri=uri)
        reconnect = TradeMilvusStore(
            client=reconnect_client,
            embedding_manager=manager,
        )
        reopened = reconnect.open(build, manager.contract)
        assert reopened.contract_sha256 == expected.contract_sha256
        assert reconnect.validate(reopened, require_complete=False).row_count == 0

        mismatch = reopened.model_copy(update={"default_ef_search": 256})
        with pytest.raises(CollectionContractMismatch):
            reconnect.validate(mismatch, require_complete=False)

        reconnect.drop_owned_collection(reopened)
        assert not reconnect_client.has_collection(reopened.collection_name)
        assert reconnect_client.query(
            collection_name=CONTRACT_COLLECTION_NAME,
            ids=[reopened.collection_name],
            output_fields=["collection_name"],
            consistency_level="Strong",
        ) == []
        reconnect_client.close()
        expected = None
    finally:
        if expected is not None:
            store.drop_owned_collection(expected, allow_incomplete_created_here=True)
        try:
            client.close()
        except Exception:
            pass


@pytest.mark.milvus
def test_real_milvus_120_chunk_filter_first_reconnect_and_cleanup(
    isolated_canonical_build,
) -> None:
    from pymilvus import MilvusClient

    from trade_agent.index.milvus_store import (
        CONTRACT_COLLECTION_NAME,
        TradeMilvusStore,
    )
    from trade_agent.retrieval.filters import RetrievalFilter

    uri = os.environ.get("MILVUS_TEST_URI", "http://127.0.0.1:19530")
    build = isolated_canonical_build
    chunks = [snapshot.restore() for snapshot in build.chunks]
    manager = _manager()
    probe = manager.embed_query("HS 850440 synthetic buyer growth")
    manager.contract.require_production()

    client = MilvusClient(uri=uri)
    store = TradeMilvusStore(client=client, embedding_manager=manager)
    expected = None
    try:
        expected = store.create(build, manager.contract)
        collection_names = set(client.list_collections())
        assert expected.collection_name in collection_names
        assert CONTRACT_COLLECTION_NAME in collection_names

        write = store.replace_chunks(chunks)
        assert write.status == "inserted"
        assert write.inserted_count == 120
        assert write.row_count == 120
        assert write.contract_sha256 == expected.contract_sha256

        repeated = store.replace_chunks(list(reversed(chunks)))
        assert repeated.status == "noop"
        assert repeated.inserted_count == 0
        assert repeated.row_count == 120

        stats = store.validate(expected)
        assert stats.row_count == 120
        assert stats.contract_source == "companion_collection"
        assert stats.contract_sha256 == expected.contract_sha256
        assert stats.index_type == "HNSW"
        assert stats.metric_type == "COSINE"
        assert stats.hnsw_m == 32
        assert stats.hnsw_ef_construction == 256
        assert stats.loaded is True
        assert stats.exact_chunk_ids is True

        unfiltered = store.search(probe, top_k=5, filter_=None)
        assert len(unfiltered) == 5
        assert [hit.rank for hit in unfiltered] == [1, 2, 3, 4, 5]
        assert all(hit.build_id == build.build_id for hit in unfiltered)
        assert all(hit.filter_expression == "" for hit in unfiltered)

        candidates = [
            chunk
            for chunk in chunks
            if chunk.metadata.region
            and chunk.metadata.hs_code
            and chunk.metadata.source_type.value == "customs_profile"
        ]
        assert candidates
        selected = Counter(
            (
                chunk.metadata.region,
                chunk.metadata.hs_code,
                chunk.metadata.source_type,
            )
            for chunk in candidates
        ).most_common(1)[0][0]
        region, hs_code, source_type = selected
        query_chunk = next(
            chunk
            for chunk in candidates
            if (
                chunk.metadata.region,
                chunk.metadata.hs_code,
                chunk.metadata.source_type,
            )
            == selected
        )
        filtered = store.search(
            manager.embed_query(query_chunk.content),
            top_k=5,
            filter_=RetrievalFilter(
                region=region,
                hs_codes=[hs_code],
                source_types=[source_type],
                is_synthetic=True,
            ),
        )
        assert filtered
        assert all(hit.metadata.region == region for hit in filtered)
        assert all(hit.metadata.hs_code == hs_code for hit in filtered)
        assert all(hit.metadata.source_type == source_type for hit in filtered)
        assert all(hit.metadata.is_synthetic is True for hit in filtered)
        assert all("{" in hit.filter_expression for hit in filtered)
        assert all(hit.filter_expression_version == "trade-filter-v1" for hit in filtered)

        client.close()
        reconnected_client = MilvusClient(uri=uri)
        reconnected = TradeMilvusStore(
            client=reconnected_client,
            embedding_manager=manager,
        )
        reopened = reconnected.open(build, manager.contract)
        assert reopened.contract_sha256 == expected.contract_sha256
        assert reconnected.validate(reopened).row_count == 120
        after_reconnect = reconnected.search(probe, top_k=3, filter_=None)
        assert len(after_reconnect) == 3

        reconnected.drop_owned_collection(reopened)
        assert not reconnected_client.has_collection(reopened.collection_name)
        contract_rows = reconnected_client.query(
            collection_name=CONTRACT_COLLECTION_NAME,
            ids=[reopened.collection_name],
            output_fields=["collection_name"],
            consistency_level="Strong",
        )
        assert contract_rows == []
        reconnected_client.close()
        expected = None
    finally:
        if expected is not None:
            # Cleanup remains scoped to the exact in-memory build contract.
            store.drop_owned_collection(expected, allow_incomplete_created_here=True)
        try:
            client.close()
        except Exception:
            pass
