"""Verify durable bundle identities through a fresh Milvus connection."""
import os
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.milvus, pytest.mark.model]


def test_full_bundle_reconnect_has_exact_frozen_dense_sparse_ids(full_index_bundle):
    from pymilvus import MilvusClient
    from trade_agent.index.builder import TradeIndexBundle
    from trade_agent.index.milvus_store import TradeMilvusStore
    bundle, _, manager = full_index_bundle
    client = MilvusClient(uri=os.environ.get("MILVUS_TEST_URI", "http://127.0.0.1:19530"))
    try:
        store = TradeMilvusStore(client=client, embedding_manager=manager)
        reloaded = TradeIndexBundle.load(bundle.descriptor_path, milvus=store, embedding_manager=manager)
        rows = client.query(collection_name=store.contract.collection_name, filter="",
                            output_fields=["chunk_id"], limit=1000, consistency_level="Strong")
        sparse_ids = {r.metadata.chunk_id for r in reloaded.bm25.records}
        assert len(rows) == len(sparse_ids) == 120
        assert {r["chunk_id"] for r in rows} == sparse_ids == set(bundle.build.chunk_ids)
        assert reloaded.descriptor == bundle.descriptor
    finally:
        client.close()
