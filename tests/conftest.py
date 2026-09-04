"""Real Task 8 bundle shared by integration and Milvus verification."""
import os
import pytest


@pytest.fixture(scope="session")
def full_index_bundle(tmp_path_factory):
    from pymilvus import MilvusClient
    from scripts.smoke_milvus_roundtrip import isolated_build
    from scripts.index_trade_corpus import _manager
    from trade_agent.index.builder import TradeIndexBuilder
    from trade_agent.index.milvus_store import TradeMilvusStore

    directory = tmp_path_factory.mktemp("full-trade-index")
    build = isolated_build(directory)
    manager = _manager()
    client = MilvusClient(uri=os.environ.get("MILVUS_TEST_URI", "http://127.0.0.1:19530"))
    store = TradeMilvusStore(client=client, embedding_manager=manager)
    bundle = None
    try:
        bundle = TradeIndexBuilder(output_dir=directory / "indexes", milvus=store,
                                   embedding_manager=manager).build(build)
        yield bundle, store, manager
    finally:
        if bundle is not None:
            store.drop_owned_collection(bundle.descriptor.collection_contract)
            assert not client.has_collection(bundle.descriptor.collection_contract.collection_name)
        client.close()
