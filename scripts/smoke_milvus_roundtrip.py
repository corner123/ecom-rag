"""Build, reopen and query one isolated full trade index, then clean it up."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

import yaml


def isolated_build(directory: Path):
    from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog
    root = Path(__file__).resolve().parents[1]
    catalog = yaml.safe_load((root / "data/sources/trade_intel_demo.yaml").read_text(encoding="utf-8"))
    catalog["root"] = (root / "demo/trade_intel_seed").as_posix()
    catalog["synthetic_notice"] += f" TASK8-ROUNDTRIP-{uuid4().hex}"
    catalog_path = directory / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False), encoding="utf-8")
    build = IngestionPipeline().run(SourceCatalog.from_yaml(catalog_path), directory / "manifest.json")
    if len(build.chunks) != 120 or not build.metadata_complete:
        raise ValueError("round-trip corpus must contain 120 metadata-complete chunks")
    return build


def run_roundtrip(*, uri: str):
    from pymilvus import MilvusClient
    from scripts.index_trade_corpus import _manager
    from trade_agent.index.builder import TradeIndexBuilder, TradeIndexBundle
    from trade_agent.index.milvus_store import TradeMilvusStore
    from scripts.smoke_embeddings import model_settings_from_environment
    from trade_agent.retrieval import QueryIntent, load_retrieval_profile
    from trade_agent.retrieval.reranker import BgeReranker
    from trade_agent.retrieval.service import RetrievalService

    with tempfile.TemporaryDirectory(prefix="trade-index-roundtrip-") as name:
        directory = Path(name)
        build = isolated_build(directory)
        manager = _manager()
        client = MilvusClient(uri=uri)
        store = TradeMilvusStore(client=client, embedding_manager=manager)
        bundle = None
        try:
            bundle = TradeIndexBuilder(output_dir=directory / "indexes", milvus=store,
                                       embedding_manager=manager).build(build)
            fresh_store = TradeMilvusStore(client=client, embedding_manager=manager)
            reloaded = TradeIndexBundle.load(bundle.descriptor_path, milvus=fresh_store,
                                              embedding_manager=manager)
            model_settings = model_settings_from_environment()
            reranker = BgeReranker(
                device=model_settings.embedding_device,
                cache_dir=model_settings.embedding_cache_dir,
                local_files_only=model_settings.embedding_offline,
                batch_size=model_settings.embedding_batch_size,
            )
            service = RetrievalService(build=reloaded.build, bm25=reloaded.bm25,
                milvus=fresh_store, embedding_manager=manager, profile=load_retrieval_profile(),
                reranker=reranker)
            result = service.search(QueryIntent(query="HS850440 charger procurement",
                                                hs_codes=("850440",)), top_k=3, candidate_limit=10)
            if not result.hits or any(h.metadata.hs_code != "850440" for h in result.hits):
                raise RuntimeError("round-trip retrieval did not return filtered evidence")
            if result.rerank_degraded or result.rerank_contract is None or any(h.trace.rerank_score is None for h in result.hits):
                raise RuntimeError("round-trip retrieval did not exercise real reranking")
            return {"status": "ok", "build_id": build.build_id,
                    "collection_name": bundle.descriptor.collection_contract.collection_name,
                    "row_count": len(build.chunks), "hit_count": len(result.hits),
                    "reranker": result.rerank_contract.provider}
        finally:
            if bundle is not None:
                store.drop_owned_collection(bundle.descriptor.collection_contract)
                if client.has_collection(bundle.descriptor.collection_contract.collection_name):
                    raise RuntimeError("temporary collection cleanup failed")
            client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milvus-uri", default=os.environ.get("MILVUS_TEST_URI", "http://127.0.0.1:19530"))
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_roundtrip(uri=args.milvus_uri), ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "error", "error": type(error).__name__}, sort_keys=True))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
