"""Build and atomically publish one immutable Milvus/BM25 trade index bundle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    from pymilvus import MilvusClient
    from scripts.index_trade_corpus import _manager
    from trade_agent.data.manifest import BuildManifest
    from trade_agent.index.builder import TradeIndexBuilder
    from trade_agent.index.milvus_store import TradeMilvusStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/indexes"))
    parser.add_argument("--milvus-uri", default=os.environ.get("MILVUS_TEST_URI", "http://127.0.0.1:19530"))
    args = parser.parse_args(argv)
    client = None
    try:
        build = BuildManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
        manager = _manager()
        client = MilvusClient(uri=args.milvus_uri)
        store = TradeMilvusStore(client=client, embedding_manager=manager)
        existing = (args.output_dir / build.build_id / "bundle.json").exists()
        bundle = TradeIndexBuilder(output_dir=args.output_dir, milvus=store,
                                   embedding_manager=manager).build(build)
        print(json.dumps({"status": "ok", "build_id": build.build_id,
                          "collection_name": bundle.descriptor.collection_contract.collection_name,
                          "bundle": str(bundle.descriptor_path), "chunk_count": len(build.chunks),
                          "write_status": "verified_noop" if existing else "published"},
                         ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "error", "error": type(error).__name__}, sort_keys=True))
        return 1
    finally:
        if client is not None:
            client.close()

if __name__ == "__main__":
    raise SystemExit(main())
