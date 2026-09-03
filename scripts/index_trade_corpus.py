"""Index the checked-in synthetic trade corpus into real Milvus."""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog
from trade_agent.index.milvus_store import TradeMilvusStore
from trade_agent.index.embeddings import BgeEmbeddingManager


def _manager() -> BgeEmbeddingManager:
    settings_module = import_module("scripts.smoke_embeddings")
    settings = settings_module.model_settings_from_environment()
    return BgeEmbeddingManager(
        model_name=settings.embedding_model,
        revision=settings.embedding_revision,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        cache_folder=settings.embedding_cache_dir,
        local_files_only=settings.embedding_offline,
    )


def _client(uri: str) -> Any:
    pymilvus = import_module("pymilvus")
    return pymilvus.MilvusClient(uri=uri)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/sources/trade_intel_demo.yaml"))
    parser.add_argument("--build-output", type=Path, default=Path("artifacts/milvus/canonical-build.json"))
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    args = parser.parse_args(argv)

    try:
        catalog = SourceCatalog.from_yaml(args.catalog)
        args.build_output.parent.mkdir(parents=True, exist_ok=True)
        build = IngestionPipeline().run(catalog, args.build_output)
        if len(build.chunks) != 120 or not build.metadata_complete:
            raise ValueError("canonical corpus must contain 120 metadata-complete chunks")

        manager = _manager()
        probe = manager.embed_query("HS 850440 synthetic buyer growth")
        embedding = manager.contract.require_production()
        client = _client(args.milvus_uri)
        store = TradeMilvusStore(client=client, embedding_manager=manager)
        if client.has_collection(f"trade_intel_chunks_{build.build_id.removeprefix('build_')[:20]}"):
            contract = store.open(build, embedding)
        else:
            contract = store.create(build, embedding)
        write = store.replace_chunks([snapshot.restore() for snapshot in build.chunks])
        stats = store.validate(contract)
        hits = store.search(probe, top_k=1, filter_=None)
        if not hits:
            raise RuntimeError("canonical Milvus search smoke returned no evidence")
        summary = {
            "status": "ok",
            "build_id": build.build_id,
            "collection_name": contract.collection_name,
            "contract_sha256": contract.contract_sha256,
            "write_status": write.status,
            "row_count": stats.row_count,
            "search_rank_1": hits[0].chunk_id,
            "vector_dimension": int(np.asarray(probe).shape[0]),
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(
            json.dumps({"status": "error", "error": type(error).__name__}, sort_keys=True),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
