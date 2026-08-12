"""Command-line entry point for the legacy demo and engineering-knowledge RAG."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv


DEFAULT_MANIFEST = "data/manifests/builds/current.json"
DEFAULT_INDEX = "data/indexes/engineering"


def _json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _legacy_engine():
    from config import RAGConfig
    from rag_core.engine import RAGEngine

    engine = RAGEngine(RAGConfig())
    engine.initialize()
    return engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI 团队多源工程知识 RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query = subparsers.add_parser("query", help="使用原有 RAG 引擎查询")
    query.add_argument("question")
    query.add_argument("--strategy", "-s")
    query.add_argument("--top-k", "-k", type=int, default=5)

    build = subparsers.add_parser("build", help="使用原有数据目录构建索引")
    build.add_argument("--data-dir", "-d")

    evaluate = subparsers.add_parser("eval", help="运行原有 RAG 评测")
    evaluate.add_argument("--eval-data", "-e", required=True)
    evaluate.add_argument("--strategies", "-s", nargs="+")

    subparsers.add_parser("stats", help="查看原有 RAG 引擎状态")

    sync = subparsers.add_parser(
        "sources-sync", help="只读采集 Mini-Nanobot 与白名单官方资料"
    )
    sync.add_argument("--catalog", default="data/sources/catalog.yaml")
    sync.add_argument("--manifest", default=DEFAULT_MANIFEST)
    sync.add_argument("--chunk-size", type=int, default=1200)
    sync.add_argument("--chunk-overlap", type=int, default=120)
    sync.add_argument(
        "--full-rebuild",
        action="store_true",
        help=(
            "ignore an existing manifest and publish a fresh current-schema "
            "snapshot; required when intentionally migrating an older schema"
        ),
    )

    index = subparsers.add_parser(
        "engineering-build", help="从 manifest 构建 internal/official 分区索引"
    )
    index.add_argument("--manifest", default=DEFAULT_MANIFEST)
    index.add_argument("--index-dir", default=DEFAULT_INDEX)
    index.add_argument("--batch-size", type=int, default=128)
    index.add_argument(
        "--backend",
        choices=("faiss", "milvus", "both"),
        default=None,
        help="build portable FAISS only, or require a Milvus mirror as well",
    )

    engineering_query = subparsers.add_parser(
        "engineering-query", help="查询工程知识并返回结构化引用"
    )
    engineering_query.add_argument("question")
    engineering_query.add_argument("--top-k", "-k", type=int, default=5)
    engineering_query.add_argument("--answer", action="store_true")
    engineering_query.add_argument("--index-dir", default=DEFAULT_INDEX)
    engineering_query.add_argument("--mini-repo")
    engineering_query.add_argument(
        "--backend",
        choices=("faiss", "milvus", "auto"),
        default=None,
        help="select the runtime dense-vector backend",
    )

    cleanup = subparsers.add_parser(
        "engineering-milvus-cleanup",
        help="list stale build-scoped Milvus collections; deletion is explicit",
    )
    cleanup.add_argument("--index-dir", default=DEFAULT_INDEX)
    cleanup.add_argument(
        "--execute",
        action="store_true",
        help="delete the dry-run candidates after old readers have drained",
    )
    cleanup.add_argument(
        "--namespace",
        help="explicitly confirm the active Milvus owner namespace when deleting",
    )

    engineering_eval = subparsers.add_parser(
        "engineering-eval", help="运行 BM25/dense/hybrid 消融评测"
    )
    engineering_eval.add_argument(
        "--dataset",
        nargs="+",
        default=[
            "data/eval/mini_nanobot_internal.jsonl",
            "data/eval/official_engineering_specs.jsonl",
        ],
    )
    engineering_eval.add_argument(
        "--factory",
        default=None,
    )
    engineering_eval.add_argument(
        "--output", default="data/eval/reports/engineering_ablation"
    )
    engineering_eval.add_argument(
        "--snapshot", default="data/eval/evaluation_snapshot.json"
    )
    engineering_eval.add_argument("--top-k", type=int, default=5)
    engineering_eval.add_argument("--baseline", default=None)
    engineering_eval.add_argument(
        "--suite", choices=("index", "e2e"), default="index"
    )

    serve = subparsers.add_parser(
        "serve", help="启动只读工程知识 API 与本地 Web 页面"
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Explicit process variables win over the developer-only .env file.
    load_dotenv(override=False)
    args = build_parser().parse_args(argv)

    if args.command == "query":
        result = _legacy_engine().query(
            args.question, strategy=args.strategy, top_k=args.top_k
        )
        _json(result)
        return 0
    if args.command == "build":
        _legacy_engine().build_index(data_dir=args.data_dir)
        print("索引构建完成")
        return 0
    if args.command == "eval":
        results = _legacy_engine().evaluate(
            args.eval_data, strategies=args.strategies
        )
        from rag_core.evaluation import ReportGenerator

        print(ReportGenerator.generate_text_report(results))
        return 0
    if args.command == "stats":
        _json(_legacy_engine().get_stats())
        return 0

    if args.command == "sources-sync":
        from rag_core.engineering import sync_engineering_sources

        result = sync_engineering_sources(
            args.catalog,
            args.manifest,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            full_rebuild=args.full_rebuild,
        )
        _json(
            {
                "build_id": result.manifest.build_id,
                "manifest": str(Path(args.manifest).resolve()),
                "sources": len(result.manifest.sources),
                "documents": len(result.manifest.documents),
                "chunks": len(result.manifest.chunks),
                "diff": {
                    "documents": {
                        "added": len(result.diff.documents.added),
                        "modified": len(result.diff.documents.modified),
                        "deleted": len(result.diff.documents.deleted),
                    },
                    "chunks": {
                        "added": len(result.diff.chunks.added),
                        "modified": len(result.diff.chunks.modified),
                        "deleted": len(result.diff.chunks.deleted),
                    },
                },
            }
        )
        return 0
    if args.command == "engineering-build":
        from rag_core.engineering import build_engineering_index

        index = build_engineering_index(
            args.manifest,
            args.index_dir,
            batch_size=args.batch_size,
            backend=args.backend,
        )
        _json(index.stats())
        return 0
    if args.command == "engineering-query":
        from rag_core.engineering import load_engineering_service

        service = load_engineering_service(
            args.index_dir,
            mini_nanobot_repo=args.mini_repo,
            runtime_backend=args.backend,
        )
        outcome = (
            service.answer(args.question, top_k=args.top_k)
            if args.answer
            else service.retrieve(args.question, top_k=args.top_k)
        )
        _json(outcome.to_dict())
        return 0
    if args.command == "engineering-milvus-cleanup":
        from rag_core.engineering import cleanup_engineering_milvus

        _json(
            cleanup_engineering_milvus(
                args.index_dir,
                execute=args.execute,
                namespace=args.namespace,
            )
        )
        return 0
    if args.command == "engineering-eval":
        from rag_core.evaluation.engineering import main as evaluation_main

        forwarded = [
                "--dataset",
                *args.dataset,
                "--output",
                args.output,
                "--snapshot",
                args.snapshot,
                "--top-k",
                str(args.top_k),
                "--suite",
                args.suite,
            ]
        if args.factory:
            forwarded.extend(["--factory", args.factory])
        if args.baseline:
            forwarded.extend(["--baseline", args.baseline])
        return evaluation_main(forwarded)
    if args.command == "serve":
        import uvicorn

        if args.host not in {"127.0.0.1", "::1", "localhost"}:
            raise SystemExit(
                "the built-in RAG server is loopback-only; bind to 127.0.0.1 "
                "and place an authenticated TLS reverse proxy in front of it "
                "for remote access"
            )
        uvicorn.run("engineering_api:app", host=args.host, port=args.port, reload=False)
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
