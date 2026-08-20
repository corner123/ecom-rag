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

    response_eval = subparsers.add_parser(
        "engineering-response-eval",
        help="run or replay a frozen response-eval/v3 generation and RAGAS experiment",
    )
    response_eval.add_argument("--dataset", required=True)
    response_eval.add_argument("--snapshot", required=True)
    response_eval.add_argument("--manifest", default=DEFAULT_MANIFEST)
    response_eval.add_argument("--index-dir", default=DEFAULT_INDEX)
    response_eval.add_argument("--mini-repo")
    response_eval.add_argument("--output", required=True)
    response_eval.add_argument("--profile-name", required=True)
    response_eval.add_argument("--dense-weight", type=float, default=1.0)
    response_eval.add_argument("--bm25-weight", type=float, default=1.0)
    response_eval.add_argument("--partition-candidate-multiplier", type=int, default=3)
    response_eval.add_argument("--federated-candidate-multiplier", type=int, default=3)
    response_eval.add_argument("--top-k", type=int, default=5)
    response_eval.add_argument(
        "--sufficiency-profile",
        choices=(
            "legacy_exact_slash",
            "split_natural_slash_concepts",
        ),
        default="legacy_exact_slash",
        help=(
            "versioned evidence-sufficiency policy; defaults to the production "
            "legacy baseline"
        ),
    )
    response_eval.add_argument(
        "--support-selection-profile",
        choices=("legacy_first", "query_aware_diverse"),
        default="legacy_first",
        help=(
            "versioned single-slot supporting-evidence selector; defaults to "
            "the production legacy baseline"
        ),
    )
    stages = response_eval.add_mutually_exclusive_group()
    stages.add_argument("--generate-only", action="store_true")
    stages.add_argument("--judge-only", action="store_true")
    response_eval.add_argument(
        "--replay", help="generation artifact required by --judge-only"
    )
    response_eval.add_argument(
        "--resume",
        action="store_true",
        help="resume only when the existing artifact has the same frozen identity",
    )
    response_eval.add_argument(
        "--retry-from",
        help=(
            "safer Judge recovery: read a partial judged artifact and write the "
            "remaining null metrics to a new --output"
        ),
    )

    response_snapshot = subparsers.add_parser(
        "engineering-response-snapshot",
        help="freeze one reviewed response-eval/v3 dataset and current FAISS build",
    )
    response_snapshot.add_argument("--dataset", required=True)
    response_snapshot.add_argument("--manifest", default=DEFAULT_MANIFEST)
    response_snapshot.add_argument("--index-dir", default=DEFAULT_INDEX)
    response_snapshot.add_argument("--output", required=True)

    response_report = subparsers.add_parser(
        "engineering-response-report",
        help="compare two judged response artifacts without retrieval or model calls",
    )
    response_report.add_argument("--baseline-artifact", required=True)
    response_report.add_argument("--candidate-artifact", required=True)
    response_report.add_argument("--baseline-name", required=True)
    response_report.add_argument("--candidate-name", required=True)
    response_report.add_argument("--output-prefix", required=True)
    response_report.add_argument(
        "--metadata-json",
        help="existing JSON-object file or inline JSON object for report annotations",
    )
    response_report.add_argument(
        "--minimum-judge-coverage", type=float, default=0.95
    )

    response_rescore = subparsers.add_parser(
        "engineering-response-rescore",
        help="recompute deterministic retrieval metrics without model or RAGAS calls",
    )
    response_rescore.add_argument("--artifact", required=True)
    response_rescore.add_argument("--dataset", required=True)
    response_rescore.add_argument("--output", required=True)

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
    if args.command == "engineering-response-eval":
        from rag_core.evaluation.response_experiment import run_response_experiment
        from rag_core.evaluation.retrieval_profiles import RetrievalExperimentProfile

        mini_repo = args.mini_repo or os.getenv("MINI_NANOBOT_REPO")
        if not args.judge_only and not mini_repo:
            raise SystemExit(
                "--mini-repo or MINI_NANOBOT_REPO is required for generation"
            )
        if args.judge_only and not (args.replay or args.retry_from):
            raise SystemExit("--judge-only requires --replay or --retry-from")
        if args.retry_from and not args.judge_only:
            raise SystemExit("--retry-from is only valid with --judge-only")
        if args.retry_from and args.resume:
            raise SystemExit("--retry-from and legacy --resume are mutually exclusive")
        profile = RetrievalExperimentProfile(
            name=args.profile_name,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
            partition_candidate_multiplier=args.partition_candidate_multiplier,
            federated_candidate_multiplier=args.federated_candidate_multiplier,
        )
        artifact = run_response_experiment(
            dataset_path=args.dataset,
            snapshot_path=args.snapshot,
            manifest_path=args.manifest,
            index_root=args.index_dir,
            mini_nanobot_repo=mini_repo or ".",
            output_path=args.output,
            profile=profile,
            sufficiency_profile=args.sufficiency_profile,
            support_selection_profile=args.support_selection_profile,
            top_k=args.top_k,
            generate_only=args.generate_only,
            judge_only=args.judge_only,
            replay_path=args.replay,
            resume=args.resume,
            retry_from_path=args.retry_from,
        )
        _json(
            {
                "schema_version": artifact["schema_version"],
                "stage": artifact["stage"],
                "experiment_id": artifact["metadata"]["experiment_id"],
                "records": len(artifact["records"]),
                "output": str(Path(args.output).resolve()),
            }
        )
        return 0
    if args.command == "engineering-response-snapshot":
        from rag_core.evaluation.response_experiment import write_response_snapshot

        snapshot = write_response_snapshot(
            dataset_path=args.dataset,
            manifest_path=args.manifest,
            index_root=args.index_dir,
            output_path=args.output,
        )
        _json(
            {
                "schema_version": snapshot["schema_version"],
                "manifest_build_id": snapshot["manifest_build_id"],
                "dataset": snapshot["dataset"],
                "output": str(Path(args.output).resolve()),
            }
        )
        return 0
    if args.command == "engineering-response-report":
        from rag_core.evaluation.response_report import (
            build_response_comparison_from_artifacts,
            load_report_metadata,
            write_response_report,
        )

        output_prefix = Path(args.output_prefix)
        if output_prefix.suffix:
            output_prefix = output_prefix.with_suffix("")
        targets = [
            output_prefix.with_suffix(".json"),
            output_prefix.with_suffix(".md"),
            output_prefix.with_suffix(".png"),
        ]
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise SystemExit(
                "response report refuses to overwrite existing outputs: "
                + ", ".join(existing)
            )
        report = build_response_comparison_from_artifacts(
            baseline_path=args.baseline_artifact,
            candidate_path=args.candidate_artifact,
            baseline_name=args.baseline_name,
            candidate_name=args.candidate_name,
            user_metadata=load_report_metadata(args.metadata_json),
            minimum_judge_coverage=args.minimum_judge_coverage,
        )
        json_path, markdown_path, plot_path = write_response_report(
            report, output_prefix
        )
        _json(
            {
                "schema_version": report["schema_version"],
                "publishable": report["publishable"],
                "json": str(json_path.resolve()),
                "markdown": str(markdown_path.resolve()),
                "plot": str(plot_path.resolve()) if plot_path else None,
            }
        )
        return 0
    if args.command == "engineering-response-rescore":
        from rag_core.evaluation.response_experiment import (
            recompute_deterministic_retrieval_metrics,
        )

        artifact = recompute_deterministic_retrieval_metrics(
            artifact_path=args.artifact,
            dataset_path=args.dataset,
            output_path=args.output,
        )
        _json(
            {
                "schema_version": artifact["schema_version"],
                "stage": artifact["stage"],
                "metric_schema": artifact["metadata"][
                    "deterministic_retrieval_rescore"
                ]["schema_version"],
                "records": len(artifact["records"]),
                "network_or_model_calls": False,
                "output": str(Path(args.output).resolve()),
            }
        )
        return 0
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
