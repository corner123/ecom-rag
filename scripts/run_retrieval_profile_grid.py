"""Run an explicit development-only hybrid retrieval weight grid."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

from rag_core.evaluation.engineering import load_jsonl_files, run_ablation, write_report
from rag_core.evaluation.engineering_adapter import create_retrieval_profile_predictors
from rag_core.evaluation.retrieval_profiles import build_retrieval_profile_grid


def _floats(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one comma-separated number is required")
    return parsed


def _ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("at least one comma-separated integer is required")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=[
            "data/eval/mini_nanobot_internal.jsonl",
            "data/eval/official_engineering_specs.jsonl",
        ],
    )
    parser.add_argument("--index-dir", default="data/indexes/engineering")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dense-weights", type=_floats, default=(1.0, 0.5, 0.25))
    parser.add_argument("--bm25-weights", type=_floats, default=(1.0, 1.5, 2.0))
    parser.add_argument("--candidate-multipliers", type=_ints, default=(3,))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    dataset_paths = [Path(value).resolve() for value in args.dataset]
    samples = [sample for sample in load_jsonl_files(dataset_paths) if sample.answerable]
    profiles = build_retrieval_profile_grid(
        dense_weights=args.dense_weights,
        bm25_weights=args.bm25_weights,
        candidate_multipliers=args.candidate_multipliers,
    )
    predictors = create_retrieval_profile_predictors(
        index_root=args.index_dir,
        profiles=profiles,
        candidate_budget_multiplier=1,
    )
    baseline = next(
        (
            profile.name
            for profile in profiles
            if profile.dense_weight == 1.0
            and profile.bm25_weight == 1.0
            and profile.partition_candidate_multiplier == 3
        ),
        profiles[0].name,
    )
    report = run_ablation(
        samples,
        predictors,
        top_k=args.top_k,
        dataset_name=" + ".join(path.name for path in dataset_paths),
        baseline=baseline,
        suite="index",
        run_metadata={
            "experiment_type": "development_retrieval_profile_grid",
            "formal_holdout": False,
            "label_binding": (
                "historical_v2_source_labels_replayed_on_current_build; "
                "development selection only"
            ),
            "dataset_files": [
                {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in dataset_paths
            ],
        },
        validate_contract=False,
        warmup=True,
    )
    json_path, markdown_path = write_report(report, args.output)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
