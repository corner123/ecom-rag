"""Validate the controlled public/private trade evaluation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trade_agent.evaluation.generator import indexed_corpus_content, read_bundle
from trade_agent.evaluation.leakage import LeakageAuditor


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--require-zero-leakage", action="store_true",
                        help="Explicitly require the zero-leakage gate (already the default)")
    args = parser.parse_args(argv)
    development = read_bundle(args.dev / "dev_public.jsonl", args.dev / "references_dev.jsonl")
    holdout = read_bundle(args.holdout / "holdout_private.jsonl", args.holdout / "references_private.jsonl")
    corpus = json.loads((ROOT / "demo/trade_intel_seed/manifests/corpus_manifest.json").read_text(encoding="utf-8"))
    report = LeakageAuditor().audit(development, holdout, indexed_corpus_content(corpus))
    print(json.dumps({"passed": report.passed, **report.__dict__}, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
