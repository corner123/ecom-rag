"""Generate the public development or private holdout trade evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trade_agent.evaluation.generator import generate_development, generate_private_holdout, write_bundle


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "demo/trade_intel_seed/manifests/corpus_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("partition", choices=("development", "holdout"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.partition == "development":
        bundle = generate_development(manifest)
        write_bundle(bundle, args.output, case_filename="dev_public.jsonl", reference_filename="references_dev.jsonl")
    else:
        if args.seed_file is None:
            parser.error("--seed-file is required for holdout generation")
        seed = int(args.seed_file.read_text(encoding="utf-8").strip(), 16)
        generate_private_holdout(manifest, seed, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
