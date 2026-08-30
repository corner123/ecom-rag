"""CLI for generating the checked-in synthetic trade-intelligence demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trade_agent.data.demo_generator import generate_demo_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate only synthetic, demo trade intelligence fixtures.")
    parser.add_argument("--output", type=Path, default=Path("demo/trade_intel_seed"))
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    summary = generate_demo_corpus(args.output, seed=args.seed, clean=args.clean)
    print(json.dumps({"manifest": summary.manifest_path.as_posix(), "source_types": sorted(summary.source_types), "file_types": sorted(summary.file_types), "synthetic_only": True}, sort_keys=True))


if __name__ == "__main__":
    main()
