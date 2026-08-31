"""Create a deterministic manifest from a source catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trade_agent.data.pipeline import IngestionPipeline, SourceCatalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest deterministic trade sources.")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = IngestionPipeline().run(SourceCatalog.from_yaml(args.catalog), args.output)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": type(exc).__name__}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"build_id": result.build_id, "chunks": len(result.chunks), "documents": len(result.documents), "metadata_complete": result.metadata_complete, "quarantined": len(result.quarantined), "status": "ok"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
