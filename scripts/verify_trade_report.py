"""Verify an immutable trade evaluation report bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trade_agent.evaluation.report import verify_report_bundle


def _latest(root: Path, kind: str) -> Path:
    candidates: list[tuple[str, str, Path]] = []
    for report_path in root.glob("*/report.json"):
        try:
            value = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if value.get("dataset_role") == kind:
            candidates.append((str(value.get("created_at", "")), report_path.parent.name, report_path.parent))
    if not candidates:
        raise ValueError(f"no {kind} report bundles found beneath {root}")
    return max(candidates)[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("path", type=Path, nargs="?")
    selection.add_argument("--latest", type=Path, metavar="ROOT")
    parser.add_argument("--kind", choices=("development", "holdout"))
    args = parser.parse_args(argv)
    if args.latest is not None and args.kind is None:
        parser.error("--latest requires --kind")
    if args.path is not None and args.kind is not None:
        parser.error("--kind is only valid with --latest")
    try:
        path = args.path if args.path is not None else _latest(args.latest, args.kind)
    except ValueError as exc:
        parser.error(str(exc))
    result = verify_report_bundle(path)
    print(json.dumps({"errors": list(result.errors), "path": str(result.path), "valid": result.valid}, sort_keys=True))
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
