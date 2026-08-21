"""Bind engineering evaluation labels to one immutable source-manifest build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from rag_core.ingestion import BuildManifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = (
    ROOT / "data/eval/mini_nanobot_internal.jsonl",
    ROOT / "data/eval/official_engineering_specs.jsonl",
    ROOT / "data/eval/engineering_holdout_v2.jsonl",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_dataset(path: Path, revision: str) -> dict:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        scope = str(row.get("source_scope") or "")
        if scope != "official_specifications":
            row["source_revision"] = revision
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return {
        "name": path.name,
        "sha256": _sha256(path),
        "question_count": len(rows),
        "dataset_roles": sorted({str(row["dataset_role"]) for row in rows}),
        "label_versions": sorted({str(row["label_version"]) for row in rows}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=str(ROOT / "data/manifests/builds/current.json")
    )
    parser.add_argument("--dataset", nargs="+", default=[str(p) for p in DEFAULT_DATASETS])
    parser.add_argument(
        "--output", default=str(ROOT / "data/eval/evaluation_snapshot.json")
    )
    parser.add_argument(
        "--source-id",
        default=os.getenv("ENGINEERING_TARGET_SOURCE_ID", "mini_nanobot"),
        help="manifest Git source bound to live-code evaluation",
    )
    args = parser.parse_args()

    manifest = BuildManifest.read(args.manifest)
    target = next(
        (record for record in manifest.sources if record.source_id == args.source_id),
        None,
    )
    if target is None or not target.commit_sha or not target.content_hash:
        raise RuntimeError(
            f"manifest lacks a reproducible Git source snapshot: {args.source_id}"
        )
    revision = (
        f"manifest={manifest.build_id};commit={target.commit_sha};"
        f"worktree_sha256={target.content_hash};dirty={str(target.dirty).lower()}"
    )
    datasets = [
        _freeze_dataset(Path(value).resolve(), revision) for value in args.dataset
    ]
    payload = {
        "schema_version": "engineering-eval-snapshot/v1",
        "manifest_build_id": manifest.build_id,
        "target_repository": {
            "source_id": target.source_id,
            "commit_sha": target.commit_sha,
            "dirty": target.dirty,
            "content_hash": target.content_hash,
        },
        "datasets": datasets,
    }
    if target.source_id == "mini_nanobot":
        payload["mini_nanobot"] = dict(payload["target_repository"])
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(f"frozen evaluation snapshot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
