"""Deterministic orchestration around the reviewed router and chunker."""

from __future__ import annotations

import json
import os
import tempfile
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import yaml
from pydantic import AnyUrl, TypeAdapter

from trade_agent.data.chunkers import ChunkRouter
from trade_agent.data.manifest import BuildManifest, ChunkSnapshot, CountEntry, DocumentSnapshot, METADATA_SCHEMA_VERSION, ParserBackends, SourceBuildRecord, canonical_hash, canonical_json
from trade_agent.data.quarantine import QuarantineRecord, sanitize_diagnostic
from trade_agent.data.router import DocumentRouter, SourceInput
from trade_agent.schemas.source import FileType, SourceType, content_sha256, stable_id


DOCUMENT_ROUTER_VERSION = "task6-document-router-v2"
CHUNK_ROUTER_VERSION = "task6-chunk-router-v2"
MAX_FROZEN_MANIFEST_BYTES = 20_000_000
_URL = TypeAdapter(AnyUrl)
_EXTENSIONS = {".html": FileType.HTML, ".htm": FileType.HTML, ".md": FileType.MARKDOWN, ".markdown": FileType.MARKDOWN, ".jsonl": FileType.JSONL, ".pdf": FileType.PDF}


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(item in {"", ".", ".."} for item in path.parts):
        raise ValueError("unsafe catalog path")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class CatalogRule:
    source_type: SourceType
    file_types: tuple[str, ...]
    paths: tuple[str, ...]
    url_pattern: str

    def normalized(self) -> dict[str, Any]:
        return {"source_type": self.source_type.value, "file_types": sorted(self.file_types), "paths": sorted(self.paths), "url_pattern": self.url_pattern}


@dataclass(frozen=True, slots=True)
class SourceCatalog:
    catalog_version: int
    root_declared: str
    corpus_root: Path
    synthetic_notice: str
    sources: tuple[CatalogRule, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> "SourceCatalog":
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError("catalog cannot be read as YAML") from exc
        if not isinstance(payload, dict) or set(payload) != {"catalog_version", "root", "synthetic_notice", "sources"}:
            raise ValueError("catalog must contain only catalog_version, root, synthetic_notice, and sources")
        if type(payload["catalog_version"]) is not int or payload["catalog_version"] != 1:
            raise ValueError("catalog_version must be exactly 1")
        if not isinstance(payload["root"], str) or not payload["root"].strip() or not isinstance(payload["synthetic_notice"], str) or not payload["synthetic_notice"].strip():
            raise ValueError("catalog root and synthetic_notice must be nonblank")
        root_path = Path(payload["root"])
        root = root_path if root_path.is_absolute() else Path.cwd() / root_path
        if root.is_symlink() or not root.is_dir():
            raise ValueError("catalog root must be an existing non-symlink directory")
        if not isinstance(payload["sources"], list) or not payload["sources"]:
            raise ValueError("catalog sources must be a nonempty list")
        rules: list[CatalogRule] = []
        for raw in payload["sources"]:
            if not isinstance(raw, dict) or set(raw) != {"source_type", "file_types", "paths", "url_pattern"}:
                raise ValueError("each catalog source must have source_type, file_types, paths, and url_pattern")
            try:
                source_type = SourceType(raw["source_type"])
            except (TypeError, ValueError) as exc:
                raise ValueError("catalog source_type is invalid") from exc
            if not isinstance(raw["file_types"], list) or not raw["file_types"] or not all(isinstance(item, str) and item for item in raw["file_types"]):
                raise ValueError("catalog file_types must be nonempty strings")
            if not isinstance(raw["paths"], list) or not raw["paths"] or not all(isinstance(item, str) and item for item in raw["paths"]):
                raise ValueError("catalog paths must be nonempty strings")
            if not isinstance(raw["url_pattern"], str) or not raw["url_pattern"].strip():
                raise ValueError("catalog url_pattern must be nonblank")
            try:
                _URL.validate_python(raw["url_pattern"].replace("*", "source"))
            except Exception as exc:
                raise ValueError("catalog url_pattern is invalid") from exc
            rules.append(CatalogRule(source_type, tuple(raw["file_types"]), tuple(raw["paths"]), raw["url_pattern"]))
        return cls(1, payload["root"], root.resolve(strict=True), payload["synthetic_notice"], tuple(rules))

    def normalized(self) -> dict[str, Any]:
        return {"catalog_version": self.catalog_version, "root": "corpus-root", "synthetic_notice": self.synthetic_notice, "sources": sorted((rule.normalized() for rule in self.sources), key=canonical_json)}


@dataclass(frozen=True, slots=True)
class _Candidate:
    rule: CatalogRule
    relative_path: str
    issue: str | None = None


class IngestionPipeline:
    def __init__(self, *, document_router: DocumentRouter | None = None, chunk_router: ChunkRouter | None = None):
        self.document_router = document_router or DocumentRouter()
        self.chunk_router = chunk_router or ChunkRouter()

    def _expand(self, catalog: SourceCatalog) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for rule in catalog.sources:
            for pattern in rule.paths:
                try:
                    safe = _relative(pattern)
                except ValueError:
                    candidates.append(_Candidate(rule, "invalid-" + stable_id("path", pattern), "unsafe_catalog_path"))
                    continue
                try:
                    matches = sorted(catalog.corpus_root.glob(safe), key=lambda item: item.as_posix()) if any(token in safe for token in "*?[") else [catalog.corpus_root / safe]
                except ValueError:
                    candidates.append(_Candidate(rule, safe, "bad_catalog_glob"))
                    continue
                if not matches or (any(token in safe for token in "*?[") and not matches):
                    candidates.append(_Candidate(rule, safe, "missing_catalog_path"))
                else:
                    for match in matches:
                        candidates.append(_Candidate(rule, match.relative_to(catalog.corpus_root).as_posix(), None if match.exists() else "missing_catalog_path"))
        grouped: dict[str, list[_Candidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.relative_path, []).append(candidate)
        collapsed: list[_Candidate] = []
        for path, matches in grouped.items():
            selected = sorted(matches, key=lambda item: (item.rule.source_type.value, item.rule.file_types, item.rule.url_pattern))[0]
            issue = selected.issue or ("duplicate_catalog_path" if len(matches) > 1 else None)
            collapsed.append(_Candidate(selected.rule, path, issue))
        return sorted(collapsed, key=lambda item: (item.relative_path, item.rule.source_type.value, item.rule.file_types))

    @staticmethod
    def _file_type(candidate: _Candidate) -> FileType | None:
        suffix = Path(candidate.relative_path).suffix.lower()
        inferred = FileType.GENERATED_PROFILE if suffix == ".json" and candidate.rule.source_type is SourceType.CUSTOMS_PROFILE else FileType.JSON if suffix == ".json" else _EXTENSIONS.get(suffix)
        return inferred if inferred and inferred.value in candidate.rule.file_types else None

    @staticmethod
    def _safe_file(root: Path, relative_path: str) -> Path | None:
        try:
            safe = _relative(relative_path)
            path = root / safe
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            return None
        if not resolved.is_relative_to(root) or not path.is_file():
            return None
        cursor = root
        for part in PurePosixPath(safe).parts:
            cursor /= part
            if cursor.is_symlink():
                return None
        return path

    @staticmethod
    def _manifest_path(root: Path) -> Path:
        try:
            root_resolved = root.resolve(strict=True)
            manifest_dir = root / "manifests"
            manifest_path = manifest_dir / "corpus_manifest.json"
            if root.is_symlink() or not root.is_dir():
                raise ValueError
            if manifest_dir.is_symlink() or not manifest_dir.is_dir():
                raise ValueError
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise ValueError
            if not manifest_path.resolve(strict=True).is_relative_to(root_resolved):
                raise ValueError
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("frozen corpus manifest is unreadable") from exc
        return manifest_path

    def _frozen_records(self, root: Path, *, maximum: int | None = None) -> dict[str, dict[str, Any]]:
        limit = MAX_FROZEN_MANIFEST_BYTES if maximum is None else maximum
        if type(limit) is not int or limit <= 0:
            raise ValueError("frozen corpus manifest is unreadable")
        try:
            manifest_path = self._manifest_path(root)
            if manifest_path.stat().st_size > limit:
                raise ValueError("frozen corpus manifest exceeds size limit")
            chunks: list[bytes] = []
            total = 0
            with manifest_path.open("rb") as handle:
                while block := handle.read(min(1024 * 1024, limit + 1 - total)):
                    total += len(block)
                    if total > limit:
                        raise ValueError("frozen corpus manifest exceeds size limit")
                    chunks.append(block)
            data = json.loads(b"".join(chunks).decode("utf-8"))["records"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("frozen corpus manifest is unreadable") from exc
        records: dict[str, dict[str, Any]] = {}
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError("frozen corpus manifest record is invalid")
            path = _relative(item["path"])
            if path in records:
                raise ValueError("frozen corpus manifest has duplicate paths")
            records[path] = item
        return records

    @staticmethod
    def _q(code: str, path: str, diagnostic: object, parser: str = "pipeline") -> QuarantineRecord:
        return QuarantineRecord(error_code=code, source_path=path, parser=parser, diagnostic=sanitize_diagnostic(diagnostic, source_path=path))

    @staticmethod
    def _hash_limited(path: Path, maximum: int) -> str:
        if path.stat().st_size > maximum:
            raise ValueError("source exceeds router byte limit")
        digest = hashlib.sha256()
        total = 0
        with path.open("rb") as handle:
            while block := handle.read(min(1024 * 1024, maximum + 1 - total)):
                total += len(block)
                if total > maximum:
                    raise ValueError("source exceeds router byte limit")
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _write_atomic(output: Path, value: BuildManifest) -> None:
        output = output.absolute()
        if output.is_symlink() or (output.exists() and (output.is_dir() or not output.is_file())):
            raise ValueError("output target must be a non-symlink regular file or absent")
        parent = output.parent
        # macOS exposes the system temporary directory as /tmp -> /private/tmp.
        # It is a platform-owned alias, not a caller-controlled output parent.
        aliases = {Path("/var"): Path("/private/var"), Path("/tmp"): Path("/private/tmp"), Path("/etc"): Path("/private/etc")}
        for ancestor in (parent, *parent.parents):
            if ancestor == Path("/"):
                break
            if ancestor.is_symlink() and aliases.get(ancestor) != ancestor.resolve(strict=True):
                raise ValueError("output parent must not have a symlink ancestor")
        parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(canonical_json(value.model_dump(mode="json")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if output.exists() and output.is_symlink():
                raise ValueError("output target must not become a symlink")
            os.replace(temporary, output)
            directory_fd = os.open(parent, os.O_RDONLY)
            try: os.fsync(directory_fd)
            finally: os.close(directory_fd)
        finally:
            if temporary.exists(): temporary.unlink()

    def run(self, catalog: SourceCatalog, output: Path) -> BuildManifest:
        records = self._frozen_records(catalog.corpus_root)
        sources: list[SourceBuildRecord] = []
        documents = []
        chunks = []
        quarantined: list[QuarantineRecord] = []
        for candidate in self._expand(catalog):
            source_id = stable_id("source", candidate.relative_path)
            file_type = self._file_type(candidate)
            record = records.get(candidate.relative_path)
            declared_hash = record.get("content_hash") if record else None
            if not isinstance(declared_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", declared_hash):
                declared_hash = None
            path = self._safe_file(catalog.corpus_root, candidate.relative_path) if candidate.issue is None else None
            issue = candidate.issue or ("unsupported_file_type" if file_type is None else None) or ("unsafe_source_path" if path is None else None) or ("manifest_record_missing" if record is None else None)
            actual_hash = None
            if issue is None and (
                not isinstance(record.get("source_type"), str)
                or not isinstance(record.get("file_type"), str)
                or record["source_type"] != candidate.rule.source_type.value
                or record["file_type"] != file_type.value
            ):
                issue = "manifest_record_mismatch"
            if issue is None and path is not None:
                try:
                    actual_hash = self._hash_limited(path, self.document_router.max_bytes)
                except ValueError as exc:
                    issue = "input_too_large"
                    quarantined.append(self._q(issue, candidate.relative_path, exc))
                except OSError as exc:
                    issue = "source_unreadable"
                    quarantined.append(self._q(issue, candidate.relative_path, exc))
                if issue is None and actual_hash != declared_hash:
                    issue = "manifest_hash_mismatch"
            if issue:
                if not quarantined or quarantined[-1].source_path != candidate.relative_path: quarantined.append(self._q(issue, candidate.relative_path, issue))
                sources.append(SourceBuildRecord(source_id=source_id, path=candidate.relative_path, source_type=candidate.rule.source_type.value, file_type=file_type.value if file_type else "unsupported", actual_content_hash=actual_hash, manifest_content_hash=declared_hash, status="quarantined"))
                continue
            assert path is not None and record is not None and file_type is not None and actual_hash is not None
            try:
                source = SourceInput(
                    path=path,
                    display_path=candidate.relative_path,
                    file_type=file_type,
                    source_type=candidate.rule.source_type,
                    source_id=source_id,
                    title=candidate.relative_path,
                    language="en",
                    fetched_at=datetime.fromisoformat(record["ingested_at"]),
                    is_synthetic=record["is_synthetic"],
                    source_url=candidate.rule.url_pattern.replace(
                        "*", quote(Path(candidate.relative_path).stem, safe="")
                    ),
                    publish_time=datetime.fromisoformat(record["publish_time"]),
                    valid_from=datetime.fromisoformat(record["valid_from"]),
                    valid_to=datetime.fromisoformat(record["valid_to"])
                    if record.get("valid_to")
                    else None,
                    manifest_attributes={
                        key: value
                        for key, value in record.items()
                        if key
                        not in {
                            "path",
                            "content_hash",
                            "file_type",
                            "source_type",
                            "is_synthetic",
                            "ingested_at",
                            "publish_time",
                            "valid_from",
                            "valid_to",
                            "expected_entity",
                        }
                    },
                )
            except (KeyError, TypeError, ValueError) as exc:
                quarantined.append(self._q("manifest_record_invalid", candidate.relative_path, exc))
                sources.append(SourceBuildRecord(source_id=source_id, path=candidate.relative_path, source_type=candidate.rule.source_type.value, file_type=file_type.value, actual_content_hash=actual_hash, manifest_content_hash=record.get("content_hash"), status="quarantined"))
                continue
            before = len(self.document_router.quarantines)
            routed = self.document_router.load(source)
            own_q = [
                QuarantineRecord(
                    error_code=item.error_code.lower(),
                    source_path=candidate.relative_path,
                    parser=item.parser,
                    diagnostic=item.diagnostic,
                )
                for item in self.document_router.quarantines[before:]
            ]
            quarantined.extend(own_q)
            documents.extend(routed)
            chunk_failed = False
            for document in routed:
                try: chunks.extend(self.chunk_router.chunk(document))
                except Exception as exc:
                    chunk_failed = True
                    quarantined.append(self._q("chunk_failed", candidate.relative_path, exc, "chunker"))
            parser = next(
                (str(item.attributes["parser"]) for item in routed if item.attributes.get("parser")),
                own_q[-1].parser if own_q else "document_router",
            )
            sources.append(
                SourceBuildRecord(
                    source_id=source_id,
                    path=candidate.relative_path,
                    source_type=candidate.rule.source_type.value,
                    file_type=file_type.value,
                    actual_content_hash=actual_hash,
                    manifest_content_hash=record.get("content_hash"),
                    status="quarantined" if own_q or chunk_failed else "parsed",
                    parser_backend=parser,
                    degraded=bool(own_q)
                    or chunk_failed
                    or any(item.attributes.get("degraded_components") for item in routed),
                )
            )
        sources.sort(key=lambda item: item.path)
        documents.sort(key=lambda item: item.document_id)
        chunks.sort(key=lambda item: item.metadata.chunk_id)
        quarantined.sort(key=lambda item: (item.source_path, item.error_code, item.parser, item.diagnostic))
        mineru_statuses = {
            str(item.attributes["mineru_status"])
            for item in documents
            if item.attributes.get("mineru_status")
        }
        degraded_components = {
            str(component)
            for item in documents
            for component in item.attributes.get("degraded_components", [])
        }
        for item in quarantined:
            match = re.search(r"mineru_status=([a-z_]+)", item.diagnostic)
            if match:
                status = match.group(1)
                mineru_statuses.add(status)
                degraded_components.add(f"mineru_{status}")
        backends = ParserBackends(
            document_router_version=DOCUMENT_ROUTER_VERSION,
            chunk_router_version=CHUNK_ROUTER_VERSION,
            max_tokens=self.chunk_router.by[SourceType.B2B].max_tokens,
            overlap_tokens=self.chunk_router.by[SourceType.B2B].overlap_tokens,
            mineru_statuses=tuple(sorted(mineru_statuses or {"not_attempted"})),
            degraded_components=tuple(sorted(degraded_components)),
        )
        config_hash = canonical_hash({"catalog": catalog.normalized(), "router": {"version": DOCUMENT_ROUTER_VERSION, "max_bytes": self.document_router.max_bytes, "max_documents": self.document_router.max_documents}, "chunker": backends.model_dump(mode="json"), "metadata_schema_version": METADATA_SCHEMA_VERSION})
        document_snapshots = tuple(DocumentSnapshot.freeze(item) for item in documents)
        chunk_snapshots = tuple(ChunkSnapshot.freeze(item) for item in chunks)
        fingerprint_documents = [json.loads(item.payload) for item in document_snapshots]
        fingerprint_chunks = [json.loads(item.payload) for item in chunk_snapshots]
        for item in fingerprint_documents:
            item.pop("fetched_at", None)
        for item in fingerprint_chunks:
            item.get("metadata", {}).pop("ingested_at", None)
        fingerprint = {"config_hash": config_hash, "sources": [item.model_dump(mode="json") for item in sources], "documents": fingerprint_documents, "chunks": fingerprint_chunks, "quarantined": [item.model_dump(mode="json") for item in quarantined]}
        fingerprint["parser_backends"] = backends.model_dump(mode="json")
        fingerprint["metadata_schema_version"] = METADATA_SCHEMA_VERSION
        fingerprint_hash = canonical_hash(fingerprint)
        manifest = BuildManifest(build_id="build_" + fingerprint_hash[:32], config_hash=config_hash, sources=tuple(sources), documents=document_snapshots, chunks=chunk_snapshots, quarantined=tuple(quarantined), counts_by_source_type=tuple(CountEntry(name=name, count=count) for name, count in sorted(Counter(item.source_type or "unknown" for item in sources).items())), counts_by_file_type=tuple(CountEntry(name=name, count=count) for name, count in sorted(Counter(item.file_type for item in sources).items())), parser_backends=backends, metadata_complete=bool(chunks) and all(item.metadata.source_locator.raw is not None for item in chunks), fingerprint=fingerprint_hash)
        self._write_atomic(output, manifest)
        return manifest
