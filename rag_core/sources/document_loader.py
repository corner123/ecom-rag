"""File-type aware document loading for the formal source-ingestion path.

Every parser returns :class:`~rag_core.sources.schema.DocumentRecord` objects.
The router deliberately keeps parsing independent from LangChain so source
snapshots remain deterministic and JSON serializable before embeddings exist.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.metadata
import io
import json
import mimetypes
import os
import platform
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import tomllib
from typing import Any, Iterable, Mapping, Protocol

from .schema import DocumentRecord


class DocumentLoaderError(RuntimeError):
    """Base class for an actionable document-ingestion failure."""


class UnsupportedDocumentType(DocumentLoaderError):
    """The router has no safe parser for the requested file type."""


class BackendUnavailable(DocumentLoaderError):
    """An optional parser executable or Python package is unavailable."""


class RecoverableParseError(DocumentLoaderError):
    """A parser understood the type but could not extract this document."""


class EmptyDocumentError(DocumentLoaderError):
    """A document contained no indexable text or structured table data."""


@dataclass(slots=True)
class ParsedDocumentUnit:
    """One logical unit emitted by a parser before provenance is attached."""

    content: str
    title: str = ""
    media_type: str = "text/plain"
    language: str = ""
    content_format: str = "text"
    parser_backend: str = "builtin"
    identity_suffix: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class PDFParserBackend(Protocol):
    name: str

    def parse(self, path: Path, output_dir: Path) -> list[ParsedDocumentUnit]:
        """Extract logical units from *path* into an isolated output directory."""


_FILE_TYPE_BY_SUFFIX = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".rst": "text",
    ".py": "code",
    ".js": "code",
    ".jsx": "code",
    ".ts": "code",
    ".tsx": "code",
    ".java": "code",
    ".go": "code",
    ".rs": "code",
    ".c": "code",
    ".h": "code",
    ".cpp": "code",
    ".hpp": "code",
    ".sh": "code",
    ".ps1": "code",
    ".sql": "code",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".csv": "csv",
    ".xlsx": "xlsx",
    ".pdf": "pdf",
}

_FILE_TYPE_BY_MIME = {
    "text/markdown": "markdown",
    "text/plain": "text",
    "text/x-python": "code",
    "text/javascript": "code",
    "application/javascript": "code",
    "application/json": "json",
    "application/yaml": "yaml",
    "text/yaml": "yaml",
    "application/toml": "toml",
    "text/csv": "csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/pdf": "pdf",
}

_FILE_TYPE_ALIASES = {
    "md": "markdown",
    "markdown": "markdown",
    "txt": "text",
    "plain": "text",
    "text": "text",
    "source": "code",
    "source_code": "code",
    "code": "code",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "csv": "csv",
    "excel": "xlsx",
    "spreadsheet": "xlsx",
    "xlsx": "xlsx",
    "pdf": "pdf",
}

_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".sh": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
}

SUPPORTED_SUFFIXES = frozenset(_FILE_TYPE_BY_SUFFIX)

SENSITIVE_FILENAMES = frozenset(
    {
        ".env",
        ".env.local",
        "credentials.json",
        "token.json",
        "secrets.json",
        "id_rsa",
        "id_ed25519",
    }
)
SENSITIVE_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".jks"})
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        ".cache",
        ".pytest_cache",
        ".mypy_cache",
    }
)


def is_safe_source_path(path: str | Path) -> bool:
    """Reject dependency/cache trees and common secret-bearing files."""

    candidate = PurePosixPath(str(path).replace("\\", "/"))
    lowered_parts = {part.lower() for part in candidate.parts}
    name = candidate.name.lower()
    suffix = candidate.suffix.lower()
    if lowered_parts & EXCLUDED_DIRECTORY_NAMES:
        return False
    if name in SENSITIVE_FILENAMES or name.startswith(".env."):
        return False
    if suffix in SENSITIVE_SUFFIXES:
        return False
    return True


class MinerUCliBackend:
    """Optional MinerU CLI adapter.

    MinerU is run without a shell in an isolated temporary directory.  The
    command itself is also the capability probe: an absent executable raises
    :class:`BackendUnavailable`, while a recognized but failed conversion is a
    :class:`RecoverableParseError` that permits the explicit PyMuPDF fallback.
    """

    name = "mineru-cli"

    def __init__(self, executable: str | None = None, *, timeout_seconds: int = 180) -> None:
        self.executable = executable or os.getenv("MINERU_EXECUTABLE", "mineru")
        if timeout_seconds <= 0:
            raise ValueError("MinerU timeout must be positive")
        self.timeout_seconds = int(timeout_seconds)

    def parse(self, path: Path, output_dir: Path) -> list[ParsedDocumentUnit]:
        executable = shutil.which(self.executable)
        if executable is None:
            raise BackendUnavailable(
                f"MinerU executable '{self.executable}' is not available on PATH"
            )
        try:
            completed = subprocess.run(
                [executable, "-p", str(path), "-o", str(output_dir)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=self.timeout_seconds,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RecoverableParseError(
                f"MinerU timed out after {self.timeout_seconds} seconds"
            ) from exc
        except FileNotFoundError as exc:
            raise BackendUnavailable("MinerU executable disappeared before invocation") from exc
        except PermissionError as exc:
            raise BackendUnavailable("MinerU executable is not runnable") from exc
        except OSError as exc:
            raise BackendUnavailable(f"MinerU could not start: {type(exc).__name__}") from exc
        if completed.returncode != 0:
            detail = _sanitize_diagnostic(
                completed.stderr or completed.stdout,
                private_paths=(path, output_dir),
            )
            raise RecoverableParseError(
                f"MinerU exited with status {completed.returncode}: {detail or 'no diagnostic'}"
            )
        return MinerUOutputAdapter().adapt(output_dir)


class MinerUOutputAdapter:
    """Normalize MinerU artifacts without trusting the converter output tree.

    Structured JSON is preferred because it preserves page, block, and table
    coordinates. Markdown is an explicit fallback for MinerU versions that do
    not emit a supported content-list or middle-JSON shape.
    """

    def __init__(
        self,
        *,
        max_artifact_bytes: int = 32 * 1024 * 1024,
        max_total_bytes: int = 128 * 1024 * 1024,
        max_artifacts: int = 512,
    ) -> None:
        if min(max_artifact_bytes, max_total_bytes, max_artifacts) <= 0:
            raise ValueError("MinerU artifact limits must be positive")
        self.max_artifact_bytes = int(max_artifact_bytes)
        self.max_total_bytes = int(max_total_bytes)
        self.max_artifacts = int(max_artifacts)

    def adapt(self, output_dir: Path) -> list[ParsedDocumentUnit]:
        root = output_dir.resolve(strict=True)
        artifacts = self._safe_artifacts(root)
        json_candidates = [item for item in artifacts if item.suffix.lower() == ".json"]
        markdown_candidates = [item for item in artifacts if item.suffix.lower() == ".md"]
        diagnostics: list[str] = []

        # Prefer MinerU's content-list contract, then its middle JSON, and only
        # then inspect other JSON artifacts. This avoids accidentally indexing
        # model/config JSON when a structured document artifact exists.
        json_groups = (
            [item for item in json_candidates if "content_list" in item.name.lower()],
            [item for item in json_candidates if "middle" in item.name.lower()],
            [
                item
                for item in json_candidates
                if "content_list" not in item.name.lower()
                and "middle" not in item.name.lower()
            ],
        )
        for group in json_groups:
            json_units: list[ParsedDocumentUnit] = []
            for candidate in group:
                relative = candidate.relative_to(root).as_posix()
                try:
                    value = json.loads(_read_text(candidate))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, DocumentLoaderError) as exc:
                    diagnostics.append(
                        f"unreadable structured artifact {relative}: {type(exc).__name__}"
                    )
                    continue
                json_units.extend(_mineru_json_units(value, artifact=relative))
            if json_units:
                return _deduplicate_mineru_units(json_units)

        markdown_units: list[ParsedDocumentUnit] = []
        for candidate in markdown_candidates:
            text = _read_text(candidate)
            if not text.strip():
                continue
            relative = candidate.relative_to(root).as_posix()
            markdown_units.append(
                ParsedDocumentUnit(
                    content=text,
                    title=_markdown_title(text) or candidate.stem,
                    media_type="text/markdown",
                    language="markdown",
                    content_format="markdown",
                    parser_backend="mineru-cli",
                    identity_suffix=f"mineru-markdown:{_stable_coordinate(relative)}",
                    metadata={
                        "mineru_artifact": relative,
                        "parse_warnings": [
                            "MinerU structured JSON was unavailable; Markdown fallback used"
                        ],
                        "parse_degraded": True,
                    },
                )
            )
        if markdown_units:
            return markdown_units

        detail = "; ".join(diagnostics[:3])
        message = "MinerU completed without supported non-empty JSON or Markdown"
        if detail:
            message = f"{message}: {detail}"
        raise RecoverableParseError(message)

    def _safe_artifacts(self, root: Path) -> list[Path]:
        candidates: list[Path] = []
        total = 0
        for candidate in sorted(root.rglob("*")):
            if candidate.suffix.lower() not in {".json", ".md"}:
                continue
            if candidate.is_symlink() or any(parent.is_symlink() for parent in candidate.parents if parent != root):
                raise RecoverableParseError("MinerU output contains a symbolic-link artifact")
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
                stat = resolved.stat()
            except (OSError, ValueError) as exc:
                raise RecoverableParseError("MinerU output artifact escaped its sandbox") from exc
            if not resolved.is_file():
                continue
            if stat.st_size > self.max_artifact_bytes:
                raise RecoverableParseError(
                    f"MinerU artifact exceeds the {self.max_artifact_bytes}-byte limit"
                )
            total += stat.st_size
            if total > self.max_total_bytes:
                raise RecoverableParseError(
                    f"MinerU text artifacts exceed the {self.max_total_bytes}-byte total limit"
                )
            candidates.append(resolved)
            if len(candidates) > self.max_artifacts:
                raise RecoverableParseError(
                    f"MinerU emitted more than {self.max_artifacts} text artifacts"
                )
        return candidates


class DocumentLoaderRouter:
    """Route a file to a parser and return canonical ``DocumentRecord`` values.

    Resolution order is intentionally explicit: caller-provided ``file_type``
    wins over caller-provided/guessed MIME, which wins over the file suffix.
    """

    def __init__(
        self,
        *,
        mineru_backend: PDFParserBackend | None = None,
        prefer_mineru: bool = True,
    ) -> None:
        self.mineru_backend = mineru_backend or MinerUCliBackend()
        self.prefer_mineru = prefer_mineru

    def resolve_file_type(
        self,
        path: str | Path,
        *,
        file_type: str | None = None,
        mime_type: str | None = None,
    ) -> str:
        candidate = Path(path)
        if file_type:
            normalized = _FILE_TYPE_ALIASES.get(file_type.strip().lower())
            if normalized is None:
                raise UnsupportedDocumentType(f"unsupported explicit file_type: {file_type}")
            return normalized
        selected_mime = (mime_type or mimetypes.guess_type(candidate.name)[0] or "").split(";", 1)[0].lower()
        if selected_mime in _FILE_TYPE_BY_MIME:
            return _FILE_TYPE_BY_MIME[selected_mime]
        filename_type = _file_type_from_name(candidate.name)
        if filename_type is not None:
            return filename_type
        suffix_type = _FILE_TYPE_BY_SUFFIX.get(candidate.suffix.lower())
        if suffix_type is None:
            raise UnsupportedDocumentType(
                f"unsupported document type for '{candidate.name}' (MIME={selected_mime or 'unknown'})"
            )
        return suffix_type

    def load(
        self,
        path: str | Path,
        *,
        source_id: str,
        relative_path: str | None = None,
        file_type: str | None = None,
        mime_type: str | None = None,
        base_metadata: Mapping[str, Any] | None = None,
    ) -> list[DocumentRecord]:
        candidate = Path(path).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"document does not exist: {candidate}")
        logical_path = (relative_path or candidate.name).replace("\\", "/")
        selected_type = self.resolve_file_type(
            candidate, file_type=file_type, mime_type=mime_type
        )
        units, warnings, degraded = self._parse(candidate, selected_type)
        if not units:
            raise EmptyDocumentError(f"parser returned no records for {logical_path}")

        records: list[DocumentRecord] = []
        multiple = len(units) > 1
        for ordinal, unit in enumerate(units, start=1):
            if not unit.content.strip():
                continue
            identity_suffix = unit.identity_suffix or (f"unit:{ordinal}" if multiple else "")
            identity = logical_path if not identity_suffix else f"{logical_path}#{identity_suffix}"
            unit_metadata = dict(unit.metadata)
            page_number = unit_metadata.get("page_number", unit_metadata.get("page"))
            sheet_name = unit_metadata.get("sheet_name", unit_metadata.get("sheet"))
            table_id = unit_metadata.get("table_id", unit_metadata.get("table"))
            block_id = unit_metadata.get("block_id")
            parse_warnings = [
                *warnings,
                *_warning_list(
                    unit_metadata.get(
                        "parse_warnings", unit_metadata.get("warnings", [])
                    )
                ),
            ]
            parse_degraded = bool(
                degraded
                or unit_metadata.get(
                    "parse_degraded", unit_metadata.get("degraded", False)
                )
            )
            metadata = {
                **dict(base_metadata or {}),
                **unit_metadata,
                "identity": identity,
                "file_type": selected_type,
                "content_format": unit.content_format,
                "parser_backend": unit.parser_backend,
                "parser_version": unit_metadata.get("parser_version")
                or _parser_version(unit.parser_backend),
                "page_number": page_number,
                "sheet_name": sheet_name,
                "table_id": table_id,
                "block_id": block_id,
                "parse_warnings": parse_warnings,
                "parse_degraded": parse_degraded,
                # Compatibility aliases retained for existing citation/UI code.
                "page": page_number,
                "sheet": sheet_name,
                "table": table_id,
                "warnings": parse_warnings,
                "degraded": parse_degraded,
            }
            metadata.setdefault("record_kind", "file")
            records.append(
                DocumentRecord(
                    source_id=source_id,
                    relative_path=logical_path,
                    content=unit.content,
                    media_type=unit.media_type,
                    title=unit.title or candidate.name,
                    language=unit.language,
                    metadata=metadata,
                )
            )
        if not records:
            raise EmptyDocumentError(f"document contained no indexable content: {logical_path}")
        return records

    def _parse(
        self, path: Path, file_type: str
    ) -> tuple[list[ParsedDocumentUnit], list[str], bool]:
        if file_type in {"markdown", "text", "code", "json", "yaml", "toml"}:
            return self._load_textual(path, file_type), [], False
        if file_type == "csv":
            return self._load_csv(path), [], False
        if file_type == "xlsx":
            return self._load_xlsx(path), [], False
        if file_type == "pdf":
            return self._load_pdf(path)
        raise UnsupportedDocumentType(f"unsupported normalized file type: {file_type}")

    def _load_textual(self, path: Path, file_type: str) -> list[ParsedDocumentUnit]:
        text = _read_text(path)
        if not text.strip():
            raise EmptyDocumentError(f"empty {file_type} document: {path.name}")
        suffix = path.suffix.lower()
        language = _LANGUAGE_BY_SUFFIX.get(suffix, file_type)
        media_type = mimetypes.guess_type(path.name)[0] or "text/plain"
        if path.name.lower().startswith("dockerfile"):
            language = "dockerfile"
            media_type = "text/x-dockerfile"
        elif path.name.lower().startswith("makefile"):
            language = "makefile"
            media_type = "text/x-makefile"
        backend = "builtin-text"
        if file_type == "json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                raise RecoverableParseError(f"invalid JSON in {path.name}: {exc.msg}") from exc
            backend = "python-json"
        elif file_type == "yaml":
            try:
                yaml = importlib.import_module("yaml")
                list(yaml.safe_load_all(text))
            except ImportError as exc:
                raise BackendUnavailable("PyYAML is required to parse YAML") from exc
            except Exception as exc:
                raise RecoverableParseError(f"invalid YAML in {path.name}") from exc
            backend = "pyyaml"
        elif file_type == "toml":
            try:
                tomllib.loads(text)
            except tomllib.TOMLDecodeError as exc:
                raise RecoverableParseError(f"invalid TOML in {path.name}: {exc}") from exc
            backend = "python-tomllib"
        return [
            ParsedDocumentUnit(
                content=text,
                title=_markdown_title(text) if file_type == "markdown" else path.name,
                media_type=media_type,
                language=language,
                content_format=file_type,
                parser_backend=backend,
            )
        ]

    def _load_csv(self, path: Path) -> list[ParsedDocumentUnit]:
        text = _read_text(path)
        try:
            # ``splitlines`` corrupts RFC 4180 quoted multiline fields.  The
            # newline-aware text stream lets ``csv`` own record boundaries.
            rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
        except csv.Error as exc:
            raise RecoverableParseError(f"invalid CSV in {path.name}: {exc}") from exc
        if not rows or not any(any(cell.strip() for cell in row) for row in rows):
            raise EmptyDocumentError(f"empty CSV document: {path.name}")
        width = max(len(row) for row in rows)
        normalized = [row + [""] * (width - len(row)) for row in rows]
        content = _rows_to_markdown(normalized)
        return [
            ParsedDocumentUnit(
                content=content,
                title=path.stem,
                media_type="text/csv",
                content_format="markdown-table",
                parser_backend="python-csv",
                identity_suffix=f"table:{path.stem}",
                metadata={
                    "table": path.stem,
                    "row_count": len(normalized),
                    "column_count": width,
                },
            )
        ]

    def _load_xlsx(self, path: Path) -> list[ParsedDocumentUnit]:
        try:
            openpyxl = importlib.import_module("openpyxl")
        except ImportError as exc:
            raise BackendUnavailable("openpyxl is required to parse XLSX files") from exc
        try:
            workbook = openpyxl.load_workbook(path, read_only=True, data_only=False)
        except Exception as exc:
            raise RecoverableParseError(f"unable to open XLSX file: {path.name}") from exc
        units: list[ParsedDocumentUnit] = []
        try:
            for worksheet in workbook.worksheets:
                rows = [
                    ["" if cell is None else str(cell) for cell in row]
                    for row in worksheet.iter_rows(values_only=True)
                ]
                while rows and not any(cell.strip() for cell in rows[-1]):
                    rows.pop()
                if not rows or not any(any(cell.strip() for cell in row) for row in rows):
                    continue
                width = max(len(row) for row in rows)
                normalized = [row + [""] * (width - len(row)) for row in rows]
                units.append(
                    ParsedDocumentUnit(
                        content=_rows_to_markdown(normalized),
                        title=f"{path.stem} / {worksheet.title}",
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        content_format="markdown-table",
                        parser_backend="openpyxl",
                        identity_suffix=f"sheet:{worksheet.title}",
                        metadata={
                            "sheet": worksheet.title,
                            "table": worksheet.title,
                            "row_count": len(normalized),
                            "column_count": width,
                        },
                    )
                )
        finally:
            workbook.close()
        if not units:
            raise EmptyDocumentError(f"XLSX contains no non-empty worksheets: {path.name}")
        return units

    def _load_pdf(
        self, path: Path
    ) -> tuple[list[ParsedDocumentUnit], list[str], bool]:
        fallback_warning = ""
        if self.prefer_mineru:
            with tempfile.TemporaryDirectory(prefix="rag-mineru-") as temporary:
                try:
                    units = self.mineru_backend.parse(path, Path(temporary))
                    if not units:
                        raise RecoverableParseError("MinerU returned no document units")
                    return units, [], False
                except (BackendUnavailable, RecoverableParseError) as exc:
                    fallback_warning = f"MinerU fallback: {exc}"
        else:
            fallback_warning = "MinerU disabled; PyMuPDF fallback used"
        units = _parse_pdf_with_pymupdf(path)
        return units, [fallback_warning] if fallback_warning else [], True


def _parse_pdf_with_pymupdf(path: Path) -> list[ParsedDocumentUnit]:
    try:
        fitz = importlib.import_module("pymupdf")
    except ImportError:
        try:
            fitz = importlib.import_module("fitz")
        except ImportError as exc:
            raise BackendUnavailable(
                "PyMuPDF is required when MinerU is unavailable; install the 'pymupdf' package"
            ) from exc
    try:
        document = fitz.open(path)
    except Exception as exc:
        raise RecoverableParseError(f"PyMuPDF could not open {path.name}") from exc
    units: list[ParsedDocumentUnit] = []
    try:
        if getattr(document, "needs_pass", False):
            raise RecoverableParseError(f"encrypted PDF requires a password: {path.name}")
        for page_index, page in enumerate(document, start=1):
            text = (page.get_text("text") or "").strip()
            page_warnings: list[str] = []
            page_unit: ParsedDocumentUnit | None = None
            if text:
                page_unit = ParsedDocumentUnit(
                    content=text,
                    title=f"{path.stem} / page {page_index}",
                    media_type="application/pdf",
                    content_format="pdf-text",
                    parser_backend="pymupdf",
                    identity_suffix=f"page:{page_index}",
                    metadata={"page_number": page_index},
                )
                units.append(page_unit)
            find_tables = getattr(page, "find_tables", None)
            if callable(find_tables):
                try:
                    tables = list(getattr(find_tables(), "tables", []))
                except (RuntimeError, ValueError) as exc:
                    tables = []
                    page_warnings.append(
                        f"PyMuPDF table extraction failed on page {page_index}: {type(exc).__name__}"
                    )
                for table_index, table in enumerate(tables, start=1):
                    try:
                        extracted = table.extract()
                    except Exception as exc:  # third-party backends expose varied errors
                        page_warnings.append(
                            "PyMuPDF table extraction failed on page "
                            f"{page_index}, table {table_index}: {type(exc).__name__}"
                        )
                        continue
                    rows = [
                        ["" if cell is None else str(cell) for cell in row]
                        for row in (extracted or [])
                    ]
                    if not rows or not any(any(cell.strip() for cell in row) for row in rows):
                        continue
                    width = max(len(row) for row in rows)
                    normalized = [row + [""] * (width - len(row)) for row in rows]
                    table_unit = ParsedDocumentUnit(
                            content=_rows_to_markdown(normalized),
                            title=f"{path.stem} / page {page_index} / table {table_index}",
                            media_type="application/pdf",
                            content_format="markdown-table",
                            parser_backend="pymupdf",
                            identity_suffix=f"page:{page_index}:table:{table_index}",
                            metadata={
                                "page_number": page_index,
                                "table_id": f"page-{page_index}-table-{table_index}",
                            },
                        )
                    units.append(table_unit)
            if page_warnings:
                page_units = [
                    unit
                    for unit in units
                    if unit.metadata.get("page_number") == page_index
                ]
                for unit in page_units:
                    unit.metadata.setdefault("parse_warnings", []).extend(page_warnings)
    finally:
        document.close()
    if not units:
        raise EmptyDocumentError(
            f"PDF has no extractable text or tables: {path.name}; it may be scanned and require OCR/MinerU"
        )
    return units


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return _validate_decoded_text(data.decode("utf-16"), path)
        except UnicodeDecodeError as exc:
            raise RecoverableParseError(
                f"invalid BOM-declared UTF-16 text: {path.name}"
            ) from exc
    if b"\x00" in data:
        raise RecoverableParseError(f"binary data is not valid text: {path.name}")
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return _validate_decoded_text(data.decode(encoding), path)
        except UnicodeDecodeError:
            continue
    raise RecoverableParseError(f"unsupported text encoding: {path.name}")


def _validate_decoded_text(text: str, path: Path) -> str:
    forbidden = [
        char
        for char in text
        if (ord(char) < 32 and char not in {"\t", "\n", "\r", "\f"})
        or ord(char) == 127
    ]
    if forbidden:
        raise RecoverableParseError(f"binary control bytes are not valid text: {path.name}")
    return text


def _markdown_title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _file_type_from_name(name: str) -> str | None:
    lowered = name.lower()
    if lowered.startswith(("dockerfile", "makefile")) or lowered == "procfile":
        return "code"
    return None


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """Render every cell without row/column truncation."""

    if not rows:
        return ""

    def escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")

    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    lines = [
        "| " + " | ".join(escape(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(escape(cell) for cell in row) + " |" for row in padded[1:]
    )
    return "\n".join(lines)


def _mineru_json_units(value: Any, *, artifact: str) -> list[ParsedDocumentUnit]:
    """Adapt supported MinerU content-list/middle-JSON shapes.

    Each logical block is kept separate.  This preserves retrieval coordinates
    and prevents a single large JSON artifact from becoming one opaque chunk.
    """

    blocks: list[tuple[Mapping[str, Any], int | None, int]] = []

    def add_blocks(items: Any, page_number: int | None) -> None:
        if not isinstance(items, list):
            return
        for ordinal, item in enumerate(items, start=1):
            if isinstance(item, Mapping):
                blocks.append((item, _mineru_page_number(item, page_number), ordinal))

    if isinstance(value, list):
        add_blocks(value, None)
    elif isinstance(value, Mapping):
        content_list = value.get("content_list")
        if isinstance(content_list, list):
            add_blocks(content_list, None)
        pdf_info = value.get("pdf_info")
        if isinstance(pdf_info, list):
            for page_ordinal, page in enumerate(pdf_info, start=1):
                if not isinstance(page, Mapping):
                    continue
                page_number = _mineru_page_number(page, page_ordinal)
                add_blocks(page.get("para_blocks") or page.get("blocks"), page_number)
        pages = value.get("pages")
        if isinstance(pages, list):
            for page_ordinal, page in enumerate(pages, start=1):
                if not isinstance(page, Mapping):
                    continue
                page_number = _mineru_page_number(page, page_ordinal)
                add_blocks(
                    page.get("para_blocks")
                    or page.get("blocks")
                    or page.get("content_list"),
                    page_number,
                )
        if not blocks:
            add_blocks(value.get("para_blocks") or value.get("blocks"), None)

    units: list[ParsedDocumentUnit] = []
    table_ordinals: dict[int | None, int] = {}
    for item, page_number, block_ordinal in blocks:
        kind = str(item.get("type") or item.get("category") or "text").strip().lower()
        content = _mineru_block_content(item, kind)
        if not content:
            continue
        explicit_block = item.get("block_id", item.get("id"))
        block_id = _stable_coordinate(
            str(explicit_block) if explicit_block not in {None, ""} else f"block-{block_ordinal:04d}"
        )
        page_coordinate = str(page_number) if page_number is not None else "unknown"
        metadata: dict[str, Any] = {
            "mineru_artifact": artifact,
            "mineru_block_type": kind,
            "page_number": page_number,
            "block_id": block_id,
        }
        if "table" in kind:
            table_ordinals[page_number] = table_ordinals.get(page_number, 0) + 1
            explicit_table = item.get("table_id")
            table_id = _stable_coordinate(
                str(explicit_table)
                if explicit_table not in {None, ""}
                else f"page-{page_coordinate}-table-{table_ordinals[page_number]:04d}"
            )
            metadata["table_id"] = table_id
            identity_suffix = f"page:{page_coordinate}:table:{table_id}"
            lowered = content.lstrip().lower()
            content_format = (
                "html-table"
                if lowered.startswith("<table")
                else "markdown-table" if "|" in content else "structured-table"
            )
            title = f"page {page_coordinate} / table {table_ordinals[page_number]}"
        else:
            identity_suffix = f"page:{page_coordinate}:block:{block_id}"
            content_format = "json-structured-text"
            title = f"page {page_coordinate} / {kind or 'block'}"
        units.append(
            ParsedDocumentUnit(
                content=content,
                title=title,
                media_type="application/pdf",
                content_format=content_format,
                parser_backend="mineru-cli",
                identity_suffix=identity_suffix,
                metadata=metadata,
            )
        )
    return units


def _mineru_page_number(item: Mapping[str, Any], default: int | None) -> int | None:
    if "page_idx" in item:
        try:
            return int(item["page_idx"]) + 1
        except (TypeError, ValueError):
            return default
    for key in ("page_number", "page_no", "page"):
        if key in item:
            try:
                return int(item[key])
            except (TypeError, ValueError):
                return default
    return default


def _mineru_block_content(item: Mapping[str, Any], kind: str) -> str:
    fragments: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized and normalized not in fragments:
                fragments.append(normalized)
        elif isinstance(value, list):
            for nested in value:
                if isinstance(nested, str):
                    add(nested)
                elif isinstance(nested, Mapping):
                    add(nested.get("text", nested.get("content")))

    if "table" in kind:
        add(item.get("table_caption"))
        for key in ("table_body", "html", "markdown", "md", "text", "content"):
            add(item.get(key))
        add(item.get("table_footnote"))
    else:
        for key in ("text", "content", "markdown", "md", "latex"):
            add(item.get(key))

    # MinerU middle JSON commonly stores text under lines -> spans.
    for line in item.get("lines", []) if isinstance(item.get("lines"), list) else []:
        if not isinstance(line, Mapping):
            continue
        spans = line.get("spans")
        if isinstance(spans, list):
            line_fragments: list[str] = []
            for span in spans:
                if not isinstance(span, Mapping):
                    continue
                value = span.get("content", span.get("text"))
                if isinstance(value, str) and value.strip():
                    line_fragments.append(value.strip())
            add(" ".join(line_fragments))
    for nested in item.get("blocks", []) if isinstance(item.get("blocks"), list) else []:
        if isinstance(nested, Mapping):
            add(_mineru_block_content(nested, str(nested.get("type") or "text")))
    return "\n\n".join(fragments)


def _deduplicate_mineru_units(units: list[ParsedDocumentUnit]) -> list[ParsedDocumentUnit]:
    result: list[ParsedDocumentUnit] = []
    identity_hashes: dict[str, str] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for unit in units:
        digest = hashlib.sha256(unit.content.encode("utf-8")).hexdigest()
        base_identity = unit.identity_suffix
        if (base_identity, digest) in seen_pairs:
            continue
        previous = identity_hashes.get(base_identity)
        if previous is not None:
            unit.identity_suffix = f"{unit.identity_suffix}:content-{digest[:12]}"
        seen_pairs.add((base_identity, digest))
        identity_hashes[unit.identity_suffix] = digest
        result.append(unit)
    return result


def _stable_coordinate(value: str) -> str:
    normalized = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", normalized):
        return normalized
    return f"sha256-{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]}"


def _sanitize_diagnostic(
    value: str | None,
    *,
    private_paths: Iterable[str | Path] = (),
) -> str:
    text = value or ""
    for private_path in private_paths:
        rendered = str(private_path)
        if rendered:
            text = text.replace(rendered, "<private-path>")
            text = text.replace(rendered.replace("\\", "/"), "<private-path>")
    text = re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password|passwd|authorization)"
        r"\s*[:=]\s*[^\s,;]+",
        r"\1=<redacted>",
        text,
    )
    text = re.sub(r"(?i)\bbearer\s+[^\s,;]+", "Bearer <redacted>", text)
    text = re.sub(r"\b(?:sk|gh[pousr])[-_][A-Za-z0-9_-]{8,}\b", "<redacted-token>", text)
    text = re.sub(r"(?i)[A-Z]:[\\/]Users[\\/][^\\/\s]+", "<user-home>", text)
    text = " ".join(text.replace("\x00", "").split())
    return text[:400]


def _warning_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _parser_version(parser_backend: str) -> str:
    package_by_backend = {
        "pymupdf": "pymupdf",
        "openpyxl": "openpyxl",
        "pyyaml": "pyyaml",
    }
    package = package_by_backend.get(parser_backend)
    if package:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"
    if parser_backend.startswith("python-") or parser_backend == "builtin-text":
        return platform.python_version()
    return "unknown"
