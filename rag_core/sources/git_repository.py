"""Read-only ingestion adapter for an external Git repository."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import fnmatch
import os
import platform
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

from .base import CollectedSource
from .document_loader import (
    DocumentLoaderRouter,
    UnsupportedDocumentType,
    is_safe_source_path,
)
from .schema import DocumentRecord, SourceRecord, canonical_hash, content_hash, utc_now


class GitRepositoryError(RuntimeError):
    pass


@dataclass(slots=True)
class _Symbol:
    name: str
    qualified_name: str
    kind: str
    line_start: int
    line_end: int
    source: str
    docstring: str


class GitRepositorySource:
    """Collect selected text files from an external Git worktree.

    The adapter never copies files and never invokes a mutating Git command.
    Tracked and non-ignored untracked files are considered so the manifest can
    accurately describe a dirty interview/demo worktree.
    """

    def __init__(
        self,
        source_id: str,
        repository_path: str | Path,
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        include_python_symbols: bool = False,
        git_history_limit: int = 0,
        max_file_size_bytes: int = 2_000_000,
        file_types: Mapping[str, str] | None = None,
        mime_types: Mapping[str, str] | None = None,
        version: str | None = None,
        license: str = "unknown",
        metadata: dict[str, Any] | None = None,
        loader: DocumentLoaderRouter | None = None,
    ) -> None:
        self.source_id = source_id
        self.repository_path = Path(repository_path).expanduser().resolve()
        self.include = tuple(include or ("*.md", "**/*.md", "**/*.py"))
        self.exclude = tuple(exclude or ())
        self.include_python_symbols = include_python_symbols
        if git_history_limit < 0 or git_history_limit > 500:
            raise ValueError("git_history_limit must be between 0 and 500")
        self.git_history_limit = git_history_limit
        if max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be positive")
        self.max_file_size_bytes = int(max_file_size_bytes)
        self.file_types = dict(file_types or {})
        self.mime_types = dict(mime_types or {})
        self.version = version
        self.license = license
        self.metadata = dict(metadata or {})
        self.loader = loader or DocumentLoaderRouter()

    def collect(self) -> CollectedSource:
        if not self.repository_path.is_dir():
            raise GitRepositoryError(f"repository does not exist: {self.repository_path}")

        commit_sha = self._git("rev-parse", "HEAD").strip()
        status_snapshot = self._git(
            "status", "--porcelain", "--untracked-files=normal"
        )
        dirty = bool(status_snapshot.strip())
        remote_value = self._git_optional("config", "--get", "remote.origin.url")
        remote = (
            _sanitize_repository_uri(remote_value)
            if remote_value
            else f"git+local://{_safe_uri_component(self.source_id)}"
        )
        documents: list[DocumentRecord] = []

        with tempfile.TemporaryDirectory(prefix="rag-git-snapshot-") as temporary:
            snapshot_root = Path(temporary)
            for file_ordinal, relative_path in enumerate(self._candidate_files(), start=1):
                if not self._selected(relative_path) or not is_safe_source_path(relative_path):
                    continue
                absolute_path = self.repository_path / Path(relative_path)
                # ``git ls-files --cached`` also reports tracked paths deleted
                # in a dirty worktree. Their absence is represented by the next
                # manifest diff, not by a failed collection.
                if not absolute_path.is_file():
                    continue
                try:
                    resolved_path = absolute_path.resolve()
                    resolved_path.relative_to(self.repository_path)
                    if absolute_path.is_symlink():
                        continue
                except (OSError, ValueError):
                    continue
                explicit_type = _matching_override(relative_path, self.file_types)
                explicit_mime = _matching_override(relative_path, self.mime_types)
                try:
                    self.loader.resolve_file_type(
                        resolved_path,
                        file_type=explicit_type,
                        mime_type=explicit_mime,
                    )
                except UnsupportedDocumentType:
                    continue
                snapshot_bytes = _read_stable_file(
                    resolved_path,
                    logical_path=relative_path,
                    max_bytes=self.max_file_size_bytes,
                )
                if snapshot_bytes is None:
                    continue
                file_hash = content_hash(snapshot_bytes)
                snapshot_directory = snapshot_root / f"{file_ordinal:08d}"
                snapshot_directory.mkdir()
                snapshot_path = snapshot_directory / resolved_path.name
                snapshot_path.write_bytes(snapshot_bytes)
                common_metadata = {
                    "record_kind": "file",
                    "repository_uri": remote,
                    "commit_sha": commit_sha,
                    "dirty": dirty,
                    "file_hash": file_hash,
                    "file_size": len(snapshot_bytes),
                }
                file_documents = self.loader.load(
                    snapshot_path,
                    source_id=self.source_id,
                    relative_path=relative_path,
                    file_type=explicit_type,
                    mime_type=explicit_mime,
                    base_metadata=common_metadata,
                )
                documents.extend(file_documents)
                if self.include_python_symbols and absolute_path.suffix.lower() == ".py":
                    text = file_documents[0].content
                    documents.extend(
                        self._python_symbol_documents(
                            relative_path,
                            text,
                            commit_sha=commit_sha,
                            dirty=dirty,
                            repository_uri=remote,
                        )
                    )

        if self.git_history_limit:
            documents.extend(self._git_history_documents(self.git_history_limit))

        end_commit = self._git("rev-parse", "HEAD").strip()
        end_status = self._git("status", "--porcelain", "--untracked-files=normal")
        if end_commit != commit_sha or end_status != status_snapshot:
            raise GitRepositoryError(
                "repository changed while the source snapshot was being collected; retry"
            )

        documents.sort(key=lambda item: (item.relative_path, item.metadata.get("record_kind", "")))
        aggregate = canonical_hash(
            [(document.relative_path, document.content_hash) for document in documents]
        )
        record_metadata = {
            **self.metadata,
            "repository_uri": remote,
            "include": list(self.include),
            "exclude": list(self.exclude),
            "python_symbol_cards": self.include_python_symbols,
            "git_history_limit": self.git_history_limit,
            "max_file_size_bytes": self.max_file_size_bytes,
            "file_types": self.file_types,
            "mime_types": self.mime_types,
            "document_count": len(documents),
        }
        record = SourceRecord(
            source_id=self.source_id,
            source_type="git_repository",
            uri=remote,
            version=self.version or commit_sha,
            license=self.license,
            fetched_at=utc_now(),
            content_hash=aggregate,
            commit_sha=commit_sha,
            dirty=dirty,
            metadata=record_metadata,
        )
        return CollectedSource(record=record, documents=documents)

    def _git_history_documents(self, limit: int) -> list[DocumentRecord]:
        """Represent immutable commit summaries as historical evidence.

        Commit messages and changed paths can explain evolution, but they are
        deliberately tagged as ``git_commit`` so callers never treat them as
        proof of the current implementation.
        """

        raw = self._git(
            "log",
            f"--max-count={limit}",
            "--date=iso-strict",
            "--format=%H%x09%aI%x09%s",
        )
        records: list[DocumentRecord] = []
        for line in raw.splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            sha, authored_at, raw_subject = parts
            subject = _redact_sensitive_text(raw_subject)
            changed_paths = [
                value.replace("\\", "/")
                for value in self._git(
                    "diff-tree",
                    "--root",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    "-z",
                    sha,
                ).split("\0")
                if _is_safe_history_path(value, selected=self._selected)
            ]
            path_lines = (
                "\n".join("- `" + path.replace("`", "\\`") + "`" for path in changed_paths)
                or "- (no paths)"
            )
            content = (
                f"# Git commit {sha[:12]}\n\n"
                f"- Subject: {subject}\n"
                f"- Authored at: {authored_at}\n"
                f"- Commit SHA: `{sha}`\n\n"
                f"## Changed paths\n\n{path_lines}\n"
            )
            records.append(
                DocumentRecord(
                    source_id=self.source_id,
                    relative_path=f".git-history/{sha}.md",
                    content=content,
                    media_type="text/markdown",
                    language="markdown",
                    title=f"Git commit {sha[:12]}: {subject}",
                    metadata={
                        "record_kind": "git_commit",
                        "identity": f"git-commit:{sha}",
                        "commit_sha": sha,
                        "authored_at": authored_at,
                        "commit_subject": subject,
                        "changed_paths": changed_paths,
                        "file_type": "markdown",
                        "content_format": "markdown",
                        "parser_backend": "git-log",
                        "parser_version": "unknown",
                    },
                )
            )
        return records

    def _candidate_files(self) -> list[str]:
        raw = self._git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
        values = {item.replace("\\", "/") for item in raw.split("\0") if item}
        return sorted(values)

    def _selected(self, relative_path: str) -> bool:
        if self.include and not _matches_any(relative_path, self.include):
            return False
        return not _matches_any(relative_path, self.exclude)

    def _git(self, *args: str) -> str:
        command = ["git", "-C", str(self.repository_path), *args]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=15,
                shell=False,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise GitRepositoryError(
                f"git command failed ({' '.join(args)}): {type(exc).__name__}"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise GitRepositoryError(f"git command failed ({' '.join(args)}): {detail}")
        return completed.stdout

    def _git_optional(self, *args: str) -> str | None:
        try:
            value = self._git(*args).strip()
        except GitRepositoryError:
            return None
        return value or None

    def _python_symbol_documents(
        self,
        relative_path: str,
        source: str,
        *,
        commit_sha: str,
        dirty: bool,
        repository_uri: str,
    ) -> list[DocumentRecord]:
        try:
            symbols = _extract_python_symbols(source)
        except SyntaxError:
            return []

        documents = []
        for symbol in symbols:
            locator = f"{relative_path}#symbol:{symbol.qualified_name}"
            heading = (
                f"# Python {symbol.kind}: {symbol.qualified_name}\n\n"
                f"- File: `{relative_path}`\n"
                f"- Lines: {symbol.line_start}-{symbol.line_end}\n"
            )
            if symbol.docstring:
                heading += f"- Docstring: {symbol.docstring.strip()}\n"
            card = f"{heading}\n```python\n{symbol.source.rstrip()}\n```\n"
            documents.append(
                DocumentRecord(
                    source_id=self.source_id,
                    relative_path=locator,
                    content=card,
                    media_type="text/markdown",
                    title=symbol.qualified_name,
                    language="python",
                    metadata={
                        "record_kind": "python_symbol",
                        "identity": locator,
                        "parent_path": relative_path,
                        "symbol_name": symbol.name,
                        "symbol_qualified_name": symbol.qualified_name,
                        "symbol_kind": symbol.kind,
                        "line_start": symbol.line_start,
                        "line_end": symbol.line_end,
                        "repository_uri": repository_uri,
                        "commit_sha": commit_sha,
                        "dirty": dirty,
                        "file_type": "code",
                        "content_format": "markdown-code-card",
                        "parser_backend": "python-ast",
                        "parser_version": platform.python_version(),
                    },
                )
            )
        return documents


def _matches_any(relative_path: str, patterns: Iterable[str]) -> bool:
    path = relative_path.replace("\\", "/")
    pure_path = PurePosixPath(path)
    return any(
        fnmatch.fnmatchcase(path, pattern.replace("\\", "/"))
        or pure_path.match(pattern.replace("\\", "/"))
        for pattern in patterns
    )


def _matching_override(relative_path: str, values: Mapping[str, str]) -> str | None:
    for pattern, value in values.items():
        if _matches_any(relative_path, (pattern,)):
            return value
    return None


def _read_stable_file(
    path: Path,
    *,
    logical_path: str,
    max_bytes: int,
) -> bytes | None:
    """Read one self-consistent worktree snapshot without a hash/content race."""

    def signature(stat: os.stat_result) -> tuple[int, int, int, int]:
        return (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino)

    try:
        before = path.stat()
        if before.st_size > max_bytes:
            return None
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            data = handle.read(max_bytes + 1)
            opened_after = os.fstat(handle.fileno())
        after = path.stat()
    except OSError as exc:
        raise GitRepositoryError(
            f"unable to capture stable file snapshot for {logical_path}; retry"
        ) from exc
    if len(data) > max_bytes:
        return None
    if not (
        signature(before)
        == signature(opened_before)
        == signature(opened_after)
        == signature(after)
    ):
        raise GitRepositoryError(
            f"file changed while collecting {logical_path}; retry the source snapshot"
        )
    return data


def _is_safe_history_path(
    value: str,
    *,
    selected: Callable[[str], bool],
) -> bool:
    if not value or len(value) > 1_000:
        return False
    normalized = value.replace("\\", "/")
    if any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        return False
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or ".." in candidate.parts:
        return False
    if re.search(
        r"(?i)(?:api[_-]?key|access[_-]?token|token|secret|password|passwd)\s*[:=]",
        normalized,
    ):
        return False
    return is_safe_source_path(normalized) and bool(selected(normalized))


def _redact_sensitive_text(value: str) -> str:
    text = " ".join(value.replace("\x00", "").split())
    text = re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password|passwd|authorization)"
        r"\s*[:=]\s*[^\s,;]+",
        r"\1=<redacted>",
        text,
    )
    text = re.sub(r"(?i)\bbearer\s+[^\s,;]+", "Bearer <redacted>", text)
    text = re.sub(r"\b(?:sk|gh[pousr])[-_][A-Za-z0-9_-]{8,}\b", "<redacted-token>", text)
    text = re.sub(
        r"(?i)\b[A-Z]:[\\/]Users[\\/][^\\/\s]+(?:[\\/][^\s]*)?",
        "<private-path>",
        text,
    )
    text = re.sub(r"(?<!\w)/(?:home|Users)/[^/\s]+(?:/[^\s]*)?", "<private-path>", text)
    return (text[:300] or "<empty commit subject>").replace("`", "\\`")


def _safe_uri_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return normalized[:80] or f"source-{canonical_hash(value)[:12]}"


def _sanitize_repository_uri(value: str) -> str:
    """Strip credentials and query fragments before persisting a Git remote."""

    text = value.strip()
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith(("/", "\\")):
        return "git+local://redacted"
    if "://" in text:
        parsed = urlsplit(text)
        if parsed.scheme.lower() == "file":
            return "git+local://redacted"
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port is not None:
            host = f"{host}:{port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    # SCP-like SSH remotes may contain a user name. It is not required for a
    # citation and can itself be sensitive, so retain only host:path.
    if "@" in text and ":" in text.split("@", 1)[1]:
        return text.split("@", 1)[1]
    return text


def _extract_python_symbols(source: str) -> list[_Symbol]:
    tree = ast.parse(source)
    lines = source.splitlines()
    symbols: list[_Symbol] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.parents: list[str] = []

        def _record(self, node: ast.AST, name: str, kind: str) -> None:
            decorators = getattr(node, "decorator_list", [])
            starts = [getattr(item, "lineno", getattr(node, "lineno", 1)) for item in decorators]
            starts.append(getattr(node, "lineno", 1))
            start = min(starts)
            end = getattr(node, "end_lineno", None) or getattr(node, "lineno", start)
            qualified_name = ".".join([*self.parents, name])
            segment = "\n".join(lines[start - 1 : end])
            symbols.append(
                _Symbol(
                    name=name,
                    qualified_name=qualified_name,
                    kind=kind,
                    line_start=start,
                    line_end=end,
                    source=segment,
                    docstring=ast.get_docstring(node) or "",  # type: ignore[arg-type]
                )
            )

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._record(node, node.name, "class")
            self.parents.append(node.name)
            self.generic_visit(node)
            self.parents.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record(node, node.name, "function")
            self.parents.append(node.name)
            self.generic_visit(node)
            self.parents.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record(node, node.name, "async_function")
            self.parents.append(node.name)
            self.generic_visit(node)
            self.parents.pop()

    Visitor().visit(tree)
    return symbols
