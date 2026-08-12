from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from rag_core.engineering.workflows import sync_engineering_sources
from rag_core.ingestion import BuildManifest, IngestionPipeline
from rag_core.sources import (
    DocumentLoaderRouter,
    FetchResponse,
    GitRepositorySource,
    OfficialWebSource,
    SourceCatalog,
)
from rag_core.sources.official_web import clean_html_document


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _create_repository(root: Path) -> Path:
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "ingestion-test@example.invalid")
    _git(root, "config", "user.name", "Ingestion Test")
    (root / ".gitignore").write_text("ignored/\n__pycache__/\n", encoding="utf-8")
    (root / "README.md").write_text("# Mini Test\n\nInitial documentation.\n", encoding="utf-8")
    package = root / "mini_nanobot"
    package.mkdir()
    (package / "agent.py").write_text(
        'class Agent:\n'
        '    """Small test agent."""\n\n'
        '    async def run(self, task: str) -> str:\n'
        '        return task\n\n'
        'def helper(value: int) -> int:\n'
        '    return value + 1\n',
        encoding="utf-8",
    )
    (package / "excluded.py").write_text("SECRET = True\n", encoding="utf-8")
    ignored = root / "ignored"
    ignored.mkdir()
    (ignored / "private.md").write_text("not indexed", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "initial")
    return root


def test_git_repository_source_tracks_revision_dirty_hashes_and_symbols(tmp_path: Path) -> None:
    repository = _create_repository(tmp_path / "mini")
    source = GitRepositorySource(
        "mini_nanobot",
        repository,
        include=["README.md", "mini_nanobot/*.py"],
        exclude=["mini_nanobot/excluded.py"],
        include_python_symbols=True,
        git_history_limit=3,
        license="unknown-local-project",
    )

    first = source.collect()
    assert first.record.commit_sha == _git(repository, "rev-parse", "HEAD")
    assert first.record.dirty is False
    assert len(first.record.commit_sha or "") == 40
    paths = {document.relative_path for document in first.documents}
    assert "README.md" in paths
    assert "mini_nanobot/agent.py" in paths
    assert "mini_nanobot/excluded.py" not in paths
    assert "mini_nanobot/agent.py#symbol:Agent" in paths
    assert "mini_nanobot/agent.py#symbol:Agent.run" in paths
    assert "mini_nanobot/agent.py#symbol:helper" in paths
    history = [
        document for document in first.documents
        if document.metadata.get("record_kind") == "git_commit"
    ]
    assert len(history) == 1
    assert history[0].relative_path.startswith(".git-history/")
    assert "README.md" in history[0].content
    symbol = next(
        document
        for document in first.documents
        if document.relative_path.endswith("#symbol:Agent.run")
    )
    assert symbol.metadata["line_start"] < symbol.metadata["line_end"]
    assert "async def run" in symbol.content
    symbol_build = IngestionPipeline(chunk_size=80, chunk_overlap=8).build([first]).manifest
    symbol_doc = next(
        document
        for document in symbol_build.documents
        if document.relative_path.endswith("#symbol:Agent.run")
    )
    symbol_chunk = next(chunk for chunk in symbol_build.chunks if chunk.doc_id == symbol_doc.doc_id)
    assert symbol_chunk.metadata["line_start"] == symbol.metadata["line_start"]
    assert "chunk_line_start" in symbol_chunk.metadata

    old_hash = first.record.content_hash
    (repository / "README.md").write_text("# Mini Test\n\nChanged documentation.\n", encoding="utf-8")
    second = source.collect()
    assert second.record.dirty is True
    assert second.record.commit_sha == first.record.commit_sha
    assert second.record.content_hash != old_hash


def test_git_repository_source_strips_credentials_from_remote(tmp_path: Path) -> None:
    repository = _create_repository(tmp_path / "mini")
    _git(
        repository,
        "remote",
        "add",
        "origin",
        "https://user:super-secret@example.test/org/repo.git?token=also-secret",
    )

    snapshot = GitRepositorySource(
        "mini_nanobot", repository, include=["README.md"]
    ).collect()

    serialized = json.dumps(snapshot.record.to_dict(), ensure_ascii=False)
    assert snapshot.record.uri == "https://example.test/org/repo.git"
    assert "super-secret" not in serialized
    assert "also-secret" not in serialized


def test_git_snapshot_hash_and_content_use_same_bytes_and_hide_local_path(
    tmp_path: Path,
) -> None:
    repository = _create_repository(tmp_path / "private-workspace")
    original = repository / "README.md"
    captured_text = "# Dirty but stable\n\nWorking tree content.\n"
    original.write_text(captured_text, encoding="utf-8")

    class MutatingLoader(DocumentLoaderRouter):
        def load(self, path, **kwargs):
            # Mutation occurs after the source captured its private temporary
            # byte snapshot but before parsing that snapshot.
            original.write_text("# Changed concurrently\n", encoding="utf-8")
            try:
                return super().load(path, **kwargs)
            finally:
                original.write_text(captured_text, encoding="utf-8")

    snapshot = GitRepositorySource(
        "mini_nanobot",
        repository,
        include=["README.md"],
        loader=MutatingLoader(prefer_mineru=False),
    ).collect()

    document = snapshot.documents[0]
    assert snapshot.record.dirty is True
    assert "Working tree content" in document.content
    assert "Changed concurrently" not in document.content
    assert document.metadata["file_hash"] == document.content_hash
    serialized = json.dumps(
        {
            "source": snapshot.record.to_dict(),
            "documents": [item.to_dict() for item in snapshot.documents],
        },
        ensure_ascii=False,
    )
    assert str(repository) not in serialized
    assert snapshot.record.uri == "git+local://mini_nanobot"


def test_git_history_filters_sensitive_paths_and_redacts_secret_subjects(
    tmp_path: Path,
) -> None:
    repository = _create_repository(tmp_path / "mini")
    (repository / ".env.production").write_text(
        "TOKEN=never-store", encoding="utf-8"
    )
    (repository / "README.md").write_text("# Safe\n", encoding="utf-8")
    _git(repository, "add", "README.md", ".env.production")
    _git(repository, "commit", "-m", "rotate API_KEY=super-secret-value")

    snapshot = GitRepositorySource(
        "mini_nanobot",
        repository,
        include=["README.md", ".env.production"],
        git_history_limit=2,
    ).collect()
    history = [
        document
        for document in snapshot.documents
        if document.metadata.get("record_kind") == "git_commit"
    ]
    serialized = json.dumps([item.to_dict() for item in history], ensure_ascii=False)

    assert "super-secret-value" not in serialized
    assert "API_KEY=<redacted>" in serialized
    assert ".env.production" not in serialized


def test_official_web_source_is_offline_whitelisted_and_cleans_main_content() -> None:
    calls: list[str] = []

    def fake_fetch(url: str) -> FetchResponse:
        calls.append(url)
        return FetchResponse(
            url="https://docs.example.org/guide",
            body="""
                <html><head><title>Agent Guide</title><script>alert('x')</script></head>
                <body><nav>Navigation noise</nav><div role="main">
                <h1>Tools</h1><p>Use explicit schemas.</p><pre>tool.call()</pre>
                </div><p>Outside main noise</p><footer>Footer noise</footer></body></html>
            """,
            content_type="text/html; charset=utf-8",
            etag='"abc"',
            last_modified="Tue, 14 Jul 2026 00:00:00 GMT",
        )

    source = OfficialWebSource(
        "official_docs",
        ["https://docs.example.org/guide#section"],
        allowed_domains=["example.org"],
        version="v1",
        license="see-source-terms",
        fetcher=fake_fetch,
    )
    snapshot = source.collect()

    assert calls == ["https://docs.example.org/guide"]
    assert snapshot.record.source_type == "official_web"
    assert snapshot.record.version == "v1"
    assert snapshot.record.license == "see-source-terms"
    assert snapshot.record.content_hash
    assert len(snapshot.documents) == 1
    document = snapshot.documents[0]
    assert document.relative_path == "https://docs.example.org/guide"
    assert "# Agent Guide" in document.content
    assert "# Tools" in document.content
    assert "Use explicit schemas." in document.content
    assert "Navigation noise" not in document.content
    assert "Outside main noise" not in document.content
    assert "Footer noise" not in document.content
    assert "alert('x')" not in document.content
    assert document.metadata["fetched_at"]
    assert document.metadata["etag"] == '"abc"'

    with pytest.raises(ValueError, match="allowlist"):
        OfficialWebSource(
            "bad",
            ["https://untrusted.invalid/docs"],
            allowed_domains=["example.org"],
            fetcher=fake_fetch,
        )


def test_html_cleaner_keeps_nested_same_name_elements_inside_main() -> None:
    content, title = clean_html_document(
        "<html><head><title>Nested docs</title></head><body>"
        "<div role='main'><div>first<div>deep</div>last</div>tail</div>"
        "<footer>noise</footer></body></html>"
    )

    assert title == "Nested docs"
    assert "first" in content
    assert "deep" in content
    assert "last" in content
    assert "tail" in content
    assert "noise" not in content


def test_official_web_fetch_retries_transient_disconnect(monkeypatch) -> None:
    calls = 0

    class _Headers(dict):
        def get_content_type(self):
            return "text/html"

        def get_content_charset(self):
            return "utf-8"

    class _Response:
        headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def geturl(self):
            return "https://docs.example.org/retry"

        def read(self, size):
            return b"<main>recovered</main>"

    def flaky_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RemoteDisconnected("transient")
        return _Response()

    from http.client import RemoteDisconnected

    monkeypatch.setattr("rag_core.sources.official_web.urlopen", flaky_urlopen)
    monkeypatch.setattr("rag_core.sources.official_web.time.sleep", lambda _: None)
    source = OfficialWebSource(
        "retry", ["https://docs.example.org/retry"],
        allowed_domains=["example.org"], fetch_retries=2,
    )

    snapshot = source.collect()
    assert calls == 2
    assert snapshot.documents[0].content == "recovered"


def test_source_catalog_loads_yaml_and_json_without_collecting_network(tmp_path: Path) -> None:
    yaml_path = tmp_path / "catalog.yaml"
    yaml_path.write_text(
        """
sources:
  - id: local_project
    type: git_repository
    path: ./external
    include: [README.md]
    python_symbol_cards: true
  - id: official
    type: official_web
    urls: [https://docs.example.org/guide]
    allowed_domains: [example.org]
    version: v1
""".strip(),
        encoding="utf-8",
    )
    yaml_catalog = SourceCatalog.load(yaml_path)
    assert [entry.source_id for entry in yaml_catalog.entries] == ["local_project", "official"]
    sources = yaml_catalog.create_sources(
        web_fetcher=lambda url: FetchResponse(url=url, body="<main>offline</main>")
    )
    assert isinstance(sources[0], GitRepositorySource)
    assert sources[0].repository_path == (tmp_path / "external").resolve()
    assert isinstance(sources[1], OfficialWebSource)

    json_path = tmp_path / "catalog.json"
    json_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "local_project",
                        "type": "git",
                        "path": "./external",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert SourceCatalog.load(json_path).entries[0].source_type == "git"


@pytest.mark.parametrize("catalog_name", ["catalog.yaml", "catalog.example.yaml"])
def test_project_catalog_allowlists_langchain_official_overview(
    catalog_name: str,
) -> None:
    catalog_path = Path(__file__).parents[1] / "data" / "sources" / catalog_name
    catalog = SourceCatalog.load(catalog_path)
    entries = {entry.source_id: entry for entry in catalog.entries}

    langchain = entries["langchain_overview"]
    assert langchain.source_type == "official_web"
    assert langchain.options["allowed_domains"] == ["docs.langchain.com"]
    assert langchain.options["urls"] == [
        "https://docs.langchain.com/oss/python/langchain/overview"
    ]
    assert langchain.options["metadata"]["corpus"] == "official"
    assert langchain.options["metadata"]["authority"] == "official"
    assert "LangChain" in langchain.options["metadata"]["scope_aliases"]
    assert "LangChain" in langchain.options["metadata"]["covered_topics"]


def test_ingestion_pipeline_stable_ids_full_manifest_and_incremental_diff(tmp_path: Path) -> None:
    repository = _create_repository(tmp_path / "mini")
    (repository / "notes.md").write_text("# Notes\n\nKeep me initially.\n", encoding="utf-8")
    _git(repository, "add", "notes.md")
    _git(repository, "commit", "-m", "add notes")
    source = GitRepositorySource(
        "mini_nanobot",
        repository,
        include=["*.md"],
    )
    pipeline = IngestionPipeline(
        chunk_size=24,
        chunk_overlap=4,
        clock=lambda: "2026-07-14T00:00:00+00:00",
    )
    manifest_path = tmp_path / "builds" / "initial.json"
    first = pipeline.build([source], output_path=manifest_path)
    loaded = BuildManifest.read(manifest_path)
    assert loaded.to_dict() == first.manifest.to_dict()
    assert first.diff.documents.added
    assert not first.diff.documents.modified
    assert not first.diff.documents.deleted

    first_by_path = {document.relative_path: document for document in first.manifest.documents}
    (repository / "README.md").write_text("# Mini Test\n\nModified in place.\n", encoding="utf-8")
    (repository / "notes.md").unlink()
    (repository / "new.md").write_text("# New\n\nAdded document.\n", encoding="utf-8")

    second = pipeline.build([source], previous=first.manifest)
    second_by_path = {document.relative_path: document for document in second.manifest.documents}
    assert second_by_path["README.md"].doc_id == first_by_path["README.md"].doc_id
    assert first_by_path["README.md"].doc_id in second.diff.documents.modified
    assert second_by_path["new.md"].doc_id in second.diff.documents.added
    assert first_by_path["notes.md"].doc_id in second.diff.documents.deleted
    assert second.diff.chunks.changed
    assert second.manifest.build_id != first.manifest.build_id


@pytest.mark.parametrize(
    ("schema_version", "message"),
    [
        (None, "no schema_version"),
        ("1.0", "older than supported"),
        ("99.0", "newer than supported"),
        ("future", "invalid"),
    ],
)
def test_build_manifest_read_rejects_missing_old_and_future_schema_versions(
    tmp_path: Path,
    schema_version: str | None,
    message: str,
) -> None:
    repository = _create_repository(tmp_path / "mini")
    manifest = IngestionPipeline().build(
        [GitRepositorySource("mini", repository, include=["README.md"])]
    ).manifest.to_dict()
    if schema_version is None:
        manifest.pop("schema_version")
    else:
        manifest["schema_version"] = schema_version
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        BuildManifest.read(path)


def test_build_manifest_rejects_pipeline_metadata_version_mismatch(tmp_path: Path) -> None:
    repository = _create_repository(tmp_path / "mini")
    manifest = IngestionPipeline().build(
        [GitRepositorySource("mini", repository, include=["README.md"])]
    ).manifest.to_dict()
    manifest["metadata"]["pipeline_version"] = "1.0"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="pipeline_version"):
        BuildManifest.read(path)


def test_sources_sync_requires_explicit_full_rebuild_for_old_manifest(tmp_path: Path) -> None:
    repository = _create_repository(tmp_path / "mini")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "mini",
                        "type": "git_repository",
                        "path": str(repository),
                        "include": ["README.md"],
                        "git_history_limit": 0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "current.json"
    old_payload = IngestionPipeline().build(
        [GitRepositorySource("mini", repository, include=["README.md"])]
    ).manifest.to_dict()
    old_payload["schema_version"] = "1.0"
    old_payload["metadata"]["pipeline_version"] = "1.0"
    manifest_path.write_text(json.dumps(old_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="older than supported"):
        sync_engineering_sources(catalog_path, manifest_path)

    rebuilt = sync_engineering_sources(
        catalog_path,
        manifest_path,
        full_rebuild=True,
    )
    loaded = BuildManifest.read(manifest_path)
    assert loaded.to_dict() == rebuilt.manifest.to_dict()
    assert loaded.schema_version == "1.1"
    assert loaded.metadata["pipeline_version"] == "1.1"
    assert rebuilt.diff.documents.added
    assert not rebuilt.diff.documents.modified
    assert not rebuilt.diff.documents.deleted
