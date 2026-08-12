from __future__ import annotations

from pathlib import Path
import os
from types import SimpleNamespace

import pytest

from rag_core.sources import (
    BackendUnavailable,
    DocumentLoaderRouter,
    EmptyDocumentError,
    LocalDirectorySource,
    GitRepositorySource,
    MinerUOutputAdapter,
    MinerUCliBackend,
    ParsedDocumentUnit,
    RecoverableParseError,
    SourceCatalog,
)


def _git(repository: Path, *args: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


class _SuccessfulMinerU:
    name = "fake-mineru"

    def parse(self, path: Path, output_dir: Path) -> list[ParsedDocumentUnit]:
        assert output_dir.is_dir()
        return [
            ParsedDocumentUnit(
                content="# Parsed PDF\n\nMinerU content.",
                title="Parsed PDF",
                media_type="text/markdown",
                language="markdown",
                content_format="markdown",
                parser_backend="fake-mineru",
                identity_suffix="page:1",
                metadata={"page": 1},
            )
        ]


class _UnavailableMinerU:
    name = "fake-unavailable"

    def parse(self, path: Path, output_dir: Path) -> list[ParsedDocumentUnit]:
        raise BackendUnavailable("not installed")


def test_router_priority_is_explicit_type_then_mime_then_suffix(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.txt"
    path.write_text('{"framework": "LangChain"}', encoding="utf-8")
    router = DocumentLoaderRouter(prefer_mineru=False)

    explicit = router.load(
        path,
        source_id="docs",
        file_type="code",
        mime_type="application/json",
    )[0]
    from_mime = router.load(
        path,
        source_id="docs",
        mime_type="application/json",
    )[0]
    from_suffix = router.load(path, source_id="docs")[0]

    assert explicit.metadata["file_type"] == "code"
    assert from_mime.metadata["file_type"] == "json"
    assert from_suffix.metadata["file_type"] == "text"
    assert from_mime.metadata["parser_backend"] == "python-json"


def test_csv_and_xlsx_keep_all_rows_and_structured_coordinates(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "name,value\n" + "\n".join(f"row-{index},{index}" for index in range(75)),
        encoding="utf-8",
    )
    router = DocumentLoaderRouter(prefer_mineru=False)
    csv_document = router.load(csv_path, source_id="tables")[0]
    assert "row-74" in csv_document.content
    assert csv_document.metadata["row_count"] == 76
    assert csv_document.metadata["table"] == "metrics"
    assert csv_document.metadata["content_format"] == "markdown-table"

    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    first = workbook.active
    first.title = "Summary"
    first.append(["metric", "value"])
    first.append(["recall", 0.9])
    details = workbook.create_sheet("Details")
    details.append(["id", "note"])
    details.append([1, "last sheet retained"])
    xlsx_path = tmp_path / "report.xlsx"
    workbook.save(xlsx_path)

    worksheets = router.load(xlsx_path, source_id="tables")
    assert [item.metadata["sheet_name"] for item in worksheets] == ["Summary", "Details"]
    assert [item.identity for item in worksheets] == [
        "report.xlsx#sheet:Summary",
        "report.xlsx#sheet:Details",
    ]
    assert "last sheet retained" in worksheets[1].content
    assert all(item.metadata["parser_backend"] == "openpyxl" for item in worksheets)


def test_csv_preserves_quoted_multiline_cells(tmp_path: Path) -> None:
    path = tmp_path / "notes.csv"
    path.write_text('id,note\n1,"first line\nsecond line"\n', encoding="utf-8")

    document = DocumentLoaderRouter(prefer_mineru=False).load(
        path, source_id="tables"
    )[0]

    assert document.metadata["row_count"] == 2
    assert "first line" in document.content
    assert "<br>second line" in document.content


def test_text_reader_accepts_bom_utf16_and_gb18030_but_rejects_binary(
    tmp_path: Path,
) -> None:
    router = DocumentLoaderRouter(prefer_mineru=False)
    utf16 = tmp_path / "utf16.txt"
    utf16.write_bytes("UTF16 文本".encode("utf-16"))
    gb = tmp_path / "gb.txt"
    gb.write_bytes("中文 GB18030".encode("gb18030"))
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"not-text\x00payload")

    assert router.load(utf16, source_id="encoding")[0].content == "UTF16 文本"
    assert router.load(gb, source_id="encoding")[0].content == "中文 GB18030"
    with pytest.raises(RecoverableParseError, match="binary"):
        router.load(binary, source_id="encoding")


def test_mineru_adapter_prefers_structured_json_and_keeps_coordinates(
    tmp_path: Path,
) -> None:
    output = tmp_path / "mineru"
    output.mkdir()
    (output / "result.md").write_text("# lossy markdown", encoding="utf-8")
    (output / "result_content_list.json").write_text(
        """
        [
          {"type": "text", "page_idx": 0, "text": "first block", "block_id": "b1"},
          {"type": "table", "page_idx": 1, "table_body": "| a | b |\\n|---|---|\\n| 1 | 2 |"}
        ]
        """.strip(),
        encoding="utf-8",
    )

    units = MinerUOutputAdapter().adapt(output)

    assert [unit.content for unit in units] == [
        "first block",
        "| a | b |\n|---|---|\n| 1 | 2 |",
    ]
    assert [unit.metadata["page_number"] for unit in units] == [1, 2]
    assert units[0].metadata["block_id"] == "b1"
    assert units[1].metadata["table_id"] == "page-2-table-0001"
    assert units[1].identity_suffix == "page:2:table:page-2-table-0001"
    assert [item.identity_suffix for item in MinerUOutputAdapter().adapt(output)] == [
        item.identity_suffix for item in units
    ]


def test_mineru_adapter_rejects_symlink_and_oversized_artifact(tmp_path: Path) -> None:
    output = tmp_path / "mineru"
    output.mkdir()
    (output / "large.json").write_text('{"text":"too large"}', encoding="utf-8")
    with pytest.raises(RecoverableParseError, match="exceeds"):
        MinerUOutputAdapter(max_artifact_bytes=4).adapt(output)

    if os.name != "nt":
        (output / "large.json").unlink()
        outside = tmp_path / "outside.md"
        outside.write_text("secret", encoding="utf-8")
        (output / "linked.md").symlink_to(outside)
        with pytest.raises(RecoverableParseError, match="symbolic-link"):
            MinerUOutputAdapter().adapt(output)


def test_mineru_failure_diagnostic_redacts_paths_and_secrets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "private" / "document.pdf"
    input_path.parent.mkdir()
    input_path.write_bytes(b"pdf")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(
        "rag_core.sources.document_loader.shutil.which", lambda _: "mineru"
    )
    monkeypatch.setattr(
        "rag_core.sources.document_loader.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2,
            stdout="",
            stderr=f"failed {input_path} API_KEY=top-secret-token",
        ),
    )

    with pytest.raises(RecoverableParseError) as captured:
        MinerUCliBackend().parse(input_path, output)

    message = str(captured.value)
    assert str(input_path) not in message
    assert "top-secret-token" not in message
    assert "<private-path>" in message
    assert "<redacted>" in message


def test_pdf_prefers_injected_mineru_and_preserves_parser_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "design.pdf"
    path.write_bytes(b"fake fixture handled by injected backend")
    document = DocumentLoaderRouter(mineru_backend=_SuccessfulMinerU()).load(
        path, source_id="design"
    )[0]

    assert document.content.startswith("# Parsed PDF")
    assert document.metadata["parser_backend"] == "fake-mineru"
    assert document.metadata["page_number"] == 1
    assert document.metadata["parse_degraded"] is False
    assert document.metadata["parse_warnings"] == []


def test_pdf_falls_back_only_for_declared_backend_failure(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "fallback.pdf"
    path.write_bytes(b"fake fixture for controlled fallback")
    fallback_calls = 0

    def fake_pymupdf(candidate: Path) -> list[ParsedDocumentUnit]:
        nonlocal fallback_calls
        fallback_calls += 1
        return [
            ParsedDocumentUnit(
                content="fallback text",
                parser_backend="pymupdf",
                content_format="pdf-text",
                metadata={"page": 1},
            )
        ]

    monkeypatch.setattr(
        "rag_core.sources.document_loader._parse_pdf_with_pymupdf",
        fake_pymupdf,
    )
    result = DocumentLoaderRouter(mineru_backend=_UnavailableMinerU()).load(
        path, source_id="pdf"
    )[0]
    assert fallback_calls == 1
    assert result.metadata["parse_degraded"] is True
    assert "MinerU fallback" in result.metadata["parse_warnings"][0]

    class ProgrammingBug:
        name = "bug"

        def parse(self, path: Path, output_dir: Path) -> list[ParsedDocumentUnit]:
            raise ValueError("implementation bug")

    with pytest.raises(ValueError, match="implementation bug"):
        DocumentLoaderRouter(mineru_backend=ProgrammingBug()).load(
            path, source_id="pdf"
        )
    assert fallback_calls == 1


def test_pymupdf_real_fixture_extracts_pages_and_rejects_scanned_empty_pdf(
    tmp_path: Path,
) -> None:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "guide.pdf"
    pdf = fitz.open()
    first = pdf.new_page()
    first.insert_text((72, 72), "LangChain is an application framework.")
    second = pdf.new_page()
    second.insert_text((72, 72), "Second page evidence.")
    pdf.save(path)
    pdf.close()

    records = DocumentLoaderRouter(prefer_mineru=False).load(path, source_id="pdf")
    assert [record.metadata["page_number"] for record in records] == [1, 2]
    assert [record.identity for record in records] == [
        "guide.pdf#page:1",
        "guide.pdf#page:2",
    ]
    assert "LangChain" in records[0].content
    assert all(record.metadata["parser_backend"] == "pymupdf" for record in records)
    assert all(record.metadata["parse_degraded"] is True for record in records)

    empty_path = tmp_path / "scanned.pdf"
    empty = fitz.open()
    empty.new_page()
    empty.save(empty_path)
    empty.close()
    with pytest.raises(EmptyDocumentError, match="scanned"):
        DocumentLoaderRouter(prefer_mineru=False).load(empty_path, source_id="pdf")


def test_pymupdf_table_extract_failure_is_isolated_and_recorded(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class BrokenTable:
        def extract(self):
            raise RuntimeError("backend detail must not abort the page")

    class Tables:
        tables = [BrokenTable()]

    class Page:
        def get_text(self, mode: str) -> str:
            assert mode == "text"
            return "page text survives"

        def find_tables(self):
            return Tables()

    class PDF:
        needs_pass = False

        def __iter__(self):
            return iter([Page()])

        def close(self):
            pass

    class FakePyMuPDF:
        @staticmethod
        def open(path: Path):
            return PDF()

    real_import = __import__("importlib").import_module

    def fake_import(name: str):
        if name == "pymupdf":
            return FakePyMuPDF()
        return real_import(name)

    monkeypatch.setattr("rag_core.sources.document_loader.importlib.import_module", fake_import)
    path = tmp_path / "broken-table.pdf"
    path.write_bytes(b"fixture")

    record = DocumentLoaderRouter(prefer_mineru=False).load(
        path, source_id="pdf"
    )[0]

    assert record.content == "page text survives"
    assert any(
        "table 1: RuntimeError" in warning
        for warning in record.metadata["parse_warnings"]
    )


def test_local_directory_is_a_catalog_source_and_never_collects_secrets(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "guide.md").write_text("# Guide\n\nPublic knowledge.", encoding="utf-8")
    (corpus / "data.csv").write_text("key,value\nmode,read-only", encoding="utf-8")
    (corpus / "Dockerfile").write_text("FROM python:3.12\n", encoding="utf-8")
    (corpus / "unknown.bin").write_bytes(b"\x00\x01\x02")
    (corpus / ".env").write_text("SECRET=never-index", encoding="utf-8")
    (corpus / "client.pem").write_text("private-key", encoding="utf-8")

    catalog = SourceCatalog.from_mapping(
        {
            "sources": [
                {
                    "id": "local_docs",
                    "type": "local_directory",
                    "path": str(corpus),
                    "include": ["*", "**/*"],
                }
            ]
        },
        base_dir=tmp_path,
    )
    source = catalog.create_sources()[0]
    assert isinstance(source, LocalDirectorySource)
    snapshot = source.collect()

    paths = {document.relative_path for document in snapshot.documents}
    assert paths == {"Dockerfile", "data.csv", "guide.md"}
    serialized = "\n".join(document.content for document in snapshot.documents)
    assert "never-index" not in serialized
    assert "private-key" not in serialized
    assert snapshot.record.source_type == "local_directory"
    docker_record = next(
        document for document in snapshot.documents
        if document.relative_path == "Dockerfile"
    )
    assert docker_record.metadata["file_type"] == "code"
    assert docker_record.language == "dockerfile"
    assert all(
        {
            "file_type",
            "content_format",
            "parser_backend",
            "parser_version",
            "parse_warnings",
            "parse_degraded",
        }
        <= set(document.metadata)
        for document in snapshot.documents
    )


def test_git_source_reuses_router_for_dockerfile_tables_and_skips_binary(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.email", "router-test@example.invalid")
    _git(repository, "config", "user.name", "Router Test")
    docker = repository / "docker"
    docker.mkdir()
    (docker / "Dockerfile.gpu").write_text("FROM python:3.12\n", encoding="utf-8")
    (repository / "metrics.csv").write_text("name,value\nrecall,0.9\n", encoding="utf-8")
    (repository / "artifact.bin").write_bytes(b"\x00\x01\x02")
    (repository / ".env.production").write_text("TOKEN=never-index", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "add heterogeneous fixtures")

    snapshot = GitRepositorySource(
        "mixed_repository", repository, include=["*", "**/*"]
    ).collect()
    paths = {document.relative_path for document in snapshot.documents}

    assert paths == {"docker/Dockerfile.gpu", "metrics.csv"}
    docker_record = next(
        document for document in snapshot.documents
        if document.relative_path == "docker/Dockerfile.gpu"
    )
    assert docker_record.language == "dockerfile"
    assert docker_record.media_type == "text/x-dockerfile"
    assert docker_record.metadata["file_type"] == "code"
    table_record = next(
        document for document in snapshot.documents
        if document.relative_path == "metrics.csv"
    )
    assert table_record.metadata["content_format"] == "markdown-table"
