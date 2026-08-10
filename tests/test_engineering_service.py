from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from fastapi.testclient import TestClient
import pytest

from engineering_api import create_app
from rag_core.engineering import (
    EngineeringIndex,
    EngineeringRAGService,
    GroundedAnswerer,
)
from rag_core.engineering.index import (
    _acquire_build_lock,
    _process_is_running,
    _recover_interrupted_publish,
    _release_build_lock,
)
from rag_core.engineering.sufficiency import EvidenceSufficiencyGuard
from rag_core.ingestion import BuildManifest
from rag_core.retrieval.engineering import (
    BM25Retriever,
    EngineeringSearchResult,
    FederatedRetriever,
    LiveASTRetriever,
    LiveCodeRetriever,
    LiveGitVerifier,
)
from rag_core.sources.schema import ChunkRecord, DocumentRecord, SourceRecord


class _FakeEmbeddingManager:
    dimension = 12

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        vector = [0.0] * cls.dimension
        for token in text.casefold().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[digest[0] % cls.dimension] += 1.0
        if not any(vector):
            vector[0] = 1.0
        return vector

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]


def _manifest() -> BuildManifest:
    internal = SourceRecord(
        source_id="mini",
        source_type="git_repository",
        uri="D:/repos/Mini-Nanobot",
        commit_sha="a" * 40,
        metadata={"corpus": "internal"},
    )
    official = SourceRecord(
        source_id="mcp",
        source_type="official_web",
        uri="https://modelcontextprotocol.io/specification/2025-06-18/server/tools",
        version="2025-06-18",
        metadata={"corpus": "official"},
    )
    documents = [
        DocumentRecord("mini", "docs/adr/0001.md", "为什么使用手写循环", doc_id="doc_design"),
        DocumentRecord("mini", "mini_nanobot/checkpoint.py", "class CheckpointStore: pass", doc_id="doc_code"),
        DocumentRecord("mini", "tests/test_checkpoint.py", "def test_checkpoint_store(): pass", doc_id="doc_test"),
        DocumentRecord(
            "mcp",
            "https://modelcontextprotocol.io/specification/2025-06-18/server/tools",
            "MCP tools have an inputSchema.",
            doc_id="doc_official",
        ),
    ]
    chunks = [
        ChunkRecord(record.source_id, record.doc_id, 0, record.content, chunk_id=f"chunk_{index}")
        for index, record in enumerate(documents)
    ]
    return BuildManifest(
        build_id="build_test",
        created_at="2026-07-14T00:00:00+00:00",
        sources=[internal, official],
        documents=documents,
        chunks=chunks,
    )


def test_engineering_index_physically_partitions_and_hybrid_searches(tmp_path: Path):
    manager = _FakeEmbeddingManager()
    index = EngineeringIndex.build(
        _manifest(), tmp_path / "index", embedding_manager=manager
    )

    partitions = {(spec.corpus, spec.authority) for spec in index.specs}
    assert partitions == {
        ("internal", "design"),
        ("internal", "code"),
        ("internal", "test"),
        ("official", "official"),
    }
    assert all((tmp_path / "index" / spec.directory / "index.faiss").is_file() for spec in index.specs)

    loaded = EngineeringIndex.load(tmp_path / "index", embedding_manager=manager)
    results = loaded.federated.search(
        "CheckpointStore", corpora="internal", authorities="code", top_k=3
    )
    assert results
    assert results[0].corpus == "internal"
    assert results[0].authority == "code"
    assert "CheckpointStore" in results[0].content
    assert set(results[0].metadata["rrf_retrievers"]) == {"dense", "bm25"}

    (tmp_path / "index" / index.specs[0].directory / "documents.json").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        EngineeringIndex.load(tmp_path / "index", embedding_manager=manager)


def test_git_commit_records_are_history_not_current_implementation():
    source = SourceRecord(
        source_id="mini",
        source_type="git_repository",
        uri="D:/repos/Mini-Nanobot",
        metadata={"corpus": "internal"},
    )
    document = DocumentRecord(
        "mini",
        ".git-history/abc.md",
        "changed checkpoint behavior",
        doc_id="doc_history",
        metadata={"record_kind": "git_commit"},
    )
    chunk = ChunkRecord(
        "mini",
        "doc_history",
        0,
        document.content,
        chunk_id="chunk_history",
        metadata={"record_kind": "git_commit"},
    )
    from rag_core.engineering import infer_partition

    assert infer_partition(chunk, document, source) == ("internal", "history")


def _result(
    content: str,
    source: str,
    *,
    corpus: str,
    authority: str,
    metadata: dict | None = None,
) -> EngineeringSearchResult:
    return EngineeringSearchResult(
        content=content,
        source=source,
        corpus=corpus,
        authority=authority,
        metadata={"document_id": source, **(metadata or {})},
    )


class _FakeLiveCode:
    def __init__(self, root: Path, *, enabled: bool = True):
        self.repo_root = root
        self.enabled = enabled
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 10):
        self.queries.append(query)
        if not self.enabled or "checkpointstore" not in query.casefold():
            return []
        return [
            EngineeringSearchResult(
                content="class CheckpointStore:",
                source=str(self.repo_root / "mini_nanobot/checkpoint.py"),
                corpus="internal",
                authority="code",
                line_start=7,
                line_end=7,
                retriever="live_code_fake",
                metadata={
                    "document_id": "live-checkpoint",
                    "relative_path": "mini_nanobot/checkpoint.py",
                    "live_verification": True,
                },
            )
        ]


def _service(tmp_path: Path, *, live_enabled: bool = True) -> EngineeringRAGService:
    internal_code = BM25Retriever(
        [
            _result(
                "CheckpointStore saves a snapshot",
                "mini_nanobot/checkpoint.py",
                corpus="internal",
                authority="code",
                metadata={"symbol": "CheckpointStore"},
            )
        ]
    )
    internal_design = BM25Retriever(
        [
            _result(
                "选择 SQLite snapshot 是为了简单的本地恢复",
                "docs/adr/0002.md",
                corpus="internal",
                authority="design",
            )
        ]
    )
    official = BM25Retriever(
        [
            _result(
                "The MCP tools capability uses inputSchema.",
                "https://modelcontextprotocol.io/specification/2025-06-18/server/tools",
                corpus="official",
                authority="official",
                metadata={
                    "scope_aliases": ["MCP", "Model Context Protocol"],
                    "covered_topics": [
                        "tools", "tools/list", "tools/call", "inputSchema", "isError"
                    ],
                },
            )
        ]
    )
    federated = FederatedRetriever()
    federated.add_partition("internal", "code", internal_code)
    federated.add_partition("internal", "design", internal_design)
    federated.add_partition("official", "official", official)
    return EngineeringRAGService(
        federated,
        live_code=_FakeLiveCode(tmp_path, enabled=live_enabled),
    )


def test_implementation_requires_and_reports_live_source_verification(tmp_path: Path):
    service = _service(tmp_path)
    outcome = service.retrieve("当前 CheckpointStore 在哪里实现？", top_k=3)

    assert outcome.sufficient_evidence is True
    assert outcome.live_verification_attempted is True
    assert "CheckpointStore" in outcome.live_verification_terms
    assert any(result.metadata["evidence_role"] == "current_implementation" for result in outcome.results)
    live_citation = next(citation for citation in outcome.citations if citation.live_verified)
    assert live_citation.line_start == 7
    assert live_citation.corpus == "internal"


def test_service_binds_live_ast_result_to_git_revision_and_dirty_state(tmp_path: Path):
    repo = tmp_path / "mini"
    package = repo / "mini_nanobot"
    package.mkdir(parents=True)
    source = package / "agent.py"
    source.write_text(
        "class QueryEngine:\n"
        "    async def submit_message(self, message: str):\n"
        "        return message\n",
        encoding="utf-8",
    )
    for command in (
        ["git", "init"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "Test"],
        ["git", "add", "."],
        ["git", "commit", "-m", "initial"],
    ):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    # The verifier must report the real current worktree, including edits that
    # have not yet been committed or included in the static RAG manifest.
    source.write_text(source.read_text(encoding="utf-8") + "\n# local edit\n", encoding="utf-8")

    indexed = BM25Retriever(
        [
            _result(
                "QueryEngine.submit_message handles one turn",
                "mini_nanobot/agent.py",
                corpus="internal",
                authority="code",
                metadata={"symbol": "QueryEngine.submit_message"},
            )
        ]
    )
    federated = FederatedRetriever({("internal", "code"): indexed})
    service = EngineeringRAGService(
        federated,
        live_code=LiveCodeRetriever(repo, prefer_rg=False, corpus="internal"),
        live_ast=LiveASTRetriever(repo),
        live_git=LiveGitVerifier(repo),
    )

    outcome = service.retrieve(
        "当前 QueryEngine.submit_message 在哪里实现？", top_k=4
    )

    assert outcome.sufficient_evidence is True
    assert outcome.live_revision["dirty"] is True
    assert len(outcome.live_revision["commit_sha"]) == 40
    ast_result = next(result for result in outcome.results if result.symbol == "QueryEngine.submit_message")
    assert ast_result.retriever == "rrf"
    assert "live_ast" in ast_result.metadata["rrf_retrievers"]
    citation = next(item for item in outcome.citations if item.symbol == "QueryEngine.submit_message")
    assert citation.revision == outcome.live_revision["commit_sha"]
    assert citation.dirty is True


def test_missing_qualified_symbol_is_not_certified_by_parent_class(tmp_path: Path):
    repo = tmp_path / "mini"
    repo.mkdir()
    (repo / "agent.py").write_text(
        "class QueryEngine:\n"
        "    async def submit_message(self, message: str):\n"
        "        return message\n",
        encoding="utf-8",
    )
    indexed = BM25Retriever(
        [
            _result(
                "class QueryEngine with submit_message",
                "agent.py",
                corpus="internal",
                authority="code",
                metadata={"symbol": "QueryEngine"},
            )
        ]
    )
    service = EngineeringRAGService(
        FederatedRetriever({("internal", "code"): indexed}),
        live_code=LiveCodeRetriever(repo, prefer_rg=False),
        live_ast=LiveASTRetriever(repo),
    )

    outcome = service.retrieve(
        "当前 QueryEngine.definitely_missing_method 在哪里实现？", top_k=3
    )

    assert outcome.live_verification_attempted is True
    assert outcome.live_verification_terms == (
        "QueryEngine.definitely_missing_method",
    )
    assert outcome.sufficient_evidence is False
    assert not any(citation.live_verified for citation in outcome.citations)


def test_official_evidence_never_proves_current_implementation(tmp_path: Path):
    service = _service(tmp_path, live_enabled=False)
    answer = service.answer("当前 CheckpointStore 在哪里实现？")

    assert answer.refused is True
    assert "没有获得实时源码核验结果" in answer.answer
    assert all(citation.evidence_role != "external_normative" for citation in answer.citations)


def test_comparison_requires_both_live_and_official_evidence(tmp_path: Path):
    service = _service(tmp_path)
    answer = service.answer("对比当前 CheckpointStore 与官方 MCP tools 规范", top_k=4)

    assert answer.refused is False
    roles = {citation.evidence_role for citation in answer.citations}
    assert "current_implementation" in roles
    assert "external_normative" in roles

    without_live = _service(tmp_path, live_enabled=False).answer(
        "对比当前 CheckpointStore 与官方 MCP tools 规范", top_k=4
    )
    assert without_live.refused is True
    assert "实时内部实现证据" in without_live.answer


def test_official_answer_is_explicitly_not_implementation_proof(tmp_path: Path):
    answer = _service(tmp_path).answer("根据官方 MCP tools 规范，inputSchema 是什么？")
    assert answer.refused is False
    assert "不能单独证明 Mini-Nanobot 的当前实现" in answer.answer
    assert {citation.evidence_role for citation in answer.citations} == {"external_normative"}


def test_sufficiency_guard_rejects_uncatalogued_or_ungrounded_topics(tmp_path: Path):
    service = _service(tmp_path)

    unknown_official = service.retrieve(
        "根据 Kubernetes 官方规范，PodSecurity admission 如何配置？"
    )
    assert unknown_official.intent.value == "out_of_scope"
    assert unknown_official.sufficient_evidence is False

    missing_mcp_subtopic = service.retrieve(
        "根据 MCP 官方规范，resources/list 如何分页？"
    )
    assert missing_mcp_subtopic.intent.value == "official"
    assert missing_mcp_subtopic.sufficient_evidence is False
    assert any("subtopic" in warning for warning in missing_mcp_subtopic.warnings)

    invented_design = service.retrieve("为什么项目选择 CQRS 架构？")
    assert invented_design.intent.value == "design"
    assert invented_design.sufficient_evidence is False
    assert any("CQRS" in warning for warning in invented_design.warnings)

    unknown_framework = service.retrieve("什么是someunknownframework？")
    assert unknown_framework.intent.value == "design"
    assert unknown_framework.sufficient_evidence is False

    guarded, warnings, reason = EvidenceSufficiencyGuard().check(
        "什么是someunknownframework？",
        unknown_framework.intent,
        [
            _result(
                "选择 SQLite snapshot 是为了简单的本地恢复",
                "docs/adr/0002.md",
                corpus="internal",
                authority="design",
                metadata={"evidence_role": "internal_design"},
            )
        ],
    )
    assert guarded is False
    assert reason == "unsupported_anchor"
    assert any("someunknownframework" in warning for warning in warnings)


@pytest.mark.parametrize("query", ["什么是langchain？", "什么是LangChain？"])
def test_langchain_queries_use_official_scope_and_refuse_without_source(
    tmp_path: Path, query: str
):
    service = _service(tmp_path)

    retrieval = service.retrieve(query)
    answer = service.answer(query)

    assert retrieval.intent.value == "official"
    assert retrieval.sufficient_evidence is False
    assert retrieval.refusal_reason == "missing_official_evidence"
    assert answer.refused is True
    assert answer.citations == []


def test_fastapi_health_retrieve_answer_and_optional_token(tmp_path: Path):
    app = create_app(_service(tmp_path), token="secret-token")
    client = TestClient(app)

    assert client.get("/health").status_code == 401
    headers = {"Authorization": "Bearer secret-token"}
    health = client.get("/health", headers=headers)
    assert health.status_code == 200
    assert health.json()["live_code_enabled"] is True
    assert health.json()["answer_provider"] == "deterministic"
    assert health.json()["answer_model"] is None
    assert health.json()["model_generation_enabled"] is False

    retrieved = client.post(
        "/retrieve",
        json={"query": "当前 CheckpointStore 在哪里实现？", "top_k": 3},
        headers=headers,
    )
    assert retrieved.status_code == 200
    assert retrieved.json()["sufficient_evidence"] is True
    assert retrieved.json()["citations"][0]["citation_id"] == "E1"

    assert client.post(
        "/retrieve",
        json={"query": "x", "top_k": 1, "repo_path": "D:/private"},
        headers=headers,
    ).status_code == 422
    assert client.post(
        "/retrieve",
        json={"query": "x", "top_k": True},
        headers=headers,
    ).status_code == 422

    answered = client.post(
        "/answer",
        json={"query": "当前 CheckpointStore 在哪里实现？"},
        headers={"X-RAG-Token": "secret-token"},
    )
    assert answered.status_code == 200
    assert answered.json()["refused"] is False
    assert answered.json()["generation_mode"] == "deterministic"
    assert answered.json()["generation_provider"] == "deterministic"


def test_fastapi_retrieve_never_calls_model_and_answer_failure_stays_available(
    tmp_path: Path,
):
    calls = 0

    def unavailable_model(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        raise TimeoutError("provider detail must remain private")

    service = _service(tmp_path)
    service.answerer = GroundedAnswerer(unavailable_model, provider="deepseek")
    client = TestClient(create_app(service))

    retrieved = client.post(
        "/retrieve",
        json={"query": "当前 CheckpointStore 在哪里实现？"},
    )
    assert retrieved.status_code == 200
    assert calls == 0

    answered = client.post(
        "/answer",
        json={"query": "当前 CheckpointStore 在哪里实现？"},
    )
    assert answered.status_code == 200
    assert calls == 1
    payload = answered.json()
    assert payload["generation_mode"] == "deterministic_fallback"
    assert payload["generation_provider"] == "deepseek"
    assert "model_generation_failed_fallback_used" in payload["warnings"]
    assert "provider detail" not in str(payload)


def test_answer_endpoint_performs_retrieval_without_a_prior_retrieve_request(
    tmp_path: Path,
):
    service = _service(tmp_path)
    original_retrieve = service.retrieve
    retrieval_calls: list[tuple[str, int]] = []

    def tracked_retrieve(query: str, *, top_k: int = 5):
        retrieval_calls.append((query, top_k))
        return original_retrieve(query, top_k=top_k)

    service.retrieve = tracked_retrieve  # type: ignore[method-assign]
    client = TestClient(create_app(service))
    query = "当前 CheckpointStore 在哪里实现？"

    answered = client.post("/answer", json={"query": query, "top_k": 3})

    assert answered.status_code == 200
    assert retrieval_calls == [(query, 3)]
    assert answered.json()["citations"]


def test_frontend_is_static_same_origin_and_does_not_load_service():
    calls = 0

    def unavailable_service():
        nonlocal calls
        calls += 1
        raise AssertionError("the frontend must not initialize the retrieval service")

    app = create_app(service_factory=unavailable_service, token="do-not-embed")
    client = TestClient(app)

    page = client.get("/")
    assert page.status_code == 200
    assert "工程知识检索台" in page.text
    assert "do-not-embed" not in page.text
    assert "http://" not in page.text
    assert "https://" not in page.text
    assert "仅检索证据" in page.text
    assert "检索并生成回答" in page.text
    assert "会自动执行检索，无需先点" in page.text
    assert 'id="metric-answer-provider"' in page.text
    assert 'id="status-popover"' in page.text
    assert 'id="about-popover"' in page.text
    assert 'id="token-settings"' in page.text
    assert 'id="evidence-drawer"' in page.text
    assert 'class="answer-content"' in page.text
    assert 'id="app-sidebar"' in page.text
    assert 'id="nav-toggle"' in page.text
    assert 'id="nav-overlay"' in page.text
    assert 'data-workspace-mode="answer"' in page.text
    assert 'data-workspace-mode="retrieve"' in page.text
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["x-frame-options"] == "DENY"

    stylesheet = client.get("/assets/app.css")
    script = client.get("/assets/app.js")
    favicon = client.get("/assets/favicon.svg")
    assert stylesheet.status_code == 200
    assert script.status_code == 200
    assert favicon.status_code == 200
    assert "localStorage" not in script.text
    assert "sessionStorage" not in script.text
    for unsafe_dom_api in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "new Function",
    ):
        assert unsafe_dom_api not in script.text
    assert "citation-ref" in script.text
    assert "allowedCitationIds.has(citationId)" in script.text
    assert 'requestJson(`/${activeRequest.mode}`' in script.text
    assert 'mode === "answer" ? "检索并生成回答" : "仅检索证据"' in script.text
    assert "会自动检索并生成回答，无需先点" in script.text
    assert 'toggleAttribute("inert", hidden)' in script.text
    assert 'setAttribute("aria-hidden", "true")' in script.text
    assert 'setAttribute("aria-current", "page")' in script.text
    assert 'new DOMException("请求超时", "TimeoutError")' in script.text
    assert 'new DOMException("请求已取消", "AbortError")' in script.text
    assert "prefers-reduced-motion: reduce" in script.text
    assert "body.nav-open .app-sidebar" in stylesheet.text
    assert "@media (max-width: 900px)" in stylesheet.text
    assert "@media (prefers-reduced-motion: reduce)" in stylesheet.text
    assert calls == 0


def test_public_retrieval_dto_does_not_expose_internal_metadata(tmp_path: Path):
    outcome = _service(tmp_path).retrieve("当前 CheckpointStore 在哪里实现？")
    outcome.results[0].metadata.update(
        {
            "repository_path": "D:/private/Mini-Nanobot",
            "repository_uri": "https://user:secret@example.test/repo.git",
            "response_headers": {"set-cookie": "secret"},
        }
    )

    payload = outcome.to_dict()
    serialized = str(payload)
    assert "D:/private" not in serialized
    assert "user:secret" not in serialized
    assert "set-cookie" not in serialized
    live = next(item for item in payload["results"] if item["live_verified"])
    assert not Path(live["source"]).is_absolute()
    assert payload["schema_version"] == "engineering-retrieval/v1"


def test_index_health_reports_manifest_freshness(tmp_path: Path):
    manager = _FakeEmbeddingManager()
    manifest = _manifest()
    index_root = tmp_path / "index"
    EngineeringIndex.build(manifest, index_root, embedding_manager=manager)
    manifest_path = tmp_path / "manifest.json"
    manifest.write(manifest_path)

    service = EngineeringRAGService.from_index(
        index_root, embedding_manager=manager, manifest_path=manifest_path
    )
    fresh = service.health()
    assert fresh["status"] == "ok"
    assert fresh["index_fresh"] is True

    payload = manifest.to_dict()
    payload["build_id"] = "build_newer"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    stale = service.health()
    assert stale["status"] == "stale_index"
    assert stale["index_fresh"] is False


@pytest.mark.skipif(os.name != "nt", reason="Windows process-probe regression")
def test_windows_build_lock_probe_never_terminates_live_process(tmp_path: Path):
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _process_is_running(child.pid) is True
        time.sleep(0.05)
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_build_lock_uses_os_lock_and_rejects_competing_builder(tmp_path: Path):
    lock = tmp_path / ".index.build.lock"
    owner = _acquire_build_lock(lock)
    try:
        with pytest.raises(FileExistsError):
            _acquire_build_lock(lock)
        assert lock.exists()
    finally:
        _release_build_lock(owner)
    # Persistent lock files avoid close/unlink races. The OS lock itself is
    # released automatically even if a builder process crashes.
    assert lock.exists()
    next_owner = _acquire_build_lock(lock)
    _release_build_lock(next_owner)


def test_engineering_index_rejects_checksum_tampering(tmp_path: Path):
    manager = _FakeEmbeddingManager()
    root = tmp_path / "index"
    EngineeringIndex.build(_manifest(), root, embedding_manager=manager)
    catalog = json.loads((root / "partitions.json").read_text(encoding="utf-8"))
    partition = root / catalog["partitions"][0]["directory"]
    documents = partition / "documents.json"
    documents.write_text(documents.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        EngineeringIndex.load(root, embedding_manager=manager)


def test_engineering_index_refuses_empty_manifest(tmp_path: Path):
    empty = BuildManifest(
        build_id="build_empty",
        created_at="2026-07-14T00:00:00+00:00",
        sources=[],
        documents=[],
        chunks=[],
    )
    with pytest.raises(ValueError, match="empty engineering index"):
        EngineeringIndex.build(
            empty, tmp_path / "index", embedding_manager=_FakeEmbeddingManager()
        )
    assert (tmp_path / ".index.build.lock").exists()


def test_engineering_index_load_rejects_empty_catalog(tmp_path: Path):
    manager = _FakeEmbeddingManager()
    root = tmp_path / "index"
    EngineeringIndex.build(_manifest(), root, embedding_manager=manager)
    catalog_path = root / "partitions.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["partitions"] = []
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="no partitions"):
        EngineeringIndex.load(root, embedding_manager=manager)


def test_recovery_quarantines_legacy_backup_after_committed_root(tmp_path: Path):
    manager = _FakeEmbeddingManager()
    root = tmp_path / "index"
    EngineeringIndex.build(_manifest(), root, embedding_manager=manager)
    backup = tmp_path / ".index.backup-legacy"
    backup.mkdir()
    (backup / "partitions.json").write_text(
        json.dumps({"schema_version": 2, "partitions": []}), encoding="utf-8"
    )

    _recover_interrupted_publish(root)

    EngineeringIndex.load(root, embedding_manager=manager)
    assert not backup.exists()
    assert len(list(tmp_path.glob(".index.quarantine-*"))) == 1


def test_recovery_quarantines_legacy_backup_when_root_is_missing(tmp_path: Path):
    root = tmp_path / "index"
    backup = tmp_path / ".index.backup-legacy"
    backup.mkdir()
    (backup / "partitions.json").write_text(
        json.dumps({"schema_version": 2, "partitions": []}), encoding="utf-8"
    )

    _recover_interrupted_publish(root)

    assert not root.exists()
    assert not backup.exists()
    assert len(list(tmp_path.glob(".index.quarantine-*"))) == 1
