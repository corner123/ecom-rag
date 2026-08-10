from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_core.engineering.index import HybridPartitionRetriever
from rag_core.retrieval.engineering import (
    BM25Retriever,
    EngineeringSearchResult,
    FederatedRetriever,
    LiveCodeRetriever,
    SourceIntent,
    SourceIntentRouter,
    rrf_fusion,
)


def _result(
    content: str,
    source: str,
    *,
    score: float = 0.0,
    corpus: str = "internal",
    authority: str = "code",
    retriever: str = "",
) -> EngineeringSearchResult:
    return EngineeringSearchResult(
        content=content,
        source=source,
        score=score,
        corpus=corpus,
        authority=authority,
        retriever=retriever,
    )


def test_bm25_ranks_code_identifiers_and_supports_filters():
    documents = [
        _result(
            "QueryEngine.submit_message restores the SQLite checkpoint",
            "core/query_engine.py",
        ),
        _result(
            "MCP client initialization is defined by the official protocol",
            "mcp/tools.md",
            corpus="official",
            authority="official",
        ),
        _result("上下文压缩会保留最近的工具结果", "context/compressor.py"),
    ]
    retriever = BM25Retriever(documents)

    ranked = retriever.search("Where does submit_message load a checkpoint?", top_k=3)
    assert ranked[0].source == "core/query_engine.py"
    assert ranked[0].score > 0
    assert ranked[0].retriever == "bm25"
    assert documents[0].score == 0.0  # inputs are not mutated

    official = retriever.search("official protocol", corpus="official")
    assert [result.source for result in official] == ["mcp/tools.md"]


def test_rrf_fuses_and_deduplicates_results():
    shared_a = _result("shared", "a.py", score=0.9, retriever="dense")
    shared_b = _result("shared", "a.py", score=7.0, retriever="bm25")
    dense_only = _result("dense only", "b.py", score=0.8, retriever="dense")
    bm25_only = _result("bm25 only", "c.py", score=6.0, retriever="bm25")

    fused = rrf_fusion(
        [[shared_a, dense_only], [shared_b, bm25_only]], k=60, top_k=3
    )

    assert [item.source for item in fused].count("a.py") == 1
    assert fused[0].source == "a.py"
    assert fused[0].retriever == "rrf"
    assert set(fused[0].metadata["rrf_retrievers"]) == {"dense", "bm25"}


class _StaticRetriever:
    def __init__(self, results):
        self.results = results

    def search(self, query: str, top_k: int = 5):
        return self.results[:top_k]


class _FailingRetriever:
    def search(self, query: str, top_k: int = 5):
        raise RuntimeError("retrieval failed")


@pytest.mark.parametrize(
    ("dense", "bm25"),
    [
        (_FailingRetriever(), _FailingRetriever()),
        (_FailingRetriever(), _StaticRetriever([])),
    ],
)
def test_hybrid_partition_can_fail_closed_even_when_no_result_carries_metadata(
    dense, bm25
):
    retriever = HybridPartitionRetriever(dense, bm25, fail_open=False)

    with pytest.raises(RuntimeError, match="failed closed"):
        retriever.search("query")


def test_federated_retriever_partitions_and_filters():
    internal = _StaticRetriever([_result("implementation", "query.py")])
    official = _StaticRetriever(
        [_result("MCP specification", "https://modelcontextprotocol.io/spec")]
    )
    retriever = FederatedRetriever(
        {
            ("internal", "code"): internal,
            ("official", "official"): official,
        },
        rrf_k=10,
    )

    all_results = retriever.search("MCP implementation", top_k=5)
    assert {(result.corpus, result.authority) for result in all_results} == {
        ("internal", "code"),
        ("official", "official"),
    }

    official_results = retriever.search(
        "MCP", top_k=5, corpora="official", authorities={"official"}
    )
    assert len(official_results) == 1
    assert official_results[0].corpus == "official"
    assert official_results[0].metadata["federated_partition"] == "official/official"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("当前 QueryEngine.submit_message 在哪个文件实现？", SourceIntent.IMPLEMENTATION),
        ("为什么 checkpoint 使用 snapshot 设计？", SourceIntent.DESIGN),
        ("根据 MCP 官方规范，client 初始化包含什么？", SourceIntent.OFFICIAL),
        ("什么是langchain？", SourceIntent.OFFICIAL),
        ("什么是LangChain？", SourceIntent.OFFICIAL),
        ("Langchian是什么", SourceIntent.OFFICIAL),
        ("对比当前 MCPToolAdapter 与官方 MCP Client", SourceIntent.COMPARISON),
        ("今天上海天气如何？", SourceIntent.OUT_OF_SCOPE),
    ],
)
def test_source_intent_router(query, expected):
    router = SourceIntentRouter()
    assert router.classify(query) is expected
    assert router.route(query).intent is expected


def test_query_normalization_only_applies_curated_unambiguous_alias():
    router = SourceIntentRouter()

    normalized, warning = router.normalize_query("Langchian是什么")
    unknown, unknown_warning = router.normalize_query("LangChaine是什么")
    literal, literal_warning = router.normalize_query("`langchian` 在哪里定义？")

    assert normalized == "LangChain是什么"
    assert warning == "query_normalized_langchian_to_langchain"
    assert unknown == "LangChaine是什么"
    assert unknown_warning is None
    assert literal == "`langchian` 在哪里定义？"
    assert literal_warning is None


def test_live_code_python_fallback_is_bounded_and_returns_lines(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "agent.py"
    source.write_text("first line\nclass CheckpointStore:\n    pass\n", encoding="utf-8")
    (repo / ".git").mkdir()
    (repo / ".git" / "secret.txt").write_text("CheckpointStore", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("CheckpointStore", encoding="utf-8")

    retriever = LiveCodeRetriever(repo, prefer_rg=False)
    results = retriever.search("checkpointstore", top_k=10)

    assert len(results) == 1
    assert results[0].source == str(source.resolve())
    assert results[0].line_start == 2
    assert results[0].metadata["relative_path"] == "agent.py"
    assert results[0].retriever == "live_code_python"
    assert str(outside) not in {result.source for result in results}


def test_live_code_rg_uses_literal_argument_vector_without_shell(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "agent.py").write_text("safe", encoding="utf-8")
    retriever = LiveCodeRetriever(repo, prefer_rg=False)
    retriever._rg_path = "rg"
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    suspicious = "--version; Remove-Item C:\\"

    assert retriever.search(suspicious) == []
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["cwd"] == str(repo.resolve())
    assert captured["command"][-3:] == ["--", suspicious, "."]


def test_live_code_rejects_multiline_queries(tmp_path):
    retriever = LiveCodeRetriever(tmp_path, prefer_rg=False)
    with pytest.raises(ValueError):
        retriever.search("first\nsecond")
