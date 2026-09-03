from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from trade_agent.schemas.source import (
    ChunkMetadata,
    ChunkRecord,
    content_sha256,
    FactType,
    FileType,
    SourceLocator,
    SourceType,
)


def _chunk(chunk_id: str, content: str, **overrides) -> ChunkRecord:
    metadata = ChunkMetadata(
        chunk_id=chunk_id,
        chunk_index=0,
        document_id=f"document_{chunk_id[-8:]}",
        entity_id="entity_harbor_01",
        company_name="Harbor CN Imports 01",
        normalized_name="harbor cn imports 01",
        country_code="CN",
        region="Asia",
        hs_code="850440",
        product_name="USB-C charger",
        sku="SKU-GAN65W",
        file_type=FileType.GENERATED_PROFILE,
        source_type=SourceType.CUSTOMS_PROFILE,
        source_weight=0.8,
        fact_type=FactType.TRADE_ACTIVITY,
        publish_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ingested_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        source_url="https://customs.synthetic.example/profile",
        canonical_url="https://customs.synthetic.example/profile",
        source_locator=SourceLocator(profile="monthly_company_hs"),
        content_hash=content_sha256(content),
        parent_document_hash=content_sha256("parent"),
        language="en",
        is_synthetic=True,
        **overrides,
    )
    return ChunkRecord(content=content, metadata=metadata)


@pytest.fixture
def chunks() -> tuple[ChunkRecord, ...]:
    return (
        _chunk("chunk_match", "HS 850440 HS 850440 Harbor CN Imports 01 imported USB-C charger"),
        _chunk("chunk_related", "HS 850440 charger demand increased"),
        _chunk("chunk_other", "HS 730890 steel structures were exported"),
    )


@pytest.fixture
def index(chunks):
    from trade_agent.retrieval.bm25 import BM25Index

    return BM25Index.build(chunks, build_id="build_0123456789abcdef0123456789abcdef")


@pytest.mark.parametrize(
    "token",
    ["850440", "ABC-TRADING-LLC", "C37700", "SKU-GAN65W"],
)
def test_tokenizer_preserves_trade_identifiers(token: str) -> None:
    from trade_agent.retrieval.tokenizer import TradeTokenizer

    tokens = TradeTokenizer.tokenize(f"查询 {token} 最近采购")
    assert token.casefold() in tokens


def test_tokenizer_is_deterministic_and_adds_cjk_bigrams() -> None:
    from trade_agent.retrieval.tokenizer import TradeTokenizer

    first = TradeTokenizer.tokenize("客户查询采购增长")
    second = TradeTokenizer.tokenize("客户查询采购增长")
    assert first == second
    assert "采购" in first
    assert "增长" in first


def test_bm25_prefers_exact_hs_code_and_records_trace(index) -> None:
    hits = index.search("HS 850440", top_k=3)
    assert [hit.chunk_id for hit in hits] == ["chunk_match", "chunk_related", "chunk_other"]
    assert hits[0].metadata.hs_code == "850440"
    assert hits[0].rank == 1
    assert hits[0].build_id == index.build_id
    assert hits[0].tokenizer_version == "trade-bm25-tokenizer-v1"


def test_bm25_applies_typed_allowed_chunk_ids_before_top_k(index) -> None:
    hits = index.search("HS 850440 charger", top_k=2, allowed_chunk_ids={"chunk_related"})
    assert [hit.chunk_id for hit in hits] == ["chunk_related"]


def test_bm25_build_rejects_duplicate_chunk_ids(chunks) -> None:
    from trade_agent.retrieval.bm25 import BM25Index

    with pytest.raises(ValueError, match="duplicate"):
        BM25Index.build((*chunks, chunks[0]), build_id="build_0123456789abcdef0123456789abcdef")


def test_bm25_artifact_roundtrip_is_deterministic_and_tamper_evident(
    index,
    tmp_path: Path,
) -> None:
    from trade_agent.retrieval.bm25 import BM25Index

    path = tmp_path / "bm25.json"
    index.save(path)
    first_bytes = path.read_bytes()
    index.save(path)
    assert path.read_bytes() == first_bytes

    loaded = BM25Index.load(path)
    assert loaded.build_id == index.build_id
    assert loaded.tokenizer_version == index.tokenizer_version
    assert [hit.chunk_id for hit in loaded.search("HS 850440", top_k=2)] == [
        hit.chunk_id for hit in index.search("HS 850440", top_k=2)
    ]

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["payload"]["ordered_chunk_ids"][0] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        BM25Index.load(path)


def test_bm25_artifact_rejects_unknown_schema_and_build_id(index, tmp_path: Path) -> None:
    from trade_agent.retrieval.bm25 import BM25Index

    path = tmp_path / "bm25.json"
    index.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["payload"]["schema_version"] = "other"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        BM25Index.load(path)
