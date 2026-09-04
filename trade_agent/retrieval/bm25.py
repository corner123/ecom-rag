"""Persistent exact-aware BM25 retrieval over immutable chunk builds."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictStr, ValidationError
from rank_bm25 import BM25Okapi

from trade_agent.data.manifest import canonical_json
from trade_agent.retrieval.models import SparseHit
from trade_agent.retrieval.tokenizer import TOKENIZER_VERSION, TradeTokenizer
from trade_agent.schemas.source import ChunkRecord


ARTIFACT_SCHEMA_VERSION = "trade-bm25-artifact-v1"
_BUILD_ID = re.compile(r"build_[0-9a-f]{32}")
_CHECKSUM = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_FILE_SUFFIX = ".tmp"


class BM25ArtifactPayload(BaseModel):
    """The durable subset required to reconstruct an index exactly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ARTIFACT_SCHEMA_VERSION
    build_id: StrictStr
    tokenizer_version: str = TOKENIZER_VERSION
    k1: StrictFloat
    b: StrictFloat
    ordered_chunk_ids: tuple[StrictStr, ...] = Field(min_length=1)
    tokenized_corpus: tuple[tuple[StrictStr, ...], ...]
    records: tuple[ChunkRecord, ...]

    @property
    def checksum(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode("utf-8")).hexdigest()


class _ArtifactEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ARTIFACT_SCHEMA_VERSION
    checksum: StrictStr
    payload: BM25ArtifactPayload


class BM25Index:
    """Immutable BM25 index for one chunk build."""

    def __init__(
        self,
        *,
        build_id: str,
        ordered_chunks: tuple[ChunkRecord, ...],
        tokenized_corpus: tuple[tuple[str, ...], ...],
        k1: float,
        b: float,
    ) -> None:
        if not _BUILD_ID.fullmatch(build_id):
            raise ValueError("build_id must be build_ followed by 32 lowercase hex characters")
        if len(ordered_chunks) == 0 or len(ordered_chunks) != len(tokenized_corpus):
            raise ValueError("ordered chunks and tokenized corpus must be nonempty and equal length")
        self.build_id = build_id
        self.tokenizer_version = TOKENIZER_VERSION
        self.k1 = k1
        self.b = b
        self._ordered_chunks = ordered_chunks
        self._tokenized_corpus = tokenized_corpus
        self._chunk_positions = {
            chunk.metadata.chunk_id: index for index, chunk in enumerate(ordered_chunks)
        }
        self._index = BM25Okapi([list(tokens) for tokens in tokenized_corpus], k1=k1, b=b)

    @classmethod
    def build(
        cls,
        chunks: Sequence[ChunkRecord],
        *,
        build_id: str,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        if isinstance(chunks, (str, bytes)):
            raise TypeError("chunks must be a sequence of ChunkRecord values")
        ordered = tuple(chunks)
        chunk_ids = [chunk.metadata.chunk_id for chunk in ordered]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunks contain duplicate chunk IDs")
        tokenized = tuple(tuple(TradeTokenizer.tokenize(chunk.content)) for chunk in ordered)
        return cls(
            build_id=build_id,
            ordered_chunks=ordered,
            tokenized_corpus=tokenized,
            k1=k1,
            b=b,
        )

    @property
    def chunk_count(self) -> int:
        return len(self._ordered_chunks)

    @property
    def records(self) -> tuple[ChunkRecord, ...]:
        """Return detached records for verification against a frozen build."""
        return tuple(record.model_copy(deep=True) for record in self._ordered_chunks)

    def validate_records(self, records: Sequence[ChunkRecord]) -> None:
        """Prove persisted IDs, payloads and cached tokens came from these records."""
        expected = tuple(records)
        if self._ordered_chunks != expected:
            raise ValueError("BM25 records do not exactly match expected records")
        retokenized = tuple(tuple(TradeTokenizer.tokenize(record.content)) for record in expected)
        if self._tokenized_corpus != retokenized:
            raise ValueError("BM25 tokenized corpus does not match record content")

    def search(
        self,
        query: str,
        *,
        top_k: int,
        allowed_chunk_ids: set[str] | None = None,
    ) -> list[SparseHit]:
        if type(top_k) is not int or not 1 <= top_k <= 512:
            raise ValueError("top_k must be an integer from 1 through 512")
        tokens = TradeTokenizer.tokenize(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        allowed = allowed_chunk_ids
        if allowed is not None and type(allowed) is not set:
            raise TypeError("allowed_chunk_ids must be a set or None")
        scored_positions: list[tuple[float, int]] = []
        for position, score_value in enumerate(scores):
            chunk = self._ordered_chunks[position]
            if allowed is not None and chunk.metadata.chunk_id not in allowed:
                continue
            score = float(score_value)
            if score <= 0.0:
                continue
            scored_positions.append((score, position))
        scored_positions.sort(key=lambda item: (-item[0], item[1]))
        return [
            SparseHit(
                chunk_id=self._ordered_chunks[position].metadata.chunk_id,
                record=self._ordered_chunks[position],
                score=score,
                rank=rank,
                build_id=self.build_id,
                tokenizer_version=self.tokenizer_version,
            )
            for rank, (score, position) in enumerate(scored_positions[:top_k], start=1)
        ]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = BM25ArtifactPayload(
            build_id=self.build_id,
            tokenizer_version=self.tokenizer_version,
            k1=self.k1,
            b=self.b,
            ordered_chunk_ids=tuple(chunk.metadata.chunk_id for chunk in self._ordered_chunks),
            tokenized_corpus=self._tokenized_corpus,
            records=self._ordered_chunks,
        )
        envelope = _ArtifactEnvelope(checksum=payload.checksum, payload=payload)
        serialized = json.dumps(
            envelope.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f"{destination.name}{_ARTIFACT_FILE_SUFFIX}",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
            envelope = _ArtifactEnvelope.model_validate(raw)
        except (OSError, ValueError, json.JSONDecodeError, ValidationError) as error:
            raise ValueError(f"invalid BM25 artifact: {type(error).__name__}") from error
        if envelope.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported BM25 artifact schema version")
        payload = envelope.payload
        if payload.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported BM25 payload schema version")
        if not _CHECKSUM.fullmatch(envelope.checksum):
            raise ValueError("invalid BM25 artifact checksum")
        if envelope.checksum != payload.checksum:
            raise ValueError("BM25 artifact checksum mismatch")
        if payload.tokenizer_version != TOKENIZER_VERSION:
            raise ValueError("unsupported BM25 tokenizer version")
        records_by_id = {
            record.metadata.chunk_id: record
            for record in payload.records
        }
        if len(records_by_id) != len(payload.records):
            raise ValueError("BM25 artifact contains duplicate records")
        try:
            ordered = tuple(records_by_id[chunk_id] for chunk_id in payload.ordered_chunk_ids)
        except KeyError as error:
            raise ValueError("BM25 artifact chunk IDs do not match records") from error
        return cls(
            build_id=payload.build_id,
            ordered_chunks=ordered,
            tokenized_corpus=payload.tokenized_corpus,
            k1=payload.k1,
            b=payload.b,
        )
