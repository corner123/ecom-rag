"""Typed models for lexical retrieval and persisted BM25 artifacts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

from trade_agent.schemas.source import ChunkRecord


class SparseHit(BaseModel):
    """One BM25 candidate with the minimum trace required for fusion."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    chunk_id: StrictStr
    record: ChunkRecord
    score: StrictFloat
    rank: StrictInt = Field(ge=1)
    build_id: StrictStr
    tokenizer_version: StrictStr

    @property
    def metadata(self) -> object:
        return self.record.metadata
