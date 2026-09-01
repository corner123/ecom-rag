"""Versioned embedding contracts and BGE-M3 vector generation."""

from .contracts import EmbeddingContract
from .embeddings import BGE_M3_DIMENSION, BGE_M3_MODEL, BGE_M3_REVISION, BgeEmbeddingManager

__all__ = [
    "BGE_M3_DIMENSION",
    "BGE_M3_MODEL",
    "BGE_M3_REVISION",
    "BgeEmbeddingManager",
    "EmbeddingContract",
]
