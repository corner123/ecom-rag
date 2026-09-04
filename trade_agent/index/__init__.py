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


def __getattr__(name):
    if name in {"TradeIndexBuilder", "TradeIndexBundle", "BundleDescriptor"}:
        from .builder import TradeIndexBuilder, TradeIndexBundle, BundleDescriptor
        return locals()[name]
    raise AttributeError(name)

__all__ += ["TradeIndexBuilder", "TradeIndexBundle", "BundleDescriptor"]
