"""Small contracts shared by local and distributed dense-vector stores."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DenseVectorStore(Protocol):
    """The query-side contract required by engineering retrieval."""

    def search_dense(
        self,
        query: str,
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[dict[str, Any]]:
        """Return ranked records with text, score and round-tripped metadata."""

    def get_stats(self) -> dict[str, Any]:
        """Return non-secret operational metadata for health reporting."""
