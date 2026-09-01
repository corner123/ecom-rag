from __future__ import annotations

import json

import numpy as np

from trade_agent.index.embeddings import BgeEmbeddingManager


class FakeEncoder:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 1024), dtype=np.float32)


def test_embedding_smoke_returns_safe_machine_readable_contract_summary() -> None:
    from scripts.smoke_embeddings import run_embedding_smoke

    summary = run_embedding_smoke(
        BgeEmbeddingManager(test_encoder=FakeEncoder(), test_mode=True)
    )

    encoded = json.dumps(summary, ensure_ascii=False)
    assert summary["documents"] == {"shape": [2, 1024], "finite": True, "normalized": True}
    assert summary["query"] == {"shape": [1024], "finite": True, "normalized": True}
    assert summary["contract"]["resolved_revision"] == "5617a9f61b028005a4858fdac845db406aefb181"
    assert "Users" not in encoded
    assert "model_cache" not in encoded


def test_embedding_smoke_reads_only_model_runtime_environment(monkeypatch) -> None:
    from scripts.smoke_embeddings import model_settings_from_environment

    monkeypatch.setenv("MODELS__EMBEDDING_BATCH_SIZE", "7")
    monkeypatch.setenv("MODELS__EMBEDDING_CACHE_DIR", "/safe-cache")
    monkeypatch.setenv("MODELS__EMBEDDING_OFFLINE", "true")

    models = model_settings_from_environment()

    assert models.embedding_batch_size == 7
    assert models.embedding_cache_dir == "/safe-cache"
    assert models.embedding_offline is True
