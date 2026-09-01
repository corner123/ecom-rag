from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from trade_agent.index.embeddings import BgeEmbeddingManager


class FakeEncoder:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 1024), dtype=np.float32)


def test_public_embedding_smoke_rejects_a_test_encoder(monkeypatch) -> None:
    from scripts.smoke_embeddings import run_embedding_smoke

    monkeypatch.setenv("TEST_EMBEDDING_PROVIDER", "deterministic")
    manager = BgeEmbeddingManager(test_encoder=FakeEncoder(), test_mode=True)

    with pytest.raises(ValueError, match="production"):
        run_embedding_smoke(manager)


def test_embedding_smoke_reads_only_model_runtime_environment(monkeypatch) -> None:
    from scripts.smoke_embeddings import model_settings_from_environment

    monkeypatch.setenv("MODELS__EMBEDDING_BATCH_SIZE", "7")
    monkeypatch.setenv("MODELS__EMBEDDING_CACHE_DIR", "/safe-cache")
    monkeypatch.setenv("MODELS__EMBEDDING_OFFLINE", "true")

    models = model_settings_from_environment()

    assert models.embedding_batch_size == 7
    assert models.embedding_cache_dir == "/safe-cache"
    assert models.embedding_offline is True


def test_smoke_cli_rejects_unknown_arguments_without_echoing_them(monkeypatch, capsys) -> None:
    import scripts.smoke_embeddings as smoke

    def must_not_run():
        raise AssertionError("model smoke must not run for invalid arguments")

    monkeypatch.setattr(smoke, "run_embedding_smoke", must_not_run)
    private_argument = "/Users/private/model-cache?token=secret"

    exit_code = smoke.main([private_argument])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == '{"error": "UsageError", "status": "failed"}\n'
    assert private_argument not in captured.err


def test_smoke_cli_failure_is_one_safe_stderr_json_line(monkeypatch, capsys) -> None:
    import scripts.smoke_embeddings as smoke

    def fail_with_private_output():
        print("/Users/private/model-cache?token=secret")
        print("secret diagnostic", file=sys.stderr)
        raise RuntimeError("/Users/private/model-cache?token=secret")

    monkeypatch.setattr(smoke, "run_embedding_smoke", fail_with_private_output)

    exit_code = smoke.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == '{"error": "RuntimeError", "status": "failed"}\n'


def test_smoke_cli_serialization_failure_is_also_safely_reported(monkeypatch, capsys) -> None:
    import scripts.smoke_embeddings as smoke

    monkeypatch.setattr(
        smoke,
        "run_embedding_smoke",
        lambda: {"contract": {"provider": "sentence-transformers"}, "unsafe": object()},
    )

    exit_code = smoke.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == '{"error": "TypeError", "status": "failed"}\n'


def test_smoke_cli_success_is_one_stdout_json_line(monkeypatch, capsys) -> None:
    import scripts.smoke_embeddings as smoke

    summary = {
        "status": "ok",
        "contract": {"provider": "sentence-transformers"},
        "documents": {"shape": [2, 1024], "finite": True, "normalized": True},
        "query": {"shape": [1024], "finite": True, "normalized": True},
    }

    def succeed_with_noisy_library_output():
        print("library progress")
        print("library warning", file=sys.stderr)
        return summary

    monkeypatch.setattr(smoke, "run_embedding_smoke", succeed_with_noisy_library_output)

    exit_code = smoke.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == summary
    assert captured.out.count("\n") == 1
