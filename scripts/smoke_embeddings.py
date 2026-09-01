"""Run a real, pinned BGE-M3 embedding smoke without exposing local paths."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from trade_agent.config.settings import ModelSettings
from trade_agent.index.embeddings import BgeEmbeddingManager

_MODEL_ENVIRONMENT = {
    "MODELS__EMBEDDING_MODEL": "embedding_model",
    "MODELS__EMBEDDING_REVISION": "embedding_revision",
    "MODELS__EMBEDDING_DEVICE": "embedding_device",
    "MODELS__EMBEDDING_BATCH_SIZE": "embedding_batch_size",
    "MODELS__EMBEDDING_CACHE_DIR": "embedding_cache_dir",
    "MODELS__EMBEDDING_OFFLINE": "embedding_offline",
    "MODELS__RERANKER_MODEL": "reranker_model",
    "MODELS__RERANKER_REVISION": "reranker_revision",
}


def model_settings_from_environment() -> ModelSettings:
    """Read model-only controls without loading database or service secrets."""

    values = {
        field: value
        for environment_name, field in _MODEL_ENVIRONMENT.items()
        if (value := os.environ.get(environment_name)) not in (None, "")
    }
    return ModelSettings.model_validate(values)


def _vector_summary(vectors: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(vectors, axis=-1)
    return {
        "shape": list(vectors.shape),
        "finite": bool(np.isfinite(vectors).all()),
        "normalized": bool(np.allclose(norms, 1.0, rtol=0.0, atol=1e-5)),
    }


def run_embedding_smoke(manager: BgeEmbeddingManager | None = None) -> dict[str, Any]:
    """Embed multilingual trade text and return only machine-safe evidence."""

    if manager is None:
        models = model_settings_from_environment()
        manager = BgeEmbeddingManager(
            model_name=models.embedding_model,
            revision=models.embedding_revision,
            device=models.embedding_device,
            batch_size=models.embedding_batch_size,
            cache_folder=models.embedding_cache_dir,
            local_files_only=models.embedding_offline,
        )
    documents = manager.embed_documents(
        ["HS 850440 chargers exported to Europe", "采购增长客户需要核验海关月度贸易画像"]
    )
    query = manager.embed_query("查询 HS 850440 采购增长客户")
    contract = manager.contract.require_production()
    return {
        "status": "ok",
        "contract": contract.model_dump(mode="json"),
        "snapshot_bytes": manager.snapshot_size_bytes,
        "documents": _vector_summary(documents),
        "query": _vector_summary(query),
    }


def _failure(error_name: str, *, exit_code: int) -> int:
    print(
        json.dumps({"status": "failed", "error": error_name}, sort_keys=True),
        file=sys.stderr,
    )
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        return _failure("UsageError", exit_code=2)

    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
            summary = run_embedding_smoke()
            if summary["contract"]["provider"] != "sentence-transformers":
                raise RuntimeError("real sentence-transformers provider was not used")
            payload = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    except Exception as error:  # The command intentionally emits no exception text or paths.
        return _failure(type(error).__name__, exit_code=1)
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
