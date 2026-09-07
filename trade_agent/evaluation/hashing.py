"""Deterministic SHA-256 helpers for immutable evaluation artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from pydantic import BaseModel


def hash_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(model: BaseModel) -> str:
    """Hash a Pydantic contract using stable JSON object-key ordering."""
    if not isinstance(model, BaseModel):
        raise TypeError("canonical_hash requires a Pydantic BaseModel")
    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()
