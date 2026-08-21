"""Resolve the independently versioned repository used for live verification."""

from __future__ import annotations

import os
from pathlib import Path


TARGET_REPOSITORY_ENV = "KNOWLEDGE_TARGET_REPO"
LEGACY_TARGET_REPOSITORY_ENV = "MINI_NANOBOT_REPO"
TARGET_SOURCE_ID_ENV = "ENGINEERING_TARGET_SOURCE_ID"
DEFAULT_TARGET_SOURCE_ID = "mini_nanobot"


def resolve_target_repository(
    target_repo: str | Path | None = None,
    *,
    legacy_repo: str | Path | None = None,
) -> str | Path | None:
    """Resolve the live-code root while retaining the legacy configuration."""

    if target_repo is not None:
        return target_repo
    configured = os.getenv(TARGET_REPOSITORY_ENV)
    if configured:
        return configured
    if legacy_repo is not None:
        return legacy_repo
    return os.getenv(LEGACY_TARGET_REPOSITORY_ENV)


def resolve_target_source_id(source_id: str | None = None) -> str:
    """Resolve the manifest source id representing the live target repo."""

    value = source_id or os.getenv(TARGET_SOURCE_ID_ENV) or DEFAULT_TARGET_SOURCE_ID
    normalized = value.strip()
    if not normalized:
        raise ValueError("target source id must not be empty")
    return normalized
