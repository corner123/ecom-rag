"""Validated quarantine records and diagnostic redaction."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator


class QuarantineRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    error_code: StrictStr
    source_path: StrictStr
    parser: StrictStr
    diagnostic: StrictStr

    @field_validator("error_code", "source_path", "parser", "diagnostic")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


def sanitize_diagnostic(value: object, limit: int = 240, *, source_path: str | None = None) -> str:
    if type(limit) is not int or limit <= 0:
        raise ValueError("diagnostic limit must be a positive integer")
    text = " ".join(str(value).replace("\n", " ").split())
    if source_path:
        variants = {source_path, str(Path(source_path).expanduser().resolve(strict=False))}
        for path in sorted(variants, key=len, reverse=True):
            if path:
                text = text.replace(path, "[SOURCE_PATH]")
    text = re.sub(r"(?i)\bbearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
    text = re.sub(
        r"(?i)([\"']?(?:password|passwd|secret|token|api[-_]?key|access[-_]?key|authorization|cookie|session(?:[-_]?id)?|signature|sig)[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s&,;}]+)",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"(https?://)[^/@\s]+@", r"\1[REDACTED]@", text)
    text = re.sub(r"([?&][^=&#\s]+)=([^&#\s]*)", r"\1=[REDACTED]", text)
    text = text[:limit].strip()
    return text or "[REDACTED]"
