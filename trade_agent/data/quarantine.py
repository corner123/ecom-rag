"""Validated quarantine records and diagnostic redaction."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator, model_validator


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

    @field_validator("source_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("quarantine source_path must be safe and relative")
        return value

    @field_validator("diagnostic")
    @classmethod
    def safe_diagnostic(cls, value: str) -> str:
        if re.search(r"(?i)(https?://[^/@\s]+@|[?&][^=&#\s]+=([^\[]|$)|\b(?:cookie|session|token|signature|password)\s*[:=]\s*(?!\[REDACTED\]))", value):
            raise ValueError("quarantine diagnostic contains unsafe data")
        return value

    def model_copy(self, *, update: dict | None = None, deep: bool = False) -> "QuarantineRecord":
        if update is None:
            return super().model_copy(deep=deep)
        value = self.model_dump(mode="python")
        value.update(update)
        return type(self).model_validate(value)

    @model_validator(mode="after")
    def diagnostic_is_canonical(self) -> "QuarantineRecord":
        if self.diagnostic != sanitize_diagnostic(self.diagnostic):
            raise ValueError("quarantine diagnostic must be sanitized")
        return self


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
