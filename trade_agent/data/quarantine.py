from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    error_code: str
    source_path: str
    parser: str
    diagnostic: str


def sanitize_diagnostic(value: object, limit: int = 240) -> str:
    text = " ".join(str(value).replace("\n", " ").split())
    text = re.sub(r"(?i)\b(password|secret|token|api[-_]?key)\s*[:=]\s*[^\s&]+", r"\1=[REDACTED]", text)
    text = re.sub(r"(https?://)[^/@\s]+@", r"\1[REDACTED]@", text)
    text = re.sub(r"([?&](?:token|key|secret|password)=)[^&\s]+", r"\1[REDACTED]", text, flags=re.I)
    return text[:limit]
