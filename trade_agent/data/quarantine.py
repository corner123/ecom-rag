from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    error_code: str
    source_path: str
    parser: str
    diagnostic: str


def sanitize_diagnostic(value: object, limit: int = 240) -> str:
    return " ".join(str(value).replace("\n", " ").split())[:limit]
