"""Database URLs with explicit least-privilege roles."""
from __future__ import annotations

import os
from urllib.parse import quote_plus


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or value.lower().startswith(("replace-", "change-", "example-")):
        raise ValueError(f"{name} must be a non-placeholder environment value")
    return value


def database_url_from_environment(*, role: str) -> str:
    if role not in {"migration", "query"}:
        raise ValueError("role must be migration or query")
    prefix = f"MYSQL__{role.upper()}"
    host = os.environ.get("MYSQL__HOST", "mysql")
    port = os.environ.get("MYSQL__PORT", "3306")
    database = os.environ.get("MYSQL__DATABASE", "foreign_trade_db")
    user = _required(f"{prefix}_USER")
    password = _required(f"{prefix}_PASSWORD")
    return f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{database}?charset=utf8mb4"
