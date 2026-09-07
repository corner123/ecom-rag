"""ASGI product entry point for the strict trade-agent API."""
from __future__ import annotations
import os
from typing import Callable

from trade_agent.api.app import app


_DEPENDENCIES = {"mysql": 3306, "milvus": 19530, "redis": 6379}


def readiness_payload(probe: Callable[[str, int], bool]) -> dict[str, object]:
    """Legacy pure helper retained for callers; product readiness uses contracts."""
    dependencies = {
        name: probe(os.getenv(f"{name.upper()}__HOST", name), port)
        for name, port in _DEPENDENCIES.items()
    }
    return {"api": True, "dependencies": dependencies, "ready": all(dependencies.values())}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "trade_agent.server:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        access_log=False,
    )

if __name__ == "__main__":
    main()
