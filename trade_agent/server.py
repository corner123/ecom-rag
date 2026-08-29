"""Small health-aware HTTP process used by the local Compose API service."""
from __future__ import annotations
import json
import os
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable

_DEPENDENCIES = {"mysql": 3306, "milvus": 19530, "redis": 6379}

def _probe(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False

def readiness_payload(probe: Callable[[str, int], bool] = _probe) -> dict[str, object]:
    dependencies = {name: probe(os.getenv(f"{name.upper()}__HOST", name), port) for name, port in _DEPENDENCIES.items()}
    return {"api": True, "dependencies": dependencies, "ready": all(dependencies.values())}

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/health", "/ready"}:
            self.send_error(404)
            return
        payload = {"api": True} if self.path == "/health" else readiness_payload()
        body = json.dumps(payload).encode()
        self.send_response(200 if payload.get("ready", True) else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return

def main() -> None:
    HTTPServer(("0.0.0.0", int(os.getenv("API_PORT", "8000"))), _Handler).serve_forever()

if __name__ == "__main__":
    main()
