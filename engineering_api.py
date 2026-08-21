"""FastAPI boundary consumed by a read-only engineering knowledge client."""

from __future__ import annotations

from collections.abc import Callable
import hmac
import os
from pathlib import Path
import threading

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from rag_core.engineering import EngineeringRAGService


WEB_ROOT = Path(__file__).resolve().parent / "web"


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=1_000)
    top_k: int = Field(default=5, ge=1, le=50)


ServiceFactory = Callable[[], EngineeringRAGService]


class _LazyService:
    def __init__(self, factory: ServiceFactory) -> None:
        self.factory = factory
        self._service: EngineeringRAGService | None = None
        self._lock = threading.Lock()

    def get(self) -> EngineeringRAGService:
        if self._service is None:
            with self._lock:
                if self._service is None:
                    self._service = self.factory()
        return self._service


def _default_service() -> EngineeringRAGService:
    index_root = Path(
        os.getenv("ENGINEERING_INDEX_DIR", "data/indexes/ecommerce_demo")
    )
    return EngineeringRAGService.from_index(
        index_root,
        target_repo=os.getenv("KNOWLEDGE_TARGET_REPO"),
        mini_nanobot_repo=os.getenv("MINI_NANOBOT_REPO"),
        manifest_path=os.getenv(
            "ENGINEERING_MANIFEST_PATH", "data/manifests/builds/ecommerce_demo.json"
        ),
    )


def create_app(
    service: EngineeringRAGService | None = None,
    *,
    service_factory: ServiceFactory | None = None,
    token: str | None = None,
) -> FastAPI:
    """Create an API without loading embeddings until the first request."""

    if service is not None and service_factory is not None:
        raise ValueError("provide either service or service_factory, not both")
    if service_factory is not None:
        factory = service_factory
    elif service is not None:
        factory = lambda: service
    else:
        factory = _default_service
    lazy = _LazyService(factory)
    expected_token = (
        os.getenv("RAG_API_TOKEN", "") if token is None else token
    ).strip()

    def authorize(
        authorization: str | None = Header(default=None),
        x_rag_token: str | None = Header(default=None),
    ) -> None:
        if not expected_token:
            return
        supplied = x_rag_token or ""
        if authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
        if not supplied or not hmac.compare_digest(supplied, expected_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid RAG API token",
            )

    app = FastAPI(
        title="E-commerce Engineering Knowledge RAG",
        version="1.0.0",
        description="Read-only retrieval over a synthetic commerce engineering corpus.",
    )

    @app.middleware("http")
    async def frontend_security_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
            )
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
        return response

    app.mount("/assets", StaticFiles(directory=WEB_ROOT / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def frontend() -> FileResponse:
        """Serve the local, read-only engineering RAG workbench."""

        return FileResponse(WEB_ROOT / "index.html", media_type="text/html")

    def resolve_service(_authorized: None = Depends(authorize)) -> EngineeringRAGService:
        try:
            return lazy.get()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="engineering RAG is not ready",
            ) from exc

    @app.get("/health")
    def health(active: EngineeringRAGService = Depends(resolve_service)) -> dict:
        return active.health()

    @app.post("/retrieve")
    def retrieve(
        request: SearchRequest,
        active: EngineeringRAGService = Depends(resolve_service),
    ) -> dict:
        try:
            return active.retrieve(request.query, top_k=request.top_k).to_dict()
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/answer")
    def answer(
        request: SearchRequest,
        active: EngineeringRAGService = Depends(resolve_service),
    ) -> dict:
        try:
            return active.answer(request.query, top_k=request.top_k).to_dict()
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
