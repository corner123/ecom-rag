"""FastAPI application exposing only validated, non-streaming v1 results."""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
import re
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status

from trade_agent.api.dependencies import (
    ApiRuntime,
    RuntimeUnavailableError,
    production_runtime_lifespan,
)
from trade_agent.api.models import (
    ErrorResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
    ResumeRequest,
    RetrieveRequest,
    RetrievalResponse,
    RunId,
)
from trade_agent.evidence.models import Evidence


RuntimeLifespan = Callable[[], Any]


def _safe_error(code: str, status_code: int) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code})


def _runtime(request: Request) -> ApiRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise _safe_error("runtime_unavailable", status.HTTP_503_SERVICE_UNAVAILABLE)
    return runtime


def _validated_result(value: object, model: type[Any]) -> Any:
    if type(value) is not model:
        raise _safe_error("invalid_runtime_contract", status.HTTP_502_BAD_GATEWAY)
    try:
        return model.model_validate(value.model_dump(mode="python"))
    except Exception:
        raise _safe_error("invalid_runtime_contract", status.HTTP_502_BAD_GATEWAY) from None


def create_app(*, runtime_lifespan: RuntimeLifespan = production_runtime_lifespan) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        async with runtime_lifespan() as runtime:
            application.state.runtime = runtime
            try:
                yield
            finally:
                application.state.runtime = None

    application = FastAPI(
        title="Trade Intelligence Agent",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "alive"}

    @application.get(
        "/ready",
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
    )
    async def ready(runtime: ApiRuntime = Depends(_runtime)):
        try:
            result = _validated_result(await runtime.readiness(), ReadinessResponse)
        except RuntimeUnavailableError as error:
            raise _safe_error(error.code, status.HTTP_503_SERVICE_UNAVAILABLE) from None
        if not result.ready:
            # Returning a Response would bypass response-model validation. This
            # exception deliberately carries the already validated object.
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=503, content=result.model_dump(mode="json"))
        return result

    @application.post(
        "/v1/query",
        response_model=QueryResponse,
        responses={502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def query(request: QueryRequest, runtime: ApiRuntime = Depends(_runtime)) -> QueryResponse:
        try:
            return _validated_result(await runtime.query(request), QueryResponse)
        except RuntimeUnavailableError as error:
            raise _safe_error(error.code, status.HTTP_503_SERVICE_UNAVAILABLE) from None

    @application.post(
        "/v1/retrieve",
        response_model=RetrievalResponse,
        responses={502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def retrieve(
        request: RetrieveRequest, runtime: ApiRuntime = Depends(_runtime)
    ) -> RetrievalResponse:
        try:
            return _validated_result(await runtime.retrieve(request), RetrievalResponse)
        except RuntimeUnavailableError as error:
            raise _safe_error(error.code, status.HTTP_503_SERVICE_UNAVAILABLE) from None

    @application.get(
        "/v1/runs/{run_id}",
        response_model=QueryResponse,
        responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def get_run(run_id: RunId, runtime: ApiRuntime = Depends(_runtime)) -> QueryResponse:
        try:
            result = await runtime.get_run(run_id)
        except RuntimeUnavailableError as error:
            raise _safe_error(error.code, status.HTTP_503_SERVICE_UNAVAILABLE) from None
        if result is None:
            raise _safe_error("run_not_found", status.HTTP_404_NOT_FOUND)
        return _validated_result(result, QueryResponse)

    @application.post(
        "/v1/runs/{run_id}/resume",
        response_model=QueryResponse,
        responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def resume_run(
        run_id: RunId,
        _body: ResumeRequest,
        runtime: ApiRuntime = Depends(_runtime),
    ) -> QueryResponse:
        try:
            result = await runtime.resume(run_id)
        except RuntimeUnavailableError as error:
            raise _safe_error(error.code, status.HTTP_503_SERVICE_UNAVAILABLE) from None
        if result is None:
            raise _safe_error("run_not_found", status.HTTP_404_NOT_FOUND)
        return _validated_result(result, QueryResponse)

    @application.get(
        "/v1/evidence/{evidence_id}",
        response_model=Evidence,
        responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def get_evidence(
        evidence_id: str, runtime: ApiRuntime = Depends(_runtime)
    ) -> Evidence:
        if re.fullmatch(r"(?:sql|rag)_[0-9a-f]{64}", evidence_id) is None:
            raise _safe_error("invalid_evidence_id", status.HTTP_422_UNPROCESSABLE_CONTENT)
        try:
            result = await runtime.get_evidence(evidence_id)
        except RuntimeUnavailableError as error:
            raise _safe_error(error.code, status.HTTP_503_SERVICE_UNAVAILABLE) from None
        if result is None:
            raise _safe_error("evidence_not_found", status.HTTP_404_NOT_FOUND)
        return _validated_result(result, Evidence)

    return application


app = create_app()
