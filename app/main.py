"""Composition root: the FastAPI app factory, lifespan, middleware, and static page.

An app *factory* (`create_app`), not a bare module-level `FastAPI()`, so
tests can build a fresh app per test with injected dependencies (a mocked
`EdgarClient`, a tmp-path `RunLog`, tuned `AppSettings`) -- the same
structural-injection convention the rest of the codebase uses, no
monkeypatching. `app = create_app()` at the bottom is what `make run`'s
`uvicorn app.main:app` actually serves.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import create_run, download_workbook, get_run_status, stream_events
from app.api.registry import RunRegistry
from app.api.schemas import ErrorResponse, RunCreateResponse, RunStatusResponse
from app.api.security import (
    BodySizeLimitMiddleware,
    RequestBodyTooLarge,
    SecurityHeadersMiddleware,
    request_body_too_large_handler,
)
from app.edgar.client import EdgarClient
from app.obs import RunLog, setup_logging
from app.settings import AppSettings, get_app_settings

logger = logging.getLogger(__name__)

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Reshape FastAPI's default `{"detail": ...}` into `{"error": ...}` (SECURITY.md item 10).

    Every `detail=` string raised in `app/api/routes.py` is already a
    short, generic message chosen deliberately -- this only makes the
    response body's shape consistent with `app.api.schemas.ErrorResponse`.
    """
    assert isinstance(exc, StarletteHTTPException)
    body = ErrorResponse(error=exc.detail)
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler: log the real exception, tell the client nothing but "internal error".

    SECURITY.md item 10: "exceptions are logged server-side; clients
    receive generic messages ... never stack traces or config values."
    Reaching this at all means a bug slipped past every narrower handler
    upstream (`app.agent.runner.run_pipeline` and every route already
    catch and reduce their own failures) -- worth a loud server-side log.
    """
    logger.exception("unhandled exception", exc_info=exc)
    body = ErrorResponse(error="internal error")
    return JSONResponse(status_code=500, content=body.model_dump())


def create_app(
    *,
    settings: AppSettings | None = None,
    edgar_client: EdgarClient | None = None,
    run_log: RunLog | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        settings: Platform configuration; defaults to `get_app_settings()`.
        edgar_client: The `EdgarClient` to use for the app's lifetime;
            defaults to a real one built from `EdgarSettings`. Tests pass
            one constructed with a mocked `httpx.MockTransport`.
        run_log: The run log to use; defaults to a real `RunLog` at
            `settings.run_log_path`. Tests pass one backed by `tmp_path`.

    Returns:
        A configured, not-yet-started `FastAPI` app.
    """
    resolved_settings = settings or get_app_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(resolved_settings.app_env)
        app.state.edgar_client = edgar_client or EdgarClient()
        app.state.run_log = run_log or RunLog(resolved_settings.run_log_path)
        app.state.registry = RunRegistry()
        try:
            yield
        finally:
            app.state.edgar_client.close()

    app = FastAPI(title="tieout", lifespan=lifespan)
    app.state.settings = resolved_settings

    # One Limiter per app instance (never a shared module-level singleton),
    # built from *this* instance's settings, so each `create_app()` call --
    # a fresh app per test, or the one real deployment -- gets its own
    # independent rate-limit state and its own configured limit. Applying
    # `.limit(...)` here, at registration time, rather than as a decorator
    # in app/api/routes.py, is what makes the limit value settings-driven
    # at all (see that module's docstring for why the decorator can't read
    # `app.state` itself).
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter

    # Order matters: Starlette applies middleware in reverse of add order,
    # so security headers, CORS, and the body-size cap end up wrapping (and
    # therefore applying to) slowapi's own 429 responses too, not just
    # successful ones. BodySizeLimitMiddleware is added last (so it runs
    # first, outermost) so an oversized body is rejected before any other
    # middleware or the router does any work with it.
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.add_middleware(BodySizeLimitMiddleware)

    # `_rate_limit_exceeded_handler`'s own type is narrower than the
    # `Exception`-typed handler `add_exception_handler` expects (it's
    # correct at runtime -- Starlette dispatches by the registered
    # exception type, `RateLimitExceeded`, not by the handler's signature)
    # -- a stub-only mismatch, not a real one.
    app.add_exception_handler(RateLimitExceeded, cast(Any, _rate_limit_exceeded_handler))
    app.add_exception_handler(RequestBodyTooLarge, request_body_too_large_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)

    rate_limited_create_run = limiter.limit(f"{resolved_settings.rate_limit_runs_per_hour}/hour")(
        create_run
    )
    app.add_api_route(
        "/runs",
        rate_limited_create_run,
        methods=["POST"],
        response_model=RunCreateResponse,
        status_code=202,
    )
    app.add_api_route(
        "/runs/{run_id}", get_run_status, methods=["GET"], response_model=RunStatusResponse
    )
    app.add_api_route("/runs/{run_id}/model.xlsx", download_workbook, methods=["GET"])
    app.add_api_route("/runs/{run_id}/events", stream_events, methods=["GET"])

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        """Serve the single static demo page (no build step, no other assets)."""
        return (_WEB_DIR / "index.html").read_text(encoding="utf-8")

    return app


app = create_app()
