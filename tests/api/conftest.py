"""Fixtures for app/api tests: a real ASGI app, hermetic dependencies, no network.

Uses `httpx.AsyncClient` over `httpx.ASGITransport` rather than
`fastapi.testclient.TestClient`: a background task started by `POST /runs`
must keep running while a later request streams its SSE output, and that
requires the test and the app to share one asyncio event loop -- verified
directly against this exact pattern (a task created in one request is
observable, still progressing, from a later request on the same client).
`app.router.lifespan_context(app)` is entered manually so `create_app`'s
lifespan (which sets up `app.state.edgar_client`/`run_log`/`registry`) runs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from app.edgar.client import EdgarClient
from app.main import create_app
from app.obs import RunLog
from app.settings import AppSettings
from tests.conftest import make_test_app_settings


@pytest.fixture
def api_settings(tmp_path: Path) -> AppSettings:
    """Generous limits by default so ordinary tests never trip rate limiting."""
    return make_test_app_settings(
        tmp_path,
        rate_limit_runs_per_hour=1000,
        daily_run_cap=1000,
        cors_origins="http://allowed.example.com",
    )


@pytest.fixture
def api_app(api_settings: AppSettings, aapl_edgar_client: EdgarClient) -> FastAPI:
    """A fully wired app: real routes/middleware, hermetic AAPL-fixture-backed EdgarClient."""
    run_log = RunLog(api_settings.run_log_path)
    return create_app(settings=api_settings, edgar_client=aapl_edgar_client, run_log=run_log)


@pytest.fixture
async def api_client(api_app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """An async client with the app's lifespan running, sharing this test's event loop."""
    async with api_app.router.lifespan_context(api_app):
        transport = httpx.ASGITransport(app=api_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
