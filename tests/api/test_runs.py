"""End-to-end tests for POST /runs, GET /runs/{id}, and the workbook download."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from app.edgar.client import EdgarClient
from app.main import create_app
from app.obs import RunLog
from tests.conftest import make_test_app_settings


async def _wait_for_terminal_status(
    client: httpx.AsyncClient, run_id: str, *, timeout_s: float = 10.0
) -> dict:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        response = await client.get(f"/runs/{run_id}")
        body = response.json()
        if body["status"] != "running":
            return body
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} never reached a terminal status")


async def test_full_run_lifecycle_start_poll_download(api_client: httpx.AsyncClient) -> None:
    create_response = await api_client.post("/runs", json={"ticker": "aapl"})
    assert create_response.status_code == 202
    run_id = create_response.json()["run_id"]

    status = await _wait_for_terminal_status(api_client, run_id)
    assert status["status"] == "done"
    assert status["tieout_passed"] is True
    assert status["ticker"] == "AAPL"  # normalized by validate_ticker

    download = await api_client.get(f"/runs/{run_id}/model.xlsx")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("application/vnd.openxmlformats")
    assert len(download.content) > 0


async def test_invalid_ticker_is_rejected_with_no_internals_leaked(
    api_client: httpx.AsyncClient,
) -> None:
    response = await api_client.post("/runs", json={"ticker": "not a ticker!!"})
    assert response.status_code == 422
    body = response.json()
    assert "error" in body
    assert "not a ticker" not in body["error"]


async def test_oversized_request_body_is_rejected(api_client: httpx.AsyncClient) -> None:
    """SECURITY.md item 7 ("request size limits"): a large body is capped, not buffered."""
    huge_ticker = "A" * 100_000
    response = await api_client.post(
        "/runs", content=b'{"ticker": "' + huge_ticker.encode() + b'"}'
    )
    assert response.status_code == 413
    assert response.json() == {"error": "request body too large", "run_id": None}


async def test_body_at_content_length_over_cap_is_rejected_without_reading_it(
    api_client: httpx.AsyncClient,
) -> None:
    """A declared Content-Length over the cap is rejected up front from the header alone."""
    response = await api_client.post(
        "/runs",
        content=b"x" * 200_000,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


async def test_unknown_run_id_is_404_everywhere(api_client: httpx.AsyncClient) -> None:
    fake_id = "00000000-0000-0000-0000-000000000000"
    assert (await api_client.get(f"/runs/{fake_id}")).status_code == 404
    assert (await api_client.get(f"/runs/{fake_id}/model.xlsx")).status_code == 404
    assert (await api_client.get(f"/runs/{fake_id}/events")).status_code == 404


async def test_malformed_run_id_is_404_not_a_path_error(api_client: httpx.AsyncClient) -> None:
    """A run id that isn't even a UUID (e.g. a path-traversal attempt) is just not found."""
    response = await api_client.get("/runs/../../etc/passwd")
    assert response.status_code == 404


async def test_download_404s_for_a_run_that_exists_but_has_no_artifact_yet(
    aapl_edgar_client: EdgarClient, tmp_path: Path
) -> None:
    """A known, still-`running` run (no `generate` yet) must 404 on download, not 500."""
    run_id = "11111111-1111-1111-1111-111111111111"
    settings = make_test_app_settings(tmp_path)
    run_log = RunLog(settings.run_log_path)
    run_log.create_run(run_id, "AAPL")  # a real UUID, deliberately never marked done
    app: FastAPI = create_app(settings=settings, edgar_client=aapl_edgar_client, run_log=run_log)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/runs/{run_id}/model.xlsx")
            assert response.status_code == 404
            assert response.json()["error"] == "workbook not found"

            status = await client.get(f"/runs/{run_id}")
            assert status.json()["status"] == "running"
            assert status.json()["tieout_passed"] is None
            assert status.json()["narrate_ok"] is None


async def test_daily_run_cap_is_enforced(aapl_edgar_client: EdgarClient, tmp_path: Path) -> None:
    settings = make_test_app_settings(tmp_path, rate_limit_runs_per_hour=1000, daily_run_cap=1)
    run_log = RunLog(settings.run_log_path)
    app: FastAPI = create_app(settings=settings, edgar_client=aapl_edgar_client, run_log=run_log)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.post("/runs", json={"ticker": "AAPL"})
            assert first.status_code == 202
            await _wait_for_terminal_status(client, first.json()["run_id"])

            second = await client.post("/runs", json={"ticker": "AAPL"})
            assert second.status_code == 429


async def test_daily_run_cap_counts_accepted_runs_immediately_not_just_completed_ones(
    aapl_edgar_client: EdgarClient, tmp_path: Path
) -> None:
    """Regression test for a TOCTOU race in the daily run cap.

    The cap must count a run the moment it's accepted, not only once its
    background task actually starts executing.
    `app.api.routes.create_run` used to rely on `run_pipeline` (gated behind
    `_run_semaphore`) to write the run log row; under a burst of concurrent
    requests, none of them would count toward `count_since` until a
    semaphore slot freed up, so far more than `daily_run_cap` runs could be
    accepted before any of them registered. `create_run` now reserves the
    row synchronously, before scheduling the task -- checked here by
    asserting the count is already correct immediately after acceptance,
    with no wait for either run to actually finish.
    """
    settings = make_test_app_settings(tmp_path, rate_limit_runs_per_hour=1000, daily_run_cap=2)
    run_log = RunLog(settings.run_log_path)
    app: FastAPI = create_app(settings=settings, edgar_client=aapl_edgar_client, run_log=run_log)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.post("/runs", json={"ticker": "AAPL"})
            second = await client.post("/runs", json={"ticker": "MSFT"})
            assert first.status_code == 202
            assert second.status_code == 202

            # Neither run need have finished (or even started); the cap
            # must already reflect both as counted.
            assert run_log.count_since(0.0) == 2

            third = await client.post("/runs", json={"ticker": "AAPL"})
            assert third.status_code == 429


async def test_per_ip_rate_limit_is_enforced(
    aapl_edgar_client: EdgarClient, tmp_path: Path
) -> None:
    settings = make_test_app_settings(tmp_path, rate_limit_runs_per_hour=1, daily_run_cap=1000)
    run_log = RunLog(settings.run_log_path)
    app: FastAPI = create_app(settings=settings, edgar_client=aapl_edgar_client, run_log=run_log)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.post("/runs", json={"ticker": "AAPL"})
            assert first.status_code == 202

            second = await client.post("/runs", json={"ticker": "MSFT"})
            assert second.status_code == 429


@pytest.mark.parametrize("path", ["/", "/runs/x", "/runs/x/events"])
async def test_security_headers_present_on_every_response(
    api_client: httpx.AsyncClient, path: str
) -> None:
    response = await api_client.get(path)
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in response.headers


async def test_cors_rejects_a_foreign_origin(api_client: httpx.AsyncClient) -> None:
    response = await api_client.options(
        "/runs",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "https://evil.example.com" != response.headers.get("access-control-allow-origin")


async def test_cors_allows_the_configured_origin(api_client: httpx.AsyncClient) -> None:
    response = await api_client.options(
        "/runs",
        headers={
            "Origin": "http://allowed.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://allowed.example.com"


async def test_index_page_is_served(api_client: httpx.AsyncClient) -> None:
    response = await api_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "tieout" in response.text.lower()
