"""Shared fixtures: trimmed real SEC data for AAPL and MSFT (see fixtures/make_fixtures.py)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.edgar.client import EdgarClient
from app.model.builder import build_statements
from app.schemas import StatementSet
from app.settings import AppSettings, EdgarSettings

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def make_test_app_settings(tmp_path: Path, **overrides: Any) -> AppSettings:
    """Build an `AppSettings` immune to whatever the developer's real `.env` happens to set.

    `AppSettings` is `pydantic-settings`-backed: any field a test doesn't
    pass explicitly still falls through to the real process environment /
    `.env` (ordinary `pydantic-settings` behavior, not a bug in that class
    -- see its docstring). That means a test that only overrides
    `runs_dir`/`run_log_path` would, the moment a real `ANTHROPIC_API_KEY`
    exists in `.env` for local narrate development, silently start making
    live, billed Anthropic API calls -- exactly the CLAUDE.md rule 7
    violation this project treats as a hard requirement. Every test that
    builds `AppSettings` should go through this helper rather than
    constructing it directly, so `anthropic_api_key` and both Langfuse keys
    are always pinned to empty unless a test deliberately overrides them.
    """
    defaults: dict[str, Any] = {
        "runs_dir": tmp_path / "runs",
        "run_log_path": tmp_path / "runs.db",
        "anthropic_api_key": "",
        "langfuse_public_key": "",
        "langfuse_secret_key": "",
    }
    defaults.update(overrides)
    return AppSettings(**defaults)


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / name).read_text())  # type: ignore[no-any-return]


@pytest.fixture
def aapl_facts() -> dict[str, Any]:
    return _load("companyfacts_AAPL.json")


@pytest.fixture
def aapl_submissions() -> dict[str, Any]:
    return _load("submissions_AAPL.json")


@pytest.fixture
def msft_facts() -> dict[str, Any]:
    return _load("companyfacts_MSFT.json")


@pytest.fixture
def msft_submissions() -> dict[str, Any]:
    return _load("submissions_MSFT.json")


@pytest.fixture
def fixture_company_tickers() -> dict[str, Any]:
    return _load("company_tickers.json")


@pytest.fixture
def aapl_statements(aapl_facts: dict[str, Any], aapl_submissions: dict[str, Any]) -> StatementSet:
    return build_statements(aapl_facts, aapl_submissions, "AAPL")


@pytest.fixture
def msft_statements(msft_facts: dict[str, Any], msft_submissions: dict[str, Any]) -> StatementSet:
    return build_statements(msft_facts, msft_submissions, "MSFT")


# --- Mocked EdgarClient fixtures ---------------------------------------------
# Shared by tests/agent and tests/api (both drive the pipeline/API end to end
# against real AAPL data with zero network, per CLAUDE.md rule 7).


@pytest.fixture
def aapl_10k_excerpt_html() -> str:
    """Load the real (trimmed) AAPL 10-K HTML excerpt fixture as text."""
    return (FIXTURES_DIR / "aapl_10k_excerpt.html").read_text(encoding="utf-8")


def _make_edgar_handler(
    *,
    facts: dict[str, Any],
    submissions: dict[str, Any],
    company_tickers: dict[str, Any],
    filing_html: str,
    filing_html_status: int = 200,
) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "company_tickers.json" in url:
            return httpx.Response(200, json=company_tickers)
        if "companyfacts" in url:
            return httpx.Response(200, json=facts)
        if "submissions" in url:
            return httpx.Response(200, json=submissions)
        if "Archives/edgar/data" in url:
            if filing_html_status != 200:
                return httpx.Response(filing_html_status)
            return httpx.Response(200, text=filing_html)
        return httpx.Response(404)

    return handler


@pytest.fixture
def aapl_edgar_client(
    tmp_path: Path,
    aapl_facts: dict[str, Any],
    aapl_submissions: dict[str, Any],
    fixture_company_tickers: dict[str, Any],
    aapl_10k_excerpt_html: str,
) -> Iterator[EdgarClient]:
    """An `EdgarClient` whose HTTP calls are all served from real AAPL fixtures."""
    settings = EdgarSettings(
        sec_user_agent="tieout-tests/0.1 (contact: tests@example.com)",
        cache_dir=tmp_path / "cache",
    )
    handler = _make_edgar_handler(
        facts=aapl_facts,
        submissions=aapl_submissions,
        company_tickers=fixture_company_tickers,
        filing_html=aapl_10k_excerpt_html,
    )
    client = EdgarClient(settings=settings, transport=httpx.MockTransport(handler))
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def aapl_edgar_client_no_filing_text(
    tmp_path: Path,
    aapl_facts: dict[str, Any],
    aapl_submissions: dict[str, Any],
    fixture_company_tickers: dict[str, Any],
) -> Iterator[EdgarClient]:
    """An `EdgarClient` whose filing-HTML request always 404s (numbers still fetch fine)."""
    settings = EdgarSettings(
        sec_user_agent="tieout-tests/0.1 (contact: tests@example.com)",
        cache_dir=tmp_path / "cache",
    )
    handler = _make_edgar_handler(
        facts=aapl_facts,
        submissions=aapl_submissions,
        company_tickers=fixture_company_tickers,
        filing_html="",
        filing_html_status=404,
    )
    client = EdgarClient(settings=settings, transport=httpx.MockTransport(handler))
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def unresolvable_edgar_client(tmp_path: Path) -> Iterator[EdgarClient]:
    """An `EdgarClient` for which every request 404s (simulates an unknown ticker)."""
    settings = EdgarSettings(
        sec_user_agent="tieout-tests/0.1 (contact: tests@example.com)",
        cache_dir=tmp_path / "cache",
    )
    client = EdgarClient(
        settings=settings, transport=httpx.MockTransport(lambda request: httpx.Response(404))
    )
    try:
        yield client
    finally:
        client.close()
