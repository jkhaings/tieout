"""Shared fixtures: trimmed real SEC data for AAPL and MSFT (see fixtures/make_fixtures.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.model.builder import build_statements
from app.schemas import StatementSet

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
