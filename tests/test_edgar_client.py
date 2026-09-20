"""Hermetic tests for app.edgar.client -- no network; SEC responses are simulated
with httpx.MockTransport throughout (CLAUDE.md rule 7).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.edgar.client import (
    ALLOWED_HOSTS,
    EdgarClient,
    EdgarError,
    _assert_allowed,
    _RateLimiter,
    validate_cik,
    validate_ticker,
)
from app.settings import EdgarSettings

_TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
}


def _settings(tmp_path: Path) -> EdgarSettings:
    return EdgarSettings(
        sec_user_agent="tieout-tests/0.1 (contact: test@example.com)",
        cache_dir=tmp_path,
    )


def _mock_client(tmp_path: Path, handler: Any) -> EdgarClient:
    return EdgarClient(settings=_settings(tmp_path), transport=httpx.MockTransport(handler))


# ---- validate_ticker --------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("aapl", "AAPL"),
        (" msft ", "MSFT"),
        ("BRK.B", "BRK.B"),
        ("brk-b", "BRK-B"),
        ("a", "A"),
    ],
)
def test_validate_ticker_accepts_valid(raw: str, expected: str) -> None:
    assert validate_ticker(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "1AAPL",  # must start with a letter
        "AAAAAAAAAAA",  # 11 chars, over the limit
        "AA PL",
        "../../etc/passwd",
        "$(rm -rf /)",
        "AAPL;ls",
        "AA\nPL",  # embedded control character (a trailing one is legitimately stripped)
        "aapl/../x",
    ],
)
def test_validate_ticker_rejects_invalid(raw: str) -> None:
    with pytest.raises(EdgarError):
        validate_ticker(raw)


# ---- validate_cik ------------------------------------------------------------


def test_validate_cik_zero_pads_string_and_int() -> None:
    assert validate_cik("320193") == "0000320193"
    assert validate_cik(320193) == "0000320193"


@pytest.mark.parametrize("raw", ["-5", "abc", "1.5", None, ""])
def test_validate_cik_rejects_invalid(raw: Any) -> None:
    with pytest.raises(EdgarError):
        validate_cik(raw)


# ---- host allowlist -----------------------------------------------------------


def test_allowed_hosts_are_exactly_the_two_sec_domains() -> None:
    assert ALLOWED_HOSTS == frozenset({"data.sec.gov", "www.sec.gov"})


@pytest.mark.parametrize(
    "url",
    [
        "http://data.sec.gov/x",  # not https
        "https://evil.com/x",
        "https://data.sec.gov.evil.com/x",  # subdomain confusion
        "https://www.sec.gov@evil.com/x",  # userinfo confusion
        "https://sec.gov/x",  # apex domain, not an allowlisted subdomain
        "ftp://data.sec.gov/x",
    ],
)
def test_assert_allowed_rejects(url: str) -> None:
    with pytest.raises(EdgarError):
        _assert_allowed(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        "https://www.sec.gov/files/company_tickers.json",
    ],
)
def test_assert_allowed_accepts(url: str) -> None:
    _assert_allowed(url)  # must not raise


# ---- rate limiter --------------------------------------------------------------


def test_rate_limiter_spaces_out_consecutive_calls() -> None:
    limiter = _RateLimiter(max_per_second=20.0)  # 50ms minimum interval
    limiter.wait()
    start = time.monotonic()
    limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.04  # small slack for scheduling jitter


# ---- EdgarClient over MockTransport --------------------------------------------


def test_resolve_cik_success(tmp_path: Path) -> None:
    seen_agents = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_agents.append(request.headers.get("User-Agent"))
        return httpx.Response(200, json=_TICKERS_JSON)

    with _mock_client(tmp_path, handler) as client:
        assert client.resolve_cik("aapl") == "0000320193"
    assert seen_agents == ["tieout-tests/0.1 (contact: test@example.com)"]


def test_resolve_cik_not_found_raises(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_TICKERS_JSON)

    with _mock_client(tmp_path, handler) as client, pytest.raises(EdgarError):
        client.resolve_cik("ZZZZ")


def test_invalid_ticker_never_reaches_the_network(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_TICKERS_JSON)

    with _mock_client(tmp_path, handler) as client, pytest.raises(EdgarError):
        client.resolve_cik("../../etc/passwd")
    assert calls == []


def test_missing_user_agent_raises_before_any_request(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_TICKERS_JSON)

    settings = EdgarSettings(sec_user_agent="", cache_dir=tmp_path)
    with EdgarClient(settings=settings, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(EdgarError):
            client.resolve_cik("AAPL")
    assert calls == []


def test_response_is_served_from_disk_cache_on_second_call(tmp_path: Path) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json=_TICKERS_JSON)

    with _mock_client(tmp_path, handler) as client:
        client.resolve_cik("AAPL")
        client.resolve_cik("MSFT")  # same underlying URL: must be a cache hit, not a 2nd request
    assert call_count == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_retries_on_429_then_succeeds(tmp_path: Path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json=_TICKERS_JSON)

    with _mock_client(tmp_path, handler) as client:
        assert client.resolve_cik("AAPL") == "0000320193"
    assert attempts == 2


def test_does_not_retry_on_404(tmp_path: Path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404, text="not found")

    with _mock_client(tmp_path, handler) as client, pytest.raises(httpx.HTTPStatusError):
        client.resolve_cik("AAPL")
    assert attempts == 1


def test_redirect_is_rejected_not_followed(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://evil.com/"})

    with _mock_client(tmp_path, handler) as client, pytest.raises(EdgarError):
        client.resolve_cik("AAPL")


def test_company_facts_builds_correct_url_from_validated_cik(tmp_path: Path) -> None:
    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(
            200, json={"cik": 320193, "entityName": "Apple Inc.", "facts": {"us-gaap": {}}}
        )

    with _mock_client(tmp_path, handler) as client:
        data = client.company_facts("320193")
    assert data["entityName"] == "Apple Inc."
    assert seen_urls == ["https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"]


def test_submissions_builds_correct_url(tmp_path: Path) -> None:
    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(
            200, json={"cik": "0000320193", "name": "Apple Inc.", "filings": {"recent": {}}}
        )

    with _mock_client(tmp_path, handler) as client:
        client.submissions("320193")
    assert seen_urls == ["https://data.sec.gov/submissions/CIK0000320193.json"]


def test_latest_10k_picks_the_first_10k_row(tmp_path: Path) -> None:
    payload = {
        "filings": {
            "recent": {
                "form": ["4", "10-K", "8-K"],
                "accessionNumber": ["a", "0000320193-24-000123", "c"],
                "primaryDocument": ["x.xml", "aapl-20240928.htm", "z.htm"],
                "filingDate": ["2024-11-05", "2024-11-01", "2024-10-15"],
            }
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with _mock_client(tmp_path, handler) as client:
        latest = client.latest_10k("320193")
    assert latest == {
        "accession_number": "0000320193-24-000123",
        "primary_document": "aapl-20240928.htm",
        "filing_date": "2024-11-01",
    }


def test_latest_10k_raises_when_none_present(tmp_path: Path) -> None:
    payload = {"filings": {"recent": {"form": ["4", "8-K"]}}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with _mock_client(tmp_path, handler) as client, pytest.raises(EdgarError):
        client.latest_10k("320193")


@pytest.mark.parametrize(
    ("accession", "doc"),
    [
        ("../../etc/passwd", "aapl-20240928.htm"),
        ("0000320193-24-000123", "../../etc/passwd"),
        ("not-an-accession-number", "aapl.htm"),
        ("0000320193-24-000123", "x" * 300),
    ],
)
def test_filing_html_rejects_invalid_parts_before_any_request(
    tmp_path: Path, accession: str, doc: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not reach the network with invalid input")

    with _mock_client(tmp_path, handler) as client, pytest.raises(EdgarError):
        client.filing_html("320193", accession, doc)


def test_filing_html_builds_correct_url(tmp_path: Path) -> None:
    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, text="<html>ok</html>")

    with _mock_client(tmp_path, handler) as client:
        html = client.filing_html("320193", "0000320193-24-000123", "aapl-20240928.htm")
    assert html == "<html>ok</html>"
    assert seen_urls == [
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    ]


def test_company_tickers_and_resolve_cik_share_one_cache_entry(tmp_path: Path) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json=_TICKERS_JSON)

    with _mock_client(tmp_path, handler) as client:
        mapping = client.company_tickers()
        client.resolve_cik("MSFT")
    assert mapping == _TICKERS_JSON
    assert call_count == 1
