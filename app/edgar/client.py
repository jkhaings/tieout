"""HTTP client for SEC EDGAR: companyfacts, submissions, and 10-K filing text.

Only `data.sec.gov` and `www.sec.gov` are ever contacted (SECURITY.md item 2,
enforced by `_assert_allowed` on every request, not just at URL construction
time). Every successful response is cached to disk under `cache_dir`, keyed
by a hash of the full request URL (SECURITY.md item 5: the URL never
influences the path beyond its hash), so repeat runs are reproducible and
demos work offline after the first fetch (ARCHITECTURE.md). Requests are
rate-limited to `max_requests_per_second` and retried with exponential
backoff + jitter on 429/5xx, per SEC's fair-use policy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Final
from urllib.parse import urlsplit

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from app.settings import EdgarSettings, get_edgar_settings

logger = logging.getLogger(__name__)

ALLOWED_HOSTS: Final[frozenset[str]] = frozenset({"data.sec.gov", "www.sec.gov"})
COMPANY_TICKERS_URL: Final[str] = "https://www.sec.gov/files/company_tickers.json"

_TICKER_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_ACCESSION_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9]{18}$")
_DOCNAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.\-]{1,200}$")


class EdgarError(RuntimeError):
    """Raised for invalid input, disallowed hosts, or unrecoverable EDGAR responses."""


def validate_ticker(raw: str) -> str:
    """Validate and normalize a ticker symbol.

    Strips whitespace and uppercases, then requires a match against
    `^[A-Z][A-Z0-9.\\-]{0,9}$` (SECURITY.md item 1). Raises `EdgarError`
    otherwise -- this is the only path user-supplied ticker text may take
    before being used to build a URL.
    """
    candidate = raw.strip().upper()
    if not _TICKER_RE.match(candidate):
        raise EdgarError(f"invalid ticker: {raw!r}")
    return candidate


def validate_cik(raw: str | int) -> str:
    """Validate a CIK and return it zero-padded to 10 digits.

    Raises `EdgarError` unless `raw` coerces to a non-negative integer
    (SECURITY.md item 1: "CIK values are validated as integers").
    """
    try:
        cik_int = int(raw)
    except (TypeError, ValueError) as exc:
        raise EdgarError(f"invalid CIK: {raw!r}") from exc
    if cik_int < 0:
        raise EdgarError(f"invalid CIK: {raw!r}")
    return f"{cik_int:010d}"


def _assert_allowed(url: str) -> None:
    """Raise `EdgarError` unless `url` is `https` and targets an allowlisted host.

    Called on every outbound request (not only when a URL is first built) so
    that no code path -- present or future -- can reach an unapproved host.
    """
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise EdgarError(f"host not allowlisted: {url!r}")


class _RateLimiter:
    """Monotonic-clock token bucket capping requests to `max_per_second`."""

    def __init__(self, max_per_second: float) -> None:
        """Cap requests to `max_per_second` (must be > 0)."""
        self._min_interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._last_request: float | None = None

    def wait(self) -> None:
        """Block until at least `1/max_per_second` seconds have passed since the last call."""
        with self._lock:
            now = time.monotonic()
            if self._last_request is not None:
                elapsed = now - self._last_request
                remaining = self._min_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            self._last_request = time.monotonic()


def _is_retryable(exc: BaseException) -> bool:
    """True for transport errors and 429/5xx responses; false otherwise."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class EdgarClient:
    """Cached, rate-limited, host-allowlisted HTTP client for SEC EDGAR."""

    def __init__(
        self,
        settings: EdgarSettings | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Build a client. `transport` lets tests inject `httpx.MockTransport`."""
        self._settings = settings or get_edgar_settings()
        self._limiter = _RateLimiter(self._settings.max_requests_per_second)
        self._cache_dir = self._settings.cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(
            timeout=self._settings.request_timeout_s,
            transport=transport,
            follow_redirects=False,
        )

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._client.close()

    def _headers(self) -> dict[str, str]:
        if not self._settings.sec_user_agent:
            raise EdgarError(
                "SEC_USER_AGENT is not configured; refusing to contact SEC without a "
                "declared User-Agent (SEC fair-use policy; see .env.example)"
            )
        return {
            "User-Agent": self._settings.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def _cache_path(self, url: str, suffix: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{digest}{suffix}"

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=30),
        reraise=True,
    )
    def _get(self, url: str) -> httpx.Response:
        _assert_allowed(url)
        self._limiter.wait()
        response = self._client.get(url, headers=self._headers())
        if response.is_redirect:
            raise EdgarError(f"unexpected redirect from {url!r}")
        response.raise_for_status()
        return response

    def _get_json(self, url: str) -> Any:
        cache_path = self._cache_path(url, ".json")
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        data = self._get(url).json()
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def _get_text(self, url: str, suffix: str) -> str:
        cache_path = self._cache_path(url, suffix)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
        text = self._get(url).text
        cache_path.write_text(text, encoding="utf-8")
        return text

    def company_tickers(self) -> dict[str, Any]:
        """Fetch (and cache) SEC's full ticker -> CIK mapping (`company_tickers.json`)."""
        result: dict[str, Any] = self._get_json(COMPANY_TICKERS_URL)
        return result

    def resolve_cik(self, ticker: str) -> str:
        """Resolve a ticker symbol to a zero-padded 10-digit CIK.

        Downloads (and caches) SEC's `company_tickers.json` mapping and
        scans it for a case-insensitive ticker match. Raises `EdgarError`
        for an invalid ticker or one absent from the mapping.
        """
        clean = validate_ticker(ticker)
        mapping = self.company_tickers()
        for entry in mapping.values():
            if str(entry.get("ticker", "")).upper() == clean:
                return validate_cik(entry["cik_str"])
        raise EdgarError(f"ticker not found in SEC company_tickers.json: {clean!r}")

    def company_facts(self, cik: str) -> dict[str, Any]:
        """Fetch the `companyfacts` XBRL JSON for a CIK (validated before use)."""
        cik10 = validate_cik(cik)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
        result: dict[str, Any] = self._get_json(url)
        return result

    def submissions(self, cik: str) -> dict[str, Any]:
        """Fetch the filing-history/submissions JSON for a CIK (validated before use)."""
        cik10 = validate_cik(cik)
        url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
        result: dict[str, Any] = self._get_json(url)
        return result

    def latest_10k(self, cik: str) -> dict[str, str]:
        """Return `{accession_number, primary_document, filing_date}` for the newest 10-K.

        Raises `EdgarError` if the submissions feed lists no 10-K.
        """
        data = self.submissions(cik)
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        for i, form in enumerate(forms):
            if form == "10-K":
                return {
                    "accession_number": recent["accessionNumber"][i],
                    "primary_document": recent["primaryDocument"][i],
                    "filing_date": recent["filingDate"][i],
                }
        raise EdgarError(f"no 10-K found in submissions for CIK {cik}")

    def filing_html(self, cik: str, accession_number: str, primary_document: str) -> str:
        """Fetch a filing's primary HTML document.

        The URL is assembled only from validated, constant-shaped parts: a
        zero-padded CIK, an 18-digit accession number (dashes stripped), and
        a conservative document-name pattern -- never from raw user input
        (SECURITY.md item 1).
        """
        cik10 = validate_cik(cik)
        accession_nodash = accession_number.replace("-", "")
        if not _ACCESSION_RE.match(accession_nodash):
            raise EdgarError(f"invalid accession number: {accession_number!r}")
        if not _DOCNAME_RE.match(primary_document):
            raise EdgarError(f"invalid primary document name: {primary_document!r}")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/"
            f"{accession_nodash}/{primary_document}"
        )
        return self._get_text(url, ".html")
