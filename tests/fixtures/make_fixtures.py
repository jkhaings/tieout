"""One-time fixture generator: downloads real SEC data and trims it for tests.

Run manually. Never invoked by `make test` or CI (CLAUDE.md rule 7: tests
are hermetic, no network) -- it exists purely to produce the committed
`companyfacts_*.json` / `submissions_*.json` / `company_tickers.json` files
that the hermetic tests read instead. Uses `app.edgar.client.EdgarClient`
itself, both because that avoids duplicating fetch/cache/allowlist logic and
because running it is a real end-to-end smoke test of that client.

Re-run only if `app.edgar.tags.CANONICAL` gains a tag with no data in the
current fixtures, or the model needs to reach further back in fiscal years.
Run as a module (a plain file path won't put the repo root on `sys.path`,
so `app` won't import):

    uv run python -m tests.fixtures.make_fixtures
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from app.edgar.client import EdgarClient
from app.edgar.config import EdgarSettings
from app.edgar.tags import CANONICAL

FIXTURES_DIR = Path(__file__).parent
TICKERS = ("AAPL", "MSFT")
MIN_PERIOD_END = date(2018, 1, 1)

# Every tag any canonical fallback chain can reach -- trimming to this set
# is what keeps fixtures small while still covering everything the code
# under test actually reads.
_WANTED_TAGS = {tag for spec in CANONICAL for tag in spec.tags}


def _trim_companyfacts(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only tags in `_WANTED_TAGS`, 10-K/10-K-A, recent periods -- every unit.

    Every unit, not just `"USD"`: EPS is filed under `"USD/shares"` and
    share counts under `"shares"`. Restricting to `"USD"` here once made
    those two concepts silently disappear from the fixtures even though
    `app.model.facts` reads them correctly -- caught by this project's own
    tests, not by inspection.
    """
    facts = raw.get("facts", {}).get("us-gaap", {})
    trimmed_tags: dict[str, Any] = {}
    for tag, tag_data in facts.items():
        if tag not in _WANTED_TAGS:
            continue
        trimmed_units: dict[str, Any] = {}
        for unit, entries in tag_data.get("units", {}).items():
            kept = [
                entry
                for entry in entries
                if entry.get("form") in ("10-K", "10-K/A")
                and date.fromisoformat(entry["end"]) >= MIN_PERIOD_END
            ]
            if kept:
                trimmed_units[unit] = kept
        if trimmed_units:
            trimmed_tags[tag] = {"units": trimmed_units}
    return {
        "cik": raw.get("cik"),
        "entityName": raw.get("entityName"),
        "facts": {"us-gaap": trimmed_tags},
    }


def _trim_submissions(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only 10-K/10-K-A rows of the filing history."""
    recent = raw["filings"]["recent"]
    keep_indices = [i for i, form in enumerate(recent["form"]) if form in ("10-K", "10-K/A")]
    trimmed_recent = {key: [values[i] for i in keep_indices] for key, values in recent.items()}
    return {
        "cik": raw.get("cik"),
        "name": raw.get("name"),
        "tickers": raw.get("tickers"),
        "filings": {"recent": trimmed_recent},
    }


def main() -> None:
    """Fetch, trim, and write fixtures for every ticker in `TICKERS`."""
    settings = EdgarSettings(sec_user_agent="tieout-fixtures/0.1 (contact: you@example.com)")
    with EdgarClient(settings=settings) as client:
        tickers_map = client.company_tickers()
        trimmed_map = {
            key: entry
            for key, entry in tickers_map.items()
            if str(entry.get("ticker", "")).upper() in TICKERS
        }
        (FIXTURES_DIR / "company_tickers.json").write_text(json.dumps(trimmed_map, indent=2) + "\n")
        print(f"wrote company_tickers.json ({len(trimmed_map)} entries)")

        for ticker in TICKERS:
            cik = client.resolve_cik(ticker)
            facts = client.company_facts(cik)
            submissions = client.submissions(cik)

            facts_path = FIXTURES_DIR / f"companyfacts_{ticker}.json"
            facts_path.write_text(json.dumps(_trim_companyfacts(facts), indent=2) + "\n")

            submissions_path = FIXTURES_DIR / f"submissions_{ticker}.json"
            submissions_path.write_text(json.dumps(_trim_submissions(submissions), indent=2) + "\n")

            print(f"wrote fixtures for {ticker} (CIK {cik}):")
            print(f"  {facts_path.name}, {submissions_path.name}")


if __name__ == "__main__":
    main()
