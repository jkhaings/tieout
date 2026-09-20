"""Generate the gold tie-out dataset directly from raw SEC companyfacts JSON.

One-time, standalone build-time utility -- like `tests/fixtures/make_fixtures.py`,
this is not part of the app and is not covered by CLAUDE.md rule 7's
hermeticity rule (that rule governs `tests/`; this is a build-time generator,
not a test). Run as a module so the repo root lands on `sys.path`:

    uv run python -m evals.generate_tieout_gold

Why this file exists and what it must never do: the tie-out eval checks
`app.model.facts.FactIndex` / `app.model.builder.build_statements` against a
gold answer key, so that key must not be produced by calling the code under
test. This script therefore reads `tests/fixtures/companyfacts_AAPL.json` and
`tests/fixtures/companyfacts_MSFT.json` directly and reimplements period
selection and value resolution from scratch. The only import from the app is
`app.edgar.tags.CANONICAL` (and its `TagSpec` type) -- a static declaration
of "which XBRL tag(s) mean which canonical line item," i.e. domain mapping,
not fact-selection logic, so reusing it does not undermine the independence
of this derivation. Nothing from `app.model` is imported.

Selection rules reimplemented here (verified against the real fixtures --
AAPL FY2024 revenue and total_assets reproduce exactly, see `_spot_check`):

- Only entries with `form` in `{"10-K", "10-K/A"}` count.
- A "duration" tag's entry counts only if it has a `"start"` field and
  `(end - start).days` falls within 330-400 inclusive (an annual span,
  excluding quarterly and year-to-date facts).
- An "instant" tag's entry counts only if it has no `"start"` field.
- Entries are read from `company_facts["facts"]["us-gaap"][tag]["units"][spec.unit]`
  -- the exact unit named by `TagSpec.unit` ("USD", "USD/shares", or
  "shares"). Using the wrong unit silently produces nothing for that tag.
- Qualifying entries are grouped by `"end"` date. Within a group, the VALUE
  comes from the entry with the latest `"filed"` date (restatements win);
  the FISCAL YEAR LABEL comes from the entry with the *earliest* `"filed"`
  date (the filing that first reported the period as current) -- these can
  differ when a figure is restated in a later filing.
- Each canonical line item resolves its tag fallback chain (`spec.tags`)
  independently *per period*: for a given period end, the first tag in the
  chain that reports it wins; a period missing from `tags[0]` can still be
  filled by `tags[1]`, etc.
- Values are copied as filed: raw dollars (or raw per-share / raw share
  counts for the other two units), never scaled, never negated.
- A cell with no qualifying entry from any tag in the chain is `None`
  (JSON `null`), never a fabricated `0`.

Fiscal-year discovery mirrors `app.model.builder._discover_fiscal_years`
conceptually (not by import): anchor on `total_assets`'s resolved periods,
take the 5 most recent distinct fiscal-year labels by period-end date
descending, then present ascending.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from app.edgar.tags import CANONICAL, TagSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
OUTPUT_PATH = DATASETS_DIR / "tieout_gold.json"

TICKERS: tuple[str, ...] = ("AAPL", "MSFT")
PRESENTED_FISCAL_YEARS = 5
ANNUAL_FORMS = frozenset({"10-K", "10-K/A"})
ANNUAL_MIN_DAYS = 330
ANNUAL_MAX_DAYS = 400
ANCHOR_KEY = "total_assets"
TOLERANCE_USD = 1.0

# (value, fiscal-year label) for one resolved period end.
_PeriodValue = tuple[float, "int | None"]

EXPECTED_FISCAL_YEARS: dict[str, list[int]] = {
    "AAPL": [2021, 2022, 2023, 2024, 2025],
    "MSFT": [2022, 2023, 2024, 2025, 2026],
}


def _load_companyfacts(ticker: str) -> dict[str, Any]:
    """Load and return the raw companyfacts JSON fixture for `ticker`."""
    path = FIXTURES_DIR / f"companyfacts_{ticker}.json"
    with path.open() as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def _is_qualifying(entry: dict[str, Any], kind: str) -> bool:
    """Return whether `entry` is an annual 10-K/10-K-A fact matching `kind`.

    A duration entry qualifies only if it has a `"start"` and its span is
    330-400 days inclusive (annual, excludes quarterly/YTD facts). An
    instant entry qualifies only if it has no `"start"`.
    """
    if entry.get("form") not in ANNUAL_FORMS:
        return False
    start = entry.get("start")
    if kind == "duration":
        if not start:
            return False
        span = (date.fromisoformat(entry["end"]) - date.fromisoformat(start)).days
        return ANNUAL_MIN_DAYS <= span <= ANNUAL_MAX_DAYS
    return not start


def _periods_for_tag(
    company_facts: dict[str, Any], tag: str, kind: str, unit: str
) -> dict[str, _PeriodValue]:
    """Resolve one `tag` (in `unit`) to `{period_end: (value, fiscal_year_label)}`.

    Within each period-end group of qualifying entries, the value is taken
    from the entry with the latest `"filed"` date; the fiscal-year label
    comes from the entry with the earliest `"filed"` date -- these can
    differ when a value is restated in a later filing.
    """
    tag_data = company_facts.get("facts", {}).get("us-gaap", {}).get(tag)
    if not tag_data:
        return {}
    entries: list[dict[str, Any]] = tag_data.get("units", {}).get(unit, [])

    by_period: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        if _is_qualifying(entry, kind):
            by_period.setdefault(entry["end"], []).append(entry)

    result: dict[str, _PeriodValue] = {}
    for period_end, group in by_period.items():
        ordered = sorted(group, key=lambda e: e["filed"])
        earliest, latest = ordered[0], ordered[-1]
        fy = earliest.get("fy")
        result[period_end] = (float(latest["val"]), int(fy) if fy is not None else None)
    return result


def _resolve_spec(company_facts: dict[str, Any], spec: TagSpec) -> dict[str, _PeriodValue]:
    """Resolve `spec`'s tag fallback chain to `{period_end: (value, fiscal_year_label)}`.

    Each period end is satisfied independently by the first tag in
    `spec.tags` that reports it; a period absent from `tags[0]` can still be
    filled by `tags[1]`, and so on.
    """
    result: dict[str, _PeriodValue] = {}
    for tag in spec.tags:
        tag_periods = _periods_for_tag(company_facts, tag, spec.kind, spec.unit)
        for period_end, value_fy in tag_periods.items():
            result.setdefault(period_end, value_fy)
    return result


def _discover_fiscal_years(
    anchor_resolved: dict[str, _PeriodValue], n: int
) -> list[tuple[int, str]]:
    """Return the `n` most recent distinct `(fiscal_year, period_end)` pairs, ascending by year."""
    dated = sorted(
        ((period_end, fy) for period_end, (_, fy) in anchor_resolved.items() if fy is not None),
        key=lambda pair: pair[0],
        reverse=True,
    )
    picked: list[tuple[int, str]] = []
    seen_years: set[int] = set()
    for period_end, fy in dated:
        if fy in seen_years:
            continue
        seen_years.add(fy)
        picked.append((fy, period_end))
        if len(picked) >= n:
            break
    picked.sort(key=lambda pair: pair[0])
    return picked


def _build_ticker_gold(ticker: str) -> dict[str, Any]:
    """Build `{"fiscal_years": [...], "cells": {...}}` for one ticker from its raw fixture."""
    company_facts = _load_companyfacts(ticker)
    anchor_spec = next(spec for spec in CANONICAL if spec.key == ANCHOR_KEY)
    anchor_resolved = _resolve_spec(company_facts, anchor_spec)
    year_periods = _discover_fiscal_years(anchor_resolved, PRESENTED_FISCAL_YEARS)
    fiscal_years = [fy for fy, _ in year_periods]
    period_end_by_fy = dict(year_periods)

    cells: dict[str, dict[str, float | None]] = {}
    for spec in CANONICAL:
        resolved = _resolve_spec(company_facts, spec)
        per_year: dict[str, float | None] = {}
        for fy in fiscal_years:
            period_end = period_end_by_fy[fy]
            value_fy = resolved.get(period_end)
            per_year[str(fy)] = value_fy[0] if value_fy is not None else None
        cells[spec.key] = per_year

    return {"fiscal_years": fiscal_years, "cells": cells}


def _spot_check(gold: dict[str, Any]) -> None:
    """Assert independently-confirmed AAPL FY2024 values hold exactly; raise otherwise.

    These two values were confirmed against the real filed companyfacts
    data before this script was written; running the check on every
    regeneration (not just once by hand) guards against a future edit to
    the selection logic silently breaking it.
    """
    cells = gold["tickers"]["AAPL"]["cells"]
    revenue_2024 = cells["revenue"]["2024"]
    assets_2024 = cells["total_assets"]["2024"]
    if revenue_2024 != 391035000000.0:
        raise AssertionError(f"spot-check failed: AAPL FY2024 revenue = {revenue_2024!r}")
    if assets_2024 != 364980000000.0:
        raise AssertionError(f"spot-check failed: AAPL FY2024 total_assets = {assets_2024!r}")


def _check_expected_fiscal_years(gold: dict[str, Any]) -> None:
    """Print a warning (never raise) if derived fiscal years diverge from the expected lists.

    A mismatch is a signal to double-check period-matching logic, not
    necessarily proof the app under test is wrong -- flagged loudly rather
    than silently picked one way or the other.
    """
    for ticker, expected in EXPECTED_FISCAL_YEARS.items():
        actual = gold["tickers"][ticker]["fiscal_years"]
        if actual != expected:
            print(
                f"WARNING: {ticker} derived fiscal years {actual} != expected {expected} "
                "-- double check period-matching logic before trusting this dataset."
            )


def build_gold() -> dict[str, Any]:
    """Build the full gold dataset dict for every ticker in `TICKERS`."""
    return {
        "generated_by": "evals/generate_tieout_gold.py",
        "source_fixtures": [
            "tests/fixtures/companyfacts_AAPL.json",
            "tests/fixtures/companyfacts_MSFT.json",
        ],
        "tolerance_usd": TOLERANCE_USD,
        "tickers": {ticker: _build_ticker_gold(ticker) for ticker in TICKERS},
    }


def main() -> None:
    """Generate `evals/datasets/tieout_gold.json`, spot-check it, and report coverage."""
    gold = build_gold()
    _spot_check(gold)
    _check_expected_fiscal_years(gold)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(gold, indent=2) + "\n")

    for ticker in TICKERS:
        cells = gold["tickers"][ticker]["cells"]
        total = sum(len(per_year) for per_year in cells.values())
        non_null = sum(
            1 for per_year in cells.values() for value in per_year.values() if value is not None
        )
        print(f"{ticker}: {non_null}/{total} non-null cells")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
