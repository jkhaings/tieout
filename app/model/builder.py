"""Build a `StatementSet` for the last five fiscal years from raw EDGAR JSON.

Design note (a deliberate simplification versus the original session plan):
roll-forward checks (cash, retained earnings) need a prior-year balance, so
the plan called for resolving a hidden sixth year internally. That would
mean either leaking a year outside `StatementSet.fiscal_years` into
`LineItem.values` (surprising for any downstream consumer who reasonably
assumes those two match) or inventing a side channel not present in the
frozen `app/schemas.py`. Instead, this module resolves exactly the five
presented fiscal years, and `app.model.verifier` computes roll-forward
checks over the four consecutive pairs *within* those five -- the oldest
presented year simply has no roll-forward check, which is a smaller and
more honest gap than a schema-shaped surprise. Every other check (balance
sheet equation, subtotals) still covers all five years.
"""

from __future__ import annotations

from typing import Any

from app.edgar.client import validate_cik
from app.edgar.tags import CANONICAL
from app.edgar.tags import get as get_spec
from app.model.facts import FactIndex
from app.schemas import LineItem, StatementSet

PRESENTED_FISCAL_YEARS = 5

_ANCHOR_KEY = "total_assets"  # ~universal single-tag concept; anchors year discovery


def _discover_fiscal_years(index: FactIndex, n: int) -> list[tuple[int, str]]:
    """Return the `n` most recent (fiscal_year, period_end) pairs, ascending by year.

    Anchored on `total_assets` (`us-gaap:Assets`), which is filed by every
    10-K balance sheet -- see `app.edgar.tags` for why chained items can't
    play this role safely (a tag's presence can vary by fiscal year).
    """
    resolved = index.resolve_all(get_spec(_ANCHOR_KEY))
    dated = sorted(
        ((period_end, rv) for period_end, rv in resolved.items() if rv.fiscal_year is not None),
        key=lambda pair: pair[0],
        reverse=True,
    )
    picked: list[tuple[int, str]] = []
    seen_years: set[int] = set()
    for period_end, rv in dated:
        fy = rv.fiscal_year
        assert fy is not None  # narrowed by the filter above
        if fy in seen_years:
            continue
        seen_years.add(fy)
        picked.append((fy, period_end))
        if len(picked) >= n:
            break
    picked.sort(key=lambda pair: pair[0])
    return picked


def _company_identity(
    company_facts: dict[str, Any], submissions: dict[str, Any]
) -> tuple[str, str]:
    """Return `(cik, company_name)`, preferring `submissions` (EDGAR's company-profile feed)."""
    raw_cik = submissions.get("cik") or company_facts.get("cik")
    if raw_cik is None:
        raise ValueError("neither submissions nor companyfacts carries a CIK")
    name = submissions.get("name") or company_facts.get("entityName")
    if not name:
        raise ValueError("neither submissions nor companyfacts carries a company name")
    return validate_cik(raw_cik), str(name)


def build_statements(
    company_facts: dict[str, Any],
    submissions: dict[str, Any],
    ticker: str,
) -> StatementSet:
    """Build a `StatementSet` covering the last `PRESENTED_FISCAL_YEARS` fiscal years.

    `ticker` is threaded through as given -- callers are expected to have
    already validated it (`app.edgar.client.validate_ticker`) since it
    originated from user input; this function does no I/O and trusts its
    typed inputs. Every canonical line item in `app.edgar.tags.CANONICAL` is
    resolved independently per year; a year with no usable fact for an item
    gets `None`, never a fabricated zero.
    """
    index = FactIndex(company_facts)
    year_periods = _discover_fiscal_years(index, PRESENTED_FISCAL_YEARS)
    fiscal_years = [fy for fy, _ in year_periods]
    period_end_by_fy = dict(year_periods)
    period_ends = set(period_end_by_fy.values())

    items: list[LineItem] = []
    for spec in CANONICAL:
        resolved = index.select(spec, period_ends)
        values: dict[int, float | None] = {}
        xbrl_tags: list[str] = []
        for fy in fiscal_years:
            period_end = period_end_by_fy[fy]
            rv = resolved.get(period_end)
            values[fy] = rv.value if rv is not None else None
            if rv is not None and rv.tag not in xbrl_tags:
                xbrl_tags.append(rv.tag)
        items.append(
            LineItem(
                key=spec.key,
                label=spec.label,
                statement=spec.statement,
                values=values,
                unit=spec.unit,
                xbrl_tags=xbrl_tags,
            )
        )

    cik, company_name = _company_identity(company_facts, submissions)
    return StatementSet(
        ticker=ticker,
        cik=cik,
        company_name=company_name,
        fiscal_years=fiscal_years,
        items=items,
    )
