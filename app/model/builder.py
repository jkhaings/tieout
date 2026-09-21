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

import math
from collections import Counter
from typing import Any

from app.edgar.client import validate_cik
from app.edgar.tags import (
    CANONICAL,
    DERIVED_PRETAX_INCOME,
    DERIVED_TOTAL_LIABILITIES,
    NONCONTROLLING_INTEREST_TAG,
    PARENT_ONLY_EQUITY_TAG,
    PRETAX_DOMESTIC,
    PRETAX_FOREIGN,
)
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
    # Which tag actually supplied each (key, year). Local to this function --
    # `LineItem.xbrl_tags` is item-level and `app/schemas.py` is frozen -- but
    # the derivation pass below needs it to tell parent-only equity from the
    # including-noncontrolling-interests variant.
    tag_by_key_year: dict[tuple[str, int], str] = {}
    for spec in CANONICAL:
        resolved = index.select(spec, period_ends)
        values: dict[int, float | None] = {}
        xbrl_tags: list[str] = []
        for fy in fiscal_years:
            period_end = period_end_by_fy[fy]
            rv = resolved.get(period_end)
            values[fy] = rv.value if rv is not None else None
            if rv is not None:
                tag_by_key_year[(spec.key, fy)] = rv.tag
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

    items = _apply_derivations(
        items,
        fiscal_years=fiscal_years,
        period_ends=period_ends,
        period_end_by_fy=period_end_by_fy,
        index=index,
        tag_by_key_year=tag_by_key_year,
        company_facts=company_facts,
    )

    cik, company_name = _company_identity(company_facts, submissions)
    return StatementSet(
        ticker=ticker,
        cik=cik,
        company_name=company_name,
        fiscal_years=fiscal_years,
        items=items,
    )


def _derive_total_liabilities(
    by_key: dict[str, LineItem],
    fiscal_years: list[int],
    tag_by_key_year: dict[tuple[str, int], str],
    company_facts: dict[str, Any],
) -> dict[int, float | None] | None:
    """Total liabilities as `L&SE - total equity`, for filers that never file `Liabilities`.

    McDonald's is the motivating case: it reports
    `LiabilitiesAndStockholdersEquity` but no `us-gaap:Liabilities` at all, so
    total liabilities is blank for every year, the debt-to-equity ratio has
    nothing to divide, and the balance-sheet equation cannot be scored once.

    Refuses rather than guesses in two situations. A filer that reports
    `Liabilities` for even one presented year keeps its real gaps untouched --
    a partly-derived item could not be described honestly by the item-level
    `LineItem.xbrl_tags`. And when equity resolved to the parent-only
    `StockholdersEquity` while the filer also reports noncontrolling
    interests, `L&SE - equity` is `liabilities + NCI`, overstating liabilities
    by exactly the noncontrolling interest -- an error no scored check could
    catch, because the balance-sheet check for such a filer compares `Assets`
    against `L&SE` and would tie regardless.

    Returns:
        Per-year derived values, or `None` if the derivation does not apply.
    """
    item = by_key.get("total_liabilities")
    if item is None or any(value is not None for value in item.values.values()):
        return None
    liabilities_and_equity = by_key.get("total_liabilities_and_equity")
    equity = by_key.get("total_equity")
    if liabilities_and_equity is None or equity is None:
        return None

    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    filer_has_nci = NONCONTROLLING_INTEREST_TAG in us_gaap

    values: dict[int, float | None] = {}
    for fy in fiscal_years:
        total, own = liabilities_and_equity.values.get(fy), equity.values.get(fy)
        parent_only = tag_by_key_year.get(("total_equity", fy)) == PARENT_ONLY_EQUITY_TAG
        if total is None or own is None or (filer_has_nci and parent_only):
            values[fy] = None
            continue
        values[fy] = total - own
    return None if all(value is None for value in values.values()) else values


def _derive_pretax_income(
    by_key: dict[str, LineItem],
    fiscal_years: list[int],
    period_ends: set[str],
    period_end_by_fy: dict[int, str],
    index: FactIndex,
) -> dict[int, float | None] | None:
    """Pretax income as `domestic + foreign`, for filers that file only the split.

    McDonald's files `...BeforeIncomeTaxesDomestic` and `...Foreign` and no
    consolidated pretax element, so `net_income_buildup` has no left-hand side
    and never scores. The split is an exhaustive partition rather than a
    component list, so summing it is exact -- and the result is immediately
    re-checked by `net_income_buildup` itself, which is what separates this
    from the component-summing `app.edgar.tags` refuses to do.

    Returns:
        Per-year derived values, or `None` if the derivation does not apply.
    """
    item = by_key.get("pretax_income")
    if item is None or any(value is not None for value in item.values.values()):
        return None
    domestic = index.select(PRETAX_DOMESTIC, period_ends)
    foreign = index.select(PRETAX_FOREIGN, period_ends)

    values: dict[int, float | None] = {}
    for fy in fiscal_years:
        period_end = period_end_by_fy[fy]
        at_home, abroad = domestic.get(period_end), foreign.get(period_end)
        values[fy] = None if at_home is None or abroad is None else at_home.value + abroad.value
    return None if all(value is None for value in values.values()) else values


def _apply_derivations(
    items: list[LineItem],
    *,
    fiscal_years: list[int],
    period_ends: set[str],
    period_end_by_fy: dict[int, str],
    index: FactIndex,
    tag_by_key_year: dict[tuple[str, int], str],
    company_facts: dict[str, Any],
) -> list[LineItem]:
    """Fill line items a filer never files but that follow exactly from ones it does.

    Each derivation replaces the whole item's values and records a marker in
    `xbrl_tags` (`app.edgar.tags.is_derived`) so a derived value is never
    mistaken for a filed one -- `app.model.verifier` reads that marker to
    avoid scoring an identity against its own derivation.

    Returns:
        `items`, with derived entries replaced. Never mutates in place, so
        `build_statements` stays pure.
    """
    by_key = {item.key: item for item in items}
    derived: dict[str, tuple[dict[int, float | None], str]] = {}

    liabilities = _derive_total_liabilities(by_key, fiscal_years, tag_by_key_year, company_facts)
    if liabilities is not None:
        derived["total_liabilities"] = (liabilities, DERIVED_TOTAL_LIABILITIES)

    pretax = _derive_pretax_income(by_key, fiscal_years, period_ends, period_end_by_fy, index)
    if pretax is not None:
        derived["pretax_income"] = (pretax, DERIVED_PRETAX_INCOME)

    if not derived:
        return items
    rebuilt: list[LineItem] = []
    for item in items:
        if item.key not in derived:
            rebuilt.append(item)
            continue
        values, marker = derived[item.key]
        rebuilt.append(item.model_copy(update={"values": values, "xbrl_tags": [marker]}))
    return rebuilt


# Filers tag share counts at whatever scale their income statement presents
# them: Meta files 2,574,000,000 diluted shares while McDonald's files 751.8
# for the same concept, both under the `shares` unit. Neither is wrong, but
# rendering them side by side without saying which is which is.
_SHARE_SCALE_CANDIDATES = (1.0, 1_000.0, 1_000_000.0)
_SHARES_IN_MILLIONS_FLOOR = 1_000_000.0


def share_filing_scale(statements: StatementSet) -> float:
    """The multiplier converting this filer's filed diluted share count to actual shares.

    Decided from the filer's own arithmetic wherever possible: net income
    divided by diluted EPS is the share count the filer itself implies, so the
    candidate scale whose product lands nearest that implied count is the
    right one. The candidates are 1,000x apart while the implied count sits
    within a fraction of a percent of the true one, so the choice is never
    close. One scale per statement set, by majority vote over the years
    carrying EPS evidence -- filers do not change presentation scale partway
    down a table, and a per-column scale would be worse than either answer.

    With no usable evidence (EPS or net income unreported, or EPS zero) this
    falls back to magnitude, where thousands is deliberately unreachable: a
    genuinely small filer's real share count would otherwise be misread by a
    factor of 1,000.

    Args:
        statements: The statement set whose share rows are being rendered.

    Returns:
        One of `_SHARE_SCALE_CANDIDATES`. Display-only -- `LineItem.values`
        always keeps the value exactly as filed.
    """
    by_key = {item.key: item for item in statements.items}
    shares = by_key.get("weighted_diluted_shares")
    if shares is None:
        return 1.0
    eps, net_income = by_key.get("eps_diluted"), by_key.get("net_income")

    votes: list[float] = []
    if eps is not None and net_income is not None:
        for fy in statements.fiscal_years:
            filed, per_share, earnings = (
                shares.values.get(fy),
                eps.values.get(fy),
                net_income.values.get(fy),
            )
            if not filed or not per_share or not earnings or filed <= 0:
                continue
            implied = earnings / per_share
            if implied <= 0:
                continue
            votes.append(
                min(
                    _SHARE_SCALE_CANDIDATES,
                    key=lambda scale: abs(math.log(filed * scale) - math.log(implied)),
                )
            )
    if votes:
        # Ties break toward the most recent year, which votes last.
        top = max(Counter(votes).values())
        return next(scale for scale in reversed(votes) if votes.count(scale) == top)

    latest = next(
        (
            shares.values[fy]
            for fy in reversed(statements.fiscal_years)
            if shares.values.get(fy) is not None
        ),
        None,
    )
    if latest is None or latest <= 0:
        return 1.0
    return 1.0 if latest >= _SHARES_IN_MILLIONS_FLOOR else 1_000_000.0
