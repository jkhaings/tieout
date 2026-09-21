"""Tie-out verification: accounting identities checked against a `StatementSet`.

`verify()` returns the frozen `TieoutReport` and scores only identities
confirmed, empirically, to tie exactly against real filings across eight
tickers spanning tech, banking, retail, energy, and health insurance (the
same discipline `app.edgar.tags` applies to tag chains). Two checks that
looked like natural candidates were tried against real numbers and rejected
because they do not actually tie for real filings -- scoring them would
fail every correct model, which is worse than not checking them:

- A current-asset subtotal built from this model's own line items (cash +
  short-term investments + receivables + inventory) undershoots the filed
  `total_current_assets` by 14-48% for every filer tested, because real
  balance sheets carry several more current-asset lines (prepaid expenses,
  non-trade receivables, ...) than any fixed ~15-line canonical set will
  ever itemize.
- Building operating income from revenue minus a fixed list of expense
  components (cost of revenue, R&D, SG&A) works for the tech-style filers
  in the validation set but is off by billions for Walmart (it reports
  additional opex lines not in this model) and is nonsensical for
  UnitedHealth (a health insurer's income statement isn't shaped like a
  manufacturer's -- "cost of revenue" is not the right concept at all).

What *does* tie exactly, and is scored below: the balance sheet equation;
cash roll-forward including the (often-absent) FX effect; gross profit and
operating income built up from revenue when a filer reports the tags that
make that meaningful; and net income built up from pretax income and tax,
which requires including noncontrolling interest in income to close exactly
for the three multi-subsidiary filers in the validation set (Exxon, Walmart,
UnitedHealth) -- see `app.edgar.tags` for that finding.

`reconcile()` returns the retained-earnings roll-forward separately, as a
plain dataclass outside the frozen `app/schemas.py` contract, specifically
because it does *not* reliably tie (see its docstring) and must never be
able to affect `TieoutReport.passed`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.edgar.tags import is_derived
from app.schemas import LineItem, StatementSet, TieoutCheck, TieoutReport

# A filed fact carries the rounding of the statement it was presented in: a
# filer presenting in millions files 8_223_000_000, not the exact dollar
# amount. Every term of a summed identity therefore contributes up to its own
# rounding unit of noise, so a check's tolerance is the *sum of its terms'
# units*, not one global constant. Measured against real filings:
# McDonald's FY2024 cash roll-forward differs by exactly $1,000,000 across
# six million-rounded terms -- ordinary rounding, not a broken statement --
# while its FY2022 balance sheet mixes a million-rounded `Assets` with a
# hundred-thousand-rounded `LiabilitiesAndStockholdersEquity` and differs by
# $400,000. Taking the *finest* unit across terms would false-FAIL that
# second case: accumulated rounding noise is bounded by the coarsest
# rounding applied to any term, never the finest.
_MIN_TOLERANCE = 1.0  # floor -- guards float representation for whole-dollar filers
_ROUNDING_UNITS = (1_000_000.0, 100_000.0, 10_000.0, 1_000.0)
_REL_EPS = 1e-12


def _rounding_unit(value: float) -> float:
    """The coarsest unit `value` is an exact multiple of, or `1.0`.

    Units below $1,000 are deliberately not candidates: a whole-dollar figure
    that happens to end in "00" is a coincidence, not a reporting scale, and
    admitting it would widen every exact filer's tolerance for nothing.

    Args:
        value: One filed term of an accounting identity, in USD.

    Returns:
        The largest of `_ROUNDING_UNITS` that divides `value` exactly, else
        `1.0`. Tested against the reconstructed value rather than with `%`,
        so it is sign-safe and scale-safe.
    """
    for unit in _ROUNDING_UNITS:
        if abs(value - round(value / unit) * unit) <= abs(value) * _REL_EPS:
            return unit
    return 1.0


def _infer_tolerance(terms: Sequence[float]) -> float:
    """Sum each term's own rounding unit -- the bound on accumulated rounding noise.

    Args:
        terms: Every filed value the identity sums. An optional term that was
            not reported contributes no rounding noise and is never passed in
            (see `verify`).

    Returns:
        The tolerance in USD, never below `_MIN_TOLERANCE`. Exact zeros are
        skipped: they divide by every unit, so they carry no evidence of the
        filer's presentation scale.
    """
    return max(_MIN_TOLERANCE, sum(_rounding_unit(t) for t in terms if t != 0.0))


class _Values:
    """O(1) `(canonical key, fiscal year) -> value` lookup over a StatementSet."""

    def __init__(self, statements: StatementSet) -> None:
        """Index `statements.items` by canonical key for repeated lookups."""
        self._by_key: dict[str, LineItem] = {item.key: item for item in statements.items}

    def get(self, key: str, fiscal_year: int) -> float | None:
        """The reported value for `key` in `fiscal_year`, or `None` if unreported."""
        item = self._by_key.get(key)
        if item is None:
            return None
        return item.values.get(fiscal_year)

    def get_or_zero(self, key: str, fiscal_year: int) -> float:
        """For terms that are legitimately zero when unreported (see `verify` docstring)."""
        value = self.get(key, fiscal_year)
        return value if value is not None else 0.0

    def is_derived(self, key: str) -> bool:
        """True if this line item's values came from algebra rather than a filed tag.

        Item-level, not per-year: `LineItem.xbrl_tags` records which tags a
        line item used across all presented years and cannot say *which* year
        used which, so `app.model.builder` only ever derives all-or-nothing.
        """
        item = self._by_key.get(key)
        return item is not None and any(is_derived(tag) for tag in item.xbrl_tags)


def _check(
    check_id: str,
    description: str,
    fiscal_year: int,
    lhs: float,
    rhs: float,
    *,
    terms: Sequence[float],
) -> TieoutCheck:
    """One scored check, with its tolerance inferred from its own inputs' rounding.

    `terms` is keyword-only so it can never be passed positionally where
    `lhs`/`rhs` belong, and the inferred tolerance is spelled out in the
    check's own description -- a reader seeing a $6,000,000 tolerance in the
    Tie-out tab is owed the arithmetic behind it.
    """
    assert terms, f"{check_id}: a scored check must declare the terms it sums"
    tolerance = _infer_tolerance(terms)
    coarsest = max((_rounding_unit(t) for t in terms if t != 0.0), default=1.0)
    return TieoutCheck(
        check_id=check_id,
        description=(
            f"{description} (tolerance ${tolerance:,.0f}: {len(terms)} filed terms, "
            f"rounded to at most the nearest ${coarsest:,.0f})"
        ),
        fiscal_year=fiscal_year,
        passed=abs(lhs - rhs) <= tolerance,
        lhs=lhs,
        rhs=rhs,
        tolerance=tolerance,
    )


def verify(statements: StatementSet) -> TieoutReport:
    """Run every scored tie-out check against `statements`.

    A check is included for a fiscal year only when every *required* input
    is a reported value -- omitted, not recorded as a pass, when data is
    missing (CLAUDE.md rule 4: fail closed). Two inputs are treated as
    optional reconciling terms that contribute `0` when unreported rather
    than causing the whole check to be omitted, because their absence is
    itself meaningful (no material FX exposure on cash; no noncontrolling
    interests) rather than missing data -- each is named explicitly in its
    check's `description` so this is visible in the Tie-out tab, not just
    in code: `fx_effect_on_cash` in the cash roll-forward, and
    `noncontrolling_interest_in_income` in the net-income build-up.

    The cash and retained-earnings-adjacent checks need a prior fiscal
    year's balance; `StatementSet` presents exactly `fiscal_years` (no
    hidden extra year -- see `app.model.builder`), so the earliest
    presented year has no prior year available and simply has no
    roll-forward check, rather than reaching outside the model's contract
    for one more year of data.

    If a fiscal year ends up with zero evaluable checks (e.g. a filer
    missing every required input), an explicit failing
    `insufficient_data_<fy>` check is emitted instead of silently omitting
    the year: `all([])` is `True` in Python, so a report with no checks at
    all must never be allowed to read as a clean tie-out.
    """
    v = _Values(statements)
    checks: list[TieoutCheck] = []

    for fy in statements.fiscal_years:
        year_checks: list[TieoutCheck] = []

        assets = v.get("total_assets", fy)
        liabilities = v.get("total_liabilities", fy)
        equity = v.get("total_equity", fy)
        if v.is_derived("total_liabilities"):
            # This filer reports no `us-gaap:Liabilities`, so total liabilities
            # was derived from `L&SE - equity` (see `app.model.builder`). The
            # three-term identity would then reduce algebraically to
            # `assets = L&SE` and score the derivation against itself -- a
            # check that cannot fail proves nothing. Compare the two
            # independently filed facts instead, and say so in the row.
            liabilities_and_equity = v.get("total_liabilities_and_equity", fy)
            if assets is not None and liabilities_and_equity is not None:
                year_checks.append(
                    _check(
                        f"balance_sheet_equation_independent_{fy}",
                        "Total assets = total liabilities and stockholders' equity, both as "
                        "filed (this filer reports no us-gaap:Liabilities, so total "
                        "liabilities is derived and the three-term identity would be circular)",
                        fy,
                        assets,
                        liabilities_and_equity,
                        terms=(assets, liabilities_and_equity),
                    )
                )
        elif assets is not None and liabilities is not None and equity is not None:
            year_checks.append(
                _check(
                    f"balance_sheet_equation_{fy}",
                    "Total assets = total liabilities + total stockholders' equity",
                    fy,
                    assets,
                    liabilities + equity,
                    terms=(assets, liabilities, equity),
                )
            )

        revenue = v.get("revenue", fy)
        cost_of_revenue = v.get("cost_of_revenue", fy)
        gross_profit = v.get("gross_profit", fy)
        if revenue is not None and cost_of_revenue is not None and gross_profit is not None:
            year_checks.append(
                _check(
                    f"gross_profit_buildup_{fy}",
                    "Gross profit = revenue - cost of revenue",
                    fy,
                    gross_profit,
                    revenue - cost_of_revenue,
                    terms=(gross_profit, revenue, cost_of_revenue),
                )
            )

        operating_expenses = v.get("operating_expenses", fy)
        operating_income = v.get("operating_income", fy)
        if (
            gross_profit is not None
            and operating_expenses is not None
            and operating_income is not None
        ):
            year_checks.append(
                _check(
                    f"operating_income_buildup_{fy}",
                    "Operating income = gross profit - total operating expenses",
                    fy,
                    operating_income,
                    gross_profit - operating_expenses,
                    terms=(operating_income, gross_profit, operating_expenses),
                )
            )

        pretax_income = v.get("pretax_income", fy)
        income_tax = v.get("income_tax_expense", fy)
        net_income = v.get("net_income", fy)
        if pretax_income is not None and income_tax is not None and net_income is not None:
            reported_nci = v.get("noncontrolling_interest_in_income", fy)
            nci = reported_nci if reported_nci is not None else 0.0
            terms: tuple[float, ...] = (net_income, pretax_income, income_tax)
            if reported_nci is not None:
                terms += (reported_nci,)
            year_checks.append(
                _check(
                    f"net_income_buildup_{fy}",
                    "Net income = pretax income - income tax expense - "
                    "noncontrolling interest in income (0 if none reported)",
                    fy,
                    net_income,
                    pretax_income - income_tax - nci,
                    terms=terms,
                )
            )

        cash = v.get("cash_and_equivalents", fy)
        cash_prior = v.get("cash_and_equivalents", fy - 1)
        cfo = v.get("cfo", fy)
        cfi = v.get("cfi", fy)
        cff = v.get("cff", fy)
        if None not in (cash, cash_prior, cfo, cfi, cff):
            assert cash is not None and cash_prior is not None
            assert cfo is not None and cfi is not None and cff is not None
            reported_fx = v.get("fx_effect_on_cash", fy)
            fx = reported_fx if reported_fx is not None else 0.0
            cash_terms: tuple[float, ...] = (cash, cash_prior, cfo, cfi, cff)
            if reported_fx is not None:
                cash_terms += (reported_fx,)
            year_checks.append(
                _check(
                    f"cash_roll_forward_{fy}",
                    "Ending cash - beginning cash = CFO + CFI + CFF + effect of "
                    "exchange rates (0 if none reported)",
                    fy,
                    cash - cash_prior,
                    cfo + cfi + cff + fx,
                    terms=cash_terms,
                )
            )

        if not year_checks:
            year_checks.append(
                TieoutCheck(
                    check_id=f"insufficient_data_{fy}",
                    description=(
                        "No scored check had every required input reported for this "
                        "fiscal year; flagged rather than silently omitted."
                    ),
                    fiscal_year=fy,
                    passed=False,
                )
            )
        checks.extend(year_checks)

    return TieoutReport(ticker=statements.ticker, checks=checks)


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """One informational roll-forward. Never scores into `TieoutReport.passed`.

    Deliberately outside `app/schemas.py`: unlike `TieoutCheck`, this is not
    a pass/fail identity, so giving it the same shape would misrepresent it.
    """

    label: str
    fiscal_year: int
    beginning: float
    net_income: float
    dividends: float
    buybacks: float
    ending_actual: float
    ending_expected: float
    residual: float
    residual_pct_of_assets: float | None


def reconcile(statements: StatementSet) -> list[Reconciliation]:
    """Retained-earnings roll-forward, informational only -- deliberately not scored.

    `RE_end = RE_begin + net income - dividends - buybacks` is the textbook
    identity, but it does not tie exactly from XBRL data alone: share
    repurchases are retired against additional paid-in capital, retained
    earnings, or a blend of both depending on the filer, and that split is
    not consistently tagged anywhere in companyfacts -- `share_repurchases`
    here is the cash actually paid (`PaymentsForRepurchaseOfCommonStock`),
    which is a good proxy but not always the exact amount later charged to
    RE. Checked against this model's own output for both fixtures: residuals
    of roughly 0.3-1.2% of total assets in every presented year for both
    Apple and Microsoft ($1.0-4.2B and $4.3-5.5B respectively) -- small
    relative to the balance sheet, but not zero, and not shrinking in a way
    that suggests it is just float noise. Scoring a check that is
    consistently *close but not exact* as binary pass/fail would need an
    arbitrary percentage tolerance found nowhere else in this module (every
    other check's tolerance is absolute and derived from the filed values'
    own rounding -- see `_infer_tolerance` -- which is exactly the
    justification a percentage of assets would lack); rather than invent one,
    this is surfaced as a named residual
    for the Tie-out tab to show honestly, without asserting a verdict
    (CLAUDE.md rule 4 says fail closed on a *failed* tie-out, not manufacture
    a check that can never cleanly pass or fail).
    """
    v = _Values(statements)
    out: list[Reconciliation] = []
    for fy in statements.fiscal_years:
        beginning = v.get("retained_earnings", fy - 1)
        ending_actual = v.get("retained_earnings", fy)
        net_income = v.get("net_income", fy)
        if beginning is None or ending_actual is None or net_income is None:
            continue
        dividends = v.get_or_zero("dividends_paid", fy)
        buybacks = v.get_or_zero("share_repurchases", fy)
        ending_expected = beginning + net_income - dividends - buybacks
        residual = ending_actual - ending_expected
        assets = v.get("total_assets", fy)
        out.append(
            Reconciliation(
                label=f"retained_earnings_rollforward_{fy}",
                fiscal_year=fy,
                beginning=beginning,
                net_income=net_income,
                dividends=dividends,
                buybacks=buybacks,
                ending_actual=ending_actual,
                ending_expected=ending_expected,
                residual=residual,
                residual_pct_of_assets=(abs(residual) / assets * 100) if assets else None,
            )
        )
    return out
