"""Canonical line items mapped to `us-gaap` XBRL tag fallback chains.

Each `TagSpec` names one line item a three-statement model presents, and the
ordered list of `us-gaap` tags that can supply its value -- filers tag the
same concept under different element names, and the same filer's own choice
of tag can drift across years as the FASB taxonomy evolves. `app.model.facts`
walks each chain independently *per fiscal year* and takes the first tag
with a usable fact for that year (see the module docstring there for why:
a single winning tag per line item is not sound -- Microsoft's FX-on-cash
tag alone changed twice between 2016 and 2026).

Chains were built from real coverage, not guessed: every candidate tag was
checked against companyfacts for eight filers spanning the shapes that break
naive mappings -- AAPL, MSFT, GOOGL, NVDA (large-cap tech), JPM (bank,
no gross profit / no cost of revenue), XOM (oil & gas, September-unrelated
custom cost tags), WMT (January fiscal year-end), UNH (health insurer).
A tag was only added to a chain when it was confirmed to represent the same
concept as the others in that chain -- same scope, same sign convention --
not merely a plausibly-similar name. Two pairs that looked like fallback
candidates were rejected after checking real values: `NonoperatingIncomeExpense`
vs `OtherNonoperatingIncomeExpense`, and `LongTermDebt` vs
`LongTermDebtNoncurrent` are simultaneously reported by the same filer in the
same period with different values -- they are distinct concepts, not
alternate spellings of one concept, and chaining them would silently swap
in the wrong number for some filers. Likewise Microsoft and Alphabet report
"selling, general and administrative" only as two separate lines (general &
administrative, selling & marketing) with no combined tag at all, so
`selling_general_admin` is its own item alongside, not instead of,
`general_administrative_expense` and `selling_marketing_expense`: summing two
different XBRL concepts to approximate a third would fabricate a number no
filer actually reported.

That discipline is why this file has ~43 items rather than the ~35 first
estimated in ARCHITECTURE.md/CLAUDE.md -- the difference is entirely the
three-way SG&A split, one extra presented balance-sheet total
(`total_liabilities_and_equity`), and one reconciling term needed to make
the net-income build-up check actually tie (`noncontrolling_interest_in_income`),
not scope creep.

`kind` says how a fact's period is matched in `app.model.facts`:
- `"instant"` -- balance-sheet items, matched by period end date.
- `"duration"` -- income-statement and cash-flow items, matched by a
  ~330-400 day (annual) start/end span, which is what excludes quarterly
  and cumulative-quarter facts from an annual model.

There is deliberately no `sign` field. Every chained tag is an XBRL element
with a fixed, well-known balance type: expense/outflow concepts (e.g.
`ResearchAndDevelopmentExpense`, `PaymentsToAcquirePropertyPlantAndEquipment`)
are always filed as non-negative magnitudes, and net concepts (e.g.
`NetCashProvidedByUsedInFinancingActivities`) are already signed as filed.
`app.model.verifier` and `app.model.excel` combine values using that natural,
as-filed convention explicitly in each formula, so a generic per-item sign
multiplier would be redundant metadata rather than a real simplification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from app.schemas import Statement

PeriodKind = Literal["instant", "duration"]


@dataclass(frozen=True, slots=True)
class TagSpec:
    """One canonical line item and its ordered `us-gaap` tag fallback chain."""

    key: str  # canonical id, matches LineItem.key
    label: str  # display label for the workbook
    statement: Statement
    kind: PeriodKind
    tags: tuple[str, ...]  # fallback chain, most-preferred first; never empty
    unit: str = "USD"  # XBRL unit of measure; "USD/shares" for EPS, "shares" for share counts


CANONICAL: Final[tuple[TagSpec, ...]] = (
    # ---- Income statement -------------------------------------------------
    TagSpec(
        "revenue",
        "Revenue",
        Statement.INCOME,
        "duration",
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ),
    ),
    TagSpec(
        "cost_of_revenue",
        "Cost of revenue",
        Statement.INCOME,
        "duration",
        ("CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"),
    ),
    TagSpec(
        "gross_profit",
        "Gross profit",
        Statement.INCOME,
        "duration",
        ("GrossProfit",),
    ),
    TagSpec(
        "research_development",
        "Research and development",
        Statement.INCOME,
        "duration",
        ("ResearchAndDevelopmentExpense",),
    ),
    TagSpec(
        "selling_general_admin",
        "Selling, general and administrative",
        Statement.INCOME,
        "duration",
        ("SellingGeneralAndAdministrativeExpense",),
    ),
    TagSpec(
        "general_administrative_expense",
        "General and administrative",
        Statement.INCOME,
        "duration",
        ("GeneralAndAdministrativeExpense",),
    ),
    TagSpec(
        "selling_marketing_expense",
        "Selling and marketing",
        Statement.INCOME,
        "duration",
        ("SellingAndMarketingExpense",),
    ),
    TagSpec(
        "operating_expenses",
        "Total operating expenses",
        Statement.INCOME,
        "duration",
        ("OperatingExpenses",),
    ),
    TagSpec(
        "operating_income",
        "Operating income",
        Statement.INCOME,
        "duration",
        ("OperatingIncomeLoss",),
    ),
    TagSpec(
        "interest_expense",
        "Interest expense",
        Statement.INCOME,
        "duration",
        ("InterestExpense", "InterestExpenseDebt"),
    ),
    TagSpec(
        "other_income_expense",
        "Other income (expense), net",
        Statement.INCOME,
        "duration",
        ("NonoperatingIncomeExpense",),
    ),
    TagSpec(
        "pretax_income",
        "Income before income taxes",
        Statement.INCOME,
        "duration",
        (
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        ),
    ),
    TagSpec(
        "income_tax_expense",
        "Income tax expense",
        Statement.INCOME,
        "duration",
        ("IncomeTaxExpenseBenefit",),
    ),
    TagSpec(
        "net_income",
        "Net income",
        Statement.INCOME,
        "duration",
        ("NetIncomeLoss", "ProfitLoss"),
    ),
    TagSpec(
        # A reconciling term, not a headline presented line: `net_income`
        # above is post-NCI (net income attributable to the parent) for
        # filers with partly-owned subsidiaries (XOM, WMT, UNH in the
        # 8-ticker validation set), so `verifier.net_income_buildup` needs
        # this to close pretax_income - income_tax_expense - NCI ==
        # net_income exactly. Confirmed empirically: without it the
        # build-up is off by 2-6% of net income for those three filers;
        # with it, exact across all eight. Absent (e.g. Apple, which has
        # no noncontrolling interests) is a true zero, not missing data.
        "noncontrolling_interest_in_income",
        "Net income attributable to noncontrolling interests",
        Statement.INCOME,
        "duration",
        (
            "NetIncomeLossAttributableToNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsAttributableToNoncontrollingEntity",
        ),
    ),
    TagSpec(
        "eps_diluted",
        "Diluted earnings per share",
        Statement.INCOME,
        "duration",
        ("EarningsPerShareDiluted",),
        "USD/shares",
    ),
    TagSpec(
        "weighted_diluted_shares",
        "Weighted-average diluted shares outstanding",
        Statement.INCOME,
        "duration",
        ("WeightedAverageNumberOfDilutedSharesOutstanding",),
        "shares",
    ),
    # ---- Balance sheet ------------------------------------------------------
    TagSpec(
        "cash_and_equivalents",
        "Cash and cash equivalents",
        Statement.BALANCE,
        "instant",
        (
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            "CashAndCashEquivalentsAtCarryingValue",
        ),
    ),
    TagSpec(
        "short_term_investments",
        "Short-term investments",
        Statement.BALANCE,
        "instant",
        ("ShortTermInvestments", "MarketableSecuritiesCurrent"),
    ),
    TagSpec(
        "accounts_receivable",
        "Accounts receivable, net",
        Statement.BALANCE,
        "instant",
        ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent", "NontradeReceivablesCurrent"),
    ),
    TagSpec(
        "inventory",
        "Inventory",
        Statement.BALANCE,
        "instant",
        ("InventoryNet",),
    ),
    TagSpec(
        "total_current_assets",
        "Total current assets",
        Statement.BALANCE,
        "instant",
        ("AssetsCurrent",),
    ),
    TagSpec(
        "ppe_net",
        "Property, plant and equipment, net",
        Statement.BALANCE,
        "instant",
        ("PropertyPlantAndEquipmentNet",),
    ),
    TagSpec(
        "goodwill",
        "Goodwill",
        Statement.BALANCE,
        "instant",
        ("Goodwill",),
    ),
    TagSpec(
        "intangibles",
        "Intangible assets, net",
        Statement.BALANCE,
        "instant",
        ("FiniteLivedIntangibleAssetsNet", "IntangibleAssetsNetExcludingGoodwill"),
    ),
    TagSpec(
        "total_assets",
        "Total assets",
        Statement.BALANCE,
        "instant",
        ("Assets",),
    ),
    TagSpec(
        "accounts_payable",
        "Accounts payable",
        Statement.BALANCE,
        "instant",
        ("AccountsPayableCurrent", "AccountsPayableTradeCurrent"),
    ),
    TagSpec(
        "total_current_liabilities",
        "Total current liabilities",
        Statement.BALANCE,
        "instant",
        ("LiabilitiesCurrent",),
    ),
    TagSpec(
        "long_term_debt",
        "Long-term debt",
        Statement.BALANCE,
        "instant",
        # Noncurrent-only tag preferred: in years where a filer discloses
        # both, `LongTermDebt` includes the current portion and is *not*
        # the same figure -- it is used here only as a fallback for filers
        # / years that never split one out (see module docstring).
        ("LongTermDebtNoncurrent", "LongTermDebt"),
    ),
    TagSpec(
        "total_liabilities",
        "Total liabilities",
        Statement.BALANCE,
        "instant",
        ("Liabilities",),
    ),
    TagSpec(
        "total_liabilities_and_equity",
        "Total liabilities and stockholders' equity",
        Statement.BALANCE,
        "instant",
        ("LiabilitiesAndStockholdersEquity",),
    ),
    TagSpec(
        "retained_earnings",
        "Retained earnings (accumulated deficit)",
        Statement.BALANCE,
        "instant",
        ("RetainedEarningsAccumulatedDeficit",),
    ),
    TagSpec(
        "total_equity",
        "Total stockholders' equity",
        Statement.BALANCE,
        "instant",
        (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
    ),
    # ---- Cash flow ------------------------------------------------------
    TagSpec(
        "depreciation_amortization",
        "Depreciation and amortization",
        Statement.CASHFLOW,
        "duration",
        (
            "DepreciationDepletionAndAmortization",
            "DepreciationAndAmortization",
            "DepreciationAmortizationAndAccretionNet",
            # Last resort only: some filers (e.g. Microsoft) tag their sole
            # CF add-back as bare "Depreciation" with no combined tag at
            # all, while others use it for depreciation alone alongside a
            # separately tagged amortization figure -- ordered last so a
            # combined tag always wins when a filer provides one.
            "Depreciation",
        ),
    ),
    TagSpec(
        "stock_based_comp",
        "Stock-based compensation",
        Statement.CASHFLOW,
        "duration",
        ("ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"),
    ),
    TagSpec(
        "cfo",
        "Net cash provided by operating activities",
        Statement.CASHFLOW,
        "duration",
        ("NetCashProvidedByUsedInOperatingActivities",),
    ),
    TagSpec(
        "capex",
        "Capital expenditures",
        Statement.CASHFLOW,
        "duration",
        ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
    ),
    TagSpec(
        "cfi",
        "Net cash used in investing activities",
        Statement.CASHFLOW,
        "duration",
        ("NetCashProvidedByUsedInInvestingActivities",),
    ),
    TagSpec(
        "dividends_paid",
        "Dividends paid",
        Statement.CASHFLOW,
        "duration",
        ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends"),
    ),
    TagSpec(
        "share_repurchases",
        "Repurchases of common stock",
        Statement.CASHFLOW,
        "duration",
        ("PaymentsForRepurchaseOfCommonStock",),
    ),
    TagSpec(
        "cff",
        "Net cash used in financing activities",
        Statement.CASHFLOW,
        "duration",
        ("NetCashProvidedByUsedInFinancingActivities",),
    ),
    TagSpec(
        "fx_effect_on_cash",
        "Effect of exchange rate changes on cash",
        Statement.CASHFLOW,
        "duration",
        # FASB has renamed this concept twice (ASU 2016-18 restricted-cash
        # inclusion, then a later discontinued-operations refinement); the
        # same filer can move across all three within one companyfacts
        # history (Microsoft: tag 3 through 2021, then tag 1 onward).
        (
            "EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations",
            "EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            "EffectOfExchangeRateOnCashAndCashEquivalents",
        ),
    ),
    TagSpec(
        "net_change_in_cash",
        "Net increase (decrease) in cash",
        Statement.CASHFLOW,
        "duration",
        (
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect",
            "CashAndCashEquivalentsPeriodIncreaseDecrease",
        ),
    ),
)

BY_KEY: Final[dict[str, TagSpec]] = {spec.key: spec for spec in CANONICAL}
assert len(BY_KEY) == len(CANONICAL), "duplicate canonical key in tags.py"


def get(key: str) -> TagSpec:
    """Look up a `TagSpec` by canonical key. Raises `KeyError` if unknown."""
    return BY_KEY[key]
