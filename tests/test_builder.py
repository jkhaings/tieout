"""Hermetic tests for app.model.builder, against trimmed real SEC fixtures.

No network: `aapl_statements`/`msft_statements` (tests/conftest.py) build
from the committed fixtures in tests/fixtures/, produced once by
tests/fixtures/make_fixtures.py (CLAUDE.md rule 7).
"""

from __future__ import annotations

from typing import Any

from app.edgar.tags import (
    CANONICAL,
    DERIVED_PRETAX_INCOME,
    DERIVED_TOTAL_LIABILITIES,
    is_derived,
)
from app.model.builder import PRESENTED_FISCAL_YEARS, build_statements
from app.schemas import LineItem, StatementSet


def _by_key(statements: StatementSet) -> dict[str, LineItem]:
    return {item.key: item for item in statements.items}


def test_aapl_fiscal_years_ascending(aapl_statements: StatementSet) -> None:
    assert aapl_statements.fiscal_years == [2021, 2022, 2023, 2024, 2025]
    assert len(aapl_statements.fiscal_years) == PRESENTED_FISCAL_YEARS


def test_msft_fiscal_years_ascending_non_calendar(msft_statements: StatementSet) -> None:
    # Microsoft's fiscal year ends June 30 -- FY2026 covers July 2025-June 2026,
    # confirming fiscal year labeling isn't tied to calendar-year heuristics.
    assert msft_statements.fiscal_years == [2022, 2023, 2024, 2025, 2026]


def test_identity_fields_prefer_submissions(
    aapl_statements: StatementSet, msft_statements: StatementSet
) -> None:
    assert aapl_statements.ticker == "AAPL"
    assert aapl_statements.cik == "0000320193"
    assert aapl_statements.company_name == "Apple Inc."
    assert msft_statements.ticker == "MSFT"
    assert msft_statements.cik == "0000789019"
    assert msft_statements.company_name == "MICROSOFT CORP"


def test_every_canonical_item_is_present_even_if_all_none(aapl_statements: StatementSet) -> None:
    by_key = _by_key(aapl_statements)
    assert len(by_key) == len(CANONICAL)
    for spec in CANONICAL:
        assert spec.key in by_key


def test_spot_check_filed_values_aapl(aapl_statements: StatementSet) -> None:
    """Cross-checked directly against the real filing (see module docstring)."""
    by_key = _by_key(aapl_statements)
    assert by_key["total_assets"].values[2024] == 364_980_000_000.0
    assert by_key["net_income"].values[2024] == 93_736_000_000.0
    assert by_key["revenue"].values[2023] == 383_285_000_000.0
    assert by_key["total_assets"].values[2021] == 351_002_000_000.0


def test_spot_check_filed_values_msft(msft_statements: StatementSet) -> None:
    by_key = _by_key(msft_statements)
    assert by_key["total_assets"].values[2026] == 758_376_000_000.0
    assert by_key["net_income"].values[2026] == 133_749_000_000.0


def test_missing_data_is_none_not_a_fabricated_zero(aapl_statements: StatementSet) -> None:
    """Apple's balance sheet reports no separate Goodwill or Intangibles line --
    every year must be None, never a fabricated 0.0 (CLAUDE.md rule 4)."""
    by_key = _by_key(aapl_statements)
    for key in ("goodwill", "intangibles"):
        item = by_key[key]
        assert set(item.values.values()) == {None}, f"{key} should be all-None for Apple"


def test_xbrl_tags_records_provenance(aapl_statements: StatementSet) -> None:
    by_key = _by_key(aapl_statements)
    assert by_key["total_assets"].xbrl_tags == ["Assets"]
    assert by_key["revenue"].xbrl_tags == ["RevenueFromContractWithCustomerExcludingAssessedTax"]
    # A line item with no reported year anywhere records no tags either.
    assert by_key["goodwill"].xbrl_tags == []


def test_msft_fx_effect_resolves_via_the_current_tag_name(msft_statements: StatementSet) -> None:
    """Microsoft's FX-on-cash tag was renamed by FASB (see app.edgar.tags); this
    fixture's 5 presented years (2022-2026) all postdate the last rename, so
    every year resolves via the same, current tag. The per-year engagement
    *across* a rename, within one window, is exercised directly against a
    synthetic history in test_facts.py -- this test just confirms Apple (no
    such tag at all) and Microsoft (one consistent tag here) both come out
    right for the real, present-day fixture."""
    fx = _by_key(msft_statements)["fx_effect_on_cash"]
    assert fx.xbrl_tags == [
        "EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations"
    ]
    assert all(value is not None for value in fx.values.values())


def test_aapl_has_no_fx_effect_tag_at_all(aapl_statements: StatementSet) -> None:
    fx = _by_key(aapl_statements)["fx_effect_on_cash"]
    assert set(fx.values.values()) == {None}
    assert fx.xbrl_tags == []


def test_unit_is_threaded_through_for_non_usd_items(aapl_statements: StatementSet) -> None:
    by_key = _by_key(aapl_statements)
    assert by_key["eps_diluted"].unit == "USD/shares"
    assert by_key["weighted_diluted_shares"].unit == "shares"
    assert by_key["total_assets"].unit == "USD"
    assert by_key["eps_diluted"].values[2024] == 6.08
    assert by_key["weighted_diluted_shares"].values[2024] == 15_408_095_000.0


def test_emerging_disclosure_is_none_before_first_reported_then_populated(
    aapl_statements: StatementSet,
) -> None:
    """Apple only began separately disclosing G&A in its FY2025 filing's 3-year
    comparative table -- 2021/2022 correctly have no fact under that tag at all,
    while 2023-2025 do. This is a real filer behavior, not a gap to paper over."""
    ga = _by_key(aapl_statements)["general_administrative_expense"]
    assert ga.values[2021] is None
    assert ga.values[2022] is None
    assert ga.values[2023] == 6_672_000_000.0
    assert ga.values[2025] == 8_077_000_000.0


def test_build_statements_is_pure_and_deterministic(
    aapl_facts: dict[str, Any], aapl_submissions: dict[str, Any]
) -> None:
    first = build_statements(aapl_facts, aapl_submissions, "AAPL")
    second = build_statements(aapl_facts, aapl_submissions, "AAPL")
    assert first == second


# ---- Production workbook audit: derived line items -------------------------
#
# MCD files no `us-gaap:Liabilities` and no consolidated pretax element, so
# total liabilities and pretax income were blank for every year, the
# debt-to-equity ratio had nothing to divide, and neither the balance-sheet
# equation nor the net-income build-up could be scored even once.


def _facts(tags: dict[str, Any]) -> dict[str, Any]:
    return {"cik": 1, "entityName": "Synthetic Co", "facts": {"us-gaap": tags}}


def _instant(tag: str, end: str, val: float, fy: int) -> tuple[str, Any]:
    return tag, {
        "units": {
            "USD": [
                {
                    "end": end,
                    "val": val,
                    "fy": fy,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": f"{fy + 1}-02-01",
                }
            ]
        }
    }


def _duration(tag: str, end: str, val: float, fy: int) -> tuple[str, Any]:
    return tag, {
        "units": {
            "USD": [
                {
                    "start": f"{fy}-01-01",
                    "end": end,
                    "val": val,
                    "fy": fy,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": f"{fy + 1}-02-01",
                }
            ]
        }
    }


_SUBMISSIONS = {"cik": 1, "name": "Synthetic Co"}


def _item(statements: StatementSet, key: str) -> LineItem:
    return next(i for i in statements.items if i.key == key)


def test_total_liabilities_is_derived_when_no_liabilities_tag_is_filed(
    mcd_statements: StatementSet,
) -> None:
    liabilities = _item(mcd_statements, "total_liabilities")
    total = _item(mcd_statements, "total_liabilities_and_equity")
    equity = _item(mcd_statements, "total_equity")

    assert liabilities.xbrl_tags == [DERIVED_TOTAL_LIABILITIES]
    assert is_derived(liabilities.xbrl_tags[0])
    for fy in mcd_statements.fiscal_years:
        assert liabilities.values[fy] == total.values[fy] - equity.values[fy]


def test_pretax_income_is_derived_from_the_jurisdiction_split(
    mcd_statements: StatementSet,
) -> None:
    pretax = _item(mcd_statements, "pretax_income")
    assert pretax.xbrl_tags == [DERIVED_PRETAX_INCOME]
    assert all(pretax.values[fy] is not None for fy in mcd_statements.fiscal_years)


def test_filed_values_are_never_overwritten_by_a_derivation(
    aapl_statements: StatementSet, meta_statements: StatementSet
) -> None:
    """Both filers report `Liabilities` directly; neither may be derived."""
    for statements in (aapl_statements, meta_statements):
        liabilities = _item(statements, "total_liabilities")
        assert liabilities.xbrl_tags == ["Liabilities"]
        assert not any(is_derived(tag) for tag in liabilities.xbrl_tags)


def test_derivation_refuses_when_noncontrolling_interests_would_overstate_it() -> None:
    """`L&SE - StockholdersEquity` is `liabilities + NCI` when equity is
    parent-only, and no scored check could catch the overstatement -- the
    balance-sheet check for such a filer compares Assets against L&SE and
    would tie either way. Refuse rather than ship a silently wrong number.
    """
    shared = dict(
        [
            _instant("Assets", "2024-12-31", 100.0, 2024),
            _instant("LiabilitiesAndStockholdersEquity", "2024-12-31", 100.0, 2024),
            _instant("StockholdersEquity", "2024-12-31", 60.0, 2024),
        ]
    )
    without_nci = build_statements(_facts(shared), _SUBMISSIONS, "TEST")
    assert _item(without_nci, "total_liabilities").values[2024] == 40.0

    with_nci = build_statements(
        _facts({**shared, **dict([_instant("MinorityInterest", "2024-12-31", 5.0, 2024)])}),
        _SUBMISSIONS,
        "TEST",
    )
    liabilities = _item(with_nci, "total_liabilities")
    assert liabilities.values[2024] is None  # refused, not 45.0
    assert liabilities.xbrl_tags == []


def test_derivation_is_skipped_without_its_inputs() -> None:
    """No `LiabilitiesAndStockholdersEquity` means nothing to derive from."""
    statements = build_statements(
        _facts(dict([_instant("Assets", "2024-12-31", 100.0, 2024)])), _SUBMISSIONS, "TEST"
    )
    liabilities = _item(statements, "total_liabilities")
    assert liabilities.values[2024] is None
    assert liabilities.xbrl_tags == []


def test_pretax_derivation_needs_both_jurisdictions() -> None:
    """Domestic alone is not an exhaustive partition, so it is not a derivation."""
    statements = build_statements(
        _facts(
            dict(
                [
                    _instant("Assets", "2024-12-31", 100.0, 2024),
                    _duration(
                        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
                        "2024-12-31",
                        10.0,
                        2024,
                    ),
                ]
            )
        ),
        _SUBMISSIONS,
        "TEST",
    )
    assert _item(statements, "pretax_income").values[2024] is None


def test_meta_ppe_populates_from_the_finance_lease_inclusive_element(
    meta_statements: StatementSet,
) -> None:
    """META files zero annual `PropertyPlantAndEquipmentNet` facts, so PP&E was
    blank for every year without the fallback."""
    ppe = _item(meta_statements, "ppe_net")
    assert all(ppe.values[fy] is not None for fy in meta_statements.fiscal_years)
    assert ppe.xbrl_tags == [
        "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization"
    ]


def test_filers_reporting_plain_ppe_never_use_the_broader_fallback(
    aapl_statements: StatementSet, mcd_statements: StatementSet
) -> None:
    """The fallback folds finance-lease right-of-use assets into PP&E, so it
    must never displace the plain tag for a filer that reports both."""
    for statements in (aapl_statements, mcd_statements):
        assert _item(statements, "ppe_net").xbrl_tags == ["PropertyPlantAndEquipmentNet"]
