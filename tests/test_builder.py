"""Hermetic tests for app.model.builder, against trimmed real SEC fixtures.

No network: `aapl_statements`/`msft_statements` (tests/conftest.py) build
from the committed fixtures in tests/fixtures/, produced once by
tests/fixtures/make_fixtures.py (CLAUDE.md rule 7).
"""

from __future__ import annotations

from typing import Any

from app.edgar.tags import CANONICAL
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
