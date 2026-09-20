"""Hermetic tests for app.model.verifier -- real fixtures plus synthetic edge cases."""

from __future__ import annotations

import pytest

from app.edgar.tags import BY_KEY
from app.model.verifier import Reconciliation, reconcile, verify
from app.schemas import LineItem, StatementSet, TieoutReport


def _statement_set(
    values_by_key: dict[str, dict[int, float | None]], fiscal_years: list[int]
) -> StatementSet:
    items = [
        LineItem(key=key, label=key, statement=BY_KEY[key].statement, values=values)
        for key, values in values_by_key.items()
    ]
    return StatementSet(
        ticker="TEST",
        cik="0000000001",
        company_name="Test Co",
        fiscal_years=fiscal_years,
        items=items,
    )


# ---- against real fixtures -----------------------------------------------------


def test_aapl_report_passes(aapl_statements: StatementSet) -> None:
    report = verify(aapl_statements)
    assert isinstance(report, TieoutReport)
    assert report.ticker == "AAPL"
    assert report.checks  # never empty
    assert report.passed is True
    assert all(check.passed for check in report.checks)


def test_msft_report_passes(msft_statements: StatementSet) -> None:
    report = verify(msft_statements)
    assert report.passed is True
    assert all(check.passed for check in report.checks)


def test_earliest_presented_year_has_no_roll_forward_check(
    aapl_statements: StatementSet, msft_statements: StatementSet
) -> None:
    """No hidden prior year exists (app.model.builder resolves exactly the
    presented years), so the earliest year can never get a cash roll-forward."""
    aapl_ids = {c.check_id for c in verify(aapl_statements).checks}
    assert "cash_roll_forward_2021" not in aapl_ids
    assert "cash_roll_forward_2022" in aapl_ids  # the next year does have one

    msft_ids = {c.check_id for c in verify(msft_statements).checks}
    assert "cash_roll_forward_2022" not in msft_ids
    assert "cash_roll_forward_2023" in msft_ids


def test_check_omitted_when_a_required_input_is_missing(msft_statements: StatementSet) -> None:
    """Microsoft has no `OperatingExpenses` fact for FY2022 -- the build-up check
    must be omitted for that year, not silently skipped-as-passed or forced."""
    ids = {c.check_id for c in verify(msft_statements).checks}
    assert "operating_income_buildup_2022" not in ids
    assert "operating_income_buildup_2023" in ids  # later years do have it


def test_optional_term_defaults_to_zero_not_omission() -> None:
    """noncontrolling_interest_in_income absent entirely must still let the
    check run, treating it as a true zero (see verify()'s docstring)."""
    st = _statement_set(
        {
            "pretax_income": {2024: 100.0},
            "income_tax_expense": {2024: 20.0},
            "net_income": {2024: 80.0},
        },
        [2024],
    )
    report = verify(st)
    check = next(c for c in report.checks if c.check_id == "net_income_buildup_2024")
    assert check.rhs == 80.0
    assert check.passed is True


# ---- synthetic edge cases -------------------------------------------------------


def test_balance_sheet_equation_fails_when_broken() -> None:
    st = _statement_set(
        {
            "total_assets": {2024: 100.0},
            "total_liabilities": {2024: 40.0},
            "total_equity": {2024: 50.0},  # 40 + 50 = 90 != 100
        },
        [2024],
    )
    report = verify(st)
    check = next(c for c in report.checks if c.check_id == "balance_sheet_equation_2024")
    assert check.passed is False
    assert check.lhs == 100.0
    assert check.rhs == 90.0
    assert report.passed is False  # one failing check fails the whole report


def test_check_within_tolerance_passes() -> None:
    st = _statement_set(
        {
            "total_assets": {2024: 100.000_4},
            "total_liabilities": {2024: 40.0},
            "total_equity": {2024: 60.0},
        },
        [2024],
    )
    check = next(c for c in verify(st).checks if c.check_id == "balance_sheet_equation_2024")
    assert check.passed is True  # within the $1 tolerance


def test_insufficient_data_year_fails_rather_than_vacuously_passing() -> None:
    """all([]) is True in Python -- a year with zero evaluable checks must not
    read as a clean tie-out (verify()'s docstring)."""
    st = _statement_set({}, [2024])
    report = verify(st)
    assert len(report.checks) == 1
    assert report.checks[0].check_id == "insufficient_data_2024"
    assert report.checks[0].passed is False
    assert report.passed is False


def test_missing_value_for_specific_year_omits_only_that_year() -> None:
    st = _statement_set(
        {
            "total_assets": {2023: 100.0, 2024: None},
            "total_liabilities": {2023: 40.0, 2024: 40.0},
            "total_equity": {2023: 60.0, 2024: 60.0},
        },
        [2023, 2024],
    )
    report = verify(st)
    ids = {c.check_id for c in report.checks}
    assert "balance_sheet_equation_2023" in ids
    assert "balance_sheet_equation_2024" not in ids
    assert "insufficient_data_2024" in ids  # 2024 had nothing else evaluable either


# ---- reconcile() ------------------------------------------------------------------


def test_reconcile_excludes_the_earliest_year(
    aapl_statements: StatementSet, msft_statements: StatementSet
) -> None:
    assert [r.fiscal_year for r in reconcile(aapl_statements)] == [2022, 2023, 2024, 2025]
    assert [r.fiscal_year for r in reconcile(msft_statements)] == [2023, 2024, 2025, 2026]


def test_reconcile_arithmetic_is_internally_consistent(aapl_statements: StatementSet) -> None:
    for r in reconcile(aapl_statements):
        assert isinstance(r, Reconciliation)
        assert r.ending_expected == pytest.approx(
            r.beginning + r.net_income - r.dividends - r.buybacks
        )
        assert r.residual == pytest.approx(r.ending_actual - r.ending_expected)
        if r.residual_pct_of_assets is not None:
            assert r.residual_pct_of_assets >= 0  # always a magnitude, sign lives in `residual`


def test_reconciliation_never_becomes_a_scored_check(aapl_statements: StatementSet) -> None:
    """Reconciliation is a separate, non-frozen-schema type by design (see
    verifier.py's module docstring) -- it must never leak into TieoutReport."""
    ids = " ".join(c.check_id for c in verify(aapl_statements).checks)
    assert "retained_earnings" not in ids
    assert "rollforward" not in ids


def test_reconcile_needs_a_prior_year_and_net_income() -> None:
    st = _statement_set({"retained_earnings": {2024: 100.0}}, [2024])
    assert reconcile(st) == []
