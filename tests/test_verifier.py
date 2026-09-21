"""Hermetic tests for app.model.verifier -- real fixtures plus synthetic edge cases."""

from __future__ import annotations

import pytest

from app.edgar.tags import BY_KEY, DERIVED_TOTAL_LIABILITIES
from app.model.verifier import Reconciliation, reconcile, verify
from app.schemas import LineItem, StatementSet, TieoutReport


def _statement_set(
    values_by_key: dict[str, dict[int, float | None]],
    fiscal_years: list[int],
    tags_by_key: dict[str, list[str]] | None = None,
) -> StatementSet:
    tags_by_key = tags_by_key or {}
    items = [
        LineItem(
            key=key,
            label=key,
            statement=BY_KEY[key].statement,
            values=values,
            xbrl_tags=tags_by_key.get(key, []),
        )
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
    assert check.tolerance == 3.0  # small synthetic values infer no reporting scale
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
    assert check.passed is True  # diff 0.0004, well inside the inferred tolerance
    assert check.tolerance == 3.0  # three whole-dollar terms, $1 rounding unit each


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


# ---- Production workbook audit: tolerances inferred from filed rounding ------
#
# MCD's cash roll-forward FAILed on a $1,000,000 difference against a $1
# tolerance. The filer presents in millions, so each of the six summed terms
# carries up to $1,000,000 of rounding -- expected noise, not a broken
# statement.


def test_mcd_cash_roll_forward_passes_with_an_inferred_tolerance(
    mcd_statements: StatementSet,
) -> None:
    report = verify(mcd_statements)
    for fy in (2024, 2025):
        check = next(c for c in report.checks if c.check_id == f"cash_roll_forward_{fy}")
        assert check.lhs is not None and check.rhs is not None
        assert abs(check.lhs - check.rhs) == 1_000_000.0  # the difference the audit flagged
        assert check.passed is True
        assert check.tolerance == 6_000_000.0  # six terms, each filed to the nearest $1M
        assert "nearest $1,000,000" in check.description


@pytest.mark.parametrize("fy", [2024, 2025])
def test_an_injected_material_discrepancy_still_fails(
    mcd_statements: StatementSet, fy: int
) -> None:
    """The tolerance must widen for rounding, not for real breaks: $50M is 8x it."""
    broken = mcd_statements.model_copy(
        update={
            "items": [
                item.model_copy(update={"values": {**item.values, fy: item.values[fy] + 50e6}})
                if item.key == "cfo"
                else item
                for item in mcd_statements.items
            ]
        }
    )
    check = next(c for c in verify(broken).checks if c.check_id == f"cash_roll_forward_{fy}")
    assert check.passed is False


def test_mixed_precision_terms_sum_their_own_rounding_units(
    mcd_statements: StatementSet,
) -> None:
    """MCD FY2022 files Assets to the nearest $1M but L&SE to the nearest
    $100k, and they differ by $400,000. Taking the *finest* unit across terms
    would allow only $200,000 and false-FAIL it: rounding noise is bounded by
    the coarsest rounding applied, never the finest."""
    check = next(
        c
        for c in verify(mcd_statements).checks
        if c.check_id == "balance_sheet_equation_independent_2022"
    )
    assert check.lhs is not None and check.rhs is not None
    assert abs(check.lhs - check.rhs) == 400_000.0
    assert check.tolerance == 1_100_000.0  # $1,000,000 + $100,000
    assert check.passed is True


def test_fixture_checks_tie_exactly_not_merely_within_tolerance(
    aapl_statements: StatementSet,
    msft_statements: StatementSet,
    meta_statements: StatementSet,
) -> None:
    """A wider inferred tolerance must never become cover for a real regression.

    Every scored check on these filers ties to the dollar today; this pins
    that, independently of whatever tolerance the inference produces.
    """
    for statements in (aapl_statements, msft_statements, meta_statements):
        for check in verify(statements).checks:
            if check.lhs is None or check.rhs is None:
                continue
            assert check.lhs == pytest.approx(check.rhs, abs=1.0), (
                f"{statements.ticker} {check.check_id} no longer ties exactly"
            )


def test_absent_optional_term_does_not_inflate_the_tolerance() -> None:
    """An unreported FX effect contributes no rounding noise, so it must not
    add a term -- five reported terms, not six."""
    st = _statement_set(
        {
            "cash_and_equivalents": {2023: 1_000_000.0, 2024: 2_000_000.0},
            "cfo": {2024: 3_000_000.0},
            "cfi": {2024: -1_000_000.0},
            "cff": {2024: -1_000_000.0},
        },
        [2023, 2024],
    )
    check = next(c for c in verify(st).checks if c.check_id == "cash_roll_forward_2024")
    assert check.tolerance == 5_000_000.0


def test_tolerance_never_falls_below_one_dollar() -> None:
    st = _statement_set(
        {
            "total_assets": {2024: 1.5},
            "total_liabilities": {2024: 0.5},
            "total_equity": {2024: 1.0},
        },
        [2024],
    )
    check = next(c for c in verify(st).checks if c.check_id == "balance_sheet_equation_2024")
    assert check.tolerance == 3.0


# ---- Production workbook audit: derived liabilities, honest scoring ----------


def test_mcd_scores_a_balance_sheet_check_in_every_presented_year(
    mcd_statements: StatementSet,
) -> None:
    """MCD previously got zero balance-sheet checks across all five years,
    because it files no `us-gaap:Liabilities`."""
    ids = {c.check_id for c in verify(mcd_statements).checks}
    for fy in mcd_statements.fiscal_years:
        assert f"balance_sheet_equation_independent_{fy}" in ids
        assert f"balance_sheet_equation_{fy}" not in ids  # never the circular form


def test_mcd_scores_net_income_buildup_in_every_presented_year(
    mcd_statements: StatementSet,
) -> None:
    """Derived pretax income is immediately re-checked by the identity it enables."""
    report = verify(mcd_statements)
    for fy in mcd_statements.fiscal_years:
        check = next(c for c in report.checks if c.check_id == f"net_income_buildup_{fy}")
        assert check.passed is True
    assert not any(c.check_id.startswith("insufficient_data") for c in report.checks)


def test_filed_liabilities_keeps_the_three_term_check(meta_statements: StatementSet) -> None:
    """META files `Liabilities`, so nothing is derived and the stronger
    three-fact identity still runs."""
    ids = {c.check_id for c in verify(meta_statements).checks}
    assert "balance_sheet_equation_2025" in ids
    assert "balance_sheet_equation_independent_2025" not in ids


def test_circular_balance_sheet_check_is_never_scored() -> None:
    """With liabilities derived, `assets = liabilities + equity` reduces to
    `assets = L&SE` and cannot fail. Rig the three-term identity to pass while
    the two filed facts disagree: the report must still fail."""
    st = _statement_set(
        {
            "total_assets": {2024: 100.0},
            "total_liabilities": {2024: 40.0},
            "total_equity": {2024: 60.0},
            "total_liabilities_and_equity": {2024: 900.0},
        },
        [2024],
        tags_by_key={"total_liabilities": [DERIVED_TOTAL_LIABILITIES]},
    )
    report = verify(st)
    assert "balance_sheet_equation_2024" not in {c.check_id for c in report.checks}
    assert report.passed is False
