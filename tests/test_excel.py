"""Hermetic tests for app.model.excel -- workbooks are written under tmp_path, never
touching the real filesystem outside pytest's sandbox.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from openpyxl import load_workbook

from app.model.excel import build_workbook, escape_cell, write_workbook
from app.model.verifier import reconcile, verify
from app.schemas import LineItem, Statement, StatementSet, TieoutCheck, TieoutReport

_DANGEROUS_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


# ---- escape_cell -----------------------------------------------------------------


@pytest.mark.parametrize("prefix", _DANGEROUS_PREFIXES)
def test_escape_cell_prefixes_dangerous_leading_characters(prefix: str) -> None:
    value = f'{prefix}HYPERLINK("https://evil.example/","clickme")'
    escaped = escape_cell(value)
    assert escaped == f"'{value}"


@pytest.mark.parametrize(
    "value", ["Apple Inc.", "Revenue", "Net income (loss)", "", "a=b", "100% grounded"]
)
def test_escape_cell_leaves_safe_strings_untouched(value: str) -> None:
    assert escape_cell(value) == value


def test_escaped_string_stays_a_string_cell_not_a_formula() -> None:
    """The actual mechanism escape_cell relies on: openpyxl converts an unescaped
    leading '=' into a live formula cell (data_type 'f'); escaping keeps it a
    plain string (data_type 's'), confirmed empirically before writing this test."""
    from openpyxl import Workbook

    hostile = '=HYPERLINK("https://evil.example/","clickme")'
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws["A1"] = hostile
    ws["A2"] = escape_cell(hostile)
    assert ws["A1"].data_type == "f"
    assert ws["A2"].data_type == "s"
    assert ws["A2"].value == "'" + hostile


# ---- build_workbook against real fixtures ------------------------------------------


def test_sheet_names_and_order(aapl_statements: StatementSet) -> None:
    report = verify(aapl_statements)
    recons = reconcile(aapl_statements)
    workbook = build_workbook(aapl_statements, report, recons)
    assert workbook.sheetnames == [
        "Income Statement",
        "Balance Sheet",
        "Cash Flow",
        "Ratios",
        "Tie-out",
    ]
    for name in workbook.sheetnames:
        assert len(name) <= 31  # Excel's sheet-name length limit
        assert not any(c in name for c in "[]:*?/\\")


def test_statement_sheets_have_one_row_per_matching_line_item(
    aapl_statements: StatementSet,
) -> None:
    report = verify(aapl_statements)
    recons = reconcile(aapl_statements)
    workbook = build_workbook(aapl_statements, report, recons)

    expected_income_items = sum(
        1 for item in aapl_statements.items if item.statement == Statement.INCOME
    )
    income_sheet = workbook["Income Statement"]
    label_column = [
        income_sheet.cell(row=r, column=1).value for r in range(4, income_sheet.max_row + 1)
    ]
    non_empty_labels = [v for v in label_column if v]
    assert len(non_empty_labels) == expected_income_items


def test_ratio_cells_are_formulas_not_pasted_numbers(aapl_statements: StatementSet) -> None:
    report = verify(aapl_statements)
    recons = reconcile(aapl_statements)
    workbook = build_workbook(aapl_statements, report, recons)
    ratios_sheet = workbook["Ratios"]

    found_formula = False
    for row in ratios_sheet.iter_rows(min_row=4):
        for cell in row:
            if cell.column == 1 or cell.value in (None, ""):
                continue
            assert isinstance(cell.value, str) and cell.value.startswith("="), (
                f"Ratios!{cell.coordinate} is {cell.value!r}, not a formula"
            )
            found_formula = True
    assert found_formula


def test_gross_margin_formula_references_income_statement(aapl_statements: StatementSet) -> None:
    report = verify(aapl_statements)
    recons = reconcile(aapl_statements)
    workbook = build_workbook(aapl_statements, report, recons)
    ratios_sheet = workbook["Ratios"]

    gross_margin_row = next(
        r
        for r in range(4, ratios_sheet.max_row + 1)
        if ratios_sheet.cell(row=r, column=1).value == "Gross margin"
    )
    formula = ratios_sheet.cell(row=gross_margin_row, column=2).value
    assert isinstance(formula, str)
    assert formula.startswith("=")
    assert "'Income Statement'!" in formula


def test_first_year_has_no_revenue_growth_formula(aapl_statements: StatementSet) -> None:
    """There is no prior presented year for the first column to reference."""
    report = verify(aapl_statements)
    recons = reconcile(aapl_statements)
    workbook = build_workbook(aapl_statements, report, recons)
    ratios_sheet = workbook["Ratios"]

    growth_row = next(
        r
        for r in range(4, ratios_sheet.max_row + 1)
        if ratios_sheet.cell(row=r, column=1).value == "Revenue growth (YoY)"
    )
    assert ratios_sheet.cell(row=growth_row, column=2).value in (None, "")
    assert ratios_sheet.cell(row=growth_row, column=3).value is not None


def test_tieout_sheet_lists_every_check_and_reconciliation(aapl_statements: StatementSet) -> None:
    report = verify(aapl_statements)
    recons = reconcile(aapl_statements)
    workbook = build_workbook(aapl_statements, report, recons)
    tieout_sheet = workbook["Tie-out"]

    all_values = [cell.value for row in tieout_sheet.iter_rows() for cell in row]
    for check in report.checks:
        assert check.check_id in all_values
    assert "PASS" in all_values  # every AAPL check passes
    assert "Reconciliations (not scored)" in all_values
    for recon in recons:
        assert recon.label in all_values


def test_failed_check_renders_fail_in_red(aapl_statements: StatementSet) -> None:
    broken_report = TieoutReport(
        ticker="AAPL",
        checks=[
            TieoutCheck(
                check_id="balance_sheet_equation_2024",
                description="Total assets = total liabilities + total stockholders' equity",
                fiscal_year=2024,
                passed=False,
                lhs=100.0,
                rhs=90.0,
                tolerance=1.0,
            )
        ],
    )
    workbook = build_workbook(aapl_statements, broken_report, [])
    tieout_sheet = workbook["Tie-out"]
    result_cell = next(
        cell for row in tieout_sheet.iter_rows() for cell in row if cell.value in ("PASS", "FAIL")
    )
    assert result_cell.value == "FAIL"
    assert result_cell.font.color.rgb.endswith("C0362C")


# ---- injection safety end-to-end, on a hostile synthetic StatementSet ----------------


def _hostile_statement_set() -> StatementSet:
    return StatementSet(
        ticker="EVIL",
        cik="0000000002",
        company_name='=HYPERLINK("https://evil.example/","clickme")',
        fiscal_years=[2024],
        items=[
            LineItem(
                key="total_assets",
                label="+cmd|' /C calc'!A1",
                statement=Statement.BALANCE,
                values={2024: 100.0},
            )
        ],
    )


def test_hostile_company_name_and_label_are_neutralized_end_to_end(tmp_path: Path) -> None:
    statements = _hostile_statement_set()
    report = TieoutReport(
        ticker="EVIL",
        checks=[
            TieoutCheck(
                check_id="=1+1",  # even a check_id is untrusted-shaped here
                description="@SUM(1,2)",
                fiscal_year=2024,
                passed=True,
            )
        ],
    )
    path = tmp_path / "evil.xlsx"
    write_workbook(statements, report, [], path)

    reloaded = load_workbook(path)

    balance_sheet = reloaded["Balance Sheet"]
    title_cell = balance_sheet.cell(row=1, column=1)
    assert title_cell.data_type == "s"
    assert isinstance(title_cell.value, str) and title_cell.value.startswith("'=HYPERLINK")

    label_cell = balance_sheet.cell(row=4, column=1)
    assert label_cell.data_type == "s"
    assert isinstance(label_cell.value, str) and label_cell.value.startswith("'+cmd")

    tieout_sheet = reloaded["Tie-out"]
    check_cells = [
        cell for row in tieout_sheet.iter_rows() for cell in row if cell.value == "'=1+1"
    ]
    assert check_cells, "escaped check_id not found as a plain string cell"
    assert check_cells[0].data_type == "s"


def test_write_workbook_creates_parent_directories(tmp_path: Path) -> None:
    statements = _hostile_statement_set()
    report = TieoutReport(ticker="EVIL", checks=[])
    nested_path = tmp_path / "nested" / "dir" / "model.xlsx"
    write_workbook(statements, report, [], nested_path)
    assert nested_path.exists()
