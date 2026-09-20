"""Excel workbook generation: three statement tabs, live-formula ratios, a tie-out tab.

Every text write goes through `escape_cell()` (SECURITY.md item 4) --
including company names and check descriptions, since both ultimately
originate from a public SEC filing another party controls. Ratio cells are
always Excel formulas referencing statement cells
(`='Income Statement'!B5/'Income Statement'!B4`), never numbers computed in
Python and pasted in, so opening the workbook and clicking a ratio shows its
derivation, and the formula still recalculates if a cell is ever edited.

A line item with no reported value is written as the text `"n/r"`, not left
blank. Excel treats a genuinely blank cell as `0` inside arithmetic, which
would let a ratio formula silently compute a plausible-looking but wrong
number from missing data; text in the cell instead surfaces as `#VALUE!`
in any ratio that depends on it -- a visible failure, matching CLAUDE.md
rule 4 (fail closed) instead of a silent one.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell.cell import Cell
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from app.model.verifier import Reconciliation
from app.schemas import Statement, StatementSet, TieoutReport

_DANGEROUS_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

_LABEL_COL = 1
_FIRST_YEAR_COL = 2
_HEADER_ROW = 3

_MONEY_FORMAT = "#,##0;(#,##0)"
_EPS_FORMAT = "#,##0.00;(#,##0.00)"
_PERCENT_FORMAT = "0.0%"
_RATIO_FORMAT = "0.00"

CellRegistry = dict[tuple[str, int], str]


def escape_cell(value: str) -> str:
    """Neutralize formula injection: prefix a leading dangerous character with `'`.

    Excel (and every major spreadsheet app) treats a cell beginning with
    `=`, `+`, `-`, `@`, a tab, or a carriage return as a formula. A leading
    `'` forces text interpretation without changing what a user sees. Apply
    to every string written to a cell, no exceptions (SECURITY.md item 4).
    """
    if value and value[0] in _DANGEROUS_PREFIXES:
        return "'" + value
    return value


def _quote_sheet(title: str) -> str:
    """Quote a sheet name for use in a formula reference, if it needs it."""
    return f"'{title}'" if any(c in title for c in " -") else title


def _plain_cell(ws: Worksheet, row: int, column: int) -> Cell:
    """`ws.cell(row, column)` narrowed from `Cell | MergedCell` to `Cell`.

    This workbook never calls `merge_cells`, so a `MergedCell` -- which the
    stub return type allows for in general -- can never actually occur
    here; asserting that lets callers set `.value`/`.font`/`.number_format`
    without every such call site repeating a `# type: ignore`.
    """
    cell = ws.cell(row=row, column=column)
    assert isinstance(cell, Cell)
    return cell


def _write_statement_sheet(
    ws: Worksheet,
    statements: StatementSet,
    statement: Statement,
    registry: CellRegistry,
) -> None:
    """Write one statement tab (Income/Balance/Cash Flow) and register every cell address."""
    ws.cell(
        row=1, column=1, value=escape_cell(f"{statements.company_name} ({statements.ticker})")
    ).font = Font(bold=True, size=13)
    ws.cell(
        row=2, column=1, value="All amounts in U.S. dollars except per-share data."
    ).font = Font(italic=True, size=9)

    ws.cell(row=_HEADER_ROW, column=_LABEL_COL, value="Line item").font = Font(bold=True)
    for i, fy in enumerate(statements.fiscal_years):
        header_cell = ws.cell(row=_HEADER_ROW, column=_FIRST_YEAR_COL + i, value=f"FY{fy}")
        header_cell.font = Font(bold=True)
        header_cell.alignment = Alignment(horizontal="right")

    row = _HEADER_ROW + 1
    quoted_title = _quote_sheet(ws.title)
    for item in statements.items:
        if item.statement != statement:
            continue
        ws.cell(row=row, column=_LABEL_COL, value=escape_cell(item.label))
        number_format = _EPS_FORMAT if item.key == "eps_diluted" else _MONEY_FORMAT
        for i, fy in enumerate(statements.fiscal_years):
            col = _FIRST_YEAR_COL + i
            cell = _plain_cell(ws, row, col)
            value = item.values.get(fy)
            if value is None:
                cell.value = "n/r"
                cell.font = Font(italic=True, color="999999")
                cell.alignment = Alignment(horizontal="right")
            else:
                cell.value = value
                cell.number_format = number_format
            registry[(item.key, fy)] = f"{quoted_title}!{get_column_letter(col)}{row}"
        row += 1


def _write_ratios_sheet(ws: Worksheet, statements: StatementSet, registry: CellRegistry) -> None:
    """Write the Ratios tab. Every populated cell is a formula, never a pasted value."""
    ws.cell(
        row=1,
        column=1,
        value=escape_cell(f"{statements.company_name} ({statements.ticker}) — Ratios"),
    ).font = Font(bold=True, size=13)
    ws.cell(
        row=2, column=1, value="Every cell below is a live formula referencing the statement tabs."
    ).font = Font(italic=True, size=9)

    ws.cell(row=_HEADER_ROW, column=_LABEL_COL, value="Ratio").font = Font(bold=True)
    for i, fy in enumerate(statements.fiscal_years):
        ws.cell(row=_HEADER_ROW, column=_FIRST_YEAR_COL + i, value=f"FY{fy}").font = Font(bold=True)

    row = _HEADER_ROW + 1
    # Same-sheet refs, for ratios built from other ratio rows (e.g. FCF margin).
    own_cells: CellRegistry = {}

    def ref(key: str, fy: int) -> str | None:
        """The registered statement-tab cell address for `key`/`fy`, if any."""
        return registry.get((key, fy))

    def safe_div(key_num: str, key_den: str, fy: int) -> str | None:
        """A `=numerator/denominator` formula, or `None` if either side is unavailable."""
        numerator, denominator = ref(key_num, fy), ref(key_den, fy)
        return f"={numerator}/{denominator}" if numerator and denominator else None

    def write_row(
        label: str, key: str, number_format: str, formula_for_year: Callable[[int], str | None]
    ) -> None:
        """Write one ratio row, one formula cell per fiscal year, and register its own cells."""
        nonlocal row
        ws.cell(row=row, column=_LABEL_COL, value=escape_cell(label))
        for i, fy in enumerate(statements.fiscal_years):
            col = _FIRST_YEAR_COL + i
            formula = formula_for_year(fy)
            cell = _plain_cell(ws, row, col)
            if formula is not None:
                cell.value = formula
                cell.number_format = number_format
            own_cells[(key, fy)] = f"{get_column_letter(col)}{row}"
        row += 1

    def quick_ratio(fy: int) -> str | None:
        """`(current assets - inventory) / current liabilities`."""
        current_assets, inventory, current_liabilities = (
            ref("total_current_assets", fy),
            ref("inventory", fy),
            ref("total_current_liabilities", fy),
        )
        if current_assets and inventory and current_liabilities:
            return f"=({current_assets}-{inventory})/{current_liabilities}"
        return None

    def fcf(fy: int) -> str | None:
        """Free cash flow: `CFO - capital expenditures`."""
        cfo, capex = ref("cfo", fy), ref("capex", fy)
        return f"={cfo}-{capex}" if cfo and capex else None

    def fcf_margin(fy: int) -> str | None:
        """Free cash flow as a fraction of revenue; references this sheet's own FCF row."""
        fcf_cell, revenue_cell = own_cells.get(("fcf", fy)), ref("revenue", fy)
        return f"={fcf_cell}/{revenue_cell}" if fcf_cell and revenue_cell else None

    def yoy_revenue_growth(fy: int) -> str | None:
        """Year-over-year revenue growth; `None` for the first presented year (no prior column)."""
        idx = statements.fiscal_years.index(fy)
        if idx == 0:
            return None
        prior_fy = statements.fiscal_years[idx - 1]
        current, prior = ref("revenue", fy), ref("revenue", prior_fy)
        return f"=({current}-{prior})/{prior}" if current and prior else None

    write_row(
        "Gross margin",
        "gross_margin",
        _PERCENT_FORMAT,
        lambda fy: safe_div("gross_profit", "revenue", fy),
    )
    write_row(
        "Operating margin",
        "operating_margin",
        _PERCENT_FORMAT,
        lambda fy: safe_div("operating_income", "revenue", fy),
    )
    write_row(
        "Net margin",
        "net_margin",
        _PERCENT_FORMAT,
        lambda fy: safe_div("net_income", "revenue", fy),
    )
    write_row(
        "Return on assets",
        "roa",
        _PERCENT_FORMAT,
        lambda fy: safe_div("net_income", "total_assets", fy),
    )
    write_row(
        "Return on equity",
        "roe",
        _PERCENT_FORMAT,
        lambda fy: safe_div("net_income", "total_equity", fy),
    )
    write_row(
        "Current ratio",
        "current_ratio",
        _RATIO_FORMAT,
        lambda fy: safe_div("total_current_assets", "total_current_liabilities", fy),
    )
    write_row("Quick ratio", "quick_ratio", _RATIO_FORMAT, quick_ratio)
    write_row(
        "Asset turnover",
        "asset_turnover",
        _RATIO_FORMAT,
        lambda fy: safe_div("revenue", "total_assets", fy),
    )
    write_row(
        "Debt-to-equity",
        "debt_to_equity",
        _RATIO_FORMAT,
        lambda fy: safe_div("total_liabilities", "total_equity", fy),
    )
    write_row("Free cash flow", "fcf", _MONEY_FORMAT, fcf)
    write_row("Free cash flow margin", "fcf_margin", _PERCENT_FORMAT, fcf_margin)
    write_row("Revenue growth (YoY)", "revenue_growth", _PERCENT_FORMAT, yoy_revenue_growth)


def _write_tieout_sheet(
    ws: Worksheet,
    statements: StatementSet,
    tieout: TieoutReport,
    reconciliations: list[Reconciliation],
) -> None:
    """Write the Tie-out tab: scored checks, then informational reconciliations."""
    ws.cell(
        row=1,
        column=1,
        value=escape_cell(f"{statements.company_name} ({statements.ticker}) — Tie-out"),
    ).font = Font(bold=True, size=13)

    row = 3
    ws.cell(row=row, column=1, value="Scored checks").font = Font(bold=True, size=11)
    row += 1
    for col, header in enumerate(
        [
            "Check",
            "Fiscal year",
            "Description",
            "Left side",
            "Right side",
            "Difference",
            "Tolerance",
            "Result",
        ],
        start=1,
    ):
        ws.cell(row=row, column=col, value=header).font = Font(bold=True)
    row += 1
    for check in tieout.checks:
        diff = None if check.lhs is None or check.rhs is None else check.lhs - check.rhs
        ws.cell(row=row, column=1, value=escape_cell(check.check_id))
        ws.cell(row=row, column=2, value=check.fiscal_year)
        ws.cell(row=row, column=3, value=escape_cell(check.description))
        if check.lhs is not None:
            ws.cell(row=row, column=4, value=check.lhs).number_format = _MONEY_FORMAT
        if check.rhs is not None:
            ws.cell(row=row, column=5, value=check.rhs).number_format = _MONEY_FORMAT
        if diff is not None:
            ws.cell(row=row, column=6, value=diff).number_format = _MONEY_FORMAT
        ws.cell(row=row, column=7, value=check.tolerance)
        result_cell = ws.cell(row=row, column=8, value="PASS" if check.passed else "FAIL")
        result_cell.font = Font(bold=True, color="1A7F37" if check.passed else "C0362C")
        row += 1

    row += 1
    ws.cell(row=row, column=1, value="Reconciliations (not scored)").font = Font(bold=True, size=11)
    row += 1
    note = (
        "Retained-earnings roll-forward does not tie exactly from XBRL data alone "
        "(buybacks charged to retained earnings are not consistently tagged); shown "
        "here as a named residual, not a pass/fail check -- see app.model.verifier.reconcile."
    )
    ws.cell(row=row, column=1, value=escape_cell(note)).font = Font(italic=True, size=9)
    row += 1
    for col, header in enumerate(
        [
            "Item",
            "Fiscal year",
            "Beginning RE",
            "Net income",
            "Dividends",
            "Buybacks",
            "Ending RE (actual)",
            "Ending RE (expected)",
            "Residual",
            "Residual % of assets",
        ],
        start=1,
    ):
        ws.cell(row=row, column=col, value=header).font = Font(bold=True)
    row += 1
    for recon in reconciliations:
        ws.cell(row=row, column=1, value=escape_cell(recon.label))
        ws.cell(row=row, column=2, value=recon.fiscal_year)
        money_values = (
            recon.beginning,
            recon.net_income,
            recon.dividends,
            recon.buybacks,
            recon.ending_actual,
            recon.ending_expected,
            recon.residual,
        )
        for col, value in enumerate(money_values, start=3):
            ws.cell(row=row, column=col, value=value).number_format = _MONEY_FORMAT
        if recon.residual_pct_of_assets is not None:
            ws.cell(
                row=row, column=10, value=recon.residual_pct_of_assets / 100
            ).number_format = _PERCENT_FORMAT
        row += 1


def build_workbook(
    statements: StatementSet,
    tieout: TieoutReport,
    reconciliations: list[Reconciliation],
) -> Workbook:
    """Build the full workbook in memory: three statement tabs, Ratios, Tie-out.

    Pure and I/O-free so tests can inspect the result directly; `write_workbook`
    below is the thin convenience wrapper that saves it to disk.
    """
    workbook = Workbook()
    registry: CellRegistry = {}

    income_sheet = workbook.active
    assert income_sheet is not None
    income_sheet.title = "Income Statement"
    _write_statement_sheet(income_sheet, statements, Statement.INCOME, registry)

    balance_sheet = workbook.create_sheet("Balance Sheet")
    _write_statement_sheet(balance_sheet, statements, Statement.BALANCE, registry)

    cashflow_sheet = workbook.create_sheet("Cash Flow")
    _write_statement_sheet(cashflow_sheet, statements, Statement.CASHFLOW, registry)

    ratios_sheet = workbook.create_sheet("Ratios")
    _write_ratios_sheet(ratios_sheet, statements, registry)

    tieout_sheet = workbook.create_sheet("Tie-out")
    _write_tieout_sheet(tieout_sheet, statements, tieout, reconciliations)

    for sheet in (income_sheet, balance_sheet, cashflow_sheet, ratios_sheet, tieout_sheet):
        sheet.column_dimensions["A"].width = 44
        for i in range(len(statements.fiscal_years)):
            sheet.column_dimensions[get_column_letter(_FIRST_YEAR_COL + i)].width = 16

    return workbook


def write_workbook(
    statements: StatementSet,
    tieout: TieoutReport,
    reconciliations: list[Reconciliation],
    path: Path,
) -> None:
    """Build the workbook and save it to `path`, creating parent directories as needed."""
    workbook = build_workbook(statements, tieout, reconciliations)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
