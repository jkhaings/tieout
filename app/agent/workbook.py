"""Append the Commentary sheet onto a workbook built by `app.model.excel`.

`app/model/excel.py`'s `build_workbook()` predates ARCHITECTURE.md's
"Commentary column with citations" promise and has no `Commentary`
parameter at all -- `app/model` is a frozen, data-engineer-owned lane for
this session, so the feature is added here instead, as a sheet appended
after the fact rather than a column threaded through the statement tabs.
Every text write goes through `app.model.excel.escape_cell` (SECURITY.md
item 4): commentary text and citation quotes both ultimately originate in a
public SEC filing another party controls -- exactly the case that control
exists for.
"""

from __future__ import annotations

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.worksheet import Worksheet

from app.model.excel import escape_cell
from app.schemas import Chunk, Commentary

_HEADERS = ("Line item", "Commentary", "Cited quote", "Source")
_WRAP = Alignment(wrap_text=True, vertical="top")


def _write_row(ws: Worksheet, row: int, *, label: str, text: str, quote: str, source: str) -> None:
    """Write one Commentary-sheet row, escaping every text cell."""
    for col, value in enumerate((label, text, quote, source), start=1):
        cell = ws.cell(row=row, column=col, value=escape_cell(value) if value else None)
        cell.alignment = _WRAP


def write_commentary_sheet(
    workbook: Workbook,
    commentary: list[Commentary],
    *,
    labels_by_key: dict[str, str],
    chunks_by_id: dict[str, Chunk],
) -> None:
    """Append a `Commentary` sheet: one row per citation, grouped by line item.

    Args:
        workbook: The workbook already built by `app.model.excel.build_workbook`.
        commentary: Narration results, one per attempted line item. Entries
            with `text is None` (no grounded commentary -- CLAUDE.md rule 4)
            are omitted entirely rather than shown as an empty row.
        labels_by_key: Display label for each line item key, e.g.
            `{"revenue": "Revenue"}` (from `StatementSet.items`).
        chunks_by_id: Every chunk offered to narration, keyed by chunk id,
            used to resolve each citation's source filing URL.
    """
    ws = workbook.create_sheet("Commentary")
    ws.cell(
        row=1, column=1, value="Grounded commentary, cited to the filing where available."
    ).font = Font(italic=True, size=9)
    for col, header in enumerate(_HEADERS, start=1):
        ws.cell(row=3, column=col, value=header).font = Font(bold=True)

    row = 4
    for item in commentary:
        if not item.text:
            continue
        label = labels_by_key.get(item.line_item_key, item.line_item_key)
        if not item.citations:
            _write_row(ws, row, label=label, text=item.text, quote="", source="")
            row += 1
            continue
        for index, citation in enumerate(item.citations):
            chunk = chunks_by_id.get(citation.chunk_id)
            _write_row(
                ws,
                row,
                label=label if index == 0 else "",
                text=item.text if index == 0 else "",
                quote=citation.quote,
                source=chunk.source_url if chunk else "",
            )
            row += 1

    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 60
    ws.column_dimensions["C"].width = 60
    ws.column_dimensions["D"].width = 50
