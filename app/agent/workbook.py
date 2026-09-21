"""Append the Commentary sheet onto a workbook built by `app.model.excel`.

`app/model/excel.py`'s `build_workbook()` predates ARCHITECTURE.md's
"Commentary column with citations" promise and has no `Commentary`
parameter at all -- `app/model` is a frozen, data-engineer-owned lane for
this session, so the feature is added here instead, as a sheet appended
after the fact rather than a column threaded through the statement tabs.

Every attempted line item gets exactly one row, whether or not it was
narrated. An earlier version skipped refusals entirely, on the reading that
CLAUDE.md rule 4 means "say nothing rather than something ungrounded" -- but
a run where nothing was narrated then shipped a sheet holding only its
headers, which is not a fail-closed signal, it is an absent one. A refusal
now renders as "no grounded commentary available: <reason>", where the
reason is a deterministic fact about the run recorded by `app.agent.graph`,
never anything the model produced.
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
_REFUSAL_PREFIX = "no grounded commentary available"
_REFUSAL_FONT = Font(italic=True, color="999999")
_DEFAULT_REASON = "narration produced no grounded text for this line item"


def _write_row(
    ws: Worksheet,
    row: int,
    *,
    label: str,
    text: str,
    quote: str,
    source: str,
    refused: bool = False,
) -> None:
    """Write one Commentary-sheet row, escaping every text cell."""
    for col, value in enumerate((label, text, quote, source), start=1):
        cell = ws.cell(row=row, column=col, value=escape_cell(value) if value else None)
        cell.alignment = _WRAP
        if refused and col == 2:
            cell.font = _REFUSAL_FONT


def write_commentary_sheet(
    workbook: Workbook,
    commentary: list[Commentary],
    *,
    labels_by_key: dict[str, str],
    chunks_by_id: dict[str, Chunk],
    refusal_reasons: dict[str, str] | None = None,
) -> None:
    """Append a `Commentary` sheet: at least one row per attempted line item.

    Args:
        workbook: The workbook already built by `app.model.excel.build_workbook`.
        commentary: Narration results, one per attempted line item. An entry
            with `text is None` is a refusal and renders as an explicit row
            saying so -- never omitted, which would make a wholly-unnarrated
            run indistinguishable from a sheet that was never written.
        labels_by_key: Display label for each line item key, e.g.
            `{"revenue": "Revenue"}` (from `StatementSet.items`).
        chunks_by_id: Every chunk offered to narration, keyed by chunk id,
            used to resolve each citation's source filing URL.
        refusal_reasons: Why each refused line item was refused, keyed by line
            item key, from `app.agent.graph`. Deterministic facts about the
            run; never model output. A missing entry falls back to a generic
            reason rather than an empty explanation.
    """
    refusal_reasons = refusal_reasons or {}
    ws = workbook.create_sheet("Commentary")
    ws.cell(
        row=1, column=1, value="Grounded commentary, cited to the filing where available."
    ).font = Font(italic=True, size=9)
    for col, header in enumerate(_HEADERS, start=1):
        ws.cell(row=3, column=col, value=header).font = Font(bold=True)

    row = 4
    if not commentary:
        # Narration never ran at all. Saying so is the whole point of the sheet.
        _write_row(
            ws,
            row,
            label="(all line items)",
            text=f"{_REFUSAL_PREFIX}: narration did not run for this workbook",
            quote="",
            source="",
            refused=True,
        )
        row += 1
    for item in commentary:
        label = labels_by_key.get(item.line_item_key, item.line_item_key)
        if not item.text:
            reason = refusal_reasons.get(item.line_item_key, _DEFAULT_REASON)
            _write_row(
                ws,
                row,
                label=label,
                text=f"{_REFUSAL_PREFIX}: {reason}",
                quote="",
                source="",
                refused=True,
            )
            row += 1
            continue
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
