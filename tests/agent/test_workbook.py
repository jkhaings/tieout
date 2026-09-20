"""Tests for app.agent.workbook.write_commentary_sheet."""

from __future__ import annotations

from openpyxl import Workbook

from app.agent.workbook import write_commentary_sheet
from app.schemas import Chunk, Citation, Commentary


def _chunk(chunk_id: str, text: str, source_url: str = "https://example.com/filing.htm") -> Chunk:
    return Chunk(chunk_id=chunk_id, section="Item 7", text=text, source_url=source_url)


def test_skips_ungrounded_commentary_entirely() -> None:
    wb = Workbook()
    commentary = [Commentary(line_item_key="revenue", text=None, citations=[])]
    write_commentary_sheet(wb, commentary, labels_by_key={"revenue": "Revenue"}, chunks_by_id={})

    ws = wb["Commentary"]
    values = [row for row in ws.iter_rows(values_only=True) if any(row)]
    # Only the intro line and the header row -- no data row for the refusal.
    assert len(values) == 2


def test_writes_one_row_per_citation_with_source_resolved() -> None:
    wb = Workbook()
    chunk = _chunk("item7-0001", "Revenue grew due to strong iPhone sales.")
    commentary = [
        Commentary(
            line_item_key="revenue",
            text="Revenue grew, driven by iPhone sales.",
            citations=[Citation(chunk_id="item7-0001", quote="strong iPhone sales")],
        )
    ]
    write_commentary_sheet(
        wb, commentary, labels_by_key={"revenue": "Revenue"}, chunks_by_id={"item7-0001": chunk}
    )

    ws = wb["Commentary"]
    data_rows = [row for row in ws.iter_rows(min_row=4, values_only=True) if any(row)]
    assert len(data_rows) == 1
    label, text, quote, source = data_rows[0]
    assert label == "Revenue"
    assert text == "Revenue grew, driven by iPhone sales."
    assert quote == "strong iPhone sales"
    assert source == "https://example.com/filing.htm"


def test_multiple_citations_repeat_label_and_text_only_on_first_row() -> None:
    wb = Workbook()
    chunk_a = _chunk("item7-0001", "First point about revenue.")
    chunk_b = _chunk("item7-0002", "Second point about revenue.")
    commentary = [
        Commentary(
            line_item_key="revenue",
            text="Revenue commentary.",
            citations=[
                Citation(chunk_id="item7-0001", quote="First point"),
                Citation(chunk_id="item7-0002", quote="Second point"),
            ],
        )
    ]
    write_commentary_sheet(
        wb,
        commentary,
        labels_by_key={"revenue": "Revenue"},
        chunks_by_id={"item7-0001": chunk_a, "item7-0002": chunk_b},
    )

    ws = wb["Commentary"]
    data_rows = [row for row in ws.iter_rows(min_row=4, values_only=True) if any(row)]
    assert len(data_rows) == 2
    assert data_rows[0][0] == "Revenue" and data_rows[0][1] == "Revenue commentary."
    assert data_rows[1][0] is None and data_rows[1][1] is None
    assert data_rows[0][2] == "First point"
    assert data_rows[1][2] == "Second point"


def test_escapes_formula_injection_in_every_text_field() -> None:
    """SECURITY.md item 4: every text write must be escaped, no exceptions."""
    wb = Workbook()
    chunk = _chunk("item7-0001", "=EVIL(A1) is not a formula, just filing text.")
    commentary = [
        Commentary(
            line_item_key="revenue",
            text="=cmd|'/c calc'!A1",
            citations=[Citation(chunk_id="item7-0001", quote="=EVIL(A1) is not a formula")],
        )
    ]
    write_commentary_sheet(
        wb,
        commentary,
        labels_by_key={"revenue": "=Revenue"},
        chunks_by_id={"item7-0001": chunk},
    )

    ws = wb["Commentary"]
    label, text, quote, _source = next(
        row for row in ws.iter_rows(min_row=4, values_only=True) if any(row)
    )
    for value in (label, text, quote):
        assert value is not None
        assert value.startswith("'="), f"{value!r} was not escaped"


def test_missing_chunk_lookup_leaves_source_blank_without_raising() -> None:
    wb = Workbook()
    commentary = [
        Commentary(
            line_item_key="revenue",
            text="Some commentary.",
            citations=[Citation(chunk_id="unknown-0001", quote="quoted text here")],
        )
    ]
    write_commentary_sheet(wb, commentary, labels_by_key={"revenue": "Revenue"}, chunks_by_id={})

    ws = wb["Commentary"]
    row = next(r for r in ws.iter_rows(min_row=4, values_only=True) if any(r))
    assert row[3] is None
