"""Shared data contracts for tieout.

FROZEN during parallel sessions (see CLAUDE.md ownership map). Every module
codes against these models; changing them requires coordinating all sessions.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Statement(StrEnum):
    """The three financial statements."""

    INCOME = "income"
    BALANCE = "balance"
    CASHFLOW = "cashflow"


class LineItem(BaseModel):
    """One canonical line item with values per fiscal year."""

    key: str  # canonical id, e.g. "revenue"
    label: str  # display label, e.g. "Revenue"
    statement: Statement
    values: dict[int, float | None]  # fiscal year -> USD value; None = not reported
    unit: str = "USD"
    xbrl_tags: list[str] = Field(default_factory=list)  # tags used, in fallback order


class StatementSet(BaseModel):
    """A company's linked statements for the modeled fiscal years."""

    ticker: str
    cik: str
    company_name: str
    fiscal_years: list[int]  # ascending
    currency: str = "USD"
    items: list[LineItem]


class TieoutCheck(BaseModel):
    """One accounting-identity check for one fiscal year."""

    check_id: str  # e.g. "balance_sheet_equation_2025"
    description: str
    fiscal_year: int
    passed: bool
    lhs: float | None = None
    rhs: float | None = None
    tolerance: float = 0.0


class TieoutReport(BaseModel):
    """All verification results for a generated model."""

    ticker: str
    checks: list[TieoutCheck]

    @property
    def passed(self) -> bool:
        """True only if every check passed."""
        return all(check.passed for check in self.checks)


class Chunk(BaseModel):
    """A retrieval unit cut from a filing."""

    chunk_id: str
    section: str  # e.g. "Item 7. Management's Discussion and Analysis"
    text: str
    source_url: str


class Citation(BaseModel):
    """A verbatim quote tying commentary to its source chunk."""

    chunk_id: str
    quote: str  # must be an exact substring of the chunk's text (<= 300 chars)


class Commentary(BaseModel):
    """Grounded commentary for one line item. text=None means: refused, no grounding."""

    line_item_key: str
    text: str | None
    citations: list[Citation] = Field(default_factory=list)


class RunEvent(BaseModel):
    """One pipeline progress event, streamed to the UI and logged."""

    run_id: str
    step: str  # fetch | build | verify | retrieve | narrate | generate | done | error
    status: str  # started | ok | failed
    detail: str = ""
    ts: float
