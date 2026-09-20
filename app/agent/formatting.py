"""Pre-format line-item figures for narration; pick which line items get commentary.

CLAUDE.md rule 1: the LLM never computes, transforms, or restates a number.
`app.rag.narrate.narrate_line_item` enforces that by validating every
number-like token in the model's commentary against exactly these
pre-formatted strings (plus any cited quote) -- so the format chosen here is
load-bearing, not cosmetic. It matches the house style already encoded in
`app/rag`'s own tests (`"$391.04B"`): a `$` prefix for dollar amounts, a
single-letter magnitude suffix (`B`/`M`/`K`) for values at or above a
thousand, and exactly two decimal places once scaled. Each figure is
prefixed with its fiscal year (`"FY2025: $391.04B"`) so the model can
reference a specific year in its commentary -- the year is then itself a
grounded token, since it appears verbatim in the figure string.
"""

from __future__ import annotations

from app.schemas import LineItem, StatementSet

# Headline line items an equity analyst would expect commentary on: one from
# each statement's topline/bottomline plus the balance-sheet totals. Not all
# 43 canonical items (app.edgar.tags.CANONICAL) are narrated -- most are
# subtotal/reconciling inputs (e.g. `noncontrolling_interest_in_income`,
# `fx_effect_on_cash`) with nothing an MD&A-grounded sentence would usefully
# say about them on their own, and narrating all 43 would multiply the
# per-run Anthropic call count roughly fivefold for no product value.
COMMENTARY_LINE_ITEMS: tuple[str, ...] = (
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "eps_diluted",
    "cfo",
    "capex",
    "total_assets",
    "total_liabilities",
    "total_equity",
)


def _magnitude_scale(magnitude: float) -> tuple[float, str]:
    """Return `(scaled_value, suffix)` for a non-negative magnitude.

    Args:
        magnitude: A non-negative number.

    Returns:
        The value divided into billions/millions/thousands with the
        matching one-letter suffix, or the value unscaled with an empty
        suffix if it is under 1,000.
    """
    if magnitude >= 1_000_000_000:
        return magnitude / 1_000_000_000, "B"
    if magnitude >= 1_000_000:
        return magnitude / 1_000_000, "M"
    if magnitude >= 1_000:
        return magnitude / 1_000, "K"
    return magnitude, ""


def format_figure(value: float, unit: str) -> str:
    """Render one already-computed value as a pre-formatted figure string.

    Args:
        value: The value to render, as already computed by `app.model`
            (never recomputed here).
        unit: The line item's `LineItem.unit` -- `"USD"`, `"USD/shares"`,
            or `"shares"`.

    Returns:
        A string like `"$391.04B"` (USD), `"$6.42"` (USD/shares), or
        `"15.55B shares"` (shares). Negative values keep a leading `-`
        immediately before the magnitude (e.g. `"-$1.23B"`).
    """
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if unit == "USD/shares":
        return f"{sign}${magnitude:,.2f}"
    scaled, suffix = _magnitude_scale(magnitude)
    body = f"{scaled:,.2f}{suffix}"
    if unit == "shares":
        return f"{sign}{body} shares"
    return f"{sign}${body}"


def build_figures(item: LineItem, statements: StatementSet) -> list[str]:
    """Build the pre-formatted, year-labeled figure strings for one line item.

    Args:
        item: The line item to render figures for.
        statements: The statement set `item` belongs to, for its ordered
            `fiscal_years`.

    Returns:
        One `"FY<year>: <figure>"` string per fiscal year with a reported
        (non-`None`) value, oldest first. A year with no reported value is
        omitted, never rendered as a fabricated zero.
    """
    lines: list[str] = []
    for fy in statements.fiscal_years:
        value = item.values.get(fy)
        if value is None:
            continue
        lines.append(f"FY{fy}: {format_figure(value, item.unit)}")
    return lines
