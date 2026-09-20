"""Tests for app.agent.formatting: figure rendering and its narrate-grounding contract."""

from __future__ import annotations

import re

from app.agent.formatting import COMMENTARY_LINE_ITEMS, build_figures, format_figure
from app.schemas import LineItem, Statement

# The exact number-grounding pattern app.rag.narrate._validate_draft uses to
# find number-like tokens in a draft's text. Every figure this module
# produces must be found by this regex as a complete, isolated token -- if
# it isn't, narrate_line_item would never be able to ground a commentary
# mentioning that figure, no matter how faithfully the model quoted it.
_NUMBER_TOKEN_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?[%BMKTbmkt]?")


def test_format_figure_usd_billions() -> None:
    assert format_figure(391_035_000_000, "USD") == "$391.04B"


def test_format_figure_usd_millions() -> None:
    assert format_figure(5_230_000, "USD") == "$5.23M"


def test_format_figure_usd_thousands() -> None:
    assert format_figure(123_456, "USD") == "$123.46K"


def test_format_figure_usd_small() -> None:
    assert format_figure(42, "USD") == "$42.00"


def test_format_figure_negative_usd() -> None:
    assert format_figure(-1_234_567_890, "USD") == "-$1.23B"


def test_format_figure_usd_per_share() -> None:
    assert format_figure(6.42, "USD/shares") == "$6.42"


def test_format_figure_shares() -> None:
    assert format_figure(15_550_000_000, "shares") == "15.55B shares"


def test_format_figure_zero() -> None:
    assert format_figure(0, "USD") == "$0.00"


def test_every_format_figure_output_is_boundary_grounded_in_itself() -> None:
    """The number-token regex must find, as an isolated token, the number it just rendered.

    This is the actual contract narrate_line_item's `_token_is_grounded`
    relies on (boundary-aware, not substring containment): a figure string
    must contain its own numeric value as a complete match, not merely as a
    substring of something else.
    """
    for value, unit in [
        (391_035_000_000, "USD"),
        (-1_234_567_890, "USD"),
        (6.42, "USD/shares"),
        (15_550_000_000, "shares"),
        (999, "USD"),
    ]:
        figure = format_figure(value, unit)
        assert _NUMBER_TOKEN_RE.search(figure), f"{figure!r} has no number-like token"


def test_build_figures_skips_unreported_years() -> None:
    item = LineItem(
        key="revenue",
        label="Revenue",
        statement=Statement.INCOME,
        values={2023: 383_285_000_000, 2024: None, 2025: 416_161_000_000},
        unit="USD",
    )

    class _Stub:
        fiscal_years = [2023, 2024, 2025]

    lines = build_figures(item, _Stub())  # type: ignore[arg-type]
    assert lines == ["FY2023: $383.29B", "FY2025: $416.16B"]
    assert "FY2024" not in " ".join(lines)


def test_build_figures_year_is_itself_a_grounded_token() -> None:
    """The fiscal year prefix must itself match the number-token regex.

    This is what lets narrate_line_item's commentary reference a specific
    fiscal year (e.g. "in FY2025") without the validator rejecting "2025"
    as an ungrounded number: it must appear, boundary-aware, in the figure
    string handed to the model.
    """
    item = LineItem(
        key="revenue",
        label="Revenue",
        statement=Statement.INCOME,
        values={2025: 416_161_000_000},
        unit="USD",
    )

    class _Stub:
        fiscal_years = [2025]

    (line,) = build_figures(item, _Stub())  # type: ignore[arg-type]
    pattern = re.compile(r"(?<![\d.])2025(?![\d])")
    assert pattern.search(line), f"{line!r} does not contain 2025 as an isolated token"


def test_commentary_line_items_is_a_fixed_curated_set() -> None:
    """A basic sanity/regression check on the curated set, not exhaustive coverage."""
    assert "revenue" in COMMENTARY_LINE_ITEMS
    assert "net_income" in COMMENTARY_LINE_ITEMS
    assert len(COMMENTARY_LINE_ITEMS) == len(set(COMMENTARY_LINE_ITEMS))
