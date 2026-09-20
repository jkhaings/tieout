"""Hermetic tests for app.edgar.tags -- pure data, no I/O."""

from __future__ import annotations

import pytest

from app.edgar.tags import BY_KEY, CANONICAL, get
from app.schemas import Statement


def test_keys_are_unique() -> None:
    keys = [spec.key for spec in CANONICAL]
    assert len(keys) == len(set(keys))
    assert len(BY_KEY) == len(CANONICAL)


def test_every_chain_is_nonempty_and_has_no_internal_duplicates() -> None:
    for spec in CANONICAL:
        assert len(spec.tags) >= 1, f"{spec.key} has an empty fallback chain"
        assert len(set(spec.tags)) == len(spec.tags), f"{spec.key} repeats a tag in its own chain"


def test_kind_is_instant_or_duration() -> None:
    for spec in CANONICAL:
        assert spec.kind in ("instant", "duration")


def test_has_at_least_35_items() -> None:
    # CLAUDE.md/ARCHITECTURE.md call for "~35 canonical line items"; this
    # landed a bit higher for well-documented reasons -- see the tags.py
    # module docstring. This test just guards against silent shrinkage.
    assert len(CANONICAL) >= 35


def test_every_statement_type_is_represented() -> None:
    assert {spec.statement for spec in CANONICAL} == {
        Statement.INCOME,
        Statement.BALANCE,
        Statement.CASHFLOW,
    }


def test_get_returns_matching_spec() -> None:
    spec = get("total_assets")
    assert spec.key == "total_assets"
    assert spec.statement == Statement.BALANCE
    assert spec.tags == ("Assets",)


def test_get_raises_keyerror_for_unknown_key() -> None:
    with pytest.raises(KeyError):
        get("not_a_real_canonical_key")


def test_no_tag_is_reused_with_a_different_kind() -> None:
    """The same us-gaap tag can't legitimately be both an instant and a duration fact."""
    seen: dict[str, str] = {}
    for spec in CANONICAL:
        for tag in spec.tags:
            if tag in seen:
                assert seen[tag] == spec.kind, f"{tag} used as both {seen[tag]} and {spec.kind}"
            seen[tag] = spec.kind


def test_rejected_lookalike_pairs_are_not_chained_together() -> None:
    """Regression guard for the two pairs confirmed to be genuinely different concepts.

    See the tags.py module docstring: these looked like fallback candidates
    but were confirmed (against real values) to be simultaneously reported
    by the same filer with different numbers.
    """
    other_income = get("other_income_expense")
    assert "OtherNonoperatingIncomeExpense" not in other_income.tags

    long_term_debt = get("long_term_debt")
    assert long_term_debt.tags[0] == "LongTermDebtNoncurrent"


def test_sga_is_not_synthesized_from_its_components() -> None:
    """Microsoft/Alphabet split SG&A; combining G&A + S&M to fake a total is exactly what
    tags.py's module docstring says this file refuses to do -- each stays a distinct item."""
    combined = get("selling_general_admin")
    ga = get("general_administrative_expense")
    sm = get("selling_marketing_expense")
    assert combined.tags == ("SellingGeneralAndAdministrativeExpense",)
    assert ga.tags == ("GeneralAndAdministrativeExpense",)
    assert sm.tags == ("SellingAndMarketingExpense",)
