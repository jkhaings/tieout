"""Hermetic tests for app.model.facts against synthetic companyfacts payloads.

Purely synthetic (not the real fixtures) so each test pins down one specific
behavior deterministically, independent of which fiscal years happen to be
"the 5 most recent" in a real filer's history at any given time.
"""

from __future__ import annotations

from typing import Any

from app.edgar.tags import TagSpec
from app.model.facts import FactIndex
from app.schemas import Statement


def _company_facts(tags: dict[str, Any]) -> dict[str, Any]:
    return {"cik": 1, "entityName": "Synthetic Co", "facts": {"us-gaap": tags}}


def _entry(
    start: str | None, end: str, val: float, fy: int, filed: str, form: str = "10-K"
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "end": end,
        "val": val,
        "fy": fy,
        "fp": "FY",
        "form": form,
        "filed": filed,
    }
    if start is not None:
        entry["start"] = start
    return entry


# ---- period matching ------------------------------------------------------------


def test_instant_matches_by_end_date_only() -> None:
    facts = _company_facts(
        {"Assets": {"units": {"USD": [_entry(None, "2024-12-31", 100.0, 2024, "2025-02-01")]}}}
    )
    spec = TagSpec("total_assets", "Total assets", Statement.BALANCE, "instant", ("Assets",))
    resolved = FactIndex(facts).resolve_all(spec)
    assert resolved["2024-12-31"].value == 100.0


def test_duration_rejects_quarterly_and_ytd_facts() -> None:
    facts = _company_facts(
        {
            "Revenues": {
                "units": {
                    "USD": [
                        _entry(
                            "2024-01-01", "2024-03-31", 25.0, 2024, "2024-05-01"
                        ),  # quarterly, ~90 days
                        _entry("2024-01-01", "2024-09-30", 75.0, 2024, "2024-11-01"),  # 9-month YTD
                        _entry(
                            "2024-01-01", "2024-12-31", 100.0, 2024, "2025-02-01"
                        ),  # annual, ~365 days
                    ]
                }
            }
        }
    )
    spec = TagSpec("revenue", "Revenue", Statement.INCOME, "duration", ("Revenues",))
    resolved = FactIndex(facts).resolve_all(spec)
    assert list(resolved) == ["2024-12-31"]
    assert resolved["2024-12-31"].value == 100.0


def test_fiscal_year_label_comes_from_earliest_filed_not_fy_field() -> None:
    """Regression test for the central finding in this module's docstring:
    `fy` on a comparative-year fact describes the *later* filing that
    happens to also report it, not the period itself."""
    facts = _company_facts(
        {
            "Assets": {
                "units": {
                    "USD": [
                        # Originally reported in the FY2024 10-K (fy=2024) ...
                        _entry(None, "2024-12-31", 100.0, 2024, "2025-02-01"),
                        # ... then repeated as a comparative in the FY2025 10-K,
                        # which mislabels it fy=2025 even though the period is
                        # unchanged. The correct label is 2024 (earliest filed).
                        _entry(None, "2024-12-31", 100.0, 2025, "2026-02-01"),
                    ]
                }
            }
        }
    )
    spec = TagSpec("total_assets", "Total assets", Statement.BALANCE, "instant", ("Assets",))
    resolved = FactIndex(facts).resolve_all(spec)
    assert resolved["2024-12-31"].fiscal_year == 2024


def test_restatement_uses_latest_filed_value() -> None:
    facts = _company_facts(
        {
            "Assets": {
                "units": {
                    "USD": [
                        _entry(None, "2024-12-31", 100.0, 2024, "2025-02-01"),
                        _entry(
                            None, "2024-12-31", 105.0, 2025, "2026-02-01"
                        ),  # later filing restates it
                    ]
                }
            }
        }
    )
    spec = TagSpec("total_assets", "Total assets", Statement.BALANCE, "instant", ("Assets",))
    resolved = FactIndex(facts).resolve_all(spec)["2024-12-31"]
    assert resolved.value == 105.0  # latest filed value ...
    assert resolved.fiscal_year == 2024  # ... but the original fiscal year label
    assert resolved.restated is True


def test_unrestated_value_is_not_flagged() -> None:
    facts = _company_facts(
        {"Assets": {"units": {"USD": [_entry(None, "2024-12-31", 100.0, 2024, "2025-02-01")]}}}
    )
    spec = TagSpec("total_assets", "Total assets", Statement.BALANCE, "instant", ("Assets",))
    resolved = FactIndex(facts).resolve_all(spec)["2024-12-31"]
    assert resolved.restated is False


# ---- the per-year fallback chain --------------------------------------------------


def test_fallback_chain_resolves_independently_per_year() -> None:
    """The regression this project's own tag-mapping work turned up: a filer can
    switch which tag reports a concept partway through its history, so a
    single winning tag per line item is unsound. Year 1 here only reports
    under the old tag, year 2 only under the new one -- both must resolve."""
    facts = _company_facts(
        {
            "OldFxTag": {
                "units": {"USD": [_entry("2022-01-01", "2022-12-31", -5.0, 2022, "2023-02-01")]}
            },
            "NewFxTag": {
                "units": {"USD": [_entry("2023-01-01", "2023-12-31", 3.0, 2023, "2024-02-01")]}
            },
        }
    )
    spec = TagSpec(
        "fx_effect_on_cash", "FX effect", Statement.CASHFLOW, "duration", ("NewFxTag", "OldFxTag")
    )
    resolved = FactIndex(facts).resolve_all(spec)
    assert resolved["2022-12-31"].value == -5.0
    assert resolved["2022-12-31"].tag == "OldFxTag"
    assert resolved["2023-12-31"].value == 3.0
    assert resolved["2023-12-31"].tag == "NewFxTag"


def test_first_chain_tag_wins_when_both_report_the_same_year() -> None:
    facts = _company_facts(
        {
            "Preferred": {
                "units": {"USD": [_entry("2024-01-01", "2024-12-31", 10.0, 2024, "2025-02-01")]}
            },
            "Fallback": {
                "units": {"USD": [_entry("2024-01-01", "2024-12-31", 999.0, 2024, "2025-02-01")]}
            },
        }
    )
    spec = TagSpec("x", "X", Statement.INCOME, "duration", ("Preferred", "Fallback"))
    resolved = FactIndex(facts).resolve_all(spec)["2024-12-31"]
    assert resolved.value == 10.0
    assert resolved.tag == "Preferred"


def test_tag_present_in_taxonomy_with_no_usable_facts_falls_through() -> None:
    """A tag can exist with zero annual 10-K facts (e.g. only quarterly ones, or
    only from a 10-Q) -- the chain must fall through, not stop there."""
    facts = _company_facts(
        {
            "NoAnnualFacts": {
                "units": {
                    "USD": [_entry("2024-01-01", "2024-03-31", 1.0, 2024, "2024-05-01")]
                }  # quarterly only
            },
            "RealTag": {
                "units": {"USD": [_entry("2024-01-01", "2024-12-31", 42.0, 2024, "2025-02-01")]}
            },
        }
    )
    spec = TagSpec("x", "X", Statement.INCOME, "duration", ("NoAnnualFacts", "RealTag"))
    resolved = FactIndex(facts).resolve_all(spec)["2024-12-31"]
    assert resolved.value == 42.0
    assert resolved.tag == "RealTag"


def test_missing_tag_entirely_yields_no_result() -> None:
    facts = _company_facts({})
    spec = TagSpec("x", "X", Statement.INCOME, "duration", ("DoesNotExist",))
    assert FactIndex(facts).resolve_all(spec) == {}


def test_select_restricts_to_requested_periods() -> None:
    facts = _company_facts(
        {
            "Assets": {
                "units": {
                    "USD": [
                        _entry(None, "2023-12-31", 90.0, 2023, "2024-02-01"),
                        _entry(None, "2024-12-31", 100.0, 2024, "2025-02-01"),
                    ]
                }
            }
        }
    )
    spec = TagSpec("total_assets", "Total assets", Statement.BALANCE, "instant", ("Assets",))
    selected = FactIndex(facts).select(spec, {"2024-12-31"})
    assert set(selected) == {"2024-12-31"}


# ---- unit handling ------------------------------------------------------------------


def test_non_usd_unit_is_read_from_its_own_bucket() -> None:
    facts = _company_facts(
        {
            "EarningsPerShareDiluted": {
                "units": {
                    "USD/shares": [_entry("2024-01-01", "2024-12-31", 6.08, 2024, "2025-02-01")]
                }
            }
        }
    )
    spec = TagSpec(
        "eps_diluted",
        "Diluted EPS",
        Statement.INCOME,
        "duration",
        ("EarningsPerShareDiluted",),
        "USD/shares",
    )
    resolved = FactIndex(facts).resolve_all(spec)["2024-12-31"]
    assert resolved.value == 6.08


def test_wrong_unit_bucket_is_not_mixed_in() -> None:
    """A tag reported only under a different unit than the spec declares must
    not resolve -- this is exactly the bug that once made eps_diluted and
    weighted_diluted_shares silently resolve to None on every fixture."""
    facts = _company_facts(
        {
            "EarningsPerShareDiluted": {
                "units": {
                    "USD/shares": [_entry("2024-01-01", "2024-12-31", 6.08, 2024, "2025-02-01")]
                }
            }
        }
    )
    spec = TagSpec(
        "eps_diluted",
        "Diluted EPS",
        Statement.INCOME,
        "duration",
        ("EarningsPerShareDiluted",),
        "USD",
    )
    assert FactIndex(facts).resolve_all(spec) == {}
