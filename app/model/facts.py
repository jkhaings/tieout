"""Select per-fiscal-year values out of raw `us-gaap` companyfacts JSON.

Two properties of SEC's companyfacts feed make naive selection wrong, both
confirmed by inspecting real filings before writing this module:

1. **`fy`/`fp` describe the filing that reported a fact, not the period the
   fact covers.** A 10-K filed as fiscal year 2024 carries prior-year
   comparatives tagged `fy: 2024, fp: "FY"` whose `start`/`end` cover fiscal
   2022. This module never uses `fy` to pick a period -- periods are matched
   purely by `start`/`end` dates (`_is_annual_duration`, and instants by
   `end` alone). `fy` is used only afterwards, to *label* a period once
   selected (`ResolvedValue.fiscal_year`), and even then only from the
   earliest-filed fact for that period -- i.e. the filing that first
   reported it as its current year, which is authoritative and requires no
   calendar heuristics for non-calendar fiscal year-ends.
2. **A tag can exist in a company's taxonomy with zero usable annual 10-K
   facts** (Apple declares `PaymentsOfDividendsCommonStock` but never files
   an annual fact under it). `app.edgar.tags` fallback chains fall through
   on an empty result here, not on tag absence -- the two are treated
   identically throughout this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from app.edgar.tags import TagSpec

_ANNUAL_MIN_DAYS = 330
_ANNUAL_MAX_DAYS = 400
_ANNUAL_FORMS = frozenset({"10-K", "10-K/A"})


@dataclass(frozen=True, slots=True)
class ResolvedValue:
    """One tag's value for one fiscal period, plus provenance."""

    value: float
    tag: str  # the us-gaap tag that supplied this value
    filed: str  # ISO date of the filing that supplied `value`
    fiscal_year: int | None  # `fy` of the filing that *first* reported this period
    restated: bool  # True if a later filing revised the value for this period


def _parse_date(raw: str) -> date:
    return date.fromisoformat(raw)


def _is_annual_duration(entry: dict[str, Any]) -> bool:
    """True for a ~330-400 day duration fact, which excludes quarterly and YTD facts."""
    start = entry.get("start")
    if not start:
        return False
    span = (_parse_date(entry["end"]) - _parse_date(start)).days
    return _ANNUAL_MIN_DAYS <= span <= _ANNUAL_MAX_DAYS


class FactIndex:
    """Period-based fact lookup over one company's `companyfacts` JSON."""

    def __init__(self, company_facts: dict[str, Any]) -> None:
        """Wrap the `us-gaap` taxonomy of one company's raw `companyfacts` JSON."""
        self._facts: dict[str, Any] = company_facts.get("facts", {}).get("us-gaap", {})

    def _annual_entries(
        self, tag: str, kind: Literal["instant", "duration"], unit: str
    ) -> list[dict[str, Any]]:
        """All 10-K/10-K-A entries for `tag` in `unit` matching `kind`, unfiltered by period.

        `unit` must be passed explicitly rather than assumed to be `"USD"`:
        EPS is filed under `"USD/shares"` and share counts under `"shares"`.
        Hardcoding `"USD"` here previously made every fact for those two
        concepts invisible (not merely unreported) on both fixtures -- this
        project's own tests caught it.
        """
        tag_data = self._facts.get(tag)
        if not tag_data:
            return []
        unit_entries: list[dict[str, Any]] = tag_data.get("units", {}).get(unit, [])
        matches = []
        for entry in unit_entries:
            if entry.get("form") not in _ANNUAL_FORMS:
                continue
            if kind == "instant":
                if entry.get("start"):
                    continue
            elif not _is_annual_duration(entry):
                continue
            matches.append(entry)
        return matches

    def periods_for(
        self, tag: str, kind: Literal["instant", "duration"], unit: str
    ) -> dict[str, ResolvedValue]:
        """Best `ResolvedValue` per period-end date for a single tag.

        Among facts reporting the same period, the latest-filed value wins
        (ARCHITECTURE.md: "restated values -- use latest filed, note it");
        the fiscal year label and `restated` flag come from comparing against
        the earliest-filed fact for that same period.
        """
        by_period: dict[str, list[dict[str, Any]]] = {}
        for entry in self._annual_entries(tag, kind, unit):
            by_period.setdefault(entry["end"], []).append(entry)

        result: dict[str, ResolvedValue] = {}
        for period_end, entries in by_period.items():
            entries.sort(key=lambda e: e["filed"])
            first, last = entries[0], entries[-1]
            fy = first.get("fy")
            result[period_end] = ResolvedValue(
                value=float(last["val"]),
                tag=tag,
                filed=last["filed"],
                fiscal_year=int(fy) if fy is not None else None,
                restated=last["filed"] != first["filed"] and last["val"] != first["val"],
            )
        return result

    def resolve_all(self, spec: TagSpec) -> dict[str, ResolvedValue]:
        """Resolve `spec`'s fallback chain for every period any chain tag reports.

        For each period end, the first tag in `spec.tags` (in order) that
        reports it wins -- decided independently per period, since which tag
        a filer uses for a concept can change across fiscal years (e.g.
        Microsoft's FX-on-cash tag changed in 2021; see `app.edgar.tags`).
        """
        result: dict[str, ResolvedValue] = {}
        for tag in spec.tags:
            for period_end, value in self.periods_for(tag, spec.kind, spec.unit).items():
                result.setdefault(period_end, value)
        return result

    def select(self, spec: TagSpec, period_ends: set[str]) -> dict[str, ResolvedValue]:
        """`resolve_all(spec)` restricted to the requested `period_ends`."""
        return {k: v for k, v in self.resolve_all(spec).items() if k in period_ends}
