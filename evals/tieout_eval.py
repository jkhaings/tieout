"""Binary, cell-level tie-out accuracy eval against `evals/datasets/tieout_gold.json`.

Scores `app.model.builder.build_statements` -- the real code under test --
against a gold answer key produced independently by
`evals/generate_tieout_gold.py` (which reimplements fact selection from raw
companyfacts JSON rather than importing `app.model`, so this eval is not
circular). Fully offline: gold values and the fixtures both come from
committed files in `tests/fixtures/`, so there is no network access and no
Anthropic call anywhere in this module (CLAUDE.md rule 7 -- this still only
runs locally, never in CI, because `evals/scorecard.py` calling it is itself
a local-only entry point).

A cell is `(ticker, line_item_key, fiscal_year)`. It is correct iff:

- both the gold value and the actual (built) value are `None`, or
- both are non-`None` and `abs(actual - gold) <= 1.0` (a $1 absolute
  tolerance, guarding only float representation, not real accounting slop).

This is deliberately *not* the tolerance `app.model.verifier` applies to its
tie-out checks. That one is inferred per check from the rounding granularity
of the facts it sums, because a filer presenting in millions injects up to a
million dollars of rounding into each term of an identity. Here there is no
identity and no summing -- one built cell is compared against the same fact
re-derived from the same JSON -- so anything beyond float noise is a real
mismatch and must be reported as one.

Anything else -- one side `None` and the other not, or both non-`None` but
more than $1 apart -- is a mismatch and is recorded, never silently
dropped.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.model.builder import build_statements

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
GOLD_PATH = Path(__file__).resolve().parent / "datasets" / "tieout_gold.json"

_TOLERANCE_USD = 1.0  # float-representation only; see the module docstring


def _load_json(path: Path) -> dict[str, Any]:
    """Read and parse one JSON file, raising a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"required file is missing: {path}")
    with path.open() as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def _load_gold() -> dict[str, Any]:
    """Load `evals/datasets/tieout_gold.json`, raising clearly if it does not exist."""
    return _load_json(GOLD_PATH)


def _cell_matches(gold_value: float | None, actual_value: float | None) -> bool:
    """Return whether one gold/actual cell pair ties out within `_TOLERANCE_USD`.

    Both `None` counts as a match (an item legitimately unreported by the
    filer); one `None` and the other not is always a mismatch; otherwise the
    two floats must agree within the documented $1 absolute tolerance.
    """
    if gold_value is None and actual_value is None:
        return True
    if gold_value is None or actual_value is None:
        return False
    return abs(actual_value - gold_value) <= _TOLERANCE_USD


def run(tickers: Sequence[str] = ("AAPL", "MSFT")) -> dict[str, Any]:
    """Score every gold cell against a freshly built `StatementSet` for each ticker.

    For each ticker in `tickers`, builds a `StatementSet` via
    `app.model.builder.build_statements` from
    `tests/fixtures/companyfacts_<TICKER>.json` and
    `tests/fixtures/submissions_<TICKER>.json`, then compares every
    `(line_item_key, fiscal_year)` cell recorded in
    `evals/datasets/tieout_gold.json` for that ticker against the built
    value using `_cell_matches`.

    Raises `KeyError` if a requested ticker has no entry in the gold file,
    and `FileNotFoundError` if a ticker's fixture files are missing --
    never silently skips a ticker.

    Returns a dict shaped:

        {
            "tolerance_usd": 1.0,
            "tickers": {
                "AAPL": {"correct": int, "total": int, "accuracy": float},
                ...
            },
            "overall": {"correct": int, "total": int, "accuracy": float},
            "mismatches": [
                {
                    "ticker": str,
                    "key": str,
                    "fiscal_year": int,
                    "gold": float | None,
                    "actual": float | None,
                },
                ...
            ],
        }

    `accuracy` is `correct / total` (`0.0` if `total` is `0`, which should
    not happen for a non-empty gold file).
    """
    gold = _load_gold()
    gold_tickers: dict[str, Any] = gold["tickers"]

    per_ticker: dict[str, dict[str, Any]] = {}
    mismatches: list[dict[str, Any]] = []
    overall_correct = 0
    overall_total = 0

    for ticker in tickers:
        if ticker not in gold_tickers:
            raise KeyError(f"ticker {ticker!r} has no entry in the gold file {GOLD_PATH}")

        facts_path = FIXTURES_DIR / f"companyfacts_{ticker}.json"
        submissions_path = FIXTURES_DIR / f"submissions_{ticker}.json"
        company_facts = _load_json(facts_path)
        submissions = _load_json(submissions_path)

        statements = build_statements(company_facts, submissions, ticker)
        actual_by_key = {item.key: item.values for item in statements.items}

        gold_cells: dict[str, dict[str, float | None]] = gold_tickers[ticker]["cells"]

        correct = 0
        total = 0
        for key, per_year_gold in gold_cells.items():
            actual_values = actual_by_key.get(key, {})
            for fy_str, gold_value in per_year_gold.items():
                fiscal_year = int(fy_str)
                actual_value = actual_values.get(fiscal_year)
                total += 1
                if _cell_matches(gold_value, actual_value):
                    correct += 1
                else:
                    mismatches.append(
                        {
                            "ticker": ticker,
                            "key": key,
                            "fiscal_year": fiscal_year,
                            "gold": gold_value,
                            "actual": actual_value,
                        }
                    )

        per_ticker[ticker] = {
            "correct": correct,
            "total": total,
            "accuracy": (correct / total) if total else 0.0,
        }
        overall_correct += correct
        overall_total += total

    return {
        "tolerance_usd": _TOLERANCE_USD,
        "tickers": per_ticker,
        "overall": {
            "correct": overall_correct,
            "total": overall_total,
            "accuracy": (overall_correct / overall_total) if overall_total else 0.0,
        },
        "mismatches": mismatches,
    }


def _print_summary(result: dict[str, Any]) -> None:
    """Pretty-print a short human-readable summary of `run()`'s result."""
    for ticker, stats in result["tickers"].items():
        print(f"{ticker}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")
    overall = result["overall"]
    print(f"overall: {overall['correct']}/{overall['total']} ({overall['accuracy']:.2%})")
    print(f"mismatches: {len(result['mismatches'])}")
    for mismatch in result["mismatches"]:
        print(f"  {mismatch}")


if __name__ == "__main__":
    _print_summary(run())
