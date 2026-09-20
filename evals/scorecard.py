"""Single entry point for the eval layer: run everything, write the scorecard.

``uv run python -m evals.scorecard`` (also exposed as ``make evals``) runs
the tie-out eval, the retrieval eval, and the LLM-as-judge eval, assembles
one dict via :func:`build_scorecard`, writes it to
``evals/scorecard.json``, and renders a compact markdown summary into
``README.md`` between the ``<!-- SCORECARD:START -->`` /
``<!-- SCORECARD:END -->`` sentinels via :func:`write_readme_section`.

Local-only (CLAUDE.md rule 7): this module makes real, billed Anthropic API
calls whenever the judge eval is not skipped. Never imported by ``tests/``
in a way that would exercise those paths, and never run in CI -- CI stays
lint + types + hermetic unit tests only.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals import judge, retrieval_eval, tieout_eval

logger = logging.getLogger(__name__)

_SCORECARD_PATH = Path(__file__).resolve().parent / "scorecard.json"
_README_PATH = Path(__file__).resolve().parent.parent / "README.md"

_SENTINEL_START = "<!-- SCORECARD:START -->"
_SENTINEL_END = "<!-- SCORECARD:END -->"


def _git_sha() -> str:
    """Return the short current git commit sha, or `"unknown"` if that fails.

    Never raises: a missing `git` binary, a non-repo working directory, or
    any other subprocess failure all fall back to `"unknown"` rather than
    aborting the whole scorecard run over a cosmetic field.

    Returns:
        The short git sha (e.g. `"eb2ad8b"`), or `"unknown"`.
    """
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except Exception:  # noqa: BLE001 - a cosmetic field, never worth failing the run over
        logger.warning("git rev-parse failed; recording git_sha as 'unknown'", exc_info=True)
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_scorecard(*, skip_judge: bool = False, judge_only: bool = False) -> dict[str, Any]:
    """Run every eval and assemble one scorecard dict.

    Runs `evals.tieout_eval.run()` and `evals.retrieval_eval.run()`
    unconditionally (both are fully offline and hermetic-in-spirit, though
    local-only per CLAUDE.md rule 7), and `evals.judge.run()` unless
    `skip_judge` is `True`. `judge.run()` itself may already return a
    `{"status": "skipped", "reason": ...}` dict if no Anthropic API key is
    configured -- that is passed through as-is, never altered or hidden.

    Args:
        skip_judge: If `True`, never call `evals.judge.run()` at all (no
            Anthropic API call is made) and record a
            `{"status": "skipped", "reason": "--skip-judge passed"}` judge
            section instead. Use this to verify the rest of the scorecard
            pipeline without spending API money. Mutually exclusive with
            `judge_only` (the caller -- `main()` -- enforces this; this
            function itself just checks `skip_judge` first).
        judge_only: If `True`, call `evals.judge.run(use_saved_generation=True)`
            instead of a normal `evals.judge.run()` -- re-grades the
            samples already persisted in `evals/generation_samples.json`
            (see `evals.judge.generate_samples`) without narrating again,
            spending judge API money only. Raises `FileNotFoundError` (from
            `evals.judge.run`) if no such file exists yet.

    Returns:
        A dict shaped:

            {
                "git_sha": str,
                "generated_at": str,  # ISO 8601 UTC timestamp
                "tieout": <evals.tieout_eval.run()'s return value>,
                "retrieval": <evals.retrieval_eval.run()'s return value>,
                "judge": <evals.judge.run()'s return value, or the
                          --skip-judge placeholder above>,
            }
    """
    tieout_result = tieout_eval.run()
    retrieval_result = retrieval_eval.run()
    judge_result: dict[str, Any] = (
        {"status": "skipped", "reason": "--skip-judge passed"}
        if skip_judge
        else judge.run(use_saved_generation=judge_only)
    )

    return {
        "git_sha": _git_sha(),
        "generated_at": datetime.now(UTC).isoformat(),
        "tieout": tieout_result,
        "retrieval": retrieval_result,
        "judge": judge_result,
    }


def _format_pct(value: float | None) -> str:
    """Format a `0..1` float as a percentage string, or `"n/a"` if `None`."""
    return f"{value:.1%}" if value is not None else "n/a"


def _format_count_pct(hits: int, total: int) -> str:
    """Format `"hits/total (pct%)"`, or `"n/a (0/0)"` if `total` is zero.

    A raw fraction sits right next to every percentage it summarizes --
    "9/13" reads more honestly than "69.2%" alone, especially at the small
    sample sizes this eval runs at.
    """
    if total == 0:
        return "n/a (0/0)"
    return f"{hits}/{total} ({hits / total:.1%})"


def _render_tieout_section(tieout: dict[str, Any]) -> list[str]:
    """Render the tie-out portion of the scorecard as markdown lines.

    Args:
        tieout: `evals.tieout_eval.run()`'s return value.

    Returns:
        Markdown lines: a heading, an overall summary line, and a per-ticker
        table.
    """
    lines = ["### Tie-out accuracy (binary, cell-level, vs. raw companyfacts JSON)", ""]
    overall = tieout["overall"]
    tolerance = tieout["tolerance_usd"]
    lines.append(
        f"Overall: **{overall['correct']}/{overall['total']}** "
        f"({_format_pct(overall['accuracy'])}), tolerance ${tolerance:g}."
    )
    lines.append("")
    lines.append("| Ticker | Correct | Total | Accuracy |")
    lines.append("| --- | --- | --- | --- |")
    for ticker, stats in tieout["tickers"].items():
        lines.append(
            f"| {ticker} | {stats['correct']} | {stats['total']} | "
            f"{_format_pct(stats['accuracy'])} |"
        )
    num_mismatches = len(tieout["mismatches"])
    lines.append("")
    lines.append(f"Mismatches: **{num_mismatches}**.")
    return lines


def _render_retrieval_section(retrieval: dict[str, Any]) -> list[str]:
    """Render the retrieval portion of the scorecard as markdown lines.

    Reports `production_labels` and `analyst_queries` as two clearly
    labeled, separate tables (never blended into one row), plus the
    fail-closed refusal rate on `irrelevant_queries` and which backend
    (BM25-only vs. embeddings) produced these numbers.

    Args:
        retrieval: `evals.retrieval_eval.run()`'s return value.

    Returns:
        Markdown lines for the retrieval section.
    """
    ks: list[int] = retrieval["ks"]
    backend = retrieval["backend"]
    backend_desc = "hybrid (embeddings + reranker)" if backend["use_embeddings"] else "BM25-only"
    lines = ["### Retrieval precision/recall@k", ""]
    lines.append(f"Backend: **{backend_desc}**.")
    lines.append("")

    for slice_key, slice_title, caveat in (
        (
            "production_labels",
            "Production query pattern (`retriever.retrieve(item.label)`)",
            "measures what the app actually issues today",
        ),
        (
            "analyst_queries",
            "Hand-written analyst questions",
            "measures a broader capability, not what production issues today",
        ),
    ):
        slice_result = retrieval[slice_key]
        lines.append(f"**{slice_title}** ({caveat}):")
        lines.append("")
        header_cells = " | ".join(f"P@{k}" for k in ks)
        header_cells += " | " + " | ".join(f"R@{k}" for k in ks)
        lines.append(f"| {header_cells} |")
        lines.append("| " + " | ".join(["---"] * (2 * len(ks))) + " |")
        precisions = [_format_pct(slice_result["precision_at_k"][str(k)]) for k in ks]
        recalls = [_format_pct(slice_result["recall_at_k"][str(k)]) for k in ks]
        lines.append("| " + " | ".join(precisions + recalls) + " |")
        lines.append(
            f"({slice_result['num_scored']} scored, "
            f"{slice_result['num_without_gold']} without gold, "
            f"{slice_result['num_skipped']} skipped, "
            f"out of {slice_result['num_queries']} queries)"
        )
        lines.append("")

    refusal = retrieval["irrelevant_queries_refusal"]
    lines.append(
        f"**Fail-closed refusal rate** (irrelevant queries, production `RagSettings()` "
        f"defaults): **{_format_pct(refusal['refusal_rate'])}** "
        f"({refusal['num_correct_refusals']}/{refusal['num_scored']} correctly refused, "
        f"{refusal['num_skipped']} skipped)."
    )
    return lines


def _render_judge_section(judge_result: dict[str, Any]) -> list[str]:
    """Render the LLM-as-judge portion of the scorecard as markdown lines.

    Reports `grounded`/`cited`/`no_invented_numbers` as three separate
    rates (never one blended vibe score), each shown as a raw
    `hits/n_judged` fraction alongside its percentage -- deliberately, at
    this eval's small sample sizes ("9/13" is more honest than "69.2%"
    alone) -- or a single clearly-worded skip line when
    `judge_result["status"] == "skipped"`.

    Args:
        judge_result: `evals.judge.run()`'s return value, or the
            `--skip-judge` placeholder.

    Returns:
        Markdown lines for the judge section.
    """
    lines = ["### LLM-as-judge (groundedness, citation presence, invented numbers)", ""]
    if judge_result.get("status") == "skipped":
        reason = judge_result.get("reason", "unknown reason")
        lines.append(f"judge: skipped ({reason})")
        return lines

    lines.append(
        f"Generator: `{judge_result['generator_model']}`; judge: `{judge_result['judge_model']}` "
        "(deliberately a different model)."
    )
    lines.append("")
    n_judged = judge_result["n_judged"]
    lines.append("| Criterion | Result |")
    lines.append("| --- | --- |")
    lines.append(f"| Grounded | {_format_count_pct(judge_result['grounded_hits'], n_judged)} |")
    lines.append(f"| Cited | {_format_count_pct(judge_result['cited_hits'], n_judged)} |")
    no_invented = _format_count_pct(judge_result["no_invented_numbers_hits"], n_judged)
    lines.append(f"| No invented numbers | {no_invented} |")
    lines.append("")
    tickers_run = ", ".join(judge_result["tickers_run"]) or "(none)"
    lines.append(
        f"({judge_result['n_judged']}/{judge_result['n_narrated']} narrated items judged; "
        f"tickers run: {tickers_run})"
    )
    if judge_result["tickers_skipped"]:
        skipped_desc = "; ".join(
            f"{ticker}: {reason}" for ticker, reason in judge_result["tickers_skipped"].items()
        )
        lines.append(f"(tickers skipped: {skipped_desc})")
    return lines


def render_markdown_table(scorecard: dict[str, Any]) -> str:
    """Render a compact, scannable markdown summary of one scorecard dict.

    Renders overall + per-ticker tie-out accuracy, retrieval precision/
    recall@k for each `k` and each slice (`production_labels` vs.
    `analyst_queries`, clearly labeled as measuring different things, never
    blended into one row), the retrieval backend used, and the three judge
    per-criterion rates as three separate rows (never one blended judge
    score) -- or a single clearly-worded `"judge: skipped (<reason>)"` line
    when judge status is `"skipped"`.

    Args:
        scorecard: A dict shaped like `build_scorecard()`'s return value.

    Returns:
        A markdown string, without any leading/trailing sentinel comments
        (those are added by `write_readme_section`).
    """
    lines = [
        f"_Generated at {scorecard['generated_at']} from commit `{scorecard['git_sha']}`._",
        "",
        *_render_tieout_section(scorecard["tieout"]),
        "",
        *_render_retrieval_section(scorecard["retrieval"]),
        "",
        *_render_judge_section(scorecard["judge"]),
    ]
    return "\n".join(lines) + "\n"


def write_readme_section(markdown_table: str, readme_path: Path = _README_PATH) -> None:
    """Replace the scorecard section of `readme_path` between its sentinels.

    Replaces everything between `<!-- SCORECARD:START -->` and
    `<!-- SCORECARD:END -->` with `markdown_table` (sentinels preserved),
    so re-running this function is idempotent -- it always replaces between
    the sentinels, never appends.

    If `readme_path` does not exist, or exists but is missing either
    sentinel, this logs a clear warning and returns without raising and
    without creating or modifying the file: `README.md` is written by a
    separate agent in this same session and may not exist yet, or may not
    yet contain the sentinels, when this function is first exercised --
    failing loudly here would be wrong; this must degrade gracefully.

    Args:
        markdown_table: The rendered markdown to place between the
            sentinels (typically `render_markdown_table`'s return value).
        readme_path: Path to the README to update. Defaults to the repo's
            top-level `README.md`.
    """
    if not readme_path.exists():
        logger.warning(
            "%s does not exist yet; skipping scorecard README update "
            "(re-run `python -m evals.scorecard` once the README with "
            "SCORECARD:START/END sentinels has been added)",
            readme_path,
        )
        return

    content = readme_path.read_text(encoding="utf-8")
    if _SENTINEL_START not in content or _SENTINEL_END not in content:
        logger.warning(
            "%s exists but is missing the %s / %s sentinels; skipping scorecard README update",
            readme_path,
            _SENTINEL_START,
            _SENTINEL_END,
        )
        return

    start_index = content.index(_SENTINEL_START) + len(_SENTINEL_START)
    end_index = content.index(_SENTINEL_END)
    if end_index < start_index:
        logger.warning(
            "%s has %s before %s; skipping scorecard README update",
            readme_path,
            _SENTINEL_END,
            _SENTINEL_START,
        )
        return

    new_content = (
        content[:start_index] + "\n" + markdown_table.rstrip("\n") + "\n" + content[end_index:]
    )
    readme_path.write_text(new_content, encoding="utf-8")


def main() -> None:
    """CLI entry point: run every eval, write the scorecard, update the README.

    Parses `--skip-judge` and `--judge-only` (mutually exclusive -- argparse
    itself rejects passing both), calls `build_scorecard()`, writes
    `evals/scorecard.json` (`json.dumps(..., indent=2)` plus a trailing
    newline), renders and writes the README scorecard section, and prints a
    short human-readable summary (git sha, tie-out accuracy, whether the
    judge ran or was skipped and why).
    """
    parser = argparse.ArgumentParser(
        description="Run the tieout eval layer and produce evals/scorecard.json."
    )
    judge_group = parser.add_mutually_exclusive_group()
    judge_group.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip the LLM-as-judge eval (no Anthropic API call, no API money spent).",
    )
    judge_group.add_argument(
        "--judge-only",
        action="store_true",
        help=(
            "Re-grade the samples already persisted in evals/generation_samples.json "
            "(see evals.judge.generate_samples) instead of narrating again -- no "
            "generator API spend, judge API spend only. Fails loudly if no saved "
            "samples exist yet (run once without this flag first)."
        ),
    )
    args = parser.parse_args()

    scorecard = build_scorecard(skip_judge=args.skip_judge, judge_only=args.judge_only)

    _SCORECARD_PATH.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")

    markdown_table = render_markdown_table(scorecard)
    write_readme_section(markdown_table)

    overall = scorecard["tieout"]["overall"]
    judge_result = scorecard["judge"]
    judge_summary = (
        f"skipped ({judge_result.get('reason', 'unknown reason')})"
        if judge_result.get("status") == "skipped"
        else (
            f"ran (grounded={_format_pct(judge_result['grounded_rate'])}, "
            f"cited={_format_pct(judge_result['cited_rate'])}, "
            f"no_invented_numbers={_format_pct(judge_result['no_invented_numbers_rate'])})"
        )
    )
    print(  # noqa: T201 - evals/ CLI output, not app code (CLAUDE.md rule 10 is app/-scoped)
        f"scorecard written: {_SCORECARD_PATH}\n"
        f"git_sha={scorecard['git_sha']}\n"
        f"tieout: {overall['correct']}/{overall['total']} ({_format_pct(overall['accuracy'])})\n"
        f"judge: {judge_summary}"
    )


if __name__ == "__main__":
    main()
