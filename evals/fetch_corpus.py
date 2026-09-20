"""One-time, network-using build utility: fetch a real MSFT 10-K for the retrieval eval corpus.

Like `tests/fixtures/make_fixtures.py`, this is a build-time utility, not a
hermetic test: it makes real network calls to SEC EDGAR and is never invoked
by `make test`, `make evals`, or CI (CLAUDE.md rule 7). Run it manually,
once, whenever `evals/datasets/corpus/msft_10k_excerpt.html` needs to be
(re)generated:

    uv run python -m evals.fetch_corpus

It:

1. Reads `tests/fixtures/submissions_MSFT.json` for every 10-K filing row,
   newest `reportDate` first.
2. For each row in that order, fetches the filing's real HTML from SEC via
   `app.edgar.client.EdgarClient` (host allowlist, rate limiting, and disk
   caching all apply exactly as in production -- this script never bypasses
   them) and parses the FULL, untrimmed HTML with `app.rag.parse_filing`,
   reporting chunk counts and total characters for Item 1A and Item 7.
3. Picks the newest row whose Item 1A *and* Item 7 chunk counts both clear
   `_MIN_CHUNKS`/`_MIN_CHARS` (see `_is_reasonable`) -- i.e. the newest
   filing this repo's ingest pipeline can actually parse into non-trivial
   sections. See "Why a fallback search, not just the newest filing" below.
4. Writes the (trimmed only if necessary) HTML of the selected filing to
   `evals/datasets/corpus/msft_10k_excerpt.html`.
5. Re-parses the final committed file and reports its sha256, byte length,
   per-section chunk counts, and the first ~150 characters of every chunk,
   so whoever hand-labels retrieval queries can see what is actually in
   each chunk.

Why a fallback search, not just the newest filing
---------------------------------------------------
The literal newest four MSFT 10-Ks on file when this was written (fiscal
years 2023-2026, reportDates 2023-06-30 through 2026-06-30) all render their
"ITEM 1A. RISK FACTORS" heading as two *sibling, identically-styled* `<span>`
elements that split the word "RISK" itself mid-word -- e.g.
`<span ...>ITEM 1A. RIS</span><span ...>K FACTORS</span>` with no tag or
style difference between them, evidently an artifact of that filer
template's PDF-to-HTML text-run conversion, not a markup or accessibility
difference our parser could reasonably see. `app.rag.ingest._clean_text`
puts a newline between every distinct source string, so this specific
mid-word split lands "ITEM 1A. RIS" and "K FACTORS" on two different
*lines* of the cleaned text `app.rag.ingest._extract_section` scans -- and
that module's line-anchored heading regex (by design; see its own
docstring on defeating the table-of-contents trap) never matches a heading
split across two lines. The result on those four filings is not a
truncated-but-real Item 1A section: it is the *table-of-contents mention*
surviving alone (that one-line TOC entry is not itself split), a 1-chunk,
~24-46 character non-result. Item 7 is unaffected on those same filings
(its heading text happens not to be split this way), which is why its
chunk counts look normal even on the affected rows -- confirmed directly
by fetching and parsing all six rows before writing anything (see the
per-row report this script prints). The two oldest 10-Ks on file (fiscal
years 2021 and 2022, a different filer template/accession prefix) do not
have this defect and parse into 25 substantial Item 1A chunks each.

This is a real, reproducible limitation of `app/rag/ingest.py` on this
specific real-world markup pattern -- but `app/rag` is the rag-engineer's
lane (CLAUDE.md ownership map) and this script must not edit it. Silently
shipping the newest filing's broken 1-chunk Item 1A "section" would poison
the retrieval eval with a corpus that cannot answer any Item 1A query by
construction, which is worse than reporting the deviation and using the
newest filing that actually parses -- so that is what this script does,
loudly (every rejected row is printed with its reason), rather than either
fabricating content or silently narrowing the eval's scope.

Trimming policy (see `docs/BUILDLOG.md` for the ingest.py bugs this is
trying to avoid repeating): only trim for repo-size reasons if the fetched
HTML is dramatically large (over ~4 MB). If trimming, keep a wide,
generous buffer of whole document structure around Item 1A and Item 7 --
erring on keeping more surrounding content, never narrowly slicing right at
a heading, since narrow/careless trimming previously truncated or displaced
a section by cutting across an inline cross-reference.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from app.edgar.client import EdgarClient
from app.rag.ingest import parse_filing
from app.schemas import Chunk
from app.settings import RagSettings, get_edgar_settings

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
CORPUS_DIR = Path(__file__).resolve().parent / "datasets" / "corpus"
MSFT_SUBMISSIONS = FIXTURES_DIR / "submissions_MSFT.json"
OUTPUT_PATH = CORPUS_DIR / "msft_10k_excerpt.html"

# Only trim for repo-size reasons above this threshold; see module docstring.
TRIM_THRESHOLD_BYTES = 4_000_000

# Generous buffer (characters) kept around a found "Item 1A"/"Item 7" heading
# if trimming is actually needed -- wide on purpose, see module docstring.
_TRIM_BUFFER_CHARS = 400_000

# A section is "reasonable" (non-trivial, usable for hand-labeling) if it has
# at least this many chunks AND at least this many total characters. The
# defective filings described in the module docstring produce exactly 1
# chunk / 24-46 characters for Item 1A, nowhere close to either bound; the
# good filings produce 20+ chunks / tens of thousands of characters for both
# sections. See "Why a fallback search" above.
_MIN_CHUNKS = 5
_MIN_CHARS = 2_000


def _all_10k_rows(submissions: dict[str, Any]) -> list[dict[str, str]]:
    """Return every 10-K row from a submissions JSON, newest `reportDate` first.

    Args:
        submissions: Parsed `submissions_MSFT.json`-shaped JSON (SEC
            "submissions" feed shape: parallel arrays under
            `filings.recent`).

    Returns:
        A list of dicts, each with `cik`, `accession_number`,
        `primary_document`, `report_date`, and `filing_date`, sorted newest
        first.

    Raises:
        ValueError: If no row has `form == "10-K"`.
    """
    cik = str(submissions["cik"])
    recent = submissions["filings"]["recent"]
    forms = recent["form"]
    report_dates = recent["reportDate"]
    candidates = [i for i, form in enumerate(forms) if form == "10-K"]
    if not candidates:
        raise ValueError("no 10-K row found in submissions feed")
    ordered = sorted(candidates, key=lambda i: report_dates[i], reverse=True)
    return [
        {
            "cik": cik,
            "accession_number": recent["accessionNumber"][i],
            "primary_document": recent["primaryDocument"][i],
            "report_date": report_dates[i],
            "filing_date": recent["filingDate"][i],
        }
        for i in ordered
    ]


def _section_chunk_counts(chunks: list[Chunk]) -> dict[str, int]:
    """Count chunks per section prefix (e.g. `"item1a"`, `"item7"`).

    Args:
        chunks: Chunks produced by `app.rag.parse_filing`.

    Returns:
        A mapping of section-key prefix (the part of `chunk_id` before the
        trailing `-NNNN` index) to chunk count.
    """
    counts: Counter[str] = Counter()
    for chunk in chunks:
        prefix = chunk.chunk_id.rsplit("-", 1)[0]
        counts[prefix] += 1
    return dict(sorted(counts.items()))


def _section_char_counts(chunks: list[Chunk]) -> dict[str, int]:
    """Sum chunk text length per section prefix (e.g. `"item1a"`, `"item7"`).

    Args:
        chunks: Chunks produced by `app.rag.parse_filing`.

    Returns:
        A mapping of section-key prefix to total character count across all
        of that section's chunks.
    """
    totals: Counter[str] = Counter()
    for chunk in chunks:
        prefix = chunk.chunk_id.rsplit("-", 1)[0]
        totals[prefix] += len(chunk.text)
    return dict(sorted(totals.items()))


def _is_reasonable(chunks: list[Chunk]) -> bool:
    """Return whether both Item 1A and Item 7 look like real, non-trivial sections.

    Args:
        chunks: Chunks produced by `app.rag.parse_filing` on one filing's
            full HTML.

    Returns:
        `True` only if at least one `item1a*` and one `item7*` chunk-id
        prefix each meet `_MIN_CHUNKS` chunks and `_MIN_CHARS` total
        characters; `False` otherwise (including when a section is absent
        entirely).
    """
    counts = _section_chunk_counts(chunks)
    chars = _section_char_counts(chunks)
    for prefix in ("item1a", "item7"):
        if counts.get(prefix, 0) < _MIN_CHUNKS or chars.get(prefix, 0) < _MIN_CHARS:
            return False
    return True


def _widen_span(html: str, start: int, end: int, buffer_chars: int) -> tuple[int, int]:
    """Widen a `[start, end)` span by `buffer_chars` on each side, clamped to `html`'s bounds.

    Args:
        html: The full document the span was found in.
        start: Span start offset.
        end: Span end offset.
        buffer_chars: Characters to add on each side.

    Returns:
        A `(widened_start, widened_end)` tuple, clamped to `[0, len(html)]`.
    """
    return max(0, start - buffer_chars), min(len(html), end + buffer_chars)


def _maybe_trim(html: str) -> str:
    """Trim `html` only if it is dramatically large; otherwise return it unchanged.

    When trimming is needed, this keeps one wide, generous, contiguous span
    of the *raw HTML* (not the cleaned text `app.rag.ingest` works on)
    covering both an "Item 1A" and an "Item 7" heading occurrence with a
    large buffer on each side -- erring on keeping more surrounding
    document structure rather than slicing tightly at a heading, per this
    module's docstring and the `docs/BUILDLOG.md` history of narrow
    trimming truncating/displacing a section.

    Args:
        html: The full, untrimmed filing HTML.

    Returns:
        `html` unchanged if it is at or under `TRIM_THRESHOLD_BYTES`;
        otherwise a trimmed slice of `html` spanning from a wide buffer
        before the first "Item 1A" heading occurrence through a wide
        buffer after the last "Item 7"/"Item 7A" heading occurrence.
    """
    if len(html.encode("utf-8")) <= TRIM_THRESHOLD_BYTES:
        return html

    # Loose, HTML-tolerant heading finders (case-insensitive, whitespace/tag
    # noise between words is fine to ignore here -- this only bounds a wide
    # slice of raw HTML, not final section extraction, which is
    # `app.rag.ingest`'s job and reruns on the trimmed file afterward).
    item1a_re = re.compile(r"item\s*1a", re.IGNORECASE)
    item7a_re = re.compile(r"item\s*7a", re.IGNORECASE)
    item8_re = re.compile(r"item\s*8\b", re.IGNORECASE)

    first_1a = item1a_re.search(html)
    # Prefer the last "Item 8" occurrence as the end anchor (covers Item 7 +
    # 7A with margin); fall back to the last "Item 7A" if Item 8 is absent.
    end_matches = list(item8_re.finditer(html)) or list(item7a_re.finditer(html))
    if first_1a is None or not end_matches:
        # Couldn't confidently bound a slice -- keep the whole document
        # rather than risk cutting through the sections we need.
        return html
    start, _ = _widen_span(html, first_1a.start(), first_1a.start(), _TRIM_BUFFER_CHARS)
    _, end = _widen_span(html, end_matches[-1].start(), end_matches[-1].start(), _TRIM_BUFFER_CHARS)
    return html[start:end]


def _print_chunk_previews(chunks: list[Chunk]) -> None:
    """Print the chunk id, section, and first ~150 characters of every chunk's text.

    Args:
        chunks: Chunks to preview.
    """
    for chunk in chunks:
        preview = chunk.text[:150].replace("\n", " ")
        print(f"  {chunk.chunk_id} [{chunk.section}]: {preview!r}")


def _select_filing(client: EdgarClient, rows: list[dict[str, str]]) -> tuple[dict[str, str], str]:
    """Fetch each row newest-first until one parses into reasonable Item 1A/7 sections.

    Prints a per-row report (bytes fetched, chunk counts, total characters)
    for every row tried, including rejected ones, so the fallback is fully
    auditable rather than silent. See the module docstring's "Why a
    fallback search" section.

    Args:
        client: An `EdgarClient` to fetch filing HTML with.
        rows: 10-K rows as returned by `_all_10k_rows`, newest first.

    Returns:
        A `(row, html)` tuple for the newest row whose full, untrimmed HTML
        parses into reasonable Item 1A and Item 7 sections per
        `_is_reasonable`.

    Raises:
        SystemExit: If no row in `rows` produces a reasonable parse -- this
            script must never fabricate a corpus, so it refuses to write
            one rather than silently shipping a defective 1-chunk section.
    """
    for row in rows:
        html = client.filing_html(row["cik"], row["accession_number"], row["primary_document"])
        full_bytes = len(html.encode("utf-8"))
        chunks = parse_filing(html, source_url="https://www.sec.gov/", settings=RagSettings())
        counts = _section_chunk_counts(chunks)
        chars = _section_char_counts(chunks)
        reasonable = _is_reasonable(chunks)
        print(
            f"row reportDate={row['report_date']} accession={row['accession_number']} "
            f"doc={row['primary_document']} bytes={full_bytes:,} "
            f"chunk_counts={counts} chunk_chars={chars} "
            f"-> {'ACCEPTED' if reasonable else 'rejected (not reasonable)'}"
        )
        if reasonable:
            return row, html
    raise SystemExit(
        "no MSFT 10-K on file parsed into reasonable Item 1A and Item 7 sections "
        "with app.rag.parse_filing -- refusing to write a fabricated or "
        "under-covered corpus; see the printed per-row report above"
    )


def main() -> None:
    """Select the newest parseable real MSFT 10-K and write the eval corpus file."""
    submissions = json.loads(MSFT_SUBMISSIONS.read_text(encoding="utf-8"))
    rows = _all_10k_rows(submissions)
    print(f"{len(rows)} 10-K row(s) on file, newest first; searching for a parseable one...")

    with EdgarClient(settings=get_edgar_settings()) as client:
        row, html = _select_filing(client, rows)

    full_bytes = len(html.encode("utf-8"))
    print(
        f"\nselected MSFT 10-K: reportDate={row['report_date']} "
        f"filingDate={row['filing_date']} accession={row['accession_number']} "
        f"doc={row['primary_document']} ({full_bytes:,} bytes)"
    )

    final_html = _maybe_trim(html)
    if len(final_html) == len(html):
        print("full HTML is not dramatically large -- keeping it as-is, no trimming")
    else:
        print(
            f"trimmed {full_bytes:,} bytes -> {len(final_html.encode('utf-8')):,} bytes "
            "(wide buffer kept around Item 1A .. Item 7/7A/8)"
        )

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(final_html, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")

    final_bytes = OUTPUT_PATH.read_bytes()
    digest = hashlib.sha256(final_bytes).hexdigest()
    final_chunks = parse_filing(
        final_html, source_url="https://www.sec.gov/", settings=RagSettings()
    )
    final_counts = _section_chunk_counts(final_chunks)

    print("\n--- final committed file ---")
    print(f"path: {OUTPUT_PATH}")
    print(f"source_report_date: {row['report_date']}")
    print(f"source_accession_number: {row['accession_number']}")
    print(f"source_primary_document: {row['primary_document']}")
    print(f"sha256: {digest}")
    print(f"byte_length: {len(final_bytes)}")
    print(f"chunk_count: {len(final_chunks)}")
    print(f"chunk_count_by_section: {final_counts}")
    print("\nchunk previews:")
    _print_chunk_previews(final_chunks)


if __name__ == "__main__":
    main()
