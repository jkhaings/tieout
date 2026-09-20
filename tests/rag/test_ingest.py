"""Tests for app.rag.ingest.parse_filing.

Hermetic (CLAUDE.md rule 7): operates only on the real, trimmed HTML fixture
in tests/fixtures/aapl_10k_excerpt.html. No network.
"""

from __future__ import annotations

import re

from app.rag.config import RagSettings
from app.rag.ingest import parse_filing
from app.schemas import Chunk

SOURCE_URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"

# The fixture's genuine table-of-contents blurbs are single lines like
# "Item 1A. Risk Factors 5" or "Item 7. Management's Discussion... 21" --
# tens of characters. Any correctly extracted section must be far longer.
TOC_BLURB_CEILING_CHARS = 500


def _chunks_for(chunks: list[Chunk], prefix: str) -> list[Chunk]:
    return [c for c in chunks if c.chunk_id.startswith(prefix)]


def test_both_target_sections_are_found(aapl_10k_excerpt_html: str) -> None:
    """Item 1A and Item 7 are both extracted despite the TOC trap."""
    chunks = parse_filing(aapl_10k_excerpt_html, source_url=SOURCE_URL)
    sections = {c.section for c in chunks}
    assert "Item 1A. Risk Factors" in sections
    assert "Item 7. Management's Discussion and Analysis" in sections


def test_extracted_sections_defeat_the_toc_trap(aapl_10k_excerpt_html: str) -> None:
    """Extracted section text is far longer than any TOC blurb and holds real body content."""
    chunks = parse_filing(aapl_10k_excerpt_html, source_url=SOURCE_URL)

    item1a_text = " ".join(c.text for c in _chunks_for(chunks, "item1a"))
    item7_text = " ".join(c.text for c in _chunks_for(chunks, "item7"))

    # Far longer than a TOC listing could ever be.
    assert len(item1a_text) > TOC_BLURB_CEILING_CHARS * 10
    assert len(item7_text) > TOC_BLURB_CEILING_CHARS * 10

    # Real body content, not just the heading -- these phrases only occur
    # deep inside the real sections in the source filing, never in the TOC.
    assert "material adverse effect" in item1a_text.lower()
    assert "results of operations" in item7_text.lower()


def test_chunk_sizes_stay_near_target(aapl_10k_excerpt_html: str) -> None:
    """Chunk sizes stay near the ~800 token (len//4) target."""
    settings = RagSettings()
    chunks = parse_filing(aapl_10k_excerpt_html, source_url=SOURCE_URL, settings=settings)

    target = settings.chunk_target_tokens
    # Every chunk but the final one in each section should land close to
    # the target; a lone trailing remainder chunk may be smaller.
    for prefix in ("item1a", "item7"):
        section_chunks = _chunks_for(chunks, prefix)
        assert len(section_chunks) >= 2
        for chunk in section_chunks[:-1]:
            approx_tokens = len(chunk.text) // 4
            assert target * 0.5 <= approx_tokens <= target * 1.5


def test_every_chunk_carries_its_section_heading(aapl_10k_excerpt_html: str) -> None:
    """Every chunk's text carries its section heading for standalone context."""
    chunks = parse_filing(aapl_10k_excerpt_html, source_url=SOURCE_URL)
    assert chunks
    for chunk in chunks:
        assert chunk.text.startswith(chunk.section)


def test_chunk_ids_are_deterministic_across_runs(aapl_10k_excerpt_html: str) -> None:
    """chunk_id lists are identical across two separate parse_filing calls on the same html."""
    first = parse_filing(aapl_10k_excerpt_html, source_url=SOURCE_URL)
    second = parse_filing(aapl_10k_excerpt_html, source_url=SOURCE_URL)

    first_ids = [c.chunk_id for c in first]
    second_ids = [c.chunk_id for c in second]

    assert first_ids == second_ids
    assert first_ids  # non-empty: something was actually extracted


def test_chunk_id_format(aapl_10k_excerpt_html: str) -> None:
    """chunk_id format matches the documented "<section-key>-<4-digit index>" shape."""
    chunks = parse_filing(aapl_10k_excerpt_html, source_url=SOURCE_URL)
    pattern = re.compile(r"^(item1a|item7)-\d{4}$")
    for chunk in chunks:
        assert pattern.match(chunk.chunk_id), chunk.chunk_id

    # Each section's ids are contiguous starting at 0000.
    for prefix in ("item1a", "item7"):
        ids = [c.chunk_id for c in _chunks_for(chunks, prefix)]
        expected = [f"{prefix}-{i:04d}" for i in range(len(ids))]
        assert ids == expected


def test_inline_item8_cross_reference_with_plain_punctuation_does_not_truncate_item7() -> None:
    """An ordinary mid-body "Item 8. Financial Statements..." mention must not be
    mistaken for the real Item 8 heading and truncate Item 7's extracted section.

    Regression test for the finding that ``_extract_section``'s boundary matching
    had no positional anchoring: a plain-punctuation inline cross-reference
    embedded mid-sentence in Item 7's own body (very common MD&A boilerplate,
    e.g. "As further described in Item 8. Financial Statements and Supplementary
    Data, our results improved.") used to be indistinguishable from a genuine
    heading, so it could win as the section's end boundary and truncate the
    returned section to almost nothing.
    """
    filler_sentence = (
        "Net sales increased compared to the prior fiscal year, driven by strong "
        "demand across our product categories and further growth in our Services "
        "segment. "
    )
    real_body_after_cross_reference = filler_sentence * 40
    html = f"""
    <html><body>
    <div>Item 7. Management's Discussion and Analysis of Financial Condition and
    Results of Operations</div>
    <div>The following discussion of our results of operations should be read
    together with our audited financial statements and the related notes.</div>
    <div>As further described in Item 8. Financial Statements and Supplementary
    Data, our results improved.</div>
    <div>{real_body_after_cross_reference}</div>
    <div>Item 8. Financial Statements and Supplementary Data</div>
    <div>Our audited consolidated financial statements are set forth below and
    are an integral part of this Annual Report on Form 10-K.</div>
    </body></html>
    """

    chunks = parse_filing(html, source_url=SOURCE_URL)
    item7_text = " ".join(c.text for c in _chunks_for(chunks, "item7"))

    # The extracted section must reach past the inline mid-body mention and
    # include the substantial real body text that follows it.
    assert len(item7_text) > 2000
    assert "strong demand across our product categories" in item7_text
    # It must stop before the genuine Item 8 heading, never leaking Item 8's
    # own body content into Item 7's section.
    assert "audited consolidated financial statements are set forth below" not in item7_text


def test_prose_cross_reference_to_item7_does_not_displace_the_real_short_section() -> None:
    """A fully spelled-out Item 7 cross-reference embedded in running prose elsewhere
    in the document must not out-span (and thus displace) the real, short Item 7
    section under the largest-span TOC-trap disambiguation.

    Regression test for the finding that a third occurrence of a heading's full
    canonical title elsewhere in the document -- e.g. a cross-reference spelling
    out "Item 7. Management's Discussion and Analysis..." in full -- could, if its
    span-to-next-boundary happened to be larger than the real section's span, win
    the "largest span wins" disambiguation entirely and displace the real section.
    """
    filler_sentence = (
        "This unrelated filler paragraph appears only after the cross-reference "
        "and should never be mistaken for part of the real Item 7 section. "
    )
    trailing_filler = filler_sentence * 60
    html = f"""
    <html><body>
    <div>Item 7. Management's Discussion and Analysis of Financial Condition and
    Results of Operations</div>
    <div>Revenue grew year-over-year due to strong iPhone demand in the current
    fiscal year.</div>
    <div>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</div>
    <div>Our primary market risk exposure relates to foreign currency exchange
    rate fluctuations.</div>
    <div>Investors should also review the discussion in Item 7. Management's
    Discussion and Analysis of Financial Condition and Results of Operations of
    this Annual Report for additional context.</div>
    <div>{trailing_filler}</div>
    <div>Item 8. Financial Statements and Supplementary Data</div>
    <div>Our audited consolidated financial statements are set forth below.</div>
    </body></html>
    """

    chunks = parse_filing(html, source_url=SOURCE_URL)
    item7_text = " ".join(c.text for c in _chunks_for(chunks, "item7"))

    assert "strong iPhone demand" in item7_text
    assert "This unrelated filler paragraph" not in item7_text
    assert "audited consolidated financial statements" not in item7_text


def test_chunks_validate_against_the_frozen_chunk_schema(aapl_10k_excerpt_html: str) -> None:
    """Every produced chunk is a valid app.schemas.Chunk with the right source_url."""
    chunks = parse_filing(aapl_10k_excerpt_html, source_url=SOURCE_URL)
    assert chunks
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.source_url == SOURCE_URL
        assert chunk.text.strip() == chunk.text
        assert chunk.text
