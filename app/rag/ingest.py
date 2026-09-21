"""Parse a 10-K filing's HTML into retrieval-ready chunks.

The only public entrypoint is :func:`parse_filing`. It makes no network
calls: it operates purely on an HTML string already fetched by the caller
(``app/edgar`` owns the actual SEC fetch; that module is a different
ownership lane, see CLAUDE.md).

Every 10-K names each Item heading twice: once in a short table-of-contents
listing ("Item 7. Management's Discussion... 21") and once as the real
section header followed by its body. Section extraction must not be fooled
into returning the TOC line — see :func:`_extract_section` for how that trap
is defeated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from app.schemas import Chunk
from app.settings import RagSettings


def _heading_re(item: str, title_start: str) -> re.Pattern[str]:
    """Build a regex matching a genuine "Item <n>. <Title>" heading.

    Only an optional period/colon and whitespace are allowed between the
    item number and the start of its title. That is enough to match both
    the table-of-contents line and the real section header, but not the
    many inline cross-references scattered through a filing's body (e.g.
    "...see Item 7 of this Form 10-K under the heading..." or
    "...Item 7, “Management's Discussion...”"), which always have
    other words or punctuation between the item number and the title.

    Args:
        item: The item number/letter, e.g. "7" or "1a".
        title_start: A regex fragment matching the start of the item's
            canonical title, e.g. r"risk\\s+factors".

    Returns:
        A compiled, case-insensitive regex.
    """
    return re.compile(rf"item\s*{item}\.?:?\s*{title_start}", re.IGNORECASE)


def _bare_heading_re(title: str) -> re.Pattern[str]:
    """Build a regex matching a section heading that carries no "Item <n>." prefix.

    Some filers label their sections only in the table of contents and head
    the bodies with the bare title (McDonald's 10-K heads its sections
    "RISK FACTORS" and "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL
    CONDITION AND RESULTS OF OPERATIONS", and the string "Item 1A" appears
    exactly once in the whole document -- in the contents table). Matching
    those needs a looser anchor than :func:`_heading_re`, so this one is
    tightened in a different direction instead: the title must be the
    *entire* line, which is what separates a heading from the many prose
    sentences that open with the same words ("Management's Discussion and
    Analysis of Financial Condition and Results of Operations is based upon
    the Company's Consolidated Financial Statements...").

    Args:
        title: A regex fragment matching the section's full canonical title.

    Returns:
        A compiled, case-insensitive, line-anchored regex.
    """
    return re.compile(rf"^[ \t]*{title}[ \t]*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class _SectionSpec:
    """Where to look for one target section and how to tell it's over."""

    key: str  # short id used in chunk_id, e.g. "item7"
    label: str  # human-readable section label stored on each Chunk
    heading_pattern: re.Pattern[str]
    boundary_patterns: tuple[re.Pattern[str], ...]
    # Some filers label sections only in the contents table and head the
    # bodies with the bare title; see `_extract_section`.
    bare_heading_pattern: re.Pattern[str]
    bare_boundary_patterns: tuple[re.Pattern[str], ...]


_ITEM1A = _SectionSpec(
    key="item1a",
    label="Item 1A. Risk Factors",
    heading_pattern=_heading_re("1a", r"risk\s+factors"),
    boundary_patterns=(
        _heading_re("1b", r"unresolved\s+staff\s+comments"),
        _heading_re("2", r"properties"),
    ),
    bare_heading_pattern=_bare_heading_re(r"risk\s+factors"),
    bare_boundary_patterns=(
        _bare_heading_re(r"unresolved\s+staff\s+comments"),
        _bare_heading_re(r"properties"),
    ),
)

_ITEM7 = _SectionSpec(
    key="item7",
    label="Item 7. Management's Discussion and Analysis",
    heading_pattern=_heading_re("7", r"management.?s?\s+discussion"),
    boundary_patterns=(
        _heading_re("7a", r"quantitative"),
        _heading_re("8", r"financial\s+statements"),
    ),
    bare_heading_pattern=_bare_heading_re(
        r"management.?s?\s+discussion\s+and\s+analysis"
        r"(?:\s+of\s+financial\s+condition\s+and\s+results\s+of\s+operations)?"
    ),
    bare_boundary_patterns=(
        _bare_heading_re(r"quantitative\s+and\s+qualitative[^\n]*"),
        _bare_heading_re(r"financial\s+statements\s+and\s+supplementary\s+data"),
    ),
)

_SECTIONS: tuple[_SectionSpec, ...] = (_ITEM1A, _ITEM7)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _clean_text(html: str) -> str:
    """Strip script/style tags and normalize whitespace, preserving block breaks.

    Block/paragraph structure is preserved as newlines: text is extracted
    with ``\\n`` as BeautifulSoup's string separator (rather than a single
    ``get_text(" ")`` call), so every distinct source text node -- roughly,
    every paragraph/block/table-cell -- lands on its own line. Only
    intra-line horizontal whitespace (spaces, tabs, ``&nbsp;``/U+00A0) is
    then collapsed; the newlines are kept as real structural markers.

    This structure is what lets :func:`_extract_section` tell a genuine
    heading (which starts its own block in the source HTML, and therefore
    its own line here) apart from an ordinary word or phrase occurring
    mid-sentence in running prose (which does not) -- see
    :func:`_is_line_start`. Blank lines (from empty tags) are dropped.

    Args:
        html: Raw filing HTML.

    Returns:
        Cleaned plain text with normalized intra-line whitespace and one
        source block per line.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def _is_line_start(text: str, pos: int) -> bool:
    """Return whether ``pos`` sits at the start of one of ``text``'s lines.

    A genuine heading occupies its own block in the source HTML and
    therefore its own line in :func:`_clean_text`'s output; an inline
    cross-reference embedded mid-sentence in ordinary prose does not --
    it is preceded on the same line by other words and a plain space, not
    a newline or start-of-string. Requiring this anchoring is what stops
    an inline mention (e.g. "...our results, see Item 8. Financial
    Statements...") from being mistaken for a real heading/boundary match.

    Args:
        text: The text a match position falls within.
        pos: Character offset to check.

    Returns:
        ``True`` if only whitespace (or nothing) precedes ``pos`` on its
        line, ``False`` otherwise.
    """
    line_start = text.rfind("\n", 0, pos) + 1
    return text[line_start:pos].strip() == ""


def _extract_section(text: str, spec: _SectionSpec) -> str | None:
    """Return the real section's body text, defeating the TOC-mention trap.

    Finds every case-insensitive candidate start position for the section's
    heading that is anchored at the start of a line (see
    :func:`_is_line_start`) -- this rejects ordinary inline cross-references
    scattered through a filing's body, which occur mid-sentence rather than
    at the start of their own block, while still matching both a genuine
    heading and a table-of-contents line (both of which really do start
    their own block in the source HTML). Each surviving candidate is then
    scored by the span from itself to the nearest following (also
    line-anchored) boundary heading, or end of document if none follows.
    The largest span wins: the table-of-contents mention is always short
    (the next TOC line follows immediately), while the real section runs
    for its full body before the next real heading appears.

    Args:
        text: Cleaned filing text with preserved block/line structure (see
            :func:`_clean_text`).
        spec: Which section to extract and what bounds it.

    Returns:
        The extracted section text (heading through the boundary), or
        ``None`` if no candidate heading was found at all.
    """
    # Both heading shapes compete on equal footing, and the span scoring
    # below picks between them. Adding the bare-title form cannot promote a
    # contents-table line over a real body heading: the bare boundary titles
    # come with it, so a contents-table candidate's span shrinks to the
    # distance to the *next* contents line, which is what already made the
    # item-prefixed form robust.
    candidates = sorted(
        {
            m.start()
            for pattern in (spec.heading_pattern, spec.bare_heading_pattern)
            for m in pattern.finditer(text)
            if _is_line_start(text, m.start())
        }
    )
    if not candidates:
        return None
    boundaries = sorted(
        {
            m.start()
            for pattern in spec.boundary_patterns + spec.bare_boundary_patterns
            for m in pattern.finditer(text)
            if _is_line_start(text, m.start())
        }
    )
    best_start, best_end, best_span = candidates[0], len(text), -1
    for start in candidates:
        end = next((b for b in boundaries if b > start), len(text))
        span = end - start
        if span > best_span:
            best_start, best_end, best_span = start, end, span
    extracted = text[best_start:best_end].strip()
    # Collapse the preserved line breaks back to single spaces in the
    # returned section text: they were only needed as structural anchors
    # for the matching above, and downstream chunking/citation code expects
    # the same single-spaced shape a model would naturally reproduce
    # verbatim (see the old _clean_text docstring behavior).
    return re.sub(r"\s+", " ", extracted).strip()


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence-ending punctuation.

    Args:
        text: Plain text to split.

    Returns:
        A list of sentence-ish fragments; the whole text as a single
        element if no sentence boundary was found.
    """
    parts = [p for p in _SENTENCE_SPLIT.split(text) if p]
    return parts or [text]


def _chunk_section(label: str, body: str, *, target_tokens: int, overlap_tokens: int) -> list[str]:
    """Split one section's body into ~target_tokens chunks with overlap.

    Tokens are approximated as ``len(text) // 4`` throughout (no tokenizer
    dependency, per CLAUDE.md rule 7 on hermetic tests). Splits fall on
    sentence boundaries, and each returned chunk is prefixed with the
    section's heading so it carries that context on its own.

    Args:
        label: The section heading to prefix onto every chunk.
        body: The section's extracted body text.
        target_tokens: Approximate target chunk size, in tokens.
        overlap_tokens: Approximate overlap carried into the next chunk, in
            tokens.

    Returns:
        A list of chunk text strings, each already prefixed with ``label``.
    """
    target_chars = max(target_tokens * 4, 1)
    overlap_chars = max(overlap_tokens * 4, 0)
    sentences = _split_sentences(body)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for index, sentence in enumerate(sentences):
        current.append(sentence)
        current_len += len(sentence) + 1
        is_last = index == len(sentences) - 1
        if current_len >= target_chars or is_last:
            chunk_body = " ".join(current).strip()
            chunks.append(f"{label} {chunk_body}".strip())
            if is_last:
                break
            # Carry the trailing ~overlap_chars worth of sentences forward.
            carried: list[str] = []
            carried_len = 0
            for prior in reversed(current):
                if carried_len >= overlap_chars:
                    break
                carried.insert(0, prior)
                carried_len += len(prior) + 1
            current, current_len = carried, sum(len(s) + 1 for s in carried)
    return chunks or [f"{label} {body}".strip()]


def parse_filing(
    html: str,
    *,
    source_url: str,
    settings: RagSettings | None = None,
) -> list[Chunk]:
    """Parse a 10-K's raw HTML into retrieval-ready :class:`Chunk` objects.

    Extracts "Item 1A. Risk Factors" and "Item 7. Management's Discussion and
    Analysis" (defeating the table-of-contents trap, see
    :func:`_extract_section`), then splits each into overlapping,
    ~``chunk_target_tokens``-sized chunks on sentence boundaries. Chunk ids
    are deterministic and readable: ``"<section-key>-<0-based index, zero
    padded to 4 digits>"``, e.g. ``"item7-0003"``, ``"item1a-0012"``. Makes
    no network calls.

    Args:
        html: Raw filing HTML, already fetched by the caller.
        source_url: The URL the filing was fetched from; copied onto every
            chunk so citations can point back to their source document.
        settings: Chunking configuration; defaults to ``RagSettings()``.

    Returns:
        A list of :class:`Chunk` objects, ordered section by section and
        sequentially within each section. Sections that cannot be found are
        silently skipped, so the result may be shorter than expected but is
        never fabricated.
    """
    cfg = settings or RagSettings()
    text = _clean_text(html)

    chunks: list[Chunk] = []
    for spec in _SECTIONS:
        body = _extract_section(text, spec)
        if not body:
            continue
        pieces = _chunk_section(
            spec.label,
            body,
            target_tokens=cfg.chunk_target_tokens,
            overlap_tokens=cfg.chunk_overlap_tokens,
        )
        for index, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{spec.key}-{index:04d}",
                    section=spec.label,
                    text=piece,
                    source_url=source_url,
                )
            )
    return chunks
