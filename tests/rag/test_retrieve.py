"""Tests for app.rag.retrieve: reciprocal_rank_fusion and Retriever.

Hermetic (CLAUDE.md rule 7): no network, no model downloads. The reranker
in these tests is always ``_FakeReranker`` (a plain in-test double), never
``CrossEncoderReranker``, so ``sentence-transformers`` is never touched.
Vector-side tests use the ``FakeEmbedder`` fixture from
``tests/rag/conftest.py``.
"""

from __future__ import annotations

import pytest

from app.rag.config import RagSettings
from app.rag.index import HybridIndex
from app.rag.retrieve import Retriever, reciprocal_rank_fusion
from app.schemas import Chunk
from tests.rag.conftest import FakeEmbedder

SOURCE_URL = "https://www.sec.gov/Archives/edgar/data/0000320193/example.htm"

_REVENUE_CHUNK = Chunk(
    chunk_id="item7-0000",
    section="Item 7. Management's Discussion and Analysis",
    text=(
        "Item 7. Management's Discussion and Analysis. Revenue increased "
        "due to strong iPhone sales in the current fiscal year."
    ),
    source_url=SOURCE_URL,
)

_LITIGATION_CHUNK = Chunk(
    chunk_id="item7-0001",
    section="Item 7. Management's Discussion and Analysis",
    text=(
        "Item 7. Management's Discussion and Analysis. Litigation "
        "contingencies related to patent disputes remain unresolved."
    ),
    source_url=SOURCE_URL,
)

_SUPPLY_CHAIN_CHUNK = Chunk(
    chunk_id="item1a-0000",
    section="Item 1A. Risk Factors",
    text=(
        "Item 1A. Risk Factors. Supply chain disruptions could materially "
        "affect manufacturing output at contract facilities."
    ),
    source_url=SOURCE_URL,
)

_CURRENCY_CHUNK = Chunk(
    chunk_id="item1a-0001",
    section="Item 1A. Risk Factors",
    text=(
        "Item 1A. Risk Factors. Foreign currency exchange rate fluctuations "
        "may reduce reported revenue in future periods."
    ),
    source_url=SOURCE_URL,
)

_ALL_CHUNKS: list[Chunk] = [
    _REVENUE_CHUNK,
    _LITIGATION_CHUNK,
    _SUPPLY_CHAIN_CHUNK,
    _CURRENCY_CHUNK,
]


class _FakeReranker:
    """Deterministic in-test double for ``Reranker``.

    Scores are looked up by exact chunk text (what ``Retriever`` actually
    passes to ``score``), not by chunk id, so a test can script exactly
    which chunk should win regardless of retrieval order.
    """

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        """Store the fixed text -> score lookup table.

        Args:
            scores_by_text: Maps a chunk's exact ``text`` to the score this
                fake should report for it.
        """
        self._scores_by_text = scores_by_text

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return the scripted score for each text, ignoring ``query``.

        Args:
            query: Unused; accepted to satisfy the ``Reranker`` protocol.
            texts: Texts to look up in the scripted table.

        Returns:
            One score per element of ``texts``, in order.
        """
        return [self._scores_by_text[text] for text in texts]


def test_reciprocal_rank_fusion_matches_hand_computed_scores() -> None:
    """RRF fused scores exactly match a hand-computed example (k=10).

    rankings = [["a", "b", "c"], ["c", "a"]]
    a: rank 1 in list 1 (1/11) + rank 2 in list 2 (1/12) = 23/132
    b: rank 2 in list 1 (1/12) only                       = 1/12
    c: rank 3 in list 1 (1/13) + rank 1 in list 2 (1/11)  = 24/143
    Expected order by descending score: a > c > b.
    """
    rankings = [["a", "b", "c"], ["c", "a"]]

    fused = reciprocal_rank_fusion(rankings, k=10)

    fused_by_id = dict(fused)
    assert fused_by_id["a"] == pytest.approx(1 / 11 + 1 / 12)
    assert fused_by_id["b"] == pytest.approx(1 / 12)
    assert fused_by_id["c"] == pytest.approx(1 / 13 + 1 / 11)
    assert [item_id for item_id, _ in fused] == ["a", "c", "b"]


def test_reciprocal_rank_fusion_of_empty_rankings_is_empty() -> None:
    """Fusing zero (or all-empty) rankings yields an empty list, not an error."""
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_retriever_orders_by_reranker_scores_not_raw_rrf_order() -> None:
    """A fake reranker's scores -- not raw BM25/RRF order -- decide the final order.

    ``BM25Index.query`` correctly excludes non-positive-score chunks (a
    genuine relevance floor, see its docstring), so the query below adds one
    distinguishing word from each non-revenue chunk (``patent``,
    ``manufacturing``, ``exchange``) on top of the original "iPhone sales
    revenue" query. That gives every chunk in ``_ALL_CHUNKS`` a genuine,
    strictly positive keyword overlap with the query -- and thus a real
    place in the pre-rerank pool -- while the revenue chunk still legitimately
    scores highest on raw BM25 (it matches three query terms instead of one),
    so the sanity check below still demonstrates a real reordering.
    """
    index = HybridIndex.build(_ALL_CHUNKS, embedder=None)
    query = "iPhone sales revenue patent manufacturing exchange"

    # Sanity check: BM25 alone ranks the revenue chunk first for this query,
    # with every other chunk also clearing the positive-score floor.
    bm25_hits = index.bm25_query(query, top_k=4)
    assert bm25_hits[0][0] == _REVENUE_CHUNK.chunk_id
    assert {chunk_id for chunk_id, _ in bm25_hits} == {chunk.chunk_id for chunk in _ALL_CHUNKS}

    # The fake reranker deliberately inverts relevance: litigation "wins".
    fake_reranker = _FakeReranker(
        {
            _REVENUE_CHUNK.text: 0.1,
            _LITIGATION_CHUNK.text: 9.0,
            _SUPPLY_CHAIN_CHUNK.text: 0.2,
            _CURRENCY_CHUNK.text: 0.3,
        }
    )
    settings = RagSettings(final_top_k=4)
    retriever = Retriever(index, fake_reranker, settings)

    results = retriever.retrieve(query)

    assert [result.chunk.chunk_id for result in results] == [
        _LITIGATION_CHUNK.chunk_id,
        _CURRENCY_CHUNK.chunk_id,
        _SUPPLY_CHAIN_CHUNK.chunk_id,
        _REVENUE_CHUNK.chunk_id,
    ]
    assert all(result.scored_by == "rerank" for result in results)
    assert [result.score for result in results] == [9.0, 0.3, 0.2, 0.1]


def test_retriever_returns_empty_when_nothing_clears_rerank_threshold() -> None:
    """A query with no reasonably relevant chunk yields [] rather than raising.

    Simulates what a real cross-encoder does with an off-topic query: every
    candidate scores below the (default) relevance threshold, so retrieval
    fails closed with an empty list -- letting narration correctly refuse.
    """
    index = HybridIndex.build(_ALL_CHUNKS, embedder=None)
    fake_reranker = _FakeReranker({chunk.text: -5.0 for chunk in _ALL_CHUNKS})
    settings = RagSettings(min_rerank_score=0.0)
    retriever = Retriever(index, fake_reranker, settings)

    results = retriever.retrieve("unrelated query about weather forecasts in Antarctica")

    assert results == []


def test_retriever_bm25_only_degraded_mode_uses_rrf_scores() -> None:
    """With no vector side and no reranker, retrieval still ranks and scores via RRF."""
    index = HybridIndex.build(_ALL_CHUNKS, embedder=None)
    assert index.has_vector_index is False

    settings = RagSettings(final_top_k=2)
    retriever = Retriever(index, None, settings)

    results = retriever.retrieve("patent litigation dispute")

    assert results
    assert len(results) <= 2
    assert results[0].chunk.chunk_id == _LITIGATION_CHUNK.chunk_id
    assert all(result.scored_by == "rrf" for result in results)


def test_retriever_fuses_bm25_and_vector_rankings_when_both_available() -> None:
    """With a vector side present, both BM25 and vector rankings feed the fusion."""
    embedder = FakeEmbedder()
    index = HybridIndex.build(_ALL_CHUNKS, embedder=embedder, persist_dir=None)
    assert index.has_vector_index is True

    settings = RagSettings(final_top_k=4)
    retriever = Retriever(index, None, settings)

    results = retriever.retrieve(_CURRENCY_CHUNK.text)

    assert results
    # Querying with a chunk's own text should surface that chunk highly via
    # the vector side even though it also participates in BM25 fusion.
    assert results[0].chunk.chunk_id == _CURRENCY_CHUNK.chunk_id
    assert all(result.scored_by == "rrf" for result in results)
