"""Tests for app.rag.index: BM25Index, InMemoryVectorIndex, and HybridIndex.

Hermetic (CLAUDE.md rule 7): only ``rank_bm25`` (a core dependency) and the
``FakeEmbedder`` fixture from ``tests/rag/conftest.py`` are used. No network,
no model downloads, and no real ``chromadb``/``sentence_transformers`` calls
-- ``HybridIndex.build`` is always exercised with ``persist_dir=None`` so it
takes the in-memory path even if the ``ml`` extra happens to be installed in
the environment running the suite.
"""

from __future__ import annotations

from app.rag.index import (
    BM25Index,
    HybridIndex,
    InMemoryVectorIndex,
    is_chromadb_available,
    is_ml_available,
    is_sentence_transformers_available,
)
from app.rag.retrieve import Retriever
from app.schemas import Chunk
from app.settings import RagSettings
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


def test_bm25_ranks_keyword_matching_chunk_above_unrelated_ones() -> None:
    """A query sharing vocabulary with one chunk ranks it first, ahead of unrelated chunks."""
    index = BM25Index.build(_ALL_CHUNKS)

    results = index.query("patent litigation dispute", top_k=4)

    assert results, "expected at least one BM25 result"
    top_chunk_id, top_score = results[0]
    assert top_chunk_id == _LITIGATION_CHUNK.chunk_id
    # Every other returned score should trail the top match.
    assert all(score <= top_score for _, score in results[1:])


def test_bm25_query_excludes_zero_score_results_for_unrelated_query() -> None:
    """A query sharing no vocabulary with any indexed chunk returns zero hits.

    Regression test: without a positive-score floor, BM25Index.query would
    still return up to top_k chunks with a raw score of exactly 0.0 for a
    completely unrelated query (zero keyword overlap), silently defeating
    the relevance threshold downstream retrieval relies on to fail closed
    (CLAUDE.md rule 4) -- RRF fusion turns any *returned* id into a
    strictly positive fused score regardless of how weak the underlying
    match was.
    """
    index = BM25Index.build(_ALL_CHUNKS)

    results = index.query("weather forecast Antarctica penguins snowstorm", top_k=4)

    assert results == []
    assert all(score > 0 for _, score in index.query("iPhone sales revenue", top_k=4))


def test_bm25_query_returns_no_more_than_top_k_results() -> None:
    """BM25Index.query never returns more than top_k results."""
    index = BM25Index.build(_ALL_CHUNKS)

    results = index.query("revenue", top_k=2)

    assert len(results) <= 2


def test_bm25_index_built_from_no_chunks_returns_empty_results() -> None:
    """An empty corpus is handled gracefully (no crash on BM25Okapi([]))."""
    index = BM25Index.build([])

    assert index.query("anything", top_k=5) == []


def test_in_memory_vector_index_ranks_embedding_nearest_chunk_highest() -> None:
    """The chunk whose embedding is nearest the query embedding ranks first."""
    embedder = FakeEmbedder()
    ids = [chunk.chunk_id for chunk in _ALL_CHUNKS]
    embeddings = embedder.embed([chunk.text for chunk in _ALL_CHUNKS])

    index = InMemoryVectorIndex()
    index.add(ids, embeddings)

    # Querying with the exact embedding of one chunk's text must put that
    # chunk first (cosine similarity to itself is maximal).
    [query_embedding] = embedder.embed([_SUPPLY_CHAIN_CHUNK.text])
    results = index.query(query_embedding, top_k=4)

    assert results
    top_chunk_id, top_similarity = results[0]
    assert top_chunk_id == _SUPPLY_CHAIN_CHUNK.chunk_id
    assert all(similarity <= top_similarity for _, similarity in results[1:])


def test_in_memory_vector_index_respects_top_k() -> None:
    """InMemoryVectorIndex.query never returns more than top_k results."""
    embedder = FakeEmbedder()
    ids = [chunk.chunk_id for chunk in _ALL_CHUNKS]
    embeddings = embedder.embed([chunk.text for chunk in _ALL_CHUNKS])

    index = InMemoryVectorIndex()
    index.add(ids, embeddings)

    [query_embedding] = embedder.embed(["irrelevant query text"])
    results = index.query(query_embedding, top_k=1)

    assert len(results) == 1


def test_hybrid_index_without_embedder_supports_bm25_but_reports_no_vector_results() -> None:
    """HybridIndex.build(embedder=None) still answers BM25 queries but has no vector side."""
    index = HybridIndex.build(_ALL_CHUNKS, embedder=None)

    assert index.has_vector_index is False

    bm25_results = index.bm25_query("patent litigation dispute", top_k=4)
    assert bm25_results
    assert bm25_results[0][0] == _LITIGATION_CHUNK.chunk_id

    # No vector side: querying fails closed with an empty list, not an error.
    assert index.vector_query("patent litigation dispute", top_k=4) == []


def test_hybrid_index_with_fake_embedder_uses_in_memory_path_end_to_end() -> None:
    """HybridIndex.build with an embedder and persist_dir=None works end-to-end in memory.

    persist_dir=None forces the in-memory path regardless of whether
    chromadb happens to be installed, keeping this test hermetic.
    """
    embedder = FakeEmbedder()
    index = HybridIndex.build(_ALL_CHUNKS, embedder=embedder, persist_dir=None)

    assert index.has_vector_index is True

    # BM25 side still works.
    bm25_results = index.bm25_query("iPhone sales revenue", top_k=4)
    assert bm25_results
    assert bm25_results[0][0] == _REVENUE_CHUNK.chunk_id

    # Vector side: querying with a chunk's own text ranks that chunk first.
    vector_results = index.vector_query(_CURRENCY_CHUNK.text, top_k=4)
    assert vector_results
    assert vector_results[0][0] == _CURRENCY_CHUNK.chunk_id

    # Embedding happened through the embedder we supplied, once for the
    # corpus and once for our vector_query call above.
    assert embedder.call_count >= 2


def test_hybrid_index_get_chunk_looks_up_indexed_chunks_by_id() -> None:
    """get_chunk returns the matching Chunk, or None for an unknown id."""
    index = HybridIndex.build(_ALL_CHUNKS, embedder=None)

    found = index.get_chunk(_SUPPLY_CHAIN_CHUNK.chunk_id)
    assert found == _SUPPLY_CHAIN_CHUNK
    assert index.get_chunk("does-not-exist") is None


def test_retriever_no_reranker_returns_empty_for_unrelated_query() -> None:
    """Retriever.retrieve(reranker=None) fails closed for a totally unrelated query.

    Mirrors the reranker-based "returns empty for an off-topic query" test in
    tests/rag/test_retrieve.py, but for the no-reranker/RRF-only path -- the
    default mode whenever the `ml` extra isn't installed (i.e. in CI and this
    sandbox). Depends on the BM25Index.query zero-score floor: without it,
    BM25 alone would still return every indexed chunk with a raw score of
    0.0, RRF would turn each into a strictly positive fused score, and this
    query would incorrectly yield non-empty "relevant" results.
    """
    index = HybridIndex.build(_ALL_CHUNKS, embedder=None)
    assert index.has_vector_index is False

    settings = RagSettings()
    retriever = Retriever(index, None, settings)

    results = retriever.retrieve("weather forecast Antarctica penguins snowstorm")

    assert results == []


def test_ml_availability_helpers_return_bool_without_raising() -> None:
    """is_ml_available and friends always return a plain bool, never raise.

    This must hold regardless of whether the optional `ml` extra is
    installed in the environment running the suite.
    """
    for detector in (is_chromadb_available, is_sentence_transformers_available, is_ml_available):
        result = detector()
        assert isinstance(result, bool)
