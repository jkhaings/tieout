"""Fusion, reranking, and thresholding on top of ``app.rag.index.HybridIndex``.

``Retriever`` is what ``app/agent`` (or whoever narrates commentary) actually
calls. It fans a query out to :class:`~app.rag.index.HybridIndex`'s two raw
candidate lists (BM25 keyword, and vector similarity when available), fuses
them into one ranking with :func:`reciprocal_rank_fusion`, optionally
reranks the fused pool with a cross-encoder, and applies the relevance
threshold that decides "no grounded commentary available"
(CLAUDE.md rule 4). An empty return from :meth:`Retriever.retrieve` is a
normal, valid outcome -- weak or absent retrieval -- never an error; it is
exactly what lets downstream narration fail closed instead of fabricating.

RRF-fused scores and cross-encoder rerank scores are on two different,
unrelated numeric scales (rank-position-derived sums-of-reciprocals vs.
model-specific logits), so they are never compared against the same
threshold constant: :attr:`~app.settings.RagSettings.min_fused_score`
gates un-reranked results, :attr:`~app.settings.RagSettings.min_rerank_score`
gates reranked ones.

Degraded-mode support (CLAUDE.md rule "guard chromadb / sentence_transformers
imports"): ``sentence_transformers`` is optional (the ``ml`` extra) and is
imported dynamically via :func:`importlib.import_module` into a module-level
variable typed ``Any``, inside a ``try/except ImportError`` that leaves the
variable ``None`` on failure -- the same pattern ``app/rag/index.py`` uses
for ``chromadb``/``sentence_transformers``. This keeps the module importable
everywhere; only *using* :class:`CrossEncoderReranker` without the extra
installed raises, and only when actually invoked.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from app.rag.index import HybridIndex
from app.schemas import Chunk
from app.settings import RagSettings

# How many of the top fused candidates get sent through the (comparatively
# expensive) cross-encoder reranker. Large enough that a good vector/BM25
# match rarely falls outside the pool, small enough to keep rerank latency
# bounded regardless of how large bm25_top_k/vector_top_k are configured.
_RERANK_POOL_MULTIPLIER = 3

try:
    _sentence_transformers: Any = importlib.import_module("sentence_transformers")
except ImportError:  # pragma: no cover - exercised by whichever env lacks it
    _sentence_transformers = None


def reciprocal_rank_fusion(rankings: list[list[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Fuse multiple ranked id lists into one ranking via Reciprocal Rank Fusion.

    Standard RRF: for every ranked list, each id at 1-indexed rank ``r``
    contributes ``1 / (k + r)`` to its running fused score; an id's total is
    the sum of its contributions across every ranking it appears in (ids
    absent from a given ranking simply contribute nothing from it). Pure
    function with no side effects -- it never touches an index or a chunk,
    so it is directly unit-testable against hand-computed numbers.

    Args:
        rankings: One or more ranked lists of ids, best match first. An id
            may repeat across different rankings; within a single ranking
            it should appear at most once.
        k: The RRF constant controlling how quickly rank position decays
            (larger ``k`` flattens the difference between adjacent ranks).

    Returns:
        ``(id, fused_score)`` pairs sorted by descending fused score. Ties
        keep the order in which ids were first encountered, since Python's
        sort is stable and ids are scored in first-seen order.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class Reranker(Protocol):
    """Anything that can score ``(query, text)`` pairs for relevance.

    Structural (``Protocol``) rather than nominal so a hermetic test can
    inject a fake scorer for :class:`Retriever` without subclassing
    :class:`CrossEncoderReranker` -- the same pattern ``app/rag/index.py``
    uses for ``Embedder``/``VectorIndex``.
    """

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text, in the same order as ``texts``.

        Args:
            query: The query each text is scored against.
            texts: Candidate texts to score.

        Returns:
            One float per element of ``texts``, in order. Higher means more
            relevant. The numeric scale is implementation-specific (e.g.
            raw cross-encoder logits), so it must only ever be compared
            against :attr:`~app.settings.RagSettings.min_rerank_score`,
            never against an RRF fused score.
        """
        ...


# Pinned revision; re-verify deliberately before bumping (prod hardening: platform-engineer lane).
#
# This hash was written without network access to check it against the real
# huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2 commit history, so it
# MUST be re-verified against that repo before this code is ever pointed at a
# live download. Pinning by revision here is a stopgap, not full hardening:
# enforcing HF_HUB_OFFLINE, adding a load timeout, and documenting the
# exception in SECURITY.md are deferred to the platform-engineer session's
# Dockerfile / app/agent wiring work.
_MS_MARCO_MINILM_L6_V2_REVISION = "ce0834f22110de6d9222af7a7a03628121708969"


class CrossEncoderReranker:
    """:class:`Reranker` backed by a local ``sentence-transformers`` cross-encoder.

    Requires the ``ml`` extra (``uv sync --all-extras``). Construction never
    raises and never loads the model: the availability check and the
    (lazy, cached) model load both happen inside :meth:`score`, mirroring
    ``SentenceTransformerEmbedder`` in ``app/rag/index.py``.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Configure which cross-encoder model to lazily load.

        Args:
            model_name: Name or path of the sentence-transformers
                cross-encoder model.
        """
        self.model_name = model_name
        self._model: Any = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score each of ``texts`` against ``query`` with the cross-encoder.

        Args:
            query: The query each text is scored against.
            texts: Candidate texts to score.

        Returns:
            One relevance score per element of ``texts``, in order.

        Raises:
            RuntimeError: If ``sentence-transformers`` is not installed.
        """
        if _sentence_transformers is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Run `uv sync --all-extras` "
                "to enable cross-encoder reranking."
            )
        if self._model is None:
            self._model = _sentence_transformers.CrossEncoder(
                self.model_name, revision=_MS_MARCO_MINILM_L6_V2_REVISION
            )
        pairs = [[query, text] for text in texts]
        scores = self._model.predict(pairs)
        return [float(score) for score in scores]


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk that survived retrieval, fusion, and (if available) reranking.

    Attributes:
        chunk: The retrieved chunk.
        score: Its relevance score, on whichever scale ``scored_by``
            indicates -- never comparable across the two scales.
        scored_by: ``"rerank"`` if a cross-encoder produced ``score``,
            ``"rrf"`` if it is the raw Reciprocal Rank Fusion score because
            no reranker was configured.
    """

    chunk: Chunk
    score: float
    scored_by: Literal["rerank", "rrf"]


class Retriever:
    """Hybrid retrieval: BM25 + vector fusion, optional reranking, thresholding.

    Prepares the chunk pool that grounds narration. Returning an empty list
    from :meth:`retrieve` is a normal, valid outcome (weak or absent
    retrieval) -- never an error -- so downstream narration can correctly
    refuse with "no grounded commentary available" (CLAUDE.md rule 4).
    """

    def __init__(
        self,
        index: HybridIndex,
        reranker: Reranker | None,
        settings: RagSettings,
    ) -> None:
        """Wire up a retriever over an already-built index.

        Args:
            index: The hybrid (BM25 + optional vector) index to query.
            reranker: A cross-encoder reranker, or ``None`` to fall back to
                raw RRF-fused scores.
            settings: Tunable retrieval parameters (top-k sizes, RRF ``k``,
                relevance thresholds, final pool size).
        """
        self._index = index
        self._reranker = reranker
        self._settings = settings

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Retrieve, fuse, (re)rank, and threshold chunks relevant to ``query``.

        Runs a BM25 top-k query and, if the index has a vector side, a
        vector top-k query; fuses both id rankings into one via
        :func:`reciprocal_rank_fusion`; takes a pre-rerank pool of the top
        fused ids (bounded by ``final_top_k * _RERANK_POOL_MULTIPLIER``, so
        reranking cost stays predictable regardless of how large
        ``bm25_top_k``/``vector_top_k`` are); looks up the corresponding
        :class:`~app.schemas.Chunk` objects; reranks the pool with the
        configured cross-encoder when one is present (``scored_by="rerank"``,
        thresholded against ``settings.min_rerank_score``), otherwise keeps
        the raw fused scores (``scored_by="rrf"``, thresholded against
        ``settings.min_fused_score`` -- a different numeric scale that must
        never share a threshold constant with rerank scores); and returns
        at most ``settings.final_top_k`` survivors, best first.

        Args:
            query: The natural-language question to retrieve chunks for.

        Returns:
            Up to ``settings.final_top_k`` :class:`RetrievedChunk` results,
            best first. Empty if nothing cleared the relevance threshold --
            a normal outcome, not an error.
        """
        bm25_hits = self._index.bm25_query(query, self._settings.bm25_top_k)
        rankings = [[chunk_id for chunk_id, _ in bm25_hits]]

        if self._index.has_vector_index:
            vector_hits = self._index.vector_query(query, self._settings.vector_top_k)
            rankings.append([chunk_id for chunk_id, _ in vector_hits])

        fused = reciprocal_rank_fusion(rankings, k=self._settings.rrf_k)
        if not fused:
            return []

        pool_size = min(len(fused), self._settings.final_top_k * _RERANK_POOL_MULTIPLIER)
        pool = fused[:pool_size]

        pooled: list[tuple[Chunk, float]] = []
        for chunk_id, fused_score in pool:
            chunk = self._index.get_chunk(chunk_id)
            if chunk is not None:
                pooled.append((chunk, fused_score))
        if not pooled:
            return []

        scored: list[RetrievedChunk]
        threshold: float
        if self._reranker is not None:
            texts = [chunk.text for chunk, _ in pooled]
            rerank_scores = self._reranker.score(query, texts)
            scored = [
                RetrievedChunk(chunk=chunk, score=rerank_score, scored_by="rerank")
                for (chunk, _fused_score), rerank_score in zip(pooled, rerank_scores, strict=True)
            ]
            threshold = self._settings.min_rerank_score
        else:
            scored = [
                RetrievedChunk(chunk=chunk, score=fused_score, scored_by="rrf")
                for chunk, fused_score in pooled
            ]
            threshold = self._settings.min_fused_score

        scored.sort(key=lambda retrieved: retrieved.score, reverse=True)
        survivors = [retrieved for retrieved in scored if retrieved.score >= threshold]
        return survivors[: self._settings.final_top_k]
