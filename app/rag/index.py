"""Hybrid (keyword + vector) retrieval index over a filing's chunks.

``HybridIndex`` is the object a later ``app/rag/retrieve.py`` module builds
once per filing and queries twice per question: once through
:meth:`HybridIndex.bm25_query` (keyword match) and once through
:meth:`HybridIndex.vector_query` (semantic match), then fuses the two
ranked lists (e.g. via Reciprocal Rank Fusion) before reranking and handing
the survivors to narration. This module only produces the two raw ranked
candidate lists plus a ``chunk_id -> Chunk`` lookup; fusion, reranking, and
the relevance threshold used to decide "no grounded commentary available"
(CLAUDE.md rule 4) live downstream.

Degraded-mode support (CLAUDE.md rule "guard chromadb / sentence_transformers
imports"): ``chromadb`` and ``sentence_transformers`` are optional (the
``ml`` extra). Both are imported dynamically via :func:`importlib.import_module`
into a module-level variable typed ``Any``, inside a ``try/except
ImportError`` that leaves the variable ``None`` on failure. This keeps
mypy happy either way -- there is no ``# type: ignore`` on a plain
``import chromadb`` that would become a hard error (via
``warn_unused_ignores``) in whichever of the two environments (extra
installed / not installed) doesn't need it -- and it means the module
always imports cleanly. Only *using* the unavailable pieces
(:class:`ChromaVectorIndex`, :class:`SentenceTransformerEmbedder`) raises,
and only when actually invoked, never at import time. :func:`is_ml_available`
lets callers check ahead of time without raising.

``rank_bm25`` (the keyword side) is a core, always-installed dependency --
see ``pyproject.toml`` -- so :class:`BM25Index` imports it directly.
"""

from __future__ import annotations

import importlib
import math
import re
from typing import Any, Protocol

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from app.schemas import Chunk

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split ``text`` into simple word tokens for BM25.

    Args:
        text: Text to tokenize.

    Returns:
        A list of lowercase alphanumeric tokens, in order.
    """
    return _WORD_RE.findall(text.lower())


class Embedder(Protocol):
    """Anything that can turn a batch of texts into embedding vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order.

        Args:
            texts: Texts to embed.

        Returns:
            One vector (list of floats) per element of ``texts``.
        """
        ...


class VectorIndex(Protocol):
    """A similarity index over stored chunk embeddings."""

    def add(self, ids: list[str], embeddings: list[list[float]]) -> None:
        """Add vectors to the index under the given ids.

        Args:
            ids: Chunk ids, parallel to ``embeddings``.
            embeddings: Embedding vectors, parallel to ``ids``.
        """
        ...

    def query(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """Return the ``top_k`` most similar stored vectors to ``embedding``.

        Args:
            embedding: The query embedding vector.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(chunk_id, similarity)`` pairs, most-similar-first.
            Higher similarity is always better, regardless of the
            implementation's internal distance metric.
        """
        ...


class BM25Index:
    """Keyword (Okapi BM25) index over a fixed set of chunks.

    Prefer the :meth:`build` classmethod over calling the constructor
    directly.
    """

    def __init__(self, chunk_ids: list[str], tokenized_corpus: list[list[str]]) -> None:
        """Store a pre-tokenized corpus and build the underlying BM25 model.

        Args:
            chunk_ids: Chunk id for each document in ``tokenized_corpus``,
                in the same order.
            tokenized_corpus: One tokenized document (list of tokens) per
                chunk, in the same order as ``chunk_ids``.
        """
        self._chunk_ids = chunk_ids
        self._bm25: Any = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    @classmethod
    def build(cls, chunks: list[Chunk]) -> BM25Index:
        """Build a :class:`BM25Index` from a list of chunks.

        Args:
            chunks: Chunks to index. Their ``text`` is lowercased and word-
                tokenized (see :func:`_tokenize`).

        Returns:
            A ready-to-query ``BM25Index``.
        """
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        tokenized_corpus = [_tokenize(chunk.text) for chunk in chunks]
        return cls(chunk_ids, tokenized_corpus)

    def query(self, text: str, top_k: int) -> list[tuple[str, float]]:
        """Return the ``top_k`` chunks whose text best matches ``text``.

        A chunk with zero keyword overlap with ``text`` (BM25 score exactly
        ``0.0``) is never returned, even if fewer than ``top_k`` chunks
        score above zero (including zero results for a query that shares no
        vocabulary with any indexed chunk). Without this floor, an
        unrelated query would still fill up to ``top_k`` "hits" with
        meaningless zero-score chunks whenever the corpus has that many
        chunks -- silently defeating the relevance threshold downstream
        retrieval relies on to fail closed (CLAUDE.md rule 4), since RRF
        fusion turns any *returned* id into a strictly positive fused score
        regardless of how weak (here, nonexistent) the underlying match was.

        Args:
            text: Query text; tokenized the same way as the indexed corpus.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(chunk_id, bm25_score)`` pairs, highest score
            first, restricted to strictly positive scores. Empty if the
            index was built from zero chunks or if no chunk has any
            keyword overlap with ``text``.
        """
        if self._bm25 is None or top_k <= 0:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(
            zip(self._chunk_ids, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [(chunk_id, float(score)) for chunk_id, score in ranked[:top_k] if score > 0]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in ``[-1, 1]``, or ``0.0`` if either vector has
        zero magnitude (defined rather than dividing by zero).
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorIndex:
    """Pure-Python cosine-similarity vector index.

    This is the graceful-degradation path used whenever the ``ml`` extra
    (``chromadb``) isn't installed or no ``persist_dir`` was requested. It
    has no third-party dependency at all -- just stored lists and
    :func:`_cosine_similarity` -- so it always works, at the cost of O(n)
    query time and no on-disk persistence.
    """

    def __init__(self) -> None:
        """Create an empty in-memory vector index."""
        self._ids: list[str] = []
        self._embeddings: list[list[float]] = []

    def add(self, ids: list[str], embeddings: list[list[float]]) -> None:
        """Append vectors to the index.

        Args:
            ids: Chunk ids, parallel to ``embeddings``.
            embeddings: Embedding vectors, parallel to ``ids``.
        """
        self._ids.extend(ids)
        self._embeddings.extend(embeddings)

    def query(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """Return the ``top_k`` stored vectors most similar to ``embedding``.

        Args:
            embedding: Query embedding vector.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(chunk_id, cosine_similarity)`` pairs, most-similar
            first.
        """
        if top_k <= 0:
            return []
        scored = [
            (chunk_id, _cosine_similarity(embedding, stored))
            for chunk_id, stored in zip(self._ids, self._embeddings, strict=True)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


# --- Optional ML dependencies: dynamic import into `Any`, never a bare
# `import chromadb` / `import sentence_transformers` (see module docstring).

try:
    _chromadb: Any = importlib.import_module("chromadb")
except ImportError:  # pragma: no cover - exercised by whichever env lacks it
    _chromadb = None

try:
    _sentence_transformers: Any = importlib.import_module("sentence_transformers")
except ImportError:  # pragma: no cover - exercised by whichever env lacks it
    _sentence_transformers = None


def is_chromadb_available() -> bool:
    """Return ``True`` if the optional ``chromadb`` dependency is importable."""
    return _chromadb is not None


def is_sentence_transformers_available() -> bool:
    """Return ``True`` if the optional ``sentence_transformers`` dependency is importable."""
    return _sentence_transformers is not None


def is_ml_available() -> bool:
    """Return ``True`` if the full ``ml`` extra (chromadb + sentence-transformers) is installed.

    Never raises. Callers use this to decide up front whether to request a
    persistent vector index / local embedder, instead of finding out via an
    exception. Both packages ship together under the ``ml`` extra (see
    ``pyproject.toml``), so this checks both.

    Returns:
        ``True`` only if both optional dependencies are importable.
    """
    return is_chromadb_available() and is_sentence_transformers_available()


class ChromaVectorIndex:
    """:class:`VectorIndex` backed by a persistent Chroma collection.

    Requires the ``ml`` extra (``uv sync --all-extras``). Construction
    raises :class:`RuntimeError` immediately if ``chromadb`` isn't
    importable, rather than failing confusingly on first use.
    """

    def __init__(self, *, persist_dir: str, collection_name: str = "tieout-chunks") -> None:
        """Open (creating if needed) a persistent Chroma collection.

        Args:
            persist_dir: Directory Chroma should persist its data under.
            collection_name: Name of the collection to use/create.

        Raises:
            RuntimeError: If ``chromadb`` is not installed.
        """
        if _chromadb is None:
            raise RuntimeError(
                "chromadb is not installed. Run `uv sync --all-extras` to enable "
                "the persistent vector index."
            )
        self._client = _chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add(self, ids: list[str], embeddings: list[list[float]]) -> None:
        """Add vectors to the Chroma collection.

        Args:
            ids: Chunk ids, parallel to ``embeddings``.
            embeddings: Embedding vectors, parallel to ``ids``.
        """
        if not ids:
            return
        self._collection.add(ids=ids, embeddings=embeddings)

    def query(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """Return the ``top_k`` stored vectors most similar to ``embedding``.

        Args:
            embedding: Query embedding vector.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(chunk_id, similarity)`` pairs, most-similar first.
            Chroma's default distance is squared L2 (smaller = closer); it
            is converted to ``1 / (1 + distance)`` so every
            :class:`VectorIndex` implementation agrees that higher is
            better.
        """
        if top_k <= 0:
            return []
        result = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        ids = result.get("ids") or [[]]
        distances = result.get("distances") or [[]]
        return [
            (chunk_id, 1.0 / (1.0 + dist))
            for chunk_id, dist in zip(ids[0], distances[0], strict=True)
        ]


# Pinned revision; re-verify deliberately before bumping (prod hardening: platform-engineer lane).
#
# This hash was written without network access to check it against the real
# huggingface.co/sentence-transformers/all-MiniLM-L6-v2 commit history, so it
# MUST be re-verified against that repo before this code is ever pointed at a
# live download. Pinning by revision here is a stopgap, not full hardening:
# enforcing HF_HUB_OFFLINE, adding a load timeout, and documenting the
# exception in SECURITY.md are deferred to the platform-engineer session's
# Dockerfile / app/agent wiring work.
_ALL_MINILM_L6_V2_REVISION = "44eb4044493a3c34bc6d7faae1a71ec76665ebc6"


class SentenceTransformerEmbedder:
    """:class:`Embedder` backed by a local ``sentence-transformers`` model.

    Requires the ``ml`` extra (``uv sync --all-extras``). Unlike
    :class:`ChromaVectorIndex`, construction never raises and never
    imports the model: the check and the (lazy, cached) model load both
    happen inside :meth:`embed`, so simply instantiating this class in a
    degraded environment (e.g. while wiring up default config) is safe.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Configure which sentence-transformers model to lazily load.

        Args:
            model_name: Name or path of the sentence-transformers model.
        """
        self.model_name = model_name
        self._model: Any = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` with the configured sentence-transformers model.

        Args:
            texts: Texts to embed.

        Returns:
            One embedding vector per input text, in order.

        Raises:
            RuntimeError: If ``sentence-transformers`` is not installed.
        """
        if _sentence_transformers is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Run `uv sync --all-extras` "
                "to enable local embeddings."
            )
        if self._model is None:
            self._model = _sentence_transformers.SentenceTransformer(
                self.model_name, revision=_ALL_MINILM_L6_V2_REVISION
            )
        vectors = self._model.encode(list(texts))
        return [[float(value) for value in vector] for vector in vectors]


class HybridIndex:
    """Hybrid keyword (BM25) + vector retrieval index over one filing's chunks.

    Always has a working BM25 side. The vector side is optional and only
    present if a :class:`build` was given an ``embedder``; when absent,
    :meth:`vector_query` fails closed by returning an empty list rather
    than raising, so downstream retrieval code can treat "no vector index"
    the same as "vector search found nothing" (CLAUDE.md rule 4).

    Prefer :meth:`build` over calling the constructor directly.
    """

    def __init__(
        self,
        *,
        bm25_index: BM25Index,
        vector_index: VectorIndex | None,
        chunks_by_id: dict[str, Chunk],
        embedder: Embedder | None,
    ) -> None:
        """Store pre-built index components.

        Args:
            bm25_index: The keyword side of the hybrid index.
            vector_index: The vector side, or ``None`` if it was not built.
            chunks_by_id: Every indexed chunk, keyed by ``chunk_id``, for
                :meth:`get_chunk` lookups.
            embedder: The embedder used to build ``vector_index`` (needed
                again in :meth:`vector_query` to embed the query text), or
                ``None`` if there is no vector side.
        """
        self._bm25_index = bm25_index
        self._vector_index = vector_index
        self._chunks_by_id = chunks_by_id
        self._embedder = embedder

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        *,
        embedder: Embedder | None = None,
        persist_dir: str | None = None,
    ) -> HybridIndex:
        """Build a hybrid index over ``chunks``.

        Always builds the BM25 keyword side. If ``embedder`` is given, also
        embeds every chunk's text and builds a vector side: a
        :class:`ChromaVectorIndex` persisted at ``persist_dir`` when
        ``chromadb`` is importable *and* ``persist_dir`` was given,
        otherwise an :class:`InMemoryVectorIndex` (the graceful-degradation
        path -- also what the hermetic test suite always exercises, by
        passing ``persist_dir=None``, so tests never touch real
        ``chromadb`` even if it happens to be installed). If ``embedder``
        is ``None``, the vector side is simply absent.

        Args:
            chunks: Chunks to index.
            embedder: Optional embedder used to build the vector side. If
                omitted, this index has no vector side at all.
            persist_dir: Optional directory for a persistent Chroma
                collection. Ignored if ``embedder`` is ``None``; also
                ignored (falling back to :class:`InMemoryVectorIndex`) if
                ``chromadb`` is not importable.

        Returns:
            A ready-to-query :class:`HybridIndex`.
        """
        bm25_index = BM25Index.build(chunks)
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

        vector_index: VectorIndex | None = None
        if embedder is not None:
            if persist_dir is not None and is_chromadb_available():
                vector_index = ChromaVectorIndex(persist_dir=persist_dir)
            else:
                vector_index = InMemoryVectorIndex()
            if chunks:
                ids = [chunk.chunk_id for chunk in chunks]
                embeddings = embedder.embed([chunk.text for chunk in chunks])
                vector_index.add(ids, embeddings)

        return cls(
            bm25_index=bm25_index,
            vector_index=vector_index,
            chunks_by_id=chunks_by_id,
            embedder=embedder,
        )

    @property
    def has_vector_index(self) -> bool:
        """``True`` if this index has a usable vector (embedding) side."""
        return self._vector_index is not None

    def bm25_query(self, text: str, top_k: int) -> list[tuple[str, float]]:
        """Run a keyword (BM25) query against the indexed chunks.

        Args:
            text: Query text.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(chunk_id, bm25_score)`` pairs, highest score
            first.
        """
        return self._bm25_index.query(text, top_k)

    def vector_query(self, text: str, top_k: int) -> list[tuple[str, float]]:
        """Run a vector similarity query against the indexed chunks, if possible.

        Embeds ``text`` with the same embedder used in :meth:`build`, then
        queries the vector index. Fails closed rather than raising when
        there is no vector side.

        Args:
            text: Query text.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(chunk_id, similarity)`` pairs, most-similar first,
            or an empty list if this index has no vector side (built with
            ``embedder=None``).
        """
        if self._vector_index is None or self._embedder is None:
            return []
        [query_embedding] = self._embedder.embed([text])
        return self._vector_index.query(query_embedding, top_k)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Look up a previously-indexed chunk by id.

        Args:
            chunk_id: The chunk id to look up.

        Returns:
            The matching :class:`Chunk`, or ``None`` if no chunk with that
            id was indexed.
        """
        return self._chunks_by_id.get(chunk_id)
