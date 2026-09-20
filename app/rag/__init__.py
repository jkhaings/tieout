"""Public API of the RAG pipeline: ingest -> index -> retrieve -> narrate.

This package turns a filing's raw HTML into grounded, citation-checked
commentary for one line item at a time:

1. :func:`~app.rag.ingest.parse_filing` splits a 10-K's HTML into
   retrieval-ready :class:`~app.schemas.Chunk` objects.
2. :class:`~app.rag.index.HybridIndex` (built with :meth:`HybridIndex.build`)
   indexes those chunks for both keyword (BM25) and, optionally, vector
   search -- see the :class:`~app.rag.index.Embedder` and
   :class:`~app.rag.index.VectorIndex` protocols it is built from.
3. :class:`~app.rag.retrieve.Retriever` fuses (:func:`
   ~app.rag.retrieve.reciprocal_rank_fusion`), optionally reranks, and
   thresholds candidates from a :class:`~app.rag.index.HybridIndex` into a
   list of :class:`~app.rag.retrieve.RetrievedChunk`. An empty result is a
   normal "no grounded commentary available" outcome (CLAUDE.md rule 4),
   never an error.
4. :func:`~app.rag.narrate.narrate_line_item` turns pre-formatted figures
   plus retrieved chunks into a validated
   :class:`~app.schemas.Commentary`, calling any :class:`
   ~app.rag.narrate.LLMClient` (typically :class:`
   ~app.rag.narrate.AnthropicNarrator`).

:class:`~app.rag.config.RagSettings` holds the tunable knobs threaded
through every stage above.

This module re-exports the pieces other packages (notably ``app/agent``)
are expected to need, so they have one place to import from instead of
reaching into individual ``app.rag.*`` submodules.
"""

from __future__ import annotations

from app.rag.config import RagSettings
from app.rag.index import (
    BM25Index,
    ChromaVectorIndex,
    Embedder,
    HybridIndex,
    InMemoryVectorIndex,
    SentenceTransformerEmbedder,
    VectorIndex,
    is_chromadb_available,
    is_ml_available,
    is_sentence_transformers_available,
)
from app.rag.ingest import parse_filing
from app.rag.narrate import (
    DRAFT_JSON_SCHEMA,
    AnthropicNarrator,
    LLMClient,
    narrate_line_item,
)
from app.rag.retrieve import (
    CrossEncoderReranker,
    Reranker,
    RetrievedChunk,
    Retriever,
    reciprocal_rank_fusion,
)

__all__ = [
    "DRAFT_JSON_SCHEMA",
    "AnthropicNarrator",
    "BM25Index",
    "ChromaVectorIndex",
    "CrossEncoderReranker",
    "Embedder",
    "HybridIndex",
    "InMemoryVectorIndex",
    "LLMClient",
    "RagSettings",
    "Reranker",
    "RetrievedChunk",
    "Retriever",
    "SentenceTransformerEmbedder",
    "VectorIndex",
    "is_chromadb_available",
    "is_ml_available",
    "is_sentence_transformers_available",
    "narrate_line_item",
    "parse_filing",
    "reciprocal_rank_fusion",
]
