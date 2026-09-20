"""Configuration for the RAG pipeline: chunking, retrieval, and narration knobs.

``RagSettings`` holds plain defaults, not environment-loaded settings —
callers construct it directly and may override any field (for tests,
experiments, or future env-driven wiring in ``app/agent``/``app/api``).
"""

from __future__ import annotations

from pydantic import BaseModel


class RagSettings(BaseModel):
    """Tunable parameters for filing ingestion, hybrid retrieval, and narration.

    Attributes:
        chunk_target_tokens: Approximate target size of one chunk, in tokens.
            Tokens are approximated as ``len(text) // 4`` (no tokenizer
            dependency), so ingestion tests stay hermetic (CLAUDE.md rule 7).
        chunk_overlap_tokens: Approximate overlap between consecutive chunks
            cut from the same section, in the same token approximation.
        bm25_top_k: Number of candidates the BM25 (keyword) retriever returns.
        vector_top_k: Number of candidates the vector (embedding) retriever
            returns.
        rrf_k: The ``k`` constant used in Reciprocal Rank Fusion when
            combining BM25 and vector result rankings.
        final_top_k: Number of chunks kept after reranking and handed to
            narration.
        min_rerank_score: Minimum cross-encoder rerank score a chunk must
            reach to survive; chunks scoring below this are dropped
            (fail closed per CLAUDE.md rule 4).
        min_fused_score: Minimum RRF-fused score a chunk must reach to be
            considered before reranking.
        narration_model: Anthropic model id used to generate grounded
            commentary.
        embedding_model: Identifier of the embedding model backing the
            vector index. Placeholder field — the concrete model is wired
            in when the vector store is built.
        cross_encoder_model: Identifier of the cross-encoder reranker
            model. Placeholder field — the concrete model is wired in when
            the reranker is built.
        max_citation_quote_chars: Maximum length, in characters, of a
            ``Citation.quote`` extracted from a chunk.
    """

    chunk_target_tokens: int = 800
    chunk_overlap_tokens: int = 100
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rrf_k: int = 60
    final_top_k: int = 6
    min_rerank_score: float = 0.0
    min_fused_score: float = 0.0
    narration_model: str = "claude-sonnet-5"
    embedding_model: str = "placeholder-embedding-model"
    cross_encoder_model: str = "placeholder-cross-encoder-model"
    max_citation_quote_chars: int = 300
