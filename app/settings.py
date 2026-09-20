"""Consolidated configuration for tieout: EDGAR, RAG, and platform settings.

Three settings classes share this module because they share one `.env` file
and, from here on, one place to read them from:

* `EdgarSettings` (`BaseSettings`, env-loaded) — the data-engineer's lane.
  Moved verbatim from the now-deleted `app/edgar/config.py`.
* `RagSettings` (plain `BaseModel`, NOT env-loaded) — the rag-engineer's
  lane. Moved verbatim from the now-deleted `app/rag/config.py`. Callers
  construct it directly and may override any field; that call pattern
  (`RagSettings(final_top_k=4)`, etc.) is unchanged by the move.
* `AppSettings` (`BaseSettings`, env-loaded) — new, for `app/agent`/`app/api`/
  `app/obs`: the Anthropic key, optional Langfuse tracing keys, and the
  public-endpoint knobs (rate limit, daily cap, CORS, run artifact paths).

This hoist was anticipated by both original modules' docstrings once the
parallel data/rag sessions landed (`EdgarSettings` explicitly: "kept separate
... until a later session hoists everything into a shared app/settings.py").
It is scoped to this session per CLAUDE.md's ownership map.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Absolute and computed once: `env_file=".env"` alone is resolved relative to
# the process's current working directory, so `uvicorn` started from a
# different directory (or a container with a different WORKDIR) would
# silently load no `.env` at all -- surfacing later as a confusing
# `EdgarError` about a missing SEC_USER_AGENT rather than a config problem.
# Both settings classes below share this constant so they can't drift.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class EdgarSettings(BaseSettings):
    """EDGAR client configuration, loaded from the environment / `.env`.

    `sec_user_agent` defaults to empty rather than a placeholder value: an
    unconfigured User-Agent must fail loudly the moment a real request is
    attempted (see `app.edgar.client`), not silently send something SEC
    would reject or that misidentifies the requester.

    Note for tests that construct this directly with only some fields
    overridden (as several already do): every field you don't pass still
    reads the real process environment / `.env`, `pydantic-settings`'
    ordinary behavior. Pass every field explicitly if a test needs to be
    immune to whatever the developer happens to have set locally.
    """

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore", case_sensitive=False)

    sec_user_agent: str = Field(default="")
    cache_dir: Path = Field(default=Path("data/cache"))
    max_requests_per_second: float = Field(default=8.0, gt=0, le=10.0)
    request_timeout_s: float = Field(default=15.0, gt=0)


@lru_cache(maxsize=1)
def get_edgar_settings() -> EdgarSettings:
    """Return the process-wide `EdgarSettings`, constructed once and cached."""
    return EdgarSettings()


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


class AppSettings(BaseSettings):
    """Platform configuration for `app/agent`, `app/api`, and `app/obs`.

    `anthropic_api_key`, `langfuse_public_key`, and `langfuse_secret_key`
    are `SecretStr` (SECURITY.md item 6: never logged or echoed in errors —
    `SecretStr.__repr__` prints `**********`, not the value, and pydantic's
    own `ValidationError` messages, which otherwise embed the raw input
    verbatim, do the same for a `SecretStr` field). Langfuse tracing is a
    no-op whenever either Langfuse key is absent (see `app.obs.tracing`);
    `anthropic_api_key` is read here only so an explicit
    `anthropic.Anthropic(api_key=...)` client can be constructed for
    narration instead of relying on ambient environment resolution deep
    inside `app/rag` (which never loads `.env` itself).

    `enable_local_embeddings` defaults to `False` -- BM25-only retrieval --
    even when the `ml` extra (`sentence-transformers`/`chromadb`) is
    installed. `app.rag.index.SentenceTransformerEmbedder` and
    `app.rag.retrieve.CrossEncoderReranker` lazily download their model
    weights from Hugging Face Hub on first use with no timeout of their
    own (confirmed directly: an unconfigured environment hangs on the
    download rather than falling back). Their own code comments flag
    exactly this -- "enforcing HF_HUB_OFFLINE, adding a load timeout ...
    deferred to the platform-engineer session" -- so defaulting this off
    is that hardening: a public request-handling path must never have an
    implicit, unbounded dependency on a third-party host outside
    SECURITY.md's EDGAR-only allowlist. Set to `True` only in an
    environment where the model weights are already cached locally (e.g.
    baked into the container image at build time).

    `cors_origins` is `Annotated[..., NoDecode]`: `pydantic-settings`
    otherwise parses any `list[str]`-typed field from its environment value
    as JSON, so a plain `CORS_ORIGINS=https://a.com,https://b.com` would
    raise a `SettingsError` before validation ever runs. `NoDecode` skips
    that and hands the raw comma-separated string to the `field_validator`
    below instead. (An earlier draft of this field used a `str` field
    aliased to `CORS_ORIGINS` with a same-named `cors_origins` property —
    with no `populate_by_name`, that alias made a `cors_origins_raw=...`
    *keyword* argument passed to the constructor silently ignored, an
    extra field swallowed by `extra="ignore"` rather than a validation
    error. Declaring the field's real name as `cors_origins` from the start
    removes the alias, and with it that failure mode.)
    """

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore", case_sensitive=False)

    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    langfuse_public_key: SecretStr = Field(default=SecretStr(""))
    langfuse_secret_key: SecretStr = Field(default=SecretStr(""))
    langfuse_host: str = Field(default="https://cloud.langfuse.com")
    app_env: str = Field(default="dev")
    rate_limit_runs_per_hour: int = Field(default=6, gt=0)
    daily_run_cap: int = Field(default=50, gt=0)
    runs_dir: Path = Field(default=Path("data/runs"))
    run_log_path: Path = Field(default=Path("data/runs.db"))
    enable_local_embeddings: bool = Field(default=False)
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:8000"]
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        """Accept a comma-separated string (env/`.env`) or an already-built list."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("runs_dir", "run_log_path")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        """Resolve to an absolute path so a later CWD change can't orphan it."""
        return value.resolve()

    @property
    def langfuse_enabled(self) -> bool:
        """True only when both Langfuse keys are configured (see `app.obs.tracing`)."""
        return bool(self.langfuse_public_key.get_secret_value()) and bool(
            self.langfuse_secret_key.get_secret_value()
        )

    @property
    def anthropic_configured(self) -> bool:
        """True only when an Anthropic API key is set.

        Checked before narration starts so a missing key fails fast
        (an immediate, cheap decision to skip narration entirely) rather
        than after fetch/build/verify/retrieve have already run, ~20+
        seconds into a run, only to hit a 401 from the Anthropic API.
        """
        return bool(self.anthropic_api_key.get_secret_value())


@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    """Return the process-wide `AppSettings`, constructed once and cached."""
    return AppSettings()
