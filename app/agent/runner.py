"""Drive the compiled graph for one run: stream events, record the outcome.

The only public entrypoint is `run_pipeline`, an async generator that yields
`RunEvent`s live as the graph runs and, on its way out, writes the run's
final outcome to the run log. `app/api` consumes it directly to feed the SSE
endpoint's replay buffer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import cast

import httpx

from app.agent.graph import GRAPH, FatalPipelineError, PipelineContext, PipelineState
from app.edgar.client import EdgarClient, EdgarError, validate_ticker
from app.obs import RunLog, RunRecord
from app.rag import (
    AnthropicNarrator,
    CrossEncoderReranker,
    Embedder,
    LLMClient,
    Reranker,
    SentenceTransformerEmbedder,
    is_sentence_transformers_available,
)
from app.schemas import RunEvent
from app.settings import AppSettings, RagSettings

logger = logging.getLogger(__name__)

# Overall wall-clock safety net for one run. `EdgarClient._get`'s own retry
# (5 attempts, exponential backoff up to 30s each) and narrate's per-item
# circuit breaker already bound most slow paths individually; this is a
# last-resort ceiling so a genuinely hung run (e.g. a stalled TCP connection
# tenacity's backoff doesn't catch) can't occupy a run slot indefinitely.
# Generous relative to ARCHITECTURE.md's ~40s target run.
RUN_TIMEOUT_S = 300.0


def _build_narrator(settings: AppSettings) -> LLMClient:
    """Construct the narrator with an explicit key from `AppSettings`.

    `app/rag` never loads `.env` itself (by design -- it is a library, see
    its config docstring history); passing an explicit `anthropic.Anthropic`
    client here is what makes narration actually work under `app/api`
    instead of silently depending on the key also happening to be a real
    process environment variable.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
    return AnthropicNarrator(client=client)


def _build_context(
    run_id: str, *, edgar_client: EdgarClient, settings: AppSettings, rag_settings: RagSettings
) -> PipelineContext:
    """Assemble one run's `PipelineContext`, degrading gracefully per CLAUDE.md rule 4."""
    narrator_client: LLMClient | None = (
        _build_narrator(settings) if settings.anthropic_configured else None
    )
    embedder: Embedder | None = None
    reranker: Reranker | None = None
    # `AppSettings.enable_local_embeddings` gates this, not just availability:
    # both classes lazily download model weights from Hugging Face Hub on
    # first use with no timeout of their own (verified directly -- an
    # unconfigured environment hangs rather than degrading), so requesting
    # them must be an explicit opt-in, never automatic just because the `ml`
    # extra happens to be installed. BM25-only is a fully supported degrade
    # path (`app.rag.HybridIndex`/`Retriever` handle `embedder=None`/
    # `reranker=None` throughout), so this is a safe, fast default.
    if settings.enable_local_embeddings and is_sentence_transformers_available():
        embedder = SentenceTransformerEmbedder()
        reranker = CrossEncoderReranker()
    return PipelineContext(
        run_id=run_id,
        edgar_client=edgar_client,
        narrator_client=narrator_client,
        embedder=embedder,
        reranker=reranker,
        rag_settings=rag_settings,
        runs_dir=settings.runs_dir,
    )


def _generic_error(exc: BaseException) -> str:
    """Reduce any exception to the short, generic message clients may see (SECURITY.md item 10)."""
    if isinstance(exc, FatalPipelineError):
        return str(exc)
    if isinstance(exc, TimeoutError):
        return "run timed out"
    return "unexpected error while generating the model"


async def _find_cached_run(
    ticker: str, *, edgar_client: EdgarClient, run_log: RunLog
) -> RunRecord | None:
    """Best-effort check for a prior completed run of the same (ticker, filing).

    SECURITY.md item 7: "generated workbooks cached by (ticker, filing) so
    repeat requests cost nothing." Never raises -- any failure resolving
    the ticker/filing here (an invalid ticker, EDGAR unreachable) is
    treated as a cache miss, and the pipeline falls through to its own,
    already-tested error handling in the `fetch` node rather than
    duplicating it here. `resolve_cik`/`latest_10k` are both backed by
    `EdgarClient`'s own disk cache, so on a warm cache this costs no real
    network round trip -- `fetch`'s identical calls a moment later (on a
    miss) just hit the same cache.
    """
    try:
        normalized_ticker = validate_ticker(ticker)
        cik = await asyncio.to_thread(edgar_client.resolve_cik, normalized_ticker)
        latest = await asyncio.to_thread(edgar_client.latest_10k, cik)
    except (EdgarError, httpx.HTTPError):
        return None
    return await asyncio.to_thread(
        run_log.find_cached_run, normalized_ticker, latest["accession_number"]
    )


# Synthesized for a cache hit so the UI shows one "ok" event per pipeline
# step (a complete, honest progression -- each explicitly labeled as reused
# rather than freshly computed) instead of jumping straight from nothing to
# "done".
_PIPELINE_STEPS = ("fetch", "build", "verify", "retrieve", "narrate", "generate")


def _cached_hit_events(run_id: str) -> list[RunEvent]:
    """One synthetic 'ok' `RunEvent` per pipeline step, for a cache hit."""
    now = time.time()
    detail = "reusing a previously verified result for this filing"
    return [
        RunEvent(run_id=run_id, step=step, status="ok", detail=detail, ts=now)
        for step in _PIPELINE_STEPS
    ]


async def run_pipeline(
    *,
    run_id: str,
    ticker: str,
    edgar_client: EdgarClient,
    settings: AppSettings,
    rag_settings: RagSettings | None = None,
    run_log: RunLog,
) -> AsyncIterator[RunEvent]:
    """Run the pipeline for `ticker`, yielding `RunEvent`s live, recording the outcome.

    Records the run as started in `run_log` before the graph runs (a no-op
    if the caller -- `app.api.routes.create_run` -- already reserved the
    row synchronously before scheduling this; `RunLog.create_run` is
    idempotent specifically so both call sites are safe), then as either
    `"done"` (with the tie-out/narration outcome and artifact path) or
    `"error"` (with a short, generic message) once it finishes -- exactly
    once, regardless of whether the pipeline succeeded, degraded, or hit a
    fatal error. Never raises: every failure path is caught here, logged
    with full detail server-side, and surfaced to the caller only as a
    terminal `RunEvent(step="error", ...)` carrying a generic message.

    Args:
        run_id: Server-generated run id (SECURITY.md item 5).
        ticker: The (not-yet-validated) ticker string; `fetch` validates it.
        edgar_client: The single, process-wide `EdgarClient` (never one per
            run -- its rate limiter is per-instance).
        settings: Platform configuration (Anthropic key, runs directory).
        rag_settings: Retrieval/narration tunables; defaults to `RagSettings()`.
        run_log: The durable run log to record this run's outcome in.

    Yields:
        `RunEvent`s in real time as the graph executes, ending with exactly
        one terminal event whose `step` is `"done"` or `"error"`.
    """
    resolved_rag_settings = rag_settings or RagSettings()

    # Everything that can fail -- directory creation, the run-log write,
    # the cache lookup, context/client construction, and the graph itself
    # -- lives inside this one try block. An earlier version left
    # `mkdir`/`create_run`/`_build_context` running *before* any exception
    # handling existed at all: a failure there (e.g. a permissions error on
    # `runs_dir`, or `anthropic.Anthropic(api_key=...)` raising) escaped
    # this generator entirely uncaught, leaving the run log row stuck at
    # `status='running'` forever and its SSE stream hanging forever (no
    # event, including the very first `fetch/started`, was ever published).
    # Verified directly that this restructuring closes that gap: every
    # step from here through the graph's execution is now reachable only
    # via a path that ends in either a normal return or the `except`
    # below.
    final_state: PipelineState | None = None
    cache_hit: RunRecord | None = None
    try:
        settings.runs_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(run_log.create_run, run_id, ticker)
        cache_hit = await _find_cached_run(ticker, edgar_client=edgar_client, run_log=run_log)

        if cache_hit is None:
            context = _build_context(
                run_id,
                edgar_client=edgar_client,
                settings=settings,
                rag_settings=resolved_rag_settings,
            )
            initial_state: PipelineState = {"ticker": ticker}
            async with asyncio.timeout(RUN_TIMEOUT_S):
                # Multi-mode `astream` yields `(mode, chunk)`; `chunk`'s
                # static type is necessarily loose (it's `Any` for
                # "custom" -- whatever a node's `writer(...)` call
                # passed), so each branch narrows it to what this graph
                # actually emits: a `RunEvent` for "custom" (see `_emit`),
                # the accumulated `PipelineState` for "values".
                async for mode, chunk in GRAPH.astream(
                    initial_state, context=context, stream_mode=["custom", "values"]
                ):
                    if mode == "custom":
                        yield cast(RunEvent, chunk)
                    else:
                        final_state = cast(PipelineState, chunk)
    except asyncio.CancelledError:
        # `asyncio.CancelledError` subclasses `BaseException`, not
        # `Exception` (Python 3.8+), so the broad handler below never
        # catches it -- deliberately: swallowing a cancellation instead of
        # re-raising it would break the caller's ability to actually
        # cancel this task. But left uncaught entirely, a run cancelled
        # mid-flight (e.g. process shutdown) stayed at `status='running'`
        # in the run log forever, with nothing to ever mark it otherwise.
        # Record it as a real terminal outcome, then re-raise so
        # cancellation still propagates correctly.
        logger.warning("run %s cancelled", run_id, extra={"run_id": run_id})
        await asyncio.to_thread(run_log.mark_error, run_id, error="run cancelled")
        raise
    except Exception as exc:  # noqa: BLE001 - the fail-closed boundary: never leak internals
        message = _generic_error(exc)
        logger.warning("run %s failed: %s", run_id, exc, extra={"run_id": run_id})
        await asyncio.to_thread(run_log.mark_error, run_id, error=message)
        yield RunEvent(run_id=run_id, step="error", status="failed", detail=message, ts=time.time())
        return

    if cache_hit is not None:
        for event in _cached_hit_events(run_id):
            yield event
        await asyncio.to_thread(
            run_log.mark_done,
            run_id,
            tieout_passed=bool(cache_hit.tieout_passed),
            narrate_ok=bool(cache_hit.narrate_ok),
            accession_number=cache_hit.accession_number,
            artifact_path=cache_hit.artifact_path,
        )
    else:
        state = final_state or {}
        tieout = state.get("tieout")
        await asyncio.to_thread(
            run_log.mark_done,
            run_id,
            tieout_passed=bool(tieout.passed) if tieout is not None else False,
            narrate_ok=bool(state.get("narrate_ok")),
            accession_number=state.get("accession_number"),
            artifact_path=state.get("artifact_path"),
        )
    yield RunEvent(run_id=run_id, step="done", status="ok", detail="run complete", ts=time.time())
