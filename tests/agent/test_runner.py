"""Tests for app.agent.runner.run_pipeline: run-log recording on every outcome."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.agent.graph import FatalPipelineError
from app.agent.runner import _build_context, _generic_error, run_pipeline
from app.edgar.client import EdgarClient
from app.obs import RunLog
from app.settings import EdgarSettings, RagSettings
from tests.conftest import make_test_app_settings


async def test_successful_run_is_recorded_done_with_outcome(
    aapl_edgar_client: EdgarClient, tmp_path: Path
) -> None:
    settings = make_test_app_settings(tmp_path)
    run_log = RunLog(settings.run_log_path)

    events = [
        event
        async for event in run_pipeline(
            run_id="run-1",
            ticker="AAPL",
            edgar_client=aapl_edgar_client,
            settings=settings,
            rag_settings=RagSettings(),
            run_log=run_log,
        )
    ]

    assert events[-1].step == "done"
    assert events[-1].status == "ok"

    record = run_log.get("run-1")
    assert record is not None
    assert record.status == "done"
    assert record.tieout_passed is True
    assert record.accession_number == "0000320193-25-000079"
    assert record.artifact_path is not None
    assert Path(record.artifact_path).exists()


async def test_fatal_failure_is_recorded_as_error_with_generic_message(
    unresolvable_edgar_client: EdgarClient, tmp_path: Path
) -> None:
    settings = make_test_app_settings(tmp_path)
    run_log = RunLog(settings.run_log_path)

    events = [
        event
        async for event in run_pipeline(
            run_id="run-2",
            ticker="NOPE",
            edgar_client=unresolvable_edgar_client,
            settings=settings,
            rag_settings=RagSettings(),
            run_log=run_log,
        )
    ]

    assert events[-1].step == "error"
    assert events[-1].status == "failed"
    assert "NOPE" not in events[-1].detail  # SECURITY.md item 10: no internals leaked

    record = run_log.get("run-2")
    assert record is not None
    assert record.status == "error"
    assert record.error == events[-1].detail


async def test_run_pipeline_never_raises_out_of_the_generator(
    unresolvable_edgar_client: EdgarClient, tmp_path: Path
) -> None:
    """Even a fatal pipeline error must surface as a terminal event, never an exception."""
    settings = make_test_app_settings(tmp_path)
    run_log = RunLog(settings.run_log_path)

    # No pytest.raises: reaching the end of iteration without an exception
    # escaping is exactly what's being asserted.
    events = [
        event
        async for event in run_pipeline(
            run_id="run-3",
            ticker="NOPE",
            edgar_client=unresolvable_edgar_client,
            settings=settings,
            rag_settings=RagSettings(),
            run_log=run_log,
        )
    ]
    assert events[-1].step == "error"


def test_build_context_defaults_to_bm25_only_even_with_ml_extra_installed(
    aapl_edgar_client: EdgarClient, tmp_path: Path
) -> None:
    """AppSettings.enable_local_embeddings defaults False: never an implicit network dependency.

    Verified directly (not asserted here, since it would require a live
    network call to prove a negative): constructing the real
    `SentenceTransformerEmbedder`/`CrossEncoderReranker` triggers an
    unbounded Hugging Face Hub download on first use. This test only
    checks the wiring decision -- that the context ends up with no
    embedder/reranker by default -- not the download behavior itself.
    """
    settings = make_test_app_settings(tmp_path)
    assert settings.enable_local_embeddings is False

    context = _build_context(
        "run-4", edgar_client=aapl_edgar_client, settings=settings, rag_settings=RagSettings()
    )
    assert context.embedder is None
    assert context.reranker is None


def test_build_context_skips_narrator_without_an_api_key(
    aapl_edgar_client: EdgarClient, tmp_path: Path
) -> None:
    settings = make_test_app_settings(tmp_path)
    assert settings.anthropic_configured is False

    context = _build_context(
        "run-5", edgar_client=aapl_edgar_client, settings=settings, rag_settings=RagSettings()
    )
    assert context.narrator_client is None


async def test_cancelling_a_run_mid_flight_records_it_as_error_not_stuck_running(
    tmp_path: Path,
) -> None:
    """Regression test: a cancelled run must not stay at status='running' forever.

    `asyncio.CancelledError` subclasses `BaseException`, so the pipeline's
    broad `except Exception` never used to catch it -- correct, since
    swallowing a cancellation would break the caller's ability to cancel
    the task at all, but it also meant nothing ever recorded the outcome.
    A slow, real-thread-blocking mock transport (`time.sleep` inside the
    handler, executed via LangGraph's executor dispatch for a sync node)
    gives a reliable real-world window to cancel the task mid-`fetch`.
    """

    def slow_handler(request: httpx.Request) -> httpx.Response:
        time.sleep(0.5)
        return httpx.Response(404)

    settings = make_test_app_settings(tmp_path)
    run_log = RunLog(settings.run_log_path)
    edgar_settings = EdgarSettings(
        sec_user_agent="tieout-tests/0.1 (contact: tests@example.com)",
        cache_dir=tmp_path / "cache",
    )
    client = EdgarClient(settings=edgar_settings, transport=httpx.MockTransport(slow_handler))

    async def drain() -> list[Any]:
        return [
            event
            async for event in run_pipeline(
                run_id="run-cancel",
                ticker="AAPL",
                edgar_client=client,
                settings=settings,
                rag_settings=RagSettings(),
                run_log=run_log,
            )
        ]

    task = asyncio.create_task(drain())
    await asyncio.sleep(0.05)  # let it start (create_run) and enter the slow fetch call
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    client.close()

    record = run_log.get("run-cancel")
    assert record is not None
    assert record.status == "error"
    assert record.error == "run cancelled"


def test_generic_error_maps_timeout_error_to_a_generic_message() -> None:
    """`_generic_error`'s TimeoutError branch, exercised directly as a pure function."""
    assert _generic_error(TimeoutError()) == "run timed out"


def test_generic_error_passes_through_a_fatal_pipeline_errors_own_message() -> None:
    assert _generic_error(FatalPipelineError("could not fetch filing data")) == (
        "could not fetch filing data"
    )


def test_generic_error_reduces_an_unexpected_exception_to_a_generic_message() -> None:
    """SECURITY.md item 10: any other exception must never leak its own text."""
    assert (
        _generic_error(ValueError("some internal detail: /etc/secret"))
        == "unexpected error while generating the model"
    )
