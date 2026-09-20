"""HTTP route handlers: validate, delegate, serialize. No business logic here.

Plain, undecorated functions -- `app.main.create_app` registers each one
(and applies the per-IP rate limit to `create_run`) so that every app
instance gets its own `slowapi.Limiter`, built from that instance's own
`AppSettings.rate_limit_runs_per_hour`. This is a real constraint, not a
style choice: slowapi's `@limiter.limit(...)` decorator can only take a
*static* limit string, or a callable that receives at most the computed
rate-limit key (a parameter literally named `key`) -- never the `Request`
or `app.state` (verified directly: passing a `request`-shaped callable
raises `TypeError`, since `LimitGroup.__iter__` calls a zero-parameter
callable with zero arguments). Deciding the limit at route-registration
time, per app instance, is what makes it configurable at all.

`POST /runs` validates the ticker and the daily-cap abuse control, then
launches the pipeline as a background task and returns immediately; the
actual work happens in `app.agent.run_pipeline`. `GET /runs/{run_id}/events`
streams that task's progress via SSE from a `RunHandle` (see
`app.api.registry`) rather than owning the pipeline itself, so a client
disconnecting or reconnecting never affects the run in progress.
`GET /runs/{run_id}/model.xlsx` never reconstructs a filesystem path from
`run_id` -- it only ever serves the `artifact_path` the pipeline itself
already wrote to the run log (SECURITY.md item 5: no path is ever built
from user input).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse
from starlette.responses import FileResponse

from app.agent.runner import run_pipeline
from app.api.registry import RunHandle, RunRegistry
from app.api.schemas import RunCreateRequest, RunCreateResponse, RunStatusResponse
from app.edgar.client import EdgarClient, EdgarError, validate_ticker
from app.obs import RunLog
from app.settings import AppSettings

logger = logging.getLogger(__name__)

# How many runs may execute concurrently in this process. Bounds the number
# of simultaneously in-flight blocking pipelines (each dispatched to the
# default executor by LangGraph) a burst of requests could start at once,
# independent of and tighter than the per-IP/daily-cap controls, which
# limit *rate* over time, not concurrency at an instant.
_RUN_CONCURRENCY_LIMIT = 3
_run_semaphore = asyncio.Semaphore(_RUN_CONCURRENCY_LIMIT)


def get_settings_dep(request: Request) -> AppSettings:
    """The `AppSettings` this app was created with (see `app.main.create_app`)."""
    settings: AppSettings = request.app.state.settings
    return settings


def get_edgar_client_dep(request: Request) -> EdgarClient:
    """The single, process-wide `EdgarClient` (see `app.main`'s lifespan)."""
    client: EdgarClient = request.app.state.edgar_client
    return client


def get_run_log_dep(request: Request) -> RunLog:
    """The process-wide `RunLog`."""
    run_log: RunLog = request.app.state.run_log
    return run_log


def get_registry_dep(request: Request) -> RunRegistry:
    """The process-wide `RunRegistry` backing the SSE endpoint."""
    registry: RunRegistry = request.app.state.registry
    return registry


def _utc_midnight_today() -> float:
    """Unix timestamp for the start of the current UTC day (the daily-cap window)."""
    now = dt.datetime.now(dt.UTC)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp()


def _log_task_exception(task: asyncio.Task[None]) -> None:
    """Surface an exception from a fire-and-forget run task instead of letting it vanish.

    `run_pipeline` itself never raises (every failure path is caught and
    turned into a terminal event), so reaching this with an exception means
    a genuine bug in the driving/publishing code around it, not a pipeline
    failure -- worth a loud server-side log either way.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.exception("run driver task failed unexpectedly", exc_info=exc)


async def _drive(handle: RunHandle, **run_pipeline_kwargs: Any) -> None:
    """Run the pipeline for one run, publishing every event onto its `RunHandle`."""
    async with _run_semaphore:
        async for event in run_pipeline(**run_pipeline_kwargs):
            handle.publish(event)


async def create_run(
    payload: RunCreateRequest,
    request: Request,  # noqa: ARG001 - required by slowapi's rate limit, applied in app.main
    settings: AppSettings = Depends(get_settings_dep),
    edgar_client: EdgarClient = Depends(get_edgar_client_dep),
    run_log: RunLog = Depends(get_run_log_dep),
    registry: RunRegistry = Depends(get_registry_dep),
) -> RunCreateResponse:
    """Start a run for `payload.ticker` and return its id immediately.

    The ticker is validated here, at the boundary (SECURITY.md item 1),
    before it's used for anything else. A run is refused with 429 once the
    global daily cap is reached (SECURITY.md item 7) -- distinct from the
    per-IP rate limit `app.main.create_app` wraps this function in, which
    bounds one client's request rate rather than the whole deployment's
    daily spend.
    """
    try:
        ticker = validate_ticker(payload.ticker)
    except EdgarError as exc:
        raise HTTPException(status_code=422, detail="invalid ticker") from exc

    run_count_today = await asyncio.to_thread(run_log.count_since, _utc_midnight_today())
    if run_count_today >= settings.daily_run_cap:
        raise HTTPException(status_code=429, detail="daily run cap reached; try again tomorrow")

    run_id = str(uuid.uuid4())
    # Reserve the row *here*, synchronously, rather than letting it happen
    # inside run_pipeline (itself gated behind `_run_semaphore`): a burst of
    # requests could otherwise all pass the cap check above and all get
    # queued -- none of them counts toward `count_since` until it actually
    # starts running, which under load could be long after this check. This
    # closes that window; `create_run` is idempotent (`INSERT OR IGNORE`),
    # so run_pipeline's own call to it is a harmless no-op.
    await asyncio.to_thread(run_log.create_run, run_id, ticker)
    handle = registry.create(run_id, ticker)
    task = asyncio.create_task(
        _drive(
            handle,
            run_id=run_id,
            ticker=ticker,
            edgar_client=edgar_client,
            settings=settings,
            run_log=run_log,
        )
    )
    task.add_done_callback(_log_task_exception)
    handle.task = task

    return RunCreateResponse(run_id=run_id)


def _parsed_run_id(run_id: str) -> str:
    """Validate `run_id` looks like a server-generated id before any lookup.

    Every run id this API ever hands out is a `uuid.uuid4()` string
    (SECURITY.md item 5); rejecting anything else immediately, before it
    reaches a database query or a registry lookup, is cheap input hygiene
    on top of (not a substitute for) those lookups already being safe by
    construction (parameterized SQL; artifact paths are read from the run
    log, never rebuilt from this value).
    """
    try:
        return str(uuid.UUID(run_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc


async def get_run_status(
    run_id: str, run_log: RunLog = Depends(get_run_log_dep)
) -> RunStatusResponse:
    """Return a run's current durable outcome.

    `tieout_passed` is surfaced explicitly here because
    `TieoutReport.passed` is a `@property` on the frozen schema and would
    otherwise never appear in any serialized response.
    """
    validated_id = _parsed_run_id(run_id)
    record = await asyncio.to_thread(run_log.get, validated_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    return RunStatusResponse(
        run_id=record.run_id,
        ticker=record.ticker,
        status=record.status,
        tieout_passed=record.tieout_passed,
        narrate_ok=record.narrate_ok,
        error=record.error,
        created_at=record.created_at,
        completed_at=record.completed_at,
    )


async def download_workbook(
    run_id: str, run_log: RunLog = Depends(get_run_log_dep)
) -> FileResponse:
    """Serve the generated workbook for a completed run.

    The path served is always `record.artifact_path` as already written by
    the pipeline's own `generate` node -- never a path this route builds
    itself from `run_id` (SECURITY.md item 5).
    """
    validated_id = _parsed_run_id(run_id)
    record = await asyncio.to_thread(run_log.get, validated_id)
    if record is None or record.artifact_path is None:
        raise HTTPException(status_code=404, detail="workbook not found")
    return FileResponse(
        record.artifact_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"{record.ticker}-model.xlsx",
    )


async def stream_events(
    run_id: str, request: Request, registry: RunRegistry = Depends(get_registry_dep)
) -> EventSourceResponse:
    """Stream a run's `RunEvent`s live via SSE, replaying history on (re)connect.

    Reads its own cursor into `handle.events` rather than draining a shared
    queue, so connecting before the first event, connecting after the run
    finished, and multiple simultaneous viewers all replay the same
    history correctly (see `app.api.registry`'s module docstring). Honors
    `Last-Event-ID` for a browser's automatic `EventSource` reconnect.
    """
    validated_id = _parsed_run_id(run_id)
    handle = registry.get(validated_id)
    if handle is None:
        raise HTTPException(status_code=404, detail="run not found")

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        index = 0
        last_event_id = request.headers.get("last-event-id")
        if last_event_id is not None:
            try:
                index = int(last_event_id) + 1
            except ValueError:
                index = 0
        while True:
            while index < len(handle.events):
                event = handle.events[index]
                # No `event:` field -- every event is the default "message"
                # type, `step`/`status` distinguished in the JSON payload
                # itself, so the client needs one `onmessage` handler
                # rather than one `addEventListener` per pipeline step.
                yield {"id": str(index), "data": event.model_dump_json()}
                index += 1
            if handle.done:
                return
            # Clear *before* checking is_disconnected() and re-draining:
            # `is_disconnected()` itself awaits internally (a real
            # suspension point), so a `publish()` landing during that await
            # must still be observed. Clearing first, then re-checking
            # `len(handle.events)` before ever awaiting `wait()`, makes
            # this level- rather than edge-triggered -- paired with
            # `RunHandle.publish()` only ever calling `.set()` (see its
            # docstring for the failure this replaced: a lost wakeup that
            # could hang the stream forever on the terminal event).
            handle.updated.clear()
            if await request.is_disconnected():
                return
            if index < len(handle.events):
                continue
            await handle.updated.wait()

    return EventSourceResponse(event_generator(), ping=15)
