"""Tests for app.api.registry: RunHandle's notify pattern, RunRegistry's bounds."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.api.registry import DEFAULT_TTL_S, RunHandle, RunRegistry
from app.schemas import RunEvent


def _event(run_id: str, step: str, status: str = "ok") -> RunEvent:
    return RunEvent(run_id=run_id, step=step, status=status, detail="", ts=time.time())


async def _reader_like_event_generator(
    handle: RunHandle, *, disconnected: asyncio.Event, drained: list[RunEvent]
) -> None:
    """A faithful port of `app.api.routes.stream_events`'s `event_generator` loop.

    `disconnected` stands in for `Request.is_disconnected()`: awaiting it
    is the real suspension point a genuine reader also hits, and the test
    controls exactly when it resolves so a publish can be made to land
    inside that window deterministically.
    """
    index = 0
    while True:
        while index < len(handle.events):
            drained.append(handle.events[index])
            index += 1
        if handle.done:
            return
        handle.updated.clear()
        await disconnected.wait()  # stands in for `await request.is_disconnected()`
        if index < len(handle.events):
            continue
        await handle.updated.wait()


async def test_publish_during_the_readers_suspension_window_is_not_lost() -> None:
    """Regression test: a publish landing while the reader awaits is never missed.

    Reproduces the exact race a set()-then-clear() "pulse" pattern is
    vulnerable to: `Request.is_disconnected()` awaits internally, a real
    suspension point in the reader's loop between draining `events` and
    calling `.wait()`. If a `publish()` (including the run's terminal
    event) completes entirely during that suspension, a pulse-based
    `publish()` loses it -- the reader's later `.wait()` then blocks
    forever with nothing left to wake it. `RunHandle.publish()` no longer
    pulses (only `.set()`, never `.clear()`), and the reader clears then
    re-checks before waiting -- this proves that combination is race-free.
    """
    handle = RunHandle(run_id="r1", ticker="AAPL")
    drained: list[RunEvent] = []
    disconnected = asyncio.Event()  # deliberately never set -- keeps the reader suspended there

    reader = asyncio.create_task(
        _reader_like_event_generator(handle, disconnected=disconnected, drained=drained)
    )
    # Let the reader run until it's blocked awaiting `disconnected` --
    # exactly the window the real bug lived in.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Publish the run's terminal event while the reader is suspended there.
    handle.publish(_event("r1", "done"))

    # Now let the reader's disconnect-check resolve and continue.
    disconnected.set()
    await asyncio.wait_for(reader, timeout=2.0)

    assert [e.step for e in drained] == ["done"]
    assert handle.done is True


async def test_multiple_simultaneous_readers_each_see_the_full_history() -> None:
    """Two independent readers of the same handle must each see every event, in full."""
    handle = RunHandle(run_id="r2", ticker="AAPL")
    steps = ["fetch", "build", "verify", "retrieve", "narrate", "generate", "done"]

    async def reader() -> list[str]:
        index = 0
        seen: list[str] = []
        while True:
            while index < len(handle.events):
                seen.append(handle.events[index].step)
                index += 1
            if handle.done:
                return seen
            handle.updated.clear()
            if index < len(handle.events):
                continue
            await handle.updated.wait()

    reader_a = asyncio.create_task(reader())
    reader_b = asyncio.create_task(reader())
    await asyncio.sleep(0)

    for step in steps:
        handle.publish(_event("r2", step, status="ok" if step != "done" else "ok"))
        await asyncio.sleep(0)

    result_a = await asyncio.wait_for(reader_a, timeout=2.0)
    result_b = await asyncio.wait_for(reader_b, timeout=2.0)
    assert result_a == steps
    assert result_b == steps


async def test_late_reader_after_completion_drains_full_history_without_waiting() -> None:
    handle = RunHandle(run_id="r3", ticker="AAPL")
    handle.publish(_event("r3", "fetch"))
    handle.publish(_event("r3", "done"))

    index = 0
    drained: list[str] = []
    while index < len(handle.events):
        drained.append(handle.events[index].step)
        index += 1
    assert handle.done is True
    assert drained == ["fetch", "done"]


def test_sweep_removes_a_finished_handle_past_its_ttl() -> None:
    registry = RunRegistry(ttl_s=1.0, max_handles=64)
    handle = registry.create("old-run", "AAPL")
    handle.publish(_event("old-run", "done"))
    handle.created_at = time.time() - 2.0  # older than ttl_s

    registry.create("new-run", "MSFT")  # triggers a sweep

    assert registry.get("old-run") is None
    assert registry.get("new-run") is not None


def test_sweep_keeps_a_finished_handle_within_ttl() -> None:
    registry = RunRegistry(ttl_s=DEFAULT_TTL_S, max_handles=64)
    handle = registry.create("recent-run", "AAPL")
    handle.publish(_event("recent-run", "done"))

    registry.create("another-run", "MSFT")

    assert registry.get("recent-run") is not None


def test_max_handles_evicts_oldest_finished_handle_first() -> None:
    registry = RunRegistry(ttl_s=DEFAULT_TTL_S, max_handles=2)
    first = registry.create("first", "AAPL")
    first.publish(_event("first", "done"))
    second = registry.create("second", "MSFT")
    second.publish(_event("second", "done"))

    # Over the cap; "first" is the oldest finished handle and must go.
    registry.create("third", "GOOGL")

    assert registry.get("first") is None
    assert registry.get("second") is not None
    assert registry.get("third") is not None


def test_max_handles_never_evicts_a_still_running_handle() -> None:
    """A live (not-done) handle must never be dropped just to make room."""
    registry = RunRegistry(ttl_s=DEFAULT_TTL_S, max_handles=1)
    still_running = registry.create("running", "AAPL")
    assert still_running.done is False

    registry.create("newcomer", "MSFT")  # would exceed max_handles=1

    # Nothing finished exists to evict, so both must still be present.
    assert registry.get("running") is not None
    assert registry.get("newcomer") is not None


@pytest.mark.parametrize("terminal_step", ["done", "error"])
def test_publish_sets_done_only_on_a_terminal_step(terminal_step: str) -> None:
    handle = RunHandle(run_id="r4", ticker="AAPL")
    handle.publish(_event("r4", "fetch"))
    assert handle.done is False
    handle.publish(
        _event("r4", terminal_step, status="ok" if terminal_step == "done" else "failed")
    )
    assert handle.done is True
