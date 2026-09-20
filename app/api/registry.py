"""In-memory registry of live and recently finished run handles for SSE.

A `RunHandle` is an append-only event log plus a completion flag -- never a
queue. `app/api/routes.py`'s SSE endpoint reads it with its own cursor, so
connecting early, connecting late (after the run finished), reconnecting,
and multiple simultaneous viewers of the same run id all work the same way:
each reader just walks `events` from wherever it left off. A shared
`asyncio.Queue` was considered and rejected -- two readers would split a
single queue's items between them, silently dropping half the stream for
each.

`RunRegistry` bounds memory two ways: a TTL sweep (run on every `create()`,
so no separate background task needs starting/cancelling) removes handles
finished more than `ttl_s` ago, and a hard cap evicts the oldest finished
handle first if the registry is still over `max_handles` afterward. Neither
sweep touches the on-disk workbook -- artifacts outlive their handle
deliberately, since `app.obs.RunLog.find_cached_run` (SECURITY.md item 7)
depends on the file still being there for a repeat request.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from app.schemas import RunEvent

# 30 minutes: long enough that a demo viewer who wanders off and comes back
# still finds their run's history; short enough that a bursty demo doesn't
# accumulate handles indefinitely between the daily cap resetting.
DEFAULT_TTL_S = 1800.0

# Hard ceiling independent of the TTL, so a sudden burst of runs within the
# TTL window can't grow the registry unboundedly.
DEFAULT_MAX_HANDLES = 64


@dataclass
class RunHandle:
    """One run's live event log, shared between its driver task and every SSE reader."""

    run_id: str
    ticker: str
    events: list[RunEvent] = field(default_factory=list)
    updated: asyncio.Event = field(default_factory=asyncio.Event)
    done: bool = False
    created_at: float = field(default_factory=time.time)
    task: asyncio.Task[None] | None = None

    def publish(self, event: RunEvent) -> None:
        """Append one event and wake every reader waiting on it.

        Only ever calls `.set()`, never `.clear()`: a reader is the one
        that clears the flag (see `app.api.routes.stream_events`'
        `event_generator`), immediately before re-checking for new events
        and only then awaiting it. A set()-then-clear() "pulse" here was
        tried first and is wrong -- verified directly with a reproduction
        matching the real read pattern: if a reader is suspended between
        draining `events` and calling `.wait()` (Starlette's own
        `Request.is_disconnected()` awaits internally, so this window is
        real, not theoretical), a publish's `set()` immediately followed by
        `clear()` can complete entirely during that suspension. The
        reader's later `.wait()` then blocks on an already-cleared event
        with no further `publish()` ever coming to un-block it if the
        missed one was the run's terminal `done`/`error` event -- the SSE
        connection hangs forever even though the run already finished.
        Never clearing here, combined with the reader's clear-then-recheck
        pattern, makes the notification level-triggered instead of
        edge-triggered, so a `set()` that happens before the reader even
        starts waiting is never lost.
        """
        self.events.append(event)
        if event.step in ("done", "error"):
            self.done = True
        self.updated.set()


class RunRegistry:
    """Bounded, TTL-swept in-memory map of `run_id -> RunHandle`."""

    def __init__(
        self, *, ttl_s: float = DEFAULT_TTL_S, max_handles: int = DEFAULT_MAX_HANDLES
    ) -> None:
        """Create an empty registry.

        Args:
            ttl_s: How long a finished handle is kept before the sweep
                removes it.
            max_handles: Hard cap; the oldest finished handle is evicted
                first once exceeded.
        """
        self._handles: OrderedDict[str, RunHandle] = OrderedDict()
        self._ttl_s = ttl_s
        self._max_handles = max_handles

    def _sweep(self) -> None:
        now = time.time()
        expired = [
            run_id
            for run_id, handle in self._handles.items()
            if handle.done and now - handle.created_at > self._ttl_s
        ]
        for run_id in expired:
            del self._handles[run_id]

        while len(self._handles) > self._max_handles:
            for run_id, handle in self._handles.items():
                if handle.done:
                    del self._handles[run_id]
                    break
            else:
                break  # nothing finished to evict; grow under load rather than drop a live run

    def create(self, run_id: str, ticker: str) -> RunHandle:
        """Register a new, empty handle for `run_id`, then sweep stale/excess ones.

        Sweeping *after* adding (not before) is what makes `max_handles` a
        real, immediately-enforced ceiling: sweeping first would only ever
        catch a registry that was already over the cap one call earlier,
        letting it transiently hold `max_handles + 1` entries until the
        next `create()`.
        """
        handle = RunHandle(run_id=run_id, ticker=ticker)
        self._handles[run_id] = handle
        self._sweep()
        return handle

    def get(self, run_id: str) -> RunHandle | None:
        """Look up a handle by run id, or `None` if it's unknown or has been swept."""
        return self._handles.get(run_id)
