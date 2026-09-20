"""Optional Langfuse tracing: a no-op whenever Langfuse keys are absent.

`span()` is the only thing callers need: it opens a Langfuse span for the
duration of a `with` block when tracing is configured, and does nothing at
all (no client construction, no network call, negligible overhead) when it
isn't. `AppSettings.langfuse_enabled` (true only when both
`langfuse_public_key` and `langfuse_secret_key` are set) is the single
switch every call site implicitly depends on, so a demo/CI environment with
no Langfuse keys runs the exact same code path with tracing simply absent
from it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from app.settings import get_app_settings

if TYPE_CHECKING:
    from langfuse import Langfuse


@lru_cache(maxsize=1)
def _client() -> Langfuse | None:
    """Build (once) and cache the Langfuse client, or `None` if tracing is disabled.

    Imported lazily inside the function so this module never requires the
    `langfuse` package to be importable in an environment that simply has
    no keys configured -- mirroring the lazy-anthropic-import pattern in
    `app.rag.narrate`.
    """
    settings = get_app_settings()
    if not settings.langfuse_enabled:
        return None
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[None]:
    """Open a Langfuse span named `name` for the duration of the block.

    A pure no-op (no client construction, no network I/O) when tracing is
    disabled, so every graph node can unconditionally wrap itself in
    `with span("fetch", ticker=ticker):` regardless of whether Langfuse
    keys are configured.

    Args:
        name: Span name (e.g. the pipeline step: `"fetch"`, `"narrate"`).
        **attrs: Arbitrary JSON-serializable attributes recorded as the
            span's input. Never pass secret values here (SECURITY.md item 6).
    """
    client = _client()
    if client is None:
        yield
        return
    with client.start_as_current_observation(name=name, as_type="span", input=attrs or None):
        yield


def flush() -> None:
    """Flush any buffered Langfuse events. A no-op when tracing is disabled.

    Langfuse batches and sends spans on a background thread; call this at
    the end of a run (or process shutdown) so a short-lived process doesn't
    exit before its spans are sent.
    """
    client = _client()
    if client is not None:
        client.flush()
