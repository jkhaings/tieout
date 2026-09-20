"""Observability: structured logging, optional Langfuse tracing, and the run log.

Re-exports the pieces `app/agent` and `app/api` need, so they have one place
to import from -- the same pattern `app/rag/__init__.py` established.
"""

from __future__ import annotations

from app.obs.logging import setup_logging
from app.obs.runlog import RunLog, RunRecord
from app.obs.tracing import flush, span

__all__ = [
    "RunLog",
    "RunRecord",
    "flush",
    "setup_logging",
    "span",
]
