"""API request/response models. Kept separate from the frozen `app/schemas.py`.

None of these are the shared pipeline contracts (those stay in
`app.schemas`, FROZEN per CLAUDE.md) -- these are the HTTP-facing shapes
`app/api`'s routes serialize, including the one place `TieoutReport.passed`
(a `@property`, so it never appears in a `model_dump()`) is surfaced
explicitly to a client.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunCreateRequest(BaseModel):
    """Body of `POST /runs`."""

    ticker: str = Field(min_length=1, max_length=16)


class RunCreateResponse(BaseModel):
    """Response of `POST /runs`: the id to poll/stream/download with."""

    run_id: str


class RunStatusResponse(BaseModel):
    """Response of `GET /runs/{run_id}`: the run's current durable outcome.

    `tieout_passed` and `narrate_ok` are `None` while the run is still in
    progress (`status == "running"`) and populated once it reaches `"done"`
    or `"error"`.
    """

    run_id: str
    ticker: str
    status: str  # "running" | "done" | "error"
    tieout_passed: bool | None
    narrate_ok: bool | None
    error: str | None
    created_at: float
    completed_at: float | None


class ErrorResponse(BaseModel):
    """Generic error body (SECURITY.md item 10): a message and a run id, never internals."""

    error: str
    run_id: str | None = None
