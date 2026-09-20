"""SQLite run log: durable run history, the daily run cap, and the workbook cache.

One row per run, in a single `runs` table. Every method opens its own
short-lived connection rather than sharing one: the graph pipeline runs on a
worker thread (`asyncio.to_thread`, see `app/api`) while the API reads the
same database from the event-loop thread, and `sqlite3` connections are not
safe to share across threads. A fresh connection per call sidesteps that
entirely, and WAL mode (set on every connection) keeps a writer and a reader
from blocking each other.

Two SECURITY.md controls live here:
- item 7 ("a global daily run cap ... generated workbooks cached by
  (ticker, filing) so repeat requests cost nothing") -- `count_since` and
  `find_cached_run`.
- item 10 ("clients receive generic messages with a run id, never stack
  traces or config values") -- `mark_error` stores only the short, generic
  string the API already decided to show the client, never an exception's
  full text.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    completed_at REAL,
    accession_number TEXT,
    artifact_path TEXT,
    tieout_passed INTEGER,
    narrate_ok INTEGER,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs (created_at);
CREATE INDEX IF NOT EXISTS idx_runs_ticker_accession ON runs (ticker, accession_number);
"""


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One row of the run log."""

    run_id: str
    ticker: str
    status: str  # "running" | "done" | "error"
    created_at: float
    completed_at: float | None
    accession_number: str | None
    artifact_path: str | None
    tieout_passed: bool | None
    narrate_ok: bool | None
    error: str | None


def _to_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        ticker=row["ticker"],
        status=row["status"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
        accession_number=row["accession_number"],
        artifact_path=row["artifact_path"],
        tieout_passed=None if row["tieout_passed"] is None else bool(row["tieout_passed"]),
        narrate_ok=None if row["narrate_ok"] is None else bool(row["narrate_ok"]),
        error=row["error"],
    )


class RunLog:
    """Durable run history, backed by a SQLite file at `db_path`."""

    def __init__(self, db_path: Path) -> None:
        """Open (creating if needed) the run log at `db_path`.

        Args:
            db_path: Path to the SQLite database file. Its parent directory
                is created if it does not exist.
        """
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived, WAL-mode connection; committed and closed on exit."""
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_run(self, run_id: str, ticker: str, *, created_at: float | None = None) -> None:
        """Insert a new `"running"` row, or do nothing if `run_id` already has one.

        `INSERT OR IGNORE` rather than a plain `INSERT`: `app.api.routes.create_run`
        calls this synchronously, before scheduling the background task, so
        the daily run cap (`count_since`, SECURITY.md item 7) reflects a
        queued-but-not-yet-started run immediately rather than only once it
        reaches the front of the concurrency semaphore -- otherwise a burst
        of requests could queue far more than `daily_run_cap` runs before
        any of them actually counts. `app.agent.runner.run_pipeline` then
        calls this again (its own row is usually already there); `OR
        IGNORE` makes the second call a safe no-op instead of a primary-key
        `IntegrityError`, so `run_pipeline` stays independently callable
        (as several tests do) without requiring a caller to have already
        reserved the row.

        Args:
            run_id: Server-generated run id (SECURITY.md item 5: never
                derived from user input).
            ticker: The validated ticker this run is for.
            created_at: Unix timestamp; defaults to now. Ignored if a row
                for `run_id` already exists.
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO runs (run_id, ticker, status, created_at) "
                "VALUES (?, ?, 'running', ?)",
                (run_id, ticker, created_at if created_at is not None else time.time()),
            )

    def mark_done(
        self,
        run_id: str,
        *,
        tieout_passed: bool,
        narrate_ok: bool,
        accession_number: str | None,
        artifact_path: str | None,
        completed_at: float | None = None,
    ) -> None:
        """Mark a run complete and record its outcome.

        Args:
            run_id: The run to update.
            tieout_passed: Whether every tie-out check passed
                (`TieoutReport.passed`) -- recorded explicitly since that
                property does not serialize on its own.
            narrate_ok: Whether narration produced at least one grounded
                commentary (`False` is a normal, fail-closed outcome, not
                an error).
            accession_number: The filing's accession number, for the
                (ticker, accession) workbook cache.
            artifact_path: Path to the generated workbook, if one shipped.
            completed_at: Unix timestamp; defaults to now.
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = 'done', completed_at = ?, accession_number = ?,
                    artifact_path = ?, tieout_passed = ?, narrate_ok = ?
                WHERE run_id = ?
                """,
                (
                    completed_at if completed_at is not None else time.time(),
                    accession_number,
                    artifact_path,
                    int(tieout_passed),
                    int(narrate_ok),
                    run_id,
                ),
            )

    def mark_error(self, run_id: str, *, error: str, completed_at: float | None = None) -> None:
        """Mark a run failed with a short, generic error string.

        SECURITY.md item 10: never pass an exception's full text or a stack
        trace here -- callers must have already reduced it to the same
        generic message shown to the client.

        Args:
            run_id: The run to update.
            error: Generic error string, safe to serve back to a client.
            completed_at: Unix timestamp; defaults to now.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = 'error', completed_at = ?, error = ? WHERE run_id = ?",
                (completed_at if completed_at is not None else time.time(), error, run_id),
            )

    def get(self, run_id: str) -> RunRecord | None:
        """Fetch one run by id, or `None` if it does not exist."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else _to_record(row)

    def count_since(self, since_ts: float) -> int:
        """Count runs created at or after `since_ts` -- backs the global daily run cap."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM runs WHERE created_at >= ?", (since_ts,)
            ).fetchone()
        return int(row["n"])

    def find_cached_run(self, ticker: str, accession_number: str) -> RunRecord | None:
        """Return the most recent completed run's record for (ticker, accession), if any.

        SECURITY.md item 7: "generated workbooks cached by (ticker, filing)
        so repeat requests cost nothing." Returns the full record (not just
        the artifact path) so a cache hit can be recorded as its own run
        with the same `tieout_passed`/`narrate_ok` outcome, rather than
        just handed a bare file. `None` if there is no cached workbook or
        its file has since been removed from disk.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM runs
                WHERE ticker = ? AND accession_number = ? AND status = 'done'
                    AND artifact_path IS NOT NULL
                ORDER BY created_at DESC LIMIT 1
                """,
                (ticker, accession_number),
            ).fetchone()
        if row is None:
            return None
        record = _to_record(row)
        assert record.artifact_path is not None  # guaranteed by the WHERE clause above
        return record if Path(record.artifact_path).exists() else None
