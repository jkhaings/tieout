"""Process-wide structured logging setup.

CLAUDE.md rule 10: no `print()` anywhere in app code. Every module gets its
logger the standard way (`logging.getLogger(__name__)`); this module only
configures the root handler/formatter once, at process start.

Never logs settings values or secrets (SECURITY.md item 6) — callers are
responsible for not passing key material into a log message; this module
adds no filtering of its own beyond keeping the default format free of any
settings dump.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(run_id)s] %(message)s"
_DEFAULT_RUN_ID = "-"


class _RunIdFilter(logging.Filter):
    """Ensure every record has a `run_id` field, defaulting to `"-"`.

    Lets call sites attach `extra={"run_id": run_id}` for run-scoped log
    lines without requiring every other log call in the process to do the
    same (a bare `logger.info("...")` still formats cleanly).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Set `record.run_id` to the default if the caller didn't supply one."""
        if not hasattr(record, "run_id"):
            record.run_id = _DEFAULT_RUN_ID
        return True


def setup_logging(app_env: str) -> None:
    """Configure the root logger once, at process start.

    Args:
        app_env: The running environment (e.g. `"dev"`, `"prod"`). Only
            affects the chosen log level: `DEBUG` in anything other than
            `"prod"`, `INFO` in `"prod"`, so local runs are verbose by
            default without needing a separate flag.
    """
    root = logging.getLogger()
    if root.handlers:
        # Idempotent: repeated calls (e.g. from tests importing app.main
        # more than once) must not stack duplicate handlers.
        return

    level = logging.INFO if app_env == "prod" else logging.DEBUG
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(_RunIdFilter())
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet third-party loggers that are noisy at DEBUG and never carry a
    # run_id anyway (they only log connection-level detail).
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
