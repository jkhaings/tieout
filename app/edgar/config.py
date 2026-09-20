"""Configuration for the EDGAR client.

A small, in-lane settings module (CLAUDE.md ownership map: `app/edgar/` is
the data-engineer's). Kept separate from a project-wide settings module so
this session cannot collide with the parallel `session/rag` worktree, which
needs its own config (e.g. `ANTHROPIC_API_KEY`). `extra="ignore"` lets both
coexist against the same `.env` file until a later session hoists everything
into a shared `app/settings.py`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EdgarSettings(BaseSettings):
    """EDGAR client configuration, loaded from the environment / `.env`.

    `sec_user_agent` defaults to empty rather than a placeholder value: an
    unconfigured User-Agent must fail loudly the moment a real request is
    attempted (see `app.edgar.client`), not silently send something SEC
    would reject or that misidentifies the requester.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    sec_user_agent: str = Field(default="")
    cache_dir: Path = Field(default=Path("data/cache"))
    max_requests_per_second: float = Field(default=8.0, gt=0, le=10.0)
    request_timeout_s: float = Field(default=15.0, gt=0)


@lru_cache(maxsize=1)
def get_settings() -> EdgarSettings:
    """Return the process-wide `EdgarSettings`, constructed once and cached."""
    return EdgarSettings()
