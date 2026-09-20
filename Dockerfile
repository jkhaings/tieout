# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Astral publishes `uv` as a static binary specifically for this
# copy-from-image pattern; the *dependency graph* it resolves is fully
# reproducible regardless of uv's own version, since `uv sync --frozen`
# below refuses to deviate from the committed `uv.lock` (SECURITY.md item
# 9). Using `latest` here only pins "some uv binary," not the app's deps.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependencies first, in their own layer: this only re-runs on a
# pyproject.toml/uv.lock change, not on every source edit.
COPY pyproject.toml uv.lock ./
# No --all-extras: the `ml` extra (chromadb, sentence-transformers/torch)
# is a large, optional dependency this app doesn't need by default --
# AppSettings.enable_local_embeddings defaults to False specifically
# because those libraries download model weights from Hugging Face Hub on
# first use with no timeout of their own (see app/settings.py's
# docstring). Matches CI, which also runs without --all-extras.
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

# Non-root (SECURITY.md/CLAUDE.md requirement). `/app` (including the
# `data/` directory the app creates at runtime for the EDGAR cache, run
# artifacts, and the run log) must be writable by this user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# --workers 1: EdgarClient's rate limiter is per-process, not shared across
# workers (see app/main.py's lifespan) -- more than one worker would let
# each independently believe it owns the full SEC request budget.
# --proxy-headers/--forwarded-allow-ips: this container sits behind a
# reverse proxy (Caddy, per SECURITY.md item 8); without these, slowapi's
# per-IP rate limit would key on the proxy's address instead of the real
# client's, collapsing it into one shared bucket for every visitor.
# `*` trusts X-Forwarded-For from any peer -- correct only because this
# container is never exposed directly to the internet, solely reachable
# through Caddy. Tighten to Caddy's actual container/network address (e.g.
# via a compose-level command override) in any deployment where that
# isn't guaranteed.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips=*"]
