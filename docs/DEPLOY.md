# DEPLOY — running tieout on the droplet behind Caddy

Live at `https://tieout.jasonkhaings.com`. One droplet (Ubuntu 24.04, 1 vCPU,
1.9 GB RAM) already runs another service on port 8000 via systemd; tieout runs
as a Docker container bound to `127.0.0.1:8010` and Caddy reverse-proxies the
new hostname to it, appended to the existing `/etc/caddy/Caddyfile` without
touching the other site blocks.

## Layout

```
/opt/tieout/
  repo/    git clone of this repository, pinned to whatever commit was last built
  .env     chmod 600, root:root — real env vars, not committed, not baked into the image
  data/    chown 1000:1000 — bind-mounted to /app/data, holds the EDGAR cache,
           runs.db, and generated workbooks
  run.sh   docker rm -f + docker run, so "recreate" is one idempotent command
```

`data/` must be owned by uid 1000: the image runs as `appuser` (uid 1000,
`Dockerfile`), and `RunLog`'s SQLite init fails at container startup against a
root-owned mount.

## First deploy

```bash
mkdir -p /opt/tieout && cd /opt/tieout
git clone https://github.com/jkhaings/tieout.git repo
mkdir -p data && chown 1000:1000 data
```

Write `.env` (see `.env.example` for the full list; below is what production
needs to differ from the defaults):

```
ANTHROPIC_API_KEY=sk-ant-...
SEC_USER_AGENT=tieout/0.1 (contact: you@example.com)
APP_ENV=prod
CORS_ORIGINS=https://tieout.jasonkhaings.com
RUNS_DIR=/app/data/runs
RUN_LOG_PATH=/app/data/runs.db
CACHE_DIR=/app/data/cache
ENABLE_LOCAL_EMBEDDINGS=false
```

Three things that bite if gotten wrong:

- **`--env-file` takes values literally** — no shell parsing, no inline
  comments, no quoting. A quoted `SEC_USER_AGENT` becomes a User-Agent with
  literal quote characters in it.
- **`APP_ENV=prod` is not cosmetic.** It is the only consumer of that
  variable (`app/obs/logging.py`) and any other value — including the
  default `dev` — leaves the deployment logging at DEBUG.
- **`ENABLE_LOCAL_EMBEDDINGS` stays `false`.** The `ml` extra isn't in the
  image (`Dockerfile` skips `--all-extras` on purpose), and those embedder
  classes download Hugging Face weights with no timeout of their own —
  outside SECURITY.md item 2's EDGAR-only allowlist. Retrieval runs
  BM25-only in production, which is what the scorecard reports.

Paths are absolute on purpose: `EdgarSettings.cache_dir` has no
resolve-validator and is CWD-relative, so it would silently move if the
container's working directory ever changed.

Build and run:

```bash
docker build -t tieout:latest /opt/tieout/repo
bash /opt/tieout/run.sh
```

`run.sh` runs the container with `--restart unless-stopped`, `-p
127.0.0.1:8010:8000` (never `0.0.0.0` — that's what makes the Dockerfile's
`--forwarded-allow-ips=*` safe; its own comment says so), the `data/` bind
mount, and a `768m` memory cap so a tieout run can never starve the other
service on this box.

## Caddy

One block appended to `/etc/caddy/Caddyfile`, proxying to `127.0.0.1:8010`.
The app ships its own CSP and security headers (`app/api/security.py`) — do
not add a CSP in Caddy. Three details that matter:

- `header_up X-Forwarded-For {remote_host}` **overwrites** rather than
  appends, closing the spoofing path where a caller supplies its own
  `X-Forwarded-For` to forge a different rate-limit identity
  (SECURITY.md item 7). Verified live: a request with a forged
  `X-Forwarded-For` still hits the real rate limit.
- `flush_interval -1` plus excluding `/runs/*/events` from `encode` — the
  SSE stream (`ping=15`) must not be buffered or compressed, or the client
  sees nothing until the whole run finishes.
- `read_timeout`/`write_timeout` of 360s, longer than `RUN_TIMEOUT_S =
  300.0` (`app/agent/runner.py`).
- `/docs`, `/redoc`, and `/openapi.json` are 404'd at the Caddy layer.
  FastAPI never disables them; `/docs` renders blank anyway (the app's own
  CSP blocks the Swagger UI's CDN), but `/openapi.json` served the full
  schema until this was added.

```bash
caddy validate --config /etc/caddy/Caddyfile
systemctl reload caddy
```

Caddy issues the Let's Encrypt certificate automatically on reload; no
separate ACME step.

## Redeploying a new commit

```bash
cd /opt/tieout/repo && git fetch origin && git reset --hard origin/main
docker build -t tieout:latest /opt/tieout/repo
bash /opt/tieout/run.sh
```

`run.sh` is a full recreate (`docker rm -f` then `docker run`), not a
restart — the running container is briefly unavailable. `data/` is untouched
by this, so the EDGAR cache and workbook cache survive.

## Rotating the Anthropic key

Edit `/opt/tieout/.env`, then `bash /opt/tieout/run.sh` — a **recreate, not
`docker restart`**. `--env-file` is read once at `docker run` time, so a
restart keeps whatever key the container was created with.

## Operational caveats

- **The EDGAR disk cache never expires or revalidates**
  (`app/edgar/client.py`): once `submissions/<CIK>.json` is cached, a newly
  filed 10-K is invisible until `data/cache/` is cleared by hand. There is
  no cron for this yet.
- **Per-IP rate-limit buckets are in-memory** and reset on every container
  restart; the daily run cap is in `runs.db` and does not.
- **Use `docker stop`, not `docker kill`.** A SIGKILL strands any in-flight
  run at `status="running"` forever — nothing reconciles it at startup. A
  graceful stop lets the pipeline's own cancellation handler mark it
  `error`.

## Rollback

Caddy: restore the timestamped backup taken before the edit
(`Caddyfile.bak.<date>`), `caddy validate`, `systemctl reload caddy`. App:
`docker rm -f tieout` leaves `/opt/tieout/data` intact for the next
`run.sh`. Neither step touches this droplet's other services.

## Known gap, not fixed here

A ticker over 16 characters is rejected by FastAPI's own request-validation
layer before the app's `invalid ticker` check runs, so the response echoes
the submitted input in a shape that doesn't match `ErrorResponse` (no stack
trace, no secret — just inconsistent and noisier than every other error
path). `app/api` is out of lane for a deploy session; noted here for
whoever picks it up next.
