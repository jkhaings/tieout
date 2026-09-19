# SESSIONS.md — the 2-day plan

Four Claude Code sessions, parallelized with git worktrees so they never touch the same files. `app/schemas.py` is the shared contract and is frozen — every session codes against it.

## 0. Worktree setup (after the scaffold is pushed)

```bash
cd ~/dev/tieout
git worktree add ../tieout-data -b session/data
git worktree add ../tieout-rag -b session/rag
```

Run Session A in `../tieout-data`, Session B in `../tieout-rag`, in parallel. Sessions C and D run later in the main checkout.

## Session A — data-engineer (Sat morning, worktree: tieout-data)

Paste into Claude Code:

> Read CLAUDE.md, ARCHITECTURE.md, SECURITY.md and app/schemas.py. Work as the data-engineer (.claude/agents/data-engineer.md): you own app/edgar and app/model plus their tests, nothing else. Build in order: (1) app/edgar/client.py — httpx client for companyfacts, submissions, and 10-K HTML with disk cache under data/cache, SEC User-Agent from settings, ticker→CIK resolution via company_tickers.json, host allowlist, backoff; (2) app/edgar/tags.py — ~35 canonical line items mapped to us-gaap tag fallback chains covering IS/BS/CF; (3) app/model/builder.py — StatementSet for the last 5 fiscal years; (4) app/model/verifier.py — tie-out checks (balance sheet equation, retained earnings roll-forward, cash roll-forward, statement subtotals) → TieoutReport; (5) app/model/excel.py — workbook with IS/BS/CF tabs, a Ratios tab using live formulas, a Tie-out tab, and escape_cell() on every text write. Download companyfacts for AAPL and MSFT once, trim them into tests/fixtures/, and write hermetic unit tests against those fixtures. Done = make test, make lint, make type all green. Commit conventionally as you go. Do not push.

## Session B — rag-engineer (Sat morning, worktree: tieout-rag, parallel with A)

> Read CLAUDE.md, ARCHITECTURE.md, SECURITY.md and app/schemas.py. Work as the rag-engineer (.claude/agents/rag-engineer.md): you own app/rag and its tests, nothing else. Build: (1) ingest.py — parse a 10-K HTML into sections (Item 7, Item 1A), chunk ~800 tokens with heading context → Chunk objects; (2) index.py — Chroma collection + BM25 index (guard ml imports so the module degrades gracefully without extras); (3) retrieve.py — hybrid search fused with reciprocal rank fusion, cross-encoder rerank, relevance threshold; (4) narrate.py — per-line-item grounded commentary via Anthropic API returning the Commentary schema, citation quotes enforced as verbatim substrings of their source chunks, injection-safe delimiters around filing text, one schema retry then fail closed. Save one real 10-K HTML excerpt as a test fixture. Tests use a fake embedder and a fake LLM — zero network. Done = make test, make lint, make type green. Commit conventionally. Do not push.

## Merge (Sat afternoon)

```bash
cd ~/dev/tieout
git merge session/data && git merge session/rag
make test && git push origin main
```

One push. CI runs once.

## Session C — platform-engineer (Sat evening → Sun morning, main checkout)

> Read CLAUDE.md, ARCHITECTURE.md, SECURITY.md, app/schemas.py, and skim app/edgar, app/model, app/rag. Work as the platform-engineer (.claude/agents/platform-engineer.md): you own app/agent, app/api, app/obs, web/, and the Dockerfile. Build: (1) app/obs — logging setup, optional Langfuse tracing (no-op without keys), SQLite run log; (2) app/agent/graph.py — LangGraph pipeline fetch → build → verify → retrieve → narrate → generate, each node emitting RunEvents, per-node error handling, fail-closed behavior on partial failures; (3) app/api — POST /runs with ticker validation and slowapi rate limiting, SSE events endpoint, xlsx download, security headers, locked-down CORS; (4) web/index.html — single Tailwind-CDN page that starts a run and streams the steps live; (5) Dockerfile — python:3.12-slim, non-root user. Then run the pipeline end-to-end on AAPL locally and fix what breaks. Done = end-to-end run produces a verified workbook, all checks green. One push at the end.

## Session D — eval-engineer (Sun afternoon, main checkout)

> Read CLAUDE.md, ARCHITECTURE.md, and the full codebase briefly. Work as the eval-engineer (.claude/agents/eval-engineer.md): you own evals/, .github/, README.md. Build: (1) evals/datasets — hand-label ~30 retrieval queries with relevant chunk ids for 2 tickers, plus a gold tie-out set from raw companyfacts values; (2) evals/tieout_eval.py — binary cell-level accuracy vs gold; (3) evals/retrieval_eval.py — precision/recall@k; (4) evals/judge.py — LLM-as-judge rubric (grounded, cited, no invented numbers) using a different model than the generator; (5) evals/scorecard.py — run everything, write scorecard.json, render a table into README.md. Then write the full README: what it is, 60-second demo, architecture summary, the scorecard, failure modes, security notes. Finish by running the security-reviewer subagent over the repo and fixing anything High. One final push.

## Sunday evening — ship it
Deploy the container to the droplet behind Caddy, record a ~60s demo GIF for the README and LinkedIn, add the repo to jasonkhaings.com.

## Study list (tonight, ~2 hours — enough to defend every choice in an interview)
1. Open `https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json` in a browser and just read the shape — us-gaap tags, units, fiscal years. This is the whole data source.
2. XBRL in one sitting: why the same concept has different tags per company, and what a tag fallback chain is.
3. Hybrid retrieval: BM25 vs embeddings, and how reciprocal rank fusion combines them. Then what a cross-encoder reranker adds.
4. LangGraph quickstart — nodes, state, edges. 20 minutes.
5. LLM-as-judge pitfalls: position bias, self-preference, why the judge should be a different model than the generator.

## Cost notes
Public repo → GitHub Actions minutes are free, and the workflow cancels superseded runs automatically. Keep the habit anyway: one push per round. LLM calls never run in CI; evals run locally. Expected weekend API spend: $5–10. If the Docker image gets fat from torch, switch embeddings to Chroma's built-in ONNX embedder and drop the rerank — noted trade-off, acceptable.
