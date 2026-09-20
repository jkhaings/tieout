# BUILDLOG — what each session shipped and what the reviews caught

Raw session summaries from the 2-day build. Kept as evidence: every bug below was
found by our own tests or adversarial review passes before anything shipped.

## Session A — data-engineer (deterministic pipeline)

Built the full deterministic pipeline (7 commits): cached, rate-limited,
host-allowlisted SEC client — verified end-to-end against live SEC, including
against subdomain/userinfo confusion attacks; 43 canonical line items → XBRL
fallback chains, validated against 8 real tickers (not just AAPL/MSFT);
per-fiscal-year fact selection engine; 5-year StatementSet builder; tie-out
checks + RE reconciliation; 5-tab workbook with live ratio formulas and
escape_cell(). 120 tests, lint/type green.

Deviations, all evidence-based:
- 43 line items, not ~35 — real filers need a 3-way SG&A split and one extra
  reconciling term to make checks tie (documented in tags.py).
- RE roll-forward is informational, not scored — the roll-forward cannot close
  from XBRL alone (AAPL leaves ~$17.7B unexplained; buybacks charged to retained
  earnings aren't reliably tagged), so it renders as a labeled reconciliation
  instead of a false FAIL. Two other candidate checks (BS current-asset
  subtotals, component-built operating income) false-failed on real filings and
  were dropped rather than shipped as unreliable.
- No hidden 6th year in StatementSet — roll-forward checks run over 4 of 5
  years, keeping the frozen schema unambiguous for parallel sessions.

Bug caught by our own tests: FactIndex hardcoded the "USD" unit bucket,
silently zeroing out eps_diluted / weighted_diluted_shares (filed as USD/shares/
shares). Fixed by threading a unit field through TagSpec. Spot-checked against
live filings (e.g. AAPL FY2024 total assets $364,980,000,000; G&A correctly
None for FY2021–2022 since Apple only began disclosing it separately in FY2025).

## Session B — rag-engineer (retrieval + grounded narration)

Built ingest.py (10-K section parsing with TOC-trap and heading-vs-prose
disambiguation, ~800-token chunking), index.py (BM25 + Chroma/in-memory hybrid,
graceful degradation without the ml extra), retrieve.py (RRF fusion +
cross-encoder reranking), narrate.py (grounded per-line-item commentary,
injection-safe delimiting, one retry then fail-closed), a real trimmed Apple
10-K fixture, 47 hermetic tests (fake embedder, fake LLM, zero network).

Adversarial review caught 9 real defects across four independent review lenses,
all confirmed by ground-truth verification and fixed with regression tests:
- High: delimiter neutralization was case-sensitive (</FILING_EXCERPT> bypassed
  it entirely); number-grounding used raw substring containment (a fabricated
  "1.04" passed because it is a substring of "$391.04B"); the no-reranker
  retrieval path never failed closed on irrelevant queries (RRF scores are
  always positive); two ingest.py bugs where inline cross-references could
  truncate or displace the real section — fixed structurally by preserving
  paragraph boundaries before heading detection.
- Medium: Commentary(text=None) could still carry citations; empty-string
  citations passed as valid quotes; a model-echoed chunk_id could forge a second
  fake delimiter block in the retry prompt; ML models fetched unpinned from
  Hugging Face Hub.
- Low: no API timeout; overly broad Any typing.

## Session C — platform-engineer (orchestration, API, UI, ops)

Consolidated settings into app/settings.py — closing a real hermeticity hole
where tests would have silently started making live, billed API calls the moment
a real .env key existed. Built app/obs (logging, no-op Langfuse tracing, SQLite
run log), the six-node LangGraph pipeline (fail-closed per CLAUDE.md rule 4,
Commentary sheet, workbook cache keyed by ticker+filing), the API (POST /runs,
SSE streaming, xlsx download, rate limiting, daily cap, security headers, CORS,
body-size limit), the demo page, and a non-root Dockerfile verified with a real
container build.

Live verification: AAPL end-to-end in 71s with real narration (6/10 line items
grounded with verbatim citations); the identical second request hit the cache
and returned in 0.07s.

Two review passes surfaced two HIGH bugs, both fixed with before/after
reproductions: setup code that could leave a run stuck at status="running"
forever on any exception, and a lost-wakeup race in the SSE notify pattern that
could hang a stream forever on the terminal event. Also closed: a TOCTOU race
letting bursts exceed the daily cap, the workbook cache existing but never
wired up, and a missing body-size limit — all named SECURITY.md requirements
that weren't actually enforced. 227 tests, lint/type green.

Carried over for the next session: an unsynchronized cache-write race in
app/edgar/client.py, and sign-unaware number-grounding in app/rag/narrate.py
(a decline could in principle be narrated as growth and still pass grounding).
Both out-of-lane at the time; fixed in Session D.

## Session D — eval-engineer (evals, README, and the two carried-over fixes)

Fixed both carried-over defects, with regression tests proving each was
broken before the fix and fixed after:

- Cache-write race: EdgarClient wrote cache entries with a bare
  `Path.write_text()`; a reader's `cache_path.exists()` check goes true the
  instant the file is truncated, before content lands, so two concurrent
  runs sharing the process-wide client (the API layer allows 3 concurrent
  runs) could genuinely race on the same cache path — a JSON reader got a
  `JSONDecodeError`, an HTML reader got silently truncated filing text fed
  into RAG, no exception at all. Fixed with a temp-file-in-the-same-
  directory + `fsync` + `os.replace` atomic publish, so a reader only ever
  sees the fully-old or fully-new file; a cache-write failure now degrades
  to "not cached" (logging only the entry's hash filename) rather than
  failing an otherwise-successful fetch. Verified with a threading.Barrier-
  synchronized concurrency test and a monkeypatched-`os.replace` spy proving
  the destination is never observed mid-write.
- Sign-/direction-blind number grounding: the check matched a claimed
  number's *magnitude* against the given figures but not its *sign* or the
  *direction of change* between years, so a claim could be narrated with
  the wrong sign or the wrong direction and still pass. Fixed by tracking
  each number's polarity and requiring it to match the figure's own
  polarity, plus a new direction-of-change check anchored to the specific
  fiscal years a clause's own numbers match, abstaining (never
  over-rejecting) on negation, hedging, ambiguous multi-direction clauses,
  or figures with no year label to compare against.

Bug caught by our own adversarial verification, on the fix above, before it
shipped: the sign-tracking fix itself had a gap — when the model's own
claim omitted the "$" character (ordinary phrasing, "generated 9.45B"
instead of "generated $9.45B"), the character immediately before the
matched digits inside a figure like `"-$9.45B"` became "$" rather than "-",
and the regex looking for a preceding minus sign didn't tolerate that "$"
in between. Both directions of the original bug reopened as a result: a
false unsigned claim against a negative-only figure wrongly grounded, and a
true signed claim without "$" wrongly rejected with a backwards error
message. Caught by a read-only adversarial verifier deliberately trying to
break the fix with inputs the implementer hadn't tried, fixed by extending
the sign-detection regex to tolerate an optional "$" between the minus and
the matched number, with regression tests pinning both directions against
the exact adversarial inputs that found it.

Also built the first-class eval layer this session owned: `evals/datasets/`
(a gold tie-out set independently re-derived from raw companyfacts JSON — not
by calling the app's own fact-selection code, which would just be checking
it against itself — plus ~35 hand-labeled retrieval queries across AAPL and
a freshly fetched MSFT 10-K corpus), `evals/tieout_eval.py` (binary
cell-level accuracy, 430/430 cells correct across both tickers against the
verifier's own $1 tolerance), `evals/retrieval_eval.py` (precision/recall@k,
BM25-only by default so it can't hang on an unconfigured embedder download),
`evals/judge.py` (LLM-as-judge on real narrated commentary, judge model
required by an in-code check to differ from the generator), and
`evals/scorecard.py` (runs all three, writes `evals/scorecard.json`, renders
the result into README.md between two sentinel comments so re-running is
idempotent). 271 tests, lint/type green (mypy now also covers `evals/`).
