"""LangGraph pipeline: fetch -> build -> verify -> retrieve -> narrate -> generate.

Design, verified against the installed langgraph (1.2.x) rather than assumed:

* Nodes are plain **synchronous** functions -- every call inside them
  (`EdgarClient`, `build_statements`, retrieval, `narrate_line_item`,
  `openpyxl`) is blocking. LangGraph auto-wraps a sync node in
  `run_in_executor` when the graph is driven with `.astream`/`.ainvoke`, so
  the event loop is never blocked; making the nodes `async def` themselves
  would buy nothing here (there is nothing to `await`) and would forfeit
  that automatic off-loop dispatch.
* Progress is reported by taking a parameter literally named `writer`
  (LangGraph's dependency injection is name-based, not type-based -- a
  differently-named parameter is silently never populated) and calling it
  with a `RunEvent`. The API drives the graph with
  `astream(..., stream_mode=["custom", "values"])`: `"custom"` yields every
  `writer(...)` call live, mid-node, the moment it happens (verified: a node
  that sleeps between two `writer()` calls delivers both events with the
  real gap between them, not batched at node completion); `"values"` yields
  the accumulated state after each node, whose last value is the final
  state -- one execution, both the live stream and the end result.
* Per-run dependencies that are not pipeline *data* (the shared
  `EdgarClient`, the narrator client, the embedder/reranker, tunables) are
  injected the same name-based way via a parameter literally named
  `runtime`, typed `Runtime[PipelineContext]`, and passed at invoke time as
  `context=...` -- never through graph state. This keeps `PipelineState`
  plain and JSON-shaped (every field is one of `app.schemas`'s frozen
  models, a dataclass, or a primitive) and keeps the `EdgarClient` a
  single, process-wide instance whose rate limiter is never bypassed by two
  concurrent runs each thinking they own the SEC request budget (see
  `app/api` for why it is constructed once, at process startup).
* Fatal nodes (fetch, build, generate, and verify's own exception path --
  never a *failing* tie-out, which is data, not an error) raise
  `FatalPipelineError` and stop there: no conditional edges are needed
  because an exception raised inside a node propagates straight out of
  `astream`/`ainvoke` (verified: a later node in the same graph never runs).
  Degradable nodes (retrieve, narrate) catch everything themselves and
  always return a safe, empty-shaped result instead of raising, so the
  pipeline reaches `generate` regardless of how retrieval or narration went
  (CLAUDE.md rule 4: weak/absent retrieval and unavailable narration are
  normal outcomes, not errors).
* The compiled graph is stateless (all per-run dependencies come through
  `context`/state, never a closure) and is built once, at import time, and
  reused for every run.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, TypedDict, cast

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import StreamWriter
from pydantic import ValidationError as PydanticValidationError

from app.agent.formatting import COMMENTARY_LINE_ITEMS, build_figures
from app.agent.workbook import write_commentary_sheet
from app.edgar.client import EdgarClient, EdgarError, validate_ticker
from app.model.builder import build_statements
from app.model.excel import build_workbook
from app.model.verifier import Reconciliation, reconcile
from app.model.verifier import verify as compute_tieout
from app.obs import span
from app.rag import Embedder, HybridIndex, LLMClient, Reranker, Retriever, parse_filing
from app.rag import narrate_line_item as run_narrate_line_item
from app.schemas import Chunk, Commentary, RunEvent, StatementSet, TieoutReport
from app.settings import RagSettings

logger = logging.getLogger(__name__)


class FatalPipelineError(RuntimeError):
    """Raised by a node whose failure must abort the run.

    Carries only a short, generic message safe to show a client
    (SECURITY.md item 10) -- never an exception's raw text, a stack trace,
    or a config value. The API's terminal `error` event uses this message
    verbatim.
    """


@dataclass
class PipelineContext:
    """Per-run dependencies threaded through LangGraph's `Runtime`, never through state.

    None of these are pipeline *data* (see the module docstring): `edgar_client`
    is a shared, stateful resource that must be exactly one instance per
    process (its rate limiter is per-instance, `app/edgar/client.py`);
    `narrator_client` is `None` whenever no Anthropic key is configured, in
    which case narrate degrades without ever calling it; `embedder`/
    `reranker` are `None` in a BM25-only environment (the `ml` extra absent
    or not requested), which `app.rag.HybridIndex`/`Retriever` already
    handle as a fully supported degrade path.
    """

    run_id: str
    edgar_client: EdgarClient
    narrator_client: LLMClient | None
    embedder: Embedder | None
    reranker: Reranker | None
    rag_settings: RagSettings
    runs_dir: Path


class PipelineState(TypedDict, total=False):
    """Data that flows and accumulates across pipeline nodes.

    Deliberately holds only plain, JSON-shaped data -- `app.schemas` models,
    one local dataclass (`Reconciliation`), and primitives -- never a client
    object or a callable (those live in `PipelineContext`). Keeping state
    this way is what lets `stream_mode="values"` hand the API a plain,
    inspectable snapshot after every node.
    """

    ticker: str
    cik: str
    company_facts: dict[str, Any]
    submissions: dict[str, Any]
    accession_number: str
    primary_document: str
    filing_date: str
    filing_html: str | None
    statements: StatementSet
    tieout: TieoutReport
    reconciliations: list[Reconciliation]
    chunks_by_item: dict[str, list[Chunk]]
    # Why retrieval produced nothing, when it produced nothing -- carried into
    # each refused line item's reason so the workbook can say which upstream
    # step failed, not merely that commentary is absent.
    retrieval_note: str
    commentary: list[Commentary]
    # line item key -> why it has no commentary. Deterministic facts about the
    # run, never model output; rendered by `app.agent.workbook`.
    commentary_refusals: dict[str, str]
    narrate_ok: bool
    artifact_path: str


# Consecutive narrate failures (real exceptions from the LLM client, never a
# normal "no grounded commentary") after which the node stops calling
# Anthropic for the remaining line items. Without this, a bad API key or an
# outage costs one full API-timeout per remaining line item -- against a
# ~40-second per-run target, that turns one bad key into minutes of
# guaranteed timeouts for no additional information.
_NARRATE_CIRCUIT_BREAKER_THRESHOLD = 3


def _emit(writer: StreamWriter, run_id: str, step: str, status: str, detail: str = "") -> None:
    """Construct and emit one `RunEvent` through the graph's custom stream."""
    writer(RunEvent(run_id=run_id, step=step, status=status, detail=detail, ts=time.time()))


def fetch(
    state: PipelineState, runtime: Runtime[PipelineContext], writer: StreamWriter
) -> dict[str, Any]:
    """Resolve the ticker, fetch company facts/submissions, and the filing text.

    Fatal if the CIK can't be resolved or `company_facts`/`submissions`
    can't be fetched -- nothing downstream is possible without them.
    Fetching the filing's HTML text is best-effort *within this same node*:
    the `fetch` step in `RunEvent.step`'s vocabulary covers everything
    `app/edgar` fetches (see ARCHITECTURE.md's pipeline diagram, where fetch
    feeds both build and ingest/retrieve), so a filing-text failure is
    folded in here rather than invented as a new step name. If it fails,
    the run continues with `filing_html=None`, which `retrieve` treats
    exactly like an empty filing: the workbook itself never needs filing
    text, only commentary does.
    """
    run_id = runtime.context.run_id
    client = runtime.context.edgar_client
    ticker = state["ticker"]
    _emit(writer, run_id, "fetch", "started", detail=ticker)
    try:
        ticker = validate_ticker(ticker)
        cik = client.resolve_cik(ticker)
        company_facts = client.company_facts(cik)
        submissions = client.submissions(cik)
        latest = client.latest_10k(cik)
    except (EdgarError, httpx.HTTPError) as exc:
        logger.warning("fetch failed for %r: %s", ticker, exc, extra={"run_id": run_id})
        message = "could not fetch filing data from SEC EDGAR for this ticker"
        _emit(writer, run_id, "fetch", "failed", detail=message)
        raise FatalPipelineError(message) from exc

    filing_html: str | None
    try:
        filing_html = client.filing_html(
            cik, latest["accession_number"], latest["primary_document"]
        )
    except (EdgarError, httpx.HTTPError) as exc:
        logger.warning("filing text fetch failed for %r: %s", ticker, exc, extra={"run_id": run_id})
        filing_html = None

    detail = f"{ticker} (CIK {cik}), filing {latest['accession_number']}"
    if filing_html is None:
        detail += " -- filing text unavailable, commentary will be skipped"
    _emit(writer, run_id, "fetch", "ok", detail=detail)
    return {
        "ticker": ticker,
        "cik": cik,
        "company_facts": company_facts,
        "submissions": submissions,
        "accession_number": latest["accession_number"],
        "primary_document": latest["primary_document"],
        "filing_date": latest["filing_date"],
        "filing_html": filing_html,
    }


def build(
    state: PipelineState, runtime: Runtime[PipelineContext], writer: StreamWriter
) -> dict[str, Any]:
    """Build the `StatementSet` for the last five fiscal years.

    Fatal if the builder raises, or if it resolves zero fiscal years / zero
    line items: `app.model.verifier.verify`'s "insufficient data" guard is
    per fiscal year, so a `StatementSet` with *no* fiscal years at all would
    otherwise reach `verify()` as an empty checks list, and
    `TieoutReport.passed` -- `all([])` -- would read as a clean pass on a
    company with no usable data. Guarding it here, at the orchestration
    boundary, keeps that fix out of `app/model` (frozen for this session).
    """
    run_id = runtime.context.run_id
    _emit(writer, run_id, "build", "started")
    try:
        statements = build_statements(state["company_facts"], state["submissions"], state["ticker"])
        if not statements.fiscal_years or not statements.items:
            raise ValueError("no usable financial data resolved for this ticker")
    except (ValueError, KeyError, PydanticValidationError) as exc:
        message = "could not build financial statements from the fetched filing data"
        _emit(writer, run_id, "build", "failed", detail=message)
        raise FatalPipelineError(message) from exc

    detail = f"{len(statements.fiscal_years)} fiscal years, {len(statements.items)} line items"
    _emit(writer, run_id, "build", "ok", detail=detail)
    return {"statements": statements}


def verify_node(
    state: PipelineState, runtime: Runtime[PipelineContext], writer: StreamWriter
) -> dict[str, Any]:
    """Run tie-out checks. Fatal only if verification itself cannot run.

    A *failing* tie-out (`tieout.passed is False`) is not fatal and never
    aborts the run: CLAUDE.md rule 4 requires it be "flagged in the
    workbook and UI, never silently shipped" -- flagged, not withheld. The
    emitted status is `"failed"` in that case purely so the live UI
    visibly flags the step; the pipeline still proceeds to `generate`,
    which always writes every check (passed or not) to the Tie-out tab.
    """
    run_id = runtime.context.run_id
    statements = state["statements"]
    _emit(writer, run_id, "verify", "started")
    try:
        tieout = compute_tieout(statements)
        reconciliations = reconcile(statements)
    except Exception as exc:  # defensive: both are documented to never raise
        message = "tie-out verification could not run"
        _emit(writer, run_id, "verify", "failed", detail=message)
        raise FatalPipelineError(message) from exc

    n_passed = sum(1 for check in tieout.checks if check.passed)
    n_total = len(tieout.checks)
    detail = f"{n_passed}/{n_total} tie-out checks passed"
    if not tieout.passed:
        detail += " -- FAILED; flagged in the Tie-out tab, workbook still ships"
    _emit(writer, run_id, "verify", "ok" if tieout.passed else "failed", detail=detail)
    return {"tieout": tieout, "reconciliations": reconciliations}


def _source_url(cik: str, accession_number: str, primary_document: str) -> str:
    """Build the filing's canonical URL from already-validated, constant-shaped parts."""
    accession_nodash = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{primary_document}"
    )


def retrieve(
    state: PipelineState, runtime: Runtime[PipelineContext], writer: StreamWriter
) -> dict[str, Any]:
    """Parse the filing, build a per-run hybrid index, and retrieve grounding chunks.

    Always degrades rather than aborts: an empty or unavailable filing, a
    parse that finds neither target section, or any unexpected exception
    all end the same way -- `chunks_by_item` stays empty for the affected
    line items, and narrate treats that exactly like "no grounded
    commentary available" (CLAUDE.md rule 4). The index is always built
    with `persist_dir=None` (ephemeral, in-memory): each run's chunks are
    specific to that run's filing, so there is nothing worth persisting
    across runs, and skipping it avoids leftover on-disk vector state.
    """
    run_id = runtime.context.run_id
    ctx = runtime.context
    _emit(writer, run_id, "retrieve", "started")

    html = state.get("filing_html")
    if not html:
        _emit(
            writer,
            run_id,
            "retrieve",
            "ok",
            detail="no filing text available; commentary will be skipped",
        )
        logger.warning("no filing text available; commentary skipped", extra={"run_id": run_id})
        return {"chunks_by_item": {}, "retrieval_note": "the filing text could not be fetched"}

    chunks_by_item: dict[str, list[Chunk]] = {}
    note = ""
    try:
        source_url = _source_url(state["cik"], state["accession_number"], state["primary_document"])
        all_chunks = parse_filing(html, source_url=source_url, settings=ctx.rag_settings)
        if not all_chunks:
            # Neither Item 1A nor Item 7 was found. Nothing raises here and
            # nothing logged before, so this failure mode was invisible in
            # production -- the empty Commentary tab was its only symptom.
            logger.warning(
                "parse_filing found no sections; commentary will be skipped",
                extra={"run_id": run_id},
            )
            note = "the filing parsed to zero sections (Item 1A / Item 7 not found)"
        if all_chunks:
            index = HybridIndex.build(all_chunks, embedder=ctx.embedder, persist_dir=None)
            retriever = Retriever(index, ctx.reranker, ctx.rag_settings)
            items_by_key = {item.key: item for item in state["statements"].items}
            for key in COMMENTARY_LINE_ITEMS:
                item = items_by_key.get(key)
                if item is None:
                    continue
                retrieved = retriever.retrieve(item.label)
                if retrieved:
                    chunks_by_item[key] = [candidate.chunk for candidate in retrieved]
    except Exception as exc:  # noqa: BLE001 - deliberately broad; retrieval always degrades
        logger.warning("retrieve failed: %s", exc, extra={"run_id": run_id})
        _emit(
            writer,
            run_id,
            "retrieve",
            "failed",
            detail="retrieval failed; commentary will be skipped",
        )
        return {"chunks_by_item": {}, "retrieval_note": "retrieval failed"}

    detail = f"{len(chunks_by_item)}/{len(COMMENTARY_LINE_ITEMS)} line items have grounding chunks"
    _emit(writer, run_id, "retrieve", "ok", detail=detail)
    return {"chunks_by_item": chunks_by_item, "retrieval_note": note}


def narrate(
    state: PipelineState, runtime: Runtime[PipelineContext], writer: StreamWriter
) -> dict[str, Any]:
    """Generate grounded commentary per curated line item. Always degrades, never aborts.

    Skips entirely (every item `text=None`) when no Anthropic key is
    configured -- a deliberate, immediate decision rather than letting each
    call fail with a 401 one at a time. Otherwise calls
    `narrate_line_item` per item, catching any exception it lets escape
    (a live network/API error; schema/grounding failures are already
    handled inside it) and recording a refusal for that item instead. A
    small circuit breaker stops calling the API after
    `_NARRATE_CIRCUIT_BREAKER_THRESHOLD` consecutive real errors, so an
    outage or bad key costs a handful of timeouts, not one per line item.
    """
    run_id = runtime.context.run_id
    ctx = runtime.context
    _emit(writer, run_id, "narrate", "started")

    statements = state["statements"]
    items_by_key = {item.key: item for item in statements.items}
    chunks_by_item = state.get("chunks_by_item", {})
    commentary: list[Commentary] = []

    note = state.get("retrieval_note", "")
    refusals: dict[str, str] = {}

    if ctx.narrator_client is None:
        commentary = [
            Commentary(line_item_key=key, text=None, citations=[]) for key in COMMENTARY_LINE_ITEMS
        ]
        reason = "ANTHROPIC_API_KEY is not configured on this deployment"
        # Logged, not only streamed: an unkeyed production run was previously
        # silent in `docker logs`, leaving an empty Commentary tab as the only
        # evidence anything had gone wrong.
        logger.warning("narration skipped: %s", reason, extra={"run_id": run_id})
        _emit(
            writer,
            run_id,
            "narrate",
            "ok",
            detail="ANTHROPIC_API_KEY not configured; commentary skipped",
        )
        return {
            "commentary": commentary,
            "narrate_ok": False,
            "commentary_refusals": dict.fromkeys(COMMENTARY_LINE_ITEMS, reason),
        }

    consecutive_failures = 0
    circuit_open = False
    had_error = False
    for key in COMMENTARY_LINE_ITEMS:
        item = items_by_key.get(key)
        if item is None:
            # This filer does not report the concept at all (McDonald's files
            # no gross profit). Previously this `continue`d, dropping the item
            # from `commentary` entirely so the workbook had no row for it.
            commentary.append(Commentary(line_item_key=key, text=None, citations=[]))
            refusals[key] = "this filer does not report this line item"
            continue
        figures = build_figures(item, statements)
        chunks = chunks_by_item.get(key, [])
        if circuit_open or not figures or not chunks:
            commentary.append(Commentary(line_item_key=key, text=None, citations=[]))
            if circuit_open:
                refusals[key] = "narration stopped early after repeated API errors"
            elif not figures:
                refusals[key] = "this filer reports no value for this line item in any year"
            else:
                refusals[key] = (
                    f"no filing passage was retrieved for this line item ({note})"
                    if note
                    else "no filing passage was retrieved for this line item"
                )
            continue
        try:
            narrated = run_narrate_line_item(
                line_item_key=key,
                label=item.label,
                figures=figures,
                chunks=chunks,
                client=ctx.narrator_client,
                settings=ctx.rag_settings,
            )
            commentary.append(narrated)
            if narrated.text is None:
                refusals[key] = (
                    "the model's draft failed grounding or schema validation twice; refused"
                )
            consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001 - the client's own transport/API errors
            logger.warning("narrate failed for %r: %s", key, exc, extra={"run_id": run_id})
            had_error = True
            consecutive_failures += 1
            commentary.append(Commentary(line_item_key=key, text=None, citations=[]))
            refusals[key] = "the narration API call failed"
            if consecutive_failures >= _NARRATE_CIRCUIT_BREAKER_THRESHOLD:
                circuit_open = True

    n_narrated = sum(1 for item in commentary if item.text is not None)
    detail = f"{n_narrated}/{len(commentary)} line items narrated"
    if circuit_open:
        detail += " -- stopped early after repeated API errors"
    _emit(writer, run_id, "narrate", "failed" if had_error else "ok", detail=detail)
    return {
        "commentary": commentary,
        "narrate_ok": n_narrated > 0,
        "commentary_refusals": refusals,
    }


def generate(
    state: PipelineState, runtime: Runtime[PipelineContext], writer: StreamWriter
) -> dict[str, Any]:
    """Build the workbook, append the Commentary sheet, and save it under `runs_dir`.

    Fatal on any failure: a run that reaches this node has a verified (or
    loudly flagged) `StatementSet` and nothing else left to produce -- the
    workbook *is* the deliverable (SECURITY.md item 5: the save path is
    `runs_dir/<run_id>/model.xlsx`, `run_id` being server-generated, never
    derived from user input).
    """
    run_id = runtime.context.run_id
    _emit(writer, run_id, "generate", "started")
    try:
        statements = state["statements"]
        workbook = build_workbook(statements, state["tieout"], state["reconciliations"])
        chunks_by_item = state.get("chunks_by_item", {})
        chunks_by_id = {
            chunk.chunk_id: chunk for chunks in chunks_by_item.values() for chunk in chunks
        }
        labels_by_key = {item.key: item.label for item in statements.items}
        write_commentary_sheet(
            workbook,
            state.get("commentary", []),
            labels_by_key=labels_by_key,
            chunks_by_id=chunks_by_id,
            refusal_reasons=state.get("commentary_refusals", {}),
        )
        artifact_dir = runtime.context.runs_dir / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "model.xlsx"
        workbook.save(path)
    except Exception as exc:
        message = "could not generate the workbook"
        _emit(writer, run_id, "generate", "failed", detail=message)
        raise FatalPipelineError(message) from exc

    _emit(writer, run_id, "generate", "ok", detail=str(path))
    return {"artifact_path": str(path)}


_Node = Callable[[PipelineState, Runtime[PipelineContext], StreamWriter], dict[str, Any]]


def _traced(step: str, node: _Node) -> _Node:
    """Wrap `node` so its execution is enclosed in a Langfuse span named `step`.

    ARCHITECTURE.md: "Every step emits a `RunEvent` ... and a trace span
    (Langfuse)." The `RunEvent` side is `_emit`, called from inside each
    node; this is the trace-span side, added once here rather than
    threaded through every node body. `app.obs.span` is a no-op
    context manager when Langfuse isn't configured (the default), so this
    costs nothing in that case.

    Declares the exact same `state`/`runtime`/`writer` parameter names
    LangGraph's injection needs (verified: it is name-based, not
    type-based, and dispatches correctly to a node reached through a
    wrapper like this one) and simply forwards them to `node` inside the
    span, rather than wrapping/re-indenting each node's own body.
    """

    @wraps(node)
    def wrapper(
        state: PipelineState, runtime: Runtime[PipelineContext], writer: StreamWriter
    ) -> dict[str, Any]:
        with span(step, run_id=runtime.context.run_id, ticker=state.get("ticker")):
            return node(state, runtime, writer)

    return wrapper


def build_graph() -> CompiledStateGraph[
    PipelineState, PipelineContext, PipelineState, PipelineState
]:
    """Compile the linear fetch->build->verify->retrieve->narrate->generate graph.

    Plain, unconditional edges throughout: fatal nodes stop the run by
    raising `FatalPipelineError` (verified to propagate straight out of
    `astream`/`ainvoke`, halting execution before any later node runs), so
    no conditional routing is needed to skip downstream nodes on failure.
    """
    graph = StateGraph(PipelineState, context_schema=PipelineContext)
    # `add_node`'s overloads (installed langgraph 1.2.x) cover a node taking
    # `writer` or `runtime` individually but have no overload for a node
    # taking both by name at once, even though that combination is exactly
    # what LangGraph's own runtime dependency injection supports and
    # dispatches correctly (verified directly against this installed
    # version: a node with both `runtime: Runtime[T]` and `writer:
    # StreamWriter` params receives both correctly under `.astream`/
    # `.ainvoke`). The `cast` below is scoped to this stub gap, not a
    # blanket suppression -- every node function itself stays fully typed.
    add_node = cast("Callable[[str, object], None]", graph.add_node)
    add_node("fetch", _traced("fetch", fetch))
    add_node("build", _traced("build", build))
    add_node("verify", _traced("verify", verify_node))
    add_node("retrieve", _traced("retrieve", retrieve))
    add_node("narrate", _traced("narrate", narrate))
    add_node("generate", _traced("generate", generate))

    graph.add_edge(START, "fetch")
    graph.add_edge("fetch", "build")
    graph.add_edge("build", "verify")
    graph.add_edge("verify", "retrieve")
    graph.add_edge("retrieve", "narrate")
    graph.add_edge("narrate", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


# Stateless and side-effect-free once compiled (every per-run dependency
# arrives through `context`/state at invoke time), so one compiled graph
# safely serves every run in the process.
GRAPH = build_graph()
