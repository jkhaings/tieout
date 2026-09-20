"""Retrieval eval: precision@k / recall@k plus fail-closed refusal behavior.

Scores `app.rag`'s hybrid retriever against the hand-labeled (query ->
relevant chunk ids) pairs in `evals/datasets/retrieval_queries.json`. Three
independent measurements, never blended into one number:

1. `production_labels` -- precision@k/recall@k for the bare line-item-label
   queries production actually issues (`app/agent/graph.py` calling
   `retriever.retrieve(item.label)`).
2. `analyst_queries` -- precision@k/recall@k for hand-written natural-
   language questions, measuring a broader capability the app does not
   literally exercise today.
3. `irrelevant_queries` -- refusal rate: the fraction of queries with no
   answer in either corpus that correctly retrieved zero chunks
   (CLAUDE.md rule 4, fail closed), which is a different property than
   ranking quality and is reported separately.

Before scoring, every ticker's corpus is rebuilt from the exact
`source_file` pinned in the dataset's `corpus_fingerprint` and checked
against the pinned `sha256` and `chunk_count`. A mismatch raises
`CorpusFingerprintMismatchError` immediately rather than silently scoring
against a different corpus than the one the labels were written against
(see `_build_and_verify_corpus`).

Backend default is BM25-only: `HybridIndex.build(..., embedder=None)` and no
reranker. `SentenceTransformerEmbedder`/`CrossEncoderReranker` (the
optional hybrid path, `run(..., use_embeddings=True)`) lazily download
model weights from Hugging Face Hub with no timeout of their own -- see
`app.settings.AppSettings.enable_local_embeddings`'s docstring -- so that
path is opt-in and is never exercised by this module's own `__main__` block
or by the default `run()` call.

Ticker coverage is read from the dataset itself (`corpus_fingerprint`'s
keys), never assumed: this module runs correctly whether the dataset ships
one ticker or several, and any query naming a ticker absent from
`corpus_fingerprint` is skipped with a recorded reason rather than raising.

Fully offline. No Anthropic calls; BM25 needs no network or model
downloads. Local-only per CLAUDE.md rule 7 (never imported by `tests/` or
run in CI) -- intended to be called by `evals/scorecard.py` and, standalone,
via `uv run python -m evals.retrieval_eval`.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.rag import (
    CrossEncoderReranker,
    HybridIndex,
    RagSettings,
    Retriever,
    SentenceTransformerEmbedder,
    is_chromadb_available,
    is_ml_available,
    is_sentence_transformers_available,
    parse_filing,
)
from app.schemas import Chunk

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "retrieval_queries.json"

# `Chunk.source_url` is never scored by this eval (only `chunk_id` is
# compared against gold ids); this placeholder mirrors the one
# `evals/fetch_corpus.py` uses when re-parsing a committed corpus file.
_SOURCE_URL_PLACEHOLDER = "https://www.sec.gov/"

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


class CorpusFingerprintMismatchError(RuntimeError):
    """A rebuilt corpus's sha256 or chunk_count disagrees with its pinned fingerprint.

    Raised instead of silently scoring: the hand-written relevant-chunk-id
    labels in `retrieval_queries.json` name specific chunk ids produced by a
    specific source file parsed with specific `RagSettings`. If the source
    file, `app.rag.ingest.parse_filing`'s chunking behavior, or the pinned
    settings have drifted since the labels were written, chunk ids silently
    stop lining up with what a human actually read -- which would make any
    precision/recall number that came out the other end meaningless without
    saying so.
    """


@dataclass(frozen=True)
class _CorpusFingerprint:
    """One ticker's pinned corpus identity, from the dataset's `corpus_fingerprint`."""

    ticker: str
    source_file: str
    sha256: str
    chunk_count: int
    chunk_target_tokens: int
    chunk_overlap_tokens: int


def _load_dataset() -> dict[str, Any]:
    """Load and return the parsed contents of `evals/datasets/retrieval_queries.json`."""
    with DATASET_PATH.open() as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def _load_fingerprints(raw: dict[str, Any]) -> dict[str, _CorpusFingerprint]:
    """Parse the dataset's `corpus_fingerprint` mapping into `_CorpusFingerprint` objects.

    Args:
        raw: The dataset's `corpus_fingerprint` value: a mapping of lowercase
            ticker key (e.g. `"aapl"`) to fingerprint fields.

    Returns:
        The same lowercase ticker keys, mapped to parsed `_CorpusFingerprint`
        objects.
    """
    fingerprints: dict[str, _CorpusFingerprint] = {}
    for ticker_key, entry in raw.items():
        rag_settings = entry["rag_settings"]
        fingerprints[ticker_key] = _CorpusFingerprint(
            ticker=ticker_key.upper(),
            source_file=entry["source_file"],
            sha256=entry["sha256"],
            chunk_count=entry["chunk_count"],
            chunk_target_tokens=rag_settings["chunk_target_tokens"],
            chunk_overlap_tokens=rag_settings["chunk_overlap_tokens"],
        )
    return fingerprints


def _build_and_verify_corpus(fingerprint: _CorpusFingerprint) -> list[Chunk]:
    """Rebuild one ticker's corpus and assert it matches its pinned fingerprint.

    Reads `fingerprint.source_file` (relative to the repo root), hashes its
    raw bytes, and compares against `fingerprint.sha256` *before* parsing --
    catching "wrong file entirely" as cheaply as possible. Only then parses
    it with `app.rag.parse_filing` at the pinned `chunk_target_tokens`/
    `chunk_overlap_tokens` and compares the resulting chunk count against
    `fingerprint.chunk_count`, catching "same file, but ingestion behavior
    has drifted since the labels were written."

    Args:
        fingerprint: The pinned corpus identity to rebuild and verify.

    Returns:
        The rebuilt list of chunks, in `parse_filing`'s deterministic order.

    Raises:
        CorpusFingerprintMismatchError: If the source file is missing, its
            sha256 disagrees with the pinned value, or the rebuilt chunk
            count disagrees with the pinned value.
    """
    path = REPO_ROOT / fingerprint.source_file
    if not path.is_file():
        raise CorpusFingerprintMismatchError(
            f"{fingerprint.ticker}: corpus_fingerprint.source_file "
            f"{fingerprint.source_file!r} does not exist at {path}."
        )

    raw_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha256 != fingerprint.sha256:
        raise CorpusFingerprintMismatchError(
            f"{fingerprint.ticker}: sha256 mismatch for {fingerprint.source_file} -- "
            f"pinned={fingerprint.sha256!r} actual={actual_sha256!r}. The retrieval labels "
            "in retrieval_queries.json were hand-written against a specific corpus "
            "snapshot; refusing to score against a different one."
        )

    settings = RagSettings(
        chunk_target_tokens=fingerprint.chunk_target_tokens,
        chunk_overlap_tokens=fingerprint.chunk_overlap_tokens,
    )
    html = raw_bytes.decode("utf-8")
    chunks = parse_filing(html, source_url=_SOURCE_URL_PLACEHOLDER, settings=settings)
    if len(chunks) != fingerprint.chunk_count:
        raise CorpusFingerprintMismatchError(
            f"{fingerprint.ticker}: chunk_count mismatch for {fingerprint.source_file} -- "
            f"pinned={fingerprint.chunk_count} rebuilt={len(chunks)} "
            f"(chunk_target_tokens={fingerprint.chunk_target_tokens}, "
            f"chunk_overlap_tokens={fingerprint.chunk_overlap_tokens}). "
            "app.rag.ingest.parse_filing's chunking behavior has drifted since these "
            "labels were written against specific chunk ids; refusing to score against "
            "a chunk set the labels no longer describe."
        )
    return chunks


def _build_retrievers(
    fingerprints: dict[str, _CorpusFingerprint],
    ks: Sequence[int],
    *,
    use_embeddings: bool,
) -> tuple[dict[str, Retriever], dict[str, Retriever], dict[str, Any]]:
    """Verify every ticker's corpus and build its ranking and refusal retrievers.

    One `HybridIndex` is built per ticker (BM25-only unless `use_embeddings`
    is `True`) and wrapped in two `Retriever`s that share it: a "ranking"
    retriever with `final_top_k=max(ks)` (large enough that truncation never
    hides a true positive at the largest scored `k`), and a "refusal"
    retriever built from `RagSettings()` at real, unmodified production
    defaults (thresholds included), used only to measure fail-closed
    behavior on queries with no real answer.

    Args:
        fingerprints: Every ticker's pinned corpus identity, keyed by
            lowercase ticker.
        ks: The `k` values that will be scored; only `max(ks)` is used here,
            to size the ranking retriever's pool.
        use_embeddings: If `True`, build a real `SentenceTransformerEmbedder`
            + `CrossEncoderReranker` hybrid index instead of BM25-only. This
            downloads model weights from Hugging Face Hub with no timeout of
            their own on first use -- see this module's docstring. Never set
            `True` for an unattended or CI run.

    Returns:
        A 3-tuple of: ranking retrievers by lowercase ticker, refusal
        retrievers by lowercase ticker, and a per-ticker fingerprint
        verification report (for inclusion in `run`'s return value).
    """
    embedder = SentenceTransformerEmbedder() if use_embeddings else None
    reranker = CrossEncoderReranker() if use_embeddings else None
    max_k = max(ks)

    ranking_retrievers: dict[str, Retriever] = {}
    refusal_retrievers: dict[str, Retriever] = {}
    fingerprint_report: dict[str, Any] = {}

    for ticker_key, fingerprint in fingerprints.items():
        chunks = _build_and_verify_corpus(fingerprint)
        index = HybridIndex.build(chunks, embedder=embedder, persist_dir=None)
        ranking_retrievers[ticker_key] = Retriever(index, reranker, RagSettings(final_top_k=max_k))
        refusal_retrievers[ticker_key] = Retriever(index, reranker, RagSettings())
        fingerprint_report[ticker_key] = {
            "ticker": fingerprint.ticker,
            "source_file": fingerprint.source_file,
            "sha256": fingerprint.sha256,
            "chunk_count": len(chunks),
            "verified": True,
        }

    return ranking_retrievers, refusal_retrievers, fingerprint_report


def _score_slice(
    queries: list[dict[str, Any]],
    retrievers: dict[str, Retriever],
    ks: Sequence[int],
    *,
    slice_name: str,
    note: str,
) -> dict[str, Any]:
    """Score one labeled query slice's precision@k / recall@k, per query and aggregated.

    A query whose `relevant_chunk_ids` is empty (a documented, genuine
    coverage gap in this dataset -- see `retrieval_queries.json`'s own
    notes -- not a labeling shortcut) is recorded but excluded from the
    aggregate: recall is undefined with zero relevant items, and including
    a forced-zero precision would blame ranking quality for a corpus
    coverage gap. A query naming a ticker with no built corpus is likewise
    recorded but excluded, with a distinct reason. Both exclusions are
    counted, never silently dropped.

    Args:
        queries: This slice's labeled query entries (each with `ticker`,
            `query`, `relevant_chunk_ids`, and optionally `note`).
        retrievers: Ranking retrievers by lowercase ticker.
        ks: `k` values to score precision/recall at.
        slice_name: Human-readable name of this slice, echoed into the
            result for clarity when slices are compared side by side.
        note: A human-readable caveat about what this slice does and does
            not measure, echoed into the result.

    Returns:
        A dict with `slice`, `note`, query counts, `precision_at_k` /
        `recall_at_k` (mean over scored queries, keyed by `str(k)`, `None`
        if no query in this slice had gold), and a `per_query` breakdown.
    """
    per_query: list[dict[str, Any]] = []
    num_skipped = 0

    for entry in queries:
        ticker = entry["ticker"]
        retriever = retrievers.get(ticker.lower())
        if retriever is None:
            num_skipped += 1
            per_query.append(
                {
                    "ticker": ticker,
                    "query": entry["query"],
                    "skipped": True,
                    "reason": f"no corpus was built for ticker {ticker!r} "
                    "(absent from corpus_fingerprint)",
                }
            )
            continue

        relevant_ids: set[str] = set(entry.get("relevant_chunk_ids") or [])
        retrieved = retriever.retrieve(entry["query"])
        retrieved_ids = [candidate.chunk.chunk_id for candidate in retrieved]

        record: dict[str, Any] = {
            "ticker": ticker,
            "query": entry["query"],
            "skipped": False,
            "has_gold": bool(relevant_ids),
            "relevant_chunk_ids": sorted(relevant_ids),
            "retrieved_chunk_ids": retrieved_ids,
        }
        if entry.get("note"):
            record["note"] = entry["note"]

        if relevant_ids:
            precision_at_k: dict[str, float] = {}
            recall_at_k: dict[str, float] = {}
            for k in ks:
                hits = sum(1 for chunk_id in retrieved_ids[:k] if chunk_id in relevant_ids)
                precision_at_k[str(k)] = hits / k
                recall_at_k[str(k)] = hits / len(relevant_ids)
            record["precision_at_k"] = precision_at_k
            record["recall_at_k"] = recall_at_k

        per_query.append(record)

    scored = [record for record in per_query if not record["skipped"] and record["has_gold"]]
    without_gold = [
        record for record in per_query if not record["skipped"] and not record["has_gold"]
    ]

    precision_at_k_mean: dict[str, float | None] = {}
    recall_at_k_mean: dict[str, float | None] = {}
    for k in ks:
        precisions = [record["precision_at_k"][str(k)] for record in scored]
        recalls = [record["recall_at_k"][str(k)] for record in scored]
        precision_at_k_mean[str(k)] = statistics.mean(precisions) if precisions else None
        recall_at_k_mean[str(k)] = statistics.mean(recalls) if recalls else None

    return {
        "slice": slice_name,
        "note": note,
        "num_queries": len(queries),
        "num_scored": len(scored),
        "num_without_gold": len(without_gold),
        "num_skipped": num_skipped,
        "precision_at_k": precision_at_k_mean,
        "recall_at_k": recall_at_k_mean,
        "per_query": per_query,
    }


def _score_refusal(
    queries: list[dict[str, Any]],
    retrievers: dict[str, Retriever],
    *,
    note: str,
) -> dict[str, Any]:
    """Score refusal behavior: the fraction of `queries` that correctly returned zero chunks.

    This measures CLAUDE.md rule 4's fail-closed behavior, not ranking
    quality: every query in `queries` is expected to have no real answer in
    either corpus, so the only correct outcome is an empty retrieval result.

    Args:
        queries: Irrelevant-query entries (each with `ticker` and `query`).
        retrievers: Refusal retrievers (production-default `RagSettings()`)
            by lowercase ticker.
        note: A human-readable caveat about what this measures, echoed into
            the result.

    Returns:
        A dict with query counts, `refusal_rate` (`None` if no query could
        be scored), and a `per_query` breakdown.
    """
    per_query: list[dict[str, Any]] = []
    num_skipped = 0
    num_correct = 0
    num_scored = 0

    for entry in queries:
        ticker = entry["ticker"]
        retriever = retrievers.get(ticker.lower())
        if retriever is None:
            num_skipped += 1
            per_query.append(
                {
                    "ticker": ticker,
                    "query": entry["query"],
                    "skipped": True,
                    "reason": f"no corpus was built for ticker {ticker!r} "
                    "(absent from corpus_fingerprint)",
                }
            )
            continue

        retrieved = retriever.retrieve(entry["query"])
        correctly_refused = len(retrieved) == 0
        num_scored += 1
        if correctly_refused:
            num_correct += 1
        per_query.append(
            {
                "ticker": ticker,
                "query": entry["query"],
                "skipped": False,
                "correctly_refused": correctly_refused,
                "retrieved_chunk_ids": [candidate.chunk.chunk_id for candidate in retrieved],
            }
        )

    return {
        "note": note,
        "num_queries": len(queries),
        "num_scored": num_scored,
        "num_skipped": num_skipped,
        "num_correct_refusals": num_correct,
        "refusal_rate": (num_correct / num_scored) if num_scored else None,
        "per_query": per_query,
    }


def run(ks: Sequence[int] = DEFAULT_KS, *, use_embeddings: bool = False) -> dict[str, Any]:
    """Score precision@k/recall@k per query slice, plus refusal behavior on irrelevant queries.

    For each ticker present in `evals/datasets/retrieval_queries.json`'s
    `corpus_fingerprint`, rebuilds its corpus with `app.rag.parse_filing` on
    the exact `source_file` named there, at the exact
    `chunk_target_tokens`/`chunk_overlap_tokens` recorded there, and asserts
    the rebuilt `chunk_count` and `sha256` match what is pinned -- raising
    `CorpusFingerprintMismatchError` immediately on any mismatch rather than
    silently scoring against a different corpus than the one the labels
    were written against.

    Builds, per ticker, a `HybridIndex` (BM25-only unless `use_embeddings`)
    and two `Retriever`s over it: a ranking retriever with
    `final_top_k=max(ks)` used for `"production_labels"` and
    `"analyst_queries"`, and a refusal retriever built from `RagSettings()`
    at real, unmodified production defaults used for `"irrelevant_queries"`.

    `"production_labels"` mirrors the app's real query pattern (bare
    `LineItem.label` strings); `"analyst_queries"` measures a broader,
    hand-written capability the app does not literally exercise today. They
    are always scored and reported separately, never blended into one
    number. A query whose ticker is absent from `corpus_fingerprint` --
    e.g. if the dataset ever ships AAPL-only -- is skipped with a recorded
    reason rather than assumed present or treated as an error.

    Args:
        ks: The `k` values to compute precision@k/recall@k at. Deduplicated
            and sorted ascending; `max(ks)` sizes the ranking retriever's
            pool so truncation never hides a true positive at the largest
            scored `k`.
        use_embeddings: If `True`, additionally build a real
            `SentenceTransformerEmbedder` + `CrossEncoderReranker` hybrid
            index instead of BM25-only. Both lazily download model weights
            from Hugging Face Hub with **no timeout of their own** on first
            use, so this can hang indefinitely in an environment without
            the weights already cached (e.g. baked into a container image).
            Defaults to `False`; never set `True` in an unattended or CI
            context.

    Returns:
        A dict with keys `ks`, `backend` (embedder/reranker used, plus
        `app.rag`'s ml-availability flags), `corpus_fingerprint_verified`
        (per-ticker verification report), `production_labels`,
        `analyst_queries` (each shaped as `_score_slice`'s return value),
        and `irrelevant_queries_refusal` (`_score_refusal`'s return value).
    """
    if not ks:
        raise ValueError("ks must be non-empty")
    ks_sorted = sorted(set(ks))

    dataset = _load_dataset()
    fingerprints = _load_fingerprints(dataset["corpus_fingerprint"])
    ranking_retrievers, refusal_retrievers, fingerprint_report = _build_retrievers(
        fingerprints, ks_sorted, use_embeddings=use_embeddings
    )

    production_labels = _score_slice(
        dataset.get("production_labels", []),
        ranking_retrievers,
        ks_sorted,
        slice_name="production_labels",
        note=(
            "Mirrors the real production query pattern: app/agent/graph.py calling "
            "retriever.retrieve(item.label) with the bare LineItem.label string, for "
            "each of app.agent.formatting.COMMENTARY_LINE_ITEMS. Several entries have "
            "an empty relevant_chunk_ids by design (the figure lives outside this "
            "Item 1A/Item 7 corpus, e.g. in Item 8's financial statements) -- see each "
            "such entry's own 'note' and this slice's 'num_without_gold'."
        ),
    )
    analyst_queries = _score_slice(
        dataset.get("analyst_queries", []),
        ranking_retrievers,
        ks_sorted,
        slice_name="analyst_queries",
        note=(
            "Broader, hand-written natural-language analyst questions measuring a "
            "retrieval capability the app does not literally exercise today -- "
            "production only ever queries the bare COMMENTARY_LINE_ITEMS labels "
            "(see 'production_labels')."
        ),
    )
    irrelevant_queries_refusal = _score_refusal(
        dataset.get("irrelevant_queries", []),
        refusal_retrievers,
        note=(
            "Fraction of queries with no answer in either corpus that correctly "
            "returned zero chunks: fail-closed refusal behavior (CLAUDE.md rule 4), "
            "not ranking quality. Uses RagSettings() at real, unmodified production "
            "defaults (thresholds included), not the ranking pass's widened final_top_k."
        ),
    )

    backend = {
        "use_embeddings": use_embeddings,
        "embedder": "SentenceTransformerEmbedder" if use_embeddings else None,
        "reranker": "CrossEncoderReranker" if use_embeddings else None,
        "is_ml_available": is_ml_available(),
        "is_chromadb_available": is_chromadb_available(),
        "is_sentence_transformers_available": is_sentence_transformers_available(),
    }

    return {
        "ks": ks_sorted,
        "backend": backend,
        "corpus_fingerprint_verified": fingerprint_report,
        "production_labels": production_labels,
        "analyst_queries": analyst_queries,
        "irrelevant_queries_refusal": irrelevant_queries_refusal,
    }


def main() -> None:
    """Run the retrieval eval at default settings and print its result as indented JSON."""
    result = run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
