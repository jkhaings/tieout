"""Hermetic tests for evals.judge: zero real Anthropic calls (CLAUDE.md rule 7).

Every real Anthropic call path in ``evals.judge`` is exercised here only
through a fake, in-process stand-in for ``anthropic.Anthropic`` (``_FakeAnthropic``
below) -- never a real client, never a real key, never network. These tests
confirm three things the task brief called out specifically:

1. The ``_JUDGE_MODEL != narration_model`` assertion actually raises when
   deliberately broken.
2. The JSON-schema/parsing/retry path around a fake client's canned
   responses works (both the happy path and the fail-closed path).
3. ``run()`` with ``anthropic_configured`` False returns the documented
   skip dict immediately, without raising and without constructing any
   real (or even fake) LLM client.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from app.schemas import Chunk, Citation, Commentary
from app.settings import RagSettings
from evals import judge as judge_module
from evals.judge import AnthropicJudge, JudgeVerdict, _parse_verdict, judge_samples, run

_SOURCE_URL = "https://www.sec.gov/Archives/edgar/data/0000320193/example.htm"

_CHUNK = Chunk(
    chunk_id="item7-0000",
    section="Item 7. Management's Discussion and Analysis",
    text="Item 7. Management's Discussion and Analysis. Revenue increased to $391.04B.",
    source_url=_SOURCE_URL,
)

_GOOD_COMMENTARY = Commentary(
    line_item_key="revenue",
    text="Revenue rose to $391.04B.",
    citations=[Citation(chunk_id=_CHUNK.chunk_id, quote="Revenue increased to $391.04B")],
)


class _FakeBlock:
    """Stand-in for one ``anthropic`` text content block."""

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResponse:
    """Stand-in for one ``anthropic.types.Message`` response."""

    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


@dataclass
class _FakeMessages:
    """Stand-in for ``anthropic.Anthropic().messages``: scripted canned responses."""

    responses: deque[str]
    calls: list[dict[str, Any]]

    def create(self, **kwargs: Any) -> _FakeResponse:
        """Record the call and return the next scripted response."""
        self.calls.append(kwargs)
        return _FakeResponse(self.responses.popleft())


class _FakeAnthropic:
    """Stand-in for ``anthropic.Anthropic``: never touches the network."""

    def __init__(self, responses: list[str]) -> None:
        self.messages = _FakeMessages(responses=deque(responses), calls=[])


def _verdict_json(*, grounded: bool, cited: bool, no_invented_numbers: bool, notes: str) -> str:
    return json.dumps(
        {
            "grounded": grounded,
            "cited": cited,
            "no_invented_numbers": no_invented_numbers,
            "notes": notes,
        }
    )


def test_judge_model_must_differ_from_generator_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """AnthropicJudge.__init__ raises ValueError when judge model == generator model."""
    monkeypatch.setattr(judge_module, "_JUDGE_MODEL", RagSettings().narration_model)
    fake_client = _FakeAnthropic(responses=[])

    with pytest.raises(ValueError, match="must differ from the generator"):
        AnthropicJudge(client=fake_client)  # type: ignore[arg-type]

    # No call was ever attempted -- the assertion fires before any use.
    assert fake_client.messages.calls == []


def test_judge_model_differs_by_default() -> None:
    """The real, unpatched _JUDGE_MODEL constant differs from the generator's default."""
    assert judge_module._JUDGE_MODEL != RagSettings().narration_model


def test_judge_happy_path_parses_valid_response_in_one_call() -> None:
    """A well-formed verdict on the first response is parsed with exactly one call."""
    fake_client = _FakeAnthropic(
        responses=[
            _verdict_json(
                grounded=True, cited=True, no_invented_numbers=True, notes="Fully grounded."
            )
        ]
    )
    judge = AnthropicJudge(client=fake_client)  # type: ignore[arg-type]

    verdict, error = judge.judge(
        label="Revenue", figures=["FY2024: $391.04B"], chunks=[_CHUNK], commentary=_GOOD_COMMENTARY
    )

    assert error == ""
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.grounded is True
    assert verdict.cited is True
    assert verdict.no_invented_numbers is True
    assert len(fake_client.messages.calls) == 1
    # No tools=... argument is ever passed (SECURITY.md item 3).
    assert "tools" not in fake_client.messages.calls[0]


def test_judge_retries_once_on_invalid_json_then_succeeds() -> None:
    """A malformed first response is retried exactly once and can then succeed."""
    fake_client = _FakeAnthropic(
        responses=[
            "not valid json at all",
            _verdict_json(
                grounded=False, cited=True, no_invented_numbers=True, notes="Missing support."
            ),
        ]
    )
    judge = AnthropicJudge(client=fake_client)  # type: ignore[arg-type]

    verdict, error = judge.judge(
        label="Revenue", figures=["FY2024: $391.04B"], chunks=[_CHUNK], commentary=_GOOD_COMMENTARY
    )

    assert error == ""
    assert verdict is not None
    assert verdict.grounded is False
    assert len(fake_client.messages.calls) == 2
    # The retry prompt must carry the specific validation error forward.
    retry_user = fake_client.messages.calls[1]["messages"][0]["content"]
    assert "not valid JSON" in retry_user


def test_judge_fails_closed_after_two_invalid_responses() -> None:
    """Two malformed responses in a row fail closed: (None, error), never a fabricated verdict."""
    fake_client = _FakeAnthropic(responses=["still not json", '{"grounded": "not-a-bool"}'])
    judge = AnthropicJudge(client=fake_client)  # type: ignore[arg-type]

    verdict, error = judge.judge(
        label="Revenue", figures=["FY2024: $391.04B"], chunks=[_CHUNK], commentary=_GOOD_COMMENTARY
    )

    assert verdict is None
    assert error != ""
    assert len(fake_client.messages.calls) == 2


def test_parse_verdict_rejects_missing_required_field() -> None:
    """A verdict JSON missing a required field fails validation, not silently defaults."""
    raw = json.dumps({"grounded": True, "cited": True, "notes": "missing no_invented_numbers"})
    verdict, error = _parse_verdict(raw)
    assert verdict is None
    assert "did not match the required JSON shape" in error


def test_run_skips_without_raising_when_anthropic_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run() returns the documented skip dict immediately, with zero client construction."""

    class _NotConfigured:
        anthropic_configured = False

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not construct a narrator/judge when skipping")

    monkeypatch.setattr(judge_module, "get_app_settings", lambda: _NotConfigured())
    monkeypatch.setattr(judge_module, "AnthropicNarrator", _explode)
    monkeypatch.setattr(judge_module, "AnthropicJudge", _explode)

    result = run()

    assert result == {"status": "skipped", "reason": "ANTHROPIC_API_KEY not configured"}


class _Configured:
    """Stand-in ``AppSettings`` with a "key present" reading, for tests below the skip check.

    Hermeticity (CLAUDE.md rule 7): without this, ``anthropic_configured``
    would fall through to the REAL ``get_app_settings()``, making these
    tests' behavior depend on whether a real key happens to be sitting in
    the local ``.env`` -- exactly what must never happen. Every test past
    the skip check installs this via ``monkeypatch`` and separately fakes
    ``_explicit_key_anthropic_client`` (never letting the real one, which
    also reads ``get_app_settings()``, run at all).
    """

    anthropic_configured = True


_SAMPLE: dict[str, Any] = {
    "ticker": "AAPL",
    "line_item_key": "revenue",
    "label": "Revenue",
    "figures": ["FY2024: $391.04B"],
    "chunks": [
        {
            "chunk_id": _CHUNK.chunk_id,
            "section": _CHUNK.section,
            "text": _CHUNK.text,
            "source_url": _CHUNK.source_url,
        }
    ],
    "commentary_text": "Revenue rose to $391.04B.",
    "citations": [{"chunk_id": _CHUNK.chunk_id, "quote": "Revenue increased to $391.04B"}],
}

_REFUSED_SAMPLE: dict[str, Any] = {
    "ticker": "AAPL",
    "line_item_key": "total_liabilities",
    "label": "Total liabilities",
    "figures": [],
    "chunks": [],
    "commentary_text": None,
    "citations": [],
}


def test_judge_samples_grades_plain_sample_dicts_without_any_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """judge_samples() reconstructs Chunk/Commentary from plain dicts and grades them.

    This is the per-sample-artifact contract judge_samples()/generate_samples()
    exist to satisfy: a caller (a saved generation_samples.json, in
    production) hands over JSON-serializable sample dicts -- never
    Chunk/Commentary objects -- and gets back exactly the same rate-summary
    shape a combined narrate-then-judge run would produce, with zero
    AnthropicNarrator construction anywhere in this path.
    """

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("judge_samples must never construct a narrator")

    monkeypatch.setattr(judge_module, "get_app_settings", lambda: _Configured())
    monkeypatch.setattr(judge_module, "AnthropicNarrator", _explode)
    fake_client = _FakeAnthropic(
        responses=[
            _verdict_json(
                grounded=True, cited=True, no_invented_numbers=True, notes="Fully grounded."
            )
        ]
    )
    monkeypatch.setattr(judge_module, "_explicit_key_anthropic_client", lambda: fake_client)

    result = judge_samples([_SAMPLE, _REFUSED_SAMPLE], tickers_run=["AAPL"], tickers_skipped={})

    assert result["status"] == "ok"
    assert result["n_narrated"] == 1  # the refused sample (text=None) is not counted
    assert result["n_judged"] == 1
    assert result["grounded_rate"] == 1.0
    assert result["cited_rate"] == 1.0
    assert result["no_invented_numbers_rate"] == 1.0
    assert len(result["details"]) == 2
    refused_detail = next(d for d in result["details"] if d["line_item_key"] == "total_liabilities")
    assert refused_detail["narrated"] is False
    assert refused_detail["judged"] is False


def test_judge_only_rerun_raises_when_no_saved_samples_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run(use_saved_generation=True) fails loudly, not by silently generating for real.

    "No new generation spend" means a judge-only rerun must never fall
    back to live (paid) narration just because no saved samples exist yet.
    """
    monkeypatch.setattr(judge_module, "get_app_settings", lambda: _Configured())
    monkeypatch.setattr(judge_module, "_GENERATION_SAMPLES_PATH", tmp_path / "missing.json")

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not narrate when no saved samples exist")

    monkeypatch.setattr(judge_module, "generate_samples", _explode)

    with pytest.raises(FileNotFoundError, match="use_saved_generation"):
        run(use_saved_generation=True)


def test_judge_only_rerun_loads_saved_samples_and_never_narrates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run(use_saved_generation=True) judges persisted samples with zero generation calls."""
    samples_path = tmp_path / "generation_samples.json"
    samples_path.write_text(
        json.dumps({"tickers_run": ["AAPL"], "tickers_skipped": {}, "samples": [_SAMPLE]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(judge_module, "get_app_settings", lambda: _Configured())
    monkeypatch.setattr(judge_module, "_GENERATION_SAMPLES_PATH", samples_path)

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not generate new samples on a judge-only rerun")

    monkeypatch.setattr(judge_module, "generate_samples", _explode)
    monkeypatch.setattr(judge_module, "AnthropicNarrator", _explode)
    fake_client = _FakeAnthropic(
        responses=[
            _verdict_json(
                grounded=True, cited=False, no_invented_numbers=True, notes="No citation given."
            )
        ]
    )
    monkeypatch.setattr(judge_module, "_explicit_key_anthropic_client", lambda: fake_client)

    result = run(use_saved_generation=True)

    assert result["status"] == "ok"
    assert result["n_judged"] == 1
    assert result["cited_rate"] == 0.0
    assert result["tickers_run"] == ["AAPL"]


def test_run_default_persists_generation_samples_before_judging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A normal run() call writes the full generation samples to disk before judging.

    This is the "evals must persist inputs/outputs per sample" contract:
    a later judge-only rerun (see the tests above) depends on this file
    existing with the real figures/chunks/commentary, not just a lean
    summary.
    """
    samples_path = tmp_path / "generation_samples.json"
    monkeypatch.setattr(judge_module, "get_app_settings", lambda: _Configured())
    monkeypatch.setattr(judge_module, "_GENERATION_SAMPLES_PATH", samples_path)
    monkeypatch.setattr(
        judge_module,
        "generate_samples",
        lambda: {"tickers_run": ["AAPL"], "tickers_skipped": {}, "samples": [_SAMPLE]},
    )
    fake_client = _FakeAnthropic(
        responses=[
            _verdict_json(
                grounded=True, cited=True, no_invented_numbers=True, notes="Fully grounded."
            )
        ]
    )
    monkeypatch.setattr(judge_module, "_explicit_key_anthropic_client", lambda: fake_client)

    result = run()

    assert result["status"] == "ok"
    persisted = json.loads(samples_path.read_text(encoding="utf-8"))
    assert persisted["samples"] == [_SAMPLE]
    assert persisted["tickers_run"] == ["AAPL"]
