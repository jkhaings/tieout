"""Shared fixtures and fakes for app/rag tests.

Everything here is hermetic (CLAUDE.md rule 7): no network, no model
downloads, no API calls. ``FakeEmbedder`` produces deterministic hash-based
vectors instead of calling a real embedding model, and ``FakeLLM`` returns
scripted canned responses instead of calling Anthropic.
"""

from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def aapl_10k_excerpt_html() -> str:
    """Load the real (trimmed) AAPL 10-K HTML excerpt fixture as text."""
    return (FIXTURES_DIR / "aapl_10k_excerpt.html").read_text(encoding="utf-8")


class FakeEmbedder:
    """Deterministic, hash-based stand-in for a real embedding model.

    No ML download and no network: each text is hashed into a small,
    fixed-length vector of floats in ``[0, 1)``. Identical input text always
    produces the identical vector, which is all hybrid-retrieval tests need.
    """

    def __init__(self, dimensions: int = 16) -> None:
        """Set up the fake embedder.

        Args:
            dimensions: Length of each returned embedding vector.
        """
        self.dimensions = dimensions
        self.call_count = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one deterministic vector per input text.

        Args:
            texts: Texts to embed.

        Returns:
            One vector (list of floats) per element of ``texts``, in order.
        """
        self.call_count += 1
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            needed = self.dimensions
            raw = (digest * (needed // len(digest) + 1))[:needed]
            vectors.append([byte / 255.0 for byte in raw])
        return vectors


class FakeLLM:
    """Scriptable stand-in for the narration LLM.

    Records every ``(system, user)`` pair it is called with, and returns
    canned responses from a queue in call order (so a test can script
    "first call returns invalid JSON, second call returns valid JSON" to
    exercise the one-retry-then-fail-closed path).
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        """Set up the fake LLM.

        Args:
            responses: Canned strings returned in order, one per call to
                :meth:`complete`. If exhausted, later calls raise
                ``IndexError`` so a test notices it under-scripted.
        """
        self._responses: deque[str] = deque(responses or [])
        self.calls: list[tuple[str, str]] = []
        self.call_count = 0

    def complete(self, *, system: str, user: str) -> str:
        """Record the call and return the next scripted response.

        Args:
            system: The system prompt passed by the caller.
            user: The user prompt passed by the caller.

        Returns:
            The next canned response string from the queue.
        """
        self.calls.append((system, user))
        self.call_count += 1
        return self._responses.popleft()
