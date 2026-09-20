"""tieout evaluation layer: labeled datasets, tie-out/retrieval/judge evals, scorecard.

Local-only (CLAUDE.md rule 7): these modules make real Anthropic API calls
and, once, real SEC network calls to build fixtures. Never imported by
`tests/` and never run in CI -- `make evals` / `python -m evals.scorecard`
is the local entry point.
"""

from __future__ import annotations
