.PHONY: setup test lint type run evals audit

setup: ; uv sync --all-extras
test: ; uv run pytest -q
lint: ; uv run ruff check . && uv run ruff format --check .
type: ; uv run mypy app evals
run: ; uv run uvicorn app.main:app --reload
evals: ; uv run python -m evals.scorecard
audit: ; uv run pip-audit
