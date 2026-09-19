"""Scaffold guarantees: contracts import cleanly, deterministic modules stay LLM-free."""

import ast
import pathlib

BANNED = {
    "anthropic",
    "langchain",
    "langchain_anthropic",
    "langchain_core",
    "langgraph",
    "chromadb",
    "sentence_transformers",
}
DETERMINISTIC_DIRS = ("app/edgar", "app/model")


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_deterministic_modules_are_llm_free() -> None:
    """CLAUDE.md rule 6: app/edgar and app/model never import LLM/vector libs."""
    for directory in DETERMINISTIC_DIRS:
        for path in pathlib.Path(directory).rglob("*.py"):
            offending = _imported_roots(ast.parse(path.read_text())) & BANNED
            assert not offending, f"{path} imports banned modules: {sorted(offending)}"


def test_contracts_import_and_behave() -> None:
    """The frozen contracts stay importable and TieoutReport aggregates correctly."""
    from app.schemas import TieoutCheck, TieoutReport

    report = TieoutReport(
        ticker="AAPL",
        checks=[
            TieoutCheck(
                check_id="balance_sheet_equation_2025",
                description="Assets = Liabilities + Equity",
                fiscal_year=2025,
                passed=True,
            )
        ],
    )
    assert report.passed
