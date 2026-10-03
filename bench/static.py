"""Static scoring of candidate code (runs anywhere, no GPU needed)."""

from __future__ import annotations

from bench.task import Task
from mathlib_mcp.checker import check_snippet


def required_api_missing(task: Task, checked: list[str]) -> list[str]:
    """Required APIs (``a|b`` = alternatives, prefixes allowed) that the code never uses."""
    missing = []
    for requirement in task.required_api:
        options = requirement.split("|")
        if not any(c == o or c.startswith(o + ".") for o in options for c in checked):
            missing.append(requirement)
    return missing


def analyze(task: Task, code: str) -> dict:
    result = check_snippet(code)
    return {
        "api_errors": [issue.__dict__ for issue in result.errors],
        "api_references": len(result.checked),
        "required_api_missing": required_api_missing(task, result.checked),
    }
