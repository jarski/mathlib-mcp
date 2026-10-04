"""The four agent-facing tools as plain functions.

They return Markdown strings meant for an LLM context window. ``server.py`` exposes
them over MCP, and the benchmark harness can call them in-process.
"""

from __future__ import annotations

from mathlib_mcp.cards import ApiCard, load_cards, load_symbols
from mathlib_mcp.checker import check_snippet as _check
from mathlib_mcp.checker import default_index
from mathlib_mcp.search import default_search_index

SERVER_INSTRUCTIONS = """\
Reference for GPU math in Python: CuPy {cupy} and nvmath-python {nvmath}, covering cuBLAS,
cuFFT, cuSOLVER, cuSPARSE, cuDSS and cuTENSOR. Workflow for writing GPU code:
1. `search_api` with a description of the task to find candidate functions.
2. `get_api_card` for each function you will call: exact signature, parameter meanings,
   pitfalls and a GPU-verified example.
3. `check_snippet` on your final code: it finds non-existent functions, wrong keyword
   arguments and missing arguments without running the code.
"""


def server_instructions() -> str:
    libraries = load_symbols().get("libraries", {})
    return SERVER_INSTRUCTIONS.format(
        cupy=libraries.get("cupy", {}).get("version", "?"),
        nvmath=libraries.get("nvmath-python", {}).get("version", "?"),
    )


def search_api(query: str, library: str | None = None, limit: int = 8) -> str:
    """Find functions and classes for a task description or a name fragment."""
    limit = max(1, min(limit, 20))
    hits = default_search_index().search(query, k=limit, library=library)
    if not hits:
        return f"No API matches for {query!r}. Try different words, e.g. the NumPy name."
    lines = [f"Top {len(hits)} matches for {query!r} ([curated] = has pitfalls and a GPU "
             "example; call get_api_card for details):"]
    lines += [hit.to_line() for hit in hits]
    return "\n".join(lines)


def get_api_card(name: str) -> str:
    """Full reference for one function/class, by its dotted name (e.g. cupy.linalg.solve)."""
    name = name.strip().strip("`").removesuffix("()")
    cards = load_cards()
    card = cards.get(name) or _card_by_alias(cards, name)
    if card is not None:
        return card.to_markdown(full=True)
    symbols = load_symbols()["symbols"]
    entry = symbols.get(name)
    if entry is not None and entry["kind"] in ("function", "class"):
        return _light_card(name, entry)
    if entry is not None:
        target = f" -> `{entry['target']}`" if entry.get("target") else ""
        return f"`{name}` is a {entry['kind']}{target}. No API card is available for it."
    result = _check(f"import {name.split('.')[0]}\n{name}\n")
    suggestions = [s for issue in result.issues for s in issue.suggestions]
    message = f"`{name}` was not found in the indexed APIs."
    if suggestions:
        message += " Did you mean: " + ", ".join(f"`{s}`" for s in suggestions) + "?"
    else:
        message += " Use search_api to find the right name."
    return message


def get_example(task: str, limit: int = 2) -> str:
    """GPU-verified example code for the curated APIs most relevant to a task."""
    cards = load_cards()
    hits = default_search_index().search(task, k=20)
    chosen = [cards[h.id] for h in hits if h.id in cards and cards[h.id].example][:limit]
    if not chosen:
        return f"No curated example matches {task!r}. Use search_api and get_api_card."
    parts = []
    for card in chosen:
        status = (f"verified on {card.example_verified_on}" if card.example_verified_on
                  else "curated example")
        parts.append(f"### `{card.id}` ({status})\n```python\n{card.example.strip()}\n```")
        if card.pitfalls:
            parts.append("Pitfalls:\n" + "\n".join(f"- {p}" for p in card.pitfalls))
    return "\n\n".join(parts)


def check_snippet(code: str) -> str:
    """Statically check code that uses CuPy / nvmath-python for API misuse."""
    return _check(code, default_index()).to_text()


def llms_txt() -> str:
    from importlib import resources

    return resources.files("mathlib_mcp").joinpath("data", "llms.txt").read_text("utf-8")


def _card_by_alias(cards: dict[str, ApiCard], name: str) -> ApiCard | None:
    for card in cards.values():
        if name in card.aliases:
            return card
    return None


def _light_card(name: str, entry: dict) -> str:
    short = name.rsplit(".", 1)[1]
    params = entry.get("params")
    if params:
        from mathlib_mcp.checker import _format_signature

        signature = _format_signature(short, params)
    else:
        signature = f"{short}(...)"
    lines = [f"### `{name}`", "", f"```python\n{signature}\n```", "",
             f"*{entry['kind']} · extracted from source, no curated card*", ""]
    hint = default_index().import_hint(name)
    if hint:
        module = hint.split()[1]
        lines += [f"**Import:** `{hint}` (`import {module.split('.')[0]}` alone does not "
                  f"load `{module}`)", ""]
    if entry.get("summary"):
        lines += [entry["summary"], ""]
    if entry.get("members"):
        lines += ["**Members:** " + ", ".join(f"`{m}`" for m in entry["members"][:60]), ""]
    return "\n".join(lines).strip() + "\n"
