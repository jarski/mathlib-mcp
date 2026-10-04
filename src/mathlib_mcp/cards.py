"""API cards: the LLM-facing representation of one public API symbol.

A card combines facts extracted from the library source (signature, parameter docs,
return value) with curated knowledge that docstrings rarely state (which CUDA-X library
does the work, pitfalls, a verified runnable example, the NumPy/SciPy equivalent).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import cache
from importlib import resources
from typing import Any


@dataclass
class CardParam:
    name: str
    kind: str
    default: str | None = None
    annotation: str | None = None
    description: str = ""


@dataclass
class CardMethod:
    name: str
    signature: str
    summary: str = ""


@dataclass
class ApiCard:
    id: str  # public qualified name, e.g. "cupy.linalg.solve"
    library: str  # distribution name, e.g. "cupy"
    library_version: str
    kind: str  # function | class
    signature: str
    summary: str = ""
    description: str = ""
    params: list[CardParam] = field(default_factory=list)
    returns: str = ""
    raises: str = ""
    notes: str = ""
    warnings: str = ""
    methods: list[CardMethod] = field(default_factory=list)
    members: list[str] = field(default_factory=list)
    doc_examples: list[str] = field(default_factory=list)
    see_also: list[str] = field(default_factory=list)
    source: str = ""
    doc_url: str = ""
    aliases: list[str] = field(default_factory=list)
    import_hint: str | None = None  # e.g. "import cupy.cublas" when `import cupy` is not enough
    # curated
    curated: bool = False
    backend: list[str] = field(default_factory=list)
    numpy_equivalent: str | None = None
    tags: list[str] = field(default_factory=list)
    pitfalls: list[str] = field(default_factory=list)
    example: str | None = None
    example_verified_on: str | None = None  # e.g. "Tesla T4, CUDA 12.8, 2026-10-04"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApiCard:
        data = dict(data)
        data["params"] = [CardParam(**p) for p in data.get("params", [])]
        data["methods"] = [CardMethod(**m) for m in data.get("methods", [])]
        return cls(**data)

    def search_text(self) -> str:
        """Text indexed for retrieval."""
        parts = [
            self.id, self.id.replace(".", " ").replace("_", " "), " ".join(self.aliases),
            self.summary, self.description[:600], " ".join(self.tags), " ".join(self.backend),
            self.numpy_equivalent or "", " ".join(p.name for p in self.params),
            " ".join(m.name for m in self.methods),
        ]
        return "\n".join(p for p in parts if p)

    def to_markdown(self, *, full: bool = True) -> str:
        """Render the card as compact Markdown for an LLM context window."""
        lines = [f"### `{self.id}`", "", f"```python\n{self.signature}\n```", ""]
        meta = [f"{self.library} {self.library_version}", self.kind]
        if self.backend:
            meta.append("backend: " + ", ".join(self.backend))
        if self.numpy_equivalent:
            meta.append(f"NumPy/SciPy equivalent: `{self.numpy_equivalent}`")
        lines += ["*" + " · ".join(meta) + "*", ""]
        if self.import_hint:
            module = self.import_hint.split()[1]
            lines += [f"**Import:** `{self.import_hint}` (`import {module.split('.')[0]}` alone "
                      f"does not load `{module}`)", ""]
        if self.summary:
            lines += [self.summary, ""]
        if full and self.description:
            lines += [_truncate(self.description, 1200), ""]
        if self.params:
            lines.append("**Parameters**")
            for p in self.params:
                default = f" = `{p.default}`" if p.default is not None else ""
                desc = f": {_truncate(p.description, 400)}" if p.description else ""
                lines.append(f"- `{p.name}`{default}{desc}")
            lines.append("")
        if self.returns:
            lines += [f"**Returns:** {_truncate(self.returns, 600)}", ""]
        if self.methods:
            lines.append("**Methods**")
            lines += [f"- `{m.signature}`" + (f": {m.summary}" if m.summary else "")
                      for m in self.methods[:40]]
            lines.append("")
        if self.members and not self.methods:
            lines += ["**Members:** " + ", ".join(f"`{m}`" for m in self.members[:60]), ""]
        if self.pitfalls:
            lines.append("**Pitfalls**")
            lines += [f"- {p}" for p in self.pitfalls]
            lines.append("")
        if full and self.warnings:
            lines += [f"**Warning:** {_truncate(self.warnings, 600)}", ""]
        if full and self.notes:
            lines += [f"**Notes:** {_truncate(self.notes, 800)}", ""]
        if self.example:
            label = (f"verified on {self.example_verified_on}" if self.example_verified_on
                     else "curated, not yet verified on a GPU")
            lines += [f"**Example** ({label})", f"```python\n{self.example.strip()}\n```", ""]
        elif full and self.doc_examples:
            lines += ["**Example** (from docstring)",
                      f"```python\n{self.doc_examples[0].strip()}\n```", ""]
        if self.see_also:
            lines += ["**See also:** " + ", ".join(f"`{s}`" for s in self.see_also[:8]), ""]
        if self.doc_url:
            lines += [f"Docs: {self.doc_url}"]
        return "\n".join(lines).strip() + "\n"


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ------------------------------------------------------------------ packaged data


@cache
def load_cards() -> dict[str, ApiCard]:
    raw = json.loads(_data_file("cards.json"))
    return {card["id"]: ApiCard.from_dict(card) for card in raw["cards"]}


@cache
def load_symbols() -> dict[str, Any]:
    return json.loads(_data_file("symbols.json"))


def _data_file(name: str) -> str:
    return resources.files("mathlib_mcp").joinpath("data", name).read_text(encoding="utf-8")
