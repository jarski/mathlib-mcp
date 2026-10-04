"""Build the packaged API data from library sources plus curated knowledge.

    uv run python scripts/fetch_sources.py      # once: download pinned sources
    uv run mathlib-mcp-build                    # regenerate src/mathlib_mcp/data/

Outputs:

* ``symbols.json``: every public name in the indexed namespaces with its kind and,
  where statically known, its signature. Used by ``check_snippet`` to detect
  hallucinated APIs.
* ``cards.json``: one API card per function/class in the card namespaces, merged with
  ``curation/cards/*.toml``.
* ``llms.txt`` / ``llms-full.txt``: the same knowledge as static documents, for the
  "docs in context" baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from mathlib_mcp.cards import ApiCard, CardMethod, CardParam
from mathlib_mcp.extract.docstrings import parse_docstring
from mathlib_mcp.extract.pysource import (
    SourceTree,
    SymbolInfo,
    describe,
    format_signature,
    function_params,
)

REPO = Path(__file__).resolve().parents[3]


@dataclass
class Library:
    name: str
    version: str
    source: Path
    packages: list[str]
    exclude: list[str]
    doc_url: str
    card_modules: list[str]
    card_names: list[str]
    resolve_packages: list[str]

    @classmethod
    def from_toml(cls, data: dict, base: Path) -> Library:
        return cls(
            name=data["name"],
            version=data["version"],
            source=base / data["source"],
            packages=data["packages"],
            exclude=data.get("exclude", []),
            doc_url=data["doc_url"],
            card_modules=data.get("card_modules", []),
            card_names=data.get("card_names", []),
            resolve_packages=data.get("resolve_packages", []),
        )

    def tree(self) -> SourceTree:
        packages = self.packages + self.resolve_packages
        return SourceTree({pkg: self.source / pkg for pkg in packages})

    def excluded(self, modname: str) -> bool:
        return any(modname == e or modname.startswith(e + ".") for e in self.exclude)


# ------------------------------------------------------------------------ symbols


def index_symbols(lib: Library, tree: SourceTree) -> tuple[dict, dict]:
    modules: dict[str, dict] = {}
    symbols: dict[str, dict] = {}
    queue = list(lib.packages)
    while queue:
        modname = queue.pop(0)
        if modname in modules or lib.excluded(modname) or tree.module(modname) is None:
            continue
        exported = tree.module(modname).all or []
        names = sorted(n for n in tree.attribute_names(modname)
                       if not n.startswith("_") or n in exported)
        submodules = sorted(sub.rsplit(".", 1)[1] for sub in tree.submodules(modname)
                            if sub.rsplit(".", 1)[1] not in names)
        modules[modname] = {"exact": tree.is_exact(modname), "names": names,
                            "submodules": submodules}
        for name in names + submodules:
            info = describe(tree, modname, name)
            if info is None:
                continue
            symbols[info.qualname] = _symbol_entry(info)
            if info.kind == "module" and info.target and tree.is_module(info.target) \
                    and not any(p.startswith("_") for p in info.target.split(".")):
                queue.append(info.target)
        queue += [sub for sub in tree.submodules(modname) if sub not in modules]
    # which indexed modules become importable attributes after `import <module>`
    for modname, entry in modules.items():
        entry["loads"] = sorted(m for m in tree.import_closure(modname)
                                if m in modules and m != modname)
    return modules, symbols


def _symbol_entry(info: SymbolInfo) -> dict:
    entry: dict = {"kind": info.kind}
    if info.params is not None:
        entry["params"] = [[p.name, p.kind, p.default] for p in info.params]
    if info.doc:
        summary = parse_docstring(info.doc).summary
        if summary:
            entry["summary"] = summary[:200]
    if info.members is not None:
        entry["members"] = info.members
        entry["members_exact"] = info.members_exact
    if info.target:
        entry["target"] = info.target
    if info.defined_in:
        entry["defined_in"] = f"{info.defined_in}:{info.lineno}"
    return entry


# -------------------------------------------------------------------------- cards


def build_cards(lib: Library, tree: SourceTree) -> list[ApiCard]:
    candidates: list[str] = []
    for modname in lib.card_modules:
        candidates += [f"{modname}.{n}" for n in tree.public_names(modname)]
    candidates += lib.card_names

    by_definition: dict[str, list[tuple[str, SymbolInfo]]] = defaultdict(list)
    for qualname in candidates:
        modname, name = qualname.rsplit(".", 1)
        info = describe(tree, modname, name)
        if info is None:
            raise SystemExit(f"[{lib.name}] card target not found: {qualname}")
        if info.kind not in ("function", "class"):
            continue
        key = f"{info.defined_in}:{info.lineno}"
        if all(q != qualname for q, _ in by_definition[key]):
            by_definition[key].append((qualname, info))

    cards = []
    for entries in by_definition.values():
        # the shortest public path is the canonical id (cupy.matmul over cupy.linalg.matmul)
        entries.sort(key=lambda e: (e[0].count("."), len(e[0])))
        qualname, info = entries[0]
        cards.append(_make_card(lib, tree, qualname, info, [q for q, _ in entries[1:]]))
    return sorted(cards, key=lambda c: c.id)


def _make_card(lib: Library, tree: SourceTree, qualname: str, info: SymbolInfo,
               aliases: list[str]) -> ApiCard:
    doc = parse_docstring(info.doc)
    name = qualname.rsplit(".", 1)[1]
    params = [
        CardParam(p.name, p.kind, p.default, p.annotation, doc.params.get(p.name, ""))
        for p in (info.params or [])
    ]
    signature = format_signature(name, info.params) if info.params is not None else f"{name}(...)"
    methods = _methods(tree, info) if info.kind == "class" else []
    return ApiCard(
        id=qualname,
        library=lib.name,
        library_version=lib.version,
        kind=info.kind,
        signature=signature,
        summary=doc.summary,
        description=doc.description,
        params=params,
        returns=doc.returns,
        raises=doc.raises,
        notes=doc.notes,
        warnings=doc.warnings,
        methods=methods,
        members=[m for m in (info.members or []) if m not in {x.name for x in methods}],
        doc_examples=_select_examples(doc.examples, name),
        see_also=doc.see_also,
        source=_relative_source(lib, tree, info),
        doc_url=lib.doc_url.format(qualname=qualname, version=lib.version,
                                  section=qualname.split(".")[1]),
        aliases=aliases,
    )


def _select_examples(blocks: list[str], name: str, max_lines: int = 30) -> list[str]:
    """Join doctest cells into one script, skipping logging set-up cells."""
    cells = [b for b in blocks if "logging." not in b]
    if not any(f"{name}(" in cell for cell in cells):
        return []
    lines = "\n\n".join(cells).splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["# ..."]
    return ["\n".join(lines)]


def _methods(tree: SourceTree, info: SymbolInfo) -> list[CardMethod]:
    import ast

    if info.defined_in is None:
        return []
    module = tree.module(info.defined_in)
    if module is None or module.tree is None:
        return []
    name = info.qualname.rsplit(".", 1)[1]
    resolved = tree.resolve(info.defined_in, name) if name in module.bindings else None
    if resolved is None or resolved[0] is None or not isinstance(resolved[1].node, ast.ClassDef):
        # the public name may differ from the class name
        nodes = [n for n in module.tree.body
                 if isinstance(n, ast.ClassDef) and n.lineno == info.lineno]
        if not nodes:
            return []
        node = nodes[0]
    else:
        node = resolved[1].node
    methods = []
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and not stmt.name.startswith("_"):
            is_property = any(ast.unparse(d) == "property" or ast.unparse(d).endswith(".setter")
                              for d in stmt.decorator_list)
            if is_property:
                continue
            summary = parse_docstring(tree.docstring(module, stmt, owner=node)).summary
            params = function_params(stmt, drop_self=True)
            methods.append(CardMethod(stmt.name, format_signature(stmt.name, params),
                                      summary[:200]))
    return methods


def _relative_source(lib: Library, tree: SourceTree, info: SymbolInfo) -> str:
    if info.defined_in is None:
        return ""
    found = tree.find(info.defined_in)
    if found is None:
        return info.defined_in
    try:
        path = found[0].relative_to(lib.source)
    except ValueError:
        path = found[0]
    return f"{path}:{info.lineno}" if info.lineno else str(path)


# ----------------------------------------------------------------------- curation

CURATED_FIELDS = {"backend", "numpy_equivalent", "tags", "pitfalls", "example"}


def apply_curation(cards: dict[str, ApiCard], curation_dir: Path) -> int:
    alias_to_id = {alias: card.id for card in cards.values() for alias in card.aliases}
    count = 0
    for path in sorted(curation_dir.glob("*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        for card_id, fields in data.get("cards", {}).items():
            target = cards.get(card_id) or cards.get(alias_to_id.get(card_id, ""))
            if target is None:
                raise SystemExit(f"{path.name}: curated card '{card_id}' has no extracted API. "
                                 "Is it in card_modules/card_names, and does it still exist?")
            unknown = set(fields) - CURATED_FIELDS
            if unknown:
                raise SystemExit(f"{path.name}: unknown fields for {card_id}: {sorted(unknown)}")
            for key, value in fields.items():
                setattr(target, key, value.strip() if isinstance(value, str) else value)
            target.curated = True
            count += 1
    return count


def apply_verification(cards: dict[str, ApiCard], path: Path) -> int:
    """Mark examples that passed ``bench/gpu_validate.py`` on a real GPU.

    ``verified.json`` is written from the validation report and records, per card, the
    example text that passed; an example edited after its run is not marked verified.
    """
    if not path.is_file():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for card_id, record in data.get("examples", {}).items():
        card = cards.get(card_id)
        if card is not None and card.example and card.example.strip() == record["code"].strip():
            card.example_verified_on = data["environment"]
            count += 1
    return count


# --------------------------------------------------------------------- llms.txt


def render_llms_txt(cards: list[ApiCard], libraries: list[Library]) -> tuple[str, str]:
    header = [
        "# GPU math libraries for LLMs: " + ", ".join(f"{lib.name} {lib.version}"
                                                       for lib in libraries),
        "",
        "> API reference for GPU-accelerated linear algebra, FFT and sparse computations "
        "(cuBLAS, cuFFT, cuSOLVER, cuSPARSE) through CuPy and nvmath-python. Curated cards "
        "add the CUDA-X backend, common pitfalls and an example verified on a GPU.",
        "",
    ]
    by_module: dict[str, list[ApiCard]] = defaultdict(list)
    for card in cards:
        by_module[card.id.rsplit(".", 1)[0]].append(card)
    index = list(header)
    for module in sorted(by_module):
        index += [f"## {module}", ""]
        for card in by_module[module]:
            star = " (curated)" if card.curated else ""
            index.append(f"- [{card.id}]({card.doc_url}){star}: {card.summary}")
        index.append("")
    full = list(header)
    for card in sorted(cards, key=lambda c: (not c.curated, c.id)):
        full.append(card.to_markdown(full=card.curated))
    return "\n".join(index).strip() + "\n", "\n".join(full).strip() + "\n"


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--curation", type=Path, default=REPO / "curation")
    parser.add_argument("--out", type=Path, default=REPO / "src" / "mathlib_mcp" / "data")
    args = parser.parse_args(argv)

    config = tomllib.loads((args.curation / "libraries.toml").read_text(encoding="utf-8"))
    libraries = [Library.from_toml(entry, REPO) for entry in config["library"]]

    all_modules, all_symbols, all_cards = {}, {}, {}
    for lib in libraries:
        if not lib.source.is_dir():
            sys.exit(f"Missing sources for {lib.name}: {lib.source}. "
                     "Run `uv run python scripts/fetch_sources.py` first.")
        tree = lib.tree()
        modules, symbols = index_symbols(lib, tree)
        for modname in modules:
            modules[modname]["library"] = lib.name
        all_modules.update(modules)
        all_symbols.update(symbols)
        cards = build_cards(lib, tree)
        all_cards.update({card.id: card for card in cards})
        print(f"{lib.name} {lib.version}: {len(modules)} modules, {len(symbols)} symbols, "
              f"{len(cards)} cards")

    from mathlib_mcp.checker import SymbolIndex

    index = SymbolIndex({"modules": all_modules, "symbols": all_symbols})
    for card in all_cards.values():
        card.import_hint = index.import_hint(card.id)
    curated = apply_curation(all_cards, args.curation / "cards")
    verified = apply_verification(all_cards, args.curation / "verified.json")
    print(f"curated: {curated} cards, GPU-verified examples: {verified}")

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"libraries": {lib.name: {"version": lib.version, "packages": lib.packages,
                                     "exclude": lib.exclude} for lib in libraries}}
    (args.out / "symbols.json").write_text(
        json.dumps({**meta, "modules": all_modules, "symbols": all_symbols},
                   separators=(",", ":"), sort_keys=True), encoding="utf-8")
    cards_sorted = sorted(all_cards.values(), key=lambda c: c.id)
    (args.out / "cards.json").write_text(
        json.dumps({**meta, "cards": [c.to_dict() for c in cards_sorted]}, indent=1,
                   ensure_ascii=False), encoding="utf-8")
    index, full = render_llms_txt(cards_sorted, libraries)
    (args.out / "llms.txt").write_text(index, encoding="utf-8")
    (args.out / "llms-full.txt").write_text(full, encoding="utf-8")
    for name in ("symbols.json", "cards.json", "llms.txt", "llms-full.txt"):
        print(f"  wrote {name}: {(args.out / name).stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
