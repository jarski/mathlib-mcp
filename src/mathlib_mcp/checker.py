"""Static checks for LLM-generated code that uses CuPy / nvmath-python.

``check_snippet`` parses the code (it is never executed) and verifies every reference
into the indexed libraries against the symbol index built from the library sources:

* the module / function / class / enum member exists (catches hallucinated APIs and
  suggests the closest real name, also across modules: ``cupy.linalg.lu_factor`` ->
  ``cupyx.scipy.linalg.lu_factor``);
* calls use parameters that exist, respect positional-only / keyword-only rules and
  provide required arguments (catches ``cg(A, b, tol=...)`` -> ``rtol``).

Names inside namespaces the index cannot enumerate completely (Cython modules,
excluded packages) are reported as *unverified* instead of as errors.
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import asdict, dataclass, field
from functools import cache
from typing import Any

from mathlib_mcp.cards import load_symbols


@dataclass
class Issue:
    severity: str  # error | info
    code: str  # syntax-error | unknown-module | unknown-symbol | unknown-attribute |
    #            unknown-keyword | positional-only | too-many-positional | missing-argument |
    #            unverified
    line: int
    symbol: str
    message: str
    suggestions: list[str] = field(default_factory=list)


@dataclass
class CheckResult:
    issues: list[Issue]
    checked: list[str]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "issues": [asdict(i) for i in self.issues],
                "checked": self.checked}

    def to_text(self) -> str:
        lines = []
        errors = self.errors
        if errors:
            lines.append(f"FAIL: {len(errors)} problem(s) found")
        else:
            lines.append("OK: no problems found")
        for issue in self.issues:
            tag = "error" if issue.severity == "error" else "note"
            text = f"- line {issue.line} [{tag}: {issue.code}] {issue.message}"
            if issue.suggestions:
                text += " Did you mean: " + ", ".join(f"`{s}`" for s in issue.suggestions) + "?"
            lines.append(text)
        lines.append(f"Checked {len(self.checked)} API reference(s): "
                     + (", ".join(f"`{c}`" for c in self.checked[:20]) or "none")
                     + (" ..." if len(self.checked) > 20 else ""))
        return "\n".join(lines)


class SymbolIndex:
    """Lookup structure over ``symbols.json``."""

    def __init__(self, data: dict[str, Any] | None = None):
        data = data if data is not None else load_symbols()
        self.modules: dict[str, dict] = data["modules"]
        self.symbols: dict[str, dict] = data["symbols"]
        self.libraries: dict[str, dict] = data.get("libraries", {})
        self.roots = {name.split(".")[0] for name in self.modules}
        self.excluded = [prefix for lib in self.libraries.values()
                         for prefix in lib.get("exclude", [])]
        self._by_name: dict[str, list[str]] = {}
        for qualname, entry in self.symbols.items():
            if entry["kind"] in ("function", "class"):
                self._by_name.setdefault(qualname.rsplit(".", 1)[1], []).append(qualname)

    def elsewhere(self, name: str, limit: int = 3) -> list[str]:
        """Public paths where a symbol with this exact name exists."""
        candidates = self._by_name.get(name, [])
        return sorted(candidates, key=lambda q: (q.count("."), q))[:limit]

    def similar(self, module: str, name: str, limit: int = 3) -> list[str]:
        names = self.modules.get(module, {}).get("names", [])
        public = [n for n in names if not n.startswith("_")]
        return [f"{module}.{n}" for n in difflib.get_close_matches(name, public, limit, 0.6)]


@cache
def default_index() -> SymbolIndex:
    return SymbolIndex()


# ----------------------------------------------------------------------------


@dataclass
class _Resolved:
    status: str  # ok | missing | unverified | external
    qualname: str
    entry: dict | None = None
    issue: Issue | None = None


def _resolve(index: SymbolIndex, parts: list[str], line: int) -> _Resolved:
    """Walk a dotted path such as ``['cupy', 'linalg', 'solve']`` through the index."""
    module = parts[0]
    if module not in index.modules:
        return _Resolved("unverified", module)
    qualname = module
    entry: dict | None = None
    for position, part in enumerate(parts[1:], start=1):
        info = index.modules.get(module)
        if info is None:
            return _Resolved("unverified", ".".join(parts), entry)
        if part not in info["names"]:
            wanted = ".".join(parts[: position + 1])
            if not info["exact"]:
                return _Resolved("unverified", wanted)
            suggestions = index.similar(module, part) + [
                s for s in index.elsewhere(part) if s != wanted]
            message = f"`{wanted}` does not exist (no `{part}` in module `{module}`)."
            if f"{module}.{part}" in _excluded_modules(index):
                return _Resolved("unverified", wanted)
            return _Resolved("missing", wanted, None, Issue(
                "error", "unknown-symbol", line, wanted, message, _dedupe(suggestions)[:4]))
        qualname = f"{module}.{part}"
        entry = index.symbols.get(qualname)
        if entry is None:
            return _Resolved("ok", ".".join(parts[: position + 1]))
        kind = entry["kind"]
        rest = parts[position + 1:]
        if kind == "module":
            target = entry.get("target") or qualname
            if not rest:
                return _Resolved("ok", qualname, entry)
            if target not in index.modules:
                return _Resolved("unverified", ".".join(parts), entry)
            module = target
            continue
        if kind == "class" and rest:
            members = entry.get("members") or []
            if entry.get("members_exact") and rest[0] not in members \
                    and not rest[0].startswith("__"):
                wanted = f"{qualname}.{rest[0]}"
                close = difflib.get_close_matches(rest[0], members, 3, 0.6)
                return _Resolved("missing", wanted, None, Issue(
                    "error", "unknown-attribute", line, wanted,
                    f"`{qualname}` has no attribute `{rest[0]}`.",
                    [f"{qualname}.{m}" for m in close]))
            return _Resolved("ok", qualname, entry)
        if kind == "external":
            return _Resolved("ok", qualname, entry)
        return _Resolved("ok", qualname, entry)
    return _Resolved("ok", qualname, index.symbols.get(qualname) or {"kind": "module"})


def _excluded_modules(index: SymbolIndex) -> set[str]:
    return set(index.excluded)


def _dedupe(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def _attr_chain(node: ast.expr) -> list[str] | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


class _Checker(ast.NodeVisitor):
    def __init__(self, index: SymbolIndex):
        self.index = index
        self.aliases: dict[str, str] = {}  # local name -> fully qualified name
        self.issues: list[Issue] = []
        self.checked: list[str] = []
        self._seen_attrs: set[int] = set()

    # --------------------------------------------------------------- imports

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in self.index.roots:
                continue
            self._check_path(alias.name.split("."), node.lineno, is_module=True)
            if alias.asname:
                self.aliases[alias.asname] = alias.name
            else:
                self.aliases[root] = root

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or not node.module or node.module.split(".")[0] not in self.index.roots:
            return
        if self._check_path(node.module.split("."), node.lineno, is_module=True) is None:
            return
        for alias in node.names:
            if alias.name == "*":
                continue
            qualname = f"{node.module}.{alias.name}"
            if self._check_path(qualname.split("."), node.lineno) is not None:
                self.aliases[alias.asname or alias.name] = qualname

    def visit_Assign(self, node: ast.Assign) -> None:
        # module aliases such as `la = cp.linalg` or `solve = cupy.linalg.solve`
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            qualname = self._qualify(node.value)
            if qualname is not None:
                self.aliases[node.targets[0].id] = qualname
            elif node.targets[0].id in self.aliases:
                del self.aliases[node.targets[0].id]  # rebound to something else

    # ------------------------------------------------------------ references

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if id(node) not in self._seen_attrs:
            self._mark_chain(node)
            qualname = self._qualify(node)
            if qualname is not None:
                self._check_path(qualname.split("."), node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        qualname = self._qualify(node.func)
        if qualname is not None:
            if isinstance(node.func, ast.Attribute):
                self._mark_chain(node.func)
            resolved = self._check_path(qualname.split("."), node.lineno)
            if resolved is not None and resolved.status == "ok" and resolved.entry:
                self._check_call(node, resolved.qualname, resolved.entry)
        self.generic_visit(node)

    def _mark_chain(self, node: ast.expr) -> None:
        while isinstance(node, ast.Attribute):
            self._seen_attrs.add(id(node))
            node = node.value

    def _qualify(self, node: ast.expr) -> str | None:
        chain = _attr_chain(node)
        if chain is None or chain[0] not in self.aliases:
            return None
        return ".".join([self.aliases[chain[0]]] + chain[1:])

    def _check_path(self, parts: list[str], line: int, is_module: bool = False
                    ) -> _Resolved | None:
        resolved = _resolve(self.index, parts, line)
        if resolved.status == "missing" and resolved.issue is not None:
            issue = resolved.issue
            if is_module:
                issue.code = "unknown-module"
            if not any(i.symbol == issue.symbol and i.code == issue.code for i in self.issues):
                self.issues.append(issue)
            return None
        if resolved.status == "unverified":
            if not any(i.symbol == resolved.qualname for i in self.issues):
                self.issues.append(Issue(
                    "info", "unverified", line, resolved.qualname,
                    f"`{resolved.qualname}` could not be verified (namespace not fully "
                    "indexed)."))
            return resolved
        if resolved.qualname not in self.checked:
            self.checked.append(resolved.qualname)
        return resolved

    # ----------------------------------------------------------------- calls

    def _check_call(self, node: ast.Call, qualname: str, entry: dict) -> None:
        params = entry.get("params")
        if params is None or entry["kind"] not in ("function", "class"):
            return
        has_star_args = any(isinstance(a, ast.Starred) for a in node.args)
        has_star_kwargs = any(k.arg is None for k in node.keywords)
        kinds = {name: kind for name, kind, _ in params}
        positional = [p for p in params if p[1] in ("positional_only", "positional_or_keyword")]
        accepts_var_pos = any(p[1] == "var_positional" for p in params)
        accepts_var_kw = any(p[1] == "var_keyword" for p in params)
        signature = _format_signature(qualname.rsplit(".", 1)[1], params)
        line = node.lineno

        if not has_star_args and not accepts_var_pos and len(node.args) > len(positional):
            self.issues.append(Issue(
                "error", "too-many-positional", line, qualname,
                f"`{qualname}` takes at most {len(positional)} positional argument(s) but "
                f"{len(node.args)} were given. Signature: `{signature}`."))
        given = {p[0] for p in positional[: len(node.args)]}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            kind = kinds.get(keyword.arg)
            if kind == "positional_only":
                self.issues.append(Issue(
                    "error", "positional-only", line, qualname,
                    f"`{keyword.arg}` is positional-only in `{qualname}`; pass it without "
                    f"the keyword. Signature: `{signature}`."))
            elif (kind is None or kind in ("var_positional", "var_keyword")) \
                    and not accepts_var_kw:
                names = [p[0] for p in params if p[1] in ("positional_or_keyword",
                                                          "keyword_only")]
                close = difflib.get_close_matches(keyword.arg, names, 3, 0.5)
                self.issues.append(Issue(
                    "error", "unknown-keyword", line, qualname,
                    f"`{qualname}` has no parameter `{keyword.arg}`. "
                    f"Signature: `{signature}`.", close))
            given.add(keyword.arg)
        if has_star_args or has_star_kwargs:
            return
        missing = [name for name, kind, default in params
                   if default is None and kind in ("positional_only", "positional_or_keyword",
                                                   "keyword_only") and name not in given]
        if missing:
            self.issues.append(Issue(
                "error", "missing-argument", line, qualname,
                f"`{qualname}` is missing required argument(s) "
                + ", ".join(f"`{m}`" for m in missing) + f". Signature: `{signature}`."))


def _format_signature(name: str, params: list[list]) -> str:
    parts: list[str] = []
    star_done = False
    for index, (pname, kind, default) in enumerate(params):
        if kind == "keyword_only" and not star_done:
            parts.append("*")
            star_done = True
        if kind == "var_positional":
            parts.append(f"*{pname}")
            star_done = True
        elif kind == "var_keyword":
            parts.append(f"**{pname}")
        else:
            parts.append(pname if default is None else f"{pname}={default}")
        if kind == "positional_only" and (index + 1 == len(params)
                                          or params[index + 1][1] != "positional_only"):
            parts.append("/")
    return f"{name}({', '.join(parts)})"


def check_snippet(code: str, index: SymbolIndex | None = None) -> CheckResult:
    """Statically check a code snippet against the indexed library APIs."""
    index = index or default_index()
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return CheckResult([Issue("error", "syntax-error", error.lineno or 0, "",
                                  f"Syntax error: {error.msg}")], [])
    checker = _Checker(index)
    checker.visit(tree)
    issues = sorted(checker.issues, key=lambda i: (i.line, i.severity != "error"))
    return CheckResult(issues, checker.checked)
