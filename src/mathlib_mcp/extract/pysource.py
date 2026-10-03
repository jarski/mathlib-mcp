"""Static (import-free) analysis of Python package sources.

GPU libraries such as CuPy and nvmath-python cannot be imported on a machine without
CUDA, so API facts are read from the source tree with :mod:`ast` instead of
:mod:`inspect`. The analysis follows re-exports (``from x import y``), star imports
with ``__all__``, simple aliases (``MatmulEpilog = cublaslt.Epilogue``) and lazy
``__getattr__`` names, which is enough to answer "does ``cupy.linalg.solve`` exist and
what is its signature?" for the public API.

Cython ``.pyx`` modules are scanned with regular expressions; their namespaces are
marked as *inexact* because the scan can miss names.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

PARAM_KINDS = ("positional_only", "positional_or_keyword", "var_positional", "keyword_only",
               "var_keyword")


@dataclass
class Param:
    name: str
    kind: str
    default: str | None = None
    annotation: str | None = None


@dataclass
class Binding:
    """What a name in a module namespace is bound to."""

    kind: str  # def | class | import | module | assign | value | pyx_def | pyx_class
    module: str  # module whose namespace holds the binding
    node: ast.AST | None = None
    target: str | None = None  # import target: "pkg.mod" or "pkg.mod:name"
    lineno: int | None = None
    members: list[str] | None = None  # pyx classes: names assigned in the class body


@dataclass
class ModuleInfo:
    name: str
    path: Path | None
    is_package: bool
    bindings: dict[str, Binding] = field(default_factory=dict)
    all: list[str] | None = None
    star_imports: list[str] = field(default_factory=list)
    lazy_names: set[str] = field(default_factory=set)
    exact: bool = True
    tree: ast.Module | None = None


class SourceTree:
    """Index over one or more top-level packages found in source form."""

    def __init__(self, roots: dict[str, Path]):
        self.roots = {name: Path(path) for name, path in roots.items()}
        self._modules: dict[str, ModuleInfo | None] = {}

    # ----------------------------------------------------------------- files

    def _base(self, modname: str) -> Path | None:
        parts = modname.split(".")
        root = self.roots.get(parts[0])
        return None if root is None else root.joinpath(*parts[1:])

    def find(self, modname: str) -> tuple[Path, bool] | None:
        """Return ``(path, is_package)`` for a module, or ``None`` if it is not in the tree."""
        base = self._base(modname)
        if base is None:
            return None
        for candidate, is_pkg in ((base / "__init__.py", True), (base.with_suffix(".py"), False),
                                  (base / "__init__.pyi", True), (base.with_suffix(".pyi"), False),
                                  (base / "__init__.pyx", True), (base.with_suffix(".pyx"), False)):
            if candidate.is_file():
                return candidate, is_pkg
        return None

    def is_module(self, modname: str) -> bool:
        return self.find(modname) is not None

    def submodules(self, modname: str) -> list[str]:
        """Public direct child modules of a package."""
        found = self.find(modname)
        if found is None or not found[1]:
            return []
        directory = found[0].parent
        names = set()
        for child in directory.iterdir():
            if child.name.startswith(("_", ".")) or child.name in ("tests", "testing"):
                continue
            if child.is_dir() and any((child / f"__init__{suffix}").is_file()
                                      for suffix in (".py", ".pyi", ".pyx")):
                names.add(child.name)
            elif child.suffix in (".py", ".pyi", ".pyx"):
                names.add(child.stem)
        return sorted(f"{modname}.{name}" for name in names)

    # --------------------------------------------------------------- parsing

    def module(self, modname: str) -> ModuleInfo | None:
        if modname not in self._modules:
            found = self.find(modname)
            if found is None:
                self._modules[modname] = None
            else:
                path, is_pkg = found
                info = ModuleInfo(modname, path, is_pkg)
                if path.suffix == ".pyx":
                    _scan_pyx(info, path.read_text(encoding="utf-8", errors="replace"))
                else:
                    _scan_py(info, path.read_text(encoding="utf-8", errors="replace"))
                self._modules[modname] = info
        return self._modules[modname]

    # ------------------------------------------------------------ resolution

    def resolve(self, modname: str, name: str, _seen: frozenset = frozenset()
                ) -> tuple[ModuleInfo | None, Binding] | None:
        """Follow re-exports and aliases of ``modname.name`` to where it is defined.

        Returns ``(module, binding)``. ``module`` is ``None`` when the name comes from a
        package outside the tree (e.g. ``numpy``); the binding then records the target.
        """
        key = (modname, name)
        if key in _seen:
            return None
        seen = _seen | {key}
        info = self.module(modname)
        if info is None:
            return None
        binding = info.bindings.get(name)
        if binding is None:
            for star in info.star_imports:
                if name in self.public_names(star):
                    return self.resolve(star, name, seen)
            if self.is_module(f"{modname}.{name}"):
                return None, Binding("module", modname, target=f"{modname}.{name}")
            if name in info.lazy_names:
                return info, Binding("value", modname)
            return None
        if binding.kind == "import":
            assert binding.target is not None
            if ":" not in binding.target:
                return None, Binding("module", modname, target=binding.target)
            target_mod, target_name = binding.target.split(":")
            if self.is_module(f"{target_mod}.{target_name}"):
                return None, Binding("module", modname, target=f"{target_mod}.{target_name}")
            if self.module(target_mod) is None:  # external package, e.g. numpy
                return None, binding
            return self.resolve(target_mod, target_name, seen) or (None, binding)
        if binding.kind == "assign" and isinstance(binding.node, (ast.Name, ast.Attribute)):
            aliased = self._resolve_expr(info, binding.node, seen)
            if aliased is not None:
                return aliased
        return info, binding

    def _resolve_expr(self, info: ModuleInfo, node: ast.expr, seen: frozenset):
        """Resolve ``name`` or ``mod.attr.attr`` expressions inside a module."""
        chain = _attr_chain(node)
        if chain is None:
            return None
        resolved = self.resolve(info.name, chain[0], seen)
        for part in chain[1:]:
            if resolved is None:
                return None
            _, binding = resolved
            if binding.kind != "module" or binding.target is None:
                return None
            resolved = self.resolve(binding.target, part, seen)
        return resolved

    def names(self, modname: str) -> set[str]:
        """Every name that attribute access on the module can reach (public or not)."""
        info = self.module(modname)
        if info is None:
            return set()
        result = set(info.bindings) | info.lazy_names
        for star in info.star_imports:
            result |= set(self.public_names(star))
        result |= {sub.rsplit(".", 1)[1] for sub in self.submodules(modname)}
        return result

    def public_names(self, modname: str) -> list[str]:
        """Public API of a module: ``__all__`` if defined, else names without underscore."""
        info = self.module(modname)
        if info is None:
            return []
        if info.all is not None:
            return list(info.all)
        return sorted(n for n in self.names(modname) if not n.startswith("_"))

    def is_exact(self, modname: str, _seen: frozenset = frozenset()) -> bool:
        """Whether :meth:`names` is believed to be complete for the module."""
        info = self.module(modname)
        if info is None or not info.exact or modname in _seen:
            return False
        return all(self.is_exact(star, _seen | {modname}) for star in info.star_imports)

    # ----------------------------------------------------- docstring support

    def docstring(self, info: ModuleInfo, node: ast.AST, owner: ast.ClassDef | None = None
                  ) -> str | None:
        """Docstring of a def/class, with ``docstring_decorator`` templates expanded.

        ``owner`` is the enclosing class of a method; a class-level decorator also
        formats the docstrings of its methods.
        """
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return None
        doc = ast.get_docstring(node, clean=True)
        if doc is None:
            return None
        decorators = list(node.decorator_list) + (owner.decorator_list if owner else [])
        for deco in decorators:
            if (isinstance(deco, ast.Call) and _call_name(deco).endswith("docstring_decorator")
                    and deco.args):
                mapping = self._eval_doc_map(info, deco.args[0])
                if mapping is not None:
                    doc = doc.format_map(_KeepMissing(mapping))
        return doc

    def _eval_doc_map(self, info: ModuleInfo, node: ast.expr, depth: int = 0) -> dict | None:
        """Evaluate the small subset of Python used to build docstring template maps."""
        if depth > 5:
            return None
        if isinstance(node, ast.Dict):
            result = {}
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    text = self._eval_str(info, value)
                    if text is not None:
                        result[key.value] = text
            return result
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "copy":
                return self._eval_doc_map(info, node.func.value, depth + 1)
            return None
        chain = _attr_chain(node)
        if chain is None:
            return None
        if len(chain) == 1:
            return self._eval_module_dict(info, chain[0], depth)
        resolved = self._resolve_expr(info, node.value, frozenset())  # type: ignore[attr-defined]
        if resolved is None or resolved[1].kind != "module" or resolved[1].target is None:
            return None
        target = self.module(resolved[1].target)
        return None if target is None else self._eval_module_dict(target, chain[-1], depth)

    def _eval_module_dict(self, info: ModuleInfo, name: str, depth: int) -> dict | None:
        if info.tree is None:
            return None
        value: dict | None = None
        for stmt in info.tree.body:
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == name):
                value = self._eval_doc_map(info, stmt.value, depth + 1)
            elif (value is not None and isinstance(stmt, ast.Expr)
                  and isinstance(stmt.value, ast.Call)
                  and isinstance(stmt.value.func, ast.Attribute)
                  and stmt.value.func.attr == "update"
                  and isinstance(stmt.value.func.value, ast.Name)
                  and stmt.value.func.value.id == name and stmt.value.args):
                update = self._eval_doc_map(info, stmt.value.args[0], depth + 1)
                if update:
                    value.update(update)
        return value

    def _eval_str(self, info: ModuleInfo, node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self._eval_str(info, node.left), self._eval_str(info, node.right)
            return None if left is None or right is None else left + right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            base = self._eval_str(info, node.func.value)
            args = [a.value for a in node.args if isinstance(a, ast.Constant)]
            if base is None or len(args) != len(node.args):
                return None
            if node.func.attr in ("replace", "strip", "lstrip", "rstrip"):
                return getattr(base, node.func.attr)(*args)
            return None
        if isinstance(node, ast.Name) and info.tree is not None:
            for stmt in info.tree.body:
                if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id == node.id):
                    return self._eval_str(info, stmt.value)
        return None


class _KeepMissing(dict):
    """``format_map`` helper that leaves unknown ``{placeholders}`` untouched."""

    def __missing__(self, key):
        return "{" + key + "}"


# --------------------------------------------------------------------- scanners


def _scan_py(info: ModuleInfo, text: str) -> None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        info.exact = False
        return
    info.tree = tree
    _scan_statements(info, tree.body)


def _scan_statements(info: ModuleInfo, body: list[ast.stmt]) -> None:
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info.bindings[stmt.name] = Binding("def", info.name, stmt, lineno=stmt.lineno)
            if stmt.name == "__getattr__":
                info.lazy_names |= _getattr_literals(stmt, info)
        elif isinstance(stmt, ast.ClassDef):
            info.bindings[stmt.name] = Binding("class", info.name, stmt, lineno=stmt.lineno)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.asname:
                    info.bindings[alias.asname] = Binding("import", info.name, target=alias.name,
                                                          lineno=stmt.lineno)
                else:
                    top = alias.name.split(".")[0]
                    info.bindings[top] = Binding("import", info.name, target=top,
                                                 lineno=stmt.lineno)
        elif isinstance(stmt, ast.ImportFrom):
            source = _absolute_module(info, stmt.module, stmt.level)
            for alias in stmt.names:
                if alias.name == "*":
                    info.star_imports.append(source)
                else:
                    info.bindings[alias.asname or alias.name] = Binding(
                        "import", info.name, target=f"{source}:{alias.name}", lineno=stmt.lineno)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                if not isinstance(target, ast.Name) or stmt.value is None:
                    continue
                if target.id == "__all__":
                    names = _string_list(stmt.value)
                    if names is not None:
                        if isinstance(stmt, ast.AugAssign) and info.all is not None:
                            info.all.extend(names)
                        else:
                            info.all = names
                else:
                    info.bindings[target.id] = Binding("assign", info.name, stmt.value,
                                                       lineno=stmt.lineno)
        elif isinstance(stmt, ast.If):
            _scan_statements(info, stmt.body)
            _scan_statements(info, stmt.orelse)
        elif isinstance(stmt, ast.Try):
            _scan_statements(info, stmt.body)
            for handler in stmt.handlers:
                _scan_statements(info, handler.body)
            _scan_statements(info, stmt.orelse)
            _scan_statements(info, stmt.finalbody)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            # __all__.extend([...]) / __all__ += ...
            func = stmt.value.func
            if (isinstance(func, ast.Attribute) and func.attr == "extend"
                    and isinstance(func.value, ast.Name) and func.value.id == "__all__"
                    and info.all is not None and stmt.value.args):
                names = _string_list(stmt.value.args[0])
                if names:
                    info.all.extend(names)


_PYX_DEF = re.compile(r"^(?:cpdef|def)\s+(?:[\w.\[\]]+\s+)*?(\w+)\s*\(", re.M)
_PYX_CLASS = re.compile(r"^(?:cdef\s+)?class\s+(\w+)\s*(?:\(|:)", re.M)
_PYX_ASSIGN = re.compile(r"^(\w+)\s*=", re.M)
_PYX_FROM = re.compile(r"^from\s+([\w.]+)\s+(?:c)?import\s+(.+)$", re.M)
_PYX_IMPORT = re.compile(r"^import\s+([\w.]+)(?:\s+as\s+(\w+))?\s*$", re.M)
_PYX_MEMBER = re.compile(r"^    (\w+)\s*=", re.M)


def _scan_pyx(info: ModuleInfo, text: str) -> None:
    info.exact = False
    for match in _PYX_FROM.finditer(text):
        source = _absolute_module(info, match.group(1).lstrip("."),
                                  len(match.group(1)) - len(match.group(1).lstrip(".")))
        for part in match.group(2).strip("()").split(","):
            pieces = part.split()
            if not pieces or pieces[0] == "*":
                continue
            name = pieces[-1] if len(pieces) == 3 and pieces[1] == "as" else pieces[0]
            info.bindings.setdefault(name, Binding("import", info.name,
                                                   target=f"{source}:{pieces[0]}"))
    for match in _PYX_IMPORT.finditer(text):
        name = match.group(2) or match.group(1).split(".")[0]
        info.bindings.setdefault(name, Binding("import", info.name,
                                               target=match.group(2) and match.group(1) or name))
    for match in _PYX_ASSIGN.finditer(text):
        info.bindings[match.group(1)] = Binding("value", info.name)
    for match in _PYX_DEF.finditer(text):
        info.bindings[match.group(1)] = Binding("pyx_def", info.name,
                                                lineno=text.count("\n", 0, match.start()) + 1)
    for match in _PYX_CLASS.finditer(text):
        body_start = match.end()
        body_end = re.compile(r"^\S", re.M).search(text, body_start)
        body = text[body_start: body_end.start() if body_end else len(text)]
        members = [m.group(1) for m in _PYX_MEMBER.finditer(body)]
        info.bindings[match.group(1)] = Binding("pyx_class", info.name, members=members,
                                                lineno=text.count("\n", 0, match.start()) + 1)


# ---------------------------------------------------------------------- helpers


def _absolute_module(info: ModuleInfo, module: str | None, level: int) -> str:
    if level == 0:
        return module or ""
    package = info.name if info.is_package else info.name.rsplit(".", 1)[0]
    for _ in range(level - 1):
        package = package.rsplit(".", 1)[0]
    return f"{package}.{module}" if module else package


def _string_list(node: ast.expr) -> list[str] | None:
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [e.value for e in node.elts if isinstance(e, ast.Constant)]
        if len(values) == len(node.elts) and all(isinstance(v, str) for v in values):
            return list(values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _string_list(node.left), _string_list(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _getattr_literals(func: ast.FunctionDef, info: ModuleInfo) -> set[str]:
    """Names a module-level ``__getattr__`` serves lazily (``if key == 'cusparse'``)."""
    names: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Compare):
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                if isinstance(op, ast.Eq) and isinstance(comparator, ast.Constant) \
                        and isinstance(comparator.value, str):
                    names.add(comparator.value)
                elif isinstance(op, ast.In) and isinstance(comparator, ast.Name):
                    binding = info.bindings.get(comparator.id)
                    if binding is not None and isinstance(binding.node, ast.expr):
                        names |= set(_string_list(binding.node) or [])
    return names


def _attr_chain(node: ast.expr) -> list[str] | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


def _call_name(call: ast.Call) -> str:
    chain = _attr_chain(call.func)
    return ".".join(chain) if chain else ""


# ------------------------------------------------------------------- signatures


def function_params(node: ast.FunctionDef | ast.AsyncFunctionDef, *, drop_self: bool = False
                    ) -> list[Param]:
    args = node.args
    params: list[Param] = []
    positional = args.posonlyargs + args.args
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
    defaults += list(args.defaults)
    for index, (arg, default) in enumerate(zip(positional, defaults, strict=True)):
        kind = "positional_only" if index < len(args.posonlyargs) else "positional_or_keyword"
        params.append(_param(arg, kind, default))
    if args.vararg:
        params.append(_param(args.vararg, "var_positional", None))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        params.append(_param(arg, "keyword_only", default))
    if args.kwarg:
        params.append(_param(args.kwarg, "var_keyword", None))
    if drop_self and params and params[0].name in ("self", "cls"):
        params = params[1:]
    return params


def dataclass_params(node: ast.ClassDef) -> list[Param] | None:
    """Constructor parameters of a ``@dataclass`` (annotated class-body fields)."""
    if not any(_decorator_name(d).split(".")[-1] == "dataclass" for d in node.decorator_list):
        return None
    params = []
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            annotation = ast.unparse(stmt.annotation)
            if annotation.startswith(("ClassVar", "typing.ClassVar")):
                continue
            default = None if stmt.value is None else _default_text(stmt.value)
            params.append(Param(stmt.target.id, "positional_or_keyword", default, annotation))
    return params


def class_params(node: ast.ClassDef) -> list[Param] | None:
    params = dataclass_params(node)
    if params is not None:
        return params
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "__init__":
            return function_params(stmt, drop_self=True)
    return None


def class_members(node: ast.ClassDef) -> list[str]:
    members = []
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not stmt.name.startswith("_") or stmt.name in ("__call__",):
                members.append(stmt.name)
        elif isinstance(stmt, ast.Assign):
            members += [t.id for t in stmt.targets
                        if isinstance(t, ast.Name) and not t.id.startswith("_")]
        elif (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
              and not stmt.target.id.startswith("_")):
            members.append(stmt.target.id)
    return members


def class_bases(node: ast.ClassDef) -> list[str]:
    return [ast.unparse(base) for base in node.bases]


def format_signature(name: str, params: list[Param]) -> str:
    parts: list[str] = []
    seen_kw_marker = False
    for index, param in enumerate(params):
        if param.kind == "keyword_only" and not seen_kw_marker:
            parts.append("*")
            seen_kw_marker = True
        if param.kind == "var_positional":
            seen_kw_marker = True
            parts.append(f"*{param.name}")
        elif param.kind == "var_keyword":
            parts.append(f"**{param.name}")
        else:
            parts.append(param.name if param.default is None
                         else f"{param.name}={param.default}")
        is_last_posonly = (param.kind == "positional_only"
                           and (index + 1 == len(params)
                                or params[index + 1].kind != "positional_only"))
        if is_last_posonly:
            parts.append("/")
    return f"{name}({', '.join(parts)})"


def _param(arg: ast.arg, kind: str, default: ast.expr | None) -> Param:
    return Param(
        name=arg.arg,
        kind=kind,
        default=None if default is None else _default_text(default),
        annotation=None if arg.annotation is None else ast.unparse(arg.annotation),
    )


def _default_text(node: ast.expr) -> str:
    text = ast.unparse(node)
    return text if len(text) <= 60 else "..."


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    chain = _attr_chain(node)
    return ".".join(chain) if chain else ""


# ------------------------------------------------------------------ description


@dataclass
class SymbolInfo:
    """Everything statically known about one public name."""

    qualname: str
    kind: str  # function | class | module | value | external
    defined_in: str | None = None
    lineno: int | None = None
    params: list[Param] | None = None
    doc: str | None = None
    members: list[str] | None = None
    members_exact: bool = False
    bases: list[str] | None = None
    target: str | None = None


# Bases whose own attributes are irrelevant for API checks (enum members are declared
# in the subclass body).
_TRANSPARENT_BASES = {"object", "enum.IntEnum", "IntEnum", "enum.Enum", "Enum", "enum.IntFlag",
                      "IntFlag", "ABC", "abc.ABC", "Generic", "typing.Generic", "Protocol"}


def describe(tree: SourceTree, modname: str, name: str) -> SymbolInfo | None:
    resolved = tree.resolve(modname, name)
    if resolved is None:
        return None
    info, binding = resolved
    qualname = f"{modname}.{name}"
    where = dict(defined_in=binding.module, lineno=binding.lineno)
    if binding.kind == "module":
        return SymbolInfo(qualname, "module", target=binding.target)
    if binding.kind == "import":
        return SymbolInfo(qualname, "external", target=binding.target)
    if binding.kind == "def" and info is not None:
        node = binding.node
        doc = tree.docstring(info, node)
        return SymbolInfo(qualname, "function", params=function_params(node), doc=doc, **where)
    if binding.kind == "class" and info is not None:
        node = binding.node
        members, exact = _inherited_members(tree, info, node)
        return SymbolInfo(qualname, "class", params=_inherited_params(tree, info, node),
                          doc=tree.docstring(info, node), members=members,
                          members_exact=exact, bases=class_bases(node), **where)
    if binding.kind == "pyx_def":
        return SymbolInfo(qualname, "function", **where)
    if binding.kind == "pyx_class":
        return SymbolInfo(qualname, "class", members=binding.members, **where)
    if binding.kind == "assign" and info is not None and isinstance(binding.node, ast.Call):
        wrapped = _describe_callable_assignment(tree, info, name, binding.node)
        if wrapped is not None:
            params, doc = wrapped
            return SymbolInfo(qualname, "function", params=params, doc=doc, **where)
    return SymbolInfo(qualname, "value", **where)


def _inherited_params(tree: SourceTree, info: ModuleInfo, node: ast.ClassDef, depth: int = 0
                      ) -> list[Param] | None:
    """Constructor parameters, following base classes when ``__init__`` is inherited."""
    params = class_params(node)
    if params is not None or depth > 6:
        return params
    for base in node.bases:
        if ast.unparse(base) in _TRANSPARENT_BASES:
            continue
        resolved = tree._resolve_expr(info, base, frozenset())
        if resolved is not None and resolved[0] is not None and resolved[1].kind == "class":
            params = _inherited_params(tree, resolved[0], resolved[1].node, depth + 1)
            if params is not None:
                return params
    return None


def _inherited_members(tree: SourceTree, info: ModuleInfo, node: ast.ClassDef, depth: int = 0
                       ) -> tuple[list[str], bool]:
    members = class_members(node)
    exact = True
    if depth > 6:
        return members, False
    for base in node.bases:
        text = ast.unparse(base)
        if text in _TRANSPARENT_BASES:
            continue
        resolved = tree._resolve_expr(info, base, frozenset())
        if resolved is None or resolved[0] is None or resolved[1].kind != "class":
            exact = False
            continue
        base_members, base_exact = _inherited_members(tree, resolved[0], resolved[1].node,
                                                      depth + 1)
        members += [m for m in base_members if m not in members]
        exact = exact and base_exact
    return members, exact


def _describe_callable_assignment(tree: SourceTree, info: ModuleInfo, name: str, call: ast.Call
                                  ) -> tuple[list[Param] | None, str | None] | None:
    """Callables created by assignment rather than ``def``.

    Handles ``x = SomeFactory(..., doc='''x(a, b, /)\n...''')`` (CuPy gufuncs, whose
    docstring starts with the signature) and ``x = functools.wraps(f)(functools.partial(f,
    k=v))`` (nvmath-python FFT functions).
    """
    for keyword in call.keywords:
        if keyword.arg == "doc":
            doc = tree._eval_str(info, keyword.value)
            if doc is not None:
                import inspect

                doc = inspect.cleandoc(doc)
                return _signature_from_doc(doc), doc
    if (isinstance(call.func, ast.Call) and _call_name(call.func).endswith("wraps")
            and call.func.args and isinstance(call.func.args[0], ast.Name)):
        resolved = tree.resolve(info.name, call.func.args[0].id)
        if resolved is None or resolved[0] is None or resolved[1].kind != "def":
            return None
        func_info, binding = resolved
        params = function_params(binding.node)
        bound = set()
        if call.args and isinstance(call.args[0], ast.Call) \
                and _call_name(call.args[0]).endswith("partial"):
            bound = {k.arg for k in call.args[0].keywords if k.arg}
        params = [p for p in params if p.name not in bound]
        doc = _assigned_doc(tree, info, name, ast.get_docstring(binding.node, clean=True))
        return params, doc
    return None


def _assigned_doc(tree: SourceTree, info: ModuleInfo, name: str, wrapped_doc: str | None
                  ) -> str | None:
    """Apply ``name.__doc__ = ...`` statements that follow a ``functools.wraps`` alias.

    Supports ``name.__doc__ = name.__doc__.format(**MAP)`` (keeps the wrapped docstring)
    and ``name.__doc__ = \"\"\"...\"\"\".format(**MAP)`` (replaces it).
    """
    import inspect

    if info.tree is None:
        return wrapped_doc
    doc = wrapped_doc
    for node in info.tree.body:
        # also look inside `if fft.__doc__ is not None:` guards
        candidates = node.body if isinstance(node, ast.If) else [node]
        for stmt in candidates:
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and _attr_chain(stmt.targets[0]) == [name, "__doc__"]):
                continue
            value = stmt.value
            doc_map = None
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) \
                    and value.func.attr == "format":
                if value.keywords and value.keywords[0].arg is None:
                    doc_map = tree._eval_doc_map(info, value.keywords[0].value)
                value = value.func.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                doc = inspect.cleandoc(value.value)
            if doc is not None and doc_map is not None:
                doc = doc.format_map(_KeepMissing(doc_map))
    return doc


_DOC_SIGNATURE = re.compile(r"^\s*([\w.]+)\((.*)\)\s*$")


def _signature_from_doc(doc: str) -> list[Param] | None:
    """Parse a ``name(a, b, /, out=None, **kwargs)`` first docstring line."""
    first = doc.strip().splitlines()[0] if doc.strip() else ""
    match = _DOC_SIGNATURE.match(first.replace("\\*", "*"))
    if match is None:
        return None
    try:
        func = ast.parse(f"def _f({match.group(2)}): pass").body[0]
    except SyntaxError:
        return None
    return function_params(func)  # type: ignore[arg-type]
