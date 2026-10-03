import textwrap
from pathlib import Path

from mathlib_mcp.extract.pysource import SourceTree, describe, format_signature


def write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))


def make_tree(tmp_path: Path) -> SourceTree:
    write(tmp_path, {
        "pkg/__init__.py": """
            from pkg import linalg
            from pkg._core import ndarray
            from numpy import pi

            def __getattr__(name):
                if name == 'lazy':
                    return 1
                raise AttributeError(name)
        """,
        "pkg/linalg/__init__.py": """
            from pkg.linalg._solve import solve
            from ._config import *
            __all__ = ['solve', 'Options', 'Epilog']
        """,
        "pkg/linalg/_solve.py": '''
            import functools
            from pkg._docs import utils

            DOCS = utils.SHARED.copy()
            DOCS.update({"b": "Right-hand side."})

            @utils.docstring_decorator(DOCS)
            def solve(a, b, /, *, rtol=1e-5):
                """Solve a system.

                Args:
                    a: {a}
                    b: {b}
                """

            def _fft(x, *, direction, check):
                """Transform {x}."""

            fft = functools.wraps(_fft)(functools.partial(_fft, direction=1, check=True))
            fft.__doc__ = fft.__doc__.format(**DOCS)
        ''',
        "pkg/linalg/_config.py": """
            from dataclasses import dataclass
            from pkg._bindings import Epilogue as Epilog
            __all__ = ['Options', 'Epilog']

            @dataclass
            class Options:
                inplace: bool = False
                compute_type: int | None = None
        """,
        "pkg/_bindings.pyi": """
            import enum
            from typing import ClassVar
            class Epilogue(enum.IntEnum):
                RELU: ClassVar[Epilogue] = ...
                BIAS: ClassVar[Epilogue] = ...
        """,
        "pkg/_docs/__init__.py": "",
        "pkg/_docs/utils.py": """
            SHARED = {"a": "The matrix.".replace("matrix", "square matrix"), "x": "the input"}
        """,
        "pkg/_core.pyx": """
            cdef class ndarray:
                pass
        """,
    })
    return SourceTree({"pkg": tmp_path / "pkg"})


def test_reexport_resolves_to_definition_with_signature(tmp_path):
    info = describe(make_tree(tmp_path), "pkg.linalg", "solve")
    assert info.kind == "function"
    assert info.defined_in == "pkg.linalg._solve"
    assert format_signature("solve", info.params) == "solve(a, b, /, *, rtol=1e-05)"


def test_docstring_templates_are_expanded(tmp_path):
    doc = describe(make_tree(tmp_path), "pkg.linalg", "solve").doc
    assert "a: The square matrix." in doc
    assert "b: Right-hand side." in doc


def test_partial_drops_bound_keywords_and_formats_doc(tmp_path):
    tree = make_tree(tmp_path)
    info = describe(tree, "pkg.linalg._solve", "fft")
    assert info.kind == "function"
    assert [p.name for p in info.params] == ["x"]
    assert info.doc == "Transform the input."


def test_star_import_alias_and_stub_enum_members(tmp_path):
    tree = make_tree(tmp_path)
    epilog = describe(tree, "pkg.linalg", "Epilog")
    assert epilog.kind == "class" and epilog.members_exact
    assert {"RELU", "BIAS"} <= set(epilog.members)
    options = describe(tree, "pkg.linalg", "Options")
    assert [p.name for p in options.params] == ["inplace", "compute_type"]


def test_namespace_names_and_exactness(tmp_path):
    tree = make_tree(tmp_path)
    assert {"linalg", "ndarray", "pi", "lazy"} <= tree.names("pkg")
    assert tree.public_names("pkg.linalg") == ["solve", "Options", "Epilog"]
    assert tree.is_exact("pkg.linalg")
    assert not tree.is_exact("pkg._core")  # Cython modules are scanned approximately
    assert describe(tree, "pkg", "pi").kind == "external"
