import pytest

from mathlib_mcp.cards import load_cards
from mathlib_mcp.checker import check_snippet


def codes(code: str) -> list[str]:
    return [issue.code for issue in check_snippet(code).errors]


def test_valid_code_passes():
    result = check_snippet("""
import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse.linalg import cg
import nvmath

A = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(10, 10), format='csr')
x, info = cg(A, cp.ones(10), rtol=1e-8)
la = cp.linalg
w, v = la.eigh(cp.eye(3))
r = nvmath.linalg.advanced.matmul(cp.ones((4, 4)), cp.ones((4, 4)),
                                  epilog=nvmath.linalg.advanced.MatmulEpilog.RELU)
""")
    assert result.ok, result.to_text()
    assert "cupy.linalg.eigh" in result.checked


@pytest.mark.parametrize(("code", "expected"), [
    ("import cupy as cp\ncp.fft.fft_shift(x)", "unknown-symbol"),
    ("from cupy.linalg import lu_factor", "unknown-symbol"),
    ("import cupyx.scipy.sparse.linalgs", "unknown-module"),
    ("import cupyx.scipy.sparse.linalg as s\ns.cg(A, b, tol=1e-8)", "unknown-keyword"),
    ("import cupy as cp\ncp.linalg.cholesky(a, upper=True)", "unknown-keyword"),
    ("from cupyx.scipy.sparse.linalg import eigsh\neigsh(A, 3, 'SA')", "too-many-positional"),
    ("import nvmath\nnvmath.linalg.advanced.matmul(a=x, b=y)", "positional-only"),
    ("import nvmath\nnvmath.linalg.advanced.MatmulEpilog.RELU_GELU", "unknown-attribute"),
    ("import cupy as cp\ncp.linalg.solve(a)", "missing-argument"),
    ("def broken(:\n  pass", "syntax-error"),
])
def test_detects_api_misuse(code, expected):
    assert expected in codes(code)


def test_suggestions_point_to_real_apis():
    issue = check_snippet("from cupy.linalg import lu_factor").errors[0]
    assert "cupyx.scipy.linalg.lu_factor" in issue.suggestions
    issue = check_snippet("import cupyx.scipy.sparse.linalg as s\ns.cg(A, b, tol=1)").errors[0]
    assert issue.suggestions[0] == "rtol"


def test_unindexed_namespaces_are_not_errors():
    result = check_snippet("import nvmath\nnvmath.bindings.cublas.foo()")
    assert result.ok
    assert any(issue.code == "unverified" for issue in result.issues)


def test_unrelated_code_is_ignored():
    assert check_snippet("import numpy as np\nnp.linalg.does_not_exist()").ok


@pytest.mark.parametrize("card_id", sorted(k for k, c in load_cards().items() if c.example))
def test_curated_examples_pass_the_checker(card_id):
    """Guards against false positives: verified examples must never be flagged."""
    result = check_snippet(load_cards()[card_id].example)
    assert result.ok, result.to_text()
