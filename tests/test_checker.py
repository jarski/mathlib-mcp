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


def test_submodules_must_be_imported():
    # found by evaluation v1: `import cupy` does not load cupy.cublas (AttributeError on GPU)
    issue = check_snippet("import cupy as cp\ncp.cublas.gemm('N', 'N', a, b)").errors[0]
    assert issue.code == "submodule-not-imported" and issue.symbol == "cupy.cublas"
    assert "import cupy.cublas" in issue.message
    for code in ("import cupy\nimport cupy.cublas\ncupy.cublas.gemm('N', 'N', a, b)",
                 "from cupy import cublas\ncublas.gemm('N', 'N', a, b)",
                 "import cupy as cp\nimport cupy.cublas\ncp.cublas.gemm('N', 'N', a, b)",
                 "import cupy as cp\ncp.lib.stride_tricks.as_strided(x, (2,), (8,))",
                 "import cupyx\ncupyx.scipy.sparse.csr_matrix(x)"):
        assert check_snippet(code).ok, code
    assert "submodule-not-imported" in codes(
        "import cupyx.scipy.sparse as sp\nsp.linalg.cg(A, b)")


def test_guarded_version_fallbacks_are_notes():
    code = ("from cupyx.scipy.sparse.linalg import cg\n"
            "try:\n    x, i = cg(A, b, rtol=1e-8)\nexcept TypeError:\n"
            "    x, i = cg(A, b, tol=1e-8)\n")
    result = check_snippet(code)
    assert result.ok and result.issues[0].severity == "info"
    wrong_guard = code.replace("TypeError", "ValueError")
    assert "unknown-keyword" in codes(wrong_guard)


def test_calling_a_module_is_an_error():
    issue = check_snippet("import nvmath\nnvmath.tensor.contract('ij,jk->ik', a, b)").errors[0]
    assert issue.code == "not-callable"
    assert "nvmath.tensor.binary_contraction" in issue.suggestions
    # a name bound more than once is not treated as an alias (seen in evaluation v1)
    fallback = ("import nvmath\nfn = getattr(nvmath.tensor, 'binary_contraction', None)\n"
                "if fn is None:\n    fn = nvmath.tensor.contract\nfn('ij,jk->ik', a, b)\n")
    assert check_snippet(fallback).ok


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
