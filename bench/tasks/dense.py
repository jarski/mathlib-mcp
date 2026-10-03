"""Dense linear algebra: cuSOLVER (and cuBLAS trsm) through CuPy."""

import numpy as np
import scipy.linalg

from bench.task import Case, Task, assert_close


def _spd(rng, n):
    x = rng.standard_normal((n, n))
    return x @ x.T + n * np.eye(n)


def _gaussian_reference(K, Y):
    c, low = scipy.linalg.cho_factor(K)
    alpha = scipy.linalg.cho_solve((c, low), Y)
    logdet = 2.0 * np.sum(np.log(np.diag(c)))
    return alpha, logdet


def _lstsq_inputs(rng, case):
    A = rng.standard_normal((300, 12))
    if case == 1:  # rank-deficient: one column duplicates another
        A[:, 5] = A[:, 2]
    return A, rng.standard_normal(300)


def _lstsq_reference(A, b):
    x, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    return x, int(rank)


def _lstsq_check(actual, expected, inputs):
    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        raise AssertionError("expected a tuple (x, rank)")
    assert_close(actual[0], expected[0], 1e-6, 1e-8)
    if int(np.asarray(actual[1])) != expected[1]:
        raise AssertionError(f"rank {actual[1]} != expected {expected[1]}")


def _eig_key(values):
    values = np.asarray(values)
    return np.sort_complex(np.round(values.real, 6) + 1j * np.round(values.imag, 6))


def _eig_check(actual, expected, inputs):
    arr = np.asarray(actual)
    if arr.dtype != np.complex128:
        raise AssertionError(f"expected complex128 output, got {arr.dtype}")
    assert_close(_eig_key(arr), _eig_key(expected), 1e-6, 1e-6)


def _topk_reference(A, k):
    w = np.linalg.eigvalsh(A)[::-1][:k]
    return w


def _topk_check(actual, expected, inputs):
    A, k = inputs
    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        raise AssertionError("expected a tuple (values, vectors)")
    values, vectors = np.asarray(actual[0]), np.asarray(actual[1])
    assert_close(values, expected, 1e-8, 1e-8)
    if vectors.shape != (A.shape[0], k):
        raise AssertionError(f"vectors shape {vectors.shape} != {(A.shape[0], k)}")
    residual = np.linalg.norm(A @ vectors - vectors * values) / np.linalg.norm(A)
    if residual > 1e-8:
        raise AssertionError(f"eigenvector residual too large: {residual:.3g}")
    assert_close(vectors.T @ vectors, np.eye(k), 1e-8, 1e-8)


def _low_rank_reference(A, k):
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    return (u[:, :k] * s[:k]) @ vh[:k]


def _singular_inputs(rng, case):
    A = rng.standard_normal((64, 64)) + 8 * np.eye(64)
    if case == 1:
        # A zero column gives an exactly-zero pivot in any LU ordering. (Two identical
        # rows are not enough: blocked LU can leave a ~1e-15 pivot instead of 0.)
        A[:, 40] = 0.0
    return A, rng.standard_normal(64)


def _upper_with_garbage(rng, case):
    n = 80
    U = np.triu(rng.standard_normal((n, n))) + n * np.eye(n)
    U += np.tril(rng.standard_normal((n, n)) * 100, k=-1)  # garbage below the diagonal
    return U, rng.standard_normal((n, 3))


TASKS = [
    Task(
        id="batched_solve_vectors",
        family="cuSOLVER",
        difficulty="medium",
        prompt="""\
Implement `batched_solve(A, y)`.

`A` is a float64 CuPy array of shape (B, M, M) (every matrix is invertible) and `y`
is float64 of shape (B, M). Return `x` of shape (B, M) such that `A[i] @ x[i] == y[i]`
for every i. Solve all systems with one batched call (no Python loop over the batch).""",
        entry_point="batched_solve",
        solution="""\
import cupy as cp


def batched_solve(A, y):
    # b must be (..., M, K) for stacked matrices, so add and remove a K=1 axis
    return cp.linalg.solve(A, y[..., None])[..., 0]
""",
        make_inputs=lambda rng, case: (rng.standard_normal((64, 12, 12)) + 6 * np.eye(12),
                                       rng.standard_normal((64, 12))),
        reference=lambda A, y: np.linalg.solve(A, y[..., None])[..., 0],
        rtol=1e-8,
        atol=1e-10,
        trap="NumPy 2 / CuPy 14 reject a stacked (B, M) right-hand side; it must be "
             "(B, M, 1).",
        naive="""\
import cupy as cp


def batched_solve(A, y):
    return cp.linalg.solve(A, y)
""",
    ),
    Task(
        id="cholesky_solve_logdet",
        family="cuSOLVER",
        difficulty="medium",
        prompt="""\
Implement `gaussian_terms(K, Y)`.

`K` is a symmetric positive-definite float64 CuPy array (N, N) and `Y` is float64
(N, R). Return a tuple `(alpha, logdet)` where `alpha = inv(K) @ Y` (shape (N, R)) and
`logdet = log(det(K))` (a 0-d CuPy array or a Python float). Use a Cholesky
factorization of K; do not form the inverse and do not call det.""",
        entry_point="gaussian_terms",
        solution="""\
import cupy as cp
from cupyx.scipy.linalg import solve_triangular


def gaussian_terms(K, Y):
    L = cp.linalg.cholesky(K)  # lower triangular, K = L @ L.T
    z = solve_triangular(L, Y, lower=True)
    alpha = solve_triangular(L, z, lower=True, trans='T')
    logdet = 2.0 * cp.sum(cp.log(cp.diagonal(L)))
    return alpha, logdet
""",
        make_inputs=lambda rng, case: (_spd(rng, 200), rng.standard_normal((200, 4))),
        reference=_gaussian_reference,
        rtol=1e-8,
        atol=1e-10,
        required_api=("cupy.linalg.cholesky",),
        trap="CuPy returns the lower factor; the second solve needs the transpose.",
        naive="""\
import cupy as cp
from cupyx.scipy.linalg import solve_triangular


def gaussian_terms(K, Y):
    U = cp.linalg.cholesky(K)  # wrongly assumed upper, K = U.T @ U
    z = solve_triangular(U, Y, trans='T')
    alpha = solve_triangular(U, z)
    return alpha, 2.0 * cp.sum(cp.log(cp.diagonal(U)))
""",
    ),
    Task(
        id="lstsq_with_rank",
        family="cuSOLVER",
        difficulty="easy",
        prompt="""\
Implement `fit(A, b)`.

`A` is a float64 CuPy array (m, n) with m > n that may be rank-deficient, and `b` is
float64 (m,). Return a tuple `(x, rank)`: `x` is the minimum-norm least-squares
solution (as computed by `numpy.linalg.lstsq(A, b, rcond=None)`) and `rank` is the
numerical rank of A as a Python int.""",
        entry_point="fit",
        solution="""\
import cupy as cp


def fit(A, b):
    x, _, rank, _ = cp.linalg.lstsq(A, b, rcond=None)
    return x, int(rank)
""",
        make_inputs=_lstsq_inputs,
        reference=_lstsq_reference,
        check=_lstsq_check,
        trap="lstsq returns a 4-tuple; rank-deficient case needs the minimum-norm "
             "solution.",
    ),
    Task(
        id="general_eigenvalues",
        family="cuSOLVER",
        difficulty="medium",
        prompt="""\
Implement `eigenvalues(A)`.

`A` is a real, non-symmetric float64 CuPy array (n, n). Return all n eigenvalues as a
complex128 CuPy array of shape (n,), in any order. Compute them on the GPU.""",
        entry_point="eigenvalues",
        solution="""\
import cupy as cp


def eigenvalues(A):
    return cp.linalg.eigvals(A).astype(cp.complex128)
""",
        make_inputs=lambda rng, case: (rng.standard_normal((40, 40)),),
        reference=lambda A: np.linalg.eigvals(A),
        check=_eig_check,
        trap="General (non-symmetric) eigenvalues are new in CuPy 14 (cuSOLVER Xgeev); "
             "older guidance says CuPy has no eig.",
        naive="""\
import cupy as cp


def eigenvalues(A):
    return cp.linalg.eigvalsh(A).astype(cp.complex128)
""",
    ),
    Task(
        id="symmetric_top_k",
        family="cuSOLVER",
        difficulty="medium",
        prompt="""\
Implement `top_k_eigh(A, k)`.

`A` is a real symmetric float64 CuPy array (n, n) and `k` is a Python int. Return a
tuple `(values, vectors)`: the k largest eigenvalues in descending order (shape (k,))
and the corresponding unit-norm eigenvectors as columns (shape (n, k)).""",
        entry_point="top_k_eigh",
        solution="""\
import cupy as cp


def top_k_eigh(A, k):
    w, v = cp.linalg.eigh(A)  # ascending order
    return w[::-1][:k], v[:, ::-1][:, :k]
""",
        make_inputs=lambda rng, case: ((lambda x: (x + x.T) / 2)(rng.standard_normal((100, 100))),
                                       5),
        reference=_topk_reference,
        check=_topk_check,
        trap="eigh returns ascending order with eigenvectors in columns.",
        naive="""\
import cupy as cp


def top_k_eigh(A, k):
    w, v = cp.linalg.eigh(A)
    return w[:k], v[:, :k]
""",
    ),
    Task(
        id="low_rank_approx",
        family="cuSOLVER",
        difficulty="medium",
        prompt="""\
Implement `low_rank(A, k)`.

`A` is a tall float64 CuPy array of shape (60000, 50) and `k` is a Python int. Return
the best rank-k approximation of A in the Frobenius norm (shape (60000, 50)) using a
truncated SVD.""",
        entry_point="low_rank",
        solution="""\
import cupy as cp


def low_rank(A, k):
    u, s, vh = cp.linalg.svd(A, full_matrices=False)
    return (u[:, :k] * s[:k]) @ vh[:k]
""",
        make_inputs=lambda rng, case: (rng.standard_normal((60000, 50)), 7),
        reference=_low_rank_reference,
        rtol=1e-8,
        atol=1e-8,
        cases=(Case(0),),
        trap="svd defaults to full_matrices=True, which allocates a 60000 x 60000 matrix "
             "(29 GB).",
        naive="""\
import cupy as cp


def low_rank(A, k):
    u, s, vh = cp.linalg.svd(A)
    return (u[:, :k] * s[:k]) @ vh[:k]
""",
    ),
    Task(
        id="solve_or_raise",
        family="cuSOLVER",
        difficulty="hard",
        prompt="""\
Implement `checked_solve(A, b)`.

`A` is a square float64 CuPy array (n, n) and `b` is float64 (n,). Return `x` with
`A @ x == b`. If A is singular, raise `numpy.linalg.LinAlgError` instead of returning
a result. Detect singularity from the factorization itself; do not compute the
determinant, the condition number or an SVD.""",
        entry_point="checked_solve",
        solution="""\
import cupy as cp
import cupyx


def checked_solve(A, b):
    # CuPy skips cuSOLVER's error flag unless asked to check it
    with cupyx.errstate(linalg='raise'):
        return cp.linalg.solve(A, b)
""",
        make_inputs=_singular_inputs,
        reference=lambda A, b: np.linalg.solve(A, b),
        rtol=1e-8,
        atol=1e-10,
        cases=(Case(0), Case(1, expect_error="numpy.linalg.LinAlgError")),
        trap="cupy.linalg.solve returns inf/nan for singular input unless "
             "cupyx.errstate(linalg='raise') is active.",
        naive="""\
import cupy as cp


def checked_solve(A, b):
    return cp.linalg.solve(A, b)  # relies on NumPy-style LinAlgError
""",
    ),
    Task(
        id="triangular_transposed_solve",
        family="cuSOLVER",
        difficulty="medium",
        prompt="""\
Implement `solve_upper_transposed(U, B)`.

`U` is a float64 CuPy array (n, n) whose upper triangle (including the diagonal)
holds an upper-triangular matrix; the entries below the diagonal are garbage and must
be ignored. `B` is float64 (n, k). Return X (n, k) with `triu(U).T @ X == B`, using a
triangular solver (no general solve, no explicit inverse).""",
        entry_point="solve_upper_transposed",
        solution="""\
from cupyx.scipy.linalg import solve_triangular


def solve_upper_transposed(U, B):
    return solve_triangular(U, B, trans='T', lower=False)
""",
        make_inputs=_upper_with_garbage,
        reference=lambda U, B: scipy.linalg.solve_triangular(U, B, trans='T', lower=False),
        rtol=1e-8,
        atol=1e-10,
        required_api=("cupyx.scipy.linalg.solve_triangular",),
        trap="Choosing the triangle and the trans flag; a general solve uses the garbage.",
        naive="""\
import cupy as cp


def solve_upper_transposed(U, B):
    return cp.linalg.solve(U.T, B)
""",
    ),
]
