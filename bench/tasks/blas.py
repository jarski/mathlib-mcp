"""Dense matrix products: cuBLAS through CuPy and cuBLASLt through nvmath-python."""

import numpy as np

from bench.task import Case, Task, assert_close


def _float32_close(rtol, atol):
    def check(actual, expected, inputs):
        if np.asarray(actual).dtype != np.float32:
            raise AssertionError(f"expected float32 output, got {np.asarray(actual).dtype}")
        assert_close(actual, expected, rtol, atol)
    return check


def _gemm_update_check(actual, expected, inputs):
    assert_close(actual, expected, 1e-10, 1e-10)
    # the input array c (inputs[2], copied back from the GPU after the call) must hold
    # the result too: the update is in place
    assert_close(inputs[2], expected, 1e-10, 1e-10)


TASKS = [
    Task(
        id="batched_matmul",
        family="cuBLAS",
        difficulty="easy",
        prompt="""\
Implement `batched_matmul(a, b)`.

`a` has shape (B, M, K) and `b` has shape (B, K, N), both float32 CuPy arrays.
Return the float32 array of shape (B, M, N) whose i-th matrix is `a[i] @ b[i]`.
Compute all products in a single batched GPU call (no Python loop over the batch).""",
        entry_point="batched_matmul",
        solution="""\
import cupy as cp


def batched_matmul(a, b):
    return cp.matmul(a, b)
""",
        make_inputs=lambda rng, case: (
            rng.random((8, 64, 48), dtype=np.float32),
            rng.random((8, 48, 32), dtype=np.float32),
        ),
        reference=lambda a, b: (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32),
        check=_float32_close(1e-4, 1e-4),
        trap="Control task: should be solved by every condition.",
    ),
    Task(
        id="gemm_update_inplace",
        family="cuBLAS",
        difficulty="medium",
        prompt="""\
Implement `gemm_update(a, b, c, alpha, beta)` with `cupy.cublas.gemm`.

`a` is (K, M), `b` is (K, N) and `c` is (M, N); all are C-contiguous float64 CuPy
arrays. `alpha` and `beta` are Python floats. Overwrite `c` in place with
`alpha * a.T @ b + beta * c` and return `c`. Let cuBLAS apply the transpose of `a`
(do not create a transposed copy).""",
        entry_point="gemm_update",
        solution="""\
from cupy import cublas


def gemm_update(a, b, c, alpha, beta):
    return cublas.gemm('T', 'N', a, b, out=c, alpha=alpha, beta=beta)
""",
        make_inputs=lambda rng, case: (
            rng.standard_normal((40, 30)),
            rng.standard_normal((40, 20)),
            rng.standard_normal((30, 20)),
            1.5 if case == 0 else 0.25,
            -0.5 if case == 0 else 2.0,
        ),
        reference=lambda a, b, c, alpha, beta: alpha * a.T @ b + beta * c,
        check=_gemm_update_check,
        required_api=("cupy.cublas.gemm",),
        trap="In-place `out=` semantics with beta, and the 'T'/'N' transpose flags.",
        naive="""\
def gemm_update(a, b, c, alpha, beta):
    c = alpha * a.T @ b + beta * c  # rebinds the name: the caller's array is unchanged
    return c
""",
    ),
    Task(
        id="batched_quadratic_forms",
        family="cuBLAS",
        difficulty="easy",
        prompt="""\
Implement `quadratic_forms(A, x)`.

`A` is a float64 CuPy array of shape (B, N, N) and `x` is float64 of shape (B, N).
Return the float64 array `q` of shape (B,) with `q[i] = x[i] @ A[i] @ x[i]`, computed
without a Python loop over the batch.""",
        entry_point="quadratic_forms",
        solution="""\
import cupy as cp


def quadratic_forms(A, x):
    return cp.einsum('bi,bij,bj->b', x, A, x)
""",
        make_inputs=lambda rng, case: (
            rng.standard_normal((50, 16, 16)),
            rng.standard_normal((50, 16)),
        ),
        reference=lambda A, x: np.einsum('bi,bij,bj->b', x, A, x),
        rtol=1e-10,
        atol=1e-10,
        trap="Control task (einsum).",
    ),
    Task(
        id="pairwise_sq_dists",
        family="cuBLAS",
        difficulty="medium",
        prompt="""\
Implement `pairwise_sq_dists(X, Y)`.

`X` is (n, d) and `Y` is (m, d), both float32 CuPy arrays. Return the float32 (n, m)
matrix of squared Euclidean distances `D[i, j] = sum((X[i] - Y[j])**2)`. The inputs
are large (n=4000, m=3000, d=512), so use a matrix-multiplication formulation and
never build an (n, m, d) intermediate. Clamp tiny negative values caused by rounding
to zero.""",
        entry_point="pairwise_sq_dists",
        solution="""\
import cupy as cp


def pairwise_sq_dists(X, Y):
    x2 = cp.sum(X * X, axis=1)[:, None]
    y2 = cp.sum(Y * Y, axis=1)[None, :]
    D = x2 + y2 - 2.0 * (X @ Y.T)
    return cp.maximum(D, 0.0)
""",
        make_inputs=lambda rng, case: (
            rng.random((4000, 512), dtype=np.float32),
            rng.random((3000, 512), dtype=np.float32),
        ),
        reference=lambda X, Y: np.maximum(
            (X.astype(np.float64) ** 2).sum(1)[:, None]
            + (Y.astype(np.float64) ** 2).sum(1)[None, :]
            - 2.0 * X.astype(np.float64) @ Y.astype(np.float64).T, 0.0),
        check=_float32_close(1e-3, 5e-2),
        cases=(Case(0),),
        trap="A broadcasting solution needs 24 GB and runs out of GPU memory.",
        naive="""\
def pairwise_sq_dists(X, Y):
    return ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1)
""",
    ),
    Task(
        id="fused_linear_relu",
        family="cuBLAS",
        difficulty="hard",
        prompt="""\
Implement `linear_relu(x, w, bias)` with nvmath-python.

`x` is (M, K), `w` is (K, N) and `bias` is (N,), all float32 CuPy arrays. Return the
float32 (M, N) array `relu(x @ w + bias)`, where `bias` is added to every row. Use
`nvmath.linalg.advanced.matmul` with a fused epilog so that the bias addition and
the ReLU are executed inside the matrix multiplication (no separate elementwise
kernels).""",
        entry_point="linear_relu",
        solution="""\
import nvmath


def linear_relu(x, w, bias):
    # cuBLASLt adds the bias per row of the result, so compute the transposed
    # product (x @ w).T = w.T @ x.T, whose rows correspond to the N output features.
    out_t = nvmath.linalg.advanced.matmul(
        w.T, x.T,
        epilog=nvmath.linalg.advanced.MatmulEpilog.RELU_BIAS,
        epilog_inputs={"bias": bias},
    )
    return out_t.T
""",
        make_inputs=lambda rng, case: (
            rng.standard_normal((96, 64)).astype(np.float32),
            rng.standard_normal((64, 48)).astype(np.float32),
            rng.standard_normal(48).astype(np.float32),
        ),
        reference=lambda x, w, bias: np.maximum(
            x.astype(np.float64) @ w.astype(np.float64) + bias, 0).astype(np.float32),
        check=_float32_close(1e-3, 1e-3),
        required_api=("nvmath.linalg.advanced.matmul",),
        requires=("nvmath",),
        trap="cuBLASLt's BIAS epilog adds one value per result ROW (length M), not per "
             "column as in `x @ w + b`.",
        naive="""\
import nvmath


def linear_relu(x, w, bias):
    return nvmath.linalg.advanced.matmul(
        x, w, epilog=nvmath.linalg.advanced.MatmulEpilog.RELU_BIAS,
        epilog_inputs={"bias": bias})
""",
    ),
    Task(
        id="matmul_plan_reuse",
        family="cuBLAS",
        difficulty="medium",
        prompt="""\
Implement `multiply_all(a, bs)` with nvmath-python's stateful matmul API.

`a` is (M, K) and `bs` is (T, K, N), both float32 CuPy arrays. Return the float32
(T, M, N) array whose t-th matrix is `a @ bs[t]`. Use
`nvmath.linalg.advanced.Matmul`: plan the multiplication once, then execute it for
each t, swapping in the new right-hand operand instead of planning again.""",
        entry_point="multiply_all",
        solution="""\
import cupy as cp
import nvmath


def multiply_all(a, bs):
    T, _, N = bs.shape
    out = cp.empty((T, a.shape[0], N), dtype=cp.float32)
    with nvmath.linalg.advanced.Matmul(a, bs[0]) as mm:
        mm.plan()
        for t in range(T):
            if t:
                mm.reset_operands(b=bs[t])
            out[t] = mm.execute()
    return out
""",
        make_inputs=lambda rng, case: (
            rng.random((64, 32), dtype=np.float32),
            rng.random((5, 32, 24), dtype=np.float32),
        ),
        reference=lambda a, bs: np.einsum('mk,tkn->tmn', a.astype(np.float64),
                                          bs.astype(np.float64)).astype(np.float32),
        check=_float32_close(1e-4, 1e-4),
        required_api=("nvmath.linalg.advanced.Matmul",),
        requires=("nvmath",),
        trap="Stateful plan/execute/reset_operands workflow of nvmath-python.",
    ),
]
