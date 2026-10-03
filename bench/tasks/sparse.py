"""Sparse linear algebra: cuSPARSE via cupyx.scipy.sparse, cuDSS via nvmath-python."""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from bench.task import Task, assert_close


def _triplets(rng, case):
    m, n, nnz = 500, 400, 3000
    rows = rng.integers(0, m, nnz).astype(np.int32)
    cols = rng.integers(0, n, nnz).astype(np.int32)
    rows[:300], cols[:300] = rows[300:600], cols[300:600]  # guaranteed duplicates
    return rows, cols, rng.standard_normal(nnz), (m, n), rng.standard_normal(n)


def _triplet_reference(rows, cols, vals, shape, x):
    A = np.zeros(shape)
    np.add.at(A, (rows, cols), vals)
    return A @ x


def _poisson_inputs(rng, case):
    n = 300
    return rng.standard_normal(n), 1.0 / (n + 1)


def _poisson_reference(f, h):
    n = f.size
    A = scipy.sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csc") / h**2
    return scipy.sparse.linalg.spsolve(A, f)


def _laplacian_eigs(nx, ny, k):
    a = 2 - 2 * np.cos(np.pi * np.arange(1, nx + 1) / (nx + 1))
    b = 2 - 2 * np.cos(np.pi * np.arange(1, ny + 1) / (ny + 1))
    return np.sort((a[:, None] + b[None, :]).ravel())[:k]


def _row_normalize_inputs(rng, case):
    A = scipy.sparse.random(300, 200, density=0.03, format="csr", random_state=rng)
    A = A.tolil()
    A[[3, 50, 299], :] = 0  # empty rows must stay empty
    return (A.tocsr(),)


def _row_normalize_reference(A):
    dense = A.toarray()
    sums = dense.sum(axis=1, keepdims=True)
    return np.divide(dense, sums, out=np.zeros_like(dense), where=sums != 0)


def _row_normalize_check(actual, expected, inputs):
    if getattr(actual, "format", None) != "csr":
        raise AssertionError(f"expected a CSR matrix, got {type(actual).__name__}")
    assert_close(actual.toarray(), expected, 1e-12, 1e-12)


def _cudss_inputs(rng, case):
    n = 500
    A = scipy.sparse.random(n, n, density=0.01, format="csr", random_state=rng)
    A = (A + scipy.sparse.diags(rng.random(n) + 10.0)).tocsr()  # non-symmetric, regular
    return A, rng.standard_normal((n, 6))


TASKS = [
    Task(
        id="spmv_from_triplets",
        family="cuSPARSE",
        difficulty="easy",
        prompt="""\
Implement `spmv_from_triplets(rows, cols, vals, shape, x)`.

`rows` and `cols` are int32 CuPy arrays and `vals` is a float64 CuPy array; together
they describe a sparse matrix A in COO (triplet) format, where duplicate (row, col)
pairs must be summed. `shape` is a Python tuple (m, n) and `x` is a float64 CuPy array
(n,). Return `y = A @ x` (float64, shape (m,)) using a GPU sparse matrix from
`cupyx.scipy.sparse`.""",
        entry_point="spmv_from_triplets",
        solution="""\
import cupyx.scipy.sparse as sp


def spmv_from_triplets(rows, cols, vals, shape, x):
    A = sp.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
    return A @ x
""",
        make_inputs=_triplets,
        reference=_triplet_reference,
        rtol=1e-10,
        atol=1e-10,
        required_api=("cupyx.scipy.sparse",),
        trap="Control task (COO construction and SpMV).",
    ),
    Task(
        id="cg_poisson_1d",
        family="cuSPARSE",
        difficulty="medium",
        prompt="""\
Implement `poisson_1d(f, h)`.

Solve the 1-D Poisson problem `-u'' = f` with zero Dirichlet boundary conditions on a
uniform grid with spacing `h` (a Python float): the tridiagonal system
`(1/h**2) * tridiag(-1, 2, -1) @ u = f`, where `f` is a float64 CuPy array (n,).
Use the conjugate gradient solver from `cupyx.scipy.sparse.linalg` with a relative
tolerance of 1e-10 and raise RuntimeError if it does not converge. Return u (float64,
shape (n,)).""",
        entry_point="poisson_1d",
        solution="""\
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse.linalg import cg


def poisson_1d(f, h):
    n = f.size
    s = 1.0 / h**2
    A = sp.diags([-s, 2.0 * s, -s], [-1, 0, 1], shape=(n, n), format='csr')
    u, info = cg(A, f, rtol=1e-10, maxiter=10 * n)
    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info})")
    return u
""",
        make_inputs=_poisson_inputs,
        reference=_poisson_reference,
        rtol=1e-5,
        atol=1e-8,
        required_api=("cupyx.scipy.sparse.linalg.cg",),
        trap="The tolerance keyword is `rtol` (keyword-only); `tol=` raises TypeError. "
             "`info` must be checked.",
        naive="""\
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse.linalg import cg


def poisson_1d(f, h):
    n = f.size
    A = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format='csr') * (1.0 / h**2)
    u, info = cg(A, f, tol=1e-10)
    return u
""",
    ),
    Task(
        id="laplacian_2d_smallest_eigs",
        family="cuSPARSE",
        difficulty="hard",
        prompt="""\
Implement `smallest_eigenvalues(nx, ny, k)`.

Build the 2-D discrete Laplacian with zero Dirichlet boundaries on an nx-by-ny grid as
a sparse matrix of size (nx*ny, nx*ny): the Kronecker sum
`kron(I_ny, T_nx) + kron(T_ny, I_nx)` where `T_n = tridiag(-1, 2, -1)` is n-by-n (no
grid-spacing scaling). Return its k smallest eigenvalues in ascending order (float64
CuPy array of shape (k,)), computed on the GPU with a sparse iterative eigensolver
from `cupyx.scipy.sparse.linalg` (do not convert the matrix to dense).""",
        entry_point="smallest_eigenvalues",
        solution="""\
import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse.linalg import eigsh


def smallest_eigenvalues(nx, ny, k):
    def lap1d(n):
        return sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format='csr')

    A = (sp.kron(sp.identity(ny, format='csr'), lap1d(nx))
         + sp.kron(lap1d(ny), sp.identity(nx, format='csr'))).tocsr()
    # 'SA' = smallest algebraic; 'SM' is not supported by CuPy, and `which` is
    # keyword-only
    w = eigsh(A, k=k, which='SA', ncv=min(A.shape[0] - 1, 4 * k + 30),
              maxiter=20000, return_eigenvectors=False)
    return cp.sort(w)
""",
        make_inputs=lambda rng, case: (20, 17, 4) if case == 0 else (15, 11, 3),
        reference=_laplacian_eigs,
        rtol=1e-6,
        atol=1e-8,
        required_api=("cupyx.scipy.sparse.linalg.eigsh|cupyx.scipy.sparse.linalg.lobpcg",),
        trap="eigsh supports which='SA' but not 'SM', and `which` is keyword-only.",
        naive="""\
import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse.linalg import eigsh


def smallest_eigenvalues(nx, ny, k):
    def lap1d(n):
        return sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format='csr')

    A = (sp.kron(sp.identity(ny, format='csr'), lap1d(nx))
         + sp.kron(lap1d(ny), sp.identity(nx, format='csr'))).tocsr()
    return cp.sort(eigsh(A, k, which='SM', return_eigenvectors=False))
""",
    ),
    Task(
        id="sparse_row_normalize",
        family="cuSPARSE",
        difficulty="medium",
        prompt="""\
Implement `row_normalize(A)`.

`A` is a `cupyx.scipy.sparse.csr_matrix` (m, n) with non-negative float64 entries.
Return a `cupyx.scipy.sparse.csr_matrix` of the same shape in which every row is
divided by its row sum. Rows whose sum is zero must stay all-zero (no NaN or inf).
Keep the computation sparse and on the GPU.""",
        entry_point="row_normalize",
        solution="""\
import cupy as cp
import cupyx.scipy.sparse as sp


def row_normalize(A):
    sums = cp.asarray(A.sum(axis=1)).ravel()
    inv = cp.zeros_like(sums)
    nonzero = sums != 0
    inv[nonzero] = 1.0 / sums[nonzero]
    return (sp.diags(inv, format='csr') @ A).tocsr()
""",
        make_inputs=_row_normalize_inputs,
        reference=_row_normalize_reference,
        check=_row_normalize_check,
        trap="Zero rows (division by zero) and keeping a CSR result.",
    ),
    Task(
        id="cudss_multiple_rhs",
        family="cuDSS",
        difficulty="hard",
        prompt="""\
Implement `sparse_solve_many(A, B)` with nvmath-python.

`A` is a non-singular `cupyx.scipy.sparse.csr_matrix` (n, n) with float64 values, and
`B` is a C-contiguous float64 CuPy array (n, r) holding r right-hand sides. Return the
float64 (n, r) solution X of `A @ X = B`, solving all right-hand sides with one sparse
factorization using nvmath-python's cuDSS direct solver
(`nvmath.sparse.advanced.direct_solver`).""",
        entry_point="sparse_solve_many",
        solution="""\
import cupy as cp
import nvmath


def sparse_solve_many(A, B):
    # cuDSS requires a column-major (Fortran-ordered) right-hand side
    return nvmath.sparse.advanced.direct_solver(A, cp.asfortranarray(B))
""",
        make_inputs=_cudss_inputs,
        reference=lambda A, B: scipy.sparse.linalg.spsolve(A.tocsc(), B),
        rtol=1e-8,
        atol=1e-10,
        required_api=("nvmath.sparse.advanced.direct_solver|nvmath.sparse.advanced.DirectSolver",),
        requires=("nvmath",),
        trap="A 2-D right-hand side must be column-major; C-ordered B is rejected.",
        naive="""\
import nvmath


def sparse_solve_many(A, B):
    return nvmath.sparse.advanced.direct_solver(A, B)
""",
    ),
]
