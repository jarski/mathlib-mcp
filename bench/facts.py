"""Behavioural claims made in the curated pitfalls, checked on a real GPU.

Each fact is a self-contained snippet plus the expected outcome:
``"ok"`` (runs without error; asserts inside check the claim) or ``"raises:Name"``
(must raise an exception whose class name is ``Name``). Facts marked ``observe`` only
record the printed output; they answer questions the sources do not settle.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    id: str
    claim: str
    code: str
    expect: str = "ok"  # ok | raises:<ExceptionName> | observe
    requires: tuple[str, ...] = ()
    cards: tuple[str, ...] = ()  # cards whose pitfalls rely on this fact


FACTS = [
    Fact(
        "no_implicit_numpy_conversion",
        "numpy.asarray(cupy_array) raises TypeError",
        "import numpy as np, cupy as cp\nnp.asarray(cp.ones(3))",
        "raises:TypeError",
        cards=("cupy.asarray",),
    ),
    Fact(
        "linalg_rejects_numpy_input",
        "cupy.linalg.solve with NumPy inputs raises LinAlgError",
        "import numpy as np, cupy as cp\ncp.linalg.solve(np.eye(3), np.ones(3))",
        "raises:LinAlgError",
        cards=("cupy.linalg.solve",),
    ),
    Fact(
        "singular_solve_is_silent",
        "cupy.linalg.solve on a singular matrix returns non-finite values without raising",
        "import cupy as cp\nA = cp.random.rand(8, 8)\nA[:, 3] = 0\n"
        "x = cp.linalg.solve(A, cp.ones(8))\nassert not bool(cp.isfinite(x).all())",
        cards=("cupy.linalg.solve", "cupy.linalg.inv"),
    ),
    Fact(
        "errstate_raise_on_singular",
        "cupyx.errstate(linalg='raise') turns a singular solve into LinAlgError",
        "import cupy as cp, cupyx\nA = cp.random.rand(8, 8)\nA[:, 3] = 0\n"
        "with cupyx.errstate(linalg='raise'):\n    cp.linalg.solve(A, cp.ones(8))",
        "raises:LinAlgError",
        cards=("cupy.linalg.solve", "cupyx.errstate"),
    ),
    Fact(
        "errstate_divide_not_implemented",
        "cupyx.errstate(divide=...) raises NotImplementedError",
        "import cupyx\nwith cupyx.errstate(divide='ignore'):\n    pass",
        "raises:NotImplementedError",
        cards=("cupyx.errstate",),
    ),
    Fact(
        "batched_solve_rejects_stacked_vectors",
        "cupy.linalg.solve rejects a (B, M) right-hand side for (B, M, M) matrices",
        "import cupy as cp\ncp.linalg.solve(cp.random.rand(5, 3, 3) + 3 * cp.eye(3), "
        "cp.random.rand(5, 3))",
        "raises:ValueError",
        cards=("cupy.linalg.solve",),
    ),
    Fact(
        "cholesky_has_no_upper",
        "cupy.linalg.cholesky has no `upper=` keyword",
        "import cupy as cp\ncp.linalg.cholesky(cp.eye(3), upper=True)",
        "raises:TypeError",
        cards=("cupy.linalg.cholesky",),
    ),
    Fact(
        "cg_has_no_tol",
        "cupyx.scipy.sparse.linalg.cg does not accept `tol=`",
        "import cupy as cp\nfrom cupyx.scipy.sparse.linalg import cg\n"
        "cg(cp.eye(4), cp.ones(4), tol=1e-8)",
        "raises:TypeError",
        cards=("cupyx.scipy.sparse.linalg.cg",),
    ),
    Fact(
        "eigsh_which_keyword_only",
        "cupyx.scipy.sparse.linalg.eigsh takes `which` as keyword only",
        "import cupyx.scipy.sparse as sp\nfrom cupyx.scipy.sparse.linalg import eigsh\n"
        "A = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(50, 50), format='csr')\n"
        "eigsh(A, 3, 'SA')",
        "raises:TypeError",
        cards=("cupyx.scipy.sparse.linalg.eigsh",),
    ),
    Fact(
        "eigsh_no_sm",
        "cupyx.scipy.sparse.linalg.eigsh rejects which='SM'",
        "import cupyx.scipy.sparse as sp\nfrom cupyx.scipy.sparse.linalg import eigsh\n"
        "A = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(50, 50), format='csr')\n"
        "eigsh(A, k=3, which='SM')",
        "raises:ValueError",
        cards=("cupyx.scipy.sparse.linalg.eigsh",),
    ),
    Fact(
        "sparse_rejects_int_data",
        "cupyx.scipy.sparse.csr_matrix rejects integer data",
        "import cupy as cp, cupyx.scipy.sparse as sp\nsp.csr_matrix(cp.eye(3, dtype=cp.int64))",
        "raises:ValueError",
        cards=("cupyx.scipy.sparse.csr_matrix",),
    ),
    Fact(
        "spmatrix_star_is_matmul",
        "csr_matrix * x is a matrix product, csr_array * x is elementwise",
        "import cupy as cp, cupyx.scipy.sparse as sp\n"
        "M = sp.csr_matrix(cp.asarray([[1.0, 2.0], [3.0, 4.0]]))\n"
        "assert cp.allclose(M * cp.asarray([1.0, 1.0]), cp.asarray([3.0, 7.0]))\n"
        "A = sp.csr_array(cp.asarray([[1.0, 2.0], [3.0, 4.0]]))\n"
        "assert cp.allclose((A * A).toarray(), cp.asarray([[1.0, 4.0], [9.0, 16.0]]))",
        cards=("cupyx.scipy.sparse.csr_matrix",),
    ),
    Fact(
        "cupy_cublas_not_loaded_by_import_cupy",
        "`import cupy` does not load the cupy.cublas submodule",
        "import cupy\ncupy.cublas",
        "raises:AttributeError",
        cards=("cupy.cublas.gemm", "cupy.cublas.batched_gesv"),
    ),
    Fact(
        "sparse_linalg_not_loaded_by_import_sparse",
        "`import cupyx.scipy.sparse` does not load cupyx.scipy.sparse.linalg",
        "import cupyx.scipy.sparse as sp\nsp.linalg",
        "raises:AttributeError",
        cards=("cupyx.scipy.sparse.linalg.cg",),
    ),
    Fact(
        "fft_keeps_single_precision",
        "cupy.fft.fft keeps float32 -> complex64 and float64 -> complex128",
        "import cupy as cp\nassert cp.fft.fft(cp.ones(8, cp.float32)).dtype == cp.complex64\n"
        "assert cp.fft.fft(cp.ones(8)).dtype == cp.complex128",
        cards=("cupy.fft.fft",),
    ),
    Fact(
        "norm_returns_0d_array",
        "cupy.linalg.norm returns a 0-d cupy.ndarray",
        "import cupy as cp\nn = cp.linalg.norm(cp.ones(4))\n"
        "assert isinstance(n, cp.ndarray) and n.ndim == 0",
        cards=("cupy.linalg.norm",),
    ),
    Fact(
        "eig_available",
        "cupy.linalg.eig works (needs cuSOLVER >= 11.7)",
        "import cupy as cp\nw, v = cp.linalg.eig(cp.asarray([[2.0, 1.0], [0.0, 3.0]]))\n"
        "assert w.dtype == cp.complex128",
        cards=("cupy.linalg.eig",),
    ),
    Fact(
        "nvmath_fft_rejects_real",
        "nvmath.fft.fft raises ValueError for real input",
        "import cupy as cp, nvmath\nnvmath.fft.fft(cp.ones(16, cp.float32))",
        "raises:ValueError",
        requires=("nvmath",),
        cards=("nvmath.fft.fft",),
    ),
    Fact(
        "nvmath_fft_default_axes_all",
        "nvmath.fft.fft without axes transforms all axes (like numpy.fft.fftn)",
        "import numpy as np, cupy as cp, nvmath\n"
        "x = cp.random.rand(4, 8).astype(cp.complex64)\n"
        "assert cp.allclose(nvmath.fft.fft(x), cp.fft.fftn(x), atol=1e-4)",
        requires=("nvmath",),
        cards=("nvmath.fft.fft",),
    ),
    Fact(
        "nvmath_ifft_unnormalized",
        "nvmath.fft.ifft is unnormalized: ifft(fft(x)) == n * x (first GPU run measured 64x "
        "for n=64)",
        "import cupy as cp, nvmath\nx = cp.random.rand(64).astype(cp.complex128)\n"
        "y = nvmath.fft.ifft(nvmath.fft.fft(x))\nassert cp.allclose(y, 64 * x)",
        requires=("nvmath",),
        cards=("nvmath.fft.fft", "nvmath.fft.ifft", "nvmath.fft.FFT"),
    ),
    Fact(
        "nvmath_irfft_unnormalized",
        "nvmath.fft.irfft is unnormalized: irfft(rfft(x)) == n * x",
        "import cupy as cp, nvmath\nx = cp.random.rand(4, 64)\n"
        "y = nvmath.fft.irfft(nvmath.fft.rfft(x, axes=[-1]), axes=[-1])\n"
        "assert cp.allclose(y, 64 * x)",
        requires=("nvmath",),
        cards=("nvmath.fft.irfft",),
    ),
    Fact(
        "nvmath_numpy_fft_runs_on_cpu",
        "nvmath.fft with a NumPy operand executes on the CPU and returns NumPy",
        "import numpy as np, nvmath\n"
        "with nvmath.fft.FFT(np.ones(16, np.complex64)) as f:\n"
        "    f.plan()\n    r = f.execute()\n"
        "    assert f.execution_space == 'cpu', f.execution_space\n"
        "assert isinstance(r, np.ndarray)",
        requires=("nvmath",),
        cards=("nvmath.fft.fft",),
    ),
    Fact(
        "nvmath_bias_is_per_row",
        "RELU_BIAS adds a length-M bias per result row",
        "import cupy as cp, nvmath\n"
        "a, b = cp.ones((3, 2), cp.float32), cp.ones((2, 5), cp.float32)\n"
        "bias = cp.asarray([0.0, 10.0, 20.0], cp.float32)\n"
        "r = nvmath.linalg.advanced.matmul(a, b, "
        "epilog=nvmath.linalg.advanced.MatmulEpilog.BIAS, epilog_inputs={'bias': bias})\n"
        "assert cp.allclose(r, 2.0 + bias[:, None])",
        requires=("nvmath",),
        cards=("nvmath.linalg.advanced.matmul",),
    ),
    Fact(
        "nvmath_aux_epilog_returns_tuple",
        "RELU_AUX returns (result, aux_dict)",
        "import cupy as cp, nvmath\n"
        "a = cp.random.rand(64, 32, dtype=cp.float32)\n"
        "b = cp.random.rand(32, 16, dtype=cp.float32)\n"
        "r = nvmath.linalg.advanced.matmul(a, b, "
        "epilog=nvmath.linalg.advanced.MatmulEpilog.RELU_AUX)\n"
        "assert isinstance(r, tuple) and isinstance(r[1], dict), type(r)",
        requires=("nvmath",),
        cards=("nvmath.linalg.advanced.matmul",),
    ),
    Fact(
        "cudss_rejects_c_ordered_rhs",
        "nvmath.sparse.advanced.direct_solver rejects a C-ordered 2-D right-hand side",
        "import cupy as cp, cupyx.scipy.sparse as sp, nvmath\n"
        "A = sp.diags([1.0, 4.0, 1.0], [-1, 0, 1], shape=(20, 20), format='csr')\n"
        "nvmath.sparse.advanced.direct_solver(A, cp.ones((20, 3)))",
        "raises:TypeError|ValueError",
        requires=("nvmath",),
        cards=("nvmath.sparse.advanced.direct_solver",),
    ),
]
