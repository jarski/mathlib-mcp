# Pitfalls

Mistakes that look right and fail on the GPU, collected from the curated cards. Each item names the function it belongs to; the full card has the signature and a verified example.

## cupy

### `cupy.asarray`
- Copies host data to the current GPU. Transfers are often slower than the computation itself, so move data once and keep it on the GPU.
- There is no implicit conversion back: `numpy.asarray(cupy_array)` raises `TypeError`. Use `cupy.asnumpy(x)` or `x.get()`.
- Python floats and NumPy's default dtype give float64. Pass `dtype=cupy.float32` when single precision is enough: many GPUs (e.g. T4, consumer cards) have far lower FP64 throughput.

### `cupy.asnumpy`
- Blocks until the GPU work that produces the array has finished (`blocking=True` by default). It is the usual implicit synchronization point; GPU timings taken without a sync measure only kernel launches.
- `x.get()` does the same for a `cupy.ndarray`.

### `cupy.cublas.batched_gesv`
- Import: `import cupy.cublas` (`import cupy` alone is not enough).
- `import cupy` does not load `cupy.cublas`: use `from cupy import cublas` or `import cupy.cublas`, otherwise `cupy.cublas.batched_gesv` raises `AttributeError`.
- Unlike `cupy.linalg.solve`, it accepts one right-hand-side vector per system, shape `(..., M)`, as well as `(..., M, K)`.
- `cupy.linalg.solve` already calls it for stacked matrices up to `cupy.cublas.get_batched_gesv_limit()`; above that size batched LU is slow, so prefer `cupy.linalg.solve` for large matrices.

### `cupy.cublas.gemm`
- Import: `import cupy.cublas` (`import cupy` alone is not enough).
- `import cupy` does not load `cupy.cublas`: `cupy.cublas.gemm(...)` after only `import cupy` raises `AttributeError`. Use `from cupy import cublas` (as in the example) or `import cupy.cublas`.
- `transa`/`transb` take 'N', 'T' or 'H' and apply op() to the inputs without materializing a transposed copy.
- Only 2-D operands. Use `cupy.matmul` for batched products.
- With `out` given, the result is accumulated in place: `out = alpha * op(a) @ op(b) + beta * out`.

### `cupy.dot`
- For N-D inputs `dot` contracts the last axis of `a` with the second-to-last axis of `b` (NumPy semantics) and does not broadcast batch dimensions. Use `cupy.matmul` (`@`) for batched matrix products.

### `cupy.einsum`
- Pairwise contractions are reduced to `tensordot`/cuBLAS GEMMs by default. With `CUPY_ACCELERATORS=cutensor` and cuTENSOR installed, CuPy can use cuTENSOR contractions instead.
- Expressions like `'bij,bj->bi'` (batched matrix-vector) are a clear, fast alternative to `matmul` with reshaping.

### `cupy.fft.fft`
- Single precision is preserved: float16/float32/complex64 input gives complex64 output; other real or integer input gives complex128. NumPy < 2 always returned complex128.
- cuFFT plans are cached per device (`cupy.fft.config.get_plan_cache()`). The first call for a new shape/dtype pays the planning cost, so warm up before timing.
- Lengths with only small prime factors (2, 3, 5, 7) are fastest; `cupyx.scipy.fft.next_fast_len(n)` gives a good padded length.

### `cupy.fft.fftn`
- Transforms all axes by default. For a batch of images shaped (B, H, W), pass `axes=(-2, -1)` or use `fft2`, otherwise the batch axis is transformed too.
- One cuFFT N-D plan covers up to 3 adjacent axes that include the first or last axis. Other axis combinations fall back to successive 1-D transforms.

### `cupy.fft.irfft`
- Without `n`, the output length is `2 * (m - 1)` (always even). Pass `n=original_length` to recover odd-length signals.
- Normalization follows NumPy (`norm='backward'`): `irfft(rfft(x), n)` returns `x` with no manual scaling.

### `cupy.fft.rfft`
- For real input, `rfft` does about half the work and memory of `fft`. The output has `n//2 + 1` points along the transformed axis.
- To invert, call `irfft` with `n=` set to the original length, otherwise odd lengths are not recovered.

### `cupy.linalg.cholesky`
- Returns the lower-triangular factor `L` with `a = L @ L.conj().T`. CuPy 14 has no `upper=` keyword (NumPy 2 does); use `L.conj().T` for the upper factor.
- A matrix that is not positive definite gives nan silently unless inside `cupyx.errstate(linalg='raise')`.
- To solve with the factor, use two `cupyx.scipy.linalg.solve_triangular` calls (`lower=True`, then `trans='C'`).

### `cupy.linalg.eig`
- Requires cuSOLVER 11.7 (CUDA 12.6) or newer; older CUDA raises `RuntimeError('geev is not available')`. For symmetric/Hermitian matrices use `eigh`, which is faster and works everywhere.
- Always returns complex arrays (complex64/complex128), even when every eigenvalue is real. NumPy returns real arrays in that case.
- The order of the eigenvalues is not guaranteed and can differ from NumPy. Sort before comparing.

### `cupy.linalg.eigh`
- Only the triangle selected by `UPLO` ('L' by default) is read. Symmetry is not checked, so a non-symmetric input gives a wrong answer silently.
- Eigenvalues are returned in ascending order; eigenvectors are the columns `v[:, i]`.
- Stacked `(..., M, M)` input uses the Jacobi solver `syevj`.

### `cupy.linalg.inv`
- To solve `a x = b`, use `cupy.linalg.solve` instead of `inv(a) @ b`.
- Stacked input `(..., M, M)` is inverted with batched cuBLAS routines in one call.
- Singular input gives inf/nan silently unless inside `cupyx.errstate(linalg='raise')`.

### `cupy.linalg.lstsq`
- Implemented with an SVD (`full_matrices=False`): robust for rank-deficient problems, but slower than QR. For tall, well-conditioned problems `qr` + `cupyx.scipy.linalg.solve_triangular` is faster.
- Returns a 4-tuple `(x, residuals, rank, s)`. `rcond=None` uses `eps * max(M, N)` (the NumPy 2 default).
- Only 2-D `a` is supported; there is no batched least squares.

### `cupy.linalg.norm`
- Returns a 0-d CuPy array, not a Python float. Use `float(...)` or `.item()` to get a host scalar (this synchronizes).

### `cupy.linalg.qr`
- `mode` is one of 'reduced' (default), 'complete', 'r', 'raw'. For an (M, N) matrix with M > N, 'complete' allocates an (M, M) Q.
- Stacked input `(..., M, N)` is supported.

### `cupy.linalg.solve`
- Both arguments must be CuPy arrays. NumPy inputs raise `LinAlgError('cupy.linalg only supports cupy.ndarray')` instead of being copied implicitly.
- For a batch `a` of shape `(..., M, M)`, `b` must be `(M,)` (shared) or `(..., M, K)`. A per-system vector of shape `(..., M)` is rejected (NumPy 2 semantics). Pass `b[..., None]` and take `[..., 0]` of the result.
- A singular matrix does not raise by default. cuSOLVER's error flag is not checked (that would need a device sync), so the result silently contains inf/nan. Use `with cupyx.errstate(linalg='raise'):` to get `LinAlgError`.
- Prefer `solve(a, b)` over `inv(a) @ b`: it is faster and more accurate.

### `cupy.linalg.svd`
- `full_matrices=True` is the default (like NumPy). For a tall (M, N) matrix this allocates an (M, M) `u`; pass `full_matrices=False` unless the full basis is needed.
- Returns `(u, s, vh)` where `vh` is V^H, not V.
- Stacked input with matrices up to 32x32 uses the fast batched Jacobi solver (`gesvdj`); larger stacked matrices are processed with `gesvd`.

### `cupy.matmul`
- Stacked inputs `(..., M, K) @ (..., K, N)` run as a single batched cuBLAS GEMM. Prefer this over a Python loop of 2-D products.
- float32 GEMM uses full FP32 by default. Setting the environment variable `CUPY_TF32=1` lets cuBLAS use TF32 tensor cores (faster, ~10-bit mantissa). It is read once, at the first float32 matmul, so set it at process start.
- Mixed dtypes promote like NumPy (float32 @ float64 -> float64), which can be much slower on GPUs with low FP64 throughput.

### `cupyx.errstate`
- Only `linalg=` is implemented, with values 'ignore' (default) or 'raise'. Passing `divide=`, `over=`, `under=` or `invalid=` raises `NotImplementedError` (unlike `numpy.errstate`).
- With `linalg='raise'`, failed factorizations in the `cupy.linalg` routines (`solve`, `inv`, `cholesky`, ...) raise `numpy.linalg.LinAlgError`. Each check copies the device info flag to the host, which costs a synchronization.
- It does not affect `cupyx.scipy.linalg.lu_factor`: on an exactly singular matrix that function always issues a `RuntimeWarning` ('Diagonal number k is exactly zero. Singular matrix.') and returns, with or without `errstate`.

### `cupyx.scipy.fft.get_fft_plan`
- The plan can be used as a context manager (`with plan: y = cupy.fft.fft(x)`) or passed as `plan=` to `cupyx.scipy.fft` functions. Reusing it avoids plan look-ups in hot loops.
- `value_type` must match the transform: 'C2C' (default), 'R2C' for rfft, 'C2R' for irfft. A plan whose dtype, size or batch does not match the call raises `ValueError`.
- The plan is tied to the array's shape, dtype, layout and the transformed axes.

### `cupyx.scipy.fft.next_fast_len`
- Pad signals to `next_fast_len(n)` (sizes with small prime factors) before FFT-based convolution or correlation; awkward prime sizes are much slower.

### `cupyx.scipy.linalg.lu_factor`
- Factor once with `lu_factor` and reuse the `(lu, piv)` tuple in `lu_solve` for many right-hand sides; this is cheaper than calling `solve` repeatedly.
- An exactly singular matrix does not raise: `lu_factor` issues a `RuntimeWarning` ('Diagonal number k is exactly zero. Singular matrix.') and returns, even inside `cupyx.errstate(linalg='raise')`. To get an exception, turn the warning into an error (`warnings.simplefilter('error', RuntimeWarning)`) or use `cupy.linalg.solve` under `errstate`.
- `check_finite=True` by default, which scans the input for inf/nan and forces a device sync. Pass `check_finite=False` in hot loops.

### `cupyx.scipy.linalg.lu_solve`
- The first argument is the `(lu, piv)` tuple returned by `lu_factor`, not the original matrix.
- `check_finite=True` by default (forces a device sync); pass `check_finite=False` in loops.

### `cupyx.scipy.linalg.solve_triangular`
- Uses the upper triangle by default (`lower=False`), as in SciPy. Pass `lower=True` for factors from `cholesky` or the L of an LU.
- `trans` accepts 0/'N', 1/'T', 2/'C' to solve with the (conjugate) transpose without forming it.

### `cupyx.scipy.sparse.csr_matrix`
- Only bool, float32, float64, complex64 and complex128 values are supported. Integer data raises `ValueError`; cast to float first.
- Create it on the GPU from `(data, indices, indptr)` CuPy arrays, from a dense CuPy array, or from a SciPy sparse matrix (copied to the device).
- `csr_matrix * x` is a matrix product (SciPy spmatrix semantics), but `csr_array * x` is elementwise. Use `A @ x` to be unambiguous.
- Index arrays are int32 by default, and some cuSPARSE/cuSOLVER paths (e.g. `spsolve`) require int32 indices.

### `cupyx.scipy.sparse.diags`
- Pass `format='csr'` when the matrix goes to SpMV or a solver. The default format is unspecified and may change.

### `cupyx.scipy.sparse.linalg.cg`
- Import: `import cupyx.scipy.sparse.linalg` (`import cupyx` alone is not enough).
- The tolerance keyword is `rtol` (keyword-only, matching SciPy >= 1.12). The old `tol=` raises `TypeError`.
- Returns `(x, info)`. `info == 0` means converged and `info > 0` is the iteration count reached without converging. Always check `info`.
- `A` must be symmetric (Hermitian) positive definite; for general matrices use `gmres` or `bicgstab`.

### `cupyx.scipy.sparse.linalg.eigsh`
- Import: `import cupyx.scipy.sparse.linalg` (`import cupyx` alone is not enough).
- `which` supports only 'LM' (default), 'LA' and 'SA'. SciPy's 'SM' and shift-invert (`sigma=`) are not available.
- Everything after `k` is keyword-only: `eigsh(A, 3, 'SA')` raises `TypeError`; write `eigsh(A, k=3, which='SA')`.
- Needs `k < n`. For small dense problems, `cupy.linalg.eigh` is simpler and faster.

### `cupyx.scipy.sparse.linalg.gmres`
- Import: `import cupyx.scipy.sparse.linalg` (`import cupyx` alone is not enough).
- Tolerance is `rtol=` (keyword-only); `tol=` raises `TypeError`.
- Returns `(x, info)` with `info == 0` on convergence.
- Increase `restart` if convergence stalls on hard problems (more memory per restart cycle).

### `cupyx.scipy.sparse.linalg.spsolve`
- Import: `import cupyx.scipy.sparse.linalg` (`import cupyx` alone is not enough).
- Uses cuSOLVER's sparse QR (`csrlsvqr`). Non-CSR input is converted with a `SparseEfficiencyWarning`, so build the matrix in CSR format.
- A 2-D `b` is solved one column at a time (slow for many right-hand sides). For large systems or many right-hand sides consider `nvmath.sparse.advanced.direct_solver` (cuDSS).
- `b` is cast to `A.dtype`.

## nvmath-python

### `nvmath.fft.FFT`
- Workflow: construct -> `plan()` -> `execute(direction=...)` (repeatable) -> `free()`. Use it as a context manager. One object can run both forward and inverse C2C transforms.
- `execute(direction='inverse')` is unnormalized: scale by `1/n` to undo a forward transform.
- `reset_operand(new)` reuses the plan for a new array with the same shape, dtype and layout.
- Prolog/epilog device callbacks (compiled with `nvmath.fft.compile_prolog`/`compile_epilog`) fuse elementwise work into the FFT, but compiling them requires Numba with CUDA support (`RuntimeError` otherwise).

### `nvmath.fft.fft`
- Complex input only: real input raises `ValueError` ("This function expects complex operand"). Use `nvmath.fft.rfft` for real data, or cast with `.astype(cupy.complex64)`.
- Without `axes` it is an N-D FFT over all axes (like `numpy.fft.fftn`, not `numpy.fft.fft`). Pass `axes=[-1]` for a batched 1-D FFT along the last axis.
- `axes` must be contiguous and include the first or the last dimension; at most 3-D transforms.
- Host (NumPy) input executes on the CPU by default, because the execution space follows the operand's memory space. Pass CuPy/PyTorch GPU arrays or `execution='cuda'` to run on the GPU.
- Unlike NumPy and CuPy, the inverse transforms are unnormalized (cuFFT convention): `nvmath.fft.ifft(nvmath.fft.fft(x))` returns `n * x`. Divide by the transform size yourself.

### `nvmath.fft.ifft`
- Unnormalized: `ifft(fft(x))` returns `n * x` (n = product of the transformed lengths), whereas `numpy.fft.ifft` divides by n. Scale the result by `1/n`.
- Complex input only, and without `axes` all axes are transformed (N-D), as for `nvmath.fft.fft`.

### `nvmath.fft.irfft`
- There is no `n` argument. The output length of the last axis is `2*(m-1)` by default; for odd-length signals pass `options={'last_axis_parity': 'odd'}` to get `2*(m-1)+1`.
- Unnormalized: `irfft(rfft(x))` returns `n * x`, not `x` as with NumPy/CuPy. Divide by the output length.
- The input must be Hermitian-symmetric along the last axis (e.g. the output of `rfft`), otherwise the result is undefined.

### `nvmath.fft.rfft`
- Real input only. The last transformed axis has `n//2 + 1` complex outputs.
- Without `axes`, all axes are transformed (N-D). Pass `axes=[-1]` for a batched 1-D transform.

### `nvmath.linalg.advanced.Matmul`
- Workflow: construct -> `plan()` -> `execute()` (repeatable) -> `free()`. Use it as a context manager (`with Matmul(a, b) as mm:`) so resources are released.
- In the stateful API the epilog is passed to `plan(epilog=..., epilog_inputs=...)`, not to the constructor.
- `reset_operands(a=..., b=...)` swaps in new operands with the same shape, dtype and layout without re-planning. That is where the stateful API saves time.
- `autotune()` after `plan()` benchmarks the candidate algorithms on the actual data.

### `nvmath.linalg.advanced.MatmulEpilog`
- BIAS epilogs add one bias value per row of the result (length M). For the `x @ w + b` layout of a linear layer, see the transposed recipe on `nvmath.linalg.advanced.matmul`.
- Epilogs with BIAS in the name need `epilog_inputs={'bias': ...}`. DRELU epilogs need 'relu_aux' and DGELU epilogs need 'gelu_aux'; those come from the aux output of the matching forward epilog (`RELU_AUX`, `GELU_AUX`, ...).
- It is an alias of `nvmath.bindings.cublasLt.Epilogue` (an `IntEnum`), so members are accessed as `MatmulEpilog.RELU_BIAS`.

### `nvmath.linalg.advanced.MatmulOptions`
- Choose the GEMM precision with `compute_type`, e.g. `MatmulOptions(compute_type=nvmath.linalg.advanced.MatmulComputeType.COMPUTE_32F_FAST_TF32)` to allow TF32 tensor cores for float32.
- A plain dict also works: `options={'compute_type': ...}`.

### `nvmath.linalg.advanced.matmul`
- `a` and `b` are positional-only; everything after `c` is keyword-only. If `c` is given, `beta` must be given too.
- Epilogs fuse post-processing into the GEMM: `epilog=MatmulEpilog.RELU_BIAS, epilog_inputs={'bias': bias}`. The bias has length M (one value per row of the result) and is broadcast across the N columns, not across rows as in `x @ W + b`.
- For a linear layer `relu(x @ w + b)` with one bias value per output column (length N), multiply the transposes so that the features become rows, then transpose back: `matmul(w.T, x.T, epilog=MatmulEpilog.RELU_BIAS, epilog_inputs={'bias': b}).T`.
- Epilogs with `AUX` in the name (e.g. `RELU_AUX`) return a tuple `(result, aux_dict)` instead of an array.
- The result uses the same package and memory space as the inputs: CuPy in, CuPy out.
- Every call plans from scratch. For repeated products with the same shapes, use the stateful `Matmul` object and `reset_operands` to amortize planning.

### `nvmath.sparse.advanced.direct_solver`
- The left-hand side must be in CSR format (SciPy, `cupyx.scipy.sparse` CSR, or a PyTorch sparse CSR tensor).
- A 2-D right-hand side must be column-major (Fortran order): use `cupy.asfortranarray(b)` or build it as `cupy.ones((k, n)).T`. C-ordered 2-D input is rejected.
- Solves many right-hand sides in one factorization, unlike `cupyx.scipy.sparse.linalg.spsolve`, which loops over the columns.
- For repeated solves with the same sparsity pattern, use the stateful `DirectSolver` (plan -> factorize -> solve) and `reset_operands`.

### `nvmath.tensor.binary_contraction`
- The expression comes first: `binary_contraction('ij,jk->ik', a, b)`. Optional `c`, `alpha` and `beta` compute `alpha * contraction + beta * c` in one call.
- Operands stay in their package (CuPy in, CuPy out); for repeated contractions use the stateful `BinaryContraction`.
