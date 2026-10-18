---
name: cupy-nvmath-gpu-math
description: Write correct GPU numerical code in Python with CuPy and nvmath-python, the Python front ends of the CUDA-X math libraries (cuBLAS, cuFFT, cuSOLVER, cuSPARSE, cuDSS, cuTENSOR). Gives exact signatures for 275 functions, the pitfalls behind most wrong GPU code, examples verified on a GPU, and a static checker that catches made-up functions and wrong arguments before the code runs. Use it whenever code imports cupy, cupyx or nvmath, or when the user wants matrix products, linear solves, least squares, eigenvalues, SVD, FFTs, sparse solvers, tensor contractions or NumPy/SciPy code moved to the GPU, even if they never name a library.
license: MIT
compatibility: Python 3.12+ with uv or pip for the checker script. No GPU needed to use the skill; the code it helps write needs CUDA to run.
metadata:
  author: jarski
  version: "0.1.0"
  source: https://github.com/jarski/mathlib-mcp
  cupy: "14.2.0"
  nvmath-python: "1.0.0"
---

# GPU math in Python: CuPy and nvmath-python

CuPy and nvmath-python look like NumPy and SciPy, which is exactly why code written
for them goes wrong: a name that exists in NumPy is guessed to exist in CuPy, a
submodule is used without being imported, a keyword argument is invented, or an inverse
FFT comes back scaled by *n*. None of that shows up until the code runs on a GPU, which
is often not the machine you are on. This skill replaces guessing with lookup.

The material comes from the library sources at pinned versions (CuPy 14.2.0,
nvmath-python 1.0.0): 275 API cards, 43 of them curated with pitfalls and examples
verified on a Tesla T4. The same cards and checker, served as MCP tools, took Claude
Haiku 4.5 from 51% to 92% on 26 GPU tasks and cut code with hallucinated APIs from 36%
to 4% of attempts ([results](https://github.com/jarski/mathlib-mcp#results)). This
skill packages the same material as files.

## Workflow

1. **Find the function.** Search `references/index.md` for what you need (one line per
   function, with the module it lives in). Entries marked `(curated)` have pitfalls and
   a verified example; prefer them when several functions would do.

2. **Read its card before you call it.** Search for `` ### `cupy.linalg.solve` `` (the
   exact dotted name) in `references/cupy.md` or `references/nvmath.md`. Read the
   signature, the **Import** line (when `import cupy` alone does not load the module),
   the **Pitfalls** and the example. The cards are long; read only the ones you use.

3. **Skim `references/pitfalls.md` for the area you are working in** (dense linear
   algebra, FFT, sparse, nvmath matmul, memory and dtypes). It is short and lists the
   mistakes that look right and fail on the GPU.

4. **Write the code.** Start from a verified example when one fits; they are also
   collected in `references/examples.md`.

5. **Check it before handing it over:**

   ```sh
   python scripts/check_gpu_code.py path/to/code.py
   ```

   The script needs the `mathlib-mcp` package. Without installing anything:

   ```sh
   uvx --from git+https://github.com/jarski/mathlib-mcp mathlib-mcp-check path/to/code.py
   ```

   It knows every public symbol and signature of the pinned versions, so a `FAIL`
   means a real problem: a function that does not exist, a wrong or missing argument,
   a submodule that was never imported. Fix every `error` line and run it again.
   `note` lines are informational (for example a guarded fallback inside `try/except`,
   or a name outside the indexed namespaces, which it reports as `unverified` rather
   than guessing).

If the **mathlib-mcp MCP server** is connected (tools `search_api`, `get_api_card`,
`get_example`, `check_snippet`), use those tools instead of the files: same content,
fetched on demand, with search over all cards.

## The pitfalls behind most wrong GPU code

Each of these is on a card in `references/pitfalls.md` with the details.

- `import cupy` does not load `cupy.cublas`, and `import cupyx` does not load
  `cupyx.scipy.sparse.linalg`: import the submodule you call, or the attribute lookup
  fails at runtime.
- `cupy.linalg` functions reject NumPy arrays (`LinAlgError`). Move data once with
  `cp.asarray`, keep it on the GPU, and bring results back with `x.get()` or
  `cp.asnumpy(x)`; `numpy.asarray(cupy_array)` raises `TypeError`.
- Singular or non-positive-definite input gives inf/nan silently in `solve`, `inv` and
  `cholesky`. Wrap the call in `cupyx.errstate(linalg='raise')` to get an exception.
  `cupyx.scipy.linalg.lu_factor` only warns, even inside `errstate`.
- Batched `cupy.linalg.solve`: with `a` of shape `(..., M, M)`, `b` must be `(M,)` or
  `(..., M, K)`; a per-system vector `(..., M)` is rejected.
- nvmath-python inverse FFTs are unnormalized: `nvmath.fft.ifft(nvmath.fft.fft(x))`
  returns `n * x`. `nvmath.fft.fft` transforms all axes unless you pass `axes=[-1]`,
  and it takes complex input only (`nvmath.fft.rfft` for real data).
- `cupy.fft.irfft` returns an even length unless you pass `n=original_length`.
- Sparse iterative solvers (`cg`, `gmres`) take `rtol=` (keyword-only); the old
  `tol=` raises `TypeError`. They return `(x, info)`; `info != 0` means not converged.
  `eigsh(A, k=3, which='SA')`: everything after `k` is keyword-only.
- `nvmath.linalg.advanced.matmul`: `a` and `b` are positional-only, and a BIAS epilog
  adds one value per row of the result (length M), not per column. For
  `relu(x @ w + b)` multiply the transposes and transpose back.
- `nvmath.sparse.advanced.direct_solver` (cuDSS) wants a CSR matrix and a Fortran-ordered
  2-D right-hand side (`cp.asfortranarray(b)`).
- Python floats give float64 arrays. Pass `dtype=cp.float32` when single precision is
  enough; FP64 is slow on many GPUs. `cupy.linalg.norm` returns a 0-d array, not a
  float.

## What the checker can and cannot tell you

It is static. It proves that the names, arguments and imports are right for
CuPy 14.2.0 and nvmath-python 1.0.0. It does not run the code, so shapes, dtypes,
memory and numerical behaviour are yours to get right from the pitfalls and examples.
Say so when you cannot run the code on a GPU yourself, and show the checker's `OK`
line as the evidence you do have.

## Scope

- `cupy`, `cupy.linalg`, `cupy.fft`, `cupy.cublas`, `cupyx.scipy.linalg`,
  `cupyx.scipy.fft`, `cupyx.scipy.sparse`, `cupyx.scipy.sparse.linalg`
- `nvmath.linalg.advanced` (matmul with epilogs), `nvmath.fft`,
  `nvmath.sparse.advanced` (cuDSS direct solver), `nvmath.linalg` and friends
- Not covered: custom kernels (`cupy.RawKernel`, `cupyx.jit`), `cupy.testing`, and
  anything outside those namespaces. The checker reports such names as `unverified`.

## Example

Task: solve 32 independent 8×8 linear systems on the GPU.

1. `references/index.md` lists `cupy.linalg.solve (curated)` and
   `cupy.cublas.batched_gesv (curated)`.
2. The card for `cupy.linalg.solve` says a batch `a` of shape `(..., M, M)` takes `b` of
   shape `(M,)` (shared) or `(..., M, K)`, and that a per-system vector `(..., M)` is
   rejected. So pass `b[..., None]` and take `[..., 0]`. It also says singular matrices
   do not raise unless you set `cupyx.errstate(linalg='raise')`.
3. Code:

   ```python
   import cupy as cp
   import cupyx

   A = cp.random.rand(32, 8, 8) + 8 * cp.eye(8)
   b = cp.random.rand(32, 8)
   with cupyx.errstate(linalg="raise"):
       x = cp.linalg.solve(A, b[..., None])[..., 0]
   ```

4. `python scripts/check_gpu_code.py solve.py` → `OK: no problems found`.
