# Verified examples

Complete, runnable examples from the curated cards. Each one was run on the GPU named in its heading; the asserts passed there.

## cupy

### `cupy.asarray` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp

x_host = np.arange(6, dtype=np.float32).reshape(2, 3)
x = cp.asarray(x_host)             # host -> device copy
y = (x * 2).sum(axis=1)            # computed on the GPU
y_host = cp.asnumpy(y)             # device -> host copy (waits for the GPU)
assert isinstance(y, cp.ndarray) and isinstance(y_host, np.ndarray)
assert np.allclose(y_host, (x_host * 2).sum(axis=1))
```

### `cupy.asnumpy` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp

x = cp.linspace(0.0, 1.0, 5, dtype=cp.float32)
host = cp.asnumpy(x)               # same as x.get()
assert isinstance(host, np.ndarray)
assert np.allclose(host, np.linspace(0.0, 1.0, 5))
```

### `cupy.cublas.batched_gesv` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
from cupy import cublas                           # `import cupy` alone does not load it

A = cp.random.rand(64, 12, 12) + 12 * cp.eye(12)
y = cp.random.rand(64, 12)                        # one right-hand side per system
x = cublas.batched_gesv(A, y)                     # stacked vectors are accepted here
assert x.shape == (64, 12)
assert np.allclose(cp.einsum('bij,bj->bi', A, x).get(), y.get())
```

### `cupy.cublas.gemm` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
from cupy import cublas

a = cp.random.rand(64, 32)
b = cp.random.rand(64, 16)
c = cublas.gemm('T', 'N', a, b)                   # a.T @ b without a transposed copy
assert np.allclose(c.get(), a.get().T @ b.get())
```

### `cupy.einsum` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp

A = cp.random.rand(32, 8, 8)
x = cp.random.rand(32, 8)
y = cp.einsum('bij,bj->bi', A, x)                 # batched matrix-vector product
assert np.allclose(y.get(), np.einsum('bij,bj->bi', A.get(), x.get()))
```

### `cupy.fft.fft` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp

x = cp.random.rand(1024, dtype=cp.float32)
X = cp.fft.fft(x)
assert X.dtype == cp.complex64                    # precision is kept
assert np.allclose(X.get(), np.fft.fft(x.get()), rtol=1e-3, atol=1e-3)
```

### `cupy.fft.fftn` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp

imgs = cp.random.rand(16, 64, 64).astype(cp.complex64)   # batch of 16 images
F = cp.fft.fftn(imgs, axes=(-2, -1))              # 2-D FFT of each image
assert F.shape == imgs.shape
assert np.allclose(F.get(), np.fft.fftn(imgs.get(), axes=(-2, -1)), rtol=1e-3, atol=1e-2)
```

### `cupy.fft.rfft` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp

x = cp.random.rand(4, 1001)                       # odd length, batched along axis 0
X = cp.fft.rfft(x, axis=-1)
assert X.shape == (4, 1001 // 2 + 1)
x_back = cp.fft.irfft(X, n=1001, axis=-1)         # pass n for odd lengths
assert np.allclose(x_back.get(), x.get())
```

### `cupy.linalg.cholesky` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import cupyx.scipy.linalg

n = 64
X = cp.random.rand(n, n)
A = X @ X.T + n * cp.eye(n)                       # symmetric positive definite
L = cp.linalg.cholesky(A)
assert np.allclose((L @ L.T).get(), A.get())

b = cp.random.rand(n)
y = cupyx.scipy.linalg.solve_triangular(L, b, lower=True)
x = cupyx.scipy.linalg.solve_triangular(L, y, lower=True, trans='C')
assert np.allclose((A @ x).get(), b.get())
```

### `cupy.linalg.eig` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp

A = cp.asarray([[2.0, 1.0], [0.0, 3.0]])
w, v = cp.linalg.eig(A)
assert w.dtype.kind == 'c'                        # complex even for real eigenvalues
assert np.allclose(np.sort(w.get().real), [2.0, 3.0])
assert np.allclose((A @ v).get(), (v * w).get())
```

### `cupy.linalg.eigh` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp

X = cp.random.rand(50, 50)
A = (X + X.T) / 2
w, v = cp.linalg.eigh(A)
assert np.all(np.diff(w.get()) >= 0)              # ascending
assert np.allclose((A @ v).get(), (v * w).get(), atol=1e-8)
```

### `cupy.linalg.inv` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp

A = cp.random.rand(16, 4, 4) + 4 * cp.eye(4)     # batch of 16 matrices
Ainv = cp.linalg.inv(A)
assert np.allclose((A @ Ainv).get(), np.broadcast_to(np.eye(4), (16, 4, 4)), atol=1e-10)
```

### `cupy.linalg.lstsq` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp

m, n = 500, 20
A = cp.random.rand(m, n)
x_true = cp.arange(n, dtype=A.dtype)
b = A @ x_true
x, residuals, rank, s = cp.linalg.lstsq(A, b, rcond=None)
assert int(rank) == n
assert np.allclose(x.get(), x_true.get(), atol=1e-6)
```

### `cupy.linalg.norm` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp

x = cp.asarray([3.0, 4.0])
n = cp.linalg.norm(x)
assert isinstance(n, cp.ndarray) and n.ndim == 0
assert float(n) == 5.0
```

### `cupy.linalg.qr` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp

A = cp.random.rand(100, 10)
Q, R = cp.linalg.qr(A)                            # reduced: Q (100, 10), R (10, 10)
assert Q.shape == (100, 10) and R.shape == (10, 10)
assert np.allclose((Q @ R).get(), A.get())
assert np.allclose((Q.T @ Q).get(), np.eye(10), atol=1e-10)
```

### `cupy.linalg.solve` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp

n = 256
a = cp.random.rand(n, n) + n * cp.eye(n)          # well-conditioned
b = cp.random.rand(n)
x = cp.linalg.solve(a, b)
assert np.allclose((a @ x).get(), b.get())

# 32 independent 8x8 systems, each with its own right-hand side vector
A = cp.random.rand(32, 8, 8) + 8 * cp.eye(8)
B = cp.random.rand(32, 8)
X = cp.linalg.solve(A, B[..., None])[..., 0]      # b must be (..., M, K)
assert np.allclose(cp.einsum('bij,bj->bi', A, X).get(), B.get())
```

### `cupy.linalg.svd` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp

A = cp.random.rand(1000, 50)
u, s, vh = cp.linalg.svd(A, full_matrices=False)  # u: (1000, 50), not (1000, 1000)
assert u.shape == (1000, 50) and s.shape == (50,) and vh.shape == (50, 50)
assert np.allclose(((u * s) @ vh).get(), A.get())
```

### `cupy.matmul` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp

a = cp.random.rand(8, 64, 32, dtype=cp.float32)   # batch of 8 matrices
b = cp.random.rand(8, 32, 16, dtype=cp.float32)
c = cp.matmul(a, b)                               # one batched GEMM; same as a @ b
assert c.shape == (8, 64, 16) and c.dtype == cp.float32
assert np.allclose(c.get(), a.get() @ b.get(), rtol=1e-4, atol=1e-4)
```

### `cupyx.errstate` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import cupyx

singular = cp.zeros((3, 3))
try:
    with cupyx.errstate(linalg='raise'):
        cp.linalg.cholesky(singular)
    raised = False
except np.linalg.LinAlgError:
    raised = True
assert raised
```

### `cupyx.scipy.fft.get_fft_plan` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft

x = cp.random.rand(64, 128).astype(cp.complex64)
plan = cufft.get_fft_plan(x, axes=(-1,))         # plan a 1-D C2C FFT along the last axis
y = cufft.fft(x, axis=-1, plan=plan)
with plan:                                        # plan also works as a context manager
    y2 = cp.fft.fft(x, axis=-1)
assert np.allclose(y.get(), y2.get())
assert np.allclose(y.get(), np.fft.fft(x.get(), axis=-1), rtol=1e-3, atol=1e-3)
```

### `cupyx.scipy.fft.next_fast_len` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import cupy as cp
from cupyx.scipy.fft import next_fast_len

n = next_fast_len(1009)                           # 1009 is prime
assert n >= 1009
x = cp.random.rand(1009)
X = cp.fft.rfft(x, n=n)                           # zero-padded transform
assert X.shape == (n // 2 + 1,)
```

### `cupyx.scipy.linalg.lu_factor` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import lu_factor, lu_solve

A = cp.random.rand(100, 100) + 100 * cp.eye(100)
lu_piv = lu_factor(A, check_finite=False)
for _ in range(3):                                # reuse the factorization
    b = cp.random.rand(100)
    x = lu_solve(lu_piv, b, check_finite=False)
    assert np.allclose((A @ x).get(), b.get())
```

### `cupyx.scipy.linalg.solve_triangular` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import solve_triangular

L = cp.tril(cp.random.rand(20, 20)) + 20 * cp.eye(20)
b = cp.random.rand(20)
x = solve_triangular(L, b, lower=True)
assert np.allclose((L @ x).get(), b.get())
```

### `cupyx.scipy.sparse.csr_matrix` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as sp

data = cp.asarray([1.0, 2.0, 3.0, 4.0])
indices = cp.asarray([0, 2, 1, 2], dtype=cp.int32)
indptr = cp.asarray([0, 2, 3, 4], dtype=cp.int32)
A = sp.csr_matrix((data, indices, indptr), shape=(3, 3))
x = cp.asarray([1.0, 1.0, 1.0])
y = A @ x                                         # cuSPARSE SpMV
assert np.allclose(y.get(), [3.0, 3.0, 4.0])
assert np.allclose(A.toarray().get(), [[1, 0, 2], [0, 3, 0], [0, 0, 4]])
```

### `cupyx.scipy.sparse.diags` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as sp

n = 5
lap = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format='csr')   # 1-D Laplacian
assert lap.format == 'csr'
assert np.allclose(lap.toarray().get()[1, :3], [-1.0, 2.0, -1.0])
```

### `cupyx.scipy.sparse.linalg.cg` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse.linalg import cg

n = 200
A = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format='csr')
b = cp.ones(n)
x, info = cg(A, b, rtol=1e-10, maxiter=1000)
assert info == 0                                  # converged
assert np.allclose((A @ x).get(), b.get(), atol=1e-6)
```

### `cupyx.scipy.sparse.linalg.eigsh` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse.linalg import eigsh

n = 100
A = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format='csr')
w = eigsh(A, k=3, which='LA', return_eigenvectors=False)
expected = 2 - 2 * np.cos(np.pi * np.arange(n - 2, n + 1) / (n + 1))   # analytic spectrum
assert np.allclose(np.sort(w.get()), np.sort(expected), atol=1e-6)
```

### `cupyx.scipy.sparse.linalg.gmres` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse.linalg import gmres

n = 100
A = sp.diags([-1.0, 3.0, -0.5], [-1, 0, 1], shape=(n, n), format='csr')   # non-symmetric
b = cp.ones(n)
x, info = gmres(A, b, rtol=1e-10, restart=50, maxiter=1000)
assert info == 0
assert np.allclose((A @ x).get(), b.get(), atol=1e-6)
```

### `cupyx.scipy.sparse.linalg.spsolve` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx.scipy.sparse.linalg import spsolve

n = 50
A = sp.diags([1.0, 4.0, 1.0], [-1, 0, 1], shape=(n, n), format='csr')
b = cp.arange(n, dtype=cp.float64)
x = spsolve(A, b)
assert np.allclose((A @ x).get(), b.get())
```

## nvmath-python

### `nvmath.fft.FFT` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import nvmath

x = cp.random.rand(32, 128).astype(cp.complex64)
with nvmath.fft.FFT(x, axes=[-1]) as f:
    f.plan()
    X = f.execute(direction="forward")
    x2 = cp.random.rand(32, 128).astype(cp.complex64)
    f.reset_operand(x2)                           # reuse the plan
    X2 = f.execute(direction="forward")
assert np.allclose(X.get(), np.fft.fft(x.get(), axis=-1), rtol=1e-3, atol=1e-3)
assert np.allclose(X2.get(), np.fft.fft(x2.get(), axis=-1), rtol=1e-3, atol=1e-3)
```

### `nvmath.fft.fft` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import nvmath

x = cp.random.rand(16, 256).astype(cp.complex64)  # complex input required
X = nvmath.fft.fft(x, axes=[-1])                  # batched 1-D FFT along the last axis
assert isinstance(X, cp.ndarray) and X.dtype == cp.complex64
assert np.allclose(X.get(), np.fft.fft(x.get(), axis=-1), rtol=1e-3, atol=1e-3)
```

### `nvmath.fft.ifft` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import nvmath

x = cp.random.rand(8, 64).astype(cp.complex128)
X = nvmath.fft.fft(x, axes=[-1])
y = nvmath.fft.ifft(X, axes=[-1])                 # unnormalized inverse
assert np.allclose(y.get(), 64 * x.get())         # NumPy's ifft would return x
x_back = y / 64                                   # normalize explicitly
assert np.allclose(x_back.get(), x.get())
```

### `nvmath.fft.irfft` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import nvmath

x = cp.random.rand(4, 999)                        # odd length
X = nvmath.fft.rfft(x, axes=[-1])                 # 500 complex values
y_even = nvmath.fft.irfft(X, axes=[-1])
y_odd = nvmath.fft.irfft(X, axes=[-1], options={"last_axis_parity": "odd"})
assert y_even.shape == (4, 998)
assert y_odd.shape == (4, 999)
assert np.allclose((y_odd / 999).get(), x.get())  # unnormalized: divide by n
```

### `nvmath.fft.rfft` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import nvmath

x = cp.random.rand(8, 1000, dtype=cp.float32)
X = nvmath.fft.rfft(x, axes=[-1])
assert X.shape == (8, 501) and X.dtype == cp.complex64
assert np.allclose(X.get(), np.fft.rfft(x.get(), axis=-1), rtol=1e-3, atol=1e-2)
```

### `nvmath.linalg.advanced.Matmul` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import nvmath

a = cp.random.rand(64, 32, dtype=cp.float32)
b = cp.random.rand(32, 16, dtype=cp.float32)
with nvmath.linalg.advanced.Matmul(a, b) as mm:
    mm.plan()
    r1 = mm.execute()
    a2 = cp.random.rand(64, 32, dtype=cp.float32)
    mm.reset_operands(a=a2)                       # same shape/dtype: no re-planning
    r2 = mm.execute()
assert np.allclose(r1.get(), a.get() @ b.get(), rtol=1e-4, atol=1e-4)
assert np.allclose(r2.get(), a2.get() @ b.get(), rtol=1e-4, atol=1e-4)
```

### `nvmath.linalg.advanced.MatmulOptions` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import nvmath

a = cp.random.rand(256, 256, dtype=cp.float32)
b = cp.random.rand(256, 256, dtype=cp.float32)
ct = nvmath.linalg.advanced.MatmulComputeType.COMPUTE_32F_FAST_TF32
opts = nvmath.linalg.advanced.MatmulOptions(compute_type=ct)
r = nvmath.linalg.advanced.matmul(a, b, options=opts)
assert np.allclose(r.get(), a.get() @ b.get(), rtol=1e-2, atol=1e-1)   # TF32 tolerance
```

### `nvmath.linalg.advanced.matmul` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import nvmath

M, N, K = 128, 64, 256
a = cp.random.rand(M, K, dtype=cp.float32)
b = cp.random.rand(K, N, dtype=cp.float32)
bias = cp.random.rand(M, dtype=cp.float32) - 0.5  # one value per output ROW

r = nvmath.linalg.advanced.matmul(
    a, b,
    epilog=nvmath.linalg.advanced.MatmulEpilog.RELU_BIAS,
    epilog_inputs={"bias": bias},
)
expected = np.maximum(a.get() @ b.get() + bias.get()[:, None], 0)
assert isinstance(r, cp.ndarray) and r.shape == (M, N)
assert np.allclose(r.get(), expected, rtol=1e-3, atol=1e-3)
```

### `nvmath.sparse.advanced.direct_solver` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as sp
import nvmath

n = 200
A = sp.diags([-1.0, 4.0, -1.0], [-1, 0, 1], shape=(n, n), format='csr')
b = cp.asfortranarray(cp.random.rand(n, 3))       # 3 right-hand sides, column-major
x = nvmath.sparse.advanced.direct_solver(A, b)
assert x.shape == (n, 3)
assert np.allclose((A @ x).get(), b.get())
```

### `nvmath.tensor.binary_contraction` (verified on Tesla T4, CUDA 12.9, CuPy 14.2.0, nvmath-python 1.0.0 (2026-10-04))

```python
import numpy as np
import cupy as cp
import nvmath

a = cp.random.rand(8, 16, 32, dtype=cp.float32)
b = cp.random.rand(32, 24, dtype=cp.float32)
c = nvmath.tensor.binary_contraction("bij,jk->bik", a, b)
assert c.shape == (8, 16, 24)
assert np.allclose(c.get(), np.einsum("bij,jk->bik", a.get(), b.get()), rtol=1e-4, atol=1e-4)
```
