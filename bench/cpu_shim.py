"""NumPy/SciPy stand-ins for ``cupy`` and ``cupyx`` (local dry runs only).

Lets the task definitions, references and canonical solutions be smoke-tested on a
machine without a GPU (``BENCH_CPU_SHIM=1``). It mirrors only the APIs the canonical
solutions use and does not model GPU-specific behaviour; the real verdict comes from
``bench/gpu_validate.py`` on a GPU. Tasks that need nvmath-python are skipped.
"""

from __future__ import annotations

import contextlib
import sys
import types

import numpy as np
import scipy.fft
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


class _Proxy(types.ModuleType):
    def __init__(self, name: str, backend, **extra):
        super().__init__(name)
        self._backend = backend
        self.__dict__.update(extra)

    def __getattr__(self, item):
        return getattr(self._backend, item)


def _gemm(transa, transb, a, b, out=None, alpha=1.0, beta=0.0):
    ops = {"N": lambda m: m, "T": lambda m: m.T, "H": lambda m: m.conj().T}
    product = alpha * (ops[transa](a) @ ops[transb](b))
    if out is None:
        return product
    out[...] = product + beta * out
    return out


@contextlib.contextmanager
def _errstate(*, linalg=None, **others):
    if others:
        raise NotImplementedError(f"cupyx.errstate does not support {sorted(others)}")
    yield


class _Device:
    def __init__(self, *_):
        pass

    def synchronize(self):
        pass


def install() -> None:
    cublas = _Proxy("cupy.cublas", types.SimpleNamespace(), gemm=_gemm)
    cuda = _Proxy("cupy.cuda", types.SimpleNamespace(), Device=_Device)
    cupy = _Proxy("cupy", np, ndarray=np.ndarray, asnumpy=lambda a, **_: np.asarray(a),
                  linalg=np.linalg, fft=np.fft, random=np.random, cublas=cublas, cuda=cuda)
    sparse = _Proxy("cupyx.scipy.sparse", scipy.sparse)
    sparse_linalg = _Proxy("cupyx.scipy.sparse.linalg", scipy.sparse.linalg)
    sparse.linalg = sparse_linalg
    cupyx_scipy = _Proxy("cupyx.scipy", types.SimpleNamespace(), sparse=sparse,
                         linalg=scipy.linalg, fft=scipy.fft)
    cupyx = _Proxy("cupyx", types.SimpleNamespace(), scipy=cupyx_scipy, errstate=_errstate)
    sys.modules.update({
        "cupy": cupy, "cupy.linalg": np.linalg, "cupy.fft": np.fft, "cupy.cublas": cublas,
        "cupy.cuda": cuda, "cupyx": cupyx, "cupyx.scipy": cupyx_scipy,
        "cupyx.scipy.linalg": scipy.linalg, "cupyx.scipy.fft": scipy.fft,
        "cupyx.scipy.sparse": sparse, "cupyx.scipy.sparse.linalg": sparse_linalg,
    })
