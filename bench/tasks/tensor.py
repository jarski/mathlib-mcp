"""Tensor contractions: cuTENSOR through nvmath-python."""

import numpy as np

from bench.task import Task

TASKS = [
    Task(
        id="cutensor_contraction",
        family="cuTENSOR",
        difficulty="medium",
        prompt="""\
Implement `contract(a, b)` with nvmath-python.

`a` is a float32 CuPy array of shape (I, J, K) and `b` is float32 of shape (K, J, L).
Return the float32 (I, L) array `c[i, l] = sum over j, k of a[i, j, k] * b[k, j, l]`,
computed with cuTENSOR through `nvmath.tensor`.""",
        entry_point="contract",
        solution="""\
import nvmath


def contract(a, b):
    return nvmath.tensor.binary_contraction("ijk,kjl->il", a, b)
""",
        make_inputs=lambda rng, case: (rng.random((24, 16, 32), dtype=np.float32),
                                       rng.random((32, 16, 20), dtype=np.float32)),
        reference=lambda a, b: np.einsum("ijk,kjl->il", a.astype(np.float64),
                                         b.astype(np.float64)).astype(np.float32),
        rtol=1e-4,
        atol=1e-3,
        required_api=("nvmath.tensor",),
        requires=("nvmath",),
        trap="nvmath.tensor's expression-first call signature.",
    ),
]
