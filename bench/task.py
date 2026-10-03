"""Benchmark task definition.

A task is a small, precisely specified GPU programming problem. The model sees only
``prompt`` (plus a common preamble added by the harness) and must define the function
``entry_point``. Correctness is judged on a GPU against a NumPy/SciPy reference
computed on the host from the same inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

FAMILIES = ("cuBLAS", "cuFFT", "cuSOLVER", "cuSPARSE", "cuDSS", "cuTENSOR")


@dataclass(frozen=True)
class Case:
    seed: int
    expect_error: str | None = None  # dotted exception name, e.g. "numpy.linalg.LinAlgError"


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    difficulty: str  # easy | medium | hard
    prompt: str
    entry_point: str
    solution: str  # canonical GPU solution, validated on a real GPU
    make_inputs: Callable[[np.random.Generator, int], tuple]  # (rng, case index) -> host args
    reference: Callable[..., Any] | None = None  # host reference: same args -> expected
    check: Callable[[Any, Any, tuple], None] | None = None  # custom assertion
    rtol: float = 1e-5
    atol: float = 1e-8
    cases: tuple[Case, ...] = (Case(0), Case(1))
    required_api: tuple[str, ...] = ()  # APIs the prompt explicitly asks for
    requires: tuple[str, ...] = ()  # extra packages, e.g. ("nvmath",)
    trap: str = ""  # what makes the task hard for a model (shown in reports)
    naive: str | None = None  # plausible solution that falls into the trap (must fail)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"{self.id}: unknown family {self.family}")
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"{self.id}: unknown difficulty {self.difficulty}")


def assert_close(actual: Any, expected: Any, rtol: float, atol: float) -> None:
    """Recursive ``allclose`` over arrays, tuples and scalars with a useful message."""
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise AssertionError(f"expected a sequence of {len(expected)} values, "
                                 f"got {type(actual).__name__}")
        for a, e in zip(actual, expected, strict=True):
            assert_close(a, e, rtol, atol)
        return
    actual_arr, expected_arr = np.asarray(actual), np.asarray(expected)
    if actual_arr.shape != expected_arr.shape:
        raise AssertionError(f"shape {actual_arr.shape} != expected {expected_arr.shape}")
    if not np.allclose(actual_arr, expected_arr, rtol=rtol, atol=atol, equal_nan=False):
        diff = np.max(np.abs(actual_arr - expected_arr))
        raise AssertionError(f"values differ (max abs diff {diff:.3g}, rtol={rtol}, "
                             f"atol={atol})")
