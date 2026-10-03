"""Execute candidate code for a task and judge it against the reference.

Each candidate runs in its own subprocess: a CUDA error (illegal address, OOM in a
bad state) or a hang must not poison the next candidate.

    python -m bench.runner --task fft_convolve --code-file candidate.py --out result.json

Statuses: pass | syntax_error | missing_entry_point | exception | wrong_answer |
not_on_gpu | expected_error_not_raised | timeout | crash
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from bench.task import Task, assert_close

REPO = Path(__file__).resolve().parents[1]
SHIM = os.environ.get("BENCH_CPU_SHIM") == "1"  # NumPy stand-in for local dry runs


class NotOnGPU(AssertionError):
    pass


# ----------------------------------------------------------------- conversions


def to_device(value: Any) -> Any:
    import cupy as cp
    import cupyx.scipy.sparse as cusparse
    import scipy.sparse

    if isinstance(value, np.ndarray):
        return cp.asarray(value)
    if scipy.sparse.issparse(value):
        return cusparse.csr_matrix(value) if value.format == "csr" else \
            cusparse.coo_matrix(value.tocoo())
    if isinstance(value, tuple) and value and all(isinstance(v, int) for v in value):
        return value  # a shape tuple
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(v) for v in value)
    return value


def to_host(value: Any, *, require_gpu: bool) -> Any:
    import cupy as cp
    import cupyx.scipy.sparse as cusparse

    if SHIM:
        require_gpu = False
    if isinstance(value, cp.ndarray) and not SHIM:
        return value.get()
    if cusparse.issparse(value):
        host = value.get() if hasattr(value, "get") and not SHIM else value
        return host
    if isinstance(value, np.ndarray):
        if require_gpu and value.ndim > 0:
            raise NotOnGPU("returned a NumPy array: results must stay on the GPU")
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(to_host(v, require_gpu=require_gpu) for v in value)
    return value


def _synchronize() -> None:
    if SHIM:
        return
    import cupy as cp

    cp.cuda.Device().synchronize()


def _exception_type(dotted: str) -> type[BaseException]:
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _copy_inputs(inputs: tuple) -> tuple:
    import copy

    return tuple(copy.deepcopy(v) for v in inputs)


# --------------------------------------------------------------------- judging


def execute(task: Task, code: str) -> dict[str, Any]:
    """Run all cases of a task in this process (the subprocess entry point)."""
    result: dict[str, Any] = {"task": task.id, "status": "pass", "cases": []}
    namespace: dict[str, Any] = {"__name__": "candidate"}
    try:
        compiled = compile(code, "<candidate>", "exec")
    except SyntaxError as error:
        return {**result, "status": "syntax_error", "message": str(error)}
    try:
        exec(compiled, namespace)  # noqa: S102 - running model output is the point
    except Exception as error:  # noqa: BLE001
        return {**result, "status": "exception", "error_type": type(error).__name__,
                "message": _short(error), "phase": "import"}
    func = namespace.get(task.entry_point)
    if not callable(func):
        return {**result, "status": "missing_entry_point",
                "message": f"no function named {task.entry_point}"}

    for index, case in enumerate(task.cases):
        rng = np.random.default_rng(1000 + case.seed)
        host_inputs = task.make_inputs(rng, index)
        expected = None
        if task.reference is not None and case.expect_error is None:
            expected = task.reference(*_copy_inputs(host_inputs))
        record: dict[str, Any] = {"case": index, "status": "pass"}
        start = time.perf_counter()
        try:
            device_inputs = to_device(_copy_inputs(host_inputs))
            output = func(*device_inputs)
            _synchronize()
            record["seconds"] = round(time.perf_counter() - start, 4)
            if case.expect_error is not None:
                record.update(status="expected_error_not_raised",
                              message=f"expected {case.expect_error} to be raised")
            else:
                actual = to_host(output, require_gpu=True)
                after = to_host(device_inputs, require_gpu=False)
                if task.check is not None:
                    task.check(actual, expected, after)
                else:
                    assert_close(actual, expected, task.rtol, task.atol)
        except NotOnGPU as error:
            record.update(status="not_on_gpu", message=str(error))
        except AssertionError as error:
            record.update(status="wrong_answer", message=_short(error))
        except Exception as error:  # noqa: BLE001
            if case.expect_error is not None and isinstance(
                    error, _exception_type(case.expect_error)):
                record.update(status="pass", message=f"raised {type(error).__name__}")
            else:
                record.update(status="exception", error_type=type(error).__name__,
                              message=_short(error),
                              where=_candidate_frame(error))
        record.setdefault("seconds", round(time.perf_counter() - start, 4))
        result["cases"].append(record)
        if record["status"] != "pass" and result["status"] == "pass":
            result["status"] = record["status"]
            for key in ("message", "error_type"):
                if key in record:
                    result[key] = record[key]
    return result


def _short(error: BaseException, limit: int = 500) -> str:
    text = f"{type(error).__name__}: {error}" if not isinstance(error, AssertionError) \
        else str(error)
    return text if len(text) <= limit else text[:limit] + "..."


def _candidate_frame(error: BaseException) -> str | None:
    frames = [f for f in traceback.extract_tb(error.__traceback__)
              if f.filename == "<candidate>"]
    return f"line {frames[-1].lineno}" if frames else None


# ------------------------------------------------------------------ isolation


def run_candidate(task_id: str, code: str, *, timeout: float = 180.0) -> dict[str, Any]:
    """Run one candidate in a fresh Python process with a timeout."""
    with tempfile.TemporaryDirectory() as tmp:
        code_file, out_file = Path(tmp) / "candidate.py", Path(tmp) / "result.json"
        code_file.write_text(code, encoding="utf-8")
        command = [sys.executable, "-m", "bench.runner", "--task", task_id,
                   "--code-file", str(code_file), "--out", str(out_file)]
        try:
            completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                                       timeout=timeout, env={**os.environ})
        except subprocess.TimeoutExpired:
            return {"task": task_id, "status": "timeout",
                    "message": f"no result within {timeout:.0f} s"}
        if out_file.is_file():
            return json.loads(out_file.read_text(encoding="utf-8"))
        tail = (completed.stderr or completed.stdout)[-800:]
        return {"task": task_id, "status": "crash",
                "message": f"exit code {completed.returncode}: {tail}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--code-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if SHIM:
        from bench import cpu_shim

        cpu_shim.install()
    from bench.tasks import TASKS_BY_ID

    result = execute(TASKS_BY_ID[args.task], args.code_file.read_text(encoding="utf-8"))
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
