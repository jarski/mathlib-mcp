"""Validate the benchmark and the curated knowledge on a real GPU.

    python -m bench.gpu_validate --out results/gpu_validation.json

1. canonical task solutions        -> must pass
2. naive solutions of trap tasks   -> must fail (the trap is real on this GPU)
3. curated API card examples       -> must pass
4. facts behind the pitfalls       -> must hold (``observe`` facts are only recorded)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

from bench.facts import FACTS
from bench.runner import REPO, run_candidate
from bench.tasks import ALL_TASKS

CARDS = REPO / "src" / "mathlib_mcp" / "data" / "cards.json"


def environment() -> dict:
    info: dict = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        info["gpu"] = props["name"].decode() if isinstance(props["name"], bytes) \
            else props["name"]
        info["compute_capability"] = f"{props['major']}.{props['minor']}"
        info["memory_gb"] = round(props["totalGlobalMem"] / 2**30, 1)
        info["cuda_runtime"] = _cuda_version(cp.cuda.runtime.runtimeGetVersion())
        info["cuda_driver"] = _cuda_version(cp.cuda.runtime.driverGetVersion())
        info["cupy"] = cp.__version__
        try:
            from cupy_backends.cuda.libs import cusolver

            info["cusolver"] = cusolver._getVersionNumber()
        except Exception as error:  # noqa: BLE001
            info["cusolver"] = f"unknown ({error})"
    except Exception as error:  # noqa: BLE001
        info["cupy_error"] = repr(error)
    for package in ("nvmath", "numpy", "scipy"):
        try:
            module = __import__(package)
            info[package] = getattr(module, "__version__", "?")
        except Exception as error:  # noqa: BLE001
            info[package] = f"not available ({error})"
    return info


def _cuda_version(number: int) -> str:
    return f"{number // 1000}.{number % 1000 // 10}"


def has(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def run_snippet(code: str, timeout: float = 300.0) -> dict:
    """Run a code snippet in a fresh interpreter; report the exception type if any."""
    start = time.perf_counter()
    try:
        completed = subprocess.run([sys.executable, "-c", code], cwd=REPO, text=True,
                                   capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "seconds": timeout}
    seconds = round(time.perf_counter() - start, 3)
    if completed.returncode == 0:
        return {"status": "ok", "stdout": completed.stdout[-2000:], "seconds": seconds}
    last = [line for line in completed.stderr.strip().splitlines() if line.strip()]
    final = last[-1] if last else ""
    match = re.match(r"^([\w.]+(?:Error|Exception|Warning|Exit)\w*)\b:?\s*(.*)$", final)
    error_type = match.group(1).rsplit(".", 1)[-1] if match else "unknown"
    return {"status": "error", "error_type": error_type, "message": final[-600:],
            "stdout": completed.stdout[-1000:], "seconds": seconds}


def validate(skip: set[str]) -> dict:
    report: dict = {"environment": environment(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "solutions": {}, "naive": {}, "examples": {}, "facts": {}}
    nvmath_ok = has("nvmath")

    if "solutions" not in skip:
        for task in ALL_TASKS:
            if "nvmath" in task.requires and not nvmath_ok:
                report["solutions"][task.id] = {"status": "skipped", "message": "no nvmath"}
                continue
            report["solutions"][task.id] = run_candidate(task.id, task.solution)
            _progress("solution", task.id, report["solutions"][task.id]["status"])

    if "naive" not in skip:
        for task in ALL_TASKS:
            if task.naive is None or ("nvmath" in task.requires and not nvmath_ok):
                continue
            result = run_candidate(task.id, task.naive)
            result["trap_confirmed"] = result["status"] != "pass"
            report["naive"][task.id] = result
            _progress("naive", task.id, result["status"])

    if "examples" not in skip:
        cards = json.loads(CARDS.read_text(encoding="utf-8"))["cards"]
        for card in cards:
            if not card.get("example"):
                continue
            if card["library"] == "nvmath-python" and not nvmath_ok:
                continue
            result = run_snippet(card["example"])
            result["code"] = card["example"]
            report["examples"][card["id"]] = result
            _progress("example", card["id"], result["status"])

    if "facts" not in skip:
        for fact in FACTS:
            if "nvmath" in fact.requires and not nvmath_ok:
                continue
            result = run_snippet(fact.code)
            result.update(claim=fact.claim, expect=fact.expect, cards=list(fact.cards))
            if fact.expect == "observe":
                result["holds"] = None
            elif fact.expect == "ok":
                result["holds"] = result["status"] == "ok"
            else:
                expected = fact.expect.split(":", 1)[1].split("|")
                result["holds"] = (result["status"] == "error"
                                   and result.get("error_type") in expected)
            report["facts"][fact.id] = result
            _progress("fact", fact.id, result["status"] if result["holds"] is None
                      else ("holds" if result["holds"] else "FAILS"))

    report["summary"] = summarize(report)
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return report


def summarize(report: dict) -> dict:
    solutions = report["solutions"].values()
    return {
        "solutions_pass": sum(r["status"] == "pass" for r in solutions),
        "solutions_total": sum(r["status"] != "skipped" for r in solutions),
        "traps_confirmed": sum(r["trap_confirmed"] for r in report["naive"].values()),
        "traps_total": len(report["naive"]),
        "examples_pass": sum(r["status"] == "ok" for r in report["examples"].values()),
        "examples_total": len(report["examples"]),
        "facts_hold": sum(bool(r["holds"]) for r in report["facts"].values()),
        "facts_checked": sum(r["holds"] is not None for r in report["facts"].values()),
    }


def _progress(kind: str, name: str, status: str) -> None:
    print(f"[{kind:8s}] {name:45s} {status}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=REPO / "bench" / "results" /
                        "gpu_validation.json")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["solutions", "naive", "examples", "facts"])
    args = parser.parse_args()
    report = validate(set(args.skip))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["environment"], indent=1))
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
