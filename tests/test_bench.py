"""The benchmark's own consistency (no GPU needed)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from bench.facts import FACTS  # noqa: E402
from bench.static import analyze  # noqa: E402
from bench.tasks import ALL_TASKS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TASK_IDS = [t.id for t in ALL_TASKS]


def test_task_set_shape():
    assert len(ALL_TASKS) == 25
    assert len(set(TASK_IDS)) == len(TASK_IDS)
    assert {t.family for t in ALL_TASKS} == {"cuBLAS", "cuFFT", "cuSOLVER", "cuSPARSE",
                                             "cuDSS", "cuTENSOR"}


@pytest.mark.parametrize("task", ALL_TASKS, ids=TASK_IDS)
def test_canonical_solution_passes_static_checks(task):
    result = analyze(task, task.solution)
    assert not result["api_errors"], result["api_errors"]
    assert not result["required_api_missing"], result["required_api_missing"]
    assert f"def {task.entry_point}(" in task.solution
    assert task.entry_point in task.prompt


@pytest.mark.parametrize("task", ALL_TASKS, ids=TASK_IDS)
def test_references_run_on_cpu(task):
    for index, case in enumerate(task.cases):
        inputs = task.make_inputs(np.random.default_rng(1000 + case.seed), index)
        if case.expect_error is None and task.reference is not None:
            task.reference(*inputs)


@pytest.mark.parametrize("task", [t for t in ALL_TASKS if "nvmath" not in t.requires],
                         ids=lambda t: t.id)
def test_canonical_solution_dry_run_with_cpu_shim(task):
    """Smoke test with NumPy standing in for CuPy; the GPU run is the real check."""
    code = (
        "import json, sys\n"
        "from bench import cpu_shim; cpu_shim.install()\n"
        "from bench.runner import execute\n"
        "from bench.tasks import TASKS_BY_ID\n"
        f"r = execute(TASKS_BY_ID[{task.id!r}], TASKS_BY_ID[{task.id!r}].solution)\n"
        "print(json.dumps(r))\n"
        "sys.exit(0 if r['status'] == 'pass' else 1)\n"
    )
    env = {**os.environ, "BENCH_CPU_SHIM": "1"}
    completed = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                               capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stdout[-1500:] + completed.stderr[-1500:]


def test_static_checker_flags_api_traps_in_naive_solutions():
    by_id = {t.id: t for t in ALL_TASKS}
    flagged = {tid for tid in ("cg_poisson_1d",)
               if analyze(by_id[tid], by_id[tid].naive)["api_errors"]}
    assert flagged == {"cg_poisson_1d"}


def test_facts_are_unique_and_reference_known_cards():
    from mathlib_mcp.cards import load_cards

    cards = load_cards()
    assert len({f.id for f in FACTS}) == len(FACTS)
    for fact in FACTS:
        assert fact.expect == "ok" or fact.expect == "observe" or \
            fact.expect.startswith("raises:")
        for card_id in fact.cards:
            assert card_id in cards, (fact.id, card_id)
