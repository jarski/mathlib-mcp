"""Execute generated solutions on a GPU, together with pipeline controls.

    python -m bench.harness.execute --candidates candidates.jsonl --out execution.jsonl

Every candidate runs in its own subprocess (``bench.runner.run_candidate``), one at a
time so that candidates cannot disturb each other's GPU memory. Three controls per task
run through the same pipeline: ``oracle`` (canonical solution, must pass), ``naive``
(trap solution, must fail) and ``null`` (returns None, must fail).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bench.gpu_validate import environment
from bench.runner import run_candidate
from bench.tasks import ALL_TASKS, TASKS_BY_ID


def controls() -> list[dict]:
    rows = []
    for task in ALL_TASKS:
        rows.append({"id": f"oracle/canonical/{task.id}/r0", "task": task.id,
                     "code": task.solution})
        if task.naive:
            rows.append({"id": f"naive/naive/{task.id}/r0", "task": task.id,
                         "code": task.naive})
        rows.append({"id": f"null/null/{task.id}/r0", "task": task.id,
                     "code": f"def {task.entry_point}(*args):\n    return None\n"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    candidates = [json.loads(line) for line in args.candidates.read_text().splitlines()
                  if line.strip()]
    rows = controls() + [c for c in candidates if c.get("code")]
    done = set()
    if args.out.is_file():
        done = {json.loads(line)["id"] for line in args.out.read_text().splitlines() if line}
    env_file = args.out.with_name("execution_environment.json")
    env_file.write_text(json.dumps(environment(), indent=1))
    print(f"{len(rows)} executions ({len(done)} done)", flush=True)
    start = time.time()
    with args.out.open("a") as handle:
        for index, row in enumerate(rows, 1):
            if row["id"] in done or row["task"] not in TASKS_BY_ID:
                continue
            result = run_candidate(row["task"], row["code"], timeout=args.timeout)
            result["id"] = row["id"]
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {row['id']:60s} {result['status']:26s} "
                  f"{time.time() - start:7.0f}s", flush=True)


if __name__ == "__main__":
    main()
