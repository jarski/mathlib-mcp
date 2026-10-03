"""Print a Markdown summary of a ``gpu_validate`` report.

    uv run python -m bench.summarize_validation bench/results/gpu_validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.tasks import TASKS_BY_ID


def summarize(report: dict) -> str:
    env, summary = report["environment"], report["summary"]
    lines = [
        f"GPU validation on **{env.get('gpu', '?')}** (compute capability "
        f"{env.get('compute_capability', '?')}, {env.get('memory_gb', '?')} GB), CUDA runtime "
        f"{env.get('cuda_runtime', '?')} / driver {env.get('cuda_driver', '?')}, CuPy "
        f"{env.get('cupy', '?')}, nvmath-python {env.get('nvmath', '?')}, "
        f"{report['started'].split(' ')[0]}.",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Canonical task solutions pass | {summary['solutions_pass']} / "
        f"{summary['solutions_total']} |",
        f"| Naive trap solutions fail as designed | {summary['traps_confirmed']} / "
        f"{summary['traps_total']} |",
        f"| Curated card examples pass | {summary['examples_pass']} / "
        f"{summary['examples_total']} |",
        f"| Pitfall facts hold | {summary['facts_hold']} / {summary['facts_checked']} |",
        "",
        "| Task | Canonical | Naive (trap) |",
        "|---|---|---|",
    ]
    for task_id, result in report["solutions"].items():
        naive = report["naive"].get(task_id)
        naive_text = "–" if naive is None else _naive_text(naive)
        family = TASKS_BY_ID[task_id].family if task_id in TASKS_BY_ID else "?"
        lines.append(f"| `{task_id}` ({family}) | {result['status']} | {naive_text} |")
    problems = [(k, v) for k, v in report["examples"].items() if v["status"] != "ok"]
    problems += [(k, v) for k, v in report["facts"].items() if v["holds"] is False]
    if problems:
        lines += ["", "Problems:"]
        lines += [f"- `{name}`: {result.get('message', result['status'])}"
                  for name, result in problems]
    observed = {k: v for k, v in report["facts"].items() if v["holds"] is None}
    if observed:
        lines += ["", "Observations:"]
        lines += [f"- `{name}`: {(v.get('stdout') or v.get('message') or v['status']).strip()}"
                  for name, v in observed.items()]
    return "\n".join(lines)


def _naive_text(result: dict) -> str:
    status = result["status"]
    if status == "exception":
        status = f"exception ({result.get('error_type', '?')})"
    return status if not result.get("trap_confirmed") else f"fails: {status}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(summarize(json.loads(args.report.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
