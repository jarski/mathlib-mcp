"""Record GPU-verified card examples from a ``gpu_validate`` report.

    uv run python -m bench.record_verification bench/results/gpu_validation.json
    uv run mathlib-mcp-build      # cards now say "verified on <GPU, CUDA, versions>"

Only examples whose code is unchanged since the GPU run count as verified (the build
compares the stored code with the current curation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.runner import REPO


def describe_environment(env: dict, date: str) -> str:
    parts = [env.get("gpu", "unknown GPU"), f"CUDA {env.get('cuda_runtime', '?')}",
             f"CuPy {env.get('cupy', '?')}"]
    if "not available" not in str(env.get("nvmath", "not available")):
        parts.append(f"nvmath-python {env['nvmath']}")
    return ", ".join(parts) + f" ({date})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, default=REPO / "curation" / "verified.json")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    passed = {card_id: {"code": result["code"]}
              for card_id, result in report["examples"].items() if result["status"] == "ok"}
    failed = sorted(set(report["examples"]) - set(passed))
    record = {
        "environment": describe_environment(report["environment"],
                                            report["started"].split(" ")[0]),
        "report": str(args.report.resolve().relative_to(REPO)),
        "examples": dict(sorted(passed.items())),
    }
    args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"{len(passed)} verified examples written to {args.out}")
    if failed:
        print("not verified:", ", ".join(failed))


if __name__ == "__main__":
    main()
