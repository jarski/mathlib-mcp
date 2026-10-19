"""Combine conditions from several runs into one run directory for a joint report.

    uv run python -m bench.harness.merge --out bench/results/eval-v1.2-all \\
        bench/results/eval-v1:none,docs bench/results/eval-v1.1-mcp:mcp \\
        bench/results/eval-v1.2-skill:skill

Each source is ``DIR:cond1,cond2``. The generations, executions and errors of those
conditions are copied as they are; the pipeline controls (oracle, naive, null) and the
execution environment come from the last source. ``run.json`` records where each
condition came from, so the joint report stays traceable to the original runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from bench.harness.report import load_jsonl

CONTROLS = ("oracle", "naive", "null")


def merge(out: Path, sources: list[tuple[Path, list[str]]]) -> dict:
    generations: list[dict] = []
    executions: list[dict] = []
    errors: list[dict] = []
    config: dict = {"merged_from": [], "conditions": [], "tasks": set(), "reps": 0,
                    "models": {}}
    for source, conditions in sources:
        wanted = set(conditions)
        rows = [r for r in load_jsonl(source / "generations.jsonl") if r["condition"] in wanted]
        ids = {r["id"] for r in rows}
        generations += rows
        executions += [r for r in load_jsonl(source / "execution.jsonl") if r["id"] in ids]
        errors += [r for r in load_jsonl(source / "errors.jsonl")
                   if r["id"].split("/")[1] in wanted]
        run = json.loads((source / "run.json").read_text())
        config["merged_from"].append({"run": source.name, "conditions": sorted(wanted),
                                      "git_commit": run.get("git_commit"),
                                      "attempts": len(rows)})
        config["conditions"] += sorted(wanted)
        config["tasks"] |= set(run["tasks"])
        config["reps"] = max(config["reps"], run["reps"])
        config["models"].update(run["models"])
    last = sources[-1][0]
    executions += [r for r in load_jsonl(last / "execution.jsonl")
                   if r["id"].split("/")[0] in CONTROLS]
    config["tasks"] = sorted(config["tasks"])
    config["git_commit"] = "merged"

    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("generations.jsonl", generations), ("execution.jsonl", executions),
                       ("errors.jsonl", errors)):
        (out / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "run.json").write_text(json.dumps(config, indent=1, default=str))
    if (last / "execution_environment.json").is_file():
        shutil.copy(last / "execution_environment.json", out / "execution_environment.json")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("sources", nargs="+", metavar="DIR:COND,COND",
                        help="run directory and the conditions to take from it")
    args = parser.parse_args()
    sources = []
    for item in args.sources:
        path, _, conditions = item.rpartition(":")
        if not path or not conditions:
            parser.error(f"expected DIR:COND[,COND], got {item!r}")
        sources.append((Path(path), conditions.split(",")))
    config = merge(args.out, sources)
    for part in config["merged_from"]:
        print(f"  {part['run']}: {', '.join(part['conditions'])} ({part['attempts']} attempts)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
