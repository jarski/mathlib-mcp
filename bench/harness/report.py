"""Combine generations, static checks and GPU execution into the evaluation report.

    uv run python -m bench.harness.report bench/results/eval-v1

Reads ``generations.jsonl``, ``errors.jsonl`` and ``execution.jsonl`` from the run
directory and writes ``report.md``, ``summary.json`` and ``attempts.csv``.

Definitions:

* **solved**: the code passes every case of the task on the GPU *and* uses the API the
  prompt asks for (where it asks for one);
* **API errors**: ``check_snippet`` errors in the final code (non-existent functions,
  modules or enum members; unknown, positional-only or missing arguments);
* confidence intervals are 95 % percentile intervals from a bootstrap that resamples
  tasks (all repetitions of a task move together); differences between conditions are
  paired by task.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bench.harness.config import CONDITION_LABELS, CONDITIONS, MODELS
from bench.static import analyze
from bench.tasks import ALL_TASKS, TASKS_BY_ID

BOOTSTRAP = 10_000
CONTROL_TASKS = {t.id for t in ALL_TASKS if t.trap.startswith("Control task")}
NVMATH_TASKS = {t.id for t in ALL_TASKS if "nvmath" in t.requires}


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def outcome(gen: dict, execution: dict | None, static: dict) -> str:
    if gen["status"] != "ok":
        return gen["status"]  # refusal | truncated | no_code
    if execution is None:
        return "not_executed"
    status = execution["status"]
    if status == "exception":
        return f"exception:{execution.get('error_type', '?')}"
    if status == "pass" and static["required_api_missing"]:
        return "required_api_missing"
    return status


def bootstrap_mean(per_task: dict[str, float], rng: np.random.Generator
                   ) -> tuple[float, float, float]:
    values = np.array(list(per_task.values()), dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    samples = rng.choice(values, size=(BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), *np.percentile(samples, [2.5, 97.5]).tolist()


def per_task(rows: list[dict], key: str = "solved", tasks: set[str] | None = None
             ) -> dict[str, float]:
    """Mean of ``key`` per task (repetitions of a task are averaged)."""
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if tasks is None or row["task"] in tasks:
            by_task[row["task"]].append(float(row[key]))
    return {t: float(np.mean(v)) for t, v in by_task.items()}


def compare(current: dict, baseline: dict, rng: np.random.Generator) -> dict:
    """Paired (by task) solved-rate differences between this run and a baseline run."""
    def groups(result: dict) -> dict[tuple[str, str], list[dict]]:
        out: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for attempt in result["attempts"]:
            out[(attempt["model"], attempt["condition"])].append(attempt)
        return out

    now, before = groups(current), groups(baseline)
    comparison: dict = {}
    for (model, condition), rows in sorted(now.items()):
        mine = per_task(rows)
        for (base_model, base_condition), base_rows in sorted(before.items()):
            if base_model != model:
                continue
            theirs = per_task(base_rows)
            common = sorted(set(mine) & set(theirs))
            comparison[f"{model}: {condition} (this run) - {base_condition} (baseline)"] = {
                "baseline_solved": bootstrap_mean(theirs, rng),
                "delta": bootstrap_mean({t: mine[t] - theirs[t] for t in common}, rng),
            }
    return comparison


def build(run: Path) -> dict:
    generations = load_jsonl(run / "generations.jsonl")
    errors = load_jsonl(run / "errors.jsonl")
    executions = {row["id"]: row for row in load_jsonl(run / "execution.jsonl")}
    rng = np.random.default_rng(0)

    attempts = []
    for gen in generations:
        task = TASKS_BY_ID[gen["task"]]
        static = analyze(task, gen["code"]) if gen.get("code") else {
            "api_errors": [], "api_references": 0, "required_api_missing": []}
        result = outcome(gen, executions.get(gen["id"]), static)
        attempts.append({
            "id": gen["id"], "model": gen["model"], "condition": gen["condition"],
            "task": gen["task"], "family": gen["family"], "rep": gen["rep"],
            "gen_status": gen["status"], "outcome": result, "solved": result == "pass",
            "functional_pass": (executions.get(gen["id"]) or {}).get("status") == "pass",
            "api_errors": len(static["api_errors"]),
            "api_error_codes": sorted({e["code"] for e in static["api_errors"]}),
            "required_api_missing": static["required_api_missing"],
            "cost_usd": gen["cost_usd"], "latency_s": gen["latency_s"],
            "model_calls": gen["model_calls"], "tool_calls": len(gen["tool_calls"]),
            "tools_used": sorted({t["name"] for t in gen["tool_calls"]}),
            "input_tokens": gen["usage"]["input_tokens"],
            "cache_read_tokens": gen["usage"]["cache_read_input_tokens"],
            "cache_write_tokens": gen["usage"]["cache_creation_input_tokens"],
            "output_tokens": gen["usage"]["output_tokens"],
            "turn_limit_reached": gen["turn_limit_reached"],
        })

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for attempt in attempts:
        groups[(attempt["model"], attempt["condition"])].append(attempt)

    summary: dict = {"models": {}, "deltas": {}, "controls": {}, "errors": len(errors)}
    for (model, condition), rows in sorted(groups.items()):
        solved = bootstrap_mean(per_task(rows), rng)
        api_err = bootstrap_mean(per_task([{**r, "has_api_error": r["api_errors"] > 0}
                                           for r in rows], "has_api_error"), rng)
        n = len(rows)
        summary["models"].setdefault(model, {})[condition] = {
            "attempts": n,
            "solved": solved,
            "solved_traps": bootstrap_mean(per_task(rows, tasks=set(TASKS_BY_ID)
                                                    - CONTROL_TASKS), rng)[0],
            "solved_controls": bootstrap_mean(per_task(rows, tasks=CONTROL_TASKS), rng)[0],
            "solved_nvmath": bootstrap_mean(per_task(rows, tasks=NVMATH_TASKS), rng)[0],
            "solved_cupy": bootstrap_mean(per_task(rows, tasks=set(TASKS_BY_ID)
                                                   - NVMATH_TASKS), rng)[0],
            "functional_pass": sum(r["functional_pass"] for r in rows) / n,
            "api_error_rate": api_err,
            "outcomes": dict(Counter(r["outcome"] for r in rows).most_common()),
            "cost_total_usd": round(sum(r["cost_usd"] for r in rows), 4),
            "cost_per_attempt_usd": sum(r["cost_usd"] for r in rows) / n,
            "input_tokens_per_attempt": sum(r["input_tokens"] + r["cache_read_tokens"]
                                            + r["cache_write_tokens"] for r in rows) / n,
            "output_tokens_per_attempt": sum(r["output_tokens"] for r in rows) / n,
            "latency_s_per_attempt": sum(r["latency_s"] for r in rows) / n,
            "model_calls_per_attempt": sum(r["model_calls"] for r in rows) / n,
            "tool_calls_per_attempt": sum(r["tool_calls"] for r in rows) / n,
            "tool_use_rate": {tool: sum(tool in r["tools_used"] for r in rows) / n
                              for tool in ("search_api", "get_api_card", "get_example",
                                           "check_snippet")},
            "turn_limit_rate": sum(r["turn_limit_reached"] for r in rows) / n,
            "per_family": {family: bootstrap_mean(per_task(
                rows, tasks={t.id for t in ALL_TASKS if t.family == family}), rng)[0]
                for family in sorted({t.family for t in ALL_TASKS})},
            "per_task": {t: f"{sum(r['solved'] for r in rows if r['task'] == t)}/"
                            f"{sum(1 for r in rows if r['task'] == t)}"
                         for t in TASKS_BY_ID},
        }
    for model in sorted({m for m, _ in groups}):
        for a, b in (("mcp", "none"), ("docs", "none"), ("mcp", "docs")):
            if (model, a) not in groups or (model, b) not in groups:
                continue
            pa, pb = per_task(groups[(model, a)]), per_task(groups[(model, b)])
            common = sorted(set(pa) & set(pb))
            delta = {t: pa[t] - pb[t] for t in common}
            summary["deltas"][f"{model}: {a} - {b}"] = bootstrap_mean(delta, rng)

    for control, expected in (("oracle", "pass"), ("naive", "fail"), ("null", "fail")):
        rows = [r for i, r in executions.items() if i.startswith(control + "/")]
        ok = sum((r["status"] == "pass") == (expected == "pass") for r in rows)
        summary["controls"][control] = {"expected": expected, "as_expected": ok,
                                        "total": len(rows)}
    return {"summary": summary, "attempts": attempts}


# --------------------------------------------------------------------- rendering


def pct(value: float) -> str:
    return "–" if value != value else f"{100 * value:.0f}%"


def ci(triple) -> str:
    mean, low, high = triple
    return f"{pct(mean)} [{pct(low)}, {pct(high)}]"


def checker_version() -> str:
    """Commit of the check_snippet used for the static columns (+dirty if modified)."""
    repo = Path(__file__).resolve().parents[2]
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                              text=True).stdout.strip()
    dirty = git("status", "--porcelain", "--", "src/mathlib_mcp")
    return git("log", "-1", "--format=%h", "--", "src/mathlib_mcp") + ("-dirty" if dirty else "")


def render(result: dict, run: Path) -> str:
    summary = result["summary"]
    config = json.loads((run / "run.json").read_text()) if (run / "run.json").is_file() else {}
    env_file = run / "execution_environment.json"
    env = json.loads(env_file.read_text()) if env_file.is_file() else {}
    lines = ["# mathlib-mcp evaluation", ""]
    lines += [f"Run `{run.name}`: {len(config.get('tasks', []))} tasks × "
              f"{config.get('reps', '?')} samples per model and condition; code generated "
              f"at commit `{config.get('git_commit', '?')}`, executed on "
              f"{env.get('gpu', '?')} (CUDA {env.get('cuda_runtime', '?')}, CuPy "
              f"{env.get('cupy', '?')}, nvmath-python {env.get('nvmath', '?')}). Static "
              f"checks (API errors, required API) use `check_snippet` at commit "
              f"`{checker_version()}`.", ""]
    lines += ["Conditions: " + "; ".join(f"**{c}** = {CONDITION_LABELS[c]}"
                                         for c in CONDITIONS) + ".", ""]
    lines += ["## Solved rate", "",
              "Passes all GPU checks and uses the API the task asks for. 95% bootstrap "
              "CI over tasks.", "",
              "| Model | " + " | ".join(CONDITIONS) + " |",
              "|---|" + "---|" * len(CONDITIONS)]
    for model, conditions in summary["models"].items():
        cells = [ci(conditions[c]["solved"]) if c in conditions else "–" for c in CONDITIONS]
        lines.append(f"| {MODELS[model].id} | " + " | ".join(cells) + " |")
    lines += ["", "Paired differences (same tasks):", ""]
    lines += [f"- {name}: {'+' if v[0] >= 0 else ''}{100 * v[0]:.0f} points "
              f"[{100 * v[1]:+.0f}, {100 * v[2]:+.0f}]"
              for name, v in summary["deltas"].items()]
    lines += ["", "## API errors in the final code", "",
              "Share of attempts whose final code has at least one `check_snippet` error "
              "(hallucinated function/module/member, wrong or missing argument).", "",
              "| Model | " + " | ".join(CONDITIONS) + " |",
              "|---|" + "---|" * len(CONDITIONS)]
    for model, conditions in summary["models"].items():
        cells = [ci(conditions[c]["api_error_rate"]) if c in conditions else "–"
                 for c in CONDITIONS]
        lines.append(f"| {MODELS[model].id} | " + " | ".join(cells) + " |")
    lines += ["", "## Breakdown", "",
              "| Model / condition | trap tasks | control tasks | nvmath tasks | CuPy tasks |"
              " functional pass |", "|---|---|---|---|---|---|"]
    for model, conditions in summary["models"].items():
        for c in CONDITIONS:
            if c in conditions:
                s = conditions[c]
                lines.append(f"| {model} / {c} | {pct(s['solved_traps'])} | "
                             f"{pct(s['solved_controls'])} | {pct(s['solved_nvmath'])} | "
                             f"{pct(s['solved_cupy'])} | {pct(s['functional_pass'])} |")
    families = sorted({t.family for t in ALL_TASKS})
    lines += ["", "Per library family:", "", "| Model / condition | " + " | ".join(families)
              + " |", "|---|" + "---|" * len(families)]
    for model, conditions in summary["models"].items():
        for c in CONDITIONS:
            if c in conditions:
                lines.append(f"| {model} / {c} | " + " | ".join(
                    pct(conditions[c]["per_family"][f]) for f in families) + " |")
    lines += ["", "## Cost and effort per attempt", "",
              "| Model / condition | cost | input tokens (incl. cached) | output tokens | "
              "model calls | tool calls | latency |", "|---|---|---|---|---|---|---|"]
    for model, conditions in summary["models"].items():
        for c in CONDITIONS:
            if c in conditions:
                s = conditions[c]
                lines.append(f"| {model} / {c} | ${s['cost_per_attempt_usd']:.4f} | "
                             f"{s['input_tokens_per_attempt']:,.0f} | "
                             f"{s['output_tokens_per_attempt']:,.0f} | "
                             f"{s['model_calls_per_attempt']:.1f} | "
                             f"{s['tool_calls_per_attempt']:.1f} | "
                             f"{s['latency_s_per_attempt']:.1f} s |")
    total = sum(s["cost_total_usd"] for conds in summary["models"].values()
                for s in conds.values())
    lines += ["", f"Total generation cost: ${total:.2f}.", ""]
    lines += ["## Outcomes", ""]
    for model, conditions in summary["models"].items():
        for c in CONDITIONS:
            if c in conditions:
                outcomes = ", ".join(f"{k} {v}" for k, v in conditions[c]["outcomes"].items())
                lines.append(f"- **{model} / {c}**: {outcomes}")
    mcp_rows = [(m, s["mcp"]) for m, s in summary["models"].items() if "mcp" in s]
    if mcp_rows:
        lines += ["", "## Tool use (mcp condition)", "",
                  "| Model | search_api | get_api_card | get_example | check_snippet | "
                  "hit turn limit |", "|---|---|---|---|---|---|"]
        for model, s in mcp_rows:
            rates = s["tool_use_rate"]
            lines.append(f"| {model} | {pct(rates['search_api'])} | "
                         f"{pct(rates['get_api_card'])} | {pct(rates['get_example'])} | "
                         f"{pct(rates['check_snippet'])} | {pct(s['turn_limit_rate'])} |")
    lines += ["", "## Per task (solved / attempts)", "",
              "| Task | " + " | ".join(f"{m}/{c}" for m, conds in summary["models"].items()
                                     for c in CONDITIONS if c in conds) + " |",
              "|---|" + "---|" * sum(len(c) for c in summary["models"].values())]
    for task in ALL_TASKS:
        cells = [conds[c]["per_task"][task.id] for conds in summary["models"].values()
                 for c in CONDITIONS if c in conds]
        lines.append(f"| `{task.id}` ({task.family}) | " + " | ".join(cells) + " |")
    if summary.get("comparison"):
        lines += ["", f"## Compared with `{summary['comparison_run']}`", "",
                  "Paired by task (same tasks, same models).", "",
                  "| Comparison | baseline solved | difference |", "|---|---|---|"]
        for name, c in summary["comparison"].items():
            d = c["delta"]
            lines.append(f"| {name} | {ci(c['baseline_solved'])} | "
                         f"{'+' if d[0] >= 0 else ''}{100 * d[0]:.0f} points "
                         f"[{100 * d[1]:+.0f}, {100 * d[2]:+.0f}] |")
    lines += ["", "## Pipeline controls", ""]
    for control, c in summary["controls"].items():
        lines.append(f"- {control} solutions ({c['expected']} expected): "
                     f"{c['as_expected']}/{c['total']} as expected")
    lines.append(f"- infrastructure errors (excluded from scoring): {summary['errors']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--compare", type=Path, help="baseline run directory")
    args = parser.parse_args()
    result = build(args.run)
    if args.compare:
        result["summary"]["comparison_run"] = args.compare.name
        result["summary"]["comparison"] = compare(result, build(args.compare),
                                                  np.random.default_rng(1))
    (args.run / "summary.json").write_text(json.dumps(result["summary"], indent=1))
    with (args.run / "attempts.csv").open("w", newline="") as handle:
        fields = [k for k in result["attempts"][0] if k not in ("tools_used",
                                                                 "api_error_codes",
                                                                 "required_api_missing")]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["attempts"])
    report = render(result, args.run)
    (args.run / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
