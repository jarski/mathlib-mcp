"""Generate candidate solutions with Claude under the three conditions.

    uv run python -m bench.harness.generate --out bench/results/eval-v1 \\
        --models haiku sonnet --conditions none docs mcp --reps 3

Writes, per run directory:

* ``generations.jsonl``: one row per (model, condition, task, rep) that produced a
  scorable response, with the extracted code, status, usage, cost and tool calls;
* ``errors.jsonl``: attempts that failed for infrastructure reasons (API errors after
  retries, served-model mismatch). They are never scored as model failures;
* ``transcripts/``: the full message history of every attempt;
* ``run.json``: the configuration snapshot.

Re-running with the same ``--out`` resumes: attempts already in ``generations.jsonl``
are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

from bench.harness.config import (
    CONDITIONS,
    DOCS_HEADER,
    FINAL_ANSWER_NUDGE,
    MAX_AGENT_TURNS,
    MODELS,
    SYSTEM_PROMPT,
    ModelConfig,
)
from bench.task import Task
from bench.tasks import ALL_TASKS, TASKS_BY_ID

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "src" / "mathlib_mcp" / "data" / "llms-full.txt"
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


class ServedModelMismatch(RuntimeError):
    pass


# --------------------------------------------------------------------- MCP tools


class McpTools:
    """The real mathlib-mcp server, called in-process (one instance per thread)."""

    _local = threading.local()

    @classmethod
    def server(cls):
        if not hasattr(cls._local, "server"):
            from mathlib_mcp.server import create_server

            cls._local.server = create_server()
        return cls._local.server

    @classmethod
    def definitions(cls) -> list[dict]:
        tools = asyncio.run(cls.server().list_tools())
        return sorted(({"name": t.name, "description": t.description,
                        "input_schema": t.input_schema} for t in tools),
                      key=lambda t: t["name"])

    @classmethod
    def instructions(cls) -> str:
        from mathlib_mcp.tools import server_instructions

        return server_instructions()

    @classmethod
    def call(cls, name: str, arguments: dict) -> tuple[str, bool]:
        try:
            result = asyncio.run(cls.server().call_tool(name, arguments))
        except Exception as error:  # noqa: BLE001 - reported to the model as a tool error
            return f"Tool error: {type(error).__name__}: {error}", True
        text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
        return text, bool(getattr(result, "is_error", False))


# ------------------------------------------------------------------- one attempt


@dataclass
class Attempt:
    model: ModelConfig
    condition: str
    task: Task
    rep: int

    @property
    def id(self) -> str:
        return f"{self.model.key}/{self.condition}/{self.task.id}/r{self.rep}"


def system_for(condition: str) -> Any:
    if condition == "none":
        return SYSTEM_PROMPT
    if condition == "docs":
        return [{"type": "text", "text": SYSTEM_PROMPT},
                {"type": "text", "text": DOCS_HEADER + DOCS.read_text(encoding="utf-8"),
                 "cache_control": {"type": "ephemeral"}}]
    return [{"type": "text", "text": SYSTEM_PROMPT + "\n\n" + McpTools.instructions()}]


def extract_code(text: str) -> str | None:
    blocks = [b for b in _CODE_BLOCK.findall(text) if "def " in b]
    return blocks[-1].strip() + "\n" if blocks else None


def run_attempt(client: anthropic.Anthropic, attempt: Attempt, tools: list[dict]) -> dict:
    model, condition = attempt.model, attempt.condition
    request: dict[str, Any] = {"model": model.id, "max_tokens": model.max_tokens,
                               "system": system_for(condition), **model.params}
    if condition == "mcp":
        request["tools"] = tools
        request["cache_control"] = {"type": "ephemeral"}  # cache the growing conversation
    messages: list[dict] = [{"role": "user", "content": attempt.task.prompt}]
    calls: list[dict] = []
    tool_calls: list[dict] = []
    transcript: list[dict] = [{"role": "user", "content": attempt.task.prompt}]
    nudged = False
    response = None
    for turn in range(MAX_AGENT_TURNS + 1):
        if turn == MAX_AGENT_TURNS:  # still calling tools: ask for the final answer
            nudged = True
            messages.append({"role": "user", "content": FINAL_ANSWER_NUDGE})
            transcript.append({"role": "user", "content": FINAL_ANSWER_NUDGE})
            request["tool_choice"] = {"type": "none"}
        start = time.perf_counter()
        response = client.messages.create(messages=messages, **request)
        latency = time.perf_counter() - start
        if not (response.model == model.id or response.model.startswith(model.id + "-")):
            raise ServedModelMismatch(f"requested {model.id}, served {response.model}")
        usage = response.usage
        calls.append({
            "turn": turn, "latency_s": round(latency, 3), "stop_reason": response.stop_reason,
            "request_id": response._request_id, "served_model": response.model,
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
            "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
        })
        # append-only history: thinking blocks are passed back unchanged
        messages.append({"role": "assistant", "content": response.content})
        transcript.append({"role": "assistant",
                           "content": [b.to_dict() for b in response.content]})
        if response.stop_reason != "tool_use" or condition != "mcp" or nudged:
            break
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            started = time.perf_counter()
            text, is_error = McpTools.call(block.name, dict(block.input))
            tool_calls.append({"turn": turn, "name": block.name, "input": dict(block.input),
                               "is_error": is_error, "output_chars": len(text),
                               "seconds": round(time.perf_counter() - started, 3)})
            result = {"type": "tool_result", "tool_use_id": block.id, "content": text}
            if is_error:
                result["is_error"] = True
            results.append(result)
        messages.append({"role": "user", "content": results})
        transcript.append({"role": "user", "content": results})

    final_text = "\n".join(b.text for b in response.content if b.type == "text")
    code = extract_code(final_text)
    if response.stop_reason == "refusal":
        status = "refusal"
    elif response.stop_reason == "max_tokens":
        status = "truncated"
    elif code is None:
        status = "no_code"
    else:
        status = "ok"
    totals = {k: sum(c[k] for c in calls) for k in (
        "input_tokens", "output_tokens", "cache_creation_input_tokens",
        "cache_read_input_tokens")}
    return {
        "id": attempt.id, "model": model.key, "model_id": model.id,
        "served_model": calls[-1]["served_model"], "condition": condition,
        "task": attempt.task.id, "family": attempt.task.family,
        "difficulty": attempt.task.difficulty, "rep": attempt.rep, "status": status,
        "stop_reason": response.stop_reason, "turn_limit_reached": nudged,
        "code": code, "final_text": final_text, "model_calls": len(calls),
        "tool_calls": tool_calls, "usage": totals, "cost_usd": round(model.cost(totals), 6),
        "latency_s": round(sum(c["latency_s"] for c in calls), 3), "calls": calls,
    }, transcript


# ---------------------------------------------------------------------- runner


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                        choices=list(CONDITIONS))
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--tasks", nargs="*", default=None, help="task ids (default: all)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    tasks = [TASKS_BY_ID[t] for t in args.tasks] if args.tasks else ALL_TASKS
    out: Path = args.out
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    tools = McpTools.definitions()
    logging.getLogger().setLevel(logging.WARNING)  # the MCP server configures INFO logging
    for name in ("httpx", "httpx2", "mcp", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)
    config = {
        "git_commit": _git_commit(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {k: MODELS[k].__dict__ for k in args.models},
        "conditions": args.conditions, "reps": args.reps, "tasks": [t.id for t in tasks],
        "max_agent_turns": MAX_AGENT_TURNS, "system_prompt_sha": _sha(SYSTEM_PROMPT),
        "docs_sha": _sha(DOCS.read_text(encoding="utf-8")),
        "tools": [t["name"] for t in tools], "tools_sha": _sha(json.dumps(tools, sort_keys=True)),
    }
    (out / "run.json").write_text(json.dumps(config, indent=1, default=str))

    done = set()
    generations = out / "generations.jsonl"
    if generations.is_file():
        done = {json.loads(line)["id"] for line in generations.read_text().splitlines() if line}
    attempts = [Attempt(MODELS[m], c, t, r) for m in args.models for c in args.conditions
                for t in tasks for r in range(args.reps)]
    pending = [a for a in attempts if a.id not in done]
    print(f"{len(attempts)} attempts, {len(done)} already done, {len(pending)} to run")

    client = anthropic.Anthropic(max_retries=6)
    lock = threading.Lock()
    totals = {"cost": 0.0, "n": 0}

    def work(attempt: Attempt) -> None:
        try:
            row, transcript = run_attempt(client, attempt, tools)
        except (anthropic.APIError, ServedModelMismatch) as error:
            record = {"id": attempt.id, "error_class": type(error).__name__,
                      "message": str(error)[:1000], "time": time.strftime("%H:%M:%S")}
            with lock, (out / "errors.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"  ERROR {attempt.id}: {record['error_class']}", flush=True)
            return
        name = attempt.id.replace("/", "__") + ".json"
        (out / "transcripts" / name).write_text(json.dumps(transcript, indent=1, default=str))
        with lock:
            with generations.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            totals["cost"] += row["cost_usd"]
            totals["n"] += 1
            print(f"  [{totals['n']}/{len(pending)}] {row['id']:55s} {row['status']:9s} "
                  f"calls={row['model_calls']} tools={len(row['tool_calls'])} "
                  f"${row['cost_usd']:.4f}  total ${totals['cost']:.3f}", flush=True)

    # One (model, condition) group at a time keeps a group's cache prefix warm. In the
    # docs condition one attempt runs first and writes the cache; the rest read it.
    groups: dict[tuple[str, str], list[Attempt]] = {}
    for attempt in pending:
        groups.setdefault((attempt.model.key, attempt.condition), []).append(attempt)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for (model_key, condition), group in groups.items():
            print(f"-- {model_key} / {condition}: {len(group)} attempts", flush=True)
            if condition == "docs":
                work(group[0])
                group = group[1:]
            for future in as_completed([pool.submit(work, a) for a in group]):
                future.result()
    print(f"done: {totals['n']} generations, ${totals['cost']:.3f}")


if __name__ == "__main__":
    main()
