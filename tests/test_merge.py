import json

from bench.harness.merge import merge


def write_run(path, conditions, reps, commit):
    path.mkdir()
    gens, execs = [], []
    for condition in conditions:
        for rep in range(reps):
            attempt = f"haiku/{condition}/task_a/r{rep}"
            gens.append({"id": attempt, "model": "haiku", "condition": condition,
                         "task": "task_a", "rep": rep})
            execs.append({"id": attempt, "status": "pass"})
    execs += [{"id": f"{control}/x/task_a/r0", "status": "pass"}
              for control in ("oracle", "naive", "null")]
    (path / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in gens))
    (path / "execution.jsonl").write_text("".join(json.dumps(r) + "\n" for r in execs))
    (path / "errors.jsonl").write_text(json.dumps({"id": f"haiku/{conditions[0]}/task_b/r0",
                                                   "error_class": "APIError"}) + "\n")
    (path / "run.json").write_text(json.dumps({"git_commit": commit, "tasks": ["task_a"],
                                               "reps": reps, "models": {"haiku": {}}}))
    (path / "execution_environment.json").write_text(json.dumps({"gpu": commit}))


def test_merge_takes_only_the_named_conditions_and_one_set_of_controls(tmp_path):
    write_run(tmp_path / "v1", ["none", "docs", "mcp"], reps=2, commit="aaa")
    write_run(tmp_path / "v2", ["skill"], reps=3, commit="bbb")
    out = tmp_path / "all"
    config = merge(out, [(tmp_path / "v1", ["none", "docs"]), (tmp_path / "v2", ["skill"])])

    rows = [json.loads(line) for line in (out / "generations.jsonl").read_text().splitlines()]
    assert sorted({r["condition"] for r in rows}) == ["docs", "none", "skill"]
    assert len(rows) == 2 * 2 + 3
    execs = [json.loads(line) for line in (out / "execution.jsonl").read_text().splitlines()]
    controls = [e for e in execs if e["id"].split("/")[0] in ("oracle", "naive", "null")]
    assert len(controls) == 3 and len(execs) == len(rows) + 3
    errors = [json.loads(line) for line in (out / "errors.jsonl").read_text().splitlines()]
    assert [e["id"].split("/")[1] for e in errors] == ["none", "skill"]
    assert config["reps"] == 3 and config["tasks"] == ["task_a"]
    assert [p["run"] for p in config["merged_from"]] == ["v1", "v2"]
    assert json.loads((out / "execution_environment.json").read_text()) == {"gpu": "bbb"}
