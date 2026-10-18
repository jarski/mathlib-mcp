import io
import json

import pytest

from mathlib_mcp import cli

GOOD = """\
import cupy as cp

a = cp.random.rand(8, 8) + 8 * cp.eye(8)
b = cp.random.rand(8)
x = cp.linalg.solve(a, b)
"""
BAD = """\
import cupy as cp

x = cp.linalg.solve_triangular(cp.eye(3), cp.ones(3), lower=True)
"""


def test_clean_file_passes(tmp_path, capsys):
    path = tmp_path / "good.py"
    path.write_text(GOOD)
    assert cli.main([str(path)]) == 0
    assert capsys.readouterr().out.startswith("OK:")


def test_hallucinated_api_fails(tmp_path, capsys):
    path = tmp_path / "bad.py"
    path.write_text(BAD)
    assert cli.main([str(path)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("FAIL:") and "solve_triangular" in out


def test_several_files_are_labelled(tmp_path, capsys):
    good, bad = tmp_path / "good.py", tmp_path / "bad.py"
    good.write_text(GOOD)
    bad.write_text(BAD)
    assert cli.main([str(good), str(bad)]) == 1
    out = capsys.readouterr().out
    assert f"== {good}" in out and f"== {bad}" in out


def test_stdin_and_json(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(BAD))
    assert cli.main(["--json", "-"]) == 1
    record = json.loads(capsys.readouterr().out)
    assert record["file"] == "<stdin>" and record["ok"] is False
    assert any(i["code"] == "unknown-symbol" for i in record["issues"])


def test_missing_file_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main([str(tmp_path / "nope.py")])
    assert exc.value.code == 2
