"""The Agent Skill is generated from the same cards as the server; check the package."""

import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from mathlib_mcp.cards import load_cards

ROOT = Path(__file__).resolve().parents[1]
NAME = "cupy-nvmath-gpu-math"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    base = tmp_path_factory.mktemp("skill")
    out, archive = base / NAME, base / f"{NAME}.zip"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_skill.py"),
                    "--out", str(out), "--zip", str(archive)],
                   check=True, capture_output=True, text=True)
    return out, archive


def test_frontmatter_follows_the_spec(built):
    out, _ = built
    text = (out / "SKILL.md").read_text()
    header = text[4:text.index("\n---\n", 4)]
    name = re.search(r"^name: (.+)$", header, re.M).group(1)
    description = re.search(r"^description: (.+)$", header, re.M).group(1)
    assert name == NAME == out.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name) and len(name) <= 64
    assert 1 <= len(description) <= 1024
    assert "license: MIT" in header
    assert text.count("\n") < 500


def test_references_cover_every_card(built):
    out, _ = built
    refs = out / "references"
    cards = load_cards().values()
    library_text = (refs / "cupy.md").read_text() + (refs / "nvmath.md").read_text()
    pitfalls = (refs / "pitfalls.md").read_text()
    examples = (refs / "examples.md").read_text()
    for card in cards:
        assert f"### `{card.id}`" in library_text
        if card.pitfalls:
            assert f"### `{card.id}`" in pitfalls
        if card.example:
            assert f"### `{card.id}`" in examples
    assert (refs / "index.md").read_text().startswith("# GPU math libraries for LLMs")
    assert "Apache License 2.0" in (out / "NOTICE.md").read_text()
    assert (out / "LICENSE").read_text().startswith("MIT License")


def test_zip_holds_the_skill_folder(built):
    _, archive = built
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
    assert f"{NAME}/SKILL.md" in names
    assert f"{NAME}/scripts/check_gpu_code.py" in names
    assert all(name.startswith(f"{NAME}/") for name in names)
    assert not any("__pycache__" in name for name in names)


def test_bundled_script_runs_the_checker(built, tmp_path):
    out, _ = built
    snippet = tmp_path / "snippet.py"
    snippet.write_text("import cupy as cp\nx = cp.linalg.solve_triangular(cp.eye(2), cp.ones(2))\n")
    run = subprocess.run([sys.executable, str(out / "scripts" / "check_gpu_code.py"),
                          str(snippet)], capture_output=True, text=True)
    assert run.returncode == 1
    assert run.stdout.startswith("FAIL:") and "solve_triangular" in run.stdout
