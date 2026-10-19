"""The skill condition exposes the skill folder and checker the way an agent uses them."""

import pytest

from bench.harness.config import CONDITION_LABELS, CONDITIONS
from bench.harness.generate import SkillTools, system_for


def test_skill_is_a_condition_with_a_label():
    assert "skill" in CONDITIONS
    assert "skill" in CONDITION_LABELS


def test_system_prompt_carries_the_skill_body_without_frontmatter():
    (block,) = system_for("skill")
    text = block["text"]
    assert "## Workflow" in text and "check_gpu_code" in text
    assert "\nname: cupy-nvmath-gpu-math" not in text
    assert block["cache_control"] == {"type": "ephemeral"}


def test_tool_definitions_have_schemas():
    tools = {t["name"]: t for t in SkillTools.definitions()}
    assert set(tools) == {"check_gpu_code", "grep", "read_file"}
    for tool in tools.values():
        assert tool["input_schema"]["type"] == "object" and tool["description"]


def test_grep_finds_a_card_with_context():
    text, is_error = SkillTools.call("grep", {"pattern": r"^### `cupy\.linalg\.solve`",
                                              "path": "references/cupy.md",
                                              "context_after": 6})
    assert not is_error
    assert text.startswith("references/cupy.md:") and "solve(a, b)" in text


def test_read_file_numbers_lines_and_pages():
    text, is_error = SkillTools.call("read_file", {"path": "references/pitfalls.md",
                                                   "offset": 1, "limit": 3})
    assert not is_error
    assert text.splitlines()[0].startswith("references/pitfalls.md (lines 1-3 of ")
    assert "continue with offset=4" in text
    assert text.splitlines()[1].startswith("     1\t# Pitfalls")


def test_paths_stay_inside_the_skill_folder():
    for path in ("../../pyproject.toml", "/etc/hosts", "references/../../README.md"):
        text, is_error = SkillTools.call("read_file", {"path": path})
        assert is_error and "Tool error" in text
    text, is_error = SkillTools.call("grep", {"pattern": "x", "path": "../README.md"})
    assert is_error


def test_checker_tool_runs_the_checker():
    bad = "import cupy as cp\nx = cp.linalg.solve_triangular(cp.eye(2), cp.ones(2))\n"
    text, is_error = SkillTools.call("check_gpu_code", {"code": bad})
    assert not is_error and text.startswith("FAIL:") and "solve_triangular" in text
    good = "import cupy as cp\nx = cp.linalg.solve(cp.eye(2), cp.ones(2))\n"
    text, _ = SkillTools.call("check_gpu_code", {"code": good})
    assert text.startswith("OK:")


@pytest.mark.parametrize("arguments", [{"pattern": "("}, {"pattern": "x", "path": "nope.md"}])
def test_bad_arguments_are_reported_not_raised(arguments):
    text, is_error = SkillTools.call("grep", arguments)
    assert is_error and text.startswith("Tool error")
