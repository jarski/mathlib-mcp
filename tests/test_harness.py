"""Evaluation harness logic that does not need the API or a GPU."""

import pytest

pytest.importorskip("anthropic")

from bench.harness.config import MODELS  # noqa: E402
from bench.harness.generate import McpTools, extract_code  # noqa: E402
from bench.harness.report import outcome  # noqa: E402


def test_extract_code_takes_the_last_block_with_a_function():
    text = ("Here is a sketch:\n```python\nx = 1\n```\nFinal:\n```python\nimport cupy as cp\n\n"
            "def f(a):\n    return a\n```\nDone.")
    assert extract_code(text) == "import cupy as cp\n\ndef f(a):\n    return a\n"
    assert extract_code("no code here") is None


def test_cost_uses_cache_rates():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000,
             "cache_creation_input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000}
    assert MODELS["haiku"].cost(usage) == pytest.approx(1.0 + 5.0 + 1.25 + 0.10)
    assert MODELS["sonnet"].cost(usage) == pytest.approx(2.0 + 10.0 + 2.5 + 0.20)


def test_outcome_mapping():
    ok = {"status": "ok"}
    clean = {"required_api_missing": []}
    assert outcome({"status": "refusal"}, None, clean) == "refusal"
    assert outcome(ok, None, clean) == "not_executed"
    assert outcome(ok, {"status": "pass"}, clean) == "pass"
    assert outcome(ok, {"status": "pass"}, {"required_api_missing": ["nvmath.fft"]}) == \
        "required_api_missing"
    assert outcome(ok, {"status": "exception", "error_type": "TypeError"}, clean) == \
        "exception:TypeError"


def test_harness_uses_the_real_mcp_tool_surface():
    names = [tool["name"] for tool in McpTools.definitions()]
    assert names == ["check_snippet", "get_api_card", "get_example", "search_api"]
    text, is_error = McpTools.call("check_snippet",
                                   {"code": "import cupy as cp\ncp.linalg.cholesky(a, upper=True)"})
    assert not is_error and "unknown-keyword" in text
