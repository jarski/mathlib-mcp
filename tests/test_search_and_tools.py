import asyncio

import pytest

from mathlib_mcp import tools
from mathlib_mcp.cards import load_cards, load_symbols
from mathlib_mcp.search import SearchIndex, default_search_index


@pytest.mark.parametrize(("query", "expected"), [
    ("solve sparse symmetric positive definite system iteratively",
     "cupyx.scipy.sparse.linalg.cg"),
    ("matrix multiply with fused bias and relu", "nvmath.linalg.advanced.matmul"),
    ("least squares fit", "cupy.linalg.lstsq"),
    ("batched 2d fft of images", "cupy.fft.fftn"),
    ("eigenvalues of a symmetric matrix", "cupy.linalg.eigh"),
    ("copy result back to host numpy", "cupy.asnumpy"),
    ("LU factorization reuse for many right hand sides", "cupyx.scipy.linalg.lu_factor"),
    ("cuDSS sparse direct solver", "nvmath.sparse.advanced.direct_solver"),
])
def test_search_finds_the_right_api_in_top_3(query, expected):
    ids = [hit.id for hit in default_search_index().search(query, k=3)]
    assert expected in ids, ids


def test_keyword_search_needs_no_embedding_model():
    index = SearchIndex(load_cards(), load_symbols(), embeddings=False)
    assert index.search("rfft", k=1)[0].id in ("cupy.fft.rfft", "cupyx.scipy.fft.rfft",
                                              "nvmath.fft.rfft")


def test_get_api_card_variants():
    assert "**Pitfalls**" in tools.get_api_card("cupyx.scipy.sparse.linalg.cg")
    assert "`cupy.matmul`" in tools.get_api_card("cupy.linalg.matmul")  # alias
    assert "no curated card" in tools.get_api_card("cupy.zeros")  # symbol-only
    assert "cupyx.scipy.linalg.lu_factor" in tools.get_api_card("cupy.linalg.lu_factor")


def test_get_example_returns_code_and_pitfalls():
    text = tools.get_example("matmul with bias and relu epilog")
    assert "```python" in text and "Pitfalls" in text


def test_mcp_server_lists_and_calls_tools():
    from mathlib_mcp.server import create_server

    async def run():
        server = create_server()
        names = {tool.name for tool in await server.list_tools()}
        result = await server.call_tool(
            "check_snippet", {"code": "import cupy as cp\ncp.linalg.cholesky(a, upper=True)"})
        return names, result.content[0].text

    names, text = asyncio.run(run())
    assert names == {"search_api", "get_api_card", "get_example", "check_snippet"}
    assert "unknown-keyword" in text
