"""MCP server exposing LLM-friendly references for GPU math libraries.

Run with ``uvx mathlib-mcp`` (stdio), or from a checkout with ``uv run mathlib-mcp``.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from mathlib_mcp import tools

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def create_server() -> MCPServer:
    server = MCPServer("mathlib-mcp", instructions=tools.server_instructions())

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def search_api(query: str, library: str | None = None, limit: int = 8) -> str:
        """Search CuPy / nvmath-python APIs (cuBLAS, cuFFT, cuSOLVER, cuSPARSE, cuDSS,
        cuTENSOR) by task description or name, e.g. "batched least squares" or "rfft".

        Args:
            query: What the code needs to do, or part of a function name.
            library: Optional prefix filter such as "cupy", "cupyx.scipy.sparse" or "nvmath".
            limit: Maximum number of results (1-20).
        """
        return tools.search_api(query, library, limit)

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def get_api_card(name: str) -> str:
        """Get the exact signature, parameters, return value, pitfalls and a GPU-verified
        example for one function or class, e.g. "cupyx.scipy.sparse.linalg.cg".

        Args:
            name: Fully qualified dotted name.
        """
        return tools.get_api_card(name)

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def get_example(task: str) -> str:
        """Get runnable, GPU-verified example code (with pitfalls) for the curated APIs
        most relevant to a task, e.g. "matmul with bias and relu epilog".

        Args:
            task: Short description of what the code should do.
        """
        return tools.get_example(task)

    @server.tool(annotations=READ_ONLY, structured_output=False)
    def check_snippet(code: str) -> str:
        """Statically check Python code that uses CuPy / nvmath-python without running it:
        reports functions, modules and enum members that do not exist (with the closest
        real names), unknown or positional-only keyword arguments, too many positional
        arguments and missing required arguments.

        Args:
            code: Python source code.
        """
        return tools.check_snippet(code)

    @server.resource("mathlib://llms.txt", name="llms.txt", mime_type="text/markdown",
                     description="Index of all API cards (llms.txt format).")
    def llms_txt() -> str:
        return tools.llms_txt()

    return server


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()
