#!/usr/bin/env python3
"""Check Python code that uses CuPy / nvmath-python for API misuse, without running it.

    python scripts/check_gpu_code.py FILE [FILE ...]
    python scripts/check_gpu_code.py - < snippet.py

Exit status 0 when no errors were found, 1 when some were, 2 on usage errors or when the
checker is not installed. The check is static: no GPU, CuPy or nvmath-python needed.

This is a thin wrapper around the ``mathlib-mcp-check`` command of the mathlib-mcp
package (https://github.com/jarski/mathlib-mcp), which carries the symbol index.
"""

import sys

INSTALL_HINT = """\
The mathlib-mcp package is not installed in this Python environment.

Either install it once:
    uv tool install git+https://github.com/jarski/mathlib-mcp
    (or: pip install git+https://github.com/jarski/mathlib-mcp)

or run the checker without installing:
    uvx --from git+https://github.com/jarski/mathlib-mcp mathlib-mcp-check FILE
"""

try:
    from mathlib_mcp.cli import main
except ImportError:
    sys.stderr.write(INSTALL_HINT)
    sys.exit(2)

sys.exit(main())
