"""Command-line front end for the static checker.

    mathlib-mcp-check FILE [FILE ...]     # check Python files
    mathlib-mcp-check - < snippet.py      # or read stdin

Exit status 0 when no errors were found, 1 when some were, 2 on usage errors. The
check is static: it needs no GPU and no CuPy or nvmath-python installation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mathlib_mcp.checker import check_snippet, default_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mathlib-mcp-check",
        description="Check Python code that uses CuPy / nvmath-python for API misuse: "
                    "functions that do not exist, wrong keyword arguments, missing "
                    "arguments and submodules that were never imported.")
    parser.add_argument("files", nargs="*", metavar="FILE",
                        help="Python files to check; '-' (or no files) reads stdin")
    parser.add_argument("--json", action="store_true", help="one JSON object per file")
    args = parser.parse_args(argv)

    files = args.files or ["-"]
    if files == ["-"] and sys.stdin.isatty():
        parser.error("no files given and nothing piped to stdin")
    index = default_index()
    failed = False
    for name in files:
        if name == "-":
            label, code = "<stdin>", sys.stdin.read()
        else:
            path = Path(name)
            if not path.is_file():
                parser.error(f"not a file: {name}")
            label, code = name, path.read_text(encoding="utf-8")
        result = check_snippet(code, index)
        failed = failed or not result.ok
        if args.json:
            print(json.dumps({"file": label, **result.to_dict()}))
        else:
            if len(files) > 1:
                print(f"== {label}")
            print(result.to_text())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
