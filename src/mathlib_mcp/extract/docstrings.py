"""Parse Google- and NumPy-style docstrings into structured sections.

CuPy and nvmath-python mostly use Google style (``Args:``) with Sphinx roles; a few
functions use NumPy style (``Parameters`` + dashes). The output is plain Markdown-ish
text: Sphinx roles are reduced to inline code so that the text reads well for an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SECTION_ALIASES = {
    "args": "params", "arguments": "params", "parameters": "params", "params": "params",
    "keyword args": "params", "keyword arguments": "params", "other parameters": "params",
    "returns": "returns", "return": "returns", "yields": "returns",
    "raises": "raises",
    "example": "examples", "examples": "examples",
    "note": "notes", "notes": "notes",
    "warning": "warnings", "warnings": "warnings",
    "see also": "see_also", "seealso": "see_also",
    "semantics": "notes", "narrow-precision support": "notes",
    "attributes": "attributes",
    "references": "references",
}

_ROLE = re.compile(r":[\w:+-]+?:?`~?([^`<]+?)(?:\s*<[^>]+>)?`")
_LABEL = re.compile(r"^\s*\.\.\s+_[\w-]+:\s*$", re.M)
_DIRECTIVE_LINE = re.compile(r"^\s*\.\.\s+[\w-]+::.*$", re.M)
_MATH = re.compile(r":math:`([^`]+)`")
_DOUBLE_TICKS = re.compile(r"``([^`]+)``")
_LINK = re.compile(r"`([^`<]+?)\s*<([^>]+)>`_+")
_REF_SUFFIX = re.compile(r"`([^`]+)`_\b")
_PARAM_LINE = re.compile(r"^(\*{0,2}[A-Za-z_][\w, *]*?)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")


@dataclass
class ParsedDoc:
    summary: str = ""
    description: str = ""
    params: dict[str, str] = field(default_factory=dict)
    returns: str = ""
    raises: str = ""
    examples: list[str] = field(default_factory=list)
    notes: str = ""
    warnings: str = ""
    see_also: list[str] = field(default_factory=list)


def clean_rst(text: str) -> str:
    """Reduce reStructuredText/Sphinx markup to Markdown-friendly text."""
    text = _LABEL.sub("", text)
    text = _DIRECTIVE_LINE.sub("", text)
    text = _LINK.sub(r"\1 (\2)", text)
    text = _ROLE.sub(r"`\1`", text)
    text = _MATH.sub(r"`\1`", text)
    text = _DOUBLE_TICKS.sub(r"`\1`", text)
    text = _REF_SUFFIX.sub(r"`\1`", text)
    text = text.replace("\\*", "*")
    return text


def parse_docstring(doc: str | None) -> ParsedDoc:
    result = ParsedDoc()
    if not doc:
        return result
    lines = doc.expandtabs().splitlines()
    lines = _drop_signature_line(lines)
    sections = _split_sections(lines)

    body = _dedent(sections.pop("_body", []))
    paragraphs = _paragraphs(body)
    examples = _doctest_blocks(body)
    prose = [p for p in paragraphs if not p.lstrip().startswith(">>>")]
    if prose:
        first = clean_rst(" ".join(prose[0].split()))
        sentence = re.match(r"^(.+?[.!?])\s+(?=[A-Z`(])", first)
        result.summary = sentence.group(1) if sentence else first
        if re.fullmatch(r"\{\w+\}", result.summary):  # unexpanded docstring template
            result.summary = ""
        rest = [first[len(result.summary):].strip()] if sentence else []
        result.description = clean_rst("\n\n".join(rest + prose[1:])).strip()

    for name, raw in sections.items():
        text = _dedent(raw)
        if name == "params":
            result.params.update(_parse_params(text))
        elif name == "returns":
            result.returns = clean_rst(_join(text))
        elif name == "raises":
            result.raises = clean_rst(_join(text))
        elif name == "examples":
            examples += _doctest_blocks(text)
        elif name == "notes":
            result.notes = (result.notes + "\n\n" + clean_rst("\n".join(text)).strip()).strip()
        elif name == "warnings":
            result.warnings = clean_rst(_join(text))
        elif name == "see_also":
            result.see_also = _see_also_names(text)
    result.examples = examples
    return result


# ---------------------------------------------------------------------------


def _drop_signature_line(lines: list[str]) -> list[str]:
    """Docstrings of C-implemented callables often start with ``name(args)``."""
    if lines and re.match(r"^\s*[\w.]+\(.*\)\s*$", lines[0]):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return lines


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"_body": []}
    current = "_body"
    numpy_style = False  # NumPy sections keep unindented lines (parameter names)
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        header = None
        # Google style: "Args:" at column 0
        if line == line.lstrip() and stripped.endswith(":"):
            header = _SECTION_ALIASES.get(stripped[:-1].lower())
        # NumPy style: "Parameters" followed by "----------"
        elif (line == line.lstrip() and index + 1 < len(lines)
              and re.fullmatch(r"-{3,}", lines[index + 1].strip() or "x")):
            header = _SECTION_ALIASES.get(stripped.lower(), "_skip")
            index += 1
        # Sphinx directives: ".. seealso::", ".. note::", ".. warning::"
        directive = re.match(r"^\.\.\s+(\w+)::\s*(.*)$", stripped) if line == line.lstrip() \
            else None
        if directive:
            header = _SECTION_ALIASES.get(directive.group(1).lower(), "_skip")
            sections.setdefault(header, [])
            if directive.group(2):
                sections[header].append("    " + directive.group(2))
            current = header
        elif header:
            numpy_style = index > 0 and bool(re.fullmatch(r"-{3,}", lines[index].strip()))
            current = header
            sections.setdefault(current, [])
        elif (current != "_body" and not numpy_style and line and line == line.lstrip()
              and stripped):
            # unindented text after an indented section returns to the body
            current = "_body"
            sections[current].append(line)
        else:
            sections.setdefault(current, []).append(line)
        index += 1
    sections.pop("_skip", None)
    return sections


def _dedent(lines: list[str]) -> list[str]:
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if not indents:
        return []
    cut = min(indents)
    return [line[cut:] for line in lines]


def _paragraphs(lines: list[str]) -> list[str]:
    paragraphs, current = [], []
    for line in lines:
        if line.strip():
            current.append(line)
        elif current:
            paragraphs.append("\n".join(current))
            current = []
    if current:
        paragraphs.append("\n".join(current))
    return paragraphs


def _join(lines: list[str]) -> str:
    return " ".join(" ".join(lines).split())


def _parse_params(lines: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    current: list[str] | None = None
    names: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        line = line.replace("\\*", "*")
        is_header = line == line.lstrip()
        match = _PARAM_LINE.match(line.strip()) if is_header else None
        if match:
            if current is not None:
                for name in names:
                    params[name] = clean_rst(" ".join(" ".join(current).split()))
            names = [n.strip().lstrip("*") for n in match.group(1).split(",") if n.strip()]
            type_text = match.group(2)
            current = [f"({type_text})"] if type_text else []
            if match.group(3):
                current.append(match.group(3))
        elif is_header and current is None:
            continue
        elif current is not None:
            current.append(line.strip())
    if current is not None:
        for name in names:
            params[name] = clean_rst(" ".join(" ".join(current).split()))
    return params


def _doctest_blocks(lines: list[str]) -> list[str]:
    blocks, current = [], []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">>>") or (stripped.startswith("...") and current):
            current.append(stripped[4:] if len(stripped) > 3 else "")
        elif current and not stripped:  # a blank line ends the doctest block
            blocks.append("\n".join(current))
            current = []
        # any other line is doctest output: ignore it
    if current:
        blocks.append("\n".join(current))
    return [b for b in blocks if b.strip()]


def _see_also_names(lines: list[str]) -> list[str]:
    text = clean_rst(" ".join(lines))
    names = re.findall(r"`([\w.]+)`", text)
    if not names:
        names = re.findall(r"\b([a-z_][\w]*(?:\.[\w]+)+)\b", text)
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen
