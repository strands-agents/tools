"""AST-based context tool for Python source files.

Returns a structural outline of a Python file: top-level imports, module-level
assignments, classes (with their methods), and free functions, each tagged with
a line range. Lets an agent fetch just the shape of a file without reading the
full body.

Background:
    Coding agents commonly burn context by reading whole files when they only
    need to know what is defined where. The Dirac TerminalBench writeup calls
    this out as `EXCESSIVE_FILE_READS`. An outline is one cheap call that
    answers "what is in this file" so the agent can decide whether to read
    further.

Usage with Strands Agent:
    from strands import Agent
    from strands_tools import ast_context

    agent = Agent(tools=[ast_context])
    agent.tool.ast_context(path="/path/to/module.py")
"""

import ast
from os.path import expanduser
from typing import Any, Dict, List, Optional

from strands import tool

_MAX_FILE_BYTES = 5 * 1024 * 1024


def _signature(node: ast.AST) -> str:
    """Render a one-line signature for a function or async function definition."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""

    args = node.args
    posonly = list(args.posonlyargs)
    pos_args = list(args.args)
    pos_count = len(posonly) + len(pos_args)
    pos_defaults: List[Optional[str]] = [None] * (pos_count - len(args.defaults)) + [
        ast.unparse(d) for d in args.defaults
    ]
    kw_defaults: List[Optional[str]] = [ast.unparse(d) if d is not None else None for d in args.kw_defaults]

    parts: List[str] = []

    pos_index = 0
    for a in posonly:
        default = pos_defaults[pos_index]
        parts.append(f"{a.arg}={default}" if default is not None else a.arg)
        pos_index += 1
    if posonly:
        parts.append("/")

    for a in pos_args:
        default = pos_defaults[pos_index]
        parts.append(f"{a.arg}={default}" if default is not None else a.arg)
        pos_index += 1

    if args.vararg is not None:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")

    for a, default in zip(args.kwonlyargs, kw_defaults, strict=False):
        parts.append(f"{a.arg}={default}" if default is not None else a.arg)

    if args.kwarg is not None:
        parts.append(f"**{args.kwarg.arg}")

    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return f"{prefix}{node.name}({', '.join(parts)})"


def _line_range(node: ast.AST) -> List[int]:
    start = getattr(node, "lineno", 0)
    end = getattr(node, "end_lineno", start) or start
    return [start, end]


def _outline_function(node: ast.AST) -> Dict[str, Any]:
    return {
        "kind": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
        "name": getattr(node, "name", ""),
        "signature": _signature(node),
        "lines": _line_range(node),
        "decorators": [ast.unparse(d) for d in getattr(node, "decorator_list", [])],
    }


def _outline_class(node: ast.ClassDef) -> Dict[str, Any]:
    methods: List[Dict[str, Any]] = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(_outline_function(child))

    return {
        "kind": "class",
        "name": node.name,
        "lines": _line_range(node),
        "bases": [ast.unparse(b) for b in node.bases],
        "decorators": [ast.unparse(d) for d in node.decorator_list],
        "methods": methods,
    }


def _outline_imports(tree: ast.Module) -> List[Dict[str, Any]]:
    imports: List[Dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {
                        "kind": "import",
                        "module": alias.name,
                        "asname": alias.asname,
                        "line": node.lineno,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append(
                    {
                        "kind": "from_import",
                        "module": node.module,
                        "name": alias.name,
                        "asname": alias.asname,
                        "level": node.level,
                        "line": node.lineno,
                    }
                )
    return imports


def _outline_assignments(tree: ast.Module) -> List[Dict[str, Any]]:
    assignments: List[Dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.append({"name": target.id, "line": node.lineno})
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments.append({"name": node.target.id, "line": node.lineno})
    return assignments


@tool
def ast_context(path: str, include_imports: bool = True, include_assignments: bool = True) -> Dict[str, Any]:
    """Return a structural outline of a Python source file.

    Parses the file with the standard library `ast` module and returns its
    top-level shape: imports, module-level assignments, classes (with their
    methods), and free functions. Each entry carries a line range so the agent
    can follow up with a targeted read of just the relevant span.

    Args:
        path: Absolute or user-relative path to a `.py` file.
        include_imports: If False, omit the imports section.
        include_assignments: If False, omit the module-level assignments section.

    Returns:
        ToolResult dict. The JSON content block carries the structured outline:
            - path: The expanded path that was read.
            - docstring: The module docstring, if any.
            - imports: list of `{kind, module, ...}` entries (when included).
            - assignments: list of `{name, line}` entries (when included).
            - classes: list of class outlines, each with name, line range,
              bases, decorators, and method outlines.
            - functions: list of free-function outlines.
            - error: present only on error, with a human-readable message.

    Raises:
        Does not raise. Errors are returned in the `error` field.

    Examples:
        >>> ast_context(path="/repo/src/module.py")
        {"status": "success", "path": "/repo/src/module.py", "imports": [...],
         "classes": [...], "functions": [...]}
    """
    expanded = expanduser(path)

    def _error(msg: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "content": [
                {"text": f"ast_context error: {msg}"},
                {"json": {"path": expanded, "error": msg}},
            ],
        }

    try:
        with open(expanded, "rb") as fh:
            raw = fh.read(_MAX_FILE_BYTES + 1)
    except OSError as exc:
        return _error(f"Could not read file: {exc}")

    if len(raw) > _MAX_FILE_BYTES:
        return _error(f"File exceeds {_MAX_FILE_BYTES} bytes; refusing to parse.")

    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return _error(f"File is not valid UTF-8: {exc}")

    try:
        tree = ast.parse(source, filename=expanded)
    except SyntaxError as exc:
        return _error(f"SyntaxError at line {exc.lineno}: {exc.msg}")

    classes: List[Dict[str, Any]] = []
    functions: List[Dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(_outline_class(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_outline_function(node))

    docstring: Optional[str] = ast.get_docstring(tree)

    payload: Dict[str, Any] = {
        "path": expanded,
        "docstring": docstring,
        "classes": classes,
        "functions": functions,
    }
    if include_imports:
        payload["imports"] = _outline_imports(tree)
    if include_assignments:
        payload["assignments"] = _outline_assignments(tree)

    summary_lines = [f"Outline of {expanded}"]
    if include_imports:
        summary_lines.append(f"  imports: {len(payload['imports'])}")
    if include_assignments:
        summary_lines.append(f"  assignments: {len(payload['assignments'])}")
    summary_lines.append(f"  classes: {len(classes)}")
    summary_lines.append(f"  functions: {len(functions)}")

    return {
        "status": "success",
        "content": [
            {"text": "\n".join(summary_lines)},
            {"json": payload},
        ],
    }
