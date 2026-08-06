"""Tests for the deprecation markers shared by every deprecated tool.

The per-tool test modules cover the runtime warning each tool logs. These tests cover
the packaging contract that makes the marker reachable for type checkers and IDEs,
which the per-tool tests cannot see because they import the tool directly.
"""

import ast
import importlib
import pathlib
import types

import pytest

import strands_tools

# (module, attribute) for every tool carrying @deprecated.
DEPRECATED_TOOLS = [
    ("batch", "batch"),
    ("calculator", "calculator"),
    ("cron", "cron"),
    ("current_time", "current_time"),
    ("diagram", "diagram"),
    ("editor", "editor"),
    ("environment", "environment"),
    ("memory", "memory"),
    ("retrieve", "retrieve"),
    ("rss", "rss"),
    ("shell", "shell"),
    ("slack", "slack"),
    ("slack", "slack_send_message"),
    ("sleep", "sleep"),
    ("think", "think"),
]

SRC = pathlib.Path(strands_tools.__file__).parent


@pytest.mark.parametrize("module_name, attr", DEPRECATED_TOOLS)
def test_tool_carries_deprecation_marker(module_name, attr):
    """Each deprecated tool exposes __deprecated__ naming the error-log release."""
    tool = getattr(importlib.import_module(f"strands_tools.{module_name}"), attr)

    marker = getattr(tool, "__deprecated__", None)
    assert marker is not None
    assert "becomes an error log in v0.9.0" in marker


@pytest.mark.parametrize("module_name, attr", DEPRECATED_TOOLS)
def test_deprecated_tool_is_reexported_for_type_checkers(module_name, attr):
    """``from strands_tools import <tool>`` must resolve to the tool for type checkers.

    At runtime that import binds the module, which carries no marker, so the
    TYPE_CHECKING re-export in __init__ is the only thing that lets a checker reach
    the @deprecated tool. Assert against the source, since the block never executes.
    """
    tree = ast.parse((SRC / "__init__.py").read_text())

    reexports = {
        (node.module, alias.name)
        for block in tree.body
        if isinstance(block, ast.If)
        for node in ast.walk(block)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname == alias.name  # ``as`` form, required for re-export
    }

    assert (module_name, attr) in reexports


def test_importing_the_package_does_not_import_tool_modules():
    """The re-export is typing-only, so it must not pull in tool modules or their extras."""
    tree = ast.parse((SRC / "__init__.py").read_text())

    module_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert [alias.name for node in module_level_imports for alias in node.names] == ["TYPE_CHECKING"]


@pytest.mark.parametrize("module_name, attr", DEPRECATED_TOOLS)
def test_reexport_leaves_runtime_binding_untouched(module_name, attr):
    """The documented import still yields the module, so existing callers keep working."""
    imported = getattr(strands_tools, module_name)

    assert isinstance(imported, types.ModuleType)
    assert hasattr(imported, attr)


@pytest.mark.parametrize("module_name, attr", DEPRECATED_TOOLS)
def test_deprecation_message_is_a_literal_in_the_decorator(module_name, attr):
    """mypy only reports @deprecated when the message is a string literal.

    Passing the shared _DEPRECATION_MESSAGE constant instead silently disables the
    check, so pin the literal in place.
    """
    tree = ast.parse((SRC / f"{module_name}.py").read_text())

    decorators = [
        decorator
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == attr
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call) and getattr(decorator.func, "id", None) == "deprecated"
    ]

    assert len(decorators) == 1
    assert isinstance(decorators[0].args[0], ast.Constant)


@pytest.mark.parametrize("module_name, attr", DEPRECATED_TOOLS)
def test_decorator_literal_matches_the_logged_message(module_name, attr):
    """The literal and _DEPRECATION_MESSAGE must not drift apart.

    Inlining the literal means each message exists twice: once for type checkers and
    once for the logger.warning users see at runtime.
    """
    module = importlib.import_module(f"strands_tools.{module_name}")

    assert getattr(module, attr).__deprecated__ == module._DEPRECATION_MESSAGE


def test_py_typed_marker_ships_with_the_package():
    """Without py.typed, type checkers skip the package and never see @deprecated."""
    assert (SRC / "py.typed").is_file()
