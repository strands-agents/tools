"""Tests for the deprecation markers shared by every deprecated tool.

The per-tool test modules cover the runtime warning each tool logs. These tests cover
the packaging contract that makes the marker reachable for type checkers and IDEs,
which the per-tool tests cannot see because they import the tool directly.
"""

import ast
import importlib
import os
import pathlib
import re
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

# Only tools whose name matches their module can be re-exported: the re-export has to
# resolve at runtime too, and ``from strands_tools import slack_send_message`` does not.
REEXPORTED_TOOLS = [(module_name, attr) for module_name, attr in DEPRECATED_TOOLS if module_name == attr]

SRC = pathlib.Path(strands_tools.__file__).parent

# ``from strands.vended_tools import bash`` inside a migration message.
MIGRATION_IMPORT = re.compile(r"from ([\w.]+) import (\w+)")

# shell pulls in termios/pty, which do not exist on Windows. Same stance as
# test_shell.py: skipped there until issue #17 is resolved.
WINDOWS_UNSUPPORTED = {"shell"}


def _import_tool_module(module_name):
    """Import a tool module, skipping the names that cannot import on this platform."""
    if os.name == "nt" and module_name in WINDOWS_UNSUPPORTED:
        pytest.skip(f"{module_name} does not import on windows (issue #17)")
    return importlib.import_module(f"strands_tools.{module_name}")


@pytest.mark.parametrize("module_name, attr", DEPRECATED_TOOLS)
def test_tool_carries_deprecation_marker(module_name, attr):
    """Each deprecated tool exposes __deprecated__ naming the error-log release."""
    tool = getattr(_import_tool_module(module_name), attr)

    marker = getattr(tool, "__deprecated__", None)
    assert marker is not None
    assert "becomes an error log in v0.9.0" in marker


def _reexports():
    """The (module, name) pairs re-exported from the TYPE_CHECKING block in __init__.

    The block never executes, so read it out of the source. Guarding on the test being
    ``TYPE_CHECKING`` matters: under any other condition the imports are dead to the
    runtime and to checkers alike.
    """
    tree = ast.parse((SRC / "__init__.py").read_text(encoding="utf-8"))

    return {
        (node.module, alias.name)
        for block in tree.body
        if isinstance(block, ast.If) and getattr(block.test, "id", None) == "TYPE_CHECKING"
        for node in ast.walk(block)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname == alias.name  # ``as`` form, required for re-export
    }


@pytest.mark.parametrize("module_name, attr", REEXPORTED_TOOLS)
def test_deprecated_tool_is_reexported_for_type_checkers(module_name, attr):
    """``from strands_tools import <tool>`` must resolve to the tool for type checkers.

    At runtime that import binds the module, which carries no marker, so the
    TYPE_CHECKING re-export in __init__ is the only thing that lets a checker reach
    the @deprecated tool.
    """
    assert (module_name, attr) in _reexports()


def _from_import(name):
    """Emulate ``from strands_tools import <name>``: attribute first, then submodule.

    A plain getattr is order-dependent: it only succeeds if something else already
    imported the submodule under its canonical name, which xdist workers do not
    guarantee.
    """
    try:
        return getattr(strands_tools, name)
    except AttributeError:
        return importlib.import_module(f"strands_tools.{name}")


def test_reexports_resolve_at_runtime():
    """A re-export a checker accepts but the runtime rejects would be worse than none.

    ``slack_send_message`` is the case to watch: it carries @deprecated but is not a
    submodule, so re-exporting it would make ``from strands_tools import
    slack_send_message`` typecheck and then raise ImportError.
    """
    for _, attr in _reexports():
        if os.name == "nt" and attr in WINDOWS_UNSUPPORTED:
            continue
        _from_import(attr)


def test_importing_the_package_does_not_import_tool_modules():
    """The re-export is typing-only, so it must not pull in tool modules or their extras."""
    tree = ast.parse((SRC / "__init__.py").read_text(encoding="utf-8"))

    module_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert [alias.name for node in module_level_imports for alias in node.names] == ["TYPE_CHECKING"]


@pytest.mark.parametrize("module_name, attr", REEXPORTED_TOOLS)
def test_reexport_leaves_runtime_binding_untouched(module_name, attr):
    """The documented import still yields the module, so existing callers keep working."""
    if os.name == "nt" and module_name in WINDOWS_UNSUPPORTED:
        pytest.skip(f"{module_name} does not import on windows (issue #17)")
    imported = _from_import(module_name)

    assert isinstance(imported, types.ModuleType)
    assert hasattr(imported, attr)


@pytest.mark.parametrize("module_name, attr", DEPRECATED_TOOLS)
def test_deprecation_message_is_a_literal_in_the_decorator(module_name, attr):
    """mypy only reports @deprecated when the message is a string literal.

    Passing the shared _DEPRECATION_MESSAGE constant instead silently disables the
    check, so pin the literal in place.
    """
    tree = ast.parse((SRC / f"{module_name}.py").read_text(encoding="utf-8"))

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
    module = _import_tool_module(module_name)

    assert getattr(module, attr).__deprecated__ == module._DEPRECATION_MESSAGE


@pytest.mark.parametrize("module_name, attr", DEPRECATED_TOOLS)
def test_migration_import_resolves(module_name, attr):
    """Every import a message spells out must actually work on the installed SDK.

    The per-tool tests assert these messages by substring, which cannot tell a real
    symbol from a plausible one: the messages shipped ``from strands.vended_tools import
    shell`` for four tools, and no released SDK exports it.
    """
    message = _import_tool_module(module_name)._DEPRECATION_MESSAGE

    for module_path, symbol in MIGRATION_IMPORT.findall(message):
        assert hasattr(importlib.import_module(module_path), symbol)


def test_py_typed_marker_ships_with_the_package():
    """Without py.typed, type checkers skip the package and never see @deprecated."""
    assert (SRC / "py.typed").is_file()
