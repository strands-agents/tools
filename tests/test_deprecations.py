"""Tests for the tools removed in v0.9.0.

Each tool module now raises ImportError on import, while its @deprecated marker and the
TYPE_CHECKING re-export in __init__ stay in place so type checkers and IDEs keep flagging it.
"""

import ast
import importlib
import pathlib
import re
import warnings

import pytest

import strands_tools

# (module, attribute) for every tool carrying @deprecated.
DEPRECATED_TOOLS = [
    ("a2a_client", "A2AClientToolProvider"),
    ("agent_core_memory", "AgentCoreMemoryToolProvider"),
    ("agent_graph", "agent_graph"),
    ("batch", "batch"),
    ("bright_data", "bright_data"),
    ("browser.browser", "Browser"),
    ("calculator", "calculator"),
    ("chat_video", "chat_video"),
    ("code_interpreter.code_interpreter", "CodeInterpreter"),
    ("cron", "cron"),
    ("current_time", "current_time"),
    ("diagram", "diagram"),
    ("editor", "editor"),
    ("elasticsearch_memory", "ElasticsearchMemoryTool"),
    ("elasticsearch_memory", "elasticsearch_memory"),
    ("environment", "environment"),
    ("file_read", "file_read"),
    ("file_write", "file_write"),
    ("generate_image", "generate_image"),
    ("generate_image_stability", "generate_image_stability"),
    ("graph", "graph"),
    ("handoff_to_user", "handoff_to_user"),
    ("http_request", "http_request"),
    ("image_reader", "image_reader"),
    ("exa", "exa_get_contents"),
    ("exa", "exa_search"),
    ("journal", "journal"),
    ("load_tool", "load_tool"),
    ("mcp_client", "mcp_client"),
    ("mem0_memory", "Mem0MemoryTool"),
    ("mem0_memory", "mem0_memory"),
    ("memory", "memory"),
    ("mongodb_memory", "MongoDBMemoryTool"),
    ("mongodb_memory", "mongodb_memory"),
    ("nova_reels", "nova_reels"),
    ("python_repl", "python_repl"),
    ("retrieve", "retrieve"),
    ("rss", "rss"),
    ("search_video", "search_video"),
    ("shell", "shell"),
    ("slack", "slack"),
    ("slack", "slack_send_message"),
    ("sleep", "sleep"),
    ("speak", "speak"),
    ("stop", "stop"),
    ("swarm", "swarm"),
    ("tavily", "tavily_crawl"),
    ("tavily", "tavily_extract"),
    ("tavily", "tavily_map"),
    ("tavily", "tavily_search"),
    ("think", "think"),
    ("use_agent", "use_agent"),
    ("use_aws", "use_aws"),
    ("use_computer", "use_computer"),
    ("use_llm", "use_llm"),
    ("workflow", "workflow"),
]

TOOL_MODULES = sorted({module_name for module_name, _ in DEPRECATED_TOOLS})

# Only tools whose name matches their module can be re-exported: the re-export has to
# resolve at runtime too, and ``from strands_tools import slack_send_message`` does not.
REEXPORTED_TOOLS = [(module_name, attr) for module_name, attr in DEPRECATED_TOOLS if module_name == attr]

SRC = pathlib.Path(strands_tools.__file__).parent

REMOVED_MESSAGE = re.compile(r"^\w+ was removed in v0\.9\.0\. ")

# Imports such as ``from strands.vended_tools import shell`` or
# ``from strands.vended_tools import http_request, web_fetch`` inside migration messages.
MIGRATION_IMPORT = re.compile(r"from ([\w.]+) import ([\w., ]+)")


def _removal_error(module_name):
    """The ImportError a removed tool raises."""
    with pytest.raises(ImportError, match=REMOVED_MESSAGE) as exc_info:
        importlib.import_module(f"strands_tools.{module_name}")
    return exc_info.value


def _module_of(path):
    return ".".join(path.relative_to(SRC).with_suffix("").parts)


def _tree(module_name):
    return ast.parse((SRC / f"{module_name.replace('.', '/')}.py").read_text(encoding="utf-8"))


def _is_removal_guard(node):
    """``if not TYPE_CHECKING: raise ImportError(...)``."""
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and getattr(node.test.operand, "id", None) == "TYPE_CHECKING"
        and isinstance(node.body[0], ast.Raise)
    )


def _message_literals(module_name):
    """Every _DEPRECATION_MESSAGE value and @deprecated message in the module's source."""
    literals = []
    for node in ast.walk(_tree(module_name)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_DEPRECATION_MESSAGE" for t in node.targets):
            literals.append(node.value.value)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            literals += [
                d.args[0].value
                for d in node.decorator_list
                if isinstance(d, ast.Call) and getattr(d.func, "id", None) == "deprecated"
            ]
    return literals


@pytest.mark.parametrize("module_name", TOOL_MODULES)
def test_tool_module_raises_import_error(module_name):
    """Importing a removed tool raises ImportError naming the module and the removal release."""
    assert _removal_error(module_name).name == f"strands_tools.{module_name}"


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

    The TYPE_CHECKING re-export in __init__ is the only thing that lets a checker reach
    the @deprecated tool and flag the import before it fails at runtime.
    """
    assert (module_name, attr) in _reexports()


def test_reexports_raise_the_removal_error_at_runtime():
    """``from strands_tools import <tool>`` must fail with the removal message.

    ``slack_send_message`` is the case to watch: it is not a submodule, so re-exporting it
    would fail with a bare "cannot import name" instead of the migration path.
    """
    for _, attr in _reexports():
        with pytest.raises(ImportError, match=REMOVED_MESSAGE):
            exec(f"from strands_tools import {attr}", {})


def test_importing_the_package_does_not_import_tool_modules():
    """The re-export is typing-only, so importing the package must not raise."""
    tree = ast.parse((SRC / "__init__.py").read_text(encoding="utf-8"))

    module_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert [alias.name for node in module_level_imports for alias in node.names] == ["TYPE_CHECKING"]


@pytest.mark.parametrize("module_name, attr", DEPRECATED_TOOLS)
def test_deprecation_message_is_a_literal_in_the_decorator(module_name, attr):
    """mypy only reports @deprecated when the message is a string literal.

    Passing the shared _DEPRECATION_MESSAGE constant instead silently disables the
    check, so pin the literal in place.
    """
    decorators = [
        decorator
        for node in ast.walk(_tree(module_name))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == attr
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call) and getattr(decorator.func, "id", None) == "deprecated"
    ]

    assert len(decorators) == 1
    assert isinstance(decorators[0].args[0], ast.Constant)


def test_every_removed_module_is_covered():
    """A module that raises the removal error but is missing from DEPRECATED_TOOLS goes untested."""
    guarded = {
        _module_of(path)
        for path in SRC.rglob("*.py")
        if any(map(_is_removal_guard, ast.parse(path.read_text(encoding="utf-8")).body))
    }
    assert guarded == set(TOOL_MODULES)


@pytest.mark.parametrize("module_name", TOOL_MODULES)
def test_removal_guard_precedes_every_import(module_name):
    """Raising before the tool's own imports shows the message even when its extra is not installed."""
    body = [node for node in _tree(module_name).body if not isinstance(node, ast.Expr)]  # skip the docstring

    assert isinstance(body[0], ast.ImportFrom) and [alias.name for alias in body[0].names] == ["TYPE_CHECKING"]
    assert _is_removal_guard(body[1])


MESSAGE_MODULES = sorted(
    _module_of(path) for path in SRC.rglob("*.py") if "@deprecated(" in path.read_text(encoding="utf-8")
)


@pytest.mark.parametrize("module_name", MESSAGE_MODULES)
def test_static_messages_match_the_removal_error(module_name):
    """Type checkers show the @deprecated text, so it must be the exact message the import raises.

    Comparing against the module's own literals pins both the tool name and its migration path.
    """
    literals = _message_literals(module_name)

    assert literals
    assert set(literals) == {str(_removal_error(module_name))}


@pytest.mark.parametrize("module_name", TOOL_MODULES)
def test_migration_import_resolves(module_name):
    """Every migration import in the removal message must resolve without a deprecation warning.

    Compatibility aliases can still exist while warning users to migrate again.
    Check attribute access separately from module import so unrelated dependency
    warnings are not mistaken for a deprecated replacement.
    """
    message = str(_removal_error(module_name))

    for module_path, symbols in MIGRATION_IMPORT.findall(message):
        module = importlib.import_module(module_path)
        for symbol in re.split(r"\s*,\s*", symbols):
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                assert hasattr(module, symbol)


def test_py_typed_marker_ships_with_the_package():
    """Without py.typed, type checkers skip the package and never see @deprecated."""
    assert (SRC / "py.typed").is_file()
