"""Tests for the ast_context tool."""

import pytest
from strands import Agent

from strands_tools import ast_context


@pytest.fixture
def agent():
    return Agent(tools=[ast_context])


@pytest.fixture
def py_file(tmp_path):
    path = tmp_path / "module.py"
    path.write_text(
        '"""A short module."""\n'
        "import os\n"
        "from collections import defaultdict as dd\n"
        "\n"
        "CONST: int = 1\n"
        "other = 2\n"
        "\n"
        "@staticmethod\n"
        "def hello(name, *args, **kwargs):\n"
        '    """Say hi."""\n'
        "    return name\n"
        "\n"
        "async def fetch(url, /, retries=3):\n"
        "    return None\n"
        "\n"
        "class Greeter(Base, metaclass=Meta):\n"
        "    def __init__(self, x):\n"
        "        self.x = x\n"
        "    async def greet(self, name):\n"
        "        return name\n"
    )
    return str(path)


def _payload(tool_response):
    for block in tool_response["content"]:
        if "json" in block:
            return block["json"]
    raise AssertionError("No JSON content block in tool response")


def test_outline_basic(agent, py_file):
    response = agent.tool.ast_context(path=py_file)
    assert response["status"] == "success"
    payload = _payload(response)

    assert payload["docstring"] == "A short module."

    assert {imp["module"] for imp in payload["imports"] if imp["kind"] == "import"} == {"os"}
    from_imports = [imp for imp in payload["imports"] if imp["kind"] == "from_import"]
    assert from_imports[0]["module"] == "collections"
    assert from_imports[0]["name"] == "defaultdict"
    assert from_imports[0]["asname"] == "dd"

    assert {a["name"] for a in payload["assignments"]} == {"CONST", "other"}

    funcs = {f["name"]: f for f in payload["functions"]}
    assert "hello" in funcs and "fetch" in funcs
    assert funcs["hello"]["signature"] == "def hello(name, *args, **kwargs)"
    assert funcs["fetch"]["kind"] == "async_function"
    assert funcs["fetch"]["signature"] == "async def fetch(url, /, retries=3)"
    assert funcs["hello"]["decorators"] == ["staticmethod"]

    classes = {c["name"]: c for c in payload["classes"]}
    assert "Greeter" in classes
    method_names = {m["name"] for m in classes["Greeter"]["methods"]}
    assert method_names == {"__init__", "greet"}
    assert "Base" in classes["Greeter"]["bases"]


def test_line_ranges_make_sense(agent, py_file):
    response = agent.tool.ast_context(path=py_file)
    payload = _payload(response)
    greeter = next(c for c in payload["classes"] if c["name"] == "Greeter")
    start, end = greeter["lines"]
    assert start < end


def test_optional_sections_can_be_omitted(agent, py_file):
    response = agent.tool.ast_context(path=py_file, include_imports=False, include_assignments=False)
    payload = _payload(response)
    assert "imports" not in payload
    assert "assignments" not in payload
    assert "classes" in payload and "functions" in payload


def test_syntax_error_is_reported(agent, tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(:\n    pass\n")

    response = agent.tool.ast_context(path=str(bad))
    assert response["status"] == "error"
    assert "SyntaxError" in _payload(response)["error"]


def test_missing_file_is_reported(agent, tmp_path):
    response = agent.tool.ast_context(path=str(tmp_path / "does_not_exist.py"))
    assert response["status"] == "error"
    assert "Could not read" in _payload(response)["error"]
