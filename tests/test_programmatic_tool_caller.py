"""Tests for programmatic_tool_caller tool."""

from unittest.mock import MagicMock, patch

import pytest

from strands_tools.programmatic_tool_caller import (
    _RESERVED_NAMESPACE_NAMES,
    _build_namespace,
    _create_async_tool_function,
    _execute_tool,
    _get_allowed_tools,
    programmatic_tool_caller,
)


class TestExecuteTool:
    """Tests for _execute_tool function."""

    def test_executes_callable_tool(self):
        mock_tool_func = MagicMock(return_value={"status": "success", "content": [{"text": "result"}]})
        mock_agent = MagicMock()
        mock_agent.tool = MagicMock()
        mock_agent.tool.test_tool = mock_tool_func

        result = _execute_tool(mock_agent, "test_tool", {"arg": "value"})
        mock_tool_func.assert_called_once_with(record_direct_tool_call=False, arg="value")
        assert result == "result"

    def test_raises_error_for_missing_tool(self):
        mock_agent = MagicMock()
        mock_agent.tool = MagicMock(spec=[])  # Empty spec means no attributes

        with pytest.raises(RuntimeError, match="not found"):
            _execute_tool(mock_agent, "missing", {})

    def test_raises_error_for_none_agent(self):
        with pytest.raises(RuntimeError, match="No agent available"):
            _execute_tool(None, "test_tool", {})


class TestCreateAsyncToolFunction:
    """Tests for _create_async_tool_function."""

    def test_creates_async_function(self):
        import asyncio

        mock_tool_func = MagicMock(return_value={"status": "success", "content": [{"text": "async result"}]})
        mock_agent = MagicMock()
        mock_agent.tool = MagicMock()
        mock_agent.tool.test_tool = mock_tool_func

        async_func = _create_async_tool_function(mock_agent, "test_tool")
        assert asyncio.iscoroutinefunction(async_func)

        result = asyncio.run(async_func(arg="value"))
        assert result == "async result"


class TestGetAllowedTools:
    """Tests for _get_allowed_tools function."""

    def test_returns_all_tools_by_default(self):
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {
            "calculator": MagicMock(),
            "file_read": MagicMock(),
            "programmatic_tool_caller": MagicMock(),
        }

        with patch.dict("os.environ", {}, clear=True):
            allowed = _get_allowed_tools(mock_agent)

        assert "calculator" in allowed
        assert "file_read" in allowed
        assert "programmatic_tool_caller" not in allowed  # Always excluded

    def test_respects_env_var(self):
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {
            "calculator": MagicMock(),
            "file_read": MagicMock(),
            "shell": MagicMock(),
        }

        with patch.dict("os.environ", {"PROGRAMMATIC_TOOL_CALLER_ALLOWED_TOOLS": "calculator,file_read"}):
            allowed = _get_allowed_tools(mock_agent)

        assert "calculator" in allowed
        assert "file_read" in allowed
        assert "shell" not in allowed

    def test_filters_nonexistent_tools_from_env(self):
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {"calculator": MagicMock()}

        with patch.dict("os.environ", {"PROGRAMMATIC_TOOL_CALLER_ALLOWED_TOOLS": "calculator,nonexistent"}):
            allowed = _get_allowed_tools(mock_agent)

        assert "calculator" in allowed
        assert "nonexistent" not in allowed


class TestBuildNamespace:
    """Tests for _build_namespace function."""

    def test_base_namespace_matches_python_repl(self):
        """Base namespace should be {"__name__": "__main__"} like python_repl."""
        import asyncio

        mock_agent = MagicMock()
        with patch.dict("os.environ", {}, clear=True):
            ns = _build_namespace(set(), mock_agent)

        assert ns["__name__"] == "__main__"
        assert ns["asyncio"] is asyncio

    def test_asyncio_always_present(self):
        """asyncio must always be in namespace (required for async wrapping)."""
        import asyncio

        mock_agent = MagicMock()
        with patch.dict("os.environ", {}, clear=True):
            ns = _build_namespace(set(), mock_agent)

        assert "asyncio" in ns
        assert ns["asyncio"] is asyncio

    def test_no_extra_modules_by_default(self):
        """Without PROGRAMMATIC_TOOL_CALLER_EXTRA_MODULES, only __name__ and asyncio."""
        mock_agent = MagicMock()
        with patch.dict("os.environ", {}, clear=True):
            ns = _build_namespace(set(), mock_agent)

        # Only __name__ and asyncio should be present (no json, re, math etc.)
        assert "json" not in ns
        assert "re" not in ns
        assert "math" not in ns

    def test_extra_modules_from_env(self):
        """PROGRAMMATIC_TOOL_CALLER_EXTRA_MODULES should inject specified modules."""
        import json
        import math
        import re

        mock_agent = MagicMock()
        with patch.dict("os.environ", {"PROGRAMMATIC_TOOL_CALLER_EXTRA_MODULES": "json,re,math"}):
            ns = _build_namespace(set(), mock_agent)

        assert ns["json"] is json
        assert ns["re"] is re
        assert ns["math"] is math

    def test_extra_modules_ignores_invalid(self):
        """Invalid module names should be skipped without error."""
        mock_agent = MagicMock()
        with patch.dict("os.environ", {"PROGRAMMATIC_TOOL_CALLER_EXTRA_MODULES": "json,nonexistent_module_xyz"}):
            ns = _build_namespace(set(), mock_agent)

        import json

        assert ns["json"] is json
        assert "nonexistent_module_xyz" not in ns

    def test_tools_injected_as_async_functions(self):
        """Tool functions should be injected as async callables."""
        import asyncio

        mock_agent = MagicMock()
        with patch.dict("os.environ", {}, clear=True):
            ns = _build_namespace({"calculator", "shell"}, mock_agent)

        assert "calculator" in ns
        assert "shell" in ns
        assert asyncio.iscoroutinefunction(ns["calculator"])
        assert asyncio.iscoroutinefunction(ns["shell"])

    def test_tool_named_asyncio_raises_error(self):
        """A tool named 'asyncio' must raise ValueError (namespace clash)."""
        mock_agent = MagicMock()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="asyncio"):
                _build_namespace({"asyncio", "calculator"}, mock_agent)

    def test_tool_clashing_with_extra_module_raises_error(self):
        """A tool whose name matches an extra module must raise ValueError."""
        mock_agent = MagicMock()
        with patch.dict("os.environ", {"PROGRAMMATIC_TOOL_CALLER_EXTRA_MODULES": "json"}):
            with pytest.raises(ValueError, match="json"):
                _build_namespace({"json", "calculator"}, mock_agent)

    def test_no_error_when_no_clashes(self):
        """No error should be raised when tool names don't clash with reserved names."""
        mock_agent = MagicMock()
        with patch.dict("os.environ", {"PROGRAMMATIC_TOOL_CALLER_EXTRA_MODULES": "json"}):
            ns = _build_namespace({"calculator", "shell"}, mock_agent)

        assert "calculator" in ns
        assert "shell" in ns
        assert "json" in ns

    def test_reserved_namespace_names_constant(self):
        """_RESERVED_NAMESPACE_NAMES should contain asyncio and __name__."""
        assert "asyncio" in _RESERVED_NAMESPACE_NAMES
        assert "__name__" in _RESERVED_NAMESPACE_NAMES


class TestProgrammaticToolCaller:
    """Tests for programmatic_tool_caller function."""

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    def test_requires_agent_context(self, mock_console, mock_input):
        mock_console.create.return_value = MagicMock()
        result = programmatic_tool_caller(code="print('hello')", tool_context=None)
        assert result["status"] == "error"
        assert "No agent context" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_executes_simple_code(self, mock_console, mock_input):
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(code="print('Hello!')", tool_context=mock_context)
        assert result["status"] == "success"
        assert "Hello!" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_async_tool_execution(self, mock_console, mock_input):
        mock_console.create.return_value = MagicMock()
        mock_tool_func = MagicMock(return_value={"status": "success", "content": [{"text": "42"}]})
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {"calculator": MagicMock()}
        mock_context.agent.tool = MagicMock()
        mock_context.agent.tool.calculator = mock_tool_func

        result = programmatic_tool_caller(
            code='result = await calculator(expression="6*7")\nprint(f"Result: {result}")',
            tool_context=mock_context,
        )
        assert result["status"] == "success"
        assert "Result: 42" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_asyncio_gather_works(self, mock_console, mock_input):
        mock_console.create.return_value = MagicMock()

        def mock_calc(record_direct_tool_call=False, **kwargs):
            return {"status": "success", "content": [{"text": str(eval(kwargs["expression"]))}]}

        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {"calculator": MagicMock()}
        mock_context.agent.tool = MagicMock()
        mock_context.agent.tool.calculator = mock_calc

        result = programmatic_tool_caller(
            code="""
results = await asyncio.gather(
    calculator(expression="1+1"),
    calculator(expression="2+2"),
)
print(f"Results: {results}")
""",
            tool_context=mock_context,
        )
        assert result["status"] == "success"
        assert "2" in result["content"][0]["text"]
        assert "4" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true", "PROGRAMMATIC_TOOL_CALLER_ALLOWED_TOOLS": "calculator"})
    def test_respects_allowed_tools_env_var(self, mock_console, mock_input):
        mock_console.create.return_value = MagicMock()
        mock_tool_func = MagicMock(return_value={"status": "success", "content": [{"text": "4"}]})
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {
            "calculator": MagicMock(),
            "shell": MagicMock(),
        }
        mock_context.agent.tool = MagicMock()
        mock_context.agent.tool.calculator = mock_tool_func

        result = programmatic_tool_caller(
            code='r = await calculator(expression="2+2")\nprint(r)',
            tool_context=mock_context,
        )
        assert result["status"] == "success"

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "false"})
    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    def test_user_cancellation(self, mock_console, mock_input):
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}
        mock_input.side_effect = ["n", "Changed my mind"]

        result = programmatic_tool_caller(code="print('hello')", tool_context=mock_context)
        assert result["status"] == "error"
        assert "cancelled" in result["content"][0]["text"].lower()

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_stderr_captured(self, mock_console, mock_input):
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(
            code="import sys; print('error', file=sys.stderr)",
            tool_context=mock_context,
        )
        assert result["status"] == "success"
        assert "[stderr]" in result["content"][0]["text"]
        assert "error" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_syntax_error_handled(self, mock_console, mock_input):
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(code="def invalid(:", tool_context=mock_context)
        assert result["status"] == "error"
        assert "yntax" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true", "PROGRAMMATIC_TOOL_CALLER_EXTRA_MODULES": "json,math"})
    def test_extra_modules_available_in_code(self, mock_console, mock_input):
        """Modules from PROGRAMMATIC_TOOL_CALLER_EXTRA_MODULES should be usable in code."""
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(
            code='print(json.dumps({"pi": math.pi}))',
            tool_context=mock_context,
        )
        assert result["status"] == "success"
        assert "3.14159" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_code_can_import_modules(self, mock_console, mock_input):
        """Code should be able to import modules on its own (like python_repl)."""
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(
            code="import json\nprint(json.dumps({'key': 'value'}))",
            tool_context=mock_context,
        )
        assert result["status"] == "success"
        assert "key" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_system_exit_caught(self, mock_console, mock_input):
        """sys.exit() in user code must NOT crash the host — it should return an error result."""
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(
            code="import sys; sys.exit(1)",
            tool_context=mock_context,
        )
        assert result["status"] == "error"
        assert "SystemExit" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_keyboard_interrupt_caught(self, mock_console, mock_input):
        """KeyboardInterrupt in user code must NOT crash the host."""
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(
            code="raise KeyboardInterrupt('test')",
            tool_context=mock_context,
        )
        assert result["status"] == "error"
        assert "KeyboardInterrupt" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_tool_named_asyncio_raises_namespace_error(self, mock_console, mock_input):
        """If a tool is named 'asyncio', it must fail with an error (not silently shadow)."""
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {
            "asyncio": MagicMock(),
            "calculator": MagicMock(),
        }

        result = programmatic_tool_caller(
            code="print('should not run')",
            tool_context=mock_context,
        )
        assert result["status"] == "error"
        assert "asyncio" in result["content"][0]["text"]
        assert "conflict" in result["content"][0]["text"].lower()

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true", "PROGRAMMATIC_TOOL_CALLER_EXTRA_MODULES": "json"})
    def test_tool_named_like_extra_module_raises_namespace_error(self, mock_console, mock_input):
        """If a tool name matches an extra module, it must fail with an error."""
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {
            "json": MagicMock(),
            "calculator": MagicMock(),
        }

        result = programmatic_tool_caller(
            code="print('should not run')",
            tool_context=mock_context,
        )
        assert result["status"] == "error"
        assert "json" in result["content"][0]["text"]
        assert "conflict" in result["content"][0]["text"].lower()
