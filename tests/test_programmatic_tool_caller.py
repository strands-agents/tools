"""Tests for programmatic_tool_caller tool."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from strands_tools.programmatic_tool_caller import (
    _create_async_tool_function,
    _execute_tool,
    _get_allowed_tools,
    _validate_code,
    programmatic_tool_caller,
)


class TestExecuteTool:
    """Tests for _execute_tool function."""

    def test_executes_callable_tool(self):
        mock_tool_func = MagicMock(return_value={"status": "success", "content": [{"text": "result"}]})
        mock_agent = MagicMock()
        # Mock agent.tool.test_tool() which is what _execute_tool now uses
        mock_agent.tool = MagicMock()
        mock_agent.tool.test_tool = mock_tool_func

        result = _execute_tool(mock_agent, "test_tool", {"arg": "value"})
        mock_tool_func.assert_called_once_with(record_direct_tool_call=False, arg="value")
        assert result == "result"

    def test_raises_error_for_missing_tool(self):
        mock_agent = MagicMock()
        # Simulate AttributeError when tool doesn't exist
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
        # Mock agent.tool.test_tool() which is what _execute_tool now uses
        mock_agent.tool = MagicMock()
        mock_agent.tool.test_tool = mock_tool_func

        async_func = _create_async_tool_function(mock_agent, "test_tool")
        assert asyncio.iscoroutinefunction(async_func)

        result = asyncio.run(async_func(arg="value"))
        assert result == "async result"


class TestValidateCode:
    """Tests for _validate_code function."""

    def test_detects_dangerous_patterns(self):
        assert any("subprocess" in w for w in _validate_code("import subprocess"))
        assert any("eval" in w for w in _validate_code("eval('1+1')"))
        assert any("exec" in w for w in _validate_code("exec('pass')"))

    def test_safe_code_has_no_warnings(self):
        code = "result = await calculator(expression='2+2')\nprint(result)"
        assert len(_validate_code(code)) == 0


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
        # Mock agent.tool.calculator() which is what _execute_tool now uses
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
        # Mock agent.tool.calculator() which is what _execute_tool now uses
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
        # Mock agent.tool.calculator() which is what _execute_tool now uses
        mock_context.agent.tool = MagicMock()
        mock_context.agent.tool.calculator = mock_tool_func

        # Should work - calculator is allowed
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
