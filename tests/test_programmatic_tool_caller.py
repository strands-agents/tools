"""Tests for programmatic_tool_caller tool."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from strands_tools.programmatic_tool_caller import (
    OutputCapture,
    _create_async_tool_function,
    _execute_tool,
    _validate_code,
    programmatic_tool_caller,
)


class TestOutputCapture:
    """Tests for OutputCapture class."""

    def test_captures_stdout(self):
        with OutputCapture() as capture:
            print("Hello, world!")
        assert "Hello, world!" in capture.get_output()

    def test_captures_stderr(self):
        with OutputCapture() as capture:
            print("Error message", file=sys.stderr)
        output = capture.get_output()
        assert "[stderr]" in output
        assert "Error message" in output

    def test_restores_streams_after_exit(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with OutputCapture():
            pass
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr


class TestExecuteTool:
    """Tests for _execute_tool function."""

    def test_executes_callable_tool(self):
        mock_tool = MagicMock(return_value={"status": "success", "content": [{"text": "result"}]})
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {"test_tool": mock_tool}

        result = _execute_tool(mock_agent, "test_tool", {"arg": "value"})

        mock_tool.assert_called_once_with(arg="value")
        assert result == "result"

    def test_raises_error_for_missing_tool(self):
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {}

        with pytest.raises(RuntimeError, match="not found"):
            _execute_tool(mock_agent, "missing", {})

    def test_raises_error_for_none_agent(self):
        with pytest.raises(RuntimeError, match="No agent available"):
            _execute_tool(None, "test_tool", {})

    def test_handles_error_status_in_result(self):
        mock_tool = MagicMock(return_value={"status": "error", "content": [{"text": "Tool failed"}]})
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {"test_tool": mock_tool}

        with pytest.raises(RuntimeError, match="Tool error"):
            _execute_tool(mock_agent, "test_tool", {})

    def test_returns_string_result_directly(self):
        mock_tool = MagicMock(return_value="direct string")
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {"test_tool": mock_tool}

        result = _execute_tool(mock_agent, "test_tool", {})
        assert result == "direct string"


class TestCreateAsyncToolFunction:
    """Tests for _create_async_tool_function."""

    def test_creates_async_function(self):
        import asyncio

        mock_tool = MagicMock(return_value={"status": "success", "content": [{"text": "async result"}]})
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {"test_tool": mock_tool}

        async_func = _create_async_tool_function(mock_agent, "test_tool")
        assert asyncio.iscoroutinefunction(async_func)

        result = asyncio.run(async_func(arg="value"))
        mock_tool.assert_called_once_with(arg="value")
        assert result == "async result"


class TestValidateCode:
    """Tests for _validate_code function."""

    def test_detects_subprocess_import(self):
        warnings = _validate_code("import subprocess")
        assert any("subprocess" in w for w in warnings)

    def test_detects_eval_call(self):
        warnings = _validate_code("result = eval('1 + 1')")
        assert any("eval" in w for w in warnings)

    def test_safe_code_has_no_warnings(self):
        code = """
result = await calculator(expression="2 + 2")
print(f"Result: {result}")
"""
        warnings = _validate_code(code)
        assert len(warnings) == 0


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

        result = programmatic_tool_caller(code="print('Hello, World!')", tool_context=mock_context)

        assert result["status"] == "success"
        assert "Hello, World!" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_returns_only_print_output(self, mock_console, mock_input):
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(
            code="""
x = 1 + 2
y = x * 10
print(f"Final: {y}")
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        assert "Final: 30" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_handles_syntax_error(self, mock_console, mock_input):
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(code="def broken(", tool_context=mock_context)

        assert result["status"] == "error"
        assert "Syntax error" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_handles_runtime_error(self, mock_console, mock_input):
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(code="x = 1 / 0", tool_context=mock_context)

        assert result["status"] == "error"
        assert "error" in result["content"][0]["text"].lower()

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_async_tool_execution(self, mock_console, mock_input):
        mock_console.create.return_value = MagicMock()
        mock_tool = MagicMock(return_value={"status": "success", "content": [{"text": "42"}]})
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {"calculator": mock_tool}

        result = programmatic_tool_caller(
            code="""
result = await calculator(expression="6 * 7")
print(f"Result: {result}")
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        assert "Result: 42" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_loop_with_await(self, mock_console, mock_input):
        mock_console.create.return_value = MagicMock()

        call_count = [0]

        def mock_calculator(**kwargs):
            call_count[0] += 1
            expr = kwargs.get("expression", "0")
            result = eval(expr)
            return {"status": "success", "content": [{"text": str(result)}]}

        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {"calculator": mock_calculator}

        result = programmatic_tool_caller(
            code="""
total = 0
for i in range(5):
    result = await calculator(expression=f"{i} * 10")
    total += int(result)
print(f"Total: {total}")
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        assert "Total: 100" in result["content"][0]["text"]
        assert call_count[0] == 5

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_empty_output_returns_no_output(self, mock_console, mock_input):
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(code="x = 1 + 2", tool_context=mock_context)

        assert result["status"] == "success"
        assert result["content"][0]["text"] == "(no output)"

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


class TestAsyncGather:
    """Tests for parallel execution with asyncio.gather."""

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_asyncio_gather_works(self, mock_console, mock_input):
        mock_console.create.return_value = MagicMock()

        def mock_calculator(**kwargs):
            expr = kwargs.get("expression", "0")
            return {"status": "success", "content": [{"text": str(eval(expr))}]}

        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {"calculator": mock_calculator}

        result = programmatic_tool_caller(
            code="""
results = await asyncio.gather(
    calculator(expression="1+1"),
    calculator(expression="2+2"),
    calculator(expression="3+3"),
)
print(f"Results: {results}")
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        assert "2" in result["content"][0]["text"]
        assert "4" in result["content"][0]["text"]
        assert "6" in result["content"][0]["text"]
