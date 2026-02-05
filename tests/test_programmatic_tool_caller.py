"""Tests for programmatic_tool_caller tool."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from strands_tools.programmatic_tool_caller import (
    OutputCapture,
    _create_tool_function,
    _execute_tool,
    _validate_code,
    programmatic_tool_caller,
)


class TestOutputCapture:
    """Tests for OutputCapture class."""

    def test_captures_stdout(self):
        """Test that stdout is captured."""
        with OutputCapture() as capture:
            print("Hello, world!")
        assert "Hello, world!" in capture.get_output()

    def test_captures_stderr(self):
        """Test that stderr is captured."""
        with OutputCapture() as capture:
            print("Error message", file=sys.stderr)
        output = capture.get_output()
        assert "[stderr]" in output
        assert "Error message" in output

    def test_captures_both_stdout_and_stderr(self):
        """Test capturing both streams."""
        with OutputCapture() as capture:
            print("Standard output")
            print("Error output", file=sys.stderr)
        output = capture.get_output()
        assert "Standard output" in output
        assert "Error output" in output

    def test_restores_streams_after_exit(self):
        """Test that original streams are restored."""
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with OutputCapture():
            pass
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr


class TestExecuteTool:
    """Tests for _execute_tool function."""

    def test_executes_callable_tool(self):
        """Test executing a callable tool."""
        mock_tool = MagicMock(return_value={"status": "success", "content": [{"text": "result"}]})
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {"test_tool": mock_tool}

        result = _execute_tool(mock_agent, "test_tool", {"arg": "value"})

        mock_tool.assert_called_once_with(arg="value")
        assert result == "result"

    def test_raises_error_for_missing_tool(self):
        """Test that missing tool raises RuntimeError."""
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {}

        with pytest.raises(RuntimeError, match="Tool 'missing' not found"):
            _execute_tool(mock_agent, "missing", {})

    def test_raises_error_for_none_agent(self):
        """Test that None agent raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No agent available"):
            _execute_tool(None, "test_tool", {})

    def test_handles_error_status_in_result(self):
        """Test handling error status in tool result."""
        mock_tool = MagicMock(return_value={"status": "error", "content": [{"text": "Tool failed"}]})
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {"test_tool": mock_tool}

        with pytest.raises(RuntimeError, match="Tool error: Tool failed"):
            _execute_tool(mock_agent, "test_tool", {})

    def test_extracts_text_from_content(self):
        """Test extracting text from content list."""
        mock_tool = MagicMock(return_value={"status": "success", "content": [{"text": "first"}, {"text": "second"}]})
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {"test_tool": mock_tool}

        result = _execute_tool(mock_agent, "test_tool", {})

        assert result == "first\nsecond"

    def test_returns_string_result_directly(self):
        """Test that string results are returned directly."""
        mock_tool = MagicMock(return_value="direct string result")
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {"test_tool": mock_tool}

        result = _execute_tool(mock_agent, "test_tool", {})

        assert result == "direct string result"

    def test_returns_int_result_directly(self):
        """Test that integer results are returned directly."""
        mock_tool = MagicMock(return_value=42)
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {"test_tool": mock_tool}

        result = _execute_tool(mock_agent, "test_tool", {})

        assert result == 42


class TestCreateToolFunction:
    """Tests for _create_tool_function."""

    def test_creates_async_and_sync_functions(self):
        """Test that both async and sync functions are created."""
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {"test_tool": MagicMock(return_value="result")}

        async_func, sync_func = _create_tool_function(mock_agent, "test_tool")

        assert callable(async_func)
        assert callable(sync_func)
        # Check async function is actually async
        import asyncio

        assert asyncio.iscoroutinefunction(async_func)
        # Check sync function is not async
        assert not asyncio.iscoroutinefunction(sync_func)

    def test_sync_function_calls_tool(self):
        """Test that sync function calls the tool correctly."""
        mock_tool = MagicMock(return_value={"status": "success", "content": [{"text": "result"}]})
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {"test_tool": mock_tool}

        _, sync_func = _create_tool_function(mock_agent, "test_tool")
        result = sync_func(arg="value")

        mock_tool.assert_called_once_with(arg="value")
        assert result == "result"

    def test_async_function_calls_tool(self):
        """Test that async function calls the tool correctly."""
        import asyncio

        mock_tool = MagicMock(return_value={"status": "success", "content": [{"text": "async result"}]})
        mock_agent = MagicMock()
        mock_agent.tool_registry.registry = {"test_tool": mock_tool}

        async_func, _ = _create_tool_function(mock_agent, "test_tool")
        result = asyncio.run(async_func(arg="value"))

        mock_tool.assert_called_once_with(arg="value")
        assert result == "async result"


class TestValidateCode:
    """Tests for _validate_code function."""

    def test_detects_subprocess_import(self):
        """Test detection of subprocess import."""
        code = "import subprocess"
        warnings = _validate_code(code)
        assert any("subprocess" in w for w in warnings)

    def test_detects_eval_call(self):
        """Test detection of eval call."""
        code = "result = eval('1 + 1')"
        warnings = _validate_code(code)
        assert any("eval" in w for w in warnings)

    def test_detects_exec_call(self):
        """Test detection of exec call."""
        code = "exec('print(1)')"
        warnings = _validate_code(code)
        assert any("exec" in w for w in warnings)

    def test_detects_open_call(self):
        """Test detection of open call."""
        code = "f = open('file.txt')"
        warnings = _validate_code(code)
        assert any("open" in w for w in warnings)

    def test_safe_code_has_no_warnings(self):
        """Test that safe code produces no warnings."""
        code = """
result = calculator(expression="2 + 2")
print(f"Result: {result}")
"""
        warnings = _validate_code(code)
        assert len(warnings) == 0


class TestProgrammaticToolCaller:
    """Tests for programmatic_tool_caller function."""

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    def test_requires_agent_context(self, mock_console, mock_input):
        """Test that tool requires agent context."""
        mock_console.create.return_value = MagicMock()

        result = programmatic_tool_caller(code="print('hello')", tool_context=None)

        assert result["status"] == "error"
        assert "No agent context" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_executes_simple_code(self, mock_console, mock_input):
        """Test executing simple code."""
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
        """Test that only print output is returned."""
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(
            code="""
x = 1 + 2  # This doesn't get returned
y = x * 10  # Neither does this
print(f"Final answer: {y}")  # Only this is returned
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert "Final answer: 30" in content

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_handles_syntax_error(self, mock_console, mock_input):
        """Test handling of syntax errors."""
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
        """Test handling of runtime errors."""
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(code="x = 1 / 0", tool_context=mock_context)

        assert result["status"] == "error"
        assert "Execution error" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_provides_builtin_modules(self, mock_console, mock_input):
        """Test that builtin modules are available."""
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(
            code="""
import math
print(f"pi = {math.pi:.4f}")
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        assert "3.1415" in result["content"][0]["text"] or "3.1416" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_injects_tools_as_functions(self, mock_console, mock_input):
        """Test that tools are injected as callable functions."""
        mock_console.create.return_value = MagicMock()
        mock_tool = MagicMock(return_value={"status": "success", "content": [{"text": "42"}]})
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {"calculator": mock_tool}

        result = programmatic_tool_caller(
            code="""
# Check both async and sync versions are available
print(f"async available: {callable(calculator)}")
print(f"sync available: {callable(calculator_sync)}")
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        assert "async available: True" in result["content"][0]["text"]
        assert "sync available: True" in result["content"][0]["text"]

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_sync_tool_execution(self, mock_console, mock_input):
        """Test calling tools using sync function."""
        mock_console.create.return_value = MagicMock()
        mock_tool = MagicMock(return_value={"status": "success", "content": [{"text": "100"}]})
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {"calculator": mock_tool}

        result = programmatic_tool_caller(
            code="""
result = calculator_sync(expression="50 * 2")
print(f"Result: {result}")
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        assert "Result: 100" in result["content"][0]["text"]
        mock_tool.assert_called_once_with(expression="50 * 2")

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_async_tool_execution(self, mock_console, mock_input):
        """Test calling tools using async function."""
        mock_console.create.return_value = MagicMock()
        mock_tool = MagicMock(return_value={"status": "success", "content": [{"text": "200"}]})
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {"calculator": mock_tool}

        result = programmatic_tool_caller(
            code="""
import asyncio

async def main():
    result = await calculator(expression="100 * 2")
    print(f"Async result: {result}")

asyncio.run(main())
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        assert "Async result: 200" in result["content"][0]["text"]
        mock_tool.assert_called_once_with(expression="100 * 2")

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_multiple_tool_calls_in_loop(self, mock_console, mock_input):
        """Test multiple tool calls in a loop."""
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
    result = calculator_sync(expression=f"{i} * 10")
    total += int(result)
print(f"Total: {total}")
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        assert "Total: 100" in result["content"][0]["text"]  # 0+10+20+30+40 = 100
        assert call_count[0] == 5

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_empty_output_returns_no_output(self, mock_console, mock_input):
        """Test that code with no print returns '(no output)'."""
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(
            code="""
x = 1 + 2
y = x * 3
# No print statement
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        assert result["content"][0]["text"] == "(no output)"

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "false"})
    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    def test_user_cancellation(self, mock_console, mock_input):
        """Test user cancellation."""
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        # First call returns "n", second call returns cancellation reason
        mock_input.side_effect = ["n", "Changed my mind"]

        result = programmatic_tool_caller(code="print('hello')", tool_context=mock_context)

        assert result["status"] == "error"
        assert "cancelled" in result["content"][0]["text"].lower()


class TestIntegrationWithRealTools:
    """Integration tests with real tool behavior."""

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_data_filtering_pattern(self, mock_console, mock_input):
        """Test the data filtering pattern - only returning relevant results."""
        mock_console.create.return_value = MagicMock()

        def mock_fetch_logs(**kwargs):
            logs = [
                "INFO: System started",
                "DEBUG: Processing request",
                "ERROR: Connection timeout",
                "INFO: Request completed",
                "ERROR: Database unavailable",
                "DEBUG: Cache hit",
            ]
            return {"status": "success", "content": [{"text": "\n".join(logs)}]}

        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {"fetch_logs": mock_fetch_logs}

        result = programmatic_tool_caller(
            code="""
logs = fetch_logs_sync(server_id="server1")
errors = [line for line in logs.split('\\n') if 'ERROR' in line]
print(f"Found {len(errors)} errors:")
for error in errors:
    print(error)
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert "Found 2 errors" in content
        assert "Connection timeout" in content
        assert "Database unavailable" in content
        assert "INFO" not in content
        assert "DEBUG" not in content

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_conditional_tool_calls(self, mock_console, mock_input):
        """Test conditional tool calling pattern."""
        mock_console.create.return_value = MagicMock()

        def mock_check_health(**kwargs):
            endpoint = kwargs.get("endpoint", "")
            if endpoint == "us-east":
                return {"status": "success", "content": [{"text": "unhealthy"}]}
            elif endpoint == "eu-west":
                return {"status": "success", "content": [{"text": "healthy"}]}
            return {"status": "success", "content": [{"text": "unknown"}]}

        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {"check_health": mock_check_health}

        result = programmatic_tool_caller(
            code="""
endpoints = ["us-east", "eu-west", "apac"]
for endpoint in endpoints:
    status = check_health_sync(endpoint=endpoint)
    if status == "healthy":
        print(f"Found healthy endpoint: {endpoint}")
        break
else:
    print("No healthy endpoint found")
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        assert "Found healthy endpoint: eu-west" in result["content"][0]["text"]


class TestEdgeCases:
    """Tests for edge cases."""

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_empty_code(self, mock_console, mock_input):
        """Test with empty code."""
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(code="", tool_context=mock_context)

        assert result["status"] == "success"
        assert result["content"][0]["text"] == "(no output)"

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_only_comments(self, mock_console, mock_input):
        """Test code with only comments."""
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(
            code="""
# This is a comment
# Another comment
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        assert result["content"][0]["text"] == "(no output)"

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_multiline_string_output(self, mock_console, mock_input):
        """Test multiline output."""
        mock_console.create.return_value = MagicMock()
        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {}

        result = programmatic_tool_caller(
            code="""
print("Line 1")
print("Line 2")
print("Line 3")
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert "Line 1" in content
        assert "Line 2" in content
        assert "Line 3" in content

    @patch("strands_tools.programmatic_tool_caller.get_user_input")
    @patch("strands_tools.programmatic_tool_caller.console_util")
    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_exception_in_tool(self, mock_console, mock_input):
        """Test handling exception from tool."""
        mock_console.create.return_value = MagicMock()

        def failing_tool(**kwargs):
            raise ValueError("Tool failed internally")

        mock_context = MagicMock()
        mock_context.agent.tool_registry.registry = {"failing_tool": failing_tool}

        result = programmatic_tool_caller(
            code="""
try:
    result = failing_tool_sync()
except Exception as e:
    print(f"Caught error: {e}")
""",
            tool_context=mock_context,
        )

        assert result["status"] == "success"
        assert "Caught error" in result["content"][0]["text"]
