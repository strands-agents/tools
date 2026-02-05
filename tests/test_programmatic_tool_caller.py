"""Tests for the programmatic_tool_caller tool."""

import os
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent, tool

from strands_tools.programmatic_tool_caller import (
    OutputCapture,
    ToolProxy,
    _execute_tool,
    _validate_code,
    programmatic_tool_caller,
)


@pytest.fixture
def mock_agent():
    """Create a mock agent with a tool registry."""
    agent = MagicMock()
    agent.tool_registry = MagicMock()

    # Create mock tools
    mock_calculator = MagicMock()
    mock_calculator.tool_spec = {
        "name": "calculator",
        "description": "Evaluates mathematical expressions",
        "inputSchema": {"json": {"properties": {"expression": {"type": "string"}}}},
    }
    mock_calculator.__call__ = MagicMock(return_value={"status": "success", "content": [{"text": "4"}]})

    mock_file_read = MagicMock()
    mock_file_read.tool_spec = {
        "name": "file_read",
        "description": "Reads file content",
        "inputSchema": {"json": {"properties": {"path": {"type": "string"}}}},
    }
    mock_file_read.__call__ = MagicMock(return_value={"status": "success", "content": [{"text": "file content"}]})

    agent.tool_registry.registry = {
        "calculator": mock_calculator,
        "file_read": mock_file_read,
        "programmatic_tool_caller": MagicMock(),  # The tool itself
    }

    return agent


@pytest.fixture
def mock_console():
    """Mock the console to prevent output during tests."""
    with patch("strands_tools.programmatic_tool_caller.console_util") as mock:
        yield mock.create.return_value


class TestOutputCapture:
    """Test the OutputCapture class."""

    def test_capture_stdout(self):
        """Test capturing standard output."""
        capture = OutputCapture()
        with capture:
            print("Hello, world!")

        output = capture.get_output()
        assert "Hello, world!" in output

    def test_capture_stderr(self):
        """Test capturing standard error."""
        import sys

        capture = OutputCapture()
        with capture:
            print("Error message", file=sys.stderr)

        output = capture.get_output()
        assert "Error message" in output
        assert "[stderr]" in output

    def test_capture_both(self):
        """Test capturing both stdout and stderr."""
        import sys

        capture = OutputCapture()
        with capture:
            print("Standard output")
            print("Standard error", file=sys.stderr)

        output = capture.get_output()
        assert "Standard output" in output
        assert "Standard error" in output


class TestToolProxy:
    """Test the ToolProxy class."""

    def test_list_tools(self, mock_agent):
        """Test listing available tools."""
        callback = MagicMock(return_value="result")
        proxy = ToolProxy(mock_agent.tool_registry, callback)

        tools = proxy.list_tools()
        assert "calculator" in tools
        assert "file_read" in tools
        assert "programmatic_tool_caller" not in tools  # Should be excluded

    def test_list_tools_with_filter(self, mock_agent):
        """Test listing available tools with filter."""
        callback = MagicMock(return_value="result")
        proxy = ToolProxy(mock_agent.tool_registry, callback, allowed_tools=["calculator"])

        tools = proxy.list_tools()
        assert tools == ["calculator"]

    def test_call_tool_success(self, mock_agent):
        """Test calling a tool successfully."""
        callback = MagicMock(return_value="4")
        proxy = ToolProxy(mock_agent.tool_registry, callback)

        result = proxy.calculator(expression="2+2")

        callback.assert_called_once_with("calculator", {"expression": "2+2"})
        assert result == "4"

    def test_call_tool_records_history(self, mock_agent):
        """Test that tool calls are recorded in history."""
        callback = MagicMock(return_value="4")
        proxy = ToolProxy(mock_agent.tool_registry, callback)

        proxy.calculator(expression="2+2")
        history = proxy.get_call_history()

        assert len(history) == 1
        assert history[0]["tool_name"] == "calculator"
        assert history[0]["input"] == {"expression": "2+2"}
        assert history[0]["status"] == "success"

    def test_call_tool_records_error(self, mock_agent):
        """Test that errors are recorded in history."""
        callback = MagicMock(side_effect=RuntimeError("Tool error"))
        proxy = ToolProxy(mock_agent.tool_registry, callback)

        with pytest.raises(RuntimeError):
            proxy.calculator(expression="invalid")

        history = proxy.get_call_history()
        assert len(history) == 1
        assert history[0]["status"] == "error"
        assert "Tool error" in history[0]["error"]

    def test_access_unavailable_tool(self, mock_agent):
        """Test accessing a tool that doesn't exist."""
        callback = MagicMock()
        proxy = ToolProxy(mock_agent.tool_registry, callback)

        with pytest.raises(AttributeError) as exc_info:
            proxy.nonexistent_tool()

        assert "nonexistent_tool" in str(exc_info.value)
        assert "not available" in str(exc_info.value)

    def test_access_private_attribute(self, mock_agent):
        """Test that private attributes raise AttributeError."""
        callback = MagicMock()
        proxy = ToolProxy(mock_agent.tool_registry, callback)

        with pytest.raises(AttributeError):
            _ = proxy._private

    def test_get_tool_info(self, mock_agent):
        """Test getting tool information."""
        callback = MagicMock()
        proxy = ToolProxy(mock_agent.tool_registry, callback)

        info = proxy.get_tool_info("calculator")
        assert info is not None
        assert info["name"] == "calculator"
        assert "description" in info

    def test_get_tool_info_not_found(self, mock_agent):
        """Test getting info for non-existent tool."""
        callback = MagicMock()
        proxy = ToolProxy(mock_agent.tool_registry, callback)

        info = proxy.get_tool_info("nonexistent")
        assert info is None


class TestValidateCode:
    """Test the code validation function."""

    def test_clean_code(self):
        """Test that clean code passes validation."""
        code = """
result = 2 + 2
print(f"Result: {result}")
"""
        warnings = _validate_code(code)
        assert len(warnings) == 0

    def test_dangerous_import_subprocess(self):
        """Test detection of subprocess import."""
        code = "import subprocess"
        warnings = _validate_code(code)
        assert len(warnings) > 0
        assert any("subprocess" in w for w in warnings)

    def test_dangerous_exec(self):
        """Test detection of exec call."""
        code = "exec('print(1)')"
        warnings = _validate_code(code)
        assert len(warnings) > 0
        assert any("exec" in w for w in warnings)

    def test_dangerous_eval(self):
        """Test detection of eval call."""
        code = "result = eval('2+2')"
        warnings = _validate_code(code)
        assert len(warnings) > 0
        assert any("eval" in w for w in warnings)

    def test_file_open(self):
        """Test detection of open() call."""
        code = "f = open('file.txt', 'r')"
        warnings = _validate_code(code)
        assert len(warnings) > 0
        assert any("open" in w for w in warnings)


class TestExecuteTool:
    """Test the _execute_tool function."""

    def test_execute_tool_success(self, mock_agent):
        """Test successful tool execution."""
        # Setup mock to return a value when called directly
        mock_agent.tool_registry.registry["calculator"].return_value = "4"
        result = _execute_tool(mock_agent, "calculator", {"expression": "2+2"})
        assert result == "4"

    def test_execute_tool_no_agent(self):
        """Test execution without agent raises error."""
        with pytest.raises(RuntimeError) as exc_info:
            _execute_tool(None, "calculator", {})
        assert "No agent available" in str(exc_info.value)

    def test_execute_tool_not_found(self, mock_agent):
        """Test execution of non-existent tool."""
        with pytest.raises(RuntimeError) as exc_info:
            _execute_tool(mock_agent, "nonexistent", {})
        assert "not found" in str(exc_info.value)

    def test_execute_tool_error_result(self, mock_agent):
        """Test handling of error result from tool."""
        mock_agent.tool_registry.registry["calculator"].return_value = {
            "status": "error",
            "content": [{"text": "Calculation error"}],
        }

        with pytest.raises(RuntimeError) as exc_info:
            _execute_tool(mock_agent, "calculator", {"expression": "invalid"})
        assert "Tool error" in str(exc_info.value)

    def test_execute_tool_dict_result_with_content(self, mock_agent):
        """Test handling of dict result with content."""
        mock_agent.tool_registry.registry["calculator"].return_value = {
            "status": "success",
            "content": [{"text": "42"}],
        }
        result = _execute_tool(mock_agent, "calculator", {"expression": "6*7"})
        assert result == "42"

    def test_execute_tool_direct_string_result(self, mock_agent):
        """Test handling of direct string result."""
        mock_agent.tool_registry.registry["calculator"].return_value = "direct result"
        result = _execute_tool(mock_agent, "calculator", {"expression": "1+1"})
        assert result == "direct result"


class TestProgrammaticToolCaller:
    """Test the main programmatic_tool_caller function."""

    def test_basic_execution(self, mock_agent, mock_console):
        """Test basic code execution."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
            result = programmatic_tool_caller(
                code='print("Hello, World!")',
                agent=mock_agent,
            )

        assert result["status"] == "success"
        assert "Hello, World!" in result["content"][0]["text"]

    def test_tool_call_in_code(self, mock_agent, mock_console):
        """Test calling a tool from within the code."""
        # Setup mock to return when called
        mock_agent.tool_registry.registry["calculator"].return_value = {
            "status": "success",
            "content": [{"text": "4"}],
        }

        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
            result = programmatic_tool_caller(
                code='result = tools.calculator(expression="2+2"); print(f"Result: {result}")',
                agent=mock_agent,
            )

        assert result["status"] == "success"

    def test_no_agent_error(self, mock_console):
        """Test that missing agent returns error."""
        result = programmatic_tool_caller(
            code='print("test")',
            agent=None,
        )

        assert result["status"] == "error"
        assert "No agent context" in result["content"][0]["text"]

    def test_syntax_error(self, mock_agent, mock_console):
        """Test handling of syntax errors."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
            result = programmatic_tool_caller(
                code="this is not valid python",
                agent=mock_agent,
            )

        assert result["status"] == "error"
        assert "Syntax error" in result["content"][0]["text"]

    def test_runtime_error(self, mock_agent, mock_console):
        """Test handling of runtime errors."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
            result = programmatic_tool_caller(
                code="1/0",  # ZeroDivisionError
                agent=mock_agent,
            )

        assert result["status"] == "error"
        assert "ZeroDivisionError" in result["content"][0]["text"]

    def test_allowed_tools_filter(self, mock_agent, mock_console):
        """Test that allowed_tools filter works."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
            # Try to access a tool that's not in allowed list
            result = programmatic_tool_caller(
                code="tools.file_read(path='test.txt')",
                allowed_tools=["calculator"],  # Only calculator allowed
                agent=mock_agent,
            )

        assert result["status"] == "error"
        assert "not available" in result["content"][0]["text"]

    def test_user_cancellation(self, mock_agent, mock_console):
        """Test user cancellation of execution."""
        with (
            patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "false"}),
            patch(
                "strands_tools.programmatic_tool_caller.get_user_input",
                side_effect=["n", "Testing cancellation"],
            ),
        ):
            result = programmatic_tool_caller(
                code='print("Should not run")',
                agent=mock_agent,
            )

        assert result["status"] == "error"
        assert "cancelled" in result["content"][0]["text"]

    def test_execution_with_imports(self, mock_agent, mock_console):
        """Test that allowed imports are available."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
            result = programmatic_tool_caller(
                code="""
import json
import math
data = json.dumps({"pi": math.pi})
print(data)
""",
                agent=mock_agent,
            )

        assert result["status"] == "success"
        assert "pi" in result["content"][0]["text"]

    def test_tool_call_history_tracking(self, mock_agent, mock_console):
        """Test that tool calls are tracked and reported."""
        # Setup mock to return when called
        mock_agent.tool_registry.registry["calculator"].return_value = {
            "status": "success",
            "content": [{"text": "10"}],
        }

        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
            result = programmatic_tool_caller(
                code="""
for i in range(3):
    result = tools.calculator(expression=f"{i} * 2")
print("Done")
""",
                agent=mock_agent,
            )

        assert result["status"] == "success"
        # Should report the number of tool calls made
        assert "Tool calls made:" in result["content"][0]["text"]


class TestIntegration:
    """Integration tests with real tools."""

    @pytest.fixture
    def real_agent(self):
        """Create a real agent with simple tools for integration testing."""

        @tool
        def simple_calculator(expression: str) -> str:
            """Evaluates a simple math expression.

            Args:
                expression: The math expression to evaluate.

            Returns:
                The result as a string.
            """
            # Safe eval for simple math
            return str(eval(expression))

        @tool
        def string_tool(text: str) -> str:
            """Transforms a string to uppercase.

            Args:
                text: The text to transform.

            Returns:
                The uppercase text.
            """
            return text.upper()

        return Agent(tools=[programmatic_tool_caller, simple_calculator, string_tool])

    def test_integration_simple_tool_call(self, real_agent):
        """Test calling a real tool through programmatic execution."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
            result = real_agent.tool.programmatic_tool_caller(
                code="""
result = tools.simple_calculator(expression="5 + 5")
print(f"5 + 5 = {result}")
"""
            )

        # The result should contain the output
        if isinstance(result, dict) and "content" in result:
            content = result["content"][0]["text"]
        else:
            content = str(result)

        assert "10" in content

    def test_integration_multiple_tools(self, real_agent):
        """Test calling multiple tools in one execution."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
            result = real_agent.tool.programmatic_tool_caller(
                code="""
num_result = tools.simple_calculator(expression="3 * 7")
str_result = tools.string_tool(text="hello")
print(f"Number: {num_result}, String: {str_result}")
"""
            )

        if isinstance(result, dict) and "content" in result:
            content = result["content"][0]["text"]
        else:
            content = str(result)

        assert "21" in content
        assert "HELLO" in content

    def test_integration_list_tools(self, real_agent):
        """Test listing available tools from code."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
            result = real_agent.tool.programmatic_tool_caller(
                code="""
available = tools.list_tools()
print(f"Available tools: {available}")
"""
            )

        if isinstance(result, dict) and "content" in result:
            content = result["content"][0]["text"]
        else:
            content = str(result)

        assert "simple_calculator" in content
        assert "string_tool" in content
        # programmatic_tool_caller should be excluded
        assert "programmatic_tool_caller" not in content


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_empty_code(self, mock_agent, mock_console):
        """Test execution of empty code."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
            result = programmatic_tool_caller(
                code="",
                agent=mock_agent,
            )

        assert result["status"] == "success"

    def test_only_comments(self, mock_agent, mock_console):
        """Test execution of code with only comments."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
            result = programmatic_tool_caller(
                code="# This is just a comment\n# Another comment",
                agent=mock_agent,
            )

        assert result["status"] == "success"

    def test_multiline_string_output(self, mock_agent, mock_console):
        """Test capturing multiline output."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
            result = programmatic_tool_caller(
                code="""
print("Line 1")
print("Line 2")
print("Line 3")
""",
                agent=mock_agent,
            )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert "Line 1" in content
        assert "Line 2" in content
        assert "Line 3" in content

    def test_exception_in_tool(self, mock_agent, mock_console):
        """Test handling of exception raised by a tool."""
        mock_agent.tool_registry.registry["calculator"].side_effect = Exception("Tool crashed")

        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
            result = programmatic_tool_caller(
                code='tools.calculator(expression="crash")',
                agent=mock_agent,
            )

        assert result["status"] == "error"
        assert "Tool crashed" in result["content"][0]["text"] or "Execution error" in result["content"][0]["text"]
