"""Integration tests for the programmatic_tool_caller tool."""

import pytest
from strands import Agent, tool

from strands_tools.programmatic_tool_caller import programmatic_tool_caller


@pytest.fixture
def math_tool():
    """Create a simple math tool."""

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression safely.

        Args:
            expression: The math expression to evaluate.

        Returns:
            The result as a string.
        """
        # Only allow safe math operations
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}
        return str(eval(expression, {"__builtins__": {}}, allowed_names))

    return calculate


@pytest.fixture
def text_tool():
    """Create a simple text transformation tool."""

    @tool
    def transform_text(text: str, operation: str = "upper") -> str:
        """Transform text using the specified operation.

        Args:
            text: The text to transform.
            operation: The operation to apply (upper, lower, title, reverse).

        Returns:
            The transformed text.
        """
        if operation == "upper":
            return text.upper()
        elif operation == "lower":
            return text.lower()
        elif operation == "title":
            return text.title()
        elif operation == "reverse":
            return text[::-1]
        return text

    return transform_text


@pytest.fixture
def data_tool():
    """Create a simple data generation tool."""

    @tool
    def generate_data(count: int, start: int = 0) -> str:
        """Generate a sequence of numbers.

        Args:
            count: How many numbers to generate.
            start: The starting number.

        Returns:
            JSON array of numbers.
        """
        import json

        return json.dumps(list(range(start, start + count)))

    return generate_data


@pytest.fixture
def agent_with_tools(math_tool, text_tool, data_tool):
    """Create an agent with programmatic_tool_caller and helper tools."""
    return Agent(tools=[programmatic_tool_caller, math_tool, text_tool, data_tool])


class TestProgrammaticToolCallerIntegration:
    """Integration tests for programmatic_tool_caller with real tools."""

    def test_simple_calculation(self, agent_with_tools):
        """Test a simple calculation using tools in code."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("BYPASS_TOOL_CONSENT", "true")

            result = agent_with_tools.tool.programmatic_tool_caller(
                code="""
result = tools.calculate(expression="2 + 3 * 4")
print(f"Result: {result}")
"""
            )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert "14" in content  # 2 + 3 * 4 = 14

    def test_multiple_tool_calls(self, agent_with_tools):
        """Test calling multiple different tools."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("BYPASS_TOOL_CONSENT", "true")

            result = agent_with_tools.tool.programmatic_tool_caller(
                code="""
# Calculate something
calc_result = tools.calculate(expression="10 * 5")

# Transform text
text_result = tools.transform_text(text="hello world", operation="upper")

# Generate data
data_result = tools.generate_data(count=3, start=1)

print(f"Calculation: {calc_result}")
print(f"Text: {text_result}")
print(f"Data: {data_result}")
"""
            )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert "50" in content
        assert "HELLO WORLD" in content
        assert "[1, 2, 3]" in content
        assert "Tool calls made: 3" in content

    def test_loop_with_tool_calls(self, agent_with_tools):
        """Test using tools inside a loop."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("BYPASS_TOOL_CONSENT", "true")

            result = agent_with_tools.tool.programmatic_tool_caller(
                code="""
# Calculate sum of squares from 1 to 5
total = 0
for i in range(1, 6):
    square = int(tools.calculate(expression=f"{i} ** 2"))
    total += square
    print(f"{i}² = {square}")

print(f"Sum of squares: {total}")
"""
            )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        # 1 + 4 + 9 + 16 + 25 = 55
        assert "55" in content
        assert "Tool calls made: 5" in content

    def test_conditional_tool_calls(self, agent_with_tools):
        """Test using tools with conditional logic."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("BYPASS_TOOL_CONSENT", "true")

            result = agent_with_tools.tool.programmatic_tool_caller(
                code="""
value = int(tools.calculate(expression="7 * 3"))

if value > 20:
    result = tools.transform_text(text="greater than twenty", operation="upper")
else:
    result = tools.transform_text(text="less or equal to twenty", operation="upper")

print(f"Value: {value}")
print(f"Result: {result}")
"""
            )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert "21" in content
        assert "GREATER THAN TWENTY" in content

    def test_chained_tool_results(self, agent_with_tools):
        """Test passing results from one tool to another."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("BYPASS_TOOL_CONSENT", "true")

            result = agent_with_tools.tool.programmatic_tool_caller(
                code="""
import json

# Generate some data
data_json = tools.generate_data(count=5, start=1)
print(f"Generated data: {data_json}")

# Parse it and calculate sum
data = json.loads(data_json)
sum_expr = " + ".join(str(x) for x in data)
total = tools.calculate(expression=sum_expr)
print(f"Sum expression: {sum_expr}")
print(f"Total: {total}")
"""
            )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        # 1 + 2 + 3 + 4 + 5 = 15
        assert "15" in content

    def test_list_available_tools(self, agent_with_tools):
        """Test listing tools from within code."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("BYPASS_TOOL_CONSENT", "true")

            result = agent_with_tools.tool.programmatic_tool_caller(
                code="""
available = tools.list_tools()
print(f"Available tools: {available}")

# Verify our tools are there
assert "calculate" in available
assert "transform_text" in available
assert "generate_data" in available
# programmatic_tool_caller should NOT be in the list
assert "programmatic_tool_caller" not in available

print("All assertions passed!")
"""
            )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert "All assertions passed!" in content

    def test_get_tool_info(self, agent_with_tools):
        """Test getting tool information from within code."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("BYPASS_TOOL_CONSENT", "true")

            result = agent_with_tools.tool.programmatic_tool_caller(
                code="""
info = tools.get_tool_info("calculate")
print(f"Tool name: {info['name']}")
print(f"Description: {info['description']}")

# Non-existent tool should return None
none_info = tools.get_tool_info("nonexistent")
print(f"Nonexistent tool info: {none_info}")
"""
            )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert "calculate" in content
        assert "None" in content

    def test_error_handling_in_code(self, agent_with_tools):
        """Test that errors in executed code are properly reported."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("BYPASS_TOOL_CONSENT", "true")

            result = agent_with_tools.tool.programmatic_tool_caller(
                code="""
# This should cause a ZeroDivisionError
result = tools.calculate(expression="1/0")
print(result)
"""
            )

        # The error could come from either the tool or the code
        assert result["status"] == "error"

    def test_unavailable_tool_error(self, agent_with_tools):
        """Test that accessing unavailable tools raises proper error."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("BYPASS_TOOL_CONSENT", "true")

            result = agent_with_tools.tool.programmatic_tool_caller(
                code="""
# Try to use a tool that doesn't exist
result = tools.nonexistent_tool(param="value")
"""
            )

        assert result["status"] == "error"
        assert "not available" in result["content"][0]["text"]

    def test_builtin_modules_available(self, agent_with_tools):
        """Test that safe builtin modules are available."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("BYPASS_TOOL_CONSENT", "true")

            result = agent_with_tools.tool.programmatic_tool_caller(
                code="""
import json
import math
import re
from datetime import datetime
from collections import Counter

# Test each module
print(f"json: {json.dumps({'a': 1})}")
print(f"math.pi: {math.pi:.4f}")

pattern = r'\\d+'
match_result = bool(re.match(pattern, '123'))
print(f"re match: {match_result}")

print(f"datetime: {type(datetime.now()).__name__}")
print(f"Counter: {Counter(['a', 'b', 'a'])}")
"""
            )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert '{"a": 1}' in content
        assert "3.141" in content  # math.pi rounded
        assert "True" in content


class TestAgentToolCall:
    """Test that agent can be prompted to use programmatic_tool_caller."""

    def test_agent_uses_programmatic_tool_caller(self, agent_with_tools):
        """Test that an agent can be asked to use the programmatic tool caller."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("BYPASS_TOOL_CONSENT", "true")

            # Ask the agent to calculate sum of first 5 squares using programmatic approach
            response = agent_with_tools(
                """Use the programmatic_tool_caller tool to calculate the sum of squares from 1 to 5.
                Write Python code that uses a loop to call the calculate tool for each square,
                then sums them up. Print the final result.
                
                If successful, respond with EXACTLY: RESULT: <number>
                """
            )

            response_text = str(response).lower()
            # The sum of squares 1² + 2² + 3² + 4² + 5² = 55
            assert "55" in response_text or "result: 55" in response_text
