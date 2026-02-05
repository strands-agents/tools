"""Integration tests for programmatic_tool_caller tool.

These tests use real Strands Agent with real tools to verify the
programmatic tool calling functionality works end-to-end.
"""

from unittest.mock import patch

import pytest
from strands import Agent, tool

from strands_tools.programmatic_tool_caller import programmatic_tool_caller


# Define real tools for testing
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate.

    Returns:
        The result of the calculation as a string.
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def transform_text(text: str, operation: str = "upper") -> str:
    """Transform text with various operations.

    Args:
        text: The text to transform.
        operation: The operation to perform (upper, lower, title, reverse).

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
    else:
        return f"Unknown operation: {operation}"


@tool
def generate_data(count: int, prefix: str = "item") -> str:
    """Generate a list of items.

    Args:
        count: Number of items to generate.
        prefix: Prefix for each item.

    Returns:
        JSON array of generated items.
    """
    import json

    items = [{"id": i, "name": f"{prefix}_{i}", "value": i * 10} for i in range(count)]
    return json.dumps(items)


class TestProgrammaticToolCallerIntegration:
    """Integration tests with real Agent and tools."""

    @pytest.fixture
    def agent(self):
        """Create an agent with test tools."""
        return Agent(tools=[programmatic_tool_caller, calculate, transform_text, generate_data])

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_simple_calculation(self, agent):
        """Test simple calculation using sync tool."""
        result = agent.tool.programmatic_tool_caller(
            code="""
result = calculate_sync(expression="2 + 2")
print(f"Result: {result}")
"""
        )

        assert result["status"] == "success"
        assert "Result: 4" in result["content"][0]["text"]

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_async_calculation(self, agent):
        """Test calculation using async tool."""
        result = agent.tool.programmatic_tool_caller(
            code="""
import asyncio

async def main():
    result = await calculate(expression="10 * 5")
    print(f"Async Result: {result}")

asyncio.run(main())
"""
        )

        assert result["status"] == "success"
        assert "Async Result: 50" in result["content"][0]["text"]

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_multiple_tool_calls(self, agent):
        """Test calling multiple tools."""
        result = agent.tool.programmatic_tool_caller(
            code="""
calc_result = calculate_sync(expression="100 / 4")
text_result = transform_text_sync(text="hello world", operation="upper")
print(f"Calculation: {calc_result}")
print(f"Text: {text_result}")
"""
        )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert "Calculation: 25" in content
        assert "Text: HELLO WORLD" in content

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_loop_with_tool_calls(self, agent):
        """Test tool calls in a loop."""
        result = agent.tool.programmatic_tool_caller(
            code="""
total = 0
for i in range(1, 6):
    result = calculate_sync(expression=f"{i} ** 2")
    total += int(result)
    print(f"{i}² = {result}")
print(f"Sum of squares: {total}")
"""
        )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert "1² = 1" in content
        assert "2² = 4" in content
        assert "Sum of squares: 55" in content  # 1+4+9+16+25

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_conditional_tool_calls(self, agent):
        """Test conditional tool calling."""
        result = agent.tool.programmatic_tool_caller(
            code="""
value = int(calculate_sync(expression="50"))
if value > 30:
    result = transform_text_sync(text="big number", operation="upper")
else:
    result = transform_text_sync(text="small number", operation="lower")
print(f"Result: {result}")
"""
        )

        assert result["status"] == "success"
        assert "Result: BIG NUMBER" in result["content"][0]["text"]

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_chained_tool_results(self, agent):
        """Test using output of one tool as input to another."""
        result = agent.tool.programmatic_tool_caller(
            code="""
# Calculate something
calc_result = calculate_sync(expression="7 * 6")
# Use that result in text transformation
message = f"The answer is {calc_result}"
transformed = transform_text_sync(text=message, operation="title")
print(transformed)
"""
        )

        assert result["status"] == "success"
        assert "The Answer Is 42" in result["content"][0]["text"]

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_data_processing_pattern(self, agent):
        """Test processing data from a tool - only returning filtered results."""
        result = agent.tool.programmatic_tool_caller(
            code="""
import json

# Generate some data
data_str = generate_data_sync(count=5, prefix="product")
data = json.loads(data_str)

# Filter and process - only items with value > 20
filtered = [item for item in data if item["value"] > 20]
print(f"Found {len(filtered)} items with value > 20:")
for item in filtered:
    print(f"  - {item['name']}: {item['value']}")
"""
        )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert "Found 2 items" in content  # product_3 (30) and product_4 (40)
        assert "product_3" in content
        assert "product_4" in content

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_error_handling_in_code(self, agent):
        """Test error handling within the code."""
        result = agent.tool.programmatic_tool_caller(
            code="""
try:
    result = calculate_sync(expression="1/0")
    print(f"Result: {result}")
except Exception as e:
    print(f"Caught error: {type(e).__name__}")
"""
        )

        assert result["status"] == "success"
        # The calculate tool returns "Error: ..." string, doesn't raise
        content = result["content"][0]["text"]
        assert "Error:" in content or "Result:" in content

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_unavailable_tool_error(self, agent):
        """Test calling a non-existent tool."""
        result = agent.tool.programmatic_tool_caller(
            code="""
try:
    result = nonexistent_tool_sync()
except NameError as e:
    print(f"Tool not found: {e}")
"""
        )

        assert result["status"] == "success"
        content_lower = result["content"][0]["text"].lower()
        assert "not found" in content_lower or "not defined" in content_lower

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_builtin_modules_available(self, agent):
        """Test that builtin modules are available."""
        result = agent.tool.programmatic_tool_caller(
            code="""
import json
import re
import math
from datetime import datetime

# Test json
data = json.dumps({"test": True})
print(f"json works: {data}")

# Test re
pattern = r'\\d+'
match_result = bool(re.match(pattern, '123'))
print(f"re works: {match_result}")

# Test math
pi_val = round(math.pi, 4)
print(f"math works: {pi_val}")
"""
        )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert "json works" in content
        assert "re works: True" in content
        assert "math works: 3.141" in content

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_async_parallel_pattern(self, agent):
        """Test async pattern for parallel-style calls."""
        result = agent.tool.programmatic_tool_caller(
            code="""
import asyncio

async def main():
    # While these run sequentially due to our implementation,
    # the pattern supports async/await syntax
    results = []
    for expr in ["1+1", "2+2", "3+3"]:
        r = await calculate(expression=expr)
        results.append(r)
    
    print(f"Results: {results}")
    print(f"Sum: {sum(int(r) for r in results)}")

asyncio.run(main())
"""
        )

        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert "Results:" in content
        assert "Sum: 12" in content  # 2+4+6


class TestProgrammaticToolCallerOnlyReturnsOutput:
    """Tests to verify only print() output is returned."""

    @pytest.fixture
    def agent(self):
        """Create an agent with test tools."""
        return Agent(tools=[programmatic_tool_caller, calculate])

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_only_print_output_returned(self, agent):
        """Verify that only print() output is returned, not intermediate values."""
        result = agent.tool.programmatic_tool_caller(
            code="""
# These intermediate values should NOT appear in output
x = calculate_sync(expression="100 * 100")  # 10000
y = calculate_sync(expression="200 * 200")  # 40000
z = int(x) + int(y)  # 50000

# Only this should appear
print(f"Final: {z}")
"""
        )

        assert result["status"] == "success"
        content = result["content"][0]["text"]

        # Should contain the final printed output
        assert "Final: 50000" in content

        # Should NOT contain the raw intermediate values
        # (unless they happen to be in the final string)
        assert "10000" not in content.replace("50000", "")  # Exclude the 50000
        assert "40000" not in content

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_no_tool_call_summary(self, agent):
        """Verify that tool call summary is NOT included in output."""
        result = agent.tool.programmatic_tool_caller(
            code="""
for i in range(3):
    calculate_sync(expression=f"{i}+1")
print("Done")
"""
        )

        assert result["status"] == "success"
        content = result["content"][0]["text"]

        # Should only have "Done"
        assert content.strip() == "Done"

        # Should NOT contain tool call summary
        assert "Tool calls made" not in content

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_no_execution_time(self, agent):
        """Verify that execution time is NOT included in output."""
        result = agent.tool.programmatic_tool_caller(
            code="""
print("Hello")
"""
        )

        assert result["status"] == "success"
        content = result["content"][0]["text"]

        # Should only have "Hello"
        assert content.strip() == "Hello"

        # Should NOT contain execution time
        assert "Execution time" not in content
        assert "seconds" not in content.lower()
