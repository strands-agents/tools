"""Integration tests for programmatic_tool_caller tool."""

from unittest.mock import patch

import pytest
from strands import Agent, tool

from strands_tools.programmatic_tool_caller import programmatic_tool_caller


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def transform_text(text: str, operation: str = "upper") -> str:
    """Transform text."""
    if operation == "upper":
        return text.upper()
    elif operation == "lower":
        return text.lower()
    return text


class TestProgrammaticToolCallerIntegration:
    """Integration tests with real Agent and tools."""

    @pytest.fixture
    def agent(self):
        return Agent(tools=[programmatic_tool_caller, calculate, transform_text])

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_simple_await(self, agent):
        result = agent.tool.programmatic_tool_caller(
            code="""
result = await calculate(expression="2 + 2")
print(f"Result: {result}")
"""
        )
        assert result["status"] == "success"
        assert "Result: 4" in result["content"][0]["text"]

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_loop_with_await(self, agent):
        result = agent.tool.programmatic_tool_caller(
            code="""
total = 0
for i in range(1, 6):
    r = await calculate(expression=f"{i} ** 2")
    total += int(r)
print(f"Sum of squares: {total}")
"""
        )
        assert result["status"] == "success"
        assert "Sum of squares: 55" in result["content"][0]["text"]

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_multiple_tools(self, agent):
        result = agent.tool.programmatic_tool_caller(
            code="""
num = await calculate(expression="7 * 6")
text = await transform_text(text="hello", operation="upper")
print(f"Number: {num}, Text: {text}")
"""
        )
        assert result["status"] == "success"
        assert "42" in result["content"][0]["text"]
        assert "HELLO" in result["content"][0]["text"]

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_asyncio_gather(self, agent):
        result = agent.tool.programmatic_tool_caller(
            code="""
results = await asyncio.gather(
    calculate(expression="10 * 1"),
    calculate(expression="10 * 2"),
    calculate(expression="10 * 3"),
)
print(f"Parallel: {results}")
total = sum(int(r) for r in results)
print(f"Total: {total}")
"""
        )
        assert result["status"] == "success"
        assert "Total: 60" in result["content"][0]["text"]

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_conditional_logic(self, agent):
        result = agent.tool.programmatic_tool_caller(
            code="""
value = int(await calculate(expression="50"))
if value > 30:
    msg = await transform_text(text="big", operation="upper")
else:
    msg = await transform_text(text="small", operation="upper")
print(f"Result: {msg}")
"""
        )
        assert result["status"] == "success"
        assert "BIG" in result["content"][0]["text"]


class TestOnlyPrintOutputReturned:
    """Verify only print() output is returned."""

    @pytest.fixture
    def agent(self):
        return Agent(tools=[programmatic_tool_caller, calculate])

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_only_print_returned(self, agent):
        result = agent.tool.programmatic_tool_caller(
            code="""
x = await calculate(expression="100 * 100")
y = await calculate(expression="200 * 200")
z = int(x) + int(y)
print(f"Final: {z}")
"""
        )
        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert "Final: 50000" in content
        # Intermediate values not in output
        assert "10000" not in content.replace("50000", "")
        assert "40000" not in content

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_no_tool_call_summary(self, agent):
        result = agent.tool.programmatic_tool_caller(
            code="""
for i in range(3):
    await calculate(expression=f"{i}+1")
print("Done")
"""
        )
        assert result["status"] == "success"
        assert result["content"][0]["text"].strip() == "Done"
