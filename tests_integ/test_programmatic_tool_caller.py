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
print(f"Sum: {total}")
"""
        )
        assert result["status"] == "success"
        assert "Sum: 55" in result["content"][0]["text"]

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
    def test_asyncio_gather(self, agent):
        result = agent.tool.programmatic_tool_caller(
            code="""
results = await asyncio.gather(
    calculate(expression="10 * 1"),
    calculate(expression="10 * 2"),
    calculate(expression="10 * 3"),
)
print(f"Total: {sum(int(r) for r in results)}")
"""
        )
        assert result["status"] == "success"
        assert "Total: 60" in result["content"][0]["text"]

    @patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true", "PROGRAMMATIC_TOOL_CALLER_ALLOWED_TOOLS": "calculate"})
    def test_allowed_tools_env_var(self, agent):
        # calculate is allowed
        result = agent.tool.programmatic_tool_caller(code='r = await calculate(expression="5*5")\nprint(r)')
        assert result["status"] == "success"
        assert "25" in result["content"][0]["text"]


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
