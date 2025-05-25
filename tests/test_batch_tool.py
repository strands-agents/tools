from unittest.mock import MagicMock

import pytest
from strands_tools.batch_tool import batch_tool


@pytest.fixture
def mock_agent():
    """Fixture to create a mock agent with tools."""
    agent = MagicMock()
    agent.tool.think = MagicMock(return_value="Think tool executed")
    agent.tool.stop = MagicMock(return_value="Stop tool executed")
    agent.tool.error_tool = MagicMock(side_effect=Exception("Tool execution failed"))
    return agent


def test_batch_tool_success(mock_agent):
    """Test successful execution of multiple tools."""
    mock_tool = ("mock_tool",)
    invocations = [
        {"name": "think", "arguments": {"thought": "How to improve AI?", "cycle_count": 2}},
        {"name": "stop", "arguments": {}},
    ]

    result = batch_tool(tool=mock_tool, agent=mock_agent, invocations=invocations)

    assert result["status"] == "success"
    assert len(result["results"]) == 2
    assert result["results"][0]["name"] == "think"
    assert result["results"][0]["status"] == "success"
    assert result["results"][0]["result"] == "Think tool executed"
    assert result["results"][1]["name"] == "stop"
    assert result["results"][1]["status"] == "success"
    assert result["results"][1]["result"] == "Stop tool executed"
