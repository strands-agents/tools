"""
Tests for the stop tool using the Agent interface.
"""

import pytest
from strands import Agent

from strands_tools import stop


@pytest.fixture
def agent():
    """Create an agent with the stop tool loaded."""
    return Agent(tools=[stop])


@pytest.fixture
def mock_request_state():
    """Create a mock request state dictionary."""
    return {}


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_stop_tool_direct(mock_request_state):
    """Test direct invocation of the stop tool."""
    # Create a tool use dictionary similar to how the agent would call it
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"reason": "Test reason"}}

    # Call the stop function directly with our mock request state
    result = stop.stop(tool=tool_use, request_state=mock_request_state)

    # Verify the result has the expected structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "Event loop cycle stop requested" in result["content"][0]["text"]
    assert "Test reason" in result["content"][0]["text"]

    # Verify the stop_event_loop flag was set in request_state
    assert mock_request_state.get("stop_event_loop") is True


def test_stop_no_reason(mock_request_state):
    """Test stop tool without providing a reason."""
    tool_use = {"toolUseId": "test-tool-use-id", "input": {}}

    result = stop.stop(tool=tool_use, request_state=mock_request_state)

    assert result["status"] == "success"
    assert "No reason provided" in result["content"][0]["text"]
    assert mock_request_state.get("stop_event_loop") is True


def test_stop_via_agent(agent):
    """Test stopping via the agent interface.

    Note: This test is more for illustration; in a real environment,
    stopping would end the agent's event loop, making verification difficult.
    """
    # This is a simplified test that doesn't actually test event loop stopping behavior
    result = agent.tool.stop(reason="Test via agent")

    result_text = extract_result_text(result)
    assert "Event loop cycle stop requested" in result_text
    assert "Test via agent" in result_text


def test_stop_flag_effect(mock_request_state):
    """Test that the stop flag has the intended effect on request state."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"reason": "Testing flag effect"},
    }

    # Verify the flag is not set initially
    assert mock_request_state.get("stop_event_loop") is None

    # Call stop tool
    stop.stop(tool=tool_use, request_state=mock_request_state)

    # Verify the flag was set
    assert mock_request_state.get("stop_event_loop") is True
