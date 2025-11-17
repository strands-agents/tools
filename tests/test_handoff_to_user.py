"""Tests for the handoff_to_user tool."""

from unittest.mock import patch

import pytest
from strands import Agent

from strands_tools import handoff_to_user


@pytest.fixture
def agent():
    """Create an agent with the handoff_to_user tool loaded."""
    return Agent(tools=[handoff_to_user])


@pytest.fixture
def mock_request_state():
    """Create a mock request state dictionary."""
    return {}


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_handoff_with_breakout_true_direct(mock_request_state):
    """Test handoff with breakout_of_loop=True stops the event loop (direct call)."""
    # Create a tool use dictionary similar to how the agent would call it
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "message": "Task completed. Please review the results.",
            "breakout_of_loop": True,
        },
    }

    # Call the handoff_to_user function directly with our mock request state
    result = handoff_to_user.handoff_to_user(tool=tool_use, request_state=mock_request_state)

    # Verify the result has the expected structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "Agent handoff completed" in result["content"][0]["text"]
    assert "Task completed. Please review the results." in result["content"][0]["text"]

    # Verify the stop_event_loop flag was set in request_state
    assert mock_request_state.get("stop_event_loop") is True


@patch("strands_tools.handoff_to_user.get_user_input", return_value="user response")
def test_handoff_with_breakout_false_direct(mock_get_user_input, mock_request_state):
    """Test handoff with breakout_of_loop=False waits for user input (direct call)."""
    # Create a tool use dictionary
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"message": "Please confirm the action.", "breakout_of_loop": False},
    }

    # Call the handoff_to_user function directly
    result = handoff_to_user.handoff_to_user(tool=tool_use, request_state=mock_request_state)

    # Verify get_user_input was called
    mock_get_user_input.assert_called_once_with(
        "<bold>Agent requested user input:</bold> Please confirm the action.\n<bold>Your response:</bold> "
    )

    # Verify the result has the expected structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "user response" in result["content"][0]["text"]

    # Verify the event loop stop flag is not set
    assert mock_request_state.get("stop_event_loop") is not True


@patch("strands_tools.handoff_to_user.get_user_input", side_effect=KeyboardInterrupt())
def test_handoff_keyboard_interrupt_direct(mock_get_user_input, mock_request_state):
    """Test handoff handles KeyboardInterrupt gracefully (direct call)."""
    # Create a tool use dictionary
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"message": "Please confirm the action.", "breakout_of_loop": False},
    }

    # Call the handoff_to_user function and expect KeyboardInterrupt to be handled
    result = handoff_to_user.handoff_to_user(tool=tool_use, request_state=mock_request_state)

    # Verify the event loop stop flag is set due to interruption
    assert mock_request_state["stop_event_loop"] is True

    # Verify the result indicates interruption
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "interrupted" in result["content"][0]["text"]


def test_handoff_missing_request_state():
    """Test handoff works even without request_state."""
    # Create a tool use dictionary
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"message": "Task completed.", "breakout_of_loop": True},
    }

    # Call the handoff_to_user function without request_state
    result = handoff_to_user.handoff_to_user(tool=tool_use)

    # Should still work and return success
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "Agent handoff completed" in result["content"][0]["text"]


@patch("strands_tools.handoff_to_user.get_user_input", side_effect=Exception("Test error"))
def test_handoff_input_error_direct(mock_get_user_input, mock_request_state):
    """Test handoff handles input errors gracefully (direct call)."""
    # Create a tool use dictionary
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"message": "Please confirm the action.", "breakout_of_loop": False},
    }

    # Call the handoff_to_user function and expect exception to be handled
    result = handoff_to_user.handoff_to_user(tool=tool_use, request_state=mock_request_state)

    # Verify the result indicates error
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "error"
    assert "Error during user handoff" in result["content"][0]["text"]


def test_handoff_default_message():
    """Test handoff with default message when none provided."""
    # Create a tool use dictionary without message
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"breakout_of_loop": True}}

    result = handoff_to_user.handoff_to_user(tool=tool_use)

    assert result["status"] == "success"
    assert "Agent requesting user handoff" in result["content"][0]["text"]


@patch("strands_tools.handoff_to_user.get_user_input", return_value="test response")
def test_handoff_default_breakout_false(mock_get_user_input):
    """Test handoff defaults to breakout_of_loop=False when not specified."""
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"message": "Test message"}}

    result = handoff_to_user.handoff_to_user(tool=tool_use)

    assert result["status"] == "success"
    assert "test response" in result["content"][0]["text"]
    # Verify get_user_input was called (indicating breakout_of_loop=False)
    mock_get_user_input.assert_called_once()


def test_handoff_via_agent(agent):
    """Test handoff via the agent interface.

    Note: This test is more for illustration; in a real environment,
    handoff behavior would depend on actual user interaction.
    """
    # This is a simplified test that doesn't actually test interactive behavior
    result = agent.tool.handoff_to_user(message="Test via agent", breakout_of_loop=True)

    result_text = extract_result_text(result)
    assert "Agent handoff completed" in result_text
    assert "Test via agent" in result_text


def test_tool_spec_structure():
    """Test that the tool spec has the correct structure."""
    spec = handoff_to_user.TOOL_SPEC

    assert spec["name"] == "handoff_to_user"
    assert "description" in spec
    assert "inputSchema" in spec
    assert "json" in spec["inputSchema"]
    assert "properties" in spec["inputSchema"]["json"]

    properties = spec["inputSchema"]["json"]["properties"]
    assert "message" in properties
    assert "breakout_of_loop" in properties
    assert properties["message"]["type"] == "string"
    assert properties["breakout_of_loop"]["type"] == "boolean"
    assert properties["breakout_of_loop"]["default"] is False

    required = spec["inputSchema"]["json"]["required"]
    assert "message" in required
