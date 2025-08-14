"""
Tests for the stop tool using the Agent interface.
"""

import logging
from unittest.mock import patch

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


def test_stop_empty_reason_string(mock_request_state):
    """Test stop tool with empty reason string."""
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"reason": ""}}

    result = stop.stop(tool=tool_use, request_state=mock_request_state)

    assert result["status"] == "success"
    assert "Event loop cycle stop requested. Reason: " in result["content"][0]["text"]
    assert mock_request_state.get("stop_event_loop") is True


def test_stop_long_reason(mock_request_state):
    """Test stop tool with a long reason string."""
    long_reason = "This is a very long reason for stopping the event loop " * 10
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"reason": long_reason}}

    result = stop.stop(tool=tool_use, request_state=mock_request_state)

    assert result["status"] == "success"
    assert long_reason in result["content"][0]["text"]
    assert mock_request_state.get("stop_event_loop") is True


def test_stop_special_characters_in_reason(mock_request_state):
    """Test stop tool with special characters in reason."""
    special_reason = "Reason with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"reason": special_reason}}

    result = stop.stop(tool=tool_use, request_state=mock_request_state)

    assert result["status"] == "success"
    assert special_reason in result["content"][0]["text"]
    assert mock_request_state.get("stop_event_loop") is True


def test_stop_unicode_reason(mock_request_state):
    """Test stop tool with unicode characters in reason."""
    unicode_reason = "停止原因: タスク完了 🎉"
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"reason": unicode_reason}}

    result = stop.stop(tool=tool_use, request_state=mock_request_state)

    assert result["status"] == "success"
    assert unicode_reason in result["content"][0]["text"]
    assert mock_request_state.get("stop_event_loop") is True


def test_stop_no_request_state():
    """Test stop tool when no request_state is provided."""
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"reason": "Test reason"}}

    # Call without request_state - should create empty dict
    result = stop.stop(tool=tool_use)

    assert result["status"] == "success"
    assert "Test reason" in result["content"][0]["text"]


def test_stop_existing_request_state_data(mock_request_state):
    """Test stop tool with existing data in request_state."""
    # Pre-populate request state with some data
    mock_request_state["existing_key"] = "existing_value"
    mock_request_state["another_key"] = 42

    tool_use = {"toolUseId": "test-tool-use-id", "input": {"reason": "Test reason"}}

    result = stop.stop(tool=tool_use, request_state=mock_request_state)

    # Verify existing data is preserved
    assert mock_request_state["existing_key"] == "existing_value"
    assert mock_request_state["another_key"] == 42
    
    # Verify stop flag was added
    assert mock_request_state.get("stop_event_loop") is True
    
    assert result["status"] == "success"


def test_stop_overwrite_existing_stop_flag(mock_request_state):
    """Test stop tool overwrites existing stop_event_loop flag."""
    # Pre-set the flag to False
    mock_request_state["stop_event_loop"] = False

    tool_use = {"toolUseId": "test-tool-use-id", "input": {"reason": "Test reason"}}

    result = stop.stop(tool=tool_use, request_state=mock_request_state)

    # Verify flag was set to True
    assert mock_request_state.get("stop_event_loop") is True
    assert result["status"] == "success"


def test_stop_logging(mock_request_state, caplog):
    """Test that stop tool logs the reason."""
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"reason": "Test logging"}}

    with caplog.at_level(logging.DEBUG):
        stop.stop(tool=tool_use, request_state=mock_request_state)

    # Check that the reason was logged
    assert "Reason: Test logging" in caplog.text


def test_stop_logging_no_reason(mock_request_state, caplog):
    """Test that stop tool logs when no reason is provided."""
    tool_use = {"toolUseId": "test-tool-use-id", "input": {}}

    with caplog.at_level(logging.DEBUG):
        stop.stop(tool=tool_use, request_state=mock_request_state)

    # Check that the default reason was logged
    assert "Reason: No reason provided" in caplog.text


def test_stop_tool_spec():
    """Test that the TOOL_SPEC is properly defined."""
    assert stop.TOOL_SPEC["name"] == "stop"
    assert "description" in stop.TOOL_SPEC
    assert "inputSchema" in stop.TOOL_SPEC
    assert "json" in stop.TOOL_SPEC["inputSchema"]
    
    schema = stop.TOOL_SPEC["inputSchema"]["json"]
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "reason" in schema["properties"]
    assert schema["properties"]["reason"]["type"] == "string"


def test_stop_multiple_calls_same_state(mock_request_state):
    """Test multiple calls to stop with the same request state."""
    tool_use1 = {"toolUseId": "test-1", "input": {"reason": "First stop"}}
    tool_use2 = {"toolUseId": "test-2", "input": {"reason": "Second stop"}}

    # First call
    result1 = stop.stop(tool=tool_use1, request_state=mock_request_state)
    assert result1["status"] == "success"
    assert mock_request_state.get("stop_event_loop") is True

    # Second call - flag should remain True
    result2 = stop.stop(tool=tool_use2, request_state=mock_request_state)
    assert result2["status"] == "success"
    assert mock_request_state.get("stop_event_loop") is True
