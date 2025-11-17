"""
Tests for the sleep tool using the Agent interface.
"""

import os
from unittest import mock

import pytest
from strands import Agent

from strands_tools import sleep


@pytest.fixture
def agent():
    """Create an agent with the sleep tool loaded."""
    return Agent(tools=[sleep])


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_sleep_zero_seconds(agent):
    """Test error handling with zero sleep duration."""
    result = agent.tool.sleep(seconds=0)
    result_text = extract_result_text(result)

    # Verify the error message
    assert "must be greater than 0" in result_text


def test_sleep_negative_seconds(agent):
    """Test error handling with negative sleep duration."""
    result = agent.tool.sleep(seconds=-1)
    result_text = extract_result_text(result)

    # Verify the error message
    assert "must be greater than 0" in result_text


def test_sleep_invalid_input(agent):
    """Test error handling with invalid input type."""
    result = agent.tool.sleep(seconds="invalid")
    result_text = extract_result_text(result)

    # Verify the error message contains validation error information
    assert "Validation failed for input parameters" in result_text
    assert "seconds" in result_text


def test_sleep_keyboard_interrupt(agent):
    """Test that sleep stops when KeyboardInterrupt is raised."""
    # Create a mock function that raises KeyboardInterrupt
    with mock.patch("time.sleep", side_effect=KeyboardInterrupt):
        # Call the sleep function through agent
        result = agent.tool.sleep(seconds=5)
        result_text = extract_result_text(result)

        # Verify the result message
        assert "Sleep interrupted by user" in result_text


def test_sleep_exceeds_max(agent):
    """Test error handling when sleep duration exceeds maximum allowed value."""
    # Temporarily set a smaller max sleep time for testing
    with mock.patch.dict(os.environ, {"MAX_SLEEP_SECONDS": "10"}):
        # This will reload the module with the new environment variable
        import importlib

        importlib.reload(sleep)

        # Test with a value exceeding the maximum
        result = agent.tool.sleep(seconds=11)
        result_text = extract_result_text(result)

        # Verify the error message
        assert "cannot exceed 10 seconds" in result_text
