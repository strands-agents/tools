"""
Tests for the sleep tool using the Agent interface.
"""

import time

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


def test_sleep_direct(agent):
    """Test direct invocation of the sleep tool."""
    start_time = time.time()
    result = agent.tool.sleep(seconds=0.5)
    elapsed_time = time.time() - start_time

    result_text = extract_result_text(result)

    # Verify the result message
    assert "Slept for 0.5 seconds" in result_text

    # Verify actual sleep time (with some tolerance)
    assert 0.4 <= elapsed_time <= 0.7


def test_sleep_zero_seconds(agent):
    """Test sleeping for zero seconds."""
    result = agent.tool.sleep(seconds=0)
    result_text = extract_result_text(result)

    assert "Slept for 0.0 seconds" in result_text


def test_sleep_negative_seconds(agent):
    """Test error handling with negative sleep duration."""
    result = agent.tool.sleep(seconds=-1)
    result_text = extract_result_text(result)

    # Verify the error message
    assert "cannot be negative" in result_text


def test_sleep_invalid_input(agent):
    """Test error handling with invalid input type."""
    result = agent.tool.sleep(seconds="invalid")
    result_text = extract_result_text(result)

    # Verify the error message contains validation error information
    assert "Validation failed for input parameters" in result_text
    assert "seconds" in result_text
