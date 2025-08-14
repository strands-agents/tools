"""
Tests for the sleep tool using the Agent interface.
"""

import os
import time
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


def test_sleep_successful_execution(agent):
    """Test successful sleep execution with mocked time."""
    with mock.patch("time.sleep") as mock_sleep, \
         mock.patch("strands_tools.sleep.datetime") as mock_datetime:
        
        # Mock datetime.now() to return a consistent time
        mock_datetime.now.return_value.strftime.return_value = "2025-01-01 12:00:00"
        
        result = agent.tool.sleep(seconds=2.5)
        result_text = extract_result_text(result)
        
        # Verify time.sleep was called with correct duration
        mock_sleep.assert_called_once_with(2.5)
        
        # Verify the success message format
        assert "Started sleep at 2025-01-01 12:00:00" in result_text
        assert "slept for 2.5 seconds" in result_text


def test_sleep_float_input(agent):
    """Test sleep with float input."""
    with mock.patch("time.sleep") as mock_sleep, \
         mock.patch("strands_tools.sleep.datetime") as mock_datetime:
        
        mock_datetime.now.return_value.strftime.return_value = "2025-01-01 12:00:00"
        
        result = agent.tool.sleep(seconds=1.5)
        result_text = extract_result_text(result)
        
        mock_sleep.assert_called_once_with(1.5)
        assert "slept for 1.5 seconds" in result_text


def test_sleep_integer_input(agent):
    """Test sleep with integer input."""
    with mock.patch("time.sleep") as mock_sleep, \
         mock.patch("strands_tools.sleep.datetime") as mock_datetime:
        
        mock_datetime.now.return_value.strftime.return_value = "2025-01-01 12:00:00"
        
        result = agent.tool.sleep(seconds=3)
        result_text = extract_result_text(result)
        
        mock_sleep.assert_called_once_with(3)
        assert "slept for 3.0 seconds" in result_text


def test_sleep_direct_function_call():
    """Test calling the sleep function directly without agent."""
    with mock.patch("time.sleep") as mock_sleep, \
         mock.patch("strands_tools.sleep.datetime") as mock_datetime:
        
        mock_datetime.now.return_value.strftime.return_value = "2025-01-01 12:00:00"
        
        result = sleep.sleep(1.0)
        
        mock_sleep.assert_called_once_with(1.0)
        assert "Started sleep at 2025-01-01 12:00:00" in result
        assert "slept for 1.0 seconds" in result


def test_sleep_direct_function_validation_errors():
    """Test direct function call validation errors."""
    # Test non-numeric input
    with pytest.raises(ValueError, match="Sleep duration must be a number"):
        sleep.sleep("invalid")
    
    # Test zero input
    with pytest.raises(ValueError, match="Sleep duration must be greater than 0"):
        sleep.sleep(0)
    
    # Test negative input
    with pytest.raises(ValueError, match="Sleep duration must be greater than 0"):
        sleep.sleep(-1)


def test_sleep_direct_function_max_exceeded():
    """Test direct function call with max sleep exceeded."""
    # Store original max and restore it
    original_max = sleep.max_sleep_seconds
    try:
        # Ensure we have the default max
        sleep.max_sleep_seconds = 300
        
        # Test with default max (300 seconds)
        with pytest.raises(ValueError, match="Sleep duration cannot exceed 300 seconds"):
            sleep.sleep(301)
    finally:
        # Restore original max
        sleep.max_sleep_seconds = original_max


def test_sleep_direct_function_keyboard_interrupt():
    """Test direct function call with KeyboardInterrupt."""
    with mock.patch("time.sleep", side_effect=KeyboardInterrupt):
        result = sleep.sleep(5)
        assert result == "Sleep interrupted by user"


def test_max_sleep_seconds_environment_variable():
    """Test that MAX_SLEEP_SECONDS environment variable is respected."""
    # Test the module-level variable
    original_max = sleep.max_sleep_seconds
    
    try:
        # Test with custom environment variable
        with mock.patch.dict(os.environ, {"MAX_SLEEP_SECONDS": "60"}):
            import importlib
            importlib.reload(sleep)
            
            # Verify the new max is loaded
            assert sleep.max_sleep_seconds == 60
            
            # Test that it's enforced
            with pytest.raises(ValueError, match="Sleep duration cannot exceed 60 seconds"):
                sleep.sleep(61)
    
    finally:
        # Restore original max
        sleep.max_sleep_seconds = original_max


def test_max_sleep_seconds_default_value():
    """Test that default MAX_SLEEP_SECONDS is 300."""
    # Remove the environment variable if it exists
    with mock.patch.dict(os.environ, {}, clear=True):
        import importlib
        importlib.reload(sleep)
        
        # Should default to 300
        assert sleep.max_sleep_seconds == 300
