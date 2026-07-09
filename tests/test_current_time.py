"""
Tests for the current_time tool using the Agent interface.
"""

import re
from datetime import datetime
from datetime import timezone as tz
from zoneinfo import ZoneInfo

import pytest
from strands import Agent

from strands_tools import current_time


@pytest.fixture
def agent():
    """Create an agent with the current_time tool loaded."""
    return Agent(tools=[current_time])


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def is_iso8601_format(time_string):
    """Check if a string is in ISO 8601 format."""
    # This pattern matches ISO 8601 format with timezone
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"
    return bool(re.match(pattern, time_string))


def test_current_time_direct_utc(agent):
    """Test direct invocation of the current_time tool with UTC timezone."""
    result = agent.tool.current_time(timezone="UTC")

    result_text = extract_result_text(result)

    # Verify the result is a string in ISO 8601 format
    assert isinstance(result_text, str)
    assert is_iso8601_format(result_text)

    # Verify the timezone is UTC (+00:00)
    assert result_text.endswith("+00:00") or result_text.endswith("Z")

    # Parse the returned time and ensure it's close to now
    # Converting to timestamp for easy comparison
    returned_time = datetime.fromisoformat(result_text).timestamp()
    now = datetime.now(tz.utc).timestamp()
    # Within 5 seconds should be reasonable
    assert abs(returned_time - now) < 5


def test_current_time_direct_custom_timezone(agent):
    """Test direct invocation with a custom timezone."""
    # Test with US/Pacific timezone
    result = agent.tool.current_time(timezone="America/Los_Angeles")

    result_text = extract_result_text(result)

    # Verify result format
    assert isinstance(result_text, str)
    assert is_iso8601_format(result_text)

    # Verify the timezone offset is as expected for Pacific time
    # This can be -07:00 or -08:00 depending on daylight saving time
    # So we'll just check that it's a negative offset
    assert "-" in result_text[-6:]

    # Verify the time is roughly correct by comparing to UTC with appropriate offset
    returned_time = datetime.fromisoformat(result_text)
    pacific_tz = ZoneInfo("America/Los_Angeles")
    now_pacific = datetime.now(pacific_tz)

    # Times should be within 5 seconds of each other
    time_diff = abs((returned_time - now_pacific).total_seconds())
    assert time_diff < 5


def test_current_time_invalid_timezone(agent):
    """Test error handling with an invalid timezone."""
    result = agent.tool.current_time(timezone="InvalidTimeZone")

    result_text = extract_result_text(result)
    # Verify the error message mentions the timezone issue
    assert "Error" in result_text


def test_current_time_via_agent(agent):
    """Test getting current time via the agent interface."""
    result = agent.tool.current_time(timezone="UTC")

    result_text = extract_result_text(result)
    # Verify the result is in ISO 8601 format
    assert is_iso8601_format(result_text)
    # Verify UTC timezone
    assert result_text.endswith("+00:00") or result_text.endswith("Z")


def test_current_time_agent_custom_timezone(agent):
    """Test getting current time with custom timezone via agent."""
    result = agent.tool.current_time(timezone="Europe/London")

    result_text = extract_result_text(result)
    # Verify the result is in ISO 8601 format
    assert is_iso8601_format(result_text)


def test_current_time_agent_invalid_timezone(agent):
    """Test error handling with invalid timezone via agent."""
    result = agent.tool.current_time(timezone="InvalidTimeZone")

    result_text = extract_result_text(result)
    assert "Error" in result_text
