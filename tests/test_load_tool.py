"""
Tests for the load_tool tool using the Agent interface.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent

from strands_tools import load_tool


@pytest.fixture
def agent():
    """Create an agent with the load_tool tool loaded."""
    return Agent(tools=[load_tool])


@pytest.fixture
def mock_request_state():
    """Create a mock request state dictionary."""
    return {}


@pytest.fixture
def temp_tool_file():
    """Create a temporary Python tool file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w+") as f:
        # Write a simple tool function to the temporary file
        f.write(
            '''
def sample_tool(tool, **kwargs):
    """Sample tool for testing."""
    tool_use_id = tool["toolUseId"]
    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": "Sample tool executed"}],
    }
        '''
        )
        temp_path = f.name

    yield temp_path

    # Clean up the temporary file after the test
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_load_tool_direct(temp_tool_file):
    """Test direct invocation of the load_tool tool."""
    # Create a mock agent with a mock tool registry
    mock_agent = MagicMock()
    mock_tool_registry = MagicMock()
    mock_agent.tool_registry = mock_tool_registry

    # Call the load_tool function directly with the new signature
    result = load_tool.load_tool(path=temp_tool_file, name="sample_tool", agent=mock_agent)

    # Verify the result has the expected structure
    assert result["status"] == "success"
    assert "loaded successfully" in result["content"][0]["text"]

    # Verify the tool was loaded via the registry
    mock_tool_registry.load_tool_from_filepath.assert_called_once_with(
        tool_name="sample_tool", tool_path=temp_tool_file
    )


def test_load_tool_disabled():
    """Test load_tool when disabled via environment variable."""
    # Set the environment variable to disable load_tool
    with patch.dict(os.environ, {"STRANDS_DISABLE_LOAD_TOOL": "true"}):
        mock_agent = MagicMock()

        # Call the load_tool function with the new signature
        result = load_tool.load_tool(path="/path/to/tool.py", name="test_tool", agent=mock_agent)

        # Verify the result indicates tool loading is disabled
        assert result["status"] == "error"
        assert "disabled" in result["content"][0]["text"]

        # Verify the tool was not loaded
        mock_agent.tool_registry.load_tool_from_filepath.assert_not_called()


def test_load_tool_file_not_found():
    """Test load_tool with a non-existent file path."""
    mock_agent = MagicMock()

    # Call the load_tool function with the new signature
    result = load_tool.load_tool(path="/non/existent/path.py", name="test_tool", agent=mock_agent)

    # Verify the result indicates file not found
    assert result["status"] == "error"
    assert "not found" in str(result["content"][0]["text"])

    # Verify the tool was not loaded
    mock_agent.tool_registry.load_tool_from_filepath.assert_not_called()


def test_load_tool_exception():
    """Test load_tool when an exception occurs during loading."""
    mock_agent = MagicMock()
    mock_agent.tool_registry.load_tool_from_filepath.side_effect = Exception("Test exception")

    with tempfile.NamedTemporaryFile(suffix=".py") as f:
        # Call the load_tool function with the new signature
        result = load_tool.load_tool(path=f.name, name="test_tool", agent=mock_agent)

        # Verify the result indicates an error
        assert result["status"] == "error"
        assert "Failed to load tool" in result["content"][0]["text"]
        assert "Test exception" in result["content"][0]["text"]


def test_load_tool_via_agent(agent, temp_tool_file):
    """Test loading a tool via the agent interface."""
    # Just verify that the agent can be instantiated with the load_tool tool
    # and that calling a method on it doesn't raise an exception
    try:
        agent.tool.load_tool(path=temp_tool_file, name="test_tool")
        # If we get here without an exception, consider the test passed
        assert True
    except Exception as e:
        pytest.fail(f"Agent load_tool call raised an exception: {e}")
