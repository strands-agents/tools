"""
Tests for the file_write tool using the Agent interface.
"""

import os
from unittest.mock import patch

import pytest
from strands import Agent

from strands_tools import file_write
from strands_tools.file_write import create_rich_panel, detect_language


@pytest.fixture
def agent():
    """Create an agent with the file_write tool loaded."""
    return Agent(tools=[file_write])


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file path."""
    return str(tmp_path / "test_file.txt")


@pytest.fixture
def temp_nested_file(tmp_path):
    """Create a temporary nested file path for directory creation testing."""
    return str(tmp_path / "nested" / "dir" / "test_file.txt")


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_detect_language():
    """Test language detection from file extensions."""
    assert detect_language("test.py") == "py"
    assert detect_language("test.txt") == "txt"
    assert detect_language("test") == "text"
    assert detect_language("test.ipynb") == "ipynb"
    assert detect_language("/path/to/test.js") == "js"


def test_create_rich_panel():
    """Test rich panel creation."""
    # Test with syntax highlighting
    panel = create_rich_panel("Test content", "Test Title", "python")
    assert panel is not None
    assert panel.title == "Test Title"

    # Test without syntax highlighting
    panel = create_rich_panel("Test content", "Test Title")
    assert panel is not None
    assert panel.title == "Test Title"


@patch("strands_tools.file_write.get_user_input")
def test_file_write_direct(mock_user_input, temp_file):
    """Test direct invocation of the file_write tool with user confirmation."""
    # Mock user confirming the write
    mock_user_input.return_value = "y"

    # Create a tool use dictionary similar to how the agent would call it
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"path": temp_file, "content": "This is a test content"},
    }

    # Call the file_write function directly
    result = file_write.file_write(tool=tool_use)

    # Verify the result has the expected structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "File write success" in result["content"][0]["text"]
    assert temp_file in result["content"][0]["text"]

    # Verify file was actually written
    assert os.path.exists(temp_file)
    with open(temp_file, "r") as f:
        assert f.read() == "This is a test content"


@patch("strands_tools.file_write.get_user_input")
def test_file_write_cancel(mock_user_input, temp_file):
    """Test cancellation of the file write operation."""
    # Mock user cancelling the write
    mock_user_input.side_effect = ["n", "User changed their mind"]

    # Ensure BYPASS_TOOL_CONSENT mode is disabled to force confirmation
    current_dev = os.environ.get("BYPASS_TOOL_CONSENT", None)
    if current_dev:
        os.environ.pop("BYPASS_TOOL_CONSENT")

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"path": temp_file, "content": "This is a test content"},
    }

    result = file_write.file_write(tool=tool_use)

    # Verify the operation was cancelled
    assert result["status"] == "error"
    assert "cancelled" in result["content"][0]["text"]
    assert "User changed their mind" in result["content"][0]["text"]

    # Verify file was not created
    assert not os.path.exists(temp_file)

    # Restore BYPASS_TOOL_CONSENT mode if it was set
    if current_dev:
        os.environ["BYPASS_TOOL_CONSENT"] = current_dev


@patch("strands_tools.file_write.get_user_input")
def test_file_write_directory_creation(mock_user_input, temp_nested_file):
    """Test that directories are created if they don't exist."""
    # Mock user confirming the write
    mock_user_input.return_value = "y"

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"path": temp_nested_file, "content": "Testing nested directories"},
    }

    result = file_write.file_write(tool=tool_use)

    # Verify the result
    assert result["status"] == "success"

    # Verify directories and file were created
    assert os.path.exists(temp_nested_file)
    with open(temp_nested_file, "r") as f:
        assert f.read() == "Testing nested directories"


@patch("strands_tools.file_write.get_user_input")
def test_file_write_error_handling(mock_user_input, temp_file):
    """Test error handling during file writing."""
    # Mock user confirming the write
    mock_user_input.return_value = "y"

    # Use a non-existent directory that we don't have permission to create
    # This should fail on most systems
    if os.name == "posix":  # Unix/Linux/Mac
        invalid_path = "/root/test_no_permission.txt"
    else:
        # Fallback - create a path that's too long
        invalid_path = os.path.join(temp_file, "a" * 1000 + ".txt")

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"path": invalid_path, "content": "This will fail"},
    }

    result = file_write.file_write(tool=tool_use)

    # Verify the error was handled correctly
    assert result["status"] == "error"
    assert "Error writing file" in result["content"][0]["text"]


@patch("strands_tools.file_write.get_user_input")
@patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"})
def test_file_write_dev_mode(mock_user_input, temp_file):
    """Test file_write in BYPASS_TOOL_CONSENT mode (skipping confirmation)."""
    # Mock should not be called in BYPASS_TOOL_CONSENT mode
    mock_user_input.return_value = "should not be called"

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"path": temp_file, "content": "BYPASS_TOOL_CONSENT mode test"},
    }

    result = file_write.file_write(tool=tool_use)

    # Verify file was written without confirmation
    assert result["status"] == "success"
    assert os.path.exists(temp_file)
    with open(temp_file, "r") as f:
        assert f.read() == "BYPASS_TOOL_CONSENT mode test"

    # Verify user input was not called
    mock_user_input.assert_not_called()


def test_file_write_via_agent(agent, temp_file):
    """Test file_write via the agent interface with BYPASS_TOOL_CONSENT mode to bypass user input."""
    # For agent testing, we need to use the BYPASS_TOOL_CONSENT environment to bypass user input
    os.environ["BYPASS_TOOL_CONSENT"] = "true"
    try:
        # Use the agent to write a file
        result = agent.tool.file_write(path=temp_file, content="Testing via agent")

        # Extract and verify the result
        result_text = extract_result_text(result)
        assert "File write success" in result_text

        # Verify the file contents
        assert os.path.exists(temp_file)
        with open(temp_file, "r") as f:
            assert f.read() == "Testing via agent"
    finally:
        # Restore the environment
        if "BYPASS_TOOL_CONSENT" in os.environ:
            del os.environ["BYPASS_TOOL_CONSENT"]


@patch("strands_tools.file_write.get_user_input")
def test_file_write_alternative_rejection(mock_user_input, temp_file):
    """Test file write rejection with an alternative response."""
    # Mock user providing an alternative rejection (not just 'n')
    mock_user_input.return_value = "I need to review this more"

    # Ensure BYPASS_TOOL_CONSENT mode is disabled to force confirmation
    current_dev = os.environ.get("BYPASS_TOOL_CONSENT", None)
    if current_dev:
        os.environ.pop("BYPASS_TOOL_CONSENT")

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"path": temp_file, "content": "This content should not be written"},
    }

    result = file_write.file_write(tool=tool_use)

    # Verify the operation was cancelled with the custom reason
    assert result["status"] == "error"
    assert "cancelled" in result["content"][0]["text"].lower()
    assert "review this more" in result["content"][0]["text"]

    # Restore BYPASS_TOOL_CONSENT mode if it was set
    if current_dev:
        os.environ["BYPASS_TOOL_CONSENT"] = current_dev
    assert "cancelled" in result["content"][0]["text"]
    assert "I need to review this more" in result["content"][0]["text"]

    # Verify file was not created
    assert not os.path.exists(temp_file)
