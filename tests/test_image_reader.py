"""
Tests for the image_reader tool.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent

from strands_tools import image_reader


@pytest.fixture
def test_image_path():
    """Return path to a test image file."""
    # This is a placeholder - we'll use a mock instead of creating real files
    return os.path.expanduser("~/test_image.jpg")


@pytest.fixture
def test_video_path():
    """Return path to a test video file."""
    # This is a placeholder - we'll use a mock instead of creating real files
    return os.path.expanduser("~/test_video.mp4")


@pytest.fixture
def agent():
    """Create an agent with the image_reader tool loaded."""
    return Agent(tools=[image_reader])


@patch("strands_tools.image_reader.os.path.exists")
@patch("strands_tools.image_reader.open")
@patch("strands_tools.image_reader.Image.open")
def test_image_reader_with_image(mock_pil_open, mock_open, mock_exists, test_image_path):
    """Test the image_reader tool with an image file."""
    # Mock file existence
    mock_exists.return_value = True

    # Mock file open
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = b"fake_image_bytes"
    mock_open.return_value = mock_file

    # Mock PIL Image
    mock_img = MagicMock()
    mock_img.__enter__.return_value.format = "JPEG"
    mock_pil_open.return_value = mock_img

    # Call the tool directly
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"image_path": test_image_path},
    }

    result = image_reader.image_reader(tool=tool_use)

    # Verify result structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "image" in result["content"][0]
    assert result["content"][0]["image"]["format"] == "jpeg"
    assert result["content"][0]["image"]["source"]["bytes"] == b"fake_image_bytes"


@patch("strands_tools.image_reader.os.path.exists")
def test_image_reader_file_not_found(mock_exists):
    """Test error handling when file does not exist."""
    mock_exists.return_value = False

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"image_path": "non_existent_file.jpg"},
    }

    result = image_reader.image_reader(tool=tool_use)

    assert result["status"] == "error"
    assert "File not found" in result["content"][0]["text"]


def test_image_reader_missing_path():
    """Test error handling when path is not provided."""
    tool_use = {"toolUseId": "test-tool-use-id", "input": {}}  # No image_path provided

    result = image_reader.image_reader(tool=tool_use)

    assert result["status"] == "error"
    assert "File path is required" in result["content"][0]["text"]


@patch("strands_tools.image_reader.os.path.exists")
@patch("strands_tools.image_reader.open")
@patch("strands_tools.image_reader.Image.open")
def test_image_reader_exception_handling(mock_pil_open, mock_open, mock_exists):
    """Test exception handling in the image_reader tool."""
    # Mock file existence
    mock_exists.return_value = True

    # Make PIL Image.open raise an exception
    mock_pil_open.side_effect = Exception("Test error")

    # Mock file open
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = b"fake_image_bytes"
    mock_open.return_value = mock_file

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"image_path": "test_image.jpg"},
    }

    result = image_reader.image_reader(tool=tool_use)

    assert result["status"] == "error"
    assert "Error reading file" in result["content"][0]["text"]
    assert "Test error" in result["content"][0]["text"]


def test_image_reader_via_agent(agent, test_image_path):
    """Test the image_reader via the agent interface."""
    # This test would be more complex in a real scenario as it would need to mock
    # the actual file operations within the agent. For now, we'll just check that
    # the method exists and can be called without errors.
    assert hasattr(agent.tool, "image_reader")

    # In a real test, you would mock the file operations and verify the result
    # But for illustration purposes, we're just checking the method existence
    pass
