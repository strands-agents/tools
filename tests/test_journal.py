"""
Tests for the journal tool using the Agent interface.
"""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from strands import Agent

from strands_tools import journal


@pytest.fixture
def agent():
    """Create an agent with the journal tool loaded."""
    return Agent(tools=[journal])


@pytest.fixture
def tmp_journal_dir(tmp_path):
    """Create a temporary directory for journal files and set as CWD."""
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_dir)


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_ensure_journal_dir():
    """Test journal directory creation."""
    original_dir = os.getcwd()
    try:
        # Use a temporary directory
        tmp_dir = Path(os.path.join(original_dir, "tmp_test_journal"))
        os.makedirs(tmp_dir, exist_ok=True)
        os.chdir(tmp_dir)

        # Call the function
        journal_dir = journal.ensure_journal_dir()

        # Check if directory exists
        assert journal_dir.exists()
        assert journal_dir.is_dir()
        assert journal_dir.name == "journal"

    finally:
        # Clean up
        os.chdir(original_dir)
        import shutil

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def test_get_journal_path():
    """Test journal path retrieval."""
    with patch("strands_tools.journal.ensure_journal_dir") as mock_ensure_dir:
        mock_dir = MagicMock()
        mock_ensure_dir.return_value = mock_dir

        # Test with default date (today)
        journal.get_journal_path()
        mock_dir.__truediv__.assert_called_once()

        # Test with specific date
        mock_dir.reset_mock()
        journal.get_journal_path("2023-01-01")
        mock_dir.__truediv__.assert_called_once_with("2023-01-01.md")


def test_journal_write(tmp_journal_dir):
    """Test writing to journal."""
    # Create a tool use dictionary similar to how the agent would call it
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"action": "write", "content": "Test journal entry"},
    }

    # Call the journal function directly
    result = journal.journal(tool=tool_use)

    # Verify the result
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "Added entry to journal" in result["content"][0]["text"]

    # Verify the file was created
    today = datetime.now().strftime("%Y-%m-%d")
    journal_path = tmp_journal_dir / "journal" / f"{today}.md"
    assert journal_path.exists()

    # Check file content
    content = journal_path.read_text()
    assert "Test journal entry" in content


def test_journal_read(tmp_journal_dir):
    """Test reading from journal."""
    # First create a journal entry
    today = datetime.now().strftime("%Y-%m-%d")
    journal_path = tmp_journal_dir / "journal"
    journal_path.mkdir(exist_ok=True)
    test_journal = journal_path / f"{today}.md"
    test_journal.write_text("## 12:00:00\nTest content")

    # Create tool use for reading
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"action": "read"}}

    # Call the journal function
    result = journal.journal(tool=tool_use)

    # Verify the result
    assert result["status"] == "success"
    assert "Test content" in result["content"][0]["text"]


def test_journal_list(tmp_journal_dir):
    """Test listing journal entries."""
    # Create a few journal entries
    journal_path = tmp_journal_dir / "journal"
    journal_path.mkdir(exist_ok=True)

    # Add 3 journal entries with tasks
    for day in ["2023-01-01", "2023-01-02", "2023-01-03"]:
        entry = journal_path / f"{day}.md"
        entry.write_text(f"## 12:00:00\nEntry for {day}\n\n## 12:30:00\n- [ ] Task 1\n- [ ] Task 2")

    # Create tool use for listing
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"action": "list"}}

    # Call the journal function
    result = journal.journal(tool=tool_use)

    # Verify the result
    assert result["status"] == "success"
    assert "Listed 3 journal entries" in result["content"][0]["text"]


def test_journal_list_empty(tmp_journal_dir):
    """Test listing when no journal entries exist."""
    # Create empty journal directory
    journal_path = tmp_journal_dir / "journal"
    journal_path.mkdir(exist_ok=True)

    # Create tool use for listing
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"action": "list"}}

    # Call the journal function
    result = journal.journal(tool=tool_use)

    # Verify the result for empty list
    assert result["status"] == "success"
    assert "No journal entries found" in result["content"][0]["text"]


def test_journal_add_task(tmp_journal_dir):
    """Test adding task to journal."""
    # Create tool use for adding a task
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"action": "add_task", "task": "Complete unit tests"},
    }

    # Call the journal function
    result = journal.journal(tool=tool_use)

    # Verify the result
    assert result["status"] == "success"
    assert "Added task to journal" in result["content"][0]["text"]

    # Verify the file was created and contains the task
    today = datetime.now().strftime("%Y-%m-%d")
    journal_path = tmp_journal_dir / "journal" / f"{today}.md"
    assert journal_path.exists()

    content = journal_path.read_text()
    assert "- [ ] Complete unit tests" in content


def test_journal_error_handling():
    """Test journal error handling."""
    # Test missing content for write action
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "action": "write"
            # Missing content
        },
    }

    result = journal.journal(tool=tool_use)
    assert result["status"] == "error"
    assert "Content is required" in result["content"][0]["text"]

    # Test missing task for add_task action
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "action": "add_task"
            # Missing task
        },
    }

    result = journal.journal(tool=tool_use)
    assert result["status"] == "error"
    assert "Task is required" in result["content"][0]["text"]

    # Test unknown action
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"action": "invalid_action"}}

    result = journal.journal(tool=tool_use)
    assert result["status"] == "error"
    assert "Unknown action" in result["content"][0]["text"]


def test_journal_general_exception():
    """Test journal general exception handling."""
    with patch("strands_tools.journal.get_journal_path") as mock_get_path:
        # Make get_journal_path raise an exception
        mock_get_path.side_effect = Exception("Test exception")

        tool_use = {
            "toolUseId": "test-tool-use-id",
            "input": {"action": "write", "content": "This will cause an exception"},
        }

        result = journal.journal(tool=tool_use)
        assert result["status"] == "error"
        assert "Error: Test exception" in result["content"][0]["text"]


def test_journal_read_nonexistent(tmp_journal_dir):
    """Test reading a journal entry that doesn't exist."""
    # Create tool use with date that doesn't exist
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "action": "read",
            "date": "2099-12-31",  # Future date that shouldn't exist
        },
    }

    # Call the journal function
    result = journal.journal(tool=tool_use)

    # Verify error response
    assert result["status"] == "error"
    assert "No journal found for date" in result["content"][0]["text"]


def test_journal_via_agent(agent, tmp_journal_dir):
    """Test journal tool via the agent interface."""
    # Test writing a journal entry through the agent
    result = agent.tool.journal(action="write", content="Agent journal entry test")

    result_text = extract_result_text(result)
    assert "Added entry to journal" in result_text

    # Verify file was created
    today = datetime.now().strftime("%Y-%m-%d")
    journal_file = tmp_journal_dir / "journal" / f"{today}.md"
    assert journal_file.exists()
    assert "Agent journal entry test" in journal_file.read_text()


def test_create_rich_response():
    """Test the rich response creation function."""
    mock_console = Mock()

    # Test for write action
    journal.create_rich_response(
        mock_console,
        "write",
        {
            "date": "2023-01-01",
            "path": "/path/to/journal",
            "content": "Test content",
            "timestamp": "12:00:00",
        },
    )
    mock_console.print.assert_called_once()

    # Test for read action
    mock_console.print.reset_mock()
    journal.create_rich_response(mock_console, "read", {"date": "2023-01-01", "content": "## 12:00:00\nTest content"})
    mock_console.print.assert_called_once()

    # Test for list action
    mock_console.print.reset_mock()
    journal.create_rich_response(
        mock_console,
        "list",
        {"entries": [{"date": "2023-01-01", "entry_count": 2, "task_count": 3}]},
    )
    mock_console.print.assert_called_once()

    # Test for add_task action
    mock_console.print.reset_mock()
    journal.create_rich_response(
        mock_console,
        "add_task",
        {"date": "2023-01-01", "task": "Test task", "timestamp": "12:00:00"},
    )
    mock_console.print.assert_called_once()
