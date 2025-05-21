"""
Tests for the notify tool using the Agent interface.
"""

import json
import os
from unittest import mock

import pytest
from strands import Agent
from strands_tools import notify


@pytest.fixture
def agent():
    """Create an agent with the notify tool loaded."""
    return Agent(tools=[notify])


@pytest.fixture
def mock_history_file(tmp_path):
    """Create a temporary notification history file."""
    history_dir = tmp_path / "notifications"
    history_dir.mkdir()
    history_file = history_dir / "notification_history.jsonl"

    # Mock the NOTIFICATIONS_DIR and HISTORY_FILE constants
    with mock.patch.object(notify, "NOTIFICATIONS_DIR", history_dir):
        with mock.patch.object(notify, "HISTORY_FILE", history_file):
            yield history_file


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_basic_notification(mock_history_file):
    """Test basic notification functionality."""
    tool_use = {"toolUseId": "test-notify-1", "input": {"message": "Test notification", "title": "Test Title"}}

    result = notify.notify(tool=tool_use)

    # Check the result structure
    assert result["toolUseId"] == "test-notify-1"
    assert result["status"] == "success"
    assert "Test Title: Test notification" in result["content"][0]["text"]

    # Verify notification was logged to history file
    assert mock_history_file.exists()
    with open(mock_history_file, "r") as f:
        content = f.read()
        assert "Test notification" in content
        assert "Test Title" in content


def test_notification_priority_levels(mock_history_file):
    """Test notifications with different priority levels."""
    priorities = ["low", "normal", "high"]
    indicators = {"low": "📢", "normal": "🔔", "high": "⚠️"}

    for priority in priorities:
        tool_use = {
            "toolUseId": f"test-priority-{priority}",
            "input": {
                "message": f"{priority} priority message",
                "title": f"{priority.capitalize()} Priority Test",
                "priority": priority,
            },
        }

        result = notify.notify(tool=tool_use)

        # Check indicator in the result
        assert indicators[priority] in result["content"][0]["text"]


@mock.patch("strands_tools.notify._show_system_notification")
def test_system_notification(mock_show_notification, mock_history_file):
    """Test system notification functionality."""
    # Configure the mock to return True (success)
    mock_show_notification.return_value = True

    tool_use = {
        "toolUseId": "test-system-notify",
        "input": {"message": "System notification test", "title": "System Test", "show_system_notification": True},
    }

    result = notify.notify(tool=tool_use)

    # Check the notification was attempted
    mock_show_notification.assert_called_once_with("System Test", "System notification test")

    # Check success message is included
    assert any("System notification displayed" in item["text"] for item in result["content"])


def test_non_persistent_notification(mock_history_file):
    """Test non-persistent notification (not saved to history)."""
    tool_use = {
        "toolUseId": "test-non-persistent",
        "input": {"message": "This should not be saved", "title": "Non-persistent Test", "persistent": False},
    }

    result = notify.notify(tool=tool_use)

    # Check the notification was processed successfully
    assert result["status"] == "success"

    # If this is the first test, the file might not exist yet
    if mock_history_file.exists():
        with open(mock_history_file, "r") as f:
            content = f.read()
            assert "This should not be saved" not in content


def test_notification_with_source_and_category(mock_history_file):
    """Test notification with source and category fields."""
    tool_use = {
        "toolUseId": "test-metadata",
        "input": {
            "message": "Notification with metadata",
            "title": "Metadata Test",
            "source": "test_source",
            "category": "test_category",
        },
    }

    result = notify.notify(tool=tool_use)

    # Check source and category are included in the response
    assert "from: test_source" in result["content"][0]["text"]

    # Check they were saved to history
    with open(mock_history_file, "r") as f:
        content = f.readlines()
        notification_data = json.loads(content[-1])  # Get the last entry
        assert notification_data["source"] == "test_source"
        assert notification_data["category"] == "test_category"


def test_notification_history_retrieval(mock_history_file):
    """Test retrieving notification history."""
    # First, create some test notifications
    for i in range(5):
        tool_use = {
            "toolUseId": f"test-history-{i}",
            "input": {
                "message": f"History test message {i}",
                "title": f"History {i}",
                "category": "history_test" if i % 2 == 0 else "other_category",
                "priority": "high" if i % 3 == 0 else "normal",
            },
        }
        notify.notify(tool=tool_use)

    # Now retrieve the history
    tool_use = {"toolUseId": "test-get-history", "input": {"action": "history", "limit": 10}}

    result = notify.notify(tool=tool_use)

    # Check that we get notifications back
    assert result["status"] == "success"
    assert "Recent Notifications" in result["content"][0]["text"]
    assert len(result["content"]) > 1  # Should have header plus notifications


def test_notification_history_filtering(mock_history_file):
    """Test filtering notification history by category and priority."""
    # First, create notifications with different categories and priorities
    categories = ["category_a", "category_b"]
    priorities = ["low", "normal", "high"]

    for cat in categories:
        for pri in priorities:
            tool_use = {
                "toolUseId": f"test-filter-{cat}-{pri}",
                "input": {
                    "message": f"Filter test for {cat} with {pri} priority",
                    "title": f"Filter Test {cat} {pri}",
                    "category": cat,
                    "priority": pri,
                },
            }
            notify.notify(tool=tool_use)

    # Test filtering by category
    cat_filter_use = {
        "toolUseId": "test-filter-by-category",
        "input": {"action": "history", "filter_category": "category_a"},
    }

    cat_result = notify.notify(tool=cat_filter_use)

    # All results should contain category_a
    for item in cat_result["content"][1:]:  # Skip the header
        assert "category_a" in item["text"] or "[category: category_a]" in item["text"]

    # Test filtering by minimum priority
    pri_filter_use = {"toolUseId": "test-filter-by-priority", "input": {"action": "history", "min_priority": "high"}}

    pri_result = notify.notify(tool=pri_filter_use)

    # We should only see results with the high priority indicator
    high_indicator = "⚠️"
    for item in pri_result["content"][1:]:  # Skip the header
        assert high_indicator in item["text"]


def test_missing_message_error():
    """Test error handling when message is missing."""
    tool_use = {
        "toolUseId": "test-missing-message",
        "input": {
            "title": "No Message Test"
            # Missing required message field
        },
    }

    result = notify.notify(tool=tool_use)

    assert result["status"] == "error"
    assert "message is required" in result["content"][0]["text"]


def test_via_agent(agent):
    """Test notification via the agent interface."""
    result = agent.tool.notify(message="Test via agent", title="Agent Test")

    result_text = extract_result_text(result)
    assert "Agent Test: Test via agent" in result_text


@mock.patch("strands_tools.notify._show_system_notification")
def test_system_notification_failure(mock_show_notification, mock_history_file):
    """Test handling of system notification failure."""
    # Configure the mock to return False (failure)
    mock_show_notification.return_value = False

    tool_use = {
        "toolUseId": "test-system-fail",
        "input": {"message": "System should fail", "title": "System Fail Test", "show_system_notification": True},
    }

    result = notify.notify(tool=tool_use)

    # Check failure message is included
    assert any("System notification failed" in item["text"] for item in result["content"])


def test_empty_history(mock_history_file):
    """Test retrieving history when empty."""
    # Ensure history file doesn't exist yet
    if mock_history_file.exists():
        os.unlink(mock_history_file)

    tool_use = {"toolUseId": "test-empty-history", "input": {"action": "history"}}

    result = notify.notify(tool=tool_use)

    assert "No notifications found" in result["content"][0]["text"]
