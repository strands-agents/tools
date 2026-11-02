import pytest

pytest.importorskip("strands_tools.slack", reason="slack was removed in v0.9.0", exc_type=ImportError)

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from strands_tools.slack import slack


class TestSlackEventStorage(unittest.TestCase):
    """Test Slack event storage and retrieval."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test events
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_events_dir = Path(self.temp_dir.name)
        self.test_events_file = self.test_events_dir / "events.jsonl"

        # Patch the module constants with our test paths
        self.events_dir_patcher = patch("strands_tools.slack.EVENTS_DIR", new=self.test_events_dir)
        self.events_file_patcher = patch("strands_tools.slack.EVENTS_FILE", new=self.test_events_file)

        self.events_dir_patcher.start()
        self.events_file_patcher.start()

    def tearDown(self):
        """Tear down test fixtures."""
        self.events_dir_patcher.stop()
        self.events_file_patcher.stop()
        self.temp_dir.cleanup()

    @patch("strands_tools.slack.Path.exists")
    def test_get_recent_events_no_file(self, mock_exists):
        """Test get_recent_events when no events file exists."""
        # Set up the mock
        mock_exists.return_value = False

        # Call the slack tool
        result = slack(action="get_recent_events", parameters={"count": 5})

        # Check the result
        self.assertEqual("No events found in storage", result)

    @patch("strands_tools.slack.initialize_slack_clients")
    def test_get_recent_events_with_file(self, mock_init):
        """Test get_recent_events when events file exists with real data."""
        # Ensure the directory exists
        self.test_events_dir.mkdir(parents=True, exist_ok=True)

        # Write some test events
        test_events = [
            {"event_type": "message", "payload": {"event": {"type": "message", "text": "test1"}}},
            {"event_type": "reaction_added", "payload": {"event": {"type": "reaction_added", "reaction": "thumbsup"}}},
            {"event_type": "message", "payload": {"event": {"type": "message", "text": "test2"}}},
        ]

        with open(self.test_events_file, "w") as f:
            for event in test_events:
                f.write(json.dumps(event) + "\n")

        # Initialize slack clients mock
        mock_init.return_value = (True, None)

        # Call the slack tool
        result = slack(action="get_recent_events", parameters={"count": 2})

        # Check the result
        self.assertIn("Slack events:", result)
        self.assertIn("test2", result)  # Should contain the latest event
        self.assertIn("reaction_added", result)  # Should contain the second latest event

        # Should not contain the oldest event as we limited to 2
        self.assertNotIn("test1", result)

    @patch("strands_tools.slack.initialize_slack_clients")
    def test_get_recent_events_invalid_json(self, mock_init):
        """Test get_recent_events with invalid JSON in the events file."""
        # Ensure the directory exists
        self.test_events_dir.mkdir(parents=True, exist_ok=True)

        # Write some test events with invalid JSON
        with open(self.test_events_file, "w") as f:
            f.write(json.dumps({"event_type": "message", "payload": {"text": "valid"}}) + "\n")
            f.write("This is not valid JSON\n")
            f.write(json.dumps({"event_type": "message", "payload": {"text": "also valid"}}) + "\n")

        # Initialize slack clients mock
        mock_init.return_value = (True, None)

        # Call the slack tool
        result = slack(action="get_recent_events", parameters={"count": 3})

        # Check the result
        self.assertIn("Slack events:", result)
        self.assertIn("valid", result)
        self.assertIn("also valid", result)

    @patch("strands_tools.slack.initialize_slack_clients")
    def test_get_recent_events_empty_file(self, mock_init):
        """Test get_recent_events with an empty events file."""
        # Ensure the directory exists
        self.test_events_dir.mkdir(parents=True, exist_ok=True)

        # Create an empty file
        open(self.test_events_file, "w").close()

        # Initialize slack clients mock
        mock_init.return_value = (True, None)

        # Call the slack tool
        result = slack(action="get_recent_events", parameters={"count": 5})

        # Check the result
        self.assertEqual("No valid events found in storage", result)
