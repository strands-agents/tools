import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from slack_sdk.socket_mode.request import SocketModeRequest

from strands_tools.slack import SocketModeHandler


class TestSocketModeHandlerProcessing(unittest.TestCase):
    """Test the event processing functions of SocketModeHandler."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = SocketModeHandler()
        self.handler.client = MagicMock()
        self.mock_agent = MagicMock()
        # Create a tool registry structure manually for the mock agent
        self.mock_agent.tool_registry = MagicMock()
        self.mock_agent.tool_registry.registry = MagicMock()
        self.mock_agent.tool_registry.registry.values.return_value = ["tool1", "tool2"]
        self.mock_agent.system_prompt = "Test system prompt"
        self.handler.agent = self.mock_agent

        # Create a mock for bot info
        self.handler.bot_info = {"user_id": "BOT_USER_ID", "bot_id": "BOT_ID"}

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.EVENTS_FILE", new=Path("./test_events.jsonl"))
    @patch("strands_tools.slack.EVENTS_DIR", new=Path("./test_events"))
    @patch("strands_tools.slack.Path.mkdir")
    @patch("strands_tools.slack.open", new_callable=mock_open)
    def test_process_event_message(self, mock_file, mock_mkdir, mock_client):
        """Test processing a message event."""
        # Create a mock event request
        event_request = MagicMock()
        event_request.type = "events_api"
        event_request.envelope_id = "123456"
        event_request.payload = {
            "event": {
                "type": "message",
                "text": "Hello test",
                "user": "USER123",
                "channel": "CHANNEL123",
                "ts": "1234.5678",
            }
        }

        # Set up listeners list
        self.handler.client.socket_mode_request_listeners = []

        # Set up the handler
        self.handler._setup_listeners()

        # There should now be one listener
        self.assertEqual(1, len(self.handler.client.socket_mode_request_listeners))

        # Call the listener with our mock request
        listener = self.handler.client.socket_mode_request_listeners[0]
        listener(self.handler.client, event_request)

        # Verify that a response was sent
        self.handler.client.send_socket_mode_response.assert_called_once()

        # Check that the event was written to file
        mock_file().write.assert_called()

        # Verify directory was created
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.Agent")
    def test_process_message(self, mock_agent_class, mock_client):
        """Test processing a message through the agent."""
        # Create a mock event
        event = {"type": "message", "text": "Hello test", "user": "USER123", "channel": "CHANNEL123", "ts": "1234.5678"}

        # Create the mock agent instance
        mock_agent_instance = MagicMock()
        mock_agent_instance.return_value = "Test response"
        mock_agent_class.return_value = mock_agent_instance

        # Mock the environment variable for auto-reply
        with patch.dict("os.environ", {"STRANDS_SLACK_AUTO_REPLY": "true"}):
            # Process the message
            self.handler._process_message(event)

            # Check that the agent was instantiated
            mock_agent_class.assert_called_once()

            # Check that reactions were added
            mock_client.reactions_add.assert_called()

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.Agent")
    def test_process_own_message(self, mock_agent_class, mock_client):
        """Test that the handler ignores its own messages."""
        # Create a mock event with the bot's user ID
        event = {
            "type": "message",
            "text": "Hello test",
            "user": "BOT_USER_ID",  # Same as bot_info user_id
            "channel": "CHANNEL123",
            "ts": "1234.5678",
        }

        # Process the message
        self.handler._process_message(event)

        # Check that no agent was created and no message was posted
        mock_agent_class.assert_not_called()
        mock_client.chat_postMessage.assert_not_called()

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.Agent")
    def test_process_message_with_tag(self, mock_agent_class, mock_client):
        """Test that the handler only processes messages with the tag."""
        # Create a mock event
        event = {
            "type": "message",
            "text": "Hello test #strands",
            "user": "USER123",
            "channel": "CHANNEL123",
            "ts": "1234.5678",
        }

        # Mock the environment variable for listen only tag
        with patch.dict("os.environ", {"STRANDS_SLACK_LISTEN_ONLY_TAG": "#strands"}):
            # Process the message
            self.handler._process_message(event)

            # Check that the agent was instantiated
            mock_agent_class.assert_called_once()

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.Agent")
    def test_process_message_without_tag(self, mock_agent_class, mock_client):
        """Test that the handler ignores messages without the tag."""
        # Create a mock event without the tag
        event = {
            "type": "message",
            "text": "Hello test",  # No tag
            "user": "USER123",
            "channel": "CHANNEL123",
            "ts": "1234.5678",
        }

        # Set up mock agent with return_value
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        # Set the listen only tag
        with patch.dict("os.environ", {"STRANDS_SLACK_LISTEN_ONLY_TAG": "#strands"}):
            # Process the message
            self.handler._process_message(event)

            # Check that the agent was not called
            # Note: The agent class is created but never called
            self.assertEqual(0, mock_agent_instance.call_count)


def create_mock_socket_mode_request(event_type="message", text="Test message"):
    """Helper function to create mock SocketModeRequest objects."""
    request = MagicMock(spec=SocketModeRequest)
    request.type = "events_api"
    request.envelope_id = "test_envelope_id"

    if event_type == "message":
        request.payload = {
            "event": {"type": "message", "text": text, "user": "USER123", "channel": "CHANNEL123", "ts": "1234.5678"}
        }
    elif event_type == "interactive":
        request.payload = {
            "type": "interactive",
            "channel": {"id": "CHANNEL123"},
            "user": {"id": "USER123"},
            "message": {"ts": "1234.5678"},
            "actions": [{"action_id": "test_action", "value": "test_value"}],
        }

    return request
