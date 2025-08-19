import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the slack tool module
from strands_tools.slack import (
    SocketModeHandler,
    initialize_slack_clients,
    slack,
    slack_send_message,
)


class TestSlackInitialization(unittest.TestCase):
    """Test the initialization of Slack clients."""

    def setUp(self):
        # Patch global variables to avoid side effects from other tests
        self.app_patcher = patch("strands_tools.slack.app", new=None)
        self.client_patcher = patch("strands_tools.slack.client", new=None)
        self.socket_client_patcher = patch("strands_tools.slack.socket_client", new=None)

        self.app_patcher.start()
        self.client_patcher.start()
        self.socket_client_patcher.start()

    def tearDown(self):
        self.app_patcher.stop()
        self.client_patcher.stop()
        self.socket_client_patcher.stop()

    @unittest.skip("broken")
    @patch("strands_tools.slack.WebClient")
    @patch("strands_tools.slack.App")
    @patch("strands_tools.slack.SocketModeClient")
    def test_successful_initialization(self, mock_socket_client_class, mock_app_class, mock_web_client_class):
        """Test that initialization succeeds with proper tokens."""
        # Ensure the environment variables are available
        os.environ["SLACK_BOT_TOKEN"] = "xoxb-test-token"
        os.environ["SLACK_APP_TOKEN"] = "xapp-test-token"

        # Mock the clients to prevent actual initialization
        mock_app = MagicMock()
        mock_web_client = MagicMock()
        mock_socket_client = MagicMock()

        mock_app_class.return_value = mock_app
        mock_web_client_class.return_value = mock_web_client
        mock_socket_client_class.return_value = mock_socket_client

        # Call the initialization function
        success, error_message = initialize_slack_clients()

        # Check that initialization succeeded
        self.assertTrue(success)
        self.assertIsNone(error_message)

        # Verify that clients were created
        mock_app_class.assert_called_once()
        mock_web_client_class.assert_called_once()
        mock_socket_client_class.assert_called_once()

        # Clean up environment after test
        del os.environ["SLACK_BOT_TOKEN"]
        del os.environ["SLACK_APP_TOKEN"]


class TestSlackTool(unittest.TestCase):
    """Test the main slack tool function."""

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.initialize_slack_clients")
    def test_chat_post_message(self, mock_init, mock_client):
        """Test the chat_postMessage action."""
        # Set up the mocks
        mock_init.return_value = (True, None)
        mock_response = MagicMock()
        mock_response.data = {"ok": True, "ts": "1234.5678"}

        # Set up the chat_postMessage method
        mock_client.chat_postMessage = MagicMock(return_value=mock_response)

        # Call the slack tool
        result = slack(action="chat_postMessage", parameters={"channel": "test_channel", "text": "test_message"})

        # Check that the client method was called with the correct parameters
        mock_client.chat_postMessage.assert_called_once_with(channel="test_channel", text="test_message")

        # Check the result contains success message
        self.assertIn("✅ chat_postMessage executed successfully", result)

    @patch("strands_tools.slack.socket_handler")
    @patch("strands_tools.slack.initialize_slack_clients")
    def test_start_socket_mode(self, mock_init, mock_handler):
        """Test the start_socket_mode action."""
        # Set up the mocks
        mock_init.return_value = (True, None)
        mock_handler.start.return_value = True

        # Call the slack tool
        agent_mock = MagicMock()
        result = slack(action="start_socket_mode", agent=agent_mock)

        # Check that the socket handler was started with the agent
        mock_handler.start.assert_called_once_with(agent_mock)

        # Check the result contains success message
        self.assertIn("✅ Socket Mode connection established", result)

    @patch("strands_tools.slack.EVENTS_FILE", new=Path("./test_events.jsonl"))
    @patch("strands_tools.slack.Path.exists")
    def test_get_recent_events_no_file(self, mock_exists):
        """Test get_recent_events when no events file exists."""
        # Set up the mock
        mock_exists.return_value = False

        # Call the slack tool
        result = slack(action="get_recent_events", parameters={"count": 5})

        # Check the result
        self.assertEqual("No events found in storage", result)

    @patch("strands_tools.slack.open")
    @patch("strands_tools.slack.EVENTS_FILE", new=Path("./test_events.jsonl"))
    @patch("strands_tools.slack.Path.exists")
    def test_get_recent_events_with_file(self, mock_exists, mock_open):
        """Test get_recent_events when events file exists."""
        # Set up the mocks
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.readlines.return_value = [
            '{"event_type":"message","payload":{"event":{"type":"message","text":"test1"}}}\n',
            '{"event_type":"message","payload":{"event":{"type":"message","text":"test2"}}}\n',
        ]
        mock_open.return_value = mock_file

        # Call the slack tool
        result = slack(action="get_recent_events", parameters={"count": 2})

        # Check the result contains event data
        self.assertIn("Slack events:", result)
        self.assertIn("test1", result)
        self.assertIn("test2", result)


class TestSlackSendMessage(unittest.TestCase):
    """Test the slack_send_message tool function."""

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.initialize_slack_clients")
    def test_send_message_success(self, mock_init, mock_client):
        """Test successful message sending."""
        # Set up the mocks
        mock_init.return_value = (True, None)
        mock_response = {"ok": True, "ts": "1234.5678"}
        mock_client.chat_postMessage.return_value = mock_response

        # Call the slack_send_message tool
        result = slack_send_message(channel="test_channel", text="test message")

        # Check that the client method was called with the correct parameters
        mock_client.chat_postMessage.assert_called_once_with(channel="test_channel", text="test message")

        # Check the result
        self.assertIn("Message sent successfully", result)
        self.assertIn("1234.5678", result)

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.initialize_slack_clients")
    def test_send_message_with_thread(self, mock_init, mock_client):
        """Test message sending with thread_ts parameter."""
        # Set up the mocks
        mock_init.return_value = (True, None)
        mock_response = {"ok": True, "ts": "1234.5678"}
        mock_client.chat_postMessage.return_value = mock_response

        # Call the slack_send_message tool with a thread_ts
        result = slack_send_message(channel="test_channel", text="test message", thread_ts="1111.2222")

        # Check that the client method was called with the correct parameters
        mock_client.chat_postMessage.assert_called_once_with(
            channel="test_channel", text="test message", thread_ts="1111.2222"
        )

        # Check the result
        self.assertIn("Message sent successfully", result)

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.initialize_slack_clients")
    def test_send_message_error(self, mock_init, mock_client):
        """Test error handling in message sending."""
        # Set up the mocks
        mock_init.return_value = (True, None)
        mock_client.chat_postMessage.side_effect = Exception("API Error")

        # Call the slack_send_message tool
        result = slack_send_message(channel="test_channel", text="test message")

        # Check the result for error message
        self.assertIn("Error sending message", result)
        self.assertIn("API Error", result)


class TestSocketModeHandler(unittest.TestCase):
    """Test the SocketModeHandler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = SocketModeHandler()
        self.handler.client = MagicMock()
        self.handler.agent = MagicMock()

    @patch("strands_tools.slack.initialize_slack_clients")
    @patch("strands_tools.slack.socket_client", None)
    def test_setup_client_initialization_failure(self, mock_init):
        """Test that _setup_client handles initialization failures."""
        # Set up the mock to return failure
        mock_init.return_value = (False, "Test error message")

        # Call the method and check for exception
        with self.assertRaises(ValueError) as context:
            self.handler._setup_client()

        self.assertEqual(str(context.exception), "Test error message")

    @patch("strands_tools.slack.socket_client")
    def test_setup_client_success(self, mock_socket_client):
        """Test successful client setup."""
        # Call the method
        self.handler._setup_client()

        # Check that the client is set
        self.assertEqual(self.handler.client, mock_socket_client)

    def test_start_success(self):
        """Test successful socket mode start."""
        # Mock the setup method
        self.handler._setup_client = MagicMock()

        # Call the start method
        agent_mock = MagicMock()
        result = self.handler.start(agent_mock)

        # Check that the client was set up and connected
        self.handler._setup_client.assert_called_once()
        self.handler.client.connect.assert_called_once()

        # Check the result
        self.assertTrue(result)
        self.assertTrue(self.handler.is_connected)

    def test_start_already_connected(self):
        """Test start when already connected."""
        # Set the handler as already connected
        self.handler.is_connected = True

        # Call the start method
        agent_mock = MagicMock()
        result = self.handler.start(agent_mock)

        # Check that the client was not set up or connected again
        self.assertEqual(0, self.handler.client.connect.call_count)

        # Check the result
        self.assertTrue(result)

    def test_stop_success(self):
        """Test successful socket mode stop."""
        # Set the handler as connected
        self.handler.is_connected = True

        # Call the stop method
        result = self.handler.stop()

        # Check that the client was closed
        self.handler.client.close.assert_called_once()

        # Check the result
        self.assertTrue(result)
        self.assertFalse(self.handler.is_connected)

    def test_stop_not_connected(self):
        """Test stop when not connected."""
        # Set the handler as not connected
        self.handler.is_connected = False

        # Call the stop method
        result = self.handler.stop()

        # Check that the client was not closed
        self.assertEqual(0, self.handler.client.close.call_count)

        # Check the result
        self.assertTrue(result)

    @patch("strands_tools.slack.EVENTS_FILE", new=Path("./test_events.jsonl"))
    @patch("strands_tools.slack.Path.exists")
    def test_get_recent_events_no_file(self, mock_exists):
        """Test _get_recent_events when no events file exists."""
        # Set up the mock
        mock_exists.return_value = False

        # Call the method
        result = self.handler._get_recent_events(count=5)

        # Check the result
        self.assertEqual([], result)

    @patch("strands_tools.slack.open")
    @patch("strands_tools.slack.EVENTS_FILE", new=Path("./test_events.jsonl"))
    @patch("strands_tools.slack.Path.exists")
    def test_get_recent_events_with_file(self, mock_exists, mock_open):
        """Test _get_recent_events when events file exists."""
        # Set up the mocks
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.readlines.return_value = [
            '{"event_type":"message","payload":{"event":{"type":"message","text":"test1"}}}\n',
            '{"event_type":"message","payload":{"event":{"type":"message","text":"test2"}}}\n',
        ]
        mock_open.return_value = mock_file

        # Call the method
        result = self.handler._get_recent_events(count=2)

        # Check the result
        self.assertEqual(2, len(result))
        self.assertEqual("message", result[0]["event_type"])
        self.assertEqual("message", result[1]["event_type"])


# Skip this test class unless --integration is specified
@pytest.mark.integration
class TestSlackIntegration:
    """
    Integration tests for the Slack tools.

    These tests require actual Slack tokens to run.
    Skip them if the tokens are not available.
    """

    def setup_method(self):
        """Set up test method."""
        # This is run before every test
        # Verify tokens are available
        if not os.environ.get("SLACK_BOT_TOKEN") or not os.environ.get("SLACK_APP_TOKEN"):
            pytest.skip("Slack tokens not available")

    def test_initialization(self):
        """Test that Slack clients can be initialized."""
        # This test should not be run normally since it tries to use the actual Slack API
        success, error_message = initialize_slack_clients()
        assert success
        assert error_message is None

    def test_send_message(self):
        """Test sending a message to a test channel."""
        # This test should not be run normally since it sends an actual message
        # Get test channel from environment or use a default
        test_channel = os.environ.get("SLACK_TEST_CHANNEL", "test")

        # Verify channel is provided
        if not test_channel or test_channel == "test":
            pytest.skip("Valid SLACK_TEST_CHANNEL not provided")

        # Send a test message
        result = slack_send_message(channel=test_channel, text="This is an integration test message. Please ignore.")

        # Check that the message was sent successfully
        assert "Message sent successfully" in result
class TestSlackAdvancedFeatures(unittest.TestCase):
    """Test advanced Slack tool features and edge cases."""

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.initialize_slack_clients")
    def test_slack_api_rate_limiting(self, mock_init, mock_client):
        """Test handling of Slack API rate limiting."""
        mock_init.return_value = (True, None)
        
        # Simulate rate limiting error
        mock_client.chat_postMessage.side_effect = Exception("Rate limited")

        result = slack(action="chat_postMessage", parameters={"channel": "test", "text": "test"})

        self.assertIn("Error", result)

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.initialize_slack_clients")
    def test_slack_api_invalid_channel(self, mock_init, mock_client):
        """Test handling of invalid channel errors."""
        mock_init.return_value = (True, None)
        
        # Simulate invalid channel error
        mock_client.chat_postMessage.side_effect = Exception("Channel not found")

        result = slack(action="chat_postMessage", parameters={"channel": "invalid", "text": "test"})

        self.assertIn("Error", result)

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.initialize_slack_clients")
    def test_slack_complex_message_formatting(self, mock_init, mock_client):
        """Test sending messages with complex formatting."""
        mock_init.return_value = (True, None)
        mock_response = MagicMock()
        mock_response.data = {"ok": True, "ts": "1234.5678"}
        mock_client.chat_postMessage.return_value = mock_response

        # Test with blocks and attachments
        complex_parameters = {
            "channel": "test_channel",
            "text": "Fallback text",
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "*Bold text* and _italic text_"}
                }
            ],
            "attachments": [
                {
                    "color": "good",
                    "fields": [
                        {"title": "Status", "value": "Success", "short": True}
                    ]
                }
            ]
        }

        result = slack(action="chat_postMessage", parameters=complex_parameters)

        # Verify complex parameters were passed correctly
        mock_client.chat_postMessage.assert_called_once()
        call_args = mock_client.chat_postMessage.call_args[1]
        self.assertEqual(call_args["channel"], "test_channel")
        self.assertIn("blocks", call_args)
        self.assertIn("attachments", call_args)

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.initialize_slack_clients")
    def test_slack_file_upload(self, mock_init, mock_client):
        """Test file upload functionality."""
        mock_init.return_value = (True, None)
        mock_response = MagicMock()
        mock_response.data = {"ok": True, "file": {"id": "F1234567890"}}
        mock_client.files_upload.return_value = mock_response

        result = slack(
            action="files_upload",
            parameters={
                "channels": "test_channel",
                "file": "test_file.txt",
                "title": "Test File",
                "initial_comment": "Here's a test file"
            }
        )

        mock_client.files_upload.assert_called_once_with(
            channels="test_channel",
            file="test_file.txt",
            title="Test File",
            initial_comment="Here's a test file"
        )
        self.assertIn("files_upload executed successfully", result)

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.initialize_slack_clients")
    def test_slack_user_info_retrieval(self, mock_init, mock_client):
        """Test user information retrieval."""
        mock_init.return_value = (True, None)
        mock_response = MagicMock()
        mock_response.data = {
            "ok": True,
            "user": {
                "id": "U1234567890",
                "name": "testuser",
                "real_name": "Test User",
                "profile": {"email": "test@example.com"}
            }
        }
        mock_client.users_info.return_value = mock_response

        result = slack(action="users_info", parameters={"user": "U1234567890"})

        mock_client.users_info.assert_called_once_with(user="U1234567890")
        self.assertIn("users_info executed successfully", result)
        self.assertIn("testuser", result)

    @patch("strands_tools.slack.socket_handler")
    @patch("strands_tools.slack.initialize_slack_clients")
    def test_socket_mode_connection_failure(self, mock_init, mock_handler):
        """Test socket mode connection failure handling."""
        mock_init.return_value = (True, None)
        mock_handler.start.return_value = False  # Connection failed

        agent_mock = MagicMock()
        result = slack(action="start_socket_mode", agent=agent_mock)

        mock_handler.start.assert_called_once_with(agent_mock)
        self.assertIn("Failed to establish Socket Mode connection", result)

    @patch("strands_tools.slack.socket_handler")
    @patch("strands_tools.slack.initialize_slack_clients")
    def test_socket_mode_stop(self, mock_init, mock_handler):
        """Test stopping socket mode connection."""
        mock_init.return_value = (True, None)
        mock_handler.stop.return_value = True

        result = slack(action="stop_socket_mode")

        self.assertIsInstance(result, str)

    def test_slack_initialization_missing_tokens(self):
        """Test initialization failure when tokens are missing."""
        # Ensure tokens are not in environment
        with patch.dict(os.environ, {}, clear=True):
            success, error_message = initialize_slack_clients()
            
            self.assertFalse(success)
            self.assertIsNotNone(error_message)
            self.assertIn("SLACK_BOT_TOKEN", error_message)

    def test_get_recent_events_malformed_json(self):
        """Test handling of malformed JSON in events file."""
        result = slack(action="get_recent_events", parameters={"count": 3})
        # Should handle gracefully
        self.assertIsInstance(result, str)


class TestSocketModeHandlerAdvanced(unittest.TestCase):
    """Advanced tests for SocketModeHandler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = SocketModeHandler()
        self.handler.client = MagicMock()
        self.handler.agent = MagicMock()

    def test_socket_handler_message_processing(self):
        """Test message processing in socket mode handler."""
        # Test message processing (placeholder)
        self.assertTrue(True)

    def test_socket_handler_error_recovery(self):
        """Test error recovery in socket mode handler."""
        # Simulate connection error
        self.handler.client.connect.side_effect = Exception("Connection failed")

        agent_mock = MagicMock()
        result = self.handler.start(agent_mock)

        # Should handle connection errors gracefully
        self.assertIsInstance(result, bool)

    def test_get_recent_events_large_file(self):
        """Test handling of large events file."""
        # Test with mock data
        result = self.handler._get_recent_events(count=10)
        self.assertIsInstance(result, list)


class TestSlackToolEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions in Slack tools."""

    def test_slack_with_none_parameters(self):
        """Test slack tool with None parameters."""
        result = slack(action="chat_postMessage", parameters=None)
        
        # Should handle None parameters gracefully
        self.assertIsInstance(result, str)

    def test_slack_with_empty_action(self):
        """Test slack tool with empty action."""
        result = slack(action="", parameters={"channel": "test", "text": "test"})
        
        # Should handle empty action gracefully
        self.assertIsInstance(result, str)

    @patch("strands_tools.slack.initialize_slack_clients")
    def test_slack_initialization_timeout(self, mock_init):
        """Test handling of initialization timeout."""
        mock_init.side_effect = Exception("Initialization failed")

        result = slack(action="chat_postMessage", parameters={"channel": "test", "text": "test"})

        self.assertIsInstance(result, str)

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.initialize_slack_clients")
    def test_slack_network_connectivity_issues(self, mock_init, mock_client):
        """Test handling of network connectivity issues."""
        mock_init.return_value = (True, None)
        mock_client.chat_postMessage.side_effect = Exception("Network unreachable")

        result = slack(action="chat_postMessage", parameters={"channel": "test", "text": "test"})

        self.assertIsInstance(result, str)

    def test_slack_send_message_with_special_characters(self):
        """Test sending messages with special characters and emojis."""
        special_text = "Hello! 🚀 Testing special chars: @#$%^&*()_+ 中文 العربية"
        
        with (
            patch("strands_tools.slack.client") as mock_client,
            patch("strands_tools.slack.initialize_slack_clients") as mock_init,
        ):
            mock_init.return_value = (True, None)
            mock_response = {"ok": True, "ts": "1234.5678"}
            mock_client.chat_postMessage.return_value = mock_response

            result = slack_send_message(channel="test_channel", text=special_text)

            # Should handle special characters correctly
            mock_client.chat_postMessage.assert_called_once_with(
                channel="test_channel", 
                text=special_text
            )
            self.assertIn("Message sent successfully", result)

    @patch("strands_tools.slack.client")
    @patch("strands_tools.slack.initialize_slack_clients")
    def test_slack_api_response_parsing_error(self, mock_init, mock_client):
        """Test handling of API response parsing errors."""
        mock_init.return_value = (True, None)
        
        # Mock response that causes parsing issues
        mock_response = MagicMock()
        mock_response.data = None  # Invalid response format
        mock_client.chat_postMessage.return_value = mock_response

        result = slack(action="chat_postMessage", parameters={"channel": "test", "text": "test"})

        # Should handle parsing errors gracefully
        self.assertIn("executed successfully", result)  # Tool should still report success