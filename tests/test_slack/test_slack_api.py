import unittest
from unittest.mock import MagicMock, patch

from slack_sdk.errors import SlackApiError

from strands_tools.slack import slack


class TestSlackApiMethods(unittest.TestCase):
    """Test interactions with the Slack API."""

    def setUp(self):
        """Set up the test case."""
        # Patch the global client to avoid side effects
        self.client_patcher = patch("strands_tools.slack.client")
        self.mock_client = self.client_patcher.start()

        # Patch initialize_slack_clients to return success
        self.init_patcher = patch("strands_tools.slack.initialize_slack_clients")
        self.mock_init = self.init_patcher.start()
        self.mock_init.return_value = (True, None)

    def tearDown(self):
        """Tear down the test case."""
        self.client_patcher.stop()
        self.init_patcher.stop()

    def test_chat_post_message_success(self):
        """Test a successful chat.postMessage API call."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.data = {"ok": True, "channel": "C123456", "ts": "1234.5678", "message": {"text": "Hello world"}}
        self.mock_client.chat_postMessage.return_value = mock_response

        # Call the slack tool
        result = slack(
            action="chat_postMessage",
            parameters={
                "channel": "C123456",
                "text": "Hello world",
                "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": "*Hello* world"}}],
            },
        )

        # Check that the client method was called with the correct parameters
        self.mock_client.chat_postMessage.assert_called_once_with(
            channel="C123456",
            text="Hello world",
            blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "*Hello* world"}}],
        )

        # Check the result
        self.assertIn("✅ chat_postMessage executed successfully", result)
        self.assertIn("1234.5678", result)

    def test_reactions_add(self):
        """Test a reactions.add API call."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.data = {"ok": True}
        self.mock_client.reactions_add.return_value = mock_response

        # Call the slack tool
        result = slack(
            action="reactions_add", parameters={"channel": "C123456", "timestamp": "1234.5678", "name": "thumbsup"}
        )

        # Check that the client method was called with the correct parameters
        self.mock_client.reactions_add.assert_called_once_with(
            channel="C123456", timestamp="1234.5678", name="thumbsup"
        )

        # Check the result
        self.assertIn("✅ reactions_add executed successfully", result)

    def test_conversations_list(self):
        """Test a conversations.list API call."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.data = {
            "ok": True,
            "channels": [{"id": "C123456", "name": "general"}, {"id": "C789012", "name": "random"}],
        }
        self.mock_client.conversations_list.return_value = mock_response

        # Call the slack tool
        result = slack(action="conversations_list", parameters={"types": "public_channel"})

        # Check that the client method was called with the correct parameters
        self.mock_client.conversations_list.assert_called_once_with(types="public_channel")

        # Check the result
        self.assertIn("✅ conversations_list executed successfully", result)
        self.assertIn("general", result)
        self.assertIn("random", result)

    def test_api_error(self):
        """Test handling of Slack API errors."""
        # Set up the mock error response
        mock_error_response = {"ok": False, "error": "channel_not_found"}
        self.mock_client.chat_postMessage.side_effect = SlackApiError(
            message="channel_not_found", response=mock_error_response
        )

        # Call the slack tool
        result = slack(action="chat_postMessage", parameters={"channel": "INVALID", "text": "This will fail"})

        # Check the result
        self.assertIn("Error: channel_not_found", result)

    def test_rate_limit_error(self):
        """Test handling of rate limit errors."""
        # Set up the mock error response
        mock_error_response = {"ok": False, "error": "ratelimited"}
        self.mock_client.chat_postMessage.side_effect = SlackApiError(
            message="ratelimited", response=mock_error_response
        )

        # Call the slack tool
        result = slack(action="chat_postMessage", parameters={"channel": "C123456", "text": "Rate limited"})

        # Check the result
        self.assertIn("Error: ratelimited", result)

    def test_files_upload(self):
        """Test a files.upload API call."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.data = {
            "ok": True,
            "file": {"id": "F123456", "name": "test.txt", "permalink": "https://slack.com/files/test"},
        }
        self.mock_client.files_upload.return_value = mock_response

        # Call the slack tool
        result = slack(
            action="files_upload",
            parameters={
                "channels": "C123456",
                "content": "Test file content",
                "filename": "test.txt",
                "title": "Test Upload",
            },
        )

        # Check that the client method was called with the correct parameters
        self.mock_client.files_upload.assert_called_once_with(
            channels="C123456", content="Test file content", filename="test.txt", title="Test Upload"
        )

        # Check the result
        self.assertIn("✅ files_upload executed successfully", result)
        self.assertIn("F123456", result)
