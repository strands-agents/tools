"""
Tests for the speak tool using the Agent interface.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from botocore.config import Config as BotocoreConfig
from strands import Agent

from strands_tools import speak


@pytest.fixture
def agent():
    """Create an agent with the speak tool loaded."""
    return Agent(tools=[speak])


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


@patch("subprocess.run")
def test_speak_fast_mode(mock_run):
    """Test the speak tool in fast mode."""
    # Create a tool use dictionary similar to how the agent would call it
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"text": "Hello world", "mode": "fast"}}

    # Call the speak function directly
    result = speak.speak(tool=tool_use)

    # Verify the result has the expected structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "Text spoken using macOS say command" in result["content"][0]["text"]

    # Verify subprocess.run was called with the right parameters
    mock_run.assert_called_once_with(["say", "Hello world"], check=True)


@patch("subprocess.run")
def test_speak_fast_mode_no_play(mock_run):
    """Test the speak tool in fast mode with play_audio=False."""
    tool_use = {"toolUseId": "test-tool-use-id", "input": {"text": "Hello world", "mode": "fast", "play_audio": False}}

    result = speak.speak(tool=tool_use)

    assert result["status"] == "success"
    assert "Text processed using macOS say command (audio not played)" in result["content"][0]["text"]

    # Verify subprocess.run was not called since we're not playing audio
    mock_run.assert_not_called()


@patch("boto3.client")
@patch("subprocess.run")
def test_speak_polly_mode(mock_run, mock_boto_client):
    """Test the speak tool in polly mode."""
    # Setup mock for boto3 client
    mock_polly = MagicMock()
    mock_boto_client.return_value = mock_polly

    # Mock the AudioStream
    mock_audio_stream = MagicMock()
    mock_polly.synthesize_speech.return_value = {"AudioStream": mock_audio_stream}

    # Mock the open function for file writing
    mock_open = MagicMock()

    with patch("builtins.open", mock_open):
        tool_use = {
            "toolUseId": "test-tool-use-id",
            "input": {"text": "Hello world", "mode": "polly", "voice_id": "Joanna", "output_path": "test_output.mp3"},
        }

        result = speak.speak(tool=tool_use)

        assert result["status"] == "success"
        assert "Generated and played speech using Polly" in result["content"][0]["text"]

        # Verify boto3 client was created correctly with user agent
        mock_boto_client.assert_called_once()
        args, kwargs = mock_boto_client.call_args
        assert args[0] == "polly"
        assert kwargs["region_name"] == "us-west-2"
        assert "config" in kwargs
        config = kwargs["config"]
        assert isinstance(config, BotocoreConfig)
        assert config.user_agent_extra == "strands-agents-speak"

        # Verify synthesize_speech was called with the right parameters
        mock_polly.synthesize_speech.assert_called_once_with(
            Engine="neural", OutputFormat="mp3", Text="Hello world", VoiceId="Joanna"
        )

        # Verify file was opened for writing
        mock_open.assert_called_once_with("test_output.mp3", "wb")

        # Verify subprocess.run was called to play the audio
        mock_run.assert_called_once_with(["afplay", "test_output.mp3"], check=True)


@patch("boto3.client")
@patch("subprocess.run")
def test_speak_polly_mode_no_play(mock_run, mock_boto_client):
    """Test the speak tool in polly mode with play_audio=False."""
    # Setup mock for boto3 client
    mock_polly = MagicMock()
    mock_boto_client.return_value = mock_polly

    # Mock the AudioStream
    mock_audio_stream = MagicMock()
    mock_polly.synthesize_speech.return_value = {"AudioStream": mock_audio_stream}

    # Mock the open function for file writing
    mock_open = MagicMock()

    with patch("builtins.open", mock_open):
        tool_use = {
            "toolUseId": "test-tool-use-id",
            "input": {
                "text": "Hello world",
                "mode": "polly",
                "voice_id": "Joanna",
                "output_path": "test_output.mp3",
                "play_audio": False,
            },
        }

        result = speak.speak(tool=tool_use)

        assert result["status"] == "success"
        assert "Generated speech using Polly" in result["content"][0]["text"]
        assert "audio not played" in result["content"][0]["text"]

        # Verify boto3 client was created correctly with user agent
        mock_boto_client.assert_called_once()
        args, kwargs = mock_boto_client.call_args
        assert args[0] == "polly"
        assert kwargs["region_name"] == "us-west-2"
        assert "config" in kwargs
        config = kwargs["config"]
        assert isinstance(config, BotocoreConfig)
        assert config.user_agent_extra == "strands-agents-speak"

        # Verify synthesize_speech was called with the right parameters
        mock_polly.synthesize_speech.assert_called_once_with(
            Engine="neural", OutputFormat="mp3", Text="Hello world", VoiceId="Joanna"
        )

        # Verify file was opened for writing
        mock_open.assert_called_once_with("test_output.mp3", "wb")

        # Verify subprocess.run was not called since we're not playing audio
        mock_run.assert_not_called()


@patch("boto3.client")
def test_speak_polly_no_audio_stream(mock_boto_client):
    """Test the speak tool in polly mode when no AudioStream is returned."""
    # Setup mock for boto3 client
    mock_polly = MagicMock()
    mock_boto_client.return_value = mock_polly

    # Return a response without AudioStream
    mock_polly.synthesize_speech.return_value = {}

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "text": "Hello world",
            "mode": "polly",
        },
    }

    result = speak.speak(tool=tool_use)

    assert result["status"] == "error"
    assert "No AudioStream in response from Polly" in result["content"][0]["text"]

    # Verify boto3 client was created correctly with user agent
    mock_boto_client.assert_called_once()
    args, kwargs = mock_boto_client.call_args
    assert args[0] == "polly"
    assert kwargs["region_name"] == "us-west-2"
    assert "config" in kwargs
    config = kwargs["config"]
    assert isinstance(config, BotocoreConfig)
    assert config.user_agent_extra == "strands-agents-speak"


@patch("subprocess.run")
def test_speak_exception_handling(mock_run):
    """Test exception handling in the speak tool."""
    # Make subprocess.run raise an exception
    mock_run.side_effect = Exception("Test exception")

    tool_use = {"toolUseId": "test-tool-use-id", "input": {"text": "Hello world"}}

    result = speak.speak(tool=tool_use)

    assert result["status"] == "error"
    assert "Error generating speech" in result["content"][0]["text"]
    assert "Test exception" in result["content"][0]["text"]


def test_create_status_table():
    """Test the create_status_table function."""
    table = speak.create_status_table("fast", "Hello world")
    assert table is not None

    # Test with polly mode
    table = speak.create_status_table("polly", "Hello world", "Joanna", "output.mp3", True)
    assert table is not None


def test_display_speech_status():
    """Test the display_speech_status function."""
    # This function just prints to console, so we're just testing it doesn't raise exceptions
    mock_console = Mock()
    speak.display_speech_status(mock_console, "Test", "Test message", "green")
    speak.display_speech_status(mock_console, "Error", "Error message", "red")


def test_via_agent(agent):
    """Test speaking via the agent interface (this is a simplified test)."""
    with patch("subprocess.run") as mock_run:
        result = agent.tool.speak(text="Hello via agent")

        result_text = extract_result_text(result)
        assert "Text spoken using macOS say command" in result_text
        mock_run.assert_called_once_with(["say", "Hello via agent"], check=True)
