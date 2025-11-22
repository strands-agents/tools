"""
Unit tests for the AgentCoreBrowser implementation.
"""

from unittest.mock import Mock, patch

import pytest

from strands_tools.browser import AgentCoreBrowser


def test_bedrock_browser_initialization():
    """Test AgentCoreBrowser initialization."""
    browser = AgentCoreBrowser()
    assert browser.region is not None  # Should resolve from environment or default
    assert browser.session_timeout == 3600
    assert browser._client_dict == {}


def test_bedrock_browser_registers_atexit_handler():
    """Test that close_platform is registered with atexit on initialization."""

    with patch("strands_tools.browser.browser.atexit.register") as mock_atexit:
        browser = AgentCoreBrowser()

        # Verify atexit.register was called once
        mock_atexit.assert_called_once()

        # Verify it registered the close_platform method
        registered_func = mock_atexit.call_args[0][0]
        assert callable(registered_func)
        # Verify it's the browser's close_platform method
        assert registered_func == browser.close_platform


def test_bedrock_browser_with_custom_params():
    """Test AgentCoreBrowser initialization with custom parameters."""
    browser = AgentCoreBrowser(region="us-east-1", session_timeout=7200)
    assert browser.region == "us-east-1"
    assert browser.session_timeout == 7200


def test_bedrock_browser_with_custom_identifier():
    """Test AgentCoreBrowser initialization with custom identifier."""
    custom_identifier = "my-custom-browser-abc123"
    browser = AgentCoreBrowser(identifier=custom_identifier)
    assert browser.identifier == custom_identifier


def test_bedrock_browser_default_identifier():
    """Test AgentCoreBrowser uses default identifier when none provided."""
    browser = AgentCoreBrowser()
    assert browser.identifier == "aws.browser.v1"


def test_bedrock_browser_none_identifier():
    """Test AgentCoreBrowser uses default identifier when None provided."""
    browser = AgentCoreBrowser(identifier=None)
    assert browser.identifier == "aws.browser.v1"


def test_bedrock_browser_empty_string_identifier():
    """Test AgentCoreBrowser uses default identifier when empty string provided."""
    browser = AgentCoreBrowser(identifier="")
    assert browser.identifier == "aws.browser.v1"


@pytest.mark.asyncio
async def test_bedrock_browser_create_browser_session_no_playwright():
    """Test creating session browser without playwright initialized."""
    browser = AgentCoreBrowser()

    with pytest.raises(RuntimeError, match="Playwright not initialized"):
        await browser.create_browser_session()


@pytest.mark.asyncio
async def test_bedrock_browser_create_browser_session_hydrates_client_dict():
    """Test that _client_dict is hydrated when create_browser_session is called."""
    from unittest.mock import AsyncMock, patch

    browser = AgentCoreBrowser()

    # Mock playwright
    mock_playwright = Mock()
    mock_chromium = Mock()
    mock_browser = Mock()
    mock_playwright.chromium = mock_chromium
    mock_chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
    browser._playwright = mock_playwright

    # Mock BrowserClient
    mock_client = Mock()
    mock_session_id = "test-session-123"
    mock_client.start.return_value = mock_session_id
    mock_client.generate_ws_headers.return_value = ("ws://test-url", {"header": "value"})

    with patch("strands_tools.browser.agent_core_browser.AgentCoreBrowserClient", return_value=mock_client):
        # Verify _client_dict is empty before
        assert browser._client_dict == {}

        # Create browser session
        await browser.create_browser_session()

        # Verify _client_dict is hydrated with the session
        assert mock_session_id in browser._client_dict
        assert browser._client_dict[mock_session_id] == mock_client
        assert len(browser._client_dict) == 1


def test_bedrock_browser_start_platform():
    """Test AgentCoreBrowser browser start platform (should be no-op)."""
    browser = AgentCoreBrowser()
    # Should not raise any exceptions
    browser.start_platform()


def test_bedrock_browser_close_platform():
    """Test AgentCoreBrowser browser close platform."""
    # Mock clients
    mock_client1 = Mock()
    mock_client2 = Mock()

    browser = AgentCoreBrowser()
    browser._client_dict = {"session1": mock_client1, "session2": mock_client2}

    browser.close_platform()

    # Verify all clients were stopped
    mock_client1.stop.assert_called_once()
    mock_client2.stop.assert_called_once()


def test_bedrock_browser_close_platform_with_errors():
    """Test AgentCoreBrowser browser close platform with client errors."""
    # Mock clients with one that raises an exception
    mock_client1 = Mock()
    mock_client2 = Mock()
    mock_client2.stop.side_effect = Exception("Stop failed")
    mock_client2.session_id = "session-2"

    browser = AgentCoreBrowser()
    browser._client_dict = {"session1": mock_client1, "session2": mock_client2}

    # Should not raise exception even if client.stop() fails
    browser.close_platform()

    # Verify all clients were attempted to be stopped
    mock_client1.stop.assert_called_once()
    mock_client2.stop.assert_called_once()
