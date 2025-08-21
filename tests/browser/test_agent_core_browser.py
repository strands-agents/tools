"""
Unit tests for the AgentCoreBrowser implementation.
"""

from unittest.mock import Mock

import pytest
from strands_tools.browser import AgentCoreBrowser


def test_bedrock_browser_initialization():
    """Test AgentCoreBrowser initialization."""
    browser = AgentCoreBrowser()
    assert browser.region is not None  # Should resolve from environment or default
    assert browser.session_timeout == 3600
    assert browser._client_dict == {}


def test_bedrock_browser_with_custom_params():
    """Test AgentCoreBrowser initialization with custom parameters."""
    browser = AgentCoreBrowser(region="us-east-1", session_timeout=7200)
    assert browser.region == "us-east-1"
    assert browser.session_timeout == 7200


def test_bedrock_browser_identifier_assignment():
    """Test AgentCoreBrowser identifier parameter assignment."""
    # Test default identifier
    browser_default = AgentCoreBrowser()
    assert hasattr(browser_default, "identifier")
    assert browser_default.identifier == "aws.browser.v1"

    # Test custom identifier
    custom_identifier = "my.custom.browser.v2"
    browser_custom = AgentCoreBrowser(identifier=custom_identifier)
    assert browser_custom.identifier == custom_identifier

    # Test identifier with other parameters
    browser_all_params = AgentCoreBrowser(region="us-west-2", identifier="test.browser.v1", session_timeout=3600)
    assert browser_all_params.identifier == "test.browser.v1"
    assert browser_all_params.region == "us-west-2"
    assert browser_all_params.session_timeout == 3600


@pytest.mark.asyncio
async def test_bedrock_browser_create_browser_session_no_playwright():
    """Test creating session browser without playwright initialized."""
    browser = AgentCoreBrowser()

    with pytest.raises(RuntimeError, match="Playwright not initialized"):
        await browser.create_browser_session()


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
