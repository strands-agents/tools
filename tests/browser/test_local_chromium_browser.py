"""
Unit tests for the LocalChromiumBrowser implementation.
"""

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from strands_tools.browser import LocalChromiumBrowser


def test_local_chromium_browser_initialization():
    """Test LocalChromiumBrowser initialization."""
    browser = LocalChromiumBrowser()
    assert browser._launch_options == {}
    assert browser._context_options == {}
    assert browser._default_launch_options == {}
    assert browser._default_context_options == {}


def test_local_chromium_browser_with_options():
    """Test LocalChromiumBrowser initialization with custom options."""
    launch_options = {"headless": True, "args": ["--no-sandbox"]}
    context_options = {"viewport": {"width": 1920, "height": 1080}}

    browser = LocalChromiumBrowser(launch_options=launch_options, context_options=context_options)

    assert browser._launch_options == launch_options
    assert browser._context_options == context_options


@patch.dict(
    os.environ,
    {
        "STRANDS_BROWSER_HEADLESS": "true",
        "STRANDS_BROWSER_WIDTH": "1920",
        "STRANDS_BROWSER_HEIGHT": "1080",
        "STRANDS_BROWSER_USER_DATA_DIR": "/tmp/test_browser",
    },
)
@patch("os.makedirs")
def test_local_chromium_browser_setup_configuration_with_env_vars(mock_makedirs):
    """Test configuration setup with environment variables."""
    browser = LocalChromiumBrowser()
    browser.start_platform()

    # Check that user data directory creation was attempted
    mock_makedirs.assert_called_once_with("/tmp/test_browser", exist_ok=True)

    # Check default launch options
    assert browser._default_launch_options["headless"] is True
    assert "--window-size=1920,1080" in browser._default_launch_options["args"]

    # Check default context options
    assert browser._default_context_options["viewport"] == {"width": 1920, "height": 1080}


@patch.dict(os.environ, {}, clear=True)
@patch("os.makedirs")
def test_local_chromium_browser_setup_configuration_defaults(mock_makedirs):
    """Test configuration setup with default values."""
    browser = LocalChromiumBrowser()
    browser.start_platform()

    # Check defaults
    assert browser._default_launch_options["headless"] is False
    assert "--window-size=1280,800" in browser._default_launch_options["args"]
    assert browser._default_context_options["viewport"] == {"width": 1280, "height": 800}


@pytest.mark.asyncio
async def test_local_chromium_browser_create_browser_session_regular():
    """Test creating a regular browser session."""
    mock_playwright_instance = Mock()
    mock_chromium = Mock()
    mock_browser = Mock()

    mock_playwright_instance.chromium = mock_chromium
    mock_chromium.launch = AsyncMock(return_value=mock_browser)

    browser = LocalChromiumBrowser()
    browser._playwright = mock_playwright_instance
    browser._default_launch_options = {"headless": True}

    result = await browser.create_browser_session()

    assert result == mock_browser
    mock_chromium.launch.assert_called_once_with(headless=True)


@pytest.mark.asyncio
async def test_local_chromium_browser_create_browser_session_persistent():
    """Test creating a persistent context browser session."""
    mock_playwright_instance = Mock()
    mock_chromium = Mock()
    mock_context = Mock()

    mock_playwright_instance.chromium = mock_chromium
    mock_chromium.launch_persistent_context = AsyncMock(return_value=mock_context)

    browser = LocalChromiumBrowser()
    browser._playwright = mock_playwright_instance
    browser._default_launch_options = {"persistent_context": True, "user_data_dir": "/tmp/test", "headless": True}

    result = await browser.create_browser_session()

    assert result == mock_context
    mock_chromium.launch_persistent_context.assert_called_once()


@pytest.mark.asyncio
async def test_local_chromium_browser_create_browser_session_no_playwright():
    """Test creating session browser without playwright initialized."""
    browser = LocalChromiumBrowser()

    with pytest.raises(RuntimeError, match="Playwright not initialized"):
        await browser.create_browser_session()


def test_local_chromium_browser_start_platform():
    """Test LocalChromiumBrowser start platform."""
    browser = LocalChromiumBrowser()
    # Should call _setup_configuration without errors
    browser.start_platform()

    # Verify configuration was set up
    assert browser._default_launch_options is not None
    assert browser._default_context_options is not None


def test_local_chromium_browser_close_platform():
    """Test LocalChromiumBrowser close platform (should be no-op)."""
    browser = LocalChromiumBrowser()
    # Should not raise any exceptions
    browser.close_platform()
