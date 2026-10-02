"""
Unit tests for the Browser base class using MockBrowser.
"""

import asyncio
import concurrent.futures
from unittest.mock import AsyncMock, Mock, patch

from playwright.async_api import Browser as PlaywrightBrowser

from strands_tools.browser import Browser


class MockBrowser(Browser):
    """Mock implementation of Browser for testing."""

    def start_platform(self) -> None:
        """Mock platform startup."""
        pass

    def close_platform(self) -> None:
        """Mock platform cleanup."""
        pass

    async def create_browser_session(self) -> PlaywrightBrowser:
        """Mock browser session creation."""
        return Mock()


@patch("strands_tools.browser.browser.async_playwright")
def test_browser_start_platform(mock_async_playwright):
    """Test platform startup."""
    mock_playwright_instance = Mock()
    mock_async_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)

    browser = MockBrowser()
    browser._start()

    # Verify playwright was started
    mock_async_playwright.assert_called_once()


def test_browser_browser_tool_creation():
    """Test that browser tool is created properly."""
    browser = MockBrowser()

    # Check that browser is a tool function
    assert hasattr(browser, "browser")
    assert callable(browser.browser)


@patch("strands_tools.browser.browser.async_playwright")
def test_browser_cleanup_platform(mock_async_playwright):
    """Test platform cleanup."""
    mock_playwright_instance = Mock()
    mock_async_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)

    browser = MockBrowser()
    browser._start()

    # Mock some sessions
    browser._sessions = {"session1": {"browser": AsyncMock()}, "session2": {"browser": AsyncMock()}}

    browser._cleanup()

    # Verify cleanup was called
    assert browser._sessions == {}


@patch("strands_tools.browser.browser.async_playwright")
def test_browser_tool_integration(mock_async_playwright):
    """Test that browser tool integrates properly with Strands."""
    mock_playwright_instance = Mock()
    mock_async_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)

    browser = MockBrowser()

    # Test that the tool is properly decorated
    assert hasattr(browser, "browser")
    assert callable(browser.browser)

    # Test tool metadata
    tool_func = browser.browser
    assert hasattr(tool_func, "__name__")
    assert tool_func.__name__ == "browser"


def test_execute_async_works_from_foreign_thread():
    """_execute_async must work when called from a thread other than the one that ran __init__.

    The strands SDK dispatches sync tools via asyncio.to_thread, which runs the tool
    on a worker thread. Without setting the event loop on that thread,
    nest_asyncio.apply() fails with 'There is no current event loop in thread'.
    """
    browser = MockBrowser()

    async def dummy():
        return "ok"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(browser._execute_async, dummy()).result(timeout=5)

    assert result == "ok"
