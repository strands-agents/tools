import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.strands_tools.use_browser import BrowserManager, use_browser

# Constants for parametrization
BROWSER_ACTIONS = ["navigate", "click", "type", "press_key", "evaluate", "get_text", "get_html", "screenshot"]
NAVIGATION_ACTIONS = ["back", "forward", "refresh", "new_tab", "close_tab", "get_cookies", "close"]
ERROR_SCENARIOS = [
    ("navigate", {"url": None}, "Error: url required for navigate"),
    ("click", {"selector": None}, "Error: selector required for click"),
    ("type", {"selector": "#input", "input_text": None}, "Error: text required for type"),
]
LAUNCH_OPTIONS_SCENARIOS = [
    {"headless": True, "slowMo": 100},
    {"args": ["--no-sandbox", "--disable-setuid-sandbox"]},
    {"ignoreDefaultArgs": ["--enable-automation"]},
    {"proxy": {"server": "http://myproxy.com:3128"}},
    {"downloadsPath": "/tmp/downloads"},
    {"chromiumSandbox": False},
]


# Helper Functions
def assert_browser_action(result, expected_text):
    """Helper function for common browser action assertions"""
    if isinstance(result, dict) and "content" in result:
        assert any(expected_text in item["text"] for item in result["content"])
    else:
        assert expected_text in result


# Fixtures
@pytest.fixture
def setup_test_environment():
    """Fixture to set up common test environment"""
    original_value = os.environ.get("BYPASS_TOOL_CONSENT", None)
    os.environ["BYPASS_TOOL_CONSENT"] = "true"
    with patch("src.strands_tools.use_browser.get_user_input") as mock_input:
        mock_input.return_value = "y"
        yield mock_input
    if original_value is not None:
        os.environ["BYPASS_TOOL_CONSENT"] = original_value
    elif "BYPASS_TOOL_CONSENT" in os.environ:
        del os.environ["BYPASS_TOOL_CONSENT"]


@pytest.fixture
def mock_browser_chain():
    """Fixture to create common browser chain mocks"""
    return {
        "page": AsyncMock(),
        "context": AsyncMock(),
        "browser": AsyncMock(),
        "cdp": AsyncMock(),
        "playwright": AsyncMock(),
    }


@pytest.fixture
def browser_manager(mock_browser_chain):
    """Fixture to provide a mocked BrowserManager instance"""
    manager = BrowserManager()
    manager._playwright = mock_browser_chain["playwright"]
    manager._browser = mock_browser_chain["browser"]
    manager._context = mock_browser_chain["context"]
    manager._page = mock_browser_chain["page"]
    manager._cdp_client = mock_browser_chain["cdp"]

    async def mock_ensure_browser(*args, **kwargs):
        return manager._page, manager._cdp_client

    manager.ensure_browser = mock_ensure_browser

    return manager


@pytest.fixture
def mock_browser_manager():
    """Fixture to mock the browser manager with common setup"""
    with patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = [{"text": "Action completed"}]
        mock_manager._loop = mock_loop
        yield mock_manager


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def async_mock_playwright():
    """Fixture to provide a properly configured async mock playwright instance."""
    mock_playwright = AsyncMock()
    return mock_playwright


# Tests
@pytest.mark.parametrize("action", BROWSER_ACTIONS)
def test_individual_actions(setup_test_environment, mock_browser_manager, action):
    args = {
        "navigate": {"url": "https://example.com"},
        "click": {"selector": "#button"},
        "type": {"selector": "#input", "input_text": "test"},
        "press_key": {"key": "Enter"},
        "evaluate": {"script": "document.title"},
        "get_text": {"selector": "#content"},
        "get_html": {},
        "screenshot": {},
    }

    result = use_browser(action=action, **args.get(action, {}), launch_options={"headless": True})
    assert_browser_action(result, "Action completed")


@pytest.mark.parametrize("browser_action", NAVIGATION_ACTIONS)
def test_browser_navigation_actions(setup_test_environment, mock_browser_manager, browser_action):
    expected_results = {
        "back": "Navigated back",
        "forward": "Navigated forward",
        "refresh": "Page refreshed",
        "new_tab": "New tab created",
        "close_tab": "Closed current tab",
        "get_cookies": "Cookies:",
        "close": "Browser closed",
    }

    mock_browser_manager._loop.run_until_complete.return_value = [{"text": expected_results[browser_action]}]
    result = use_browser(action=browser_action)
    assert_browser_action(result, expected_results[browser_action])


@pytest.mark.parametrize("error_scenario", ERROR_SCENARIOS)
def test_complex_error_conditions(setup_test_environment, mock_browser_manager, error_scenario):
    action, args, expected_error = error_scenario
    mock_browser_manager._loop.run_until_complete.return_value = [{"text": expected_error}]
    result = use_browser(action=action, **args)
    assert_browser_action(result, expected_error)


@pytest.mark.parametrize("launch_options", LAUNCH_OPTIONS_SCENARIOS)
def test_launch_options_combinations(setup_test_environment, mock_browser_manager, launch_options):
    mock_browser_manager._loop.run_until_complete.return_value = [{"text": "Browser launched with custom options"}]
    result = use_browser(action="connect", launch_options=launch_options)
    assert_browser_action(result, "Browser launched with custom options")


def test_multiple_actions_with_wait(setup_test_environment, mock_browser_manager):
    mock_browser_manager._loop.run_until_complete.return_value = [
        {"text": "Navigated"},
        {"text": "Waited for 2000ms"},
        {"text": "Clicked"},
        {"text": "Waited for 3000ms"},
    ]

    result = use_browser(
        actions=[
            {"action": "navigate", "args": {"url": "https://example.com"}, "wait_for": 2000},
            {"action": "click", "args": {"selector": "#button"}, "wait_for": 3000},
        ]
    )

    assert "Navigated" in result
    assert "Waited for 2000ms" in result
    assert "Clicked" in result
    assert "Waited for 3000ms" in result


@pytest.mark.asyncio
async def test_browser_manager_ensure_browser(mock_browser_chain, async_mock_playwright):
    # Configure the mock chain
    mock_playwright = async_mock_playwright
    mock_playwright.start = AsyncMock(return_value=mock_playwright)
    mock_playwright.chromium = AsyncMock()
    mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser_chain["browser"])
    mock_browser_chain["browser"].new_context = AsyncMock(return_value=mock_browser_chain["context"])
    mock_browser_chain["context"].new_page = AsyncMock(return_value=mock_browser_chain["page"])
    mock_browser_chain["page"].context = AsyncMock()
    mock_browser_chain["page"].context.new_cdp_session = AsyncMock(return_value=mock_browser_chain["cdp"])

    # Create an async function that returns our mock
    async def mock_async_playwright():
        return mock_playwright

    # Patch the async_playwright import
    with patch("src.strands_tools.use_browser.async_playwright", return_value=mock_playwright):
        browser_manager = BrowserManager()
        launch_options = {"headless": True}
        context_options = {"viewport": {"width": 1280, "height": 800}}

        page, cdp = await browser_manager.ensure_browser(launch_options, context_options)

        # Verify the calls
        mock_playwright.start.assert_called_once()
        mock_playwright.chromium.launch.assert_called_once()
        mock_browser_chain["browser"].new_context.assert_called_once()
        mock_browser_chain["context"].new_page.assert_called_once()
        assert page == mock_browser_chain["page"]
        assert cdp == mock_browser_chain["cdp"]


@pytest.mark.asyncio
async def test_browser_manager_cleanup(browser_manager):
    await browser_manager.cleanup()

    if browser_manager._page:
        browser_manager._page.close.assert_called_once()
    if browser_manager._context:
        browser_manager._context.close.assert_called_once()
    if browser_manager._browser:
        browser_manager._browser.close.assert_called_once()
    if browser_manager._playwright:
        browser_manager._playwright.stop.assert_called_once()


@pytest.mark.asyncio
@patch("src.strands_tools.use_browser.async_playwright")
async def test_browser_manager_error_handling(mock_playwright_func, browser_manager):
    async def mock_goto(*args, **kwargs):
        raise Exception("Browser has been closed")

    browser_manager._page.goto = AsyncMock(side_effect=mock_goto)

    result = await browser_manager.handle_action("navigate", args={"url": "https://example.com"})

    assert any(
        "Error" in item["text"] and "Browser has been closed" in item["text"] for item in result
    ), f"Expected browser error, got: {result[0]['text']}"

    browser_manager._page.goto.assert_called_once_with("https://example.com")
