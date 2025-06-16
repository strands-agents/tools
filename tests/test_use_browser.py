import asyncio
import json
import os
import types
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import pytest_asyncio

from src.strands_tools.use_browser import BrowserManager, use_browser, validate_required_param

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
    manager = BrowserManager()
    manager._playwright = mock_browser_chain["playwright"]
    manager._browser = mock_browser_chain["browser"]
    manager._context = mock_browser_chain["context"]
    manager._page = mock_browser_chain["page"]
    manager._cdp_client = mock_browser_chain["cdp"]

    async def mock_ensure_browser(*args, **kwargs):
        return manager._page, manager._cdp_client

    manager.ensure_browser = mock_ensure_browser

    manager._page.goto = AsyncMock(return_value=None)
    manager._page.click = AsyncMock(return_value=None)
    manager._page.fill = AsyncMock(return_value=None)
    manager._page.keyboard.press = AsyncMock(return_value=None)
    manager._page.evaluate = AsyncMock(return_value="Test Title")
    manager._page.text_content = AsyncMock(return_value="Test Content")
    manager._page.content = AsyncMock(return_value="<html>")
    manager._page.reload = AsyncMock(return_value=None)
    manager._page.go_back = AsyncMock(return_value=None)
    manager._page.go_forward = AsyncMock(return_value=None)
    manager._page.screenshot = AsyncMock(return_value=None)

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


# Tests for helper functions


def test_validate_required_param():
    assert validate_required_param(None, "test_param", "test_action") == [
        {"text": "Error: test_param required for test_action"}
    ]
    assert validate_required_param("value", "test_param", "test_action") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action, args, expected_error",
    [
        ("navigate", {}, "Error: url required for navigate"),
        ("click", {}, "Error: selector required for click"),
        ("type", {"selector": "#input"}, "Error: text required for type"),
        ("type", {}, "Error: selector required for type"),
        ("press_key", {}, "Error: key required for press_key"),
        ("evaluate", {}, "Error: script required for evaluate"),
        ("get_text", {}, "Error: selector required for get_text"),
        ("execute_cdp", {}, "Error: method required for execute_cdp"),
    ],
)
async def test_handle_action_errors(browser_manager, action, args, expected_error):
    result = await browser_manager.handle_action(action, args=args)
    assert result[0]["text"] == expected_error


# Test BYPASS_TOOL_CONSENT environment variable functions correctly
def test_use_browser_with_bypass_consent():
    """Test use_browser with bypassed consent"""
    with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
        with patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
            mock_manager._loop = MagicMock()
            mock_manager._loop.run_until_complete.return_value = [{"text": "Success"}]
            result = use_browser(action="test")
            assert "Success" in result


def test_use_browser_without_bypass_consent():
    """Test use_browser without bypassed consent"""
    with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "false"}):
        with patch("src.strands_tools.use_browser.get_user_input") as mock_input:
            mock_input.return_value = "n"
            result = use_browser(action="test")
            assert isinstance(result, dict)
            assert "error" in result["status"]


def test_use_browser_with_invalid_action():
    """Test use_browser with invalid action"""
    with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
        with patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
            mock_manager._loop = MagicMock()
            mock_manager._loop.run_until_complete.side_effect = Exception("Invalid action")

            with pytest.raises(Exception) as excinfo:
                use_browser(action="invalid")

            assert str(excinfo.value) == "Invalid action"


# Browser setup tests
@pytest.mark.asyncio
async def test_browser_manager_initialization():
    """Test BrowserManager initialization"""
    browser_manager = BrowserManager()
    assert browser_manager._playwright is None
    assert browser_manager._browser is None
    assert browser_manager._context is None
    assert browser_manager._page is None
    assert browser_manager._cdp_client is None
    assert browser_manager._user_data_dir is None
    assert browser_manager._profile_name is None
    assert isinstance(browser_manager._loop, asyncio.AbstractEventLoop)


@pytest.mark.parametrize("launch_options", LAUNCH_OPTIONS_SCENARIOS)
def test_launch_options_combinations(setup_test_environment, mock_browser_manager, launch_options):
    mock_browser_manager._loop.run_until_complete.return_value = [{"text": "Browser launched with custom options"}]
    result = use_browser(action="connect", launch_options=launch_options)
    assert_browser_action(result, "Browser launched with custom options")


@pytest.mark.asyncio
async def test_browser_manager_ensure_browser(mock_browser_chain, async_mock_playwright):
    mock_playwright = async_mock_playwright
    mock_playwright.start = AsyncMock(return_value=mock_playwright)
    mock_playwright.chromium = AsyncMock()
    mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser_chain["browser"])
    mock_browser_chain["browser"].new_context = AsyncMock(return_value=mock_browser_chain["context"])
    mock_browser_chain["context"].new_page = AsyncMock(return_value=mock_browser_chain["page"])
    mock_browser_chain["page"].context = AsyncMock()
    mock_browser_chain["page"].context.new_cdp_session = AsyncMock(return_value=mock_browser_chain["cdp"])

    async def mock_async_playwright():
        return mock_playwright

    with patch("src.strands_tools.use_browser.async_playwright", return_value=mock_playwright):
        browser_manager = BrowserManager()
        launch_options = {"headless": True}
        context_options = {"viewport": {"width": 1280, "height": 800}}

        page, cdp = await browser_manager.ensure_browser(launch_options, context_options)

        mock_playwright.start.assert_called_once()
        mock_playwright.chromium.launch.assert_called_once()
        mock_browser_chain["browser"].new_context.assert_called_once()
        mock_browser_chain["context"].new_page.assert_called_once()
        assert page == mock_browser_chain["page"]
        assert cdp == mock_browser_chain["cdp"]


@pytest.mark.asyncio
async def test_persistent_context_creation():
    """Test creation of persistent context with mocked responses"""
    with patch("src.strands_tools.use_browser.async_playwright") as mock_playwright_init:
        mock_playwright = AsyncMock()
        mock_chromium = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_cdp = AsyncMock()

        mock_playwright_init.return_value = mock_playwright
        mock_playwright.start = AsyncMock(return_value=mock_playwright)
        mock_playwright.chromium = mock_chromium
        mock_chromium.launch_persistent_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_page.context = mock_context
        mock_context.new_cdp_session = AsyncMock(return_value=mock_cdp)

        browser_manager = BrowserManager()
        launch_options = {"persistent_context": True, "user_data_dir": "/tmp/test_profile", "headless": True}

        page, cdp = await browser_manager.ensure_browser(launch_options)

        # Verify the calls
        mock_chromium.launch_persistent_context.assert_called_once()
        mock_context.new_page.assert_called_once()
        assert page == mock_page
        assert cdp == mock_cdp
        assert browser_manager._browser is None


@pytest.mark.asyncio
async def test_browser_manager_loop_setup():
    """Test event loop setup in BrowserManager"""
    with patch("asyncio.new_event_loop") as mock_new_loop:
        with patch("asyncio.set_event_loop") as mock_set_loop:
            mock_loop = AsyncMock()
            mock_new_loop.return_value = mock_loop

            browser_manager = BrowserManager()

            mock_new_loop.assert_called_once()
            mock_set_loop.assert_called_once_with(mock_loop)
            assert browser_manager._loop == mock_loop


# Tests for calling use_browser with multiple actions


def test_use_browser_with_multiple_actions():
    """Test use_browser with multiple actions"""
    with patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
        mock_manager._loop = MagicMock()
        mock_manager.handle_action = AsyncMock()

        mock_manager.handle_action.side_effect = [
            [{"text": "Navigated to https://example.com"}],
            [{"text": "Clicked #button"}],
            [{"text": "Typed 'Hello, World!' into #input"}],
        ]

        mock_manager._loop.run_until_complete.return_value = [
            {"text": "Navigated to https://example.com"},
            {"text": "Clicked #button"},
            {"text": "Typed 'Hello, World!' into #input"},
        ]

        actions = [
            {"action": "navigate", "args": {"url": "https://example.com"}, "wait_for": 2000},
            {"action": "click", "args": {"selector": "#button"}, "wait_for": 1000},
            {"action": "type", "args": {"selector": "#input", "text": "Hello, World!"}},
        ]

        with patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"}):
            result = use_browser(actions=actions)

        assert mock_manager._loop.run_until_complete.call_count == 1

        call = mock_manager._loop.run_until_complete.call_args
        assert isinstance(call[0][0], types.CoroutineType)

        expected_result = "Navigated to https://example.com\n" "Clicked #button\n" "Typed 'Hello, World!' into #input"
        assert result == expected_result

        with patch("src.strands_tools.use_browser.logger") as mock_logger:
            use_browser(actions=actions)
            mock_logger.info.assert_any_call("Multiple actions requested: ['navigate', 'click', 'type']")


@pytest.mark.asyncio
async def test_use_browser_with_multiple_actions_approval():
    """Test use_browser with multiple actions and user approval"""
    with patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "false"}):
        with patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
            mock_manager._loop = MagicMock()

            mock_manager._loop.run_until_complete.return_value = [
                {"text": "Navigated to https://example.com"},
                {"text": "Clicked #button"},
                {"text": "Typed 'Hello, World!' into #input"},
            ]

            actions = [
                {"action": "navigate", "args": {"url": "https://example.com"}},
                {"action": "click", "args": {"selector": "#button"}},
                {"action": "type", "args": {"selector": "#input", "text": "Hello, World!"}},
            ]

            with patch("src.strands_tools.use_browser.console") as mock_console:
                with patch("src.strands_tools.use_browser.get_user_input") as mock_input:
                    with patch("src.strands_tools.use_browser.Panel") as mock_panel:
                        mock_input.return_value = "y"

                        result = use_browser(actions=actions)

                        mock_panel.assert_called_once()
                        panel_args = mock_panel.call_args[0][0]

                        assert "User requested multiple actions:" in str(panel_args)
                        assert "navigate" in str(panel_args)
                        assert "click" in str(panel_args)
                        assert "type" in str(panel_args)

                        assert mock_console.print.call_count == 1

                        mock_input.assert_called_once_with("Do you want to proceed with multiple actions? (y/n)")

                        expected_result = (
                            "Navigated to https://example.com\n" "Clicked #button\n" "Typed 'Hello, World!' into #input"
                        )
                        assert result == expected_result

                        assert mock_manager._loop.run_until_complete.call_count == 1


@pytest.mark.asyncio
async def test_run_all_actions_coroutine():
    """Test that run_all_actions coroutine is created and executed correctly"""
    with patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
        mock_manager._loop = MagicMock()
        mock_manager.handle_action = AsyncMock()

        mock_manager.handle_action.side_effect = [
            [{"text": "Navigated to https://example.com"}],
            [{"text": "Clicked #button"}],
            [{"text": "Typed 'Hello, World!' into #input"}],
        ]
        combined_results = [
            {"text": "Navigated to https://example.com"},
            {"text": "Clicked #button"},
            {"text": "Typed 'Hello, World!' into #input"},
        ]

        mock_manager._loop.run_until_complete = MagicMock(return_value=combined_results)
        actions = [
            {"action": "navigate", "args": {"url": "https://example.com"}, "wait_for": 2000},
            {"action": "click", "args": {"selector": "#button"}, "wait_for": 1000},
            {"action": "type", "args": {"selector": "#input", "text": "Hello, World!"}},
        ]

        launch_options = {"headless": True}
        default_wait_time = 1

        with patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"}):
            result = use_browser(actions=actions, launch_options=launch_options)

            run_all_actions_coroutine = mock_manager._loop.run_until_complete.call_args[0][0]

            assert asyncio.iscoroutine(run_all_actions_coroutine)

            expected_calls = [
                call(
                    action="navigate",
                    args={"url": "https://example.com", "launchOptions": launch_options},
                    selector=None,
                    wait_for=2000,
                ),
                call(
                    action="click",
                    args={"selector": "#button", "launchOptions": launch_options},
                    selector=None,
                    wait_for=1000,
                ),
                call(
                    action="type",
                    args={"selector": "#input", "text": "Hello, World!", "launchOptions": launch_options},
                    selector=None,
                    wait_for=default_wait_time * 1000,
                ),
            ]

            await run_all_actions_coroutine

            assert mock_manager.handle_action.call_args_list == expected_calls

            expected_result = (
                "Navigated to https://example.com\n" "Clicked #button\n" "Typed 'Hello, World!' into #input"
            )
            assert result == expected_result


# Tests covering if statements in use_browser main function (lines ~ 510-525)


@pytest.mark.asyncio
async def test_use_browser_single_action_url(setup_test_environment):
    with patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
        mock_manager._loop = MagicMock()
        mock_manager.handle_action = AsyncMock(return_value=[{"text": "Navigated to https://example.com"}])
        mock_manager._loop.run_until_complete.return_value = [{"text": "Navigated to https://example.com"}]

        result = use_browser(action="navigate", url="https://example.com")

        mock_manager._loop.run_until_complete.assert_called_once()
        assert result == "Navigated to https://example.com"


@pytest.mark.asyncio
async def test_use_browser_single_action_input_text(setup_test_environment):
    with patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
        mock_manager._loop = MagicMock()
        mock_manager.handle_action = AsyncMock(return_value=[{"text": "Typed 'Hello World' into #input"}])
        mock_manager._loop.run_until_complete.return_value = [{"text": "Typed 'Hello World' into #input"}]

        result = use_browser(action="type", selector="#input", input_text="Hello World")

        mock_manager._loop.run_until_complete.assert_called_once()
        assert result == "Typed 'Hello World' into #input"


@pytest.mark.asyncio
async def test_use_browser_single_action_script(setup_test_environment):
    """Test use_browser with script evaluation"""
    with patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
        # Set up mock responses
        mock_manager._loop = MagicMock()
        mock_manager.handle_action = AsyncMock()
        mock_manager.cleanup = AsyncMock()

        async def mock_handle_action(**kwargs):
            return [{"text": "Evaluated: 42"}]

        mock_manager.handle_action.side_effect = mock_handle_action
        mock_manager._loop.run_until_complete = lambda x: asyncio.get_event_loop().run_until_complete(x)

        result = use_browser(action="evaluate", script="return 6 * 7;")

        assert mock_manager.handle_action.call_count == 1
        assert result == "Evaluated: 42"


@pytest.mark.asyncio
async def test_use_browser_single_action_cdp_method(setup_test_environment):
    """Test use_browser with CDP method execution"""
    with patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
        # Set up mock responses
        mock_manager._loop = MagicMock()
        mock_manager.handle_action = AsyncMock()
        mock_manager.cleanup = AsyncMock()

        async def mock_handle_action(**kwargs):
            return [{"text": "CDP command executed"}]

        mock_manager.handle_action.side_effect = mock_handle_action
        mock_manager._loop.run_until_complete = lambda x: asyncio.get_event_loop().run_until_complete(x)

        result = use_browser(
            action="execute_cdp", cdp_method="Network.enable", cdp_params={"maxTotalBufferSize": 10000000}
        )

        assert mock_manager.handle_action.call_count == 1
        call_args = mock_manager.handle_action.call_args[1]
        assert call_args["action"] == "execute_cdp"
        assert call_args["args"]["method"] == "Network.enable"
        assert result == "CDP command executed"


@pytest.mark.asyncio
async def test_use_browser_single_action_key(setup_test_environment):
    """Test use_browser with key press"""
    with patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
        # Set up mock responses
        mock_manager._loop = MagicMock()
        mock_manager.handle_action = AsyncMock()
        mock_manager.cleanup = AsyncMock()

        async def mock_handle_action(**kwargs):
            return [{"text": "Pressed key: Enter"}]

        mock_manager.handle_action.side_effect = mock_handle_action
        mock_manager._loop.run_until_complete = lambda x: asyncio.get_event_loop().run_until_complete(x)

        result = use_browser(action="press_key", key="Enter")

        assert mock_manager.handle_action.call_count == 1
        assert result == "Pressed key: Enter"


# Tests covering when specific if statements are false (throughout the whole tool)


@pytest.mark.asyncio
async def test_ensure_browser_with_existing_playwright():
    """Test ensure_browser when playwright is already initialized"""
    with patch("src.strands_tools.use_browser.async_playwright") as mock_playwright_func:
        mock_playwright = AsyncMock()
        mock_page = AsyncMock()
        mock_cdp = AsyncMock()

        browser_manager = BrowserManager()
        browser_manager._playwright = mock_playwright
        browser_manager._page = mock_page
        browser_manager._cdp_client = mock_cdp

        returned_page, returned_cdp = await browser_manager.ensure_browser()

        mock_playwright_func.assert_not_called()

        assert returned_page == mock_page
        assert returned_cdp == mock_cdp


@pytest.mark.asyncio
async def test_ensure_browser_fresh_start_no_options():
    """Test ensure_browser with no existing playwright and no launch options"""
    with patch("src.strands_tools.use_browser.async_playwright") as mock_playwright_func:
        mock_playwright = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_cdp = AsyncMock()

        mock_playwright_func.return_value.start = AsyncMock(return_value=mock_playwright)
        mock_playwright.chromium = AsyncMock()
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_page.context = mock_context
        mock_context.new_cdp_session = AsyncMock(return_value=mock_cdp)

        browser_manager = BrowserManager()
        returned_page, returned_cdp = await browser_manager.ensure_browser()

        mock_playwright_func.assert_called_once()
        mock_playwright.chromium.launch.assert_called_once_with(headless=False, args=["--window-size=1280,800"])

        mock_browser.new_context.assert_called_once_with(viewport={"width": 1280, "height": 800})

        mock_context.new_page.assert_called_once()
        mock_context.new_cdp_session.assert_called_once_with(mock_page)

        assert returned_page == mock_page
        assert returned_cdp == mock_cdp

        assert browser_manager._playwright == mock_playwright
        assert browser_manager._browser == mock_browser
        assert browser_manager._context == mock_context
        assert browser_manager._page == mock_page
        assert browser_manager._cdp_client == mock_cdp


@pytest.mark.asyncio
async def test_use_browser_exception_handling(setup_test_environment):
    with patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
        mock_manager._loop = MagicMock()
        mock_manager.handle_action = AsyncMock(side_effect=Exception("Test exception"))
        mock_manager.cleanup = AsyncMock()

        first_call = True

        def mock_run_until_complete(coro):
            nonlocal first_call
            if first_call:
                first_call = False
                raise Exception("Test exception")
            return None

        mock_manager._loop.run_until_complete = MagicMock(side_effect=mock_run_until_complete)

        with patch("src.strands_tools.use_browser.logger") as mock_logger:
            result = use_browser(action="test_action")

        mock_logger.error.assert_called_once_with("Error in use_browser: Test exception")

        mock_logger.info.assert_called_with(
            "Cleaning up browser due to explicit request or error with non-persistent session"
        )
        assert mock_manager._loop.run_until_complete.call_count == 2
        assert result == "Error: Test exception"


@pytest.mark.asyncio
async def test_use_browser_cdp_method_without_params(setup_test_environment):
    """Test use_browser with CDP method but no params"""
    with patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
        mock_manager._loop = MagicMock()
        mock_manager.handle_action = AsyncMock()
        mock_manager.cleanup = AsyncMock()

        async def mock_handle_action(**kwargs):
            return [{"text": "CDP command executed"}]

        mock_manager.handle_action.side_effect = mock_handle_action
        mock_manager._loop.run_until_complete = lambda x: asyncio.get_event_loop().run_until_complete(x)

        result = use_browser(action="execute_cdp", cdp_method="Network.enable")

        assert mock_manager.handle_action.call_count == 1
        call_args = mock_manager.handle_action.call_args[1]
        assert call_args["action"] == "execute_cdp"
        assert call_args["args"] == {"method": "Network.enable"}
        assert call_args["wait_for"] == 1000
        assert result == "CDP command executed"


# Tests for handle_action function
@pytest.mark.asyncio
async def test_handle_connect_action(browser_manager):
    result = await browser_manager.handle_action(action="connect")
    assert "Successfully connected to browser" in result[0]["text"]


@pytest.mark.asyncio
async def test_all_browser_actions(browser_manager):
    """Test all browser actions with mocked responses"""
    mock_cookies = [{"name": "test_cookie", "value": "test_value"}]

    with patch.object(BrowserManager, "_handle_get_cookies_action", new_callable=AsyncMock) as mock_get_cookies:
        mock_get_cookies.return_value = [{"text": f"Cookies: {json.dumps(mock_cookies, indent=2)}"}]

        with patch.object(BrowserManager, "_handle_set_cookies_action", new_callable=AsyncMock) as mock_set_cookies:
            mock_set_cookies.return_value = [{"text": "Cookies set successfully"}]

            test_cases = [
                {
                    "action": "navigate",
                    "args": {"url": "https://example.com"},
                    "expected": "Navigated to https://example.com",
                },
                {"action": "click", "args": {"selector": "#button"}, "expected": "Clicked #button"},
                {
                    "action": "type",
                    "args": {"selector": "#input", "text": "test text"},
                    "expected": "Typed 'test text' into #input",
                },
                {"action": "press_key", "args": {"key": "Enter"}, "expected": "Pressed key: Enter"},
                {"action": "evaluate", "args": {"script": "document.title"}, "expected": "Evaluated: Test Title"},
                {"action": "get_text", "args": {"selector": "#content"}, "expected": "Text content: Test Content"},
                {"action": "get_html", "args": {}, "expected": "HTML content: <html>..."},
                {"action": "refresh", "args": {}, "expected": "Page refreshed"},
                {"action": "back", "args": {}, "expected": "Navigated back"},
                {"action": "forward", "args": {}, "expected": "Navigated forward"},
                {"action": "screenshot", "args": {"path": "test.png"}, "expected": "Screenshot saved as test.png"},
                {"action": "get_cookies", "args": {}, "expected": f"Cookies: {json.dumps(mock_cookies, indent=2)}"},
                {
                    "action": "set_cookies",
                    "args": {"cookies": [{"name": "new_cookie", "value": "new_value"}]},
                    "expected": "Cookies set successfully",
                },
                {
                    "action": "network_intercept",
                    "args": {"pattern": "*.js", "handler": "log"},
                    "expected": "Network interception set for *.js",
                },
                {"action": "close", "args": {}, "expected": "Browser closed"},
            ]

            for test_case in test_cases:
                action = test_case["action"]
                args = test_case["args"]
                expected = test_case["expected"]

                result = await browser_manager.handle_action(action, args=args)
                assert result[0]["text"] == expected, f"Failed on action: {action}"

                if action == "set_cookies":
                    mock_set_cookies.assert_called_with(args)
                elif action == "network_intercept":
                    browser_manager._page.route.assert_called_once()

            mock_get_cookies.assert_called_once()
            mock_set_cookies.assert_called_once()


@pytest.mark.asyncio
async def test_cookie_management(browser_manager):
    """Test cookie management with mocked responses"""
    mock_cookies = [{"name": "test", "value": "123"}]
    browser_manager._context.cookies = AsyncMock(return_value=mock_cookies)

    result = await browser_manager._handle_get_cookies_action()
    assert "Cookies:" in result[0]["text"]
    assert "test" in result[0]["text"]

    test_cookies = [{"name": "test2", "value": "456"}]
    result = await browser_manager._handle_set_cookies_action({"cookies": test_cookies})
    assert "Cookies set successfully" in result[0]["text"]
    browser_manager._context.add_cookies.assert_called_once_with(test_cookies)


@pytest.mark.asyncio
async def test_network_interception(browser_manager):
    """Test network interception with mocked responses"""
    browser_manager._page.route = AsyncMock()

    result = await browser_manager._handle_network_intercept_action(
        browser_manager._page, {"pattern": "*.js", "handler": "log"}
    )

    browser_manager._page.route.assert_called_once()
    assert "Network interception set for *.js" in result[0]["text"]


@pytest.mark.asyncio
async def test_network_intercept_with_custom_handler(browser_manager):
    """Test network interception with custom handler"""

    async def custom_handler(route):
        await route.continue_()

    result = await browser_manager._handle_network_intercept_action(
        browser_manager._page, {"pattern": "*.js", "handler": custom_handler}
    )
    assert "Network interception set" in result[0]["text"]


@pytest.mark.asyncio
async def test_cdp_commands(browser_manager):
    """Test CDP command execution with mocked responses"""
    mock_response = {"result": "success"}
    browser_manager._cdp_client.send = AsyncMock(return_value=mock_response)

    result = await browser_manager._handle_execute_cdp_action(
        browser_manager._cdp_client, {"method": "Test.method", "params": {"param1": "value1"}}
    )

    browser_manager._cdp_client.send.assert_called_once_with("Test.method", {"param1": "value1"})
    assert "CDP Test.method result:" in result[0]["text"]


@pytest.mark.asyncio
async def test_new_tab_and_close_tab_sequence(browser_manager):
    """Test creating a new tab and then closing it"""
    mock_new_page = AsyncMock()
    mock_new_cdp = AsyncMock()
    browser_manager._context.new_page = AsyncMock(return_value=mock_new_page)
    mock_new_page.context = AsyncMock()
    mock_new_page.context.new_cdp_session = AsyncMock(return_value=mock_new_cdp)

    result_new = await browser_manager.handle_action(action="new_tab")
    assert result_new[0]["text"] == "New tab created"
    assert browser_manager._page == mock_new_page

    mock_original_page = AsyncMock()
    browser_manager._context.pages = [mock_original_page]
    mock_original_page.context = AsyncMock()
    mock_original_cdp = AsyncMock()
    mock_original_page.context.new_cdp_session = AsyncMock(return_value=mock_original_cdp)

    result_close = await browser_manager.handle_action(action="close_tab")
    assert "Closed current tab" in result_close[0]["text"]
    mock_new_page.close.assert_called_once()
    assert browser_manager._page == mock_original_page


@pytest.mark.asyncio
async def test_handle_close_tab_action_last_tab(browser_manager):
    """Test closing the last remaining tab"""
    browser_manager._page.close = AsyncMock()

    browser_manager._context.pages = []

    mock_new_cdp_session = browser_manager._page.context.new_cdp_session

    result = await browser_manager._handle_close_tab_action()

    browser_manager._page.close.assert_called_once()

    assert result == [{"text": "Closed the last tab. Browser may close."}]

    mock_new_cdp_session.assert_not_called()


@pytest.mark.asyncio
async def test_handle_action_with_wait_for(browser_manager):
    mock_page = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()

    browser_manager.ensure_browser = AsyncMock(return_value=(mock_page, AsyncMock()))

    browser_manager._handle_navigate_action = AsyncMock(return_value=[{"text": "Navigated successfully"}])

    result = await browser_manager.handle_action("navigate", args={"url": "https://example.com"}, wait_for=1000)

    assert result == [{"text": "Navigated successfully"}]

    mock_page.wait_for_timeout.assert_called_once_with(1000)

    browser_manager._handle_navigate_action.assert_called_once()


@pytest.mark.asyncio
async def test_handle_connect_action_with_launch_options(browser_manager):
    launch_options = {"headless": True, "slowMo": 100, "args": ["--no-sandbox", "--disable-setuid-sandbox"]}

    browser_manager.cleanup = AsyncMock()
    browser_manager.ensure_browser = AsyncMock(return_value=(AsyncMock(), AsyncMock()))

    result = await browser_manager._handle_connect_action(launch_options)

    browser_manager.cleanup.assert_called_once()

    browser_manager.ensure_browser.assert_called_once_with(launch_options=launch_options)

    assert len(result) == 2
    assert result[0] == {"text": "Successfully connected to browser"}
    assert "Launched browser with options:" in result[1]["text"]

    launched_options = json.loads(result[1]["text"].split(": ", 1)[1])
    assert launched_options == launch_options


# Testing errors


@pytest.mark.asyncio
async def test_error_handling_scenarios(browser_manager):
    """Test various error handling scenarios"""
    browser_manager._page.goto = AsyncMock(side_effect=Exception("browser has been closed"))
    result = await browser_manager.handle_action("navigate", args={"url": "https://example.com"})
    assert "Error: browser has been closed" in result[0]["text"]

    result = await browser_manager.handle_action("click", args={})
    assert "Error: selector required for click" in result[0]["text"]


@pytest.mark.asyncio
async def test_cleanup_error_handling(browser_manager):
    """Test cleanup error handling"""
    page_mock = browser_manager._page
    context_mock = browser_manager._context
    browser_mock = browser_manager._browser
    playwright_mock = browser_manager._playwright

    page_mock.close = AsyncMock(side_effect=Exception("Page close error"))
    context_mock.close = AsyncMock(side_effect=Exception("Context close error"))
    browser_mock.close = AsyncMock(side_effect=Exception("Browser close error"))
    playwright_mock.stop = AsyncMock(side_effect=Exception("Playwright stop error"))

    await browser_manager.cleanup()

    page_mock.close.assert_called_once()
    context_mock.close.assert_called_once()
    browser_mock.close.assert_called_once()
    playwright_mock.stop.assert_called_once()

    assert browser_manager._page is None
    assert browser_manager._context is None
    assert browser_manager._browser is None
    assert browser_manager._playwright is None
    assert browser_manager._cdp_client is None


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


@pytest.mark.parametrize("error_scenario", ERROR_SCENARIOS)
def test_complex_error_conditions(setup_test_environment, mock_browser_manager, error_scenario):
    action, args, expected_error = error_scenario
    mock_browser_manager._loop.run_until_complete.return_value = [{"text": expected_error}]
    result = use_browser(action=action, **args)
    assert_browser_action(result, expected_error)


@pytest.mark.asyncio
async def test_handle_action_unknown_action(browser_manager):
    """Test handling of unknown actions"""
    result = await browser_manager.handle_action("unknown_action")
    assert "Error: Unknown action" in result[0]["text"]


@pytest.mark.asyncio
async def test_handle_action_cdp_failure(browser_manager):
    """Test CDP command failure handling"""
    browser_manager._cdp_client.send = AsyncMock(side_effect=Exception("CDP command failed"))
    result = await browser_manager.handle_action("unknown_action", args={"method": "test"})
    assert "Error: Unknown action or CDP command failed" in result[0]["text"]


@pytest.mark.asyncio
async def test_cdp_command_execution_error(browser_manager):
    """Test CDP command execution with error"""
    browser_manager._cdp_client.send = AsyncMock(side_effect=Exception("CDP Error"))

    with pytest.raises(Exception) as excinfo:
        await browser_manager._handle_execute_cdp_action(browser_manager._cdp_client, {"method": "invalid.method"})

    assert str(excinfo.value) == "CDP Error"


@pytest.mark.asyncio
async def test_browser_connection_error():
    """Test browser connection error handling"""
    with patch("src.strands_tools.use_browser.async_playwright") as mock_playwright_factory:
        mock_playwright = AsyncMock()
        mock_playwright.start.side_effect = Exception("Connection failed")

        mock_playwright_factory.return_value = mock_playwright

        browser_manager = BrowserManager()

        with pytest.raises(Exception) as exc_info:
            await browser_manager.ensure_browser()

        assert "Connection failed" in str(exc_info.value)
        mock_playwright.start.assert_called_once()

        assert browser_manager._playwright is None
        assert browser_manager._browser is None
        assert browser_manager._context is None
        assert browser_manager._page is None
        assert browser_manager._cdp_client is None


@pytest.mark.asyncio
async def test_persistent_context_without_user_data_dir():
    """Test that ensure_browser raises ValueError when persistent_context is True but user_data_dir is not provided"""
    browser_manager = BrowserManager()

    launch_options = {"persistent_context": True, "headless": True}
    with pytest.raises(ValueError) as exc_info:
        await browser_manager.ensure_browser(launch_options=launch_options)

    assert "user_data_dir is required for persistent context" in str(exc_info.value)


# Cleanup tests


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
async def test_cleanup_with_no_resources():
    """Test cleanup when no browser resources are initialized"""
    browser_manager = BrowserManager()
    browser_manager._page = None
    browser_manager._context = None
    browser_manager._browser = None
    browser_manager._playwright = None
    browser_manager._cdp_client = None

    with patch("src.strands_tools.use_browser.logger") as mock_logger:
        await browser_manager.cleanup()

        mock_logger.info.assert_called_once_with("Cleanup completed successfully")

        mock_logger.warning.assert_not_called()

        assert browser_manager._page is None
        assert browser_manager._context is None
        assert browser_manager._browser is None
        assert browser_manager._playwright is None
        assert browser_manager._cdp_client is None
