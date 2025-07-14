import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, call, patch

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
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Fixture to set up common test environment"""
    mock_env = {}
    with patch.dict(os.environ, mock_env, clear=True):
        mock_env["BYPASS_TOOL_CONSENT"] = "true"
        with patch("src.strands_tools.use_browser.get_user_input") as mock_input:
            mock_input.return_value = "y"
            yield mock_env


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


@pytest.mark.asyncio
async def test_fix_javascript_syntax_edge_cases():
    browser_manager = BrowserManager()

    assert await browser_manager._fix_javascript_syntax("", "any error") is None
    assert await browser_manager._fix_javascript_syntax(None, "error") is None
    assert await browser_manager._fix_javascript_syntax("script", None) is None
    assert await browser_manager._fix_javascript_syntax("script", "") is None


@pytest.mark.asyncio
async def test_fix_javascript_syntax():
    browser_manager = BrowserManager()

    # Test case 1: Illegal return statement
    script = "return 42;"
    error_msg = "Illegal return statement"
    fixed = await browser_manager._fix_javascript_syntax(script, error_msg)
    assert fixed == "(function() { return 42; })()"

    # Test case 2: Unexpected token (template literals)
    script = "console.log(`Hello ${name}!`);"
    error_msg = "Unexpected token '`'"
    fixed = await browser_manager._fix_javascript_syntax(script, error_msg)
    assert fixed == "console.log('Hello ' + name + '!');"

    # Test case 3: Unexpected token (arrow function)
    script = "const add = (a, b) => a + b;"
    error_msg = "Unexpected token '=>'"
    fixed = await browser_manager._fix_javascript_syntax(script, error_msg)
    assert fixed == "const add = (a, b) function() { return  a + b; }"

    # Test case 4: Unexpected end of input (missing closing brace)
    script = "function test() { console.log('Hello')"
    error_msg = "Unexpected end of input"
    fixed = await browser_manager._fix_javascript_syntax(script, error_msg)
    assert fixed == "function test() { console.log('Hello')}"

    # Test case 5: Uncaught reference error
    script = "console.log(undefinedVar);"
    error_msg = "'undefinedVar' is not defined"
    fixed = await browser_manager._fix_javascript_syntax(script, error_msg)
    assert fixed == "var undefinedVar = undefined;\nconsole.log(undefinedVar);"

    # Test case 6: No fix needed
    script = "console.log('Hello, World!');"
    error_msg = "Some other error"
    fixed = await browser_manager._fix_javascript_syntax(script, error_msg)
    assert fixed is None

    # Test case 7: Empty script
    fixed = await browser_manager._fix_javascript_syntax("", "Any error")
    assert fixed is None

    # Test case 8: Empty error message
    fixed = await browser_manager._fix_javascript_syntax("var x = 5;", "")
    assert fixed is None

    # Test case 9: Both script and error message are empty
    fixed = await browser_manager._fix_javascript_syntax("", "")
    assert fixed is None


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
def test_launch_options_combinations(mock_browser_manager, launch_options):
    mock_browser_manager._loop.run_until_complete.return_value = [{"text": "Browser launched with custom options"}]
    result = use_browser(action="connect", launch_options=launch_options)
    assert_browser_action(result, "Browser launched with custom options")


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
async def test_use_browser_single_action_url():
    with patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
        mock_manager._loop = MagicMock()
        mock_manager.handle_action = AsyncMock(return_value=[{"text": "Navigated to https://example.com"}])
        mock_manager._loop.run_until_complete.return_value = [{"text": "Navigated to https://example.com"}]

        result = use_browser(action="navigate", url="https://example.com")

        mock_manager._loop.run_until_complete.assert_called_once()
        assert result == "Navigated to https://example.com"


@pytest.mark.asyncio
async def test_use_browser_single_action_input_text():
    with patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
        mock_manager._loop = MagicMock()
        mock_manager.handle_action = AsyncMock(return_value=[{"text": "Typed 'Hello World' into #input"}])
        mock_manager._loop.run_until_complete.return_value = [{"text": "Typed 'Hello World' into #input"}]

        result = use_browser(action="type", selector="#input", input_text="Hello World")

        mock_manager._loop.run_until_complete.assert_called_once()
        assert result == "Typed 'Hello World' into #input"


# Testing errors


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


@pytest.mark.parametrize("error_scenario", ERROR_SCENARIOS)
def test_complex_error_conditions(mock_browser_manager, error_scenario):
    action, args, expected_error = error_scenario
    mock_browser_manager._loop.run_until_complete.return_value = [{"text": expected_error}]
    result = use_browser(action=action, **args)
    assert_browser_action(result, expected_error)


@pytest.mark.asyncio
async def test_handle_action_unknown_action(browser_manager):
    """Test handling of unknown actions"""
    result = await browser_manager.handle_action("unknown_action")
    assert "Error: Unknown action" in result[0]["text"]


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
