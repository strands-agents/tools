"""
Comprehensive tests for Browser base class to improve coverage.
"""

import asyncio
import json
import os
import signal
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from strands_tools.browser import Browser
from strands_tools.browser.models import (
    BackAction,
    BrowserInput,
    ClickAction,
    CloseAction,
    CloseTabAction,
    EvaluateAction,
    ForwardAction,
    GetCookiesAction,
    GetHtmlAction,
    GetTextAction,
    InitSessionAction,
    ListLocalSessionsAction,
    ListTabsAction,
    NavigateAction,
    NewTabAction,
    PressKeyAction,
    RefreshAction,
    ScreenshotAction,
    SetCookiesAction,
    SwitchTabAction,
    TypeAction,
)


class MockContext:
    """Mock context object."""
    async def cookies(self):
        return [{"name": "test", "value": "cookie"}]
        
    async def add_cookies(self, cookies):
        pass


class MockPage:
    """Mock page object with serializable properties."""
    def __init__(self, url="https://example.com"):
        self.url = url
        self.context = MockContext()
        
    async def goto(self, url):
        self.url = url
        
    async def click(self, selector):
        pass
        
    async def fill(self, selector, text):
        pass
        
    async def press(self, key):
        pass
        
    async def text_content(self, selector):
        return "Mock text content"
        
    async def inner_html(self, selector):
        return "<div>Mock HTML</div>"
        
    async def content(self):
        return "<html><body>Mock page content</body></html>"
        
    async def screenshot(self, path=None):
        pass
        
    async def reload(self):
        pass
        
    async def go_back(self):
        pass
        
    async def go_forward(self):
        pass
        
    async def evaluate(self, script):
        return "Mock evaluation result"
        
    async def wait_for_selector(self, selector):
        pass
        
    async def wait_for_load_state(self, state="load"):
        pass
        
    @property
    def keyboard(self):
        """Mock keyboard object."""
        keyboard_mock = Mock()
        keyboard_mock.press = AsyncMock()
        return keyboard_mock
        
    async def close(self):
        pass


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
        mock_browser = Mock()
        mock_context = AsyncMock()
        mock_page = MockPage()
        
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        
        return mock_browser


@pytest.fixture
def mock_browser():
    """Create a mock browser instance."""
    with patch("strands_tools.browser.browser.async_playwright") as mock_playwright:
        mock_playwright_instance = Mock()
        mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)
        
        browser = MockBrowser()
        yield browser


class TestBrowserInitialization:
    """Test browser initialization and cleanup."""

    def test_browser_initialization(self):
        """Test browser initialization."""
        browser = MockBrowser()
        assert not browser._started
        assert browser._playwright is None
        assert browser._sessions == {}

    def test_browser_start(self, mock_browser):
        """Test browser startup."""
        mock_browser._start()
        assert mock_browser._started
        assert mock_browser._playwright is not None

    def test_browser_cleanup(self, mock_browser):
        """Test browser cleanup."""
        mock_browser._start()
        mock_browser._cleanup()
        assert not mock_browser._started

    def test_browser_destructor(self, mock_browser):
        """Test browser destructor cleanup."""
        mock_browser._start()
        with patch.object(mock_browser, '_cleanup') as mock_cleanup:
            mock_browser.__del__()
            mock_cleanup.assert_called_once()


class TestBrowserActions:
    """Test browser action handling."""

    def test_browser_dict_input(self, mock_browser):
        """Test browser with dict input."""
        browser_input = {
            "action": {
                "type": "list_local_sessions"
            }
        }
        
        result = mock_browser.browser(browser_input)
        assert result["status"] == "success"

    def test_unknown_action_type(self, mock_browser):
        """Test handling of unknown action types."""
        with pytest.raises(ValueError):
            # This should raise a validation error due to invalid action type
            BrowserInput(action={"type": "unknown_action"})


class TestSessionManagement:
    """Test browser session management."""

    def test_init_session_success(self, mock_browser):
        """Test successful session initialization."""
        action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        
        result = mock_browser.init_session(action)
        assert result["status"] == "success"
        assert "test-session-001" in mock_browser._sessions

    def test_init_session_duplicate(self, mock_browser):
        """Test initializing duplicate session."""
        action = InitSessionAction(
            type="init_session",
            session_name="test-session-002",
            description="Test session"
        )
        
        # Initialize first session
        mock_browser.init_session(action)
        
        # Try to initialize duplicate
        result = mock_browser.init_session(action)
        assert result["status"] == "error"
        assert "already exists" in result["content"][0]["text"]

    def test_init_session_error(self, mock_browser):
        """Test session initialization error."""
        action = InitSessionAction(
            type="init_session",
            session_name="test-session-003",
            description="Test session"
        )
        
        with patch.object(mock_browser, 'create_browser_session', side_effect=Exception("Mock error")):
            result = mock_browser.init_session(action)
            assert result["status"] == "error"
            assert "Failed to initialize session" in result["content"][0]["text"]

    def test_list_local_sessions_empty(self, mock_browser):
        """Test listing sessions when none exist."""
        result = mock_browser.list_local_sessions()
        assert result["status"] == "success"
        assert result["content"][0]["json"]["totalSessions"] == 0

    def test_list_local_sessions_with_sessions(self, mock_browser):
        """Test listing sessions with existing sessions."""
        # Create a session first
        action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(action)
        
        result = mock_browser.list_local_sessions()
        assert result["status"] == "success"
        assert result["content"][0]["json"]["totalSessions"] == 1

    def test_validate_session_not_found(self, mock_browser):
        """Test session validation with non-existent session."""
        result = mock_browser.validate_session("nonexistent")
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_validate_session_exists(self, mock_browser):
        """Test session validation with existing session."""
        # Create a session first
        action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(action)
        
        result = mock_browser.validate_session("test-session-001")
        assert result is None

    def test_get_session_page_not_found(self, mock_browser):
        """Test getting page for non-existent session."""
        page = mock_browser.get_session_page("nonexistent")
        assert page is None

    def test_get_session_page_exists(self, mock_browser):
        """Test getting page for existing session."""
        # Create a session first
        action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(action)
        
        page = mock_browser.get_session_page("test-session-001")
        assert page is not None


class TestNavigationActions:
    """Test browser navigation actions."""

    def test_navigate_success(self, mock_browser):
        """Test successful navigation."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        action = NavigateAction(type="navigate", 
            session_name="test-session-001",
            url="https://example.com"
        )
        
        result = mock_browser.navigate(action)
        assert result["status"] == "success"
        assert "Navigated to" in result["content"][0]["text"]

    def test_navigate_session_not_found(self, mock_browser):
        """Test navigation with non-existent session."""
        action = NavigateAction(type="navigate", 
            session_name="nonexistent",
            url="https://example.com"
        )
        
        result = mock_browser.navigate(action)
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_navigate_network_errors(self, mock_browser):
        """Test navigation with various network errors."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        error_cases = [
            ("ERR_NAME_NOT_RESOLVED", "Could not resolve domain"),
            ("ERR_CONNECTION_REFUSED", "Connection refused"),
            ("ERR_CONNECTION_TIMED_OUT", "Connection timed out"),
            ("ERR_SSL_PROTOCOL_ERROR", "SSL/TLS error"),
            ("ERR_CERT_INVALID", "Certificate error"),
        ]
        
        for error_code, expected_message in error_cases:
            with patch.object(mock_browser._sessions["test-session-001"].page, 'goto', 
                            side_effect=Exception(error_code)):
                action = NavigateAction(type="navigate", 
                    session_name="test-session-001",
                    url="https://example.com"
                )
                
                result = mock_browser.navigate(action)
                assert result["status"] == "error"
                assert expected_message in result["content"][0]["text"]

    def test_back_action(self, mock_browser):
        """Test back navigation."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        action = BackAction(type="back", session_name="test-session-001")
        result = mock_browser.back(action)
        assert result["status"] == "success"

    def test_forward_action(self, mock_browser):
        """Test forward navigation."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        action = ForwardAction(type="forward", session_name="test-session-001")
        result = mock_browser.forward(action)
        assert result["status"] == "success"

    def test_refresh_action(self, mock_browser):
        """Test page refresh."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        action = RefreshAction(type="refresh", session_name="test-session-001")
        result = mock_browser.refresh(action)
        assert result["status"] == "success"


class TestInteractionActions:
    """Test browser interaction actions."""

    def test_click_success(self, mock_browser):
        """Test successful click action."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        action = ClickAction(type="click", 
            session_name="test-session-001",
            selector="button"
        )
        
        result = mock_browser.click(action)
        assert result["status"] == "success"

    def test_click_error(self, mock_browser):
        """Test click action with error."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        with patch.object(mock_browser._sessions["test-session-001"].page, 'click', 
                        side_effect=Exception("Element not found")):
            action = ClickAction(type="click", 
                session_name="test-session-001",
                selector="button"
            )
            
            result = mock_browser.click(action)
            assert result["status"] == "error"

    def test_type_success(self, mock_browser):
        """Test successful type action."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        action = TypeAction(type="type", 
            session_name="test-session-001",
            selector="input",
            text="test text"
        )
        
        result = mock_browser.type(action)
        assert result["status"] == "success"

    def test_type_error(self, mock_browser):
        """Test type action with error."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        with patch.object(mock_browser._sessions["test-session-001"].page, 'fill', 
                        side_effect=Exception("Element not found")):
            action = TypeAction(type="type", 
                session_name="test-session-001",
                selector="input",
                text="test text"
            )
            
            result = mock_browser.type(action)
            assert result["status"] == "error"

    def test_press_key_success(self, mock_browser):
        """Test successful key press."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        action = PressKeyAction(type="press_key", 
            session_name="test-session-001",
            key="Enter"
        )
        
        result = mock_browser.press_key(action)
        assert result["status"] == "success"


class TestContentActions:
    """Test browser content retrieval actions."""

    def test_get_text_success(self, mock_browser):
        """Test successful text retrieval."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        # Mock text content
        mock_browser._sessions["test-session-001"].page.text_content = AsyncMock(return_value="Test text")
        
        action = GetTextAction(type="get_text", 
            session_name="test-session-001",
            selector="p"
        )
        
        result = mock_browser.get_text(action)
        assert result["status"] == "success"
        assert "Test text" in result["content"][0]["text"]

    def test_get_html_full_page(self, mock_browser):
        """Test getting full page HTML."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        # Mock page content
        mock_browser._sessions["test-session-001"].page.content = AsyncMock(return_value="<html><body>Test</body></html>")
        
        action = GetHtmlAction(type="get_html", 
            session_name="test-session-001"
        )
        
        result = mock_browser.get_html(action)
        assert result["status"] == "success"
        assert "<html>" in result["content"][0]["text"]

    def test_get_html_with_selector(self, mock_browser):
        """Test getting HTML with selector."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        # Mock element HTML
        mock_browser._sessions["test-session-001"].page.wait_for_selector = AsyncMock()
        mock_browser._sessions["test-session-001"].page.inner_html = AsyncMock(return_value="<div>Test</div>")
        
        action = GetHtmlAction(type="get_html", 
            session_name="test-session-001",
            selector="div"
        )
        
        result = mock_browser.get_html(action)
        assert result["status"] == "success"
        assert "<div>Test</div>" in result["content"][0]["text"]

    def test_get_html_selector_timeout(self, mock_browser):
        """Test getting HTML with selector timeout."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        # Mock timeout error
        mock_browser._sessions["test-session-001"].page.wait_for_selector = AsyncMock(
            side_effect=PlaywrightTimeoutError("Timeout")
        )
        
        action = GetHtmlAction(type="get_html", 
            session_name="test-session-001",
            selector="div"
        )
        
        result = mock_browser.get_html(action)
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_get_html_long_content_truncation(self, mock_browser):
        """Test HTML content truncation for long content."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        # Mock long content
        long_content = "x" * 2000
        mock_browser._sessions["test-session-001"].page.content = AsyncMock(return_value=long_content)
        
        action = GetHtmlAction(type="get_html", 
            session_name="test-session-001"
        )
        
        result = mock_browser.get_html(action)
        assert result["status"] == "success"
        assert "..." in result["content"][0]["text"]


class TestScreenshotAction:
    """Test screenshot functionality."""

    def test_screenshot_default_path(self, mock_browser):
        """Test screenshot with default path."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        with patch("os.makedirs"), patch("time.time", return_value=1234567890):
            action = ScreenshotAction(type="screenshot", session_name="test-session-001")
            result = mock_browser.screenshot(action)
            assert result["status"] == "success"
            assert "screenshot_1234567890.png" in result["content"][0]["text"]

    def test_screenshot_custom_path(self, mock_browser):
        """Test screenshot with custom path."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        with patch("os.makedirs"):
            action = ScreenshotAction(type="screenshot", 
                session_name="test-session-001",
                path="custom.png"
            )
            result = mock_browser.screenshot(action)
            assert result["status"] == "success"
            assert "custom.png" in result["content"][0]["text"]

    def test_screenshot_absolute_path(self, mock_browser):
        """Test screenshot with absolute path."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        action = ScreenshotAction(type="screenshot", 
            session_name="test-session-001",
            path="/tmp/screenshot.png"
        )
        result = mock_browser.screenshot(action)
        assert result["status"] == "success"
        assert "/tmp/screenshot.png" in result["content"][0]["text"]

    def test_screenshot_no_active_page(self, mock_browser):
        """Test screenshot with no active page."""
        # Create session but mock no active page
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        with patch.object(mock_browser, 'get_session_page', return_value=None):
            action = ScreenshotAction(type="screenshot", session_name="test-session-001")
            result = mock_browser.screenshot(action)
            assert result["status"] == "error"
            assert "No active page" in result["content"][0]["text"]


class TestEvaluateAction:
    """Test JavaScript evaluation."""

    def test_evaluate_success(self, mock_browser):
        """Test successful JavaScript evaluation."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        # Mock evaluation result
        mock_browser._sessions["test-session-001"].page.evaluate = AsyncMock(return_value="result")
        
        action = EvaluateAction(type="evaluate", 
            session_name="test-session-001",
            script="document.title"
        )
        
        result = mock_browser.evaluate(action)
        assert result["status"] == "success"
        assert "result" in result["content"][0]["text"]

    def test_evaluate_with_syntax_fix(self, mock_browser):
        """Test JavaScript evaluation with syntax error fix."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        # Mock evaluation to fail first, then succeed
        mock_browser._sessions["test-session-001"].page.evaluate = AsyncMock(
            side_effect=[
                Exception("Illegal return statement"),
                "fixed result"
            ]
        )
        
        action = EvaluateAction(type="evaluate", 
            session_name="test-session-001",
            script="return 'test'"
        )
        
        result = mock_browser.evaluate(action)
        assert result["status"] == "success"
        assert "fixed result" in result["content"][0]["text"]

    def test_evaluate_fix_template_literals(self, mock_browser):
        """Test fixing template literals in JavaScript."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        # Mock evaluation to fail first, then succeed
        mock_browser._sessions["test-session-001"].page.evaluate = AsyncMock(
            side_effect=[
                Exception("Unexpected token"),
                "fixed result"
            ]
        )
        
        action = EvaluateAction(type="evaluate", 
            session_name="test-session-001",
            script="`Hello ${name}`"
        )
        
        result = mock_browser.evaluate(action)
        assert result["status"] == "success"

    def test_evaluate_fix_arrow_functions(self, mock_browser):
        """Test fixing arrow functions in JavaScript."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        # Mock evaluation to fail first, then succeed
        mock_browser._sessions["test-session-001"].page.evaluate = AsyncMock(
            side_effect=[
                Exception("Unexpected token"),
                "fixed result"
            ]
        )
        
        action = EvaluateAction(type="evaluate", 
            session_name="test-session-001",
            script="arr => arr.length"
        )
        
        result = mock_browser.evaluate(action)
        assert result["status"] == "success"

    def test_evaluate_fix_missing_braces(self, mock_browser):
        """Test fixing missing braces in JavaScript."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        # Mock evaluation to fail first, then succeed
        mock_browser._sessions["test-session-001"].page.evaluate = AsyncMock(
            side_effect=[
                Exception("Unexpected end of input"),
                "fixed result"
            ]
        )
        
        action = EvaluateAction(type="evaluate", 
            session_name="test-session-001",
            script="if (true) { console.log('test'"
        )
        
        result = mock_browser.evaluate(action)
        assert result["status"] == "success"

    def test_evaluate_fix_undefined_variable(self, mock_browser):
        """Test fixing undefined variables in JavaScript."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        # Mock evaluation to fail first, then succeed
        mock_browser._sessions["test-session-001"].page.evaluate = AsyncMock(
            side_effect=[
                Exception("'undefinedVar' is not defined"),
                "fixed result"
            ]
        )
        
        action = EvaluateAction(type="evaluate", 
            session_name="test-session-001",
            script="console.log(undefinedVar)"
        )
        
        result = mock_browser.evaluate(action)
        assert result["status"] == "success"

    def test_evaluate_unfixable_error(self, mock_browser):
        """Test JavaScript evaluation with unfixable error."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        # Mock evaluation to always fail
        mock_browser._sessions["test-session-001"].page.evaluate = AsyncMock(
            side_effect=Exception("Unfixable error")
        )
        
        action = EvaluateAction(type="evaluate", 
            session_name="test-session-001",
            script="invalid script"
        )
        
        result = mock_browser.evaluate(action)
        assert result["status"] == "error"


class TestTabManagement:
    """Test browser tab management."""

    def test_new_tab_success(self, mock_browser):
        """Test creating new tab."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        action = NewTabAction(type="new_tab", 
            session_name="test-session-001",
            tab_id="new_tab"
        )
        
        result = mock_browser.new_tab(action)
        assert result["status"] == "success"
        assert "new_tab" in result["content"][0]["text"]

    def test_new_tab_auto_id(self, mock_browser):
        """Test creating new tab with auto-generated ID."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        action = NewTabAction(type="new_tab", session_name="test-session-001")
        result = mock_browser.new_tab(action)
        assert result["status"] == "success"

    def test_new_tab_duplicate_id(self, mock_browser):
        """Test creating tab with duplicate ID."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        # Create first tab
        action1 = NewTabAction(type="new_tab", 
            session_name="test-session-001",
            tab_id="duplicate_tab"
        )
        mock_browser.new_tab(action1)
        
        # Try to create duplicate
        action2 = NewTabAction(type="new_tab", 
            session_name="test-session-001",
            tab_id="duplicate_tab"
        )
        result = mock_browser.new_tab(action2)
        assert result["status"] == "error"
        assert "already exists" in result["content"][0]["text"]

    def test_switch_tab_success(self, mock_browser):
        """Test switching tabs."""
        # Create session and tab first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        new_tab_action = NewTabAction(type="new_tab", 
            session_name="test-session-001",
            tab_id="target_tab"
        )
        mock_browser.new_tab(new_tab_action)
        
        # Switch to tab
        switch_action = SwitchTabAction(type="switch_tab", 
            session_name="test-session-001",
            tab_id="target_tab"
        )
        result = mock_browser.switch_tab(switch_action)
        assert result["status"] == "success"

    def test_switch_tab_not_found(self, mock_browser):
        """Test switching to non-existent tab."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        action = SwitchTabAction(type="switch_tab", 
            session_name="test-session-001",
            tab_id="nonexistent_tab"
        )
        result = mock_browser.switch_tab(action)
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_close_tab_success(self, mock_browser):
        """Test closing tab."""
        # Create session and tab first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        new_tab_action = NewTabAction(type="new_tab", 
            session_name="test-session-001",
            tab_id="closable_tab"
        )
        mock_browser.new_tab(new_tab_action)
        
        # Close tab
        close_action = CloseTabAction(
            type="close_tab",
            session_name="test-session-001",
            tab_id="closable_tab"
        )
        result = mock_browser.close_tab(close_action)
        assert result["status"] == "success"

    def test_list_tabs(self, mock_browser):
        """Test listing tabs."""
        # Create session and tabs first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        new_tab_action = NewTabAction(type="new_tab", 
            session_name="test-session-001",
            tab_id="listed_tab"
        )
        mock_browser.new_tab(new_tab_action)
        
        # List tabs
        list_action = ListTabsAction(type="list_tabs", session_name="test-session-001")
        result = mock_browser.list_tabs(list_action)
        assert result["status"] == "success"
        
        # Parse JSON response
        tabs_info = json.loads(result["content"][0]["text"])
        assert "main" in tabs_info
        assert "listed_tab" in tabs_info


class TestCookieActions:
    """Test cookie management."""

    def test_get_cookies(self, mock_browser):
        """Test getting cookies."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        # Mock cookies
        mock_cookies = [{"name": "test", "value": "cookie"}]
        mock_browser._sessions["test-session-001"].page.context.cookies = AsyncMock(return_value=mock_cookies)
        
        action = GetCookiesAction(type="get_cookies", session_name="test-session-001")
        result = mock_browser.get_cookies(action)
        assert result["status"] == "success"

    def test_set_cookies(self, mock_browser):
        """Test setting cookies."""
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        cookies = [{"name": "test", "value": "cookie", "domain": "example.com"}]
        action = SetCookiesAction(
            type="set_cookies",
            session_name="test-session-001",
            cookies=cookies
        )
        result = mock_browser.set_cookies(action)
        assert result["status"] == "success"


class TestCloseAction:
    """Test browser close action."""

    def test_close_browser(self, mock_browser):
        """Test closing browser."""
        action = CloseAction(type="close", session_name="test-session-001")
        result = mock_browser.close(action)
        assert result["status"] == "success"
        assert "Browser closed" in result["content"][0]["text"]

    def test_close_browser_error(self, mock_browser):
        """Test browser close with error."""
        with patch.object(mock_browser, '_execute_async', side_effect=Exception("Close error")):
            action = CloseAction(type="close", session_name="test-session-001")
            result = mock_browser.close(action)
            assert result["status"] == "error"


class TestAsyncExecution:
    """Test async execution handling."""

    def test_execute_async_with_nest_asyncio(self, mock_browser):
        """Test async execution with nest_asyncio."""
        async def test_coro():
            return "test result"
        
        # Mock nest_asyncio not applied
        mock_browser._nest_asyncio_applied = False
        
        with patch("nest_asyncio.apply") as mock_apply:
            result = mock_browser._execute_async(test_coro())
            mock_apply.assert_called_once()
            assert mock_browser._nest_asyncio_applied

    def test_execute_async_already_applied(self, mock_browser):
        """Test async execution when nest_asyncio already applied."""
        async def test_coro():
            return "test result"
        
        # Mock nest_asyncio already applied
        mock_browser._nest_asyncio_applied = True
        
        with patch("nest_asyncio.apply") as mock_apply:
            result = mock_browser._execute_async(test_coro())
            mock_apply.assert_not_called()


class TestAsyncCleanup:
    """Test async cleanup functionality."""

    def test_async_cleanup_with_sessions(self, mock_browser):
        """Test async cleanup with active sessions."""
        # Start the browser first
        mock_browser._start()
        
        # Create session first
        init_action = InitSessionAction(
            type="init_session",
            session_name="test-session-001",
            description="Test session"
        )
        mock_browser.init_session(init_action)
        
        # Mock session close to return errors
        mock_browser._sessions["test-session-001"].close = AsyncMock(return_value=["Test error"])
        
        # Run cleanup
        mock_browser._cleanup()
        
        # Verify sessions were cleared
        assert len(mock_browser._sessions) == 0

    def test_async_cleanup_playwright_error(self, mock_browser):
        """Test async cleanup with playwright stop error."""
        mock_browser._start()
        
        # Mock playwright stop to raise error
        mock_browser._playwright.stop = AsyncMock(side_effect=Exception("Stop error"))
        
        # Should not raise exception
        mock_browser._cleanup()
        assert mock_browser._playwright is None