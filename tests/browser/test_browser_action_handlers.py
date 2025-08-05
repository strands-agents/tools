"""
Comprehensive tests for Browser action handlers and error exception sections.

This module provides complete test coverage for all browser action handlers,
including success cases, error handling, and edge cases.
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from strands_tools.browser.browser import Browser
from strands_tools.browser.models import (
    BackAction,
    BrowserInput,
    ClickAction,
    CloseAction,
    CloseTabAction,
    EvaluateAction,
    ExecuteCdpAction,
    ForwardAction,
    GetCookiesAction,
    GetHtmlAction,
    GetTextAction,
    InitSessionAction,
    ListLocalSessionsAction,
    ListTabsAction,
    NavigateAction,
    NetworkInterceptAction,
    NewTabAction,
    PressKeyAction,
    RefreshAction,
    ScreenshotAction,
    SetCookiesAction,
    SwitchTabAction,
    TypeAction,
)


class MockBrowser(Browser):
    """Mock implementation of Browser for testing."""

    def start_platform(self) -> None:
        """Mock platform startup."""
        pass

    def close_platform(self) -> None:
        """Mock platform cleanup."""
        pass

    async def create_browser_session(self):
        """Mock browser session creation."""
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        
        return mock_browser

@pytest.fixture
def mock_browser():
    """Create a mock browser instance for testing."""
    with patch("strands_tools.browser.browser.async_playwright") as mock_playwright:
        mock_playwright_instance = AsyncMock()
        mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)
        
        browser = MockBrowser()
        return browser


@pytest.fixture
def mock_session(mock_browser):
    """Create a mock session for testing."""
    # Initialize a session
    action = InitSessionAction(
        type="init_session",
        session_name="test-session-main",
        description="Test session for unit tests"
    )
    result = mock_browser.init_session(action)
    assert result["status"] == "success"
    return "test-session-main"


class TestBrowserActionDispatcher:
    """Test the main action dispatcher and unknown action handling."""

    def test_unknown_action_type(self, mock_browser):
        """Test handling of unknown action types."""
        # Create a mock action that's not in the handler list
        class UnknownAction:
            pass
        
        unknown_action = UnknownAction()
        browser_input = Mock()
        browser_input.action = unknown_action
        
        result = mock_browser.browser(browser_input)
        
        assert result["status"] == "error"
        assert "Unknown action type" in result["content"][0]["text"]
        assert str(type(unknown_action)) in result["content"][0]["text"]

    def test_dict_input_conversion(self, mock_browser):
        """Test conversion of dict input to BrowserInput."""
        dict_input = {
            "action": {
                "type": "list_local_sessions"
            }
        }
        
        result = mock_browser.browser(dict_input)
        
        # Should successfully process the dict input
        assert result["status"] == "success"
        assert "sessions" in result["content"][0]["json"]


class TestInitSessionAction:
    """Test InitSessionAction handler and error cases."""

    def test_init_session_success(self, mock_browser):
        """Test successful session initialization."""
        action = InitSessionAction(
            type="init_session",
            session_name="new-session-test",
            description="Test session"
        )
        
        result = mock_browser.init_session(action)
        
        assert result["status"] == "success"
        assert result["content"][0]["json"]["sessionName"] == "new-session-test"
        assert result["content"][0]["json"]["description"] == "Test session"

    def test_init_session_already_exists(self, mock_browser, mock_session):
        """Test error when session already exists."""
        action = InitSessionAction(
            type="init_session",
            session_name="test-session-main",  # Same as mock_session
            description="Duplicate session"
        )
        
        result = mock_browser.init_session(action)
        
        assert result["status"] == "error"
        assert "already exists" in result["content"][0]["text"]

    def test_init_session_browser_creation_error(self, mock_browser):
        """Test error during browser session creation."""
        with patch.object(mock_browser, 'create_browser_session', side_effect=Exception("Browser creation failed")):
            action = InitSessionAction(
                type="init_session",
                session_name="error-session-test",
                description="Error test session"
            )
            
            result = mock_browser.init_session(action)
            
            assert result["status"] == "error"
            assert "Failed to initialize session" in result["content"][0]["text"]
            assert "Browser creation failed" in result["content"][0]["text"]


class TestListLocalSessionsAction:
    """Test ListLocalSessionsAction handler."""

    def test_list_sessions_empty(self, mock_browser):
        """Test listing sessions when none exist."""
        result = mock_browser.list_local_sessions()
        
        assert result["status"] == "success"
        assert result["content"][0]["json"]["sessions"] == []
        assert result["content"][0]["json"]["totalSessions"] == 0

    def test_list_sessions_with_data(self, mock_browser, mock_session):
        """Test listing sessions with existing sessions."""
        result = mock_browser.list_local_sessions()
        
        assert result["status"] == "success"
        sessions = result["content"][0]["json"]["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["sessionName"] == "test-session-main"
        assert result["content"][0]["json"]["totalSessions"] == 1


class TestNavigateAction:
    """Test NavigateAction handler and error cases."""

    def test_navigate_success(self, mock_browser, mock_session):
        """Test successful navigation."""
        action = NavigateAction(
            type="navigate",
            session_name="test-session-main",
            url="https://example.com"
        )
        
        result = mock_browser.navigate(action)
        
        assert result["status"] == "success"
        assert "Navigated to https://example.com" in result["content"][0]["text"]

    def test_navigate_session_not_found(self, mock_browser):
        """Test navigation with non-existent session."""
        action = NavigateAction(
            type="navigate",
            session_name="nonexistent-session",
            url="https://example.com"
        )
        
        result = mock_browser.navigate(action)
        
        assert result["status"] == "error"
        assert "Session 'nonexistent-session' not found" in result["content"][0]["text"]

    def test_navigate_no_active_page(self, mock_browser):
        """Test navigation when no active page exists."""
        # Create session but remove the page
        mock_browser._sessions["test-session-main"] = Mock()
        mock_browser._sessions["test-session-main"].get_active_page = Mock(return_value=None)
        
        action = NavigateAction(
            type="navigate",
            session_name="test-session-main",
            url="https://example.com"
        )
        
        result = mock_browser.navigate(action)
        
        assert result["status"] == "error"
        assert "No active page for session" in result["content"][0]["text"]

    @pytest.mark.parametrize("error_type,expected_message", [
        ("ERR_NAME_NOT_RESOLVED", "Could not resolve domain"),
        ("ERR_CONNECTION_REFUSED", "Connection refused"),
        ("ERR_CONNECTION_TIMED_OUT", "Connection timed out"),
        ("ERR_SSL_PROTOCOL_ERROR", "SSL/TLS error"),
        ("ERR_CERT_INVALID", "Certificate error"),
        ("Generic error", "Generic error"),
    ])
    def test_navigate_network_errors(self, mock_browser, mock_session, error_type, expected_message):
        """Test navigation with various network errors."""
        # Mock the page to raise an exception
        session = mock_browser._sessions["test-session-main"]
        session.get_active_page().goto = AsyncMock(side_effect=Exception(error_type))
        
        action = NavigateAction(
            type="navigate",
            session_name="test-session-main",
            url="https://example.com"
        )
        
        result = mock_browser.navigate(action)
        
        assert result["status"] == "error"
        assert expected_message in result["content"][0]["text"]


class TestClickAction:
    """Test ClickAction handler and error cases."""

    def test_click_success(self, mock_browser, mock_session):
        """Test successful click action."""
        action = ClickAction(
            type="click",
            session_name="test-session-main",
            selector="button#submit"
        )
        
        result = mock_browser.click(action)
        
        assert result["status"] == "success"
        assert "Clicked element: button#submit" in result["content"][0]["text"]

    def test_click_session_not_found(self, mock_browser):
        """Test click with non-existent session."""
        action = ClickAction(
            type="click",
            session_name="nonexistent-session",
            selector="button"
        )
        
        result = mock_browser.click(action)
        
        assert result["status"] == "error"
        assert "Session 'nonexistent-session' not found" in result["content"][0]["text"]

    def test_click_element_not_found(self, mock_browser, mock_session):
        """Test click when element is not found."""
        # Mock the page to raise an exception
        session = mock_browser._sessions["test-session-main"]
        session.get_active_page().click = AsyncMock(side_effect=Exception("Element not found"))
        
        action = ClickAction(
            type="click",
            session_name="test-session-main",
            selector="button#nonexistent"
        )
        
        result = mock_browser.click(action)
        
        assert result["status"] == "error"
        assert "Element not found" in result["content"][0]["text"]


class TestTypeAction:
    """Test TypeAction handler and error cases."""

    def test_type_success(self, mock_browser, mock_session):
        """Test successful type action."""
        action = TypeAction(
            type="type",
            session_name="test-session-main",
            selector="input#username",
            text="testuser"
        )
        
        result = mock_browser.type(action)
        
        assert result["status"] == "success"
        assert "Typed 'testuser' into input#username" in result["content"][0]["text"]

    def test_type_session_not_found(self, mock_browser):
        """Test type with non-existent session."""
        action = TypeAction(
            type="type",
            session_name="nonexistent-session",
            selector="input",
            text="test"
        )
        
        result = mock_browser.type(action)
        
        assert result["status"] == "error"
        assert "Session 'nonexistent-session' not found" in result["content"][0]["text"]

    def test_type_element_error(self, mock_browser, mock_session):
        """Test type when element interaction fails."""
        # Mock the page to raise an exception
        session = mock_browser._sessions["test-session-main"]
        session.get_active_page().fill = AsyncMock(side_effect=Exception("Input field not found"))
        
        action = TypeAction(
            type="type",
            session_name="test-session-main",
            selector="input#nonexistent",
            text="test"
        )
        
        result = mock_browser.type(action)
        
        assert result["status"] == "error"
        assert "Input field not found" in result["content"][0]["text"]


class TestEvaluateAction:
    """Test EvaluateAction handler and JavaScript error handling."""

    def test_evaluate_success(self, mock_browser, mock_session):
        """Test successful JavaScript evaluation."""
        # Mock the page to return a result
        session = mock_browser._sessions["test-session-main"]
        session.get_active_page().evaluate = AsyncMock(return_value="Hello World")
        
        action = EvaluateAction(
            type="evaluate",
            session_name="test-session-main",
            script="document.title"
        )
        
        result = mock_browser.evaluate(action)
        
        assert result["status"] == "success"
        assert "Evaluation result: Hello World" in result["content"][0]["text"]

    def test_evaluate_illegal_return_statement_fix(self, mock_browser, mock_session):
        """Test JavaScript syntax fix for illegal return statement."""
        # Mock the page to fail first, then succeed with fixed script
        session = mock_browser._sessions["test-session-main"]
        page = session.get_active_page()
        
        # First call fails with illegal return, second succeeds
        page.evaluate = AsyncMock(side_effect=[
            Exception("Illegal return statement"),
            "Fixed result"
        ])
        
        action = EvaluateAction(
            type="evaluate",
            session_name="test-session-main",
            script="return 'test'"
        )
        
        result = mock_browser.evaluate(action)
        
        assert result["status"] == "success"
        assert "Evaluation result (fixed): Fixed result" in result["content"][0]["text"]

    def test_evaluate_template_literal_fix(self, mock_browser, mock_session):
        """Test JavaScript syntax fix for template literals."""
        session = mock_browser._sessions["test-session-main"]
        page = session.get_active_page()
        
        page.evaluate = AsyncMock(side_effect=[
            Exception("Unexpected token"),
            "Fixed result"
        ])
        
        action = EvaluateAction(
            type="evaluate",
            session_name="test-session-main",
            script="`Hello ${name}`"
        )
        
        result = mock_browser.evaluate(action)
        
        assert result["status"] == "success"
        assert "Evaluation result (fixed): Fixed result" in result["content"][0]["text"]

    def test_evaluate_arrow_function_fix(self, mock_browser, mock_session):
        """Test JavaScript syntax fix for arrow functions."""
        session = mock_browser._sessions["test-session-main"]
        page = session.get_active_page()
        
        page.evaluate = AsyncMock(side_effect=[
            Exception("Unexpected token"),
            "Fixed result"
        ])
        
        action = EvaluateAction(
            type="evaluate",
            session_name="test-session-main",
            script="() => 'test'"
        )
        
        result = mock_browser.evaluate(action)
        
        assert result["status"] == "success"
        assert "Evaluation result (fixed): Fixed result" in result["content"][0]["text"]

    def test_evaluate_missing_braces_fix(self, mock_browser, mock_session):
        """Test JavaScript syntax fix for missing braces."""
        session = mock_browser._sessions["test-session-main"]
        page = session.get_active_page()
        
        page.evaluate = AsyncMock(side_effect=[
            Exception("Unexpected end of input"),
            "Fixed result"
        ])
        
        action = EvaluateAction(
            type="evaluate",
            session_name="test-session-main",
            script="if (true) { console.log('test'"
        )
        
        result = mock_browser.evaluate(action)
        
        assert result["status"] == "success"
        assert "Evaluation result (fixed): Fixed result" in result["content"][0]["text"]

    def test_evaluate_undefined_variable_fix(self, mock_browser, mock_session):
        """Test JavaScript syntax fix for undefined variables."""
        session = mock_browser._sessions["test-session-main"]
        page = session.get_active_page()
        
        page.evaluate = AsyncMock(side_effect=[
            Exception("'testVar' is not defined"),
            "Fixed result"
        ])
        
        action = EvaluateAction(
            type="evaluate",
            session_name="test-session-main",
            script="console.log(testVar)"
        )
        
        result = mock_browser.evaluate(action)
        
        assert result["status"] == "success"
        assert "Evaluation result (fixed): Fixed result" in result["content"][0]["text"]

    def test_evaluate_unfixable_error(self, mock_browser, mock_session):
        """Test JavaScript evaluation with unfixable error."""
        session = mock_browser._sessions["test-session-main"]
        page = session.get_active_page()
        
        page.evaluate = AsyncMock(side_effect=Exception("Unfixable syntax error"))
        
        action = EvaluateAction(
            type="evaluate",
            session_name="test-session-main",
            script="invalid javascript $$$ syntax"
        )
        
        result = mock_browser.evaluate(action)
        
        assert result["status"] == "error"
        assert "Unfixable syntax error" in result["content"][0]["text"]

    def test_evaluate_fix_fails_too(self, mock_browser, mock_session):
        """Test JavaScript evaluation where both original and fix fail."""
        session = mock_browser._sessions["test-session-main"]
        page = session.get_active_page()
        
        page.evaluate = AsyncMock(side_effect=[
            Exception("Illegal return statement"),
            Exception("Still broken after fix")
        ])
        
        action = EvaluateAction(
            type="evaluate",
            session_name="test-session-main",
            script="return 'test'"
        )
        
        result = mock_browser.evaluate(action)
        
        assert result["status"] == "error"
        assert "Still broken after fix" in result["content"][0]["text"]


class TestGetTextAction:
    """Test GetTextAction handler and error cases."""

    def test_get_text_success(self, mock_browser, mock_session):
        """Test successful text extraction."""
        session = mock_browser._sessions["test-session-main"]
        session.get_active_page().text_content = AsyncMock(return_value="Sample text content")
        
        action = GetTextAction(
            type="get_text",
            session_name="test-session-main",
            selector="h1"
        )
        
        result = mock_browser.get_text(action)
        
        assert result["status"] == "success"
        assert "Text content: Sample text content" in result["content"][0]["text"]

    def test_get_text_session_not_found(self, mock_browser):
        """Test text extraction with non-existent session."""
        action = GetTextAction(
            type="get_text",
            session_name="nonexistent-session",
            selector="h1"
        )
        
        result = mock_browser.get_text(action)
        
        assert result["status"] == "error"
        assert "Session 'nonexistent-session' not found" in result["content"][0]["text"]

    def test_get_text_element_error(self, mock_browser, mock_session):
        """Test text extraction with element error."""
        session = mock_browser._sessions["test-session-main"]
        session.get_active_page().text_content = AsyncMock(side_effect=Exception("Element not found"))
        
        action = GetTextAction(
            type="get_text",
            session_name="test-session-main",
            selector="h1#nonexistent"
        )
        
        result = mock_browser.get_text(action)
        
        assert result["status"] == "error"
        assert "Element not found" in result["content"][0]["text"]


class TestGetHtmlAction:
    """Test GetHtmlAction handler and error cases."""

    def test_get_html_full_page(self, mock_browser, mock_session):
        """Test getting full page HTML."""
        session = mock_browser._sessions["test-session-main"]
        session.get_active_page().content = AsyncMock(return_value="<html><body>Test</body></html>")
        
        action = GetHtmlAction(
            type="get_html",
            session_name="test-session-main"
        )
        
        result = mock_browser.get_html(action)
        
        assert result["status"] == "success"
        assert "<html><body>Test</body></html>" in result["content"][0]["text"]

    def test_get_html_with_selector(self, mock_browser, mock_session):
        """Test getting HTML from specific element."""
        session = mock_browser._sessions["test-session-main"]
        page = session.get_active_page()
        page.wait_for_selector = AsyncMock()
        page.inner_html = AsyncMock(return_value="<div>Inner content</div>")
        
        action = GetHtmlAction(
            type="get_html",
            session_name="test-session-main",
            selector="div.content"
        )
        
        result = mock_browser.get_html(action)
        
        assert result["status"] == "success"
        assert "<div>Inner content</div>" in result["content"][0]["text"]

    def test_get_html_selector_timeout(self, mock_browser, mock_session):
        """Test getting HTML when selector times out."""
        session = mock_browser._sessions["test-session-main"]
        page = session.get_active_page()
        page.wait_for_selector = AsyncMock(side_effect=PlaywrightTimeoutError("Timeout"))
        
        action = GetHtmlAction(
            type="get_html",
            session_name="test-session-main",
            selector="div.nonexistent"
        )
        
        result = mock_browser.get_html(action)
        
        assert result["status"] == "error"
        assert "Element with selector 'div.nonexistent' not found" in result["content"][0]["text"]

    def test_get_html_long_content_truncation(self, mock_browser, mock_session):
        """Test HTML content truncation for long content."""
        long_html = "<html>" + "x" * 2000 + "</html>"
        session = mock_browser._sessions["test-session-main"]
        session.get_active_page().content = AsyncMock(return_value=long_html)
        
        action = GetHtmlAction(
            type="get_html",
            session_name="test-session-main"
        )
        
        result = mock_browser.get_html(action)
        
        assert result["status"] == "success"
        content = result["content"][0]["text"]
        assert len(content) <= 1003  # 1000 chars + "..."
        assert content.endswith("...")

    def test_get_html_error(self, mock_browser, mock_session):
        """Test HTML extraction with general error."""
        session = mock_browser._sessions["test-session-main"]
        session.get_active_page().content = AsyncMock(side_effect=Exception("HTML extraction failed"))
        
        action = GetHtmlAction(
            type="get_html",
            session_name="test-session-main"
        )
        
        result = mock_browser.get_html(action)
        
        assert result["status"] == "error"
        assert "HTML extraction failed" in result["content"][0]["text"]


class TestScreenshotAction:
    """Test ScreenshotAction handler and error cases."""

    def test_screenshot_success_default_path(self, mock_browser, mock_session):
        """Test successful screenshot with default path."""
        with patch('os.makedirs'), patch('time.time', return_value=1234567890):
            action = ScreenshotAction(
            type="screenshot",
            session_name="test-session-main"
            )
            
            result = mock_browser.screenshot(action)
            
            assert result["status"] == "success"
            assert "screenshot_1234567890.png" in result["content"][0]["text"]

    def test_screenshot_success_custom_path(self, mock_browser, mock_session):
        """Test successful screenshot with custom path."""
        with patch('os.makedirs'):
            action = ScreenshotAction(
            type="screenshot",
            session_name="test-session-main",
                path="custom_screenshot.png"
            )
            
            result = mock_browser.screenshot(action)
            
            assert result["status"] == "success"
            assert "custom_screenshot.png" in result["content"][0]["text"]

    def test_screenshot_absolute_path(self, mock_browser, mock_session):
        """Test screenshot with absolute path."""
        with patch('os.makedirs'):
            action = ScreenshotAction(
            type="screenshot",
            session_name="test-session-main",
                path="/tmp/absolute_screenshot.png"
            )
            
            result = mock_browser.screenshot(action)
            
            assert result["status"] == "success"
            assert "/tmp/absolute_screenshot.png" in result["content"][0]["text"]

    def test_screenshot_session_not_found(self, mock_browser):
        """Test screenshot with non-existent session."""
        action = ScreenshotAction(
            type="screenshot",
            session_name="nonexistent-session"
        )
        
        result = mock_browser.screenshot(action)
        
        assert result["status"] == "error"
        assert "Session 'nonexistent-session' not found" in result["content"][0]["text"]

    def test_screenshot_no_active_page(self, mock_browser):
        """Test screenshot when no active page exists."""
        mock_browser._sessions["test-session-main"] = Mock()
        mock_browser._sessions["test-session-main"].get_active_page = Mock(return_value=None)
        
        action = ScreenshotAction(
            type="screenshot",
            session_name="test-session-main"
        )
        
        result = mock_browser.screenshot(action)
        
        assert result["status"] == "error"
        assert "No active page for session" in result["content"][0]["text"]

    def test_screenshot_error(self, mock_browser, mock_session):
        """Test screenshot with error."""
        session = mock_browser._sessions["test-session-main"]
        session.get_active_page().screenshot = AsyncMock(side_effect=Exception("Screenshot failed"))
        
        action = ScreenshotAction(
            type="screenshot",
            session_name="test-session-main"
        )
        
        result = mock_browser.screenshot(action)
        
        assert result["status"] == "error"
        assert "Screenshot failed" in result["content"][0]["text"]


class TestRefreshAction:
    """Test RefreshAction handler and error cases."""

    def test_refresh_success(self, mock_browser, mock_session):
        """Test successful page refresh."""
        action = RefreshAction(
            type="refresh",
            session_name="test-session-main"
        )
        
        result = mock_browser.refresh(action)
        
        assert result["status"] == "success"
        assert "Page refreshed" in result["content"][0]["text"]

    def test_refresh_session_not_found(self, mock_browser):
        """Test refresh with non-existent session."""
        action = RefreshAction(
            type="refresh",
            session_name="nonexistent-session"
        )
        
        result = mock_browser.refresh(action)
        
        assert result["status"] == "error"
        assert "Session 'nonexistent-session' not found" in result["content"][0]["text"]

    def test_refresh_error(self, mock_browser, mock_session):
        """Test refresh with error."""
        session = mock_browser._sessions["test-session-main"]
        session.get_active_page().reload = AsyncMock(side_effect=Exception("Refresh failed"))
        
        action = RefreshAction(
            type="refresh",
            session_name="test-session-main"
        )
        
        result = mock_browser.refresh(action)
        
        assert result["status"] == "error"
        assert "Refresh failed" in result["content"][0]["text"]


class TestBackAction:
    """Test BackAction handler and error cases."""

    def test_back_success(self, mock_browser, mock_session):
        """Test successful back navigation."""
        action = BackAction(
            type="back",
            session_name="test-session-main"
        )
        
        result = mock_browser.back(action)
        
        assert result["status"] == "success"
        assert "Navigated back" in result["content"][0]["text"]

    def test_back_session_not_found(self, mock_browser):
        """Test back navigation with non-existent session."""
        action = BackAction(
            type="back",
            session_name="nonexistent-session"
        )
        
        result = mock_browser.back(action)
        
        assert result["status"] == "error"
        assert "Session 'nonexistent-session' not found" in result["content"][0]["text"]

    def test_back_error(self, mock_browser, mock_session):
        """Test back navigation with error."""
        session = mock_browser._sessions["test-session-main"]
        session.get_active_page().go_back = AsyncMock(side_effect=Exception("Back navigation failed"))
        
        action = BackAction(
            type="back",
            session_name="test-session-main"
        )
        
        result = mock_browser.back(action)
        
        assert result["status"] == "error"
        assert "Back navigation failed" in result["content"][0]["text"]


class TestForwardAction:
    """Test ForwardAction handler and error cases."""

    def test_forward_success(self, mock_browser, mock_session):
        """Test successful forward navigation."""
        action = ForwardAction(
            type="forward",
            session_name="test-session-main"
        )
        
        result = mock_browser.forward(action)
        
        assert result["status"] == "success"
        assert "Navigated forward" in result["content"][0]["text"]

    def test_forward_session_not_found(self, mock_browser):
        """Test forward navigation with non-existent session."""
        action = ForwardAction(
            type="forward",
            session_name="nonexistent-session"
        )
        
        result = mock_browser.forward(action)
        
        assert result["status"] == "error"
        assert "Session 'nonexistent-session' not found" in result["content"][0]["text"]

    def test_forward_error(self, mock_browser, mock_session):
        """Test forward navigation with error."""
        session = mock_browser._sessions["test-session-main"]
        session.get_active_page().go_forward = AsyncMock(side_effect=Exception("Forward navigation failed"))
        
        action = ForwardAction(
            type="forward",
            session_name="test-session-main"
        )
        
        result = mock_browser.forward(action)
        
        assert result["status"] == "error"
        assert "Forward navigation failed" in result["content"][0]["text"]
        
class TestNewTabAction:
    """Test NewTabAction handler and error cases."""

    def test_new_tab_success_default_id(self, mock_browser, mock_session):
        """Test successful new tab creation with default ID."""
        action = NewTabAction(
            type="new_tab",
            session_name="test-session-main"
        )
        
        result = mock_browser.new_tab(action)
        
        assert result["status"] == "success"
        assert "Created new tab with ID: tab_2" in result["content"][0]["text"]

    def test_new_tab_success_custom_id(self, mock_browser, mock_session):
        """Test successful new tab creation with custom ID."""
        action = NewTabAction(
            type="new_tab",
            session_name="test-session-main",
            tab_id="custom-tab"
        )
        
        result = mock_browser.new_tab(action)
        
        assert result["status"] == "success"
        assert "Created new tab with ID: custom-tab" in result["content"][0]["text"]

    def test_new_tab_duplicate_id(self, mock_browser, mock_session):
        """Test new tab creation with duplicate ID."""
        # First create a tab
        action1 = NewTabAction(
            type="new_tab",
            session_name="test-session-main",
            tab_id="duplicate-tab"
        )
        mock_browser.new_tab(action1)
        
        # Try to create another with same ID
        action2 = NewTabAction(
            type="new_tab",
            session_name="test-session-main",
            tab_id="duplicate-tab"
        )
        
        result = mock_browser.new_tab(action2)
        
        assert result["status"] == "error"
        assert "Tab with ID duplicate-tab already exists" in result["content"][0]["text"]

    def test_new_tab_session_not_found(self, mock_browser):
        """Test new tab creation with non-existent session."""
        action = NewTabAction(
            type="new_tab",
            session_name="nonexistent-session"
        )
        
        result = mock_browser.new_tab(action)
        
        assert result["status"] == "error"
        assert "Session 'nonexistent-session' not found" in result["content"][0]["text"]

    def test_new_tab_error(self, mock_browser, mock_session):
        """Test new tab creation with error."""
        session = mock_browser._sessions["test-session-main"]
        session.context.new_page = AsyncMock(side_effect=Exception("Tab creation failed"))
        
        action = NewTabAction(
            type="new_tab",
            session_name="test-session-main"
        )
        
        result = mock_browser.new_tab(action)
        
        assert result["status"] == "error"
        assert "Tab creation failed" in result["content"][0]["text"]


class TestSwitchTabAction:
    """Test SwitchTabAction handler and error cases."""

    def test_switch_tab_success(self, mock_browser, mock_session):
        """Test successful tab switching."""
        # Create a new tab first
        new_tab_action = NewTabAction(
            type="new_tab",
            session_name="test-session-main",
            tab_id="tab-to-switch"
        )
        mock_browser.new_tab(new_tab_action)
        
        # Now switch to it
        action = SwitchTabAction(
            type="switch_tab",
            session_name="test-session-main",
            tab_id="tab-to-switch"
        )
        
        result = mock_browser.switch_tab(action)
        
        assert result["status"] == "success"
        assert "Switched to tab: tab-to-switch" in result["content"][0]["text"]

    def test_switch_tab_not_found(self, mock_browser, mock_session):
        """Test switching to non-existent tab."""
        action = SwitchTabAction(
            type="switch_tab",
            session_name="test-session-main",
            tab_id="nonexistent-tab"
        )
        
        result = mock_browser.switch_tab(action)
        
        assert result["status"] == "error"
        assert "Tab with ID 'nonexistent-tab' not found" in result["content"][0]["text"]
        assert "Available tabs:" in result["content"][0]["text"]

    def test_switch_tab_session_not_found(self, mock_browser):
        """Test tab switching with non-existent session."""
        action = SwitchTabAction(
            type="switch_tab",
            session_name="nonexistent-session",
            tab_id="some-tab"
        )
        
        result = mock_browser.switch_tab(action)
        
        assert result["status"] == "error"
        assert "Session 'nonexistent-session' not found" in result["content"][0]["text"]

    def test_switch_tab_bring_to_front_error(self, mock_browser, mock_session):
        """Test tab switching when bring_to_front fails."""
        # Create a new tab first
        new_tab_action = NewTabAction(
            type="new_tab",
            session_name="test-session-main",
            tab_id="tab-with-error"
        )
        mock_browser.new_tab(new_tab_action)
        
        # Mock bring_to_front to fail
        session = mock_browser._sessions["test-session-main"]
        session.get_active_page().bring_to_front = AsyncMock(side_effect=Exception("Bring to front failed"))
        
        action = SwitchTabAction(
            type="switch_tab",
            session_name="test-session-main",
            tab_id="tab-with-error"
        )
        
        result = mock_browser.switch_tab(action)
        
        # Should still succeed even if bring_to_front fails
        assert result["status"] == "success"
        assert "Switched to tab: tab-with-error" in result["content"][0]["text"]


class TestCloseAction:
    """Test CloseAction handler and error cases."""

    def test_close_success(self, mock_browser, mock_session):
        """Test successful browser close."""
        action = CloseAction(
            type="close",
            session_name="test-session-main"
        )
        
        result = mock_browser.close(action)
        
        assert result["status"] == "success"
        assert "Browser closed" in result["content"][0]["text"]

    def test_close_error(self, mock_browser, mock_session):
        """Test browser close with error."""
        # Mock _async_cleanup to raise an exception
        with patch.object(mock_browser, '_async_cleanup', side_effect=Exception("Close failed")):
            action = CloseAction(
                type="close",
                session_name="test-session-main"
            )
            
            result = mock_browser.close(action)
            
            assert result["status"] == "error"
            assert "Close failed" in result["content"][0]["text"]


class TestSessionValidation:
    """Test session validation helper methods."""

    def test_validate_session_exists(self, mock_browser, mock_session):
        """Test session validation when session exists."""
        result = mock_browser.validate_session("test-session-main")
        
        assert result is None  # No error

    def test_validate_session_not_found(self, mock_browser):
        """Test session validation when session doesn't exist."""
        result = mock_browser.validate_session("nonexistent-session")
        
        assert result is not None
        assert result["status"] == "error"
        assert "Session 'nonexistent-session' not found" in result["content"][0]["text"]

    def test_get_session_page_exists(self, mock_browser, mock_session):
        """Test getting session page when it exists."""
        page = mock_browser.get_session_page("test-session-main")
        
        assert page is not None

    def test_get_session_page_not_found(self, mock_browser):
        """Test getting session page when session doesn't exist."""
        page = mock_browser.get_session_page("nonexistent-session")
        
        assert page is None


class TestAsyncExecutionAndCleanup:
    """Test async execution and cleanup functionality."""

    def test_execute_async_applies_nest_asyncio(self, mock_browser):
        """Test that _execute_async applies nest_asyncio when needed."""
        # Reset the flag
        mock_browser._nest_asyncio_applied = False
        
        async def dummy_coro():
            return "test"
        
        with patch('nest_asyncio.apply') as mock_apply:
            result = mock_browser._execute_async(dummy_coro())
            
            mock_apply.assert_called_once()
            assert mock_browser._nest_asyncio_applied is True
            assert result == "test"

    def test_execute_async_skips_nest_asyncio_when_applied(self, mock_browser):
        """Test that _execute_async skips nest_asyncio when already applied."""
        # Set the flag
        mock_browser._nest_asyncio_applied = True
        
        async def dummy_coro():
            return "test"
        
        with patch('nest_asyncio.apply') as mock_apply:
            result = mock_browser._execute_async(dummy_coro())
            
            mock_apply.assert_not_called()
            assert result == "test"

    def test_destructor_cleanup(self, mock_browser):
        """Test that destructor calls cleanup properly."""
        with patch.object(mock_browser, '_cleanup') as mock_cleanup:
            mock_browser.__del__()
            
            mock_cleanup.assert_called_once()

    def test_destructor_cleanup_with_exception(self, mock_browser):
        """Test that destructor handles cleanup exceptions gracefully."""
        with patch.object(mock_browser, '_cleanup', side_effect=Exception("Cleanup failed")):
            # This should not raise an exception
            mock_browser.__del__()


if __name__ == "__main__":
    pytest.main([__file__])