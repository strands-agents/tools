"""
Tests for wait_until configuration on navigation-style browser actions.
"""

from unittest.mock import AsyncMock, Mock

from strands_tools.browser.browser import Browser
from strands_tools.browser.models import BackAction, BrowserSession, ForwardAction, NavigateAction, RefreshAction


class MockBrowser(Browser):
    def start_platform(self) -> None:
        pass

    def close_platform(self) -> None:
        pass

    async def create_browser_session(self):
        return Mock()


def _make_browser_with_page(mock_page):
    browser = MockBrowser()
    session = BrowserSession(session_name="test-session-0001", description="test", page=mock_page)
    session.add_tab("main", mock_page)
    browser._sessions[session.session_name] = session
    return browser


def test_navigate_default_wait_until_load():
    action = NavigateAction(session_name="test-session-0001", url="https://example.com")
    assert action.wait_until == "load"


def test_navigate_uses_wait_until_on_goto():
    mock_page = Mock()
    mock_page.goto = AsyncMock()
    mock_page.wait_for_load_state = AsyncMock()

    browser = _make_browser_with_page(mock_page)
    action = NavigateAction(session_name="test-session-0001", url="https://example.com", wait_until="load")
    browser.navigate(action)

    mock_page.goto.assert_awaited_once_with("https://example.com", wait_until="load")
    mock_page.wait_for_load_state.assert_not_awaited()


def test_refresh_uses_wait_until_on_reload():
    mock_page = Mock()
    mock_page.reload = AsyncMock()
    mock_page.wait_for_load_state = AsyncMock()

    browser = _make_browser_with_page(mock_page)
    action = RefreshAction(session_name="test-session-0001", wait_until="commit")
    browser.refresh(action)

    mock_page.reload.assert_awaited_once_with(wait_until="commit")
    mock_page.wait_for_load_state.assert_not_awaited()


def test_back_forward_use_wait_until():
    mock_page = Mock()
    mock_page.go_back = AsyncMock()
    mock_page.go_forward = AsyncMock()
    mock_page.wait_for_load_state = AsyncMock()

    browser = _make_browser_with_page(mock_page)
    browser.back(BackAction(session_name="test-session-0001", wait_until="domcontentloaded"))
    browser.forward(ForwardAction(session_name="test-session-0001", wait_until="networkidle"))

    mock_page.go_back.assert_awaited_once_with(wait_until="domcontentloaded")
    mock_page.go_forward.assert_awaited_once_with(wait_until="networkidle")
    mock_page.wait_for_load_state.assert_not_awaited()
