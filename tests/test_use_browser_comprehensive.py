"""
Comprehensive tests for use_browser.py to improve coverage.
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from src.strands_tools.use_browser import BrowserApiMethods, BrowserManager, use_browser


class TestBrowserApiMethods:
    """Test the BrowserApiMethods class."""

    @pytest.mark.asyncio
    async def test_navigate_success(self):
        """Test successful navigation."""
        page = AsyncMock()
        page.goto = AsyncMock()
        page.wait_for_load_state = AsyncMock()
        
        result = await BrowserApiMethods.navigate(page, "https://example.com")
        
        page.goto.assert_called_once_with("https://example.com")
        page.wait_for_load_state.assert_called_once_with("networkidle")
        assert result == "Navigated to https://example.com"

    @pytest.mark.asyncio
    async def test_navigate_name_not_resolved_error(self):
        """Test navigation with DNS resolution error."""
        page = AsyncMock()
        page.goto.side_effect = Exception("ERR_NAME_NOT_RESOLVED: Could not resolve host")
        
        with pytest.raises(ValueError, match="Could not resolve domain"):
            await BrowserApiMethods.navigate(page, "https://nonexistent.example")

    @pytest.mark.asyncio
    async def test_navigate_connection_refused_error(self):
        """Test navigation with connection refused error."""
        page = AsyncMock()
        page.goto.side_effect = Exception("ERR_CONNECTION_REFUSED: Connection refused")
        
        with pytest.raises(ValueError, match="Connection refused"):
            await BrowserApiMethods.navigate(page, "https://example.com")

    @pytest.mark.asyncio
    async def test_navigate_connection_timeout_error(self):
        """Test navigation with connection timeout error."""
        page = AsyncMock()
        page.goto.side_effect = Exception("ERR_CONNECTION_TIMED_OUT: Connection timed out")
        
        with pytest.raises(ValueError, match="Connection timed out"):
            await BrowserApiMethods.navigate(page, "https://example.com")

    @pytest.mark.asyncio
    async def test_navigate_ssl_protocol_error(self):
        """Test navigation with SSL protocol error."""
        page = AsyncMock()
        page.goto.side_effect = Exception("ERR_SSL_PROTOCOL_ERROR: SSL protocol error")
        
        with pytest.raises(ValueError, match="SSL/TLS error"):
            await BrowserApiMethods.navigate(page, "https://example.com")

    @pytest.mark.asyncio
    async def test_navigate_cert_error(self):
        """Test navigation with certificate error."""
        page = AsyncMock()
        page.goto.side_effect = Exception("ERR_CERT_AUTHORITY_INVALID: Certificate error")
        
        with pytest.raises(ValueError, match="Certificate error"):
            await BrowserApiMethods.navigate(page, "https://example.com")

    @pytest.mark.asyncio
    async def test_navigate_other_error(self):
        """Test navigation with other error that should be re-raised."""
        page = AsyncMock()
        page.goto.side_effect = Exception("Some other error")
        
        with pytest.raises(Exception, match="Some other error"):
            await BrowserApiMethods.navigate(page, "https://example.com")

    @pytest.mark.asyncio
    async def test_click(self):
        """Test click action."""
        page = AsyncMock()
        page.click = AsyncMock()
        
        result = await BrowserApiMethods.click(page, "#button")
        
        page.click.assert_called_once_with("#button")
        assert result == "Clicked element: #button"

    @pytest.mark.asyncio
    async def test_type(self):
        """Test type action."""
        page = AsyncMock()
        page.fill = AsyncMock()
        
        result = await BrowserApiMethods.type(page, "#input", "test text")
        
        page.fill.assert_called_once_with("#input", "test text")
        assert result == "Typed 'test text' into #input"

    @pytest.mark.asyncio
    async def test_evaluate(self):
        """Test evaluate action."""
        page = AsyncMock()
        page.evaluate = AsyncMock(return_value="evaluation result")
        
        result = await BrowserApiMethods.evaluate(page, "document.title")
        
        page.evaluate.assert_called_once_with("document.title")
        assert result == "Evaluation result: evaluation result"

    @pytest.mark.asyncio
    async def test_press_key(self):
        """Test press key action."""
        page = AsyncMock()
        page.keyboard = AsyncMock()
        page.keyboard.press = AsyncMock()
        
        result = await BrowserApiMethods.press_key(page, "Enter")
        
        page.keyboard.press.assert_called_once_with("Enter")
        assert result == "Pressed key: Enter"

    @pytest.mark.asyncio
    async def test_get_text(self):
        """Test get text action."""
        page = AsyncMock()
        page.text_content = AsyncMock(return_value="element text")
        
        result = await BrowserApiMethods.get_text(page, "#element")
        
        page.text_content.assert_called_once_with("#element")
        assert result == "Text content: element text"

    @pytest.mark.asyncio
    async def test_get_html_no_selector(self):
        """Test get HTML without selector."""
        page = AsyncMock()
        page.content = AsyncMock(return_value="<html><body>content</body></html>")
        
        result = await BrowserApiMethods.get_html(page)
        
        page.content.assert_called_once()
        assert result == ("<html><body>content</body></html>",)

    @pytest.mark.asyncio
    async def test_get_html_with_selector(self):
        """Test get HTML with selector."""
        page = AsyncMock()
        page.wait_for_selector = AsyncMock()
        page.inner_html = AsyncMock(return_value="<div>inner content</div>")
        
        result = await BrowserApiMethods.get_html(page, "#element")
        
        page.wait_for_selector.assert_called_once_with("#element", timeout=5000)
        page.inner_html.assert_called_once_with("#element")
        assert result == ("<div>inner content</div>",)

    @pytest.mark.asyncio
    async def test_get_html_with_selector_timeout(self):
        """Test get HTML with selector timeout."""
        page = AsyncMock()
        page.wait_for_selector.side_effect = PlaywrightTimeoutError("Timeout")
        
        with pytest.raises(ValueError, match="Element with selector.*not found"):
            await BrowserApiMethods.get_html(page, "#nonexistent")

    @pytest.mark.asyncio
    async def test_get_html_long_content_truncation(self):
        """Test get HTML with long content truncation."""
        page = AsyncMock()
        long_content = "x" * 1500  # More than 1000 characters
        page.content = AsyncMock(return_value=long_content)
        
        result = await BrowserApiMethods.get_html(page)
        
        assert result == (long_content[:1000] + "...",)

    @pytest.mark.asyncio
    async def test_screenshot_default_path(self):
        """Test screenshot with default path."""
        page = AsyncMock()
        page.screenshot = AsyncMock()
        
        with patch("os.makedirs") as mock_makedirs, \
             patch("time.time", return_value=1234567890), \
             patch.dict(os.environ, {"STRANDS_BROWSER_SCREENSHOTS_DIR": "test_screenshots"}):
            
            result = await BrowserApiMethods.screenshot(page)
            
            mock_makedirs.assert_called_once_with("test_screenshots", exist_ok=True)
            expected_path = os.path.join("test_screenshots", "screenshot_1234567890.png")
            page.screenshot.assert_called_once_with(path=expected_path)
            assert result == f"Screenshot saved as {expected_path}"

    @pytest.mark.asyncio
    async def test_screenshot_custom_path(self):
        """Test screenshot with custom path."""
        page = AsyncMock()
        page.screenshot = AsyncMock()
        
        with patch("os.makedirs") as mock_makedirs, \
             patch.dict(os.environ, {"STRANDS_BROWSER_SCREENSHOTS_DIR": "test_screenshots"}):
            
            result = await BrowserApiMethods.screenshot(page, "custom.png")
            
            mock_makedirs.assert_called_once_with("test_screenshots", exist_ok=True)
            expected_path = os.path.join("test_screenshots", "custom.png")
            page.screenshot.assert_called_once_with(path=expected_path)
            assert result == f"Screenshot saved as {expected_path}"

    @pytest.mark.asyncio
    async def test_screenshot_absolute_path(self):
        """Test screenshot with absolute path."""
        page = AsyncMock()
        page.screenshot = AsyncMock()
        absolute_path = "/tmp/screenshot.png"
        
        with patch("os.makedirs") as mock_makedirs, \
             patch("os.path.isabs", return_value=True):
            
            result = await BrowserApiMethods.screenshot(page, absolute_path)
            
            mock_makedirs.assert_called_once_with("screenshots", exist_ok=True)
            page.screenshot.assert_called_once_with(path=absolute_path)
            assert result == f"Screenshot saved as {absolute_path}"

    @pytest.mark.asyncio
    async def test_refresh(self):
        """Test refresh action."""
        page = AsyncMock()
        page.reload = AsyncMock()
        page.wait_for_load_state = AsyncMock()
        
        result = await BrowserApiMethods.refresh(page)
        
        page.reload.assert_called_once()
        page.wait_for_load_state.assert_called_once_with("networkidle")
        assert result == "Page refreshed"

    @pytest.mark.asyncio
    async def test_back(self):
        """Test back navigation."""
        page = AsyncMock()
        page.go_back = AsyncMock()
        page.wait_for_load_state = AsyncMock()
        
        result = await BrowserApiMethods.back(page)
        
        page.go_back.assert_called_once()
        page.wait_for_load_state.assert_called_once_with("networkidle")
        assert result == "Navigated back"

    @pytest.mark.asyncio
    async def test_forward(self):
        """Test forward navigation."""
        page = AsyncMock()
        page.go_forward = AsyncMock()
        page.wait_for_load_state = AsyncMock()
        
        result = await BrowserApiMethods.forward(page)
        
        page.go_forward.assert_called_once()
        page.wait_for_load_state.assert_called_once_with("networkidle")
        assert result == "Navigated forward"

    @pytest.mark.asyncio
    async def test_new_tab_default_id(self):
        """Test creating new tab with default ID."""
        page = AsyncMock()
        browser_manager = Mock()
        browser_manager._tabs = {}
        browser_manager._context = AsyncMock()
        new_page = AsyncMock()
        browser_manager._context.new_page = AsyncMock(return_value=new_page)
        
        with patch.object(BrowserApiMethods, 'switch_tab', return_value="switched") as mock_switch:
            result = await BrowserApiMethods.new_tab(page, browser_manager)
            
            browser_manager._context.new_page.assert_called_once()
            assert "tab_1" in browser_manager._tabs
            assert browser_manager._tabs["tab_1"] == new_page
            mock_switch.assert_called_once_with(new_page, browser_manager, "tab_1")
            assert result == "Created new tab with ID: tab_1"

    @pytest.mark.asyncio
    async def test_new_tab_custom_id(self):
        """Test creating new tab with custom ID."""
        page = AsyncMock()
        browser_manager = Mock()
        browser_manager._tabs = {}
        browser_manager._context = AsyncMock()
        new_page = AsyncMock()
        browser_manager._context.new_page = AsyncMock(return_value=new_page)
        
        with patch.object(BrowserApiMethods, 'switch_tab', return_value="switched") as mock_switch:
            result = await BrowserApiMethods.new_tab(page, browser_manager, "custom_tab")
            
            assert "custom_tab" in browser_manager._tabs
            assert browser_manager._tabs["custom_tab"] == new_page
            mock_switch.assert_called_once_with(new_page, browser_manager, "custom_tab")
            assert result == "Created new tab with ID: custom_tab"

    @pytest.mark.asyncio
    async def test_new_tab_existing_id(self):
        """Test creating new tab with existing ID."""
        page = AsyncMock()
        browser_manager = Mock()
        browser_manager._tabs = {"existing_tab": AsyncMock()}
        
        result = await BrowserApiMethods.new_tab(page, browser_manager, "existing_tab")
        
        assert result == "Error: Tab with ID existing_tab already exists"

    @pytest.mark.asyncio
    async def test_switch_tab_success(self):
        """Test successful tab switching."""
        page = AsyncMock()
        browser_manager = Mock()
        target_page = AsyncMock()
        target_page.context = AsyncMock()
        cdp_session = AsyncMock()
        target_page.context.new_cdp_session = AsyncMock(return_value=cdp_session)
        cdp_session.send = AsyncMock()
        
        browser_manager._tabs = {"target_tab": target_page}
        
        result = await BrowserApiMethods.switch_tab(page, browser_manager, "target_tab")
        
        assert browser_manager._page == target_page
        assert browser_manager._cdp_client == cdp_session
        assert browser_manager._active_tab_id == "target_tab"
        cdp_session.send.assert_called_once_with("Page.bringToFront")
        assert result == "Switched to tab: target_tab"

    @pytest.mark.asyncio
    async def test_switch_tab_no_id(self):
        """Test tab switching without ID."""
        page = AsyncMock()
        browser_manager = Mock()
        browser_manager._tabs = {}
        
        with patch.object(BrowserApiMethods, '_get_tab_info_for_logs', return_value={}):
            with pytest.raises(ValueError, match="tab_id is required"):
                await BrowserApiMethods.switch_tab(page, browser_manager, "")

    @pytest.mark.asyncio
    async def test_switch_tab_not_found(self):
        """Test tab switching with non-existent tab."""
        page = AsyncMock()
        browser_manager = Mock()
        browser_manager._tabs = {}
        
        with patch.object(BrowserApiMethods, '_get_tab_info_for_logs', return_value={}):
            with pytest.raises(ValueError, match="Tab with ID.*not found"):
                await BrowserApiMethods.switch_tab(page, browser_manager, "nonexistent")

    @pytest.mark.asyncio
    async def test_switch_tab_cdp_error(self):
        """Test tab switching with CDP error."""
        page = AsyncMock()
        browser_manager = Mock()
        target_page = AsyncMock()
        target_page.context = AsyncMock()
        cdp_session = AsyncMock()
        target_page.context.new_cdp_session = AsyncMock(return_value=cdp_session)
        cdp_session.send = AsyncMock(side_effect=Exception("CDP error"))
        
        browser_manager._tabs = {"target_tab": target_page}
        
        with patch("src.strands_tools.use_browser.logger") as mock_logger:
            result = await BrowserApiMethods.switch_tab(page, browser_manager, "target_tab")
            
            mock_logger.warning.assert_called_once()
            assert result == "Switched to tab: target_tab"

    @pytest.mark.asyncio
    async def test_close_tab_specific_id(self):
        """Test closing specific tab."""
        page = AsyncMock()
        browser_manager = Mock()
        tab_to_close = AsyncMock()
        tab_to_close.close = AsyncMock()
        browser_manager._tabs = {"tab1": tab_to_close, "tab2": AsyncMock()}
        browser_manager._active_tab_id = "tab2"
        
        result = await BrowserApiMethods.close_tab(page, browser_manager, "tab1")
        
        tab_to_close.close.assert_called_once()
        assert "tab1" not in browser_manager._tabs
        assert "tab2" in browser_manager._tabs
        assert result == "Closed tab: tab1"

    @pytest.mark.asyncio
    async def test_close_tab_active_tab(self):
        """Test closing active tab with other tabs available."""
        page = AsyncMock()
        browser_manager = Mock()
        active_tab = AsyncMock()
        active_tab.close = AsyncMock()
        other_tab = AsyncMock()
        browser_manager._tabs = {"active": active_tab, "other": other_tab}
        browser_manager._active_tab_id = "active"
        
        with patch.object(BrowserApiMethods, 'switch_tab', return_value="switched") as mock_switch:
            result = await BrowserApiMethods.close_tab(page, browser_manager, "active")
            
            active_tab.close.assert_called_once()
            assert "active" not in browser_manager._tabs
            mock_switch.assert_called_once_with(page, browser_manager, "other")
            assert result == "Closed tab: active"

    @pytest.mark.asyncio
    async def test_close_tab_last_tab(self):
        """Test closing the last tab."""
        page = AsyncMock()
        browser_manager = Mock()
        last_tab = AsyncMock()
        last_tab.close = AsyncMock()
        browser_manager._tabs = {"last": last_tab}
        browser_manager._active_tab_id = "last"
        
        result = await BrowserApiMethods.close_tab(page, browser_manager, "last")
        
        last_tab.close.assert_called_once()
        assert browser_manager._tabs == {}
        assert browser_manager._page is None
        assert browser_manager._cdp_client is None
        assert browser_manager._active_tab_id is None
        assert result == "Closed tab: last"

    @pytest.mark.asyncio
    async def test_close_tab_not_found(self):
        """Test closing non-existent tab."""
        page = AsyncMock()
        browser_manager = Mock()
        browser_manager._tabs = {"existing": AsyncMock()}
        
        with pytest.raises(ValueError, match="Tab with ID.*not found"):
            await BrowserApiMethods.close_tab(page, browser_manager, "nonexistent")

    @pytest.mark.asyncio
    async def test_list_tabs(self):
        """Test listing tabs."""
        page = AsyncMock()
        browser_manager = Mock()
        
        with patch.object(BrowserApiMethods, '_get_tab_info_for_logs', return_value={"tab1": {"url": "https://example.com", "active": True}}):
            result = await BrowserApiMethods.list_tabs(page, browser_manager)
            
            expected = json.dumps({"tab1": {"url": "https://example.com", "active": True}}, indent=2)
            assert result == expected

    @pytest.mark.asyncio
    async def test_get_cookies(self):
        """Test getting cookies."""
        page = AsyncMock()
        cookies = [{"name": "test", "value": "value"}]
        page.context = AsyncMock()
        page.context.cookies = AsyncMock(return_value=cookies)
        
        result = await BrowserApiMethods.get_cookies(page)
        
        expected = json.dumps(cookies, indent=2)
        assert result == expected

    @pytest.mark.asyncio
    async def test_set_cookies(self):
        """Test setting cookies."""
        page = AsyncMock()
        cookies = [{"name": "test", "value": "value"}]
        page.context = AsyncMock()
        page.context.add_cookies = AsyncMock()
        
        result = await BrowserApiMethods.set_cookies(page, cookies)
        
        page.context.add_cookies.assert_called_once_with(cookies)
        assert result == "Cookies set successfully"

    @pytest.mark.asyncio
    async def test_network_intercept(self):
        """Test network interception."""
        page = AsyncMock()
        page.route = AsyncMock()
        
        result = await BrowserApiMethods.network_intercept(page, "**/*.js")
        
        page.route.assert_called_once()
        assert result == "Network interception set for **/*.js"

    @pytest.mark.asyncio
    async def test_execute_cdp(self):
        """Test CDP execution."""
        page = AsyncMock()
        page.context = AsyncMock()
        cdp_client = AsyncMock()
        page.context.new_cdp_session = AsyncMock(return_value=cdp_client)
        cdp_result = {"result": "success"}
        cdp_client.send = AsyncMock(return_value=cdp_result)
        
        result = await BrowserApiMethods.execute_cdp(page, "Runtime.evaluate", {"expression": "1+1"})
        
        cdp_client.send.assert_called_once_with("Runtime.evaluate", {"expression": "1+1"})
        expected = json.dumps(cdp_result, indent=2)
        assert result == expected

    @pytest.mark.asyncio
    async def test_execute_cdp_no_params(self):
        """Test CDP execution without parameters."""
        page = AsyncMock()
        page.context = AsyncMock()
        cdp_client = AsyncMock()
        page.context.new_cdp_session = AsyncMock(return_value=cdp_client)
        cdp_result = {"result": "success"}
        cdp_client.send = AsyncMock(return_value=cdp_result)
        
        result = await BrowserApiMethods.execute_cdp(page, "Runtime.enable")
        
        cdp_client.send.assert_called_once_with("Runtime.enable", {})

    @pytest.mark.asyncio
    async def test_close(self):
        """Test browser close."""
        page = AsyncMock()
        browser_manager = AsyncMock()
        browser_manager.cleanup = AsyncMock()
        
        result = await BrowserApiMethods.close(page, browser_manager)
        
        browser_manager.cleanup.assert_called_once()
        assert result == "Browser closed"


class TestBrowserManager:
    """Test the BrowserManager class."""

    def test_init(self):
        """Test BrowserManager initialization."""
        manager = BrowserManager()
        
        assert manager._playwright is None
        assert manager._browser is None
        assert manager._context is None
        assert manager._page is None
        assert manager._cdp_client is None
        assert manager._tabs == {}
        assert manager._active_tab_id is None
        assert manager._nest_asyncio_applied is False
        assert isinstance(manager._actions, dict)
        assert "navigate" in manager._actions
        assert "click" in manager._actions

    def test_load_actions(self):
        """Test loading actions from BrowserApiMethods."""
        manager = BrowserManager()
        actions = manager._load_actions()
        
        # Should include public methods from BrowserApiMethods
        assert "navigate" in actions
        assert "click" in actions
        assert "type" in actions
        assert "screenshot" in actions
        
        # Should not include private methods
        assert "_get_tab_info_for_logs" not in actions

    @pytest.mark.asyncio
    async def test_ensure_browser_first_time(self):
        """Test ensuring browser for the first time."""
        manager = BrowserManager()
        
        with patch("nest_asyncio.apply") as mock_nest_asyncio, \
             patch("os.makedirs") as mock_makedirs, \
             patch("src.strands_tools.use_browser.async_playwright") as mock_playwright, \
             patch.dict(os.environ, {
                 "STRANDS_BROWSER_USER_DATA_DIR": "/tmp/browser",
                 "STRANDS_BROWSER_HEADLESS": "true",
                 "STRANDS_BROWSER_WIDTH": "1920",
                 "STRANDS_BROWSER_HEIGHT": "1080"
             }):
            
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()
            mock_cdp = AsyncMock()
            
            mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)
            mock_context.new_cdp_session = AsyncMock(return_value=mock_cdp)
            
            await manager.ensure_browser()
            
            mock_nest_asyncio.assert_called_once()
            mock_makedirs.assert_called_once_with("/tmp/browser", exist_ok=True)
            assert manager._nest_asyncio_applied is True
            assert manager._playwright == mock_playwright_instance
            assert manager._browser == mock_browser
            assert manager._context == mock_context
            assert manager._page == mock_page

    @pytest.mark.asyncio
    async def test_ensure_browser_already_applied_nest_asyncio(self):
        """Test ensuring browser when nest_asyncio already applied."""
        manager = BrowserManager()
        manager._nest_asyncio_applied = True
        
        with patch("nest_asyncio.apply") as mock_nest_asyncio, \
             patch("src.strands_tools.use_browser.async_playwright") as mock_playwright:
            
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()
            
            mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)
            
            await manager.ensure_browser()
            
            mock_nest_asyncio.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_browser_with_launch_options(self):
        """Test ensuring browser with custom launch options."""
        manager = BrowserManager()
        
        launch_options = {"headless": False, "slowMo": 100}
        
        with patch("nest_asyncio.apply"), \
             patch("os.makedirs"), \
             patch("src.strands_tools.use_browser.async_playwright") as mock_playwright:
            
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()
            
            mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_browser.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=mock_page)
            
            await manager.ensure_browser(launch_options=launch_options)
            
            # Check that launch was called with merged options
            call_args = mock_playwright_instance.chromium.launch.call_args[1]
            assert call_args["headless"] is False
            assert call_args["slowMo"] == 100

    @pytest.mark.asyncio
    async def test_ensure_browser_persistent_context(self):
        """Test ensuring browser with persistent context."""
        manager = BrowserManager()
        
        launch_options = {"persistent_context": True, "user_data_dir": "/custom/path"}
        
        with patch("nest_asyncio.apply"), \
             patch("os.makedirs"), \
             patch("src.strands_tools.use_browser.async_playwright") as mock_playwright:
            
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)
            mock_context = AsyncMock()
            mock_page = AsyncMock()
            
            mock_playwright_instance.chromium.launch_persistent_context = AsyncMock(return_value=mock_context)
            mock_context.pages = [mock_page]
            mock_context.new_cdp_session = AsyncMock()
            
            await manager.ensure_browser(launch_options=launch_options)
            
            mock_playwright_instance.chromium.launch_persistent_context.assert_called_once()
            call_args = mock_playwright_instance.chromium.launch_persistent_context.call_args
            assert call_args[1]["user_data_dir"] == "/custom/path"
            assert manager._context == mock_context

    @pytest.mark.asyncio
    async def test_ensure_browser_already_initialized(self):
        """Test ensuring browser when already initialized."""
        manager = BrowserManager()
        manager._playwright = AsyncMock()
        manager._page = AsyncMock()  # Set page to avoid the error
        manager._nest_asyncio_applied = True  # Mark as already applied
        
        with patch("nest_asyncio.apply") as mock_nest_asyncio:
            await manager.ensure_browser()
            
            # Should not apply nest_asyncio again
            mock_nest_asyncio.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_full(self):
        """Test full cleanup with all resources."""
        manager = BrowserManager()
        mock_page = AsyncMock()
        mock_context = AsyncMock()
        mock_browser = AsyncMock()
        mock_playwright = AsyncMock()
        
        manager._page = mock_page
        manager._context = mock_context
        manager._browser = mock_browser
        manager._playwright = mock_playwright
        manager._tabs = {"tab1": AsyncMock(), "tab2": AsyncMock()}
        
        await manager.cleanup()
        
        mock_page.close.assert_called_once()
        mock_context.close.assert_called_once()
        mock_browser.close.assert_called_once()
        mock_playwright.stop.assert_called_once()
        
        assert manager._page is None
        assert manager._context is None
        assert manager._browser is None
        assert manager._playwright is None
        assert manager._tabs == {}

    @pytest.mark.asyncio
    async def test_cleanup_partial(self):
        """Test cleanup with only some resources."""
        manager = BrowserManager()
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        
        manager._page = mock_page
        manager._browser = mock_browser
        
        await manager.cleanup()
        
        mock_page.close.assert_called_once()
        mock_browser.close.assert_called_once()
        assert manager._page is None
        assert manager._browser is None

    @pytest.mark.asyncio
    async def test_cleanup_with_errors(self):
        """Test cleanup with errors during cleanup."""
        manager = BrowserManager()
        mock_page = AsyncMock()
        mock_browser = AsyncMock()
        mock_page.close.side_effect = Exception("Close error")
        
        manager._page = mock_page
        manager._browser = mock_browser
        
        # Should not raise exception
        await manager.cleanup()
        
        mock_page.close.assert_called_once()
        mock_browser.close.assert_called_once()
        assert manager._page is None
        assert manager._browser is None

    @pytest.mark.asyncio
    async def test_get_tab_info_for_logs(self):
        """Test getting tab info for logs."""
        manager = BrowserManager()
        page1 = AsyncMock()
        page1.url = "https://example.com"
        page2 = AsyncMock()
        page2.url = "https://test.com"
        
        manager._tabs = {"tab1": page1, "tab2": page2}
        manager._active_tab_id = "tab1"
        
        result = await BrowserApiMethods._get_tab_info_for_logs(manager)
        
        expected = {
            "tab1": {"url": "https://example.com", "active": True},
            "tab2": {"url": "https://test.com", "active": False}
        }
        assert result == expected

    @pytest.mark.asyncio
    async def test_get_tab_info_for_logs_with_error(self):
        """Test getting tab info with error."""
        manager = BrowserManager()
        page1 = Mock()
        # Create a property that raises an exception when accessed
        type(page1).url = property(lambda self: (_ for _ in ()).throw(Exception("URL error")))
        
        manager._tabs = {"tab1": page1}
        manager._active_tab_id = "tab1"
        
        result = await BrowserApiMethods._get_tab_info_for_logs(manager)
        
        assert "tab1" in result
        assert "error" in result["tab1"]
        assert "Could not retrieve tab info" in result["tab1"]["error"]


class TestUseBrowserFunction:
    """Test the main use_browser function."""

    def test_use_browser_bypass_consent(self):
        """Test use_browser with bypassed consent."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}), \
             patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
            
            mock_manager._loop = MagicMock()
            mock_manager._loop.run_until_complete.return_value = [{"text": "Success"}]
            
            result = use_browser(action="navigate", url="https://example.com")
            
            # The function returns a string, not a dict
            assert isinstance(result, str)
            assert "Success" in result

    def test_use_browser_user_consent_yes(self):
        """Test use_browser with user consent."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "false"}), \
             patch("src.strands_tools.use_browser.get_user_input", return_value="y"), \
             patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
            
            mock_manager._loop = MagicMock()
            mock_manager._loop.run_until_complete.return_value = [{"text": "Success"}]
            
            result = use_browser(action="navigate", url="https://example.com")
            
            assert isinstance(result, str)
            assert "Success" in result

    def test_use_browser_user_consent_no(self):
        """Test use_browser with user denial."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "false"}), \
             patch("src.strands_tools.use_browser.get_user_input", return_value="n"):
            
            result = use_browser(action="navigate", url="https://example.com")
            
            # The @tool decorator returns a dict format for errors
            assert isinstance(result, dict)
            assert result["status"] == "error"
            assert "cancelled" in result["content"][0]["text"].lower()

    def test_use_browser_invalid_action(self):
        """Test use_browser with invalid action."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}), \
             patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
            
            mock_manager._loop = MagicMock()
            # Mock both calls - first for the action, second for cleanup
            mock_manager._loop.run_until_complete.side_effect = [Exception("Invalid action"), None]
            
            result = use_browser(action="invalid_action")
            
            assert isinstance(result, str)
            assert "Error:" in result
            assert "Invalid action" in result

    def test_use_browser_manager_initialization(self):
        """Test browser manager initialization."""
        with patch("src.strands_tools.use_browser.BrowserManager") as mock_browser_manager_class:
            mock_manager = MagicMock()
            mock_browser_manager_class.return_value = mock_manager
            
            # Import should trigger manager creation
            from src.strands_tools.use_browser import _playwright_manager
            
            # The manager should be created when the module is imported
            assert _playwright_manager is not None

    def test_use_browser_with_multiple_parameters(self):
        """Test use_browser with multiple parameters."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}), \
             patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
            
            mock_manager._loop = MagicMock()
            mock_manager._loop.run_until_complete.return_value = [{"text": "Success"}]
            
            result = use_browser(
                action="type",
                selector="#input",
                input_text="test text",
                url="https://example.com",
                wait_time=2
            )
            
            assert isinstance(result, str)
            assert "Success" in result

    def test_use_browser_exception_handling(self):
        """Test use_browser exception handling."""
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}), \
             patch("src.strands_tools.use_browser._playwright_manager") as mock_manager:
            
            mock_manager._loop = MagicMock()
            # Mock both calls - first for the action, second for cleanup
            mock_manager._loop.run_until_complete.side_effect = [RuntimeError("Test error"), None]
            
            result = use_browser(action="navigate", url="https://example.com")
            
            assert isinstance(result, str)
            assert "Error:" in result
            assert "Test error" in result


class TestBrowserManagerJavaScriptFixes:
    """Test JavaScript syntax fixing functionality."""

    @pytest.mark.asyncio
    async def test_fix_javascript_syntax_illegal_return(self):
        """Test fixing illegal return statement."""
        manager = BrowserManager()
        
        script = "return document.title;"
        error = "Illegal return statement"
        
        result = await manager._fix_javascript_syntax(script, error)
        
        assert result == "(function() { return document.title; })()"

    @pytest.mark.asyncio
    async def test_fix_javascript_syntax_template_literals(self):
        """Test fixing template literals."""
        manager = BrowserManager()
        
        script = "console.log(`Hello ${name}!`);"
        error = "Unexpected token '`'"
        
        result = await manager._fix_javascript_syntax(script, error)
        
        assert result == "console.log('Hello ' + name + '!');"

    @pytest.mark.asyncio
    async def test_fix_javascript_syntax_arrow_function(self):
        """Test fixing arrow functions."""
        manager = BrowserManager()
        
        script = "const add = (a, b) => a + b;"
        error = "Unexpected token '=>'"
        
        result = await manager._fix_javascript_syntax(script, error)
        
        assert result == "const add = (a, b) function() { return  a + b; }"

    @pytest.mark.asyncio
    async def test_fix_javascript_syntax_missing_brace(self):
        """Test fixing missing closing brace."""
        manager = BrowserManager()
        
        script = "function test() { console.log('Hello')"
        error = "Unexpected end of input"
        
        result = await manager._fix_javascript_syntax(script, error)
        
        assert result == "function test() { console.log('Hello')}"

    @pytest.mark.asyncio
    async def test_fix_javascript_syntax_undefined_variable(self):
        """Test fixing undefined variable."""
        manager = BrowserManager()
        
        script = "console.log(undefinedVar);"
        error = "'undefinedVar' is not defined"
        
        result = await manager._fix_javascript_syntax(script, error)
        
        assert result == "var undefinedVar = undefined;\nconsole.log(undefinedVar);"

    @pytest.mark.asyncio
    async def test_fix_javascript_syntax_no_fix_needed(self):
        """Test when no fix is needed."""
        manager = BrowserManager()
        
        script = "console.log('Hello');"
        error = "Some other error"
        
        result = await manager._fix_javascript_syntax(script, error)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fix_javascript_syntax_empty_inputs(self):
        """Test with empty inputs."""
        manager = BrowserManager()
        
        # Empty script
        result = await manager._fix_javascript_syntax("", "error")
        assert result is None
        
        # Empty error
        result = await manager._fix_javascript_syntax("script", "")
        assert result is None
        
        # Both empty
        result = await manager._fix_javascript_syntax("", "")
        assert result is None
        
        # None inputs
        result = await manager._fix_javascript_syntax(None, "error")
        assert result is None
        
        result = await manager._fix_javascript_syntax("script", None)
        assert result is None