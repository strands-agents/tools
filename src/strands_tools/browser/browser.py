"""
Browser Tool implementation using Strands @tool decorator with Playwright.

This module contains the base browser tool class that provides a concrete
Playwright implementation that can be used directly or extended by specific
platform implementations.
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import nest_asyncio
from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from strands import tool

from .models import (
    BackAction,
    BrowserInput,
    BrowserSession,
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

logger = logging.getLogger(__name__)


class Browser(ABC):
    """Browser tool implementation using Playwright."""

    def __init__(self):
        self._started = False
        self._playwright = None
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._nest_asyncio_applied = False
        self._sessions: Dict[str, BrowserSession] = {}

    @tool
    def browser(self, browser_input: BrowserInput) -> Dict[str, Any]:
        """
        Browser automation tool for web scraping, testing, and automation tasks.

        This tool provides comprehensive browser automation capabilities using Playwright
        with support for multiple browser engines. It offers session management, tab control,
        page interactions, content extraction, and advanced automation features.

        Usage with Strands Agent:
        ```python
        from strands import Agent
        from strands_tools.browser import Browser

        # Create the browser tool
        browser = Browser()
        agent = Agent(tools=[browser.browser])

        # Initialize a session
        agent.tool.browser(
            browser_input={
                "action": {
                    "type": "init_session",
                    "description": "Example ession",
                    "session_name": "example-session"
                }
            }
        )

        # Navigate to a page
        agent.tool.browser(
            browser_input={
                "action": {
                    "type": "navigate",
                    "url": "https://example.com",
                    "session_name": "example-session"
                }
            }
        )

        # Close the browser
        agent.tool.browser(
            browser_input={
                "action": {
                    "type": "close",
                    "session_name": "example-session"
                }
            }
        )
        ```

        Args:
            browser_input: Structured input containing the action to perform.

        Returns:
            Dict containing execution results.
        """
        # Auto-start platform on first use
        if not self._started:
            self._start()

        if isinstance(browser_input, dict):
            logger.debug("Action was passed as Dict, mapping to BrowserInput type action")
            action = BrowserInput.model_validate(browser_input).action
        else:
            action = browser_input.action

        logger.debug(f"processing browser action {type(action)}")

        # Delegate to specific action handlers
        if isinstance(action, InitSessionAction):
            return self.init_session(action)
        elif isinstance(action, ListLocalSessionsAction):
            return self.list_local_sessions()
        elif isinstance(action, NavigateAction):
            return self.navigate(action)
        elif isinstance(action, ClickAction):
            return self.click(action)
        elif isinstance(action, TypeAction):
            return self.type(action)
        elif isinstance(action, GetTextAction):
            return self.get_text(action)
        elif isinstance(action, GetHtmlAction):
            return self.get_html(action)
        elif isinstance(action, ScreenshotAction):
            return self.screenshot(action)
        elif isinstance(action, NewTabAction):
            return self.new_tab(action)
        elif isinstance(action, SwitchTabAction):
            return self.switch_tab(action)
        elif isinstance(action, CloseTabAction):
            return self.close_tab(action)
        elif isinstance(action, ListTabsAction):
            return self.list_tabs(action)
        elif isinstance(action, BackAction):
            return self.back(action)
        elif isinstance(action, ForwardAction):
            return self.forward(action)
        elif isinstance(action, RefreshAction):
            return self.refresh(action)
        elif isinstance(action, EvaluateAction):
            return self.evaluate(action)
        elif isinstance(action, GetCookiesAction):
            return self.get_cookies(action)
        elif isinstance(action, SetCookiesAction):
            return self.set_cookies(action)
        elif isinstance(action, NetworkInterceptAction):
            return self.network_intercept(action)
        elif isinstance(action, ExecuteCdpAction):
            return self.execute_cdp(action)
        elif isinstance(action, CloseAction):
            return self.close(action)
        else:
            return {"status": "error", "content": [{"text": f"Unknown action type: {type(action)}"}]}

    def _start(self) -> None:
        """Start the platform and initialize any required connections."""
        if not self._started:
            self._playwright = self._execute_async(async_playwright().start())
            self.start_platform()
            self._started = True

    def _cleanup(self) -> None:
        """Clean up platform resources and connections."""
        if self._started:
            self._execute_async(self._async_cleanup())
            self._started = False

    def __del__(self):
        """Cleanup: Clear platform resources when tool is destroyed."""
        try:
            logger.debug("browser tool destructor called - cleaning up platform")
            self._cleanup()
            logger.debug("platform cleanup completed successfully")
        except Exception as e:
            logger.debug("exception=<%s> | platform cleanup during destruction skipped", str(e))

    @abstractmethod
    def start_platform(self) -> None:
        """Initialize platform-specific resources and establish browser connection."""
        ...

    @abstractmethod
    def close_platform(self) -> None:
        """Close platform-specific resources."""
        ...

    @abstractmethod
    async def create_browser_session(self) -> PlaywrightBrowser:
        """Create a new browser instance for a session.

        This method must be implemented by all platform-specific subclasses.
        It should return a Playwright Browser instance that will be used for
        creating new browser sessions.

        Returns:
            Browser: A Playwright Browser instance
        """
        ...

    async def _setup_session_from_browser(self, browser_or_context):
        """Setup session components from browser or context.

        This method can be overridden by subclasses to customize how
        browser, context, and page are extracted from the session object.

        Args:
            browser_or_context: The object returned by create_browser_session()

        Returns:
            Tuple of (browser, context, page)
        """
        if isinstance(browser_or_context, PlaywrightBrowser):
            # Normal non-persistent case
            session_browser = browser_or_context
            session_context = await session_browser.new_context()
            session_page = await session_context.new_page()
        else:
            # Persistent context case
            session_context = browser_or_context
            session_browser = session_context.browser
            session_page = await session_context.new_page()

        return session_browser, session_context, session_page

    # Session Management Methods
    def init_session(self, action: InitSessionAction) -> Dict[str, Any]:
        """Initialize a new browser session."""
        return self._execute_async(self._async_init_session(action))

    async def _async_init_session(self, action: InitSessionAction) -> Dict[str, Any]:
        """Async initialize session implementation."""
        logger.info(f"initializing browser session: {action.description}")

        session_name = action.session_name

        # Check if session already exists
        if session_name in self._sessions:
            return {"status": "error", "content": [{"text": f"Session '{session_name}' already exists"}]}

        try:
            # Create new browser instance for this session
            browser_or_context = await self.create_browser_session()

            # Let subclasses customize how to extract browser, context, and page
            session_browser, session_context, session_page = await self._setup_session_from_browser(browser_or_context)

            # Create and store session object
            session = BrowserSession(
                session_name=session_name,
                description=action.description,
                browser=session_browser,
                context=session_context,
                page=session_page,
            )
            session.add_tab("main", session_page)

            self._sessions[session_name] = session

            logger.info(f"initialized session: {session_name}")

            return {
                "status": "success",
                "content": [
                    {
                        "json": {
                            "sessionName": session_name,
                            "description": action.description,
                        }
                    }
                ],
            }

        except Exception as e:
            logger.debug("exception=<%s> | failed to initialize session", str(e))
            return {"status": "error", "content": [{"text": f"Failed to initialize session: {str(e)}"}]}

    def list_local_sessions(self) -> Dict[str, Any]:
        """List all sessions created by this platform instance."""
        sessions_info = []
        for session_name, session in self._sessions.items():
            sessions_info.append(
                {
                    "sessionName": session_name,
                    "description": session.description,
                }
            )

        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "sessions": sessions_info,
                        "totalSessions": len(sessions_info),
                    }
                }
            ],
        }

    def get_session_page(self, session_name: str) -> Optional[Page]:
        """Get the active page for a session."""
        session = self._sessions.get(session_name)
        if session:
            return session.get_active_page()
        return None

    def validate_session(self, session_name: str) -> Optional[Dict[str, Any]]:
        """Validate that a session exists and return error response if not."""
        if session_name not in self._sessions:
            return {"status": "error", "content": [{"text": f"Session '{session_name}' not found"}]}
        return None

    # Shared browser action implementations
    def navigate(self, action: NavigateAction) -> Dict[str, Any]:
        """Navigate to a URL."""
        return self._execute_async(self._async_navigate(action))

    async def _async_navigate(self, action: NavigateAction) -> Dict[str, Any]:
        """Async navigate implementation."""
        logger.info(f"navigating using: {action}")

        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            await page.goto(action.url)
            await page.wait_for_load_state("networkidle")
            return {"status": "success", "content": [{"text": f"Navigated to {action.url}"}]}
        except Exception as e:
            error_str = str(e)
            if "ERR_NAME_NOT_RESOLVED" in error_str:
                error_msg = (
                    f"Could not resolve domain '{action.url}'. "
                    "The website might not exist or a network connectivity issue."
                )
            elif "ERR_CONNECTION_REFUSED" in error_str:
                error_msg = f"Connection refused for '{action.url}'. The server might be down or blocking requests."
            elif "ERR_CONNECTION_TIMED_OUT" in error_str:
                error_msg = f"Connection timed out for '{action.url}'. The server might be slow or unreachable."
            elif "ERR_SSL_PROTOCOL_ERROR" in error_str:
                error_msg = (
                    f"SSL/TLS error when connecting to '{action.url}'. "
                    "The site might have an invalid or expired certificate."
                )
            elif "ERR_CERT_" in error_str:
                error_msg = (
                    f"Certificate error when connecting to '{action.url}'. "
                    "The site's security certificate might be invalid."
                )
            else:
                error_msg = str(e)
            return {"status": "error", "content": [{"text": f"Error: {error_msg}"}]}

    def click(self, action: ClickAction) -> Dict[str, Any]:
        """Click on an element."""
        return self._execute_async(self._async_click(action))

    async def _async_click(self, action: ClickAction) -> Dict[str, Any]:
        """Async click implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            await page.click(action.selector)
            return {"status": "success", "content": [{"text": f"Clicked element: {action.selector}"}]}
        except Exception as e:
            logger.debug("exception=<%s> | click action failed on selector '%s'", str(e), action.selector)
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def type(self, action: TypeAction) -> Dict[str, Any]:
        """Type text into an element."""
        return self._execute_async(self._async_type(action))

    async def _async_type(self, action: TypeAction) -> Dict[str, Any]:
        """Async type implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            await page.fill(action.selector, action.text)
            return {"status": "success", "content": [{"text": f"Typed '{action.text}' into {action.selector}"}]}
        except Exception as e:
            logger.debug("exception=<%s> | type action failed on selector '%s'", str(e), action.selector)
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def evaluate(self, action: EvaluateAction) -> Dict[str, Any]:
        """Execute JavaScript code."""
        return self._execute_async(self._async_evaluate(action))

    async def _async_evaluate(self, action: EvaluateAction) -> Dict[str, Any]:
        """Async evaluate implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            result = await page.evaluate(action.script)
            return {"status": "success", "content": [{"text": f"Evaluation result: {result}"}]}
        except Exception as e:
            # Try to fix common JavaScript syntax errors
            fixed_script = await self._fix_javascript_syntax(action.script, str(e))
            if fixed_script:
                try:
                    result = await page.evaluate(fixed_script)
                    return {"status": "success", "content": [{"text": f"Evaluation result (fixed): {result}"}]}
                except Exception as e2:
                    logger.debug("exception=<%s> | evaluate action failed even after fix", str(e2))
                    return {"status": "error", "content": [{"text": f"Error: {str(e2)}"}]}
            logger.debug("exception=<%s> | evaluate action failed", str(e))
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    async def _fix_javascript_syntax(self, script: str, error_msg: str) -> Optional[str]:
        """Attempt to fix common JavaScript syntax errors."""
        if not script or not error_msg:
            return None

        fixed_script: Optional[str] = None

        # Handle illegal return statements
        if "Illegal return statement" in error_msg:
            fixed_script = f"(function() {{ {script} }})()"
            logger.info("Fixing 'Illegal return statement' by wrapping in function")

        # Handle unexpected token errors
        elif "Unexpected token" in error_msg:
            if "`" in script:  # Fix template literals
                fixed_script = script.replace("`", "'").replace("${", "' + ").replace("}", " + '")
                logger.info("Fixing template literals in script")
            elif "=>" in script:  # Fix arrow functions in old browsers
                fixed_script = script.replace("=>", "function() { return ")
                if not fixed_script.strip().endswith("}"):
                    fixed_script += " }"
                logger.info("Fixing arrow functions in script")

        # Handle missing braces/parentheses
        elif "Unexpected end of input" in error_msg:
            open_chars = script.count("{") + script.count("(") + script.count("[")
            close_chars = script.count("}") + script.count(")") + script.count("]")

            if open_chars > close_chars:
                missing = open_chars - close_chars
                fixed_script = script + ("}" * missing)
                logger.info(f"Added {missing} missing closing braces")

        # Handle uncaught reference errors
        elif "is not defined" in error_msg:
            var_name = error_msg.split("'")[1] if "'" in error_msg else ""
            if var_name:
                fixed_script = f"var {var_name} = undefined;\n{script}"
                logger.info(f"Adding undefined variable declaration for '{var_name}'")

        return fixed_script

    def press_key(self, action: PressKeyAction) -> Dict[str, Any]:
        """Press a keyboard key."""
        return self._execute_async(self._async_press_key(action))

    async def _async_press_key(self, action: PressKeyAction) -> Dict[str, Any]:
        """Async press key implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            await page.keyboard.press(action.key)
            return {"status": "success", "content": [{"text": f"Pressed key: {action.key}"}]}
        except Exception as e:
            logger.debug("exception=<%s> | press key action failed for key '%s'", str(e), action.key)
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def get_text(self, action: GetTextAction) -> Dict[str, Any]:
        """Get text content from an element."""
        return self._execute_async(self._async_get_text(action))

    async def _async_get_text(self, action: GetTextAction) -> Dict[str, Any]:
        """Async get text implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            text = await page.text_content(action.selector)
            return {"status": "success", "content": [{"text": f"Text content: {text}"}]}
        except Exception as e:
            logger.debug("exception=<%s> | get text action failed on selector '%s'", str(e), action.selector)
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def get_html(self, action: GetHtmlAction) -> Dict[str, Any]:
        """Get HTML content."""
        return self._execute_async(self._async_get_html(action))

    async def _async_get_html(self, action: GetHtmlAction) -> Dict[str, Any]:
        """Async get HTML implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            if not action.selector:
                result = await page.content()
            else:
                try:
                    await page.wait_for_selector(action.selector, timeout=5000)
                    result = await page.inner_html(action.selector)
                except PlaywrightTimeoutError:
                    logger.debug(
                        "exception=<%s> | get HTML action failed - selector '%s' not found",
                        "PlaywrightTimeoutError",
                        action.selector,
                    )
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"Element with selector '{action.selector}' not found on the page. "
                                    "Please verify the selector is correct."
                                )
                            }
                        ],
                    }

            # Truncate long HTML content
            truncated_result = result[:1000] + "..." if len(result) > 1000 else result
            return {"status": "success", "content": [{"text": truncated_result}]}
        except Exception as e:
            logger.debug("exception=<%s> | get HTML action failed", str(e))
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def screenshot(self, action: ScreenshotAction) -> Dict[str, Any]:
        """Take a screenshot."""
        logger.debug(f"Trying to screenshot {action}")
        return self._execute_async(self._async_screenshot(action))

    async def _async_screenshot(self, action: ScreenshotAction) -> Dict[str, Any]:
        """Async screenshot implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            logger.debug(f"No active page for session '{action.session_name}' to screenshot")
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            screenshots_dir = os.getenv("STRANDS_BROWSER_SCREENSHOTS_DIR", "screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)

            if not action.path:
                filename = f"screenshot_{int(time.time())}.png"
                path = os.path.join(screenshots_dir, filename)
            elif not os.path.isabs(action.path):
                path = os.path.join(screenshots_dir, action.path)
            else:
                path = action.path

            logger.debug(f"About to take screenshot with page: {page}")
            await page.screenshot(path=path)
            return {"status": "success", "content": [{"text": f"Screenshot saved as {path}"}]}
        except Exception as e:
            logger.debug("exception=<%s> | screenshot action failed", str(e))
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def refresh(self, action: RefreshAction) -> Dict[str, Any]:
        """Refresh the current page."""
        return self._execute_async(self._async_refresh(action))

    async def _async_refresh(self, action: RefreshAction) -> Dict[str, Any]:
        """Async refresh implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            await page.reload()
            await page.wait_for_load_state("networkidle")
            return {"status": "success", "content": [{"text": "Page refreshed"}]}
        except Exception as e:
            logger.debug("exception=<%s> | refresh action failed", str(e))
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def back(self, action: BackAction) -> Dict[str, Any]:
        """Navigate back in browser history."""
        return self._execute_async(self._async_back(action))

    async def _async_back(self, action: BackAction) -> Dict[str, Any]:
        """Async back implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            await page.go_back()
            await page.wait_for_load_state("networkidle")
            return {"status": "success", "content": [{"text": "Navigated back"}]}
        except Exception as e:
            logger.debug("exception=<%s> | back action failed", str(e))
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def forward(self, action: ForwardAction) -> Dict[str, Any]:
        """Navigate forward in browser history."""
        return self._execute_async(self._async_forward(action))

    async def _async_forward(self, action: ForwardAction) -> Dict[str, Any]:
        """Async forward implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            await page.go_forward()
            await page.wait_for_load_state("networkidle")
            return {"status": "success", "content": [{"text": "Navigated forward"}]}
        except Exception as e:
            logger.debug("exception=<%s> | forward action failed", str(e))
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def new_tab(self, action: NewTabAction) -> Dict[str, Any]:
        """Create a new browser tab."""
        return self._execute_async(self._async_new_tab(action))

    async def _async_new_tab(self, action: NewTabAction) -> Dict[str, Any]:
        """Async new tab implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        session = self._sessions.get(action.session_name)
        if not session:
            return {"status": "error", "content": [{"text": f"Session '{action.session_name}' not found"}]}

        try:
            tab_id = action.tab_id or f"tab_{len(session.tabs) + 1}"

            if tab_id in session.tabs:
                return {"status": "error", "content": [{"text": f"Tab with ID {tab_id} already exists"}]}

            new_page = await session.context.new_page()
            session.add_tab(tab_id, new_page)

            return {
                "status": "success",
                "content": [{"text": f"Created new tab with ID: {tab_id} and switched active tab to {tab_id}."}],
            }
        except Exception as e:
            logger.debug("exception=<%s> | new tab action failed", str(e))
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def switch_tab(self, action: SwitchTabAction) -> Dict[str, Any]:
        """Switch to a different tab."""
        return self._execute_async(self._async_switch_tab(action))

    async def _async_switch_tab(self, action: SwitchTabAction) -> Dict[str, Any]:
        """Async switch tab implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        session = self._sessions.get(action.session_name)
        if not session:
            return {"status": "error", "content": [{"text": f"Session '{action.session_name}' not found"}]}

        try:
            if action.tab_id not in session.tabs:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"Tab with ID '{action.tab_id}' not found. Available tabs: {list(session.tabs.keys())}"
                            )
                        }
                    ],
                }

            # Switch tab in session
            session.switch_tab(action.tab_id)

            # Bring the tab to the foreground
            page = session.get_active_page()
            if page:
                try:
                    await page.bring_to_front()
                    logger.info(f"Successfully switched to tab '{action.tab_id}' and brought it to the foreground")
                except Exception as e:
                    logger.debug("")
                    logger.warning(f"Failed to bring tab '{action.tab_id}' to foreground: {str(e)}")

            return {"status": "success", "content": [{"text": f"Switched to tab: {action.tab_id}"}]}
        except Exception as e:
            logger.debug("exception=<%s> | switch tab action failed", str(e))
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def close_tab(self, action: CloseTabAction) -> Dict[str, Any]:
        """Close a browser tab."""
        return self._execute_async(self._async_close_tab(action))

    async def _async_close_tab(self, action: CloseTabAction) -> Dict[str, Any]:
        """Async close tab implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        session = self._sessions.get(action.session_name)
        if not session:
            return {"status": "error", "content": [{"text": f"Session '{action.session_name}' not found"}]}

        try:
            tab_id = action.tab_id or session.active_tab_id

            if not tab_id or tab_id not in session.tabs:
                return {
                    "status": "error",
                    "content": [
                        {"text": f"Tab with ID '{tab_id}' not found. Available tabs: {list(session.tabs.keys())}"}
                    ],
                }

            # Close the tab
            await session.tabs[tab_id].close()
            session.remove_tab(tab_id)

            return {"status": "success", "content": [{"text": f"Closed tab: {tab_id}"}]}
        except Exception as e:
            logger.debug("exception=<%s> | close tab action failed", str(e))
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def list_tabs(self, action: ListTabsAction) -> Dict[str, Any]:
        """List all open browser tabs."""
        return self._execute_async(self._async_list_tabs(action))

    async def _async_list_tabs(self, action: ListTabsAction) -> Dict[str, Any]:
        """Async list tabs implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        session = self._sessions.get(action.session_name)
        if not session:
            return {"status": "error", "content": [{"text": f"Session '{action.session_name}' not found"}]}

        try:
            tabs_info = {}
            for tab_id, page in session.tabs.items():
                try:
                    is_active = tab_id == session.active_tab_id
                    tabs_info[tab_id] = {"url": page.url, "active": is_active}
                except Exception as e:
                    tabs_info[tab_id] = {"error": f"Could not retrieve tab info: {str(e)}"}

            return {"status": "success", "content": [{"text": json.dumps(tabs_info, indent=2)}]}
        except Exception as e:
            logger.debug("exception=<%s> | list tabs action failed", str(e))
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def get_cookies(self, action: GetCookiesAction) -> Dict[str, Any]:
        """Get all cookies for the current page."""
        return self._execute_async(self._async_get_cookies(action))

    async def _async_get_cookies(self, action: GetCookiesAction) -> Dict[str, Any]:
        """Async get cookies implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            cookies = await page.context.cookies()
            return {"status": "success", "content": [{"text": json.dumps(cookies, indent=2)}]}
        except Exception as e:
            logger.debug("exception=<%s> | get cookies action failed", str(e))
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def set_cookies(self, action: SetCookiesAction) -> Dict[str, Any]:
        """Set cookies for the current page."""
        return self._execute_async(self._async_set_cookies(action))

    async def _async_set_cookies(self, action: SetCookiesAction) -> Dict[str, Any]:
        """Async set cookies implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            await page.context.add_cookies(action.cookies)
            return {"status": "success", "content": [{"text": "Cookies set successfully"}]}
        except Exception as e:
            logger.debug("exception=<%s> | set cookies action failed", str(e))
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def network_intercept(self, action: NetworkInterceptAction) -> Dict[str, Any]:
        """Set up network request interception."""
        return self._execute_async(self._async_network_intercept(action))

    async def _async_network_intercept(self, action: NetworkInterceptAction) -> Dict[str, Any]:
        """Async network intercept implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            await page.route(action.pattern, lambda route: route.continue_())
            return {"status": "success", "content": [{"text": f"Network interception set for {action.pattern}"}]}
        except Exception as e:
            logger.debug("exception=<%s> | network intercept action failed", str(e))
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def execute_cdp(self, action: ExecuteCdpAction) -> Dict[str, Any]:
        """Execute Chrome DevTools Protocol command."""
        return self._execute_async(self._async_execute_cdp(action))

    async def _async_execute_cdp(self, action: ExecuteCdpAction) -> Dict[str, Any]:
        """Async execute CDP implementation."""
        # Validate session exists
        error_response = self.validate_session(action.session_name)
        if error_response:
            return error_response

        page = self.get_session_page(action.session_name)
        if not page:
            return {"status": "error", "content": [{"text": "Error: No active page for session"}]}

        try:
            cdp_session = await page.context.new_cdp_session(page)
            result = await cdp_session.send(action.method, action.params or {})
            return {"status": "success", "content": [{"text": json.dumps(result, indent=2)}]}
        except Exception as e:
            logger.debug("exception=<%s> | execute CDP action failed", str(e))
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def close(self, action: CloseAction) -> Dict[str, Any]:
        """Close the browser."""
        try:
            self._execute_async(self._async_cleanup())
            return {"status": "success", "content": [{"text": "Browser closed"}]}
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}

    def _execute_async(self, action_coro) -> Any:
        # Apply nest_asyncio if not already applied
        if not self._nest_asyncio_applied:
            nest_asyncio.apply()
            self._nest_asyncio_applied = True

        return self._loop.run_until_complete(action_coro)

    async def _async_cleanup(self) -> None:
        """Common async cleanup logic for all Playwright platforms."""
        cleanup_errors = []

        # Close all session browsers
        for session_name, session in list(self._sessions.items()):
            try:
                session_errors = await session.close()
                cleanup_errors.extend(session_errors)
                logger.debug(f"closed session: {session_name}")
            except Exception as e:
                cleanup_errors.append(f"Error closing session {session_name}: {str(e)}")

        # Stop Playwright
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                cleanup_errors.append(f"Error stopping Playwright: {str(e)}")
        self._playwright = None

        self.close_platform()
        self._sessions.clear()

        if cleanup_errors:
            for error in cleanup_errors:
                logger.debug("exception=<%s> | cleanup error occurred", error)
        else:
            logger.info("cleanup completed successfully")
