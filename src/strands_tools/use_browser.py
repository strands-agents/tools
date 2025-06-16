import asyncio
import json

# Configure logging
import logging
import os
from typing import Dict, List, Optional

import nest_asyncio
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from strands import tool

from strands_tools.utils.user_input import get_user_input

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

console = Console()

# Global browser manager instance
_playwright_manager = None

# Apply nested event loop support
nest_asyncio.apply()


# Browser manager class for handling browser interactions
class BrowserManager:
    def __init__(self):
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._cdp_client = None
        self._user_data_dir = None
        self._profile_name = None
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    async def ensure_browser(self, launch_options=None, context_options=None):
        """Initialize browser if not already running."""
        logger.debug("Ensuring browser is running...")

        if self._playwright is None:
            self._playwright = await async_playwright().start()

            default_launch_options = {"headless": False, "args": ["--window-size=1280,800"]}

            if launch_options:
                default_launch_options.update(launch_options)

            # Handle persistent context
            if launch_options and launch_options.get("persistent_context"):
                user_data_dir = launch_options.get("user_data_dir")
                if user_data_dir:
                    logger.debug(f"Creating persistent context with user_data_dir: {user_data_dir}")
                    self._context = await self._playwright.chromium.launch_persistent_context(
                        user_data_dir=user_data_dir,
                        **{
                            k: v
                            for k, v in default_launch_options.items()
                            if k not in ["persistent_context", "user_data_dir"]
                        },
                    )
                    self._browser = None  # No separate browser instance for persistent context
                else:
                    raise ValueError("user_data_dir is required for persistent context")
            else:
                # Regular browser launch
                logger.debug("Launching browser with options: %s", default_launch_options)
                self._browser = await self._playwright.chromium.launch(**default_launch_options)

                # Create context
                context_options = context_options or {}
                default_context_options = {"viewport": {"width": 1280, "height": 800}}
                default_context_options.update(context_options)

                self._context = await self._browser.new_context(**default_context_options)

            self._page = await self._context.new_page()
            self._cdp_client = await self._page.context.new_cdp_session(self._page)

        return self._page, self._cdp_client

    async def cleanup(self):
        """Clean up all browser resources."""
        logger.info("Starting browser cleanup...")

        cleanup_errors = []

        if self._page:
            try:
                await self._page.close()
                logger.debug("Page closed successfully")
            except Exception as e:
                error_msg = f"Error closing page: {str(e)}"
                logger.warning(error_msg)
                cleanup_errors.append(error_msg)

        if self._context:
            try:
                await self._context.close()
                logger.debug("Context closed successfully")
            except Exception as e:
                error_msg = f"Error closing context: {str(e)}"
                logger.warning(error_msg)
                cleanup_errors.append(error_msg)

        if self._browser:
            try:
                await self._browser.close()
                logger.debug("Browser closed successfully")
            except Exception as e:
                error_msg = f"Error closing browser: {str(e)}"
                logger.warning(error_msg)
                cleanup_errors.append(error_msg)

        if self._playwright:
            try:
                await self._playwright.stop()
                logger.debug("Playwright stopped successfully")
            except Exception as e:
                error_msg = f"Error stopping playwright: {str(e)}"
                logger.warning(error_msg)
                cleanup_errors.append(error_msg)

        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._cdp_client = None

        if cleanup_errors:
            logger.warning(f"Cleanup completed with {len(cleanup_errors)} errors:")
            for error in cleanup_errors:
                logger.warning(error)
        else:
            logger.info("Cleanup completed successfully")

    async def handle_action(self, action: str, **kwargs) -> List[Dict[str, str]]:
        """Handle both high-level actions and direct CDP commands."""
        logger.debug(f"Handling action: {action}")
        logger.debug(f"Action arguments: {kwargs}")

        try:
            result = []
            args = kwargs.get("args", {})
            launch_options = args.get("launchOptions")
            page, cdp = await self.ensure_browser(
                launch_options=launch_options,
            )

            # High-level actions
            if action == "connect":
                result = await self._handle_connect_action(launch_options)

            elif action == "navigate":
                logger.info("attempting navigate")
                result += await self._handle_navigate_action(page, args)

            elif action == "click":
                result += await self._handle_click_action(page, args)

            elif action == "type":
                result += await self._handle_type_action(page, args)

            elif action == "press_key":
                result += await self._handle_press_key_action(page, args)

            elif action == "evaluate":
                result += await self._handle_evaluate_action(page, args)

            elif action == "get_text":
                result += await self._handle_get_text_action(page, args)

            elif action == "get_html":
                result += await self._handle_get_html_action(page, args)

            elif action == "refresh":
                result += await self._handle_refresh_action(page, args)

            elif action == "back":
                result += await self._handle_back_action(page, args)

            elif action == "forward":
                result += await self._handle_forward_action(page, args)

            elif action == "new_tab":
                result += await self._handle_new_tab_action()

            elif action == "close_tab":
                result += await self._handle_close_tab_action()

            elif action == "get_cookies":
                result += await self._handle_get_cookies_action()

            elif action == "set_cookies":
                result += await self._handle_set_cookies_action(args)

            elif action == "network_intercept":
                result += await self._handle_network_intercept_action(page, args)

            elif action == "execute_cdp":
                result += await self._handle_execute_cdp_action(cdp, args)

            elif action == "close":
                result += await self._handle_close_action()

            elif action == "screenshot":
                result += await self._handle_screenshot_action(page, args)

            else:
                # Try to execute as CDP command directly
                try:
                    logger.info(f"Trying direct CDP command: {action}")
                    cdp_result = await cdp.send(action, args)
                    result.append({"text": f"CDP command result: {json.dumps(cdp_result, indent=2)}"})
                except Exception as e:
                    return [{"text": f"Error: Unknown action or CDP command failed: {str(e)}"}]

            # Handle wait_for if specified
            if kwargs.get("wait_for"):
                wait_time = kwargs["wait_for"]
                logger.debug(f"Waiting for {wait_time}ms")
                await page.wait_for_timeout(wait_time)

            logger.debug(f"Action '{action}' completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error executing action '{action}': {str(e)}")
            if "browser has been closed" in str(e) or "browser disconnected" in str(e):
                logger.debug("Cleaning up browser due to error or non-persistent session")
                await self.cleanup()
            return [{"text": f"Error: {str(e)}"}]

    # The following are helper functions being called to handle each CDP action called by the agent

    async def _handle_connect_action(self, launch_options):
        """Handle browser connection and initialization."""
        logger.debug("Handling connect action")

        await self.cleanup()  # cleanup existing browser
        page, cdp = await self.ensure_browser(launch_options=launch_options)

        result = [{"text": "Successfully connected to browser"}]

        if launch_options:
            result.append({"text": f"Launched browser with options: {json.dumps(launch_options, indent=2)}"})

        logger.debug("Connection completed")
        return result

    async def _handle_navigate_action(self, page, args):
        url = args.get("url")
        error = validate_required_param(url, "url", "navigate")
        if error:
            return error
        logger.debug(f"Navigating to URL: {url}")
        await page.goto(url)
        await page.wait_for_load_state("networkidle")
        return [{"text": f"Navigated to {url}"}]

    async def _handle_click_action(self, page, args):
        selector = args.get("selector")
        error = validate_required_param(selector, "selector", "click")
        if error:
            return error
        await page.click(selector)
        return [{"text": f"Clicked {selector}"}]

    async def _handle_type_action(self, page, args):
        selector = args.get("selector")
        text = args.get("text")
        error = validate_required_param(selector, "selector", "type")
        if error:
            return error
        error = validate_required_param(text, "text", "type")
        if error:
            return error
        await page.fill(selector, text)
        return [{"text": f"Typed '{text}' into {selector}"}]

    async def _handle_press_key_action(self, page, args):
        key = args.get("key")
        error = validate_required_param(key, "key", "press_key")
        if error:
            return error
        await page.keyboard.press(key)
        return [{"text": f"Pressed key: {key}"}]

    async def _handle_evaluate_action(self, page, args):
        script = args.get("script")

        error = validate_required_param(script, "script", "evaluate")
        if error:
            return error
        eval_result = await page.evaluate(script)
        return [{"text": f"Evaluated: {eval_result}"}]

    async def _handle_get_text_action(self, page, args):
        selector = args.get("selector")
        error = validate_required_param(selector, "selector", "get_text")
        if error:
            return error
        text_content = await page.text_content(selector)
        return [{"text": f"Text content: {text_content}"}]

    async def _handle_get_html_action(self, page, args=None):
        html = await page.content()
        return [{"text": f"HTML content: {html[:1000]}..."}]

    async def _handle_back_action(self, page, args=None):
        await page.go_back()
        await page.wait_for_load_state("networkidle")
        return [{"text": "Navigated back"}]

    async def _handle_forward_action(self, page, args=None):
        await page.go_forward()
        await page.wait_for_load_state("networkidle")
        return [{"text": "Navigated forward"}]

    async def _handle_refresh_action(self, page, args=None):
        await page.reload()
        await page.wait_for_load_state("networkidle")
        return [{"text": "Page refreshed"}]

    # Tab management actions
    async def _handle_new_tab_action(self):
        logger.debug("Creating new tab")
        new_page = await self._context.new_page()
        self._page = new_page
        self._cdp_client = await new_page.context.new_cdp_session(new_page)
        return [{"text": "New tab created"}]

    async def _handle_close_tab_action(self):
        logger.debug("Closing current tab")
        await self._page.close()
        pages = self._context.pages
        if pages:
            self._page = pages[0]
            self._cdp_client = await self._page.context.new_cdp_session(self._page)
            return [{"text": "Closed current tab and switched to another tab"}]
        return [{"text": "Closed the last tab. Browser may close."}]

    # Cookie management actions
    async def _handle_get_cookies_action(self):
        logger.debug("Getting cookies")
        cookies = await self._context.cookies()
        return [{"text": f"Cookies: {json.dumps(cookies, indent=2)}"}]

    async def _handle_set_cookies_action(self, args):
        cookies = args.get("cookies", [])
        logger.debug(f"Setting cookies: {cookies}")
        await self._context.add_cookies(cookies)
        return [{"text": "Cookies set successfully"}]

    # Network and CDP actions
    async def _handle_network_intercept_action(self, page, args):
        pattern = args.get("pattern", "*")
        handler = args.get("handler", "log")
        logger.debug(f"Setting up network interception for: {pattern}")
        if handler == "log":
            await page.route(pattern, lambda route: route.continue_())
        return [{"text": f"Network interception set for {pattern}"}]

    async def _handle_execute_cdp_action(self, cdp, args):
        method = args.get("method")
        params = args.get("params", {})
        error = validate_required_param(method, "method", "execute_cdp")
        if error:
            return error
        logger.debug(f"[BrowserManager] Executing CDP command: {method} with params: {params}")
        cdp_result = await cdp.send(method, params)
        return [{"text": f"CDP {method} result: {json.dumps(cdp_result, indent=2)}"}]

    # Browser management actions
    async def _handle_close_action(self):
        logger.debug("Closing browser")
        await self.cleanup()
        return [{"text": "Browser closed"}]

    async def _handle_screenshot_action(self, page, args):
        path = args.get("path", "screenshot.png")
        logger.debug(f"Taking screenshot: {path}")
        await page.screenshot(path=path)
        return [{"text": f"Screenshot saved as {path}"}]


# Initialize global browser manager
_playwright_manager = BrowserManager()

# Some helper functions used throughout the code


def validate_required_param(param_value, param_name, action_name):
    """Validate that a required parameter is provided"""
    if not param_value:
        return [{"text": f"Error: {param_name} required for {action_name}"}]
    return None


@tool
def use_browser(
    url: str = None,  # set a default value
    wait_time: int = 1,
    action: str = None,
    new_tab: bool = False,
    selector: str = None,
    input_text: str = None,
    script: str = None,
    cdp_method: str = None,
    cdp_params: dict = None,
    launch_options: dict = None,
    actions: list = None,
    key: str = None,  # Add key parameter for press_key action
) -> str:
    """
    Perform browser operations using Playwright.

    Important Usage Guidelines:
    - For clicking or typing into elements, first use get_html or get_text to find the correct selector
    - If initial selector search fails, use evaluate to parse the HTML contents
    - For web searches:
    1. Start with Google (https://www.google.com)
    2. Use get_html/get_text to find search box
    3. If CAPTCHA appears, fallback to DuckDuckGo (https://duckduckgo.com)

    Args:
        action: The action to perform: 'back', 'forward', 'refresh', 'new_tab', 'close_tab',
            'navigate', 'click', 'type', 'evaluate', 'get_text', 'get_html',
            'get_cookies', 'set_cookies', 'network_intercept', 'execute_cdp', 'close', 'connect', 'screenshot',
            'press_key'.
        url: The URL to navigate to (required only when action is 'navigate')
        wait_time: Time to wait after action in seconds
        selector: Element selector for interactions
        input_text: Text to type into elements
        script: JavaScript to evaluate
        cdp_method: CDP method to execute
        cdp_params: Parameters for CDP method
        launch_options: Browser launch configuration options including:
            - headless (bool): Whether to run browser in headless mode
            - args (list): Additional browser command line arguments
            - ignoreDefaultArgs (bool): Whether to ignore default Playwright arguments
            - proxy (dict): Proxy server configuration
            - downloadsPath (str): Path for downloaded files
            - chromiumSandbox (bool): Whether to enable Chromium sandbox
            - port (int): Port to connect to browser
            - userDataDir (str): Path to Chrome user data directory for persistent sessions
            - profileName (str): Name of the Chrome profile to use
            - persistentContext (bool): Whether to create a persistent browser context
        actions: List of sequential actions to perform
        key: Key to press when using the press_key action

    Returns:
    str: Message indicating the result of the operation and extracted content if requested.
    """
    logger.info(f"use_browser tool called with action: {action}")

    if actions:
        logger.info(
            f"Multiple actions requested: {[a.get('action') for a in actions if isinstance(a, dict) and 'action' in a]}"
        )

    strands_dev = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

    if not strands_dev:
        # Get user confirmation
        if actions:
            action_description = "multiple actions"
            action_list = [a.get("action") for a in actions if isinstance(a, dict) and "action" in a]
            message = Text("User requested multiple actions: ", style="yellow")
            message.append(Text(", ".join(action_list), style="bold cyan"))
        else:
            action_description = action or "unknown"
            message = Text("User requested action: ", style="yellow")
            message.append(Text(action_description, style="bold cyan"))

        console.print(Panel(message, title="[bold green]BrowserManager", border_style="green"))

        user_input = get_user_input(f"Do you want to proceed with {action_description}? (y/n)")
        if user_input.lower().strip() != "y":
            cancellation_reason = (
                user_input if user_input.strip() != "n" else get_user_input("Please provide a reason for cancellation:")
            )
            error_message = f"Python code execution cancelled by the user. Reason: {cancellation_reason}"

            return {
                "status": "error",
                "content": [{"text": error_message}],
            }

    logger.info(f"Tool parameters: {locals()}")
    try:
        all_content = []

        # Handle multiple actions case
        if actions:
            # Create a coroutine that runs all actions sequentially
            async def run_all_actions():
                results = []
                for action_item in actions:
                    action_name = action_item.get("action")
                    action_args = action_item.get("args", {})
                    action_selector = action_item.get("selector")
                    action_wait_for = action_item.get("wait_for", wait_time * 1000 if wait_time else None)

                    if launch_options:
                        action_args["launchOptions"] = launch_options

                    logger.info(f"Executing action: {action_name}")

                    # Execute the action and collect results
                    content = await _playwright_manager.handle_action(
                        action=action_name,
                        args=action_args,
                        selector=action_selector,
                        wait_for=action_wait_for,
                    )
                    results.extend(content)
                return results

            # Run all actions in a single event loop call
            all_content = _playwright_manager._loop.run_until_complete(run_all_actions())
            return "\n".join([item["text"] for item in all_content])

        # Handle single action case
        else:
            # Prepare args based on parameters
            args = {}
            if url:
                args["url"] = url
            if input_text:
                args["text"] = input_text
            if script:
                args["script"] = script
            if selector:
                args["selector"] = selector
            if cdp_method:
                args["method"] = cdp_method
                if cdp_params:
                    args["params"] = cdp_params
            if key:
                args["key"] = key
            if launch_options:
                args["launchOptions"] = launch_options

            # Execute the action
            logger.info(f"calling action {action} to handle_action")
            content = _playwright_manager._loop.run_until_complete(
                _playwright_manager.handle_action(
                    action=action, args=args, selector=selector, wait_for=wait_time * 1000 if wait_time else None
                )
            )
            all_content.extend(content)
            return "\n".join([item["text"] for item in all_content])

    except Exception as e:
        logger.error(f"Error in use_browser: {str(e)}")
        # Cleanup only if explicitly requested or non-persistent session
        logger.info("Cleaning up browser due to explicit request or error with non-persistent session")
        _playwright_manager._loop.run_until_complete(_playwright_manager.cleanup())
        return f"Error: {str(e)}"
