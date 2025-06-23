import asyncio
import json

# Configure logging
import logging
import os
import time  # Added for timestamp in screenshot filenames
from typing import Dict, List, Optional

import nest_asyncio
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from strands import tool

from strands_tools.utils.user_input import get_user_input

# Only configure this module's logger, not the root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a handler for this logger if it doesn't have one
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

# Prevent propagation to parent loggers to avoid duplicate logs
logger.propagate = False

console = Console()

# Global browser manager instance
_playwright_manager = None

# Apply nested event loop support
nest_asyncio.apply()

# Environment Variables

screenshots_dir = os.getenv("BROWSER_SCREENSHOTS_DIR", "screenshots")


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
        self._tabs = {}  # Dictionary to track tabs by ID
        self._active_tab_id = None  # Currently active tab ID
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.action_configs = {
            "navigate": {
                "method": lambda page, args: self._safe_navigation(page, args["url"]),
                "required_params": [("url", str)],
                "post_action": lambda page: page.wait_for_load_state("networkidle"),
                "result_template": "Navigated to {url}",
            },
            "click": {
                "method": lambda page, args: page.click(args["selector"]),
                "required_params": [("selector", str)],
                "result_template": "Clicked {selector}",
            },
            "type": {
                "method": lambda page, args: page.fill(args["selector"], args["text"]),
                "required_params": [("selector", str), ("text", str)],
                "result_template": "Typed '{text}' into {selector}",
            },
            "evaluate": {
                "method": lambda page, args: page.evaluate(args["script"]),
                "required_params": [("script", str)],
                "result_template": "Evaluation result: {result}",
            },
            "press_key": {
                "method": lambda page, args: page.keyboard.press(args["key"]),
                "required_params": [("key", str)],
                "result_template": "Pressed key: {key}",
            },
            "get_text": {
                "method": lambda page, args: page.text_content(args["selector"]),
                "required_params": [("selector", str)],
                "post_process": lambda result: result,
                "result_template": "Text content: {result}",
            },
            "get_html": {
                "method": lambda page, args: page.content()
                if not args.get("selector")
                else page.inner_html(args.get("selector")),
                "required_params": [],
                "post_process": lambda result: result[:1000] + "..." if len(result) > 1000 else result,
                "result_template": "HTML content: {result}",
            },
            "refresh": {
                "method": lambda page, args: page.reload(),
                "required_params": [],
                "post_action": lambda page: page.wait_for_load_state("networkidle"),
                "result_template": "Page refreshed",
            },
            "back": {
                "method": lambda page, args: page.go_back(),
                "required_params": [],
                "post_action": lambda page: page.wait_for_load_state("networkidle"),
                "result_template": "Navigated back",
            },
            "forward": {
                "method": lambda page, args: page.go_forward(),
                "required_params": [],
                "post_action": lambda page: page.wait_for_load_state("networkidle"),
                "result_template": "Navigated forward",
            },
            "screenshot": {
                "method": lambda page, args: self._take_screenshot(page, args),
                "required_params": [],
                "result_template": "Screenshot saved as {path}",
            },
            "new_tab": {
                "method": lambda page, args: self._create_new_tab(args.get("tab_id")),
                "required_params": [],
                "result_template": "New tab created with ID: {result}",
            },
            "switch_tab": {
                "method": lambda page, args: self._switch_to_tab(args.get("tab_id")),
                "required_params": [("tab_id", str)],
                "result_template": "Switched to tab: {tab_id}",
            },
            "close_tab": {
                "method": lambda page, args: self._close_tab_by_id(args.get("tab_id", self._active_tab_id)),
                "required_params": [],
                "result_template": "Tab closed successfully",
            },
            "list_tabs": {
                "method": lambda page, args: self._list_tabs(),
                "required_params": [],
                "post_process": lambda result: json.dumps(result, indent=2),
                "result_template": "Tabs: {result}",
            },
            "get_cookies": {
                "method": lambda page, args: self._context.cookies(),
                "required_params": [],
                "post_process": lambda result: json.dumps(result, indent=2),
                "result_template": "Cookies: {result}",
            },
            "set_cookies": {
                "method": lambda page, args: self._context.add_cookies(args.get("cookies", [])),
                "required_params": [("cookies", list)],
                "result_template": "Cookies set successfully",
            },
            "network_intercept": {
                "method": lambda page, args: page.route(args.get("pattern", "*"), lambda route: route.continue_()),
                "required_params": [],
                "result_template": "Network interception set for {pattern}",
            },
            "execute_cdp": {
                "method": lambda page, args: self._cdp_client.send(args["method"], args.get("params", {})),
                "required_params": [("method", str)],
                "post_process": lambda result: json.dumps(result, indent=2),
                "result_template": "CDP {method} result: {result}",
            },
            "close": {
                "method": lambda page, args: self.cleanup(),
                "required_params": [],
                "result_template": "Browser closed",
            },
        }

    async def ensure_browser(self, launch_options=None, context_options=None):
        """Initialize browser if not already running."""
        logger.debug("Ensuring browser is running...")

        # Ensure required directories exist
        user_data_dir = os.getenv("BROWSER_USER_DATA_DIR", os.path.join(os.path.expanduser("~"), ".browser_automation"))
        headless = os.getenv("BROWSER_HEADLESS", "false").lower() == "true"
        width = int(os.getenv("BROWSER_WIDTH", "1280"))
        height = int(os.getenv("BROWSER_HEIGHT", "800"))
        os.makedirs(screenshots_dir, exist_ok=True)
        os.makedirs(user_data_dir, exist_ok=True)

        try:
            if self._playwright is None:
                self._playwright = await async_playwright().start()

                default_launch_options = {"headless": headless, "args": [f"--window-size={width},{height}"]}

                if launch_options:
                    default_launch_options.update(launch_options)

                # Handle persistent context
                if launch_options and launch_options.get("persistent_context"):
                    if launch_options and launch_options.get("persistent_context"):
                        # Use the environment variable by default, but allow override from launch_options
                        persistent_user_data_dir = launch_options.get("user_data_dir", user_data_dir)
                        self._context = await self._playwright.chromium.launch_persistent_context(
                            user_data_dir=persistent_user_data_dir,
                            **{
                                k: v
                                for k, v in default_launch_options.items()
                                if k not in ["persistent_context", "user_data_dir"]
                            },
                        )
                        self._browser = None
                    else:
                        raise ValueError("user_data_dir is required for persistent context")
                else:
                    # Regular browser launch
                    logger.debug("Launching browser with options: %s", default_launch_options)
                    self._browser = await self._playwright.chromium.launch(**default_launch_options)

                    # Create context
                    context_options = context_options or {}
                    default_context_options = {"viewport": {"width": width, "height": height}}
                    default_context_options.update(context_options)

                    self._context = await self._browser.new_context(**default_context_options)

                self._page = await self._context.new_page()
                self._cdp_client = await self._page.context.new_cdp_session(self._page)

                # Initialize tab tracking with the first tab
                first_tab_id = "main"
                self._tabs[first_tab_id] = self._page
                self._active_tab_id = first_tab_id

            if not self._page:
                raise ValueError("Browser initialized but page is not available")

            return self._page, self._cdp_client

        except Exception as e:
            logger.error(f"Failed to initialize browser: {str(e)}")
            # Clean up any partial initialization
            await self.cleanup()
            # Re-raise the exception so it's caught by the error handling in handle_action
            raise

    async def cleanup(self):
        cleanup_errors = []

        for resource in ["_page", "_context", "_browser", "_playwright"]:
            attr = getattr(self, resource)
            if attr:
                try:
                    if resource == "_playwright":
                        await attr.stop()
                    else:
                        await attr.close()
                except Exception as e:
                    cleanup_errors.append(f"Error closing {resource}: {str(e)}")

        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        self._cdp_client = None
        self._tabs = {}  # Clear tab dictionary
        self._active_tab_id = None

        if cleanup_errors:
            for error in cleanup_errors:
                logger.error(error)
        else:
            logger.info("Cleanup completed successfully")

    async def _fix_javascript_syntax(self, script, error_msg):
        """
        Attempts to fix common JavaScript syntax errors based on error messages.

        Args:
            script: The original JavaScript code with syntax errors
            error_msg: The error message from the JavaScript engine

        Returns:
            Fixed JavaScript code if a fix was found, otherwise None
        """
        if not script or not error_msg:
            return None

        fixed_script = None
        # Handle illegal return statements
        if "Illegal return statement" in error_msg:
            # Wrap in IIFE (Immediately Invoked Function Expression)
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
            # Count opening and closing braces/parentheses to see if they're balanced
            open_chars = script.count("{") + script.count("(") + script.count("[")
            close_chars = script.count("}") + script.count(")") + script.count("]")

            if open_chars > close_chars:
                # Add missing closing characters
                missing = open_chars - close_chars
                fixed_script = script + ("}" * missing)
                logger.info(f"Added {missing} missing closing braces")

        # Handle uncaught reference errors
        elif "is not defined" in error_msg:
            var_name = error_msg.split("'")[1] if "'" in error_msg else ""
            if var_name:
                fixed_script = f"var {var_name} = undefined;\n{script}"
                logger.info(f"Adding undefined variable declaration for '{var_name}'")

        # Return the fixed script or None if no fix was applied
        return fixed_script

    async def handle_action(self, action: str, **kwargs) -> List[Dict[str, str]]:
        try:
            # Extract args here at the top level so it's available for retry_action
            args = kwargs.get("args", {})

            async def action_operation():
                result = []
                launch_options = args.get("launchOptions")
                page, cdp = await self.ensure_browser(
                    launch_options=launch_options,
                )

                # Actions that are defined in BrowserManager actions config
                if action in self.action_configs:
                    result = await self._generic_action_handler(action, page, args)
                    if not result:
                        result = [{"text": f"{action} completed successfully"}]
                    # Only log success if no exceptions were raised
                    logger.debug(f"Action '{action}' completed successfully")
                    return result
                else:
                    # Try to execute as CDP command directly
                    try:
                        logger.info(f"Trying direct CDP command: {action}")
                        cdp_result = await cdp.send(action, args)
                        result.append({"text": f"CDP command result: {json.dumps(cdp_result, indent=2)}"})
                        logger.debug(f"Action '{action}' completed successfully")
                    except Exception as e:
                        return [{"text": f"Error: Unknown action or CDP command failed: {str(e)}"}]

                # Handle wait_for if specified
                if kwargs.get("wait_for"):
                    await page.wait_for_timeout(kwargs["wait_for"])

                return result

            result = await self.retry_action(action_operation, action_name=action, args=args)

            # Check if result is already a list of dictionaries with text entries
            # (which happens when retry_action catches non-retryable errors)
            if isinstance(result, list) and all(isinstance(item, dict) and "text" in item for item in result):
                return result

            return result
        except Exception as e:
            logger.error(f"Error executing action '{action}': {str(e)}")
            if "ERR_SOCKET_NOT_CONNECTED" in str(e):  # Adding special case for when network connection issues
                return [{"text": "Error: Connection issue detected. Please verify network connectivity and try again."}]
            if "browser has been closed" in str(e) or "browser disconnected" in str(e):
                await self.cleanup()
            return [{"text": f"Error: {str(e)}"}]

    async def retry_action(self, action_func, action_name=None, args=None):
        """
        Retry an async operation with exponential backoff.

        Args:
            action_func: Async function to execute
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries (doubles with each attempt)
            action_name: Name of the action being retried
            args: Arguments passed to the action (to allow fixing JavaScript for evaluate action)
        """
        last_exception = None
        max_retries = int(os.getenv("BROWSER_MAX_RETRIES", 3))
        retry_delay = int(os.getenv("BROWSER_RETRY_DELAY", "1"))

        for attempt in range(max_retries):
            try:
                return await action_func()
            except Exception as e:
                last_exception = e
                error_msg = str(e)

                # Check for non-retryable errors (DNS, connection refused, etc.)
                non_retryable_errors = [
                    "Could not resolve domain",
                    "Connection refused",
                    "Connection timed out",
                    "SSL/TLS error",
                    "Certificate error",
                    "Protocol error (Page.navigate): Cannot navigate to invalid URL",
                ]

                # If this is a non-retryable error, don't retry and return the error message
                if any(msg in error_msg for msg in non_retryable_errors):
                    logger.warning(f"Non-retryable error detected: {error_msg}")
                    return [{"text": f"Error: {error_msg}"}]

                # Log every failed attempt
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {error_msg}")

                # Only process retry if this attempt wasn't the last
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)

                    # Handle JavaScript errors more broadly - not just syntax errors
                    if action_name == "evaluate" and args and "script" in args:
                        error_types = [
                            "SyntaxError",
                            "ReferenceError",
                            "TypeError",
                            "Illegal return",
                            "Unexpected token",
                            "Unexpected end",
                            "is not defined",
                        ]
                        if any(err_type in error_msg for err_type in error_types):
                            # Try to fix common JavaScript errors using our helper
                            script = args["script"]
                            fixed_script = await self._fix_javascript_syntax(script, error_msg)

                            if fixed_script:
                                logger.warning("Detected JavaScript error. Trying with modified script.")
                                logger.warning(f"Original: {script}")
                                logger.warning(f"Modified: {fixed_script}")

                                # Update args for next attempt
                                args["script"] = fixed_script

                                # No need for delay on retrying with fixed script
                                logger.warning("Attempting retry with fixed JavaScript")
                                continue

                    logger.warning(f"Retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)

        logger.error(f"Action failed after {max_retries} attempts: {str(last_exception)}")
        raise last_exception

    async def _generic_action_handler(self, action: str, page, args: dict) -> List[Dict[str, str]]:
        """
        Generic handler for actions defined in action_configs.

        Args:
            action: The action to perform
            page: The Playwright page object
            args: Dictionary of arguments for the action

        Returns:
            List of dictionaries with text results

        Raises:
            ValueError: If required parameters are missing
        """

        if args is None:
            raise ValueError(f"Args dictionary is required for {action} action")

        if action not in self.action_configs:
            raise ValueError(f"Unknown action: {action}")

        config = self.action_configs[action]

        # Validate required parameters
        for param_name, _ in config.get("required_params", []):
            param_value = args.get(param_name)
            if not param_value:
                # Special handling for specific actions
                if action == "switch_tab" and param_name == "tab_id":
                    tab_info = await self._get_tab_info_for_logs()
                    error_msg = f"Error: '{param_name}' is required for {action} action. {tab_info}"
                else:
                    error_msg = f"Error: '{param_name}' is required for {action} action"

                logger.error(error_msg)
                raise ValueError(error_msg)

        try:
            # Execute the action method
            method = config["method"]
            result = await method(page, args)

            # Execute any post-action steps
            if "post_action" in config:
                await config["post_action"](page)

            # Apply post-processing to the result if needed
            if "post_process" in config and result is not None:
                processed_result = config["post_process"](result)
                args.update({"result": processed_result})
            elif result is not None:
                args.update({"result": result})

            # Format the result message using the template
            template = config.get("result_template", f"{action} completed")
            formatted_message = template.format(**args)

            # Always return a list containing a dict with text key
            return [{"text": formatted_message}]
        except PlaywrightTimeoutError as e:
            logger.error(f"Timeout error in {action}: {str(e)}")
            raise ValueError(
                f"Action '{action}' timed out. The element might not be available or the page is still loading."
            ) from e
        except PlaywrightError as e:
            logger.error(f"Playwright error in {action}: {str(e)}")
            # Handle specific Playwright errors
            error_msg = str(e).lower()
            if "element not found" in error_msg or "no such element" in error_msg:
                raise ValueError(
                    f"Element not found for action '{action}'. Please verify the selector is correct."
                ) from e
            elif "element not visible" in error_msg or "not visible" in error_msg:
                raise ValueError(
                    f"Element is not visible for action '{action}'. "
                    f"The element might be hidden or not yet rendered."
                ) from e
            elif "element not interactable" in error_msg or "not interactable" in error_msg:
                raise ValueError(
                    f"Element is not interactable for action '{action}'. "
                    f"The element might be disabled or covered by another element."
                ) from e
            else:
                raise ValueError(f"Playwright error in action '{action}': {str(e)}") from e
        except Exception as e:
            logger.error(f"Error in generic action handler for {action}: {str(e)}")
            # Don't log action success here, and make sure to raise the exception
            # so the retry mechanism works properly
            raise

    async def _create_new_tab(self, tab_id=None):
        """Create a new tab and track it with the given ID"""
        if tab_id is None:
            tab_id = f"tab_{len(self._tabs) + 1}"

        # Check if tab_id already exists
        if tab_id in self._tabs:
            return [{"text": f"Error: Tab with ID {tab_id} already exists"}]

        new_page = await self._context.new_page()
        self._tabs[tab_id] = new_page

        # Switch to the new tab
        await self._switch_to_tab(tab_id)

        return tab_id

    async def _switch_to_tab(self, tab_id):
        """Switch to the tab with the given ID"""
        if not tab_id:
            tab_info = await self._get_tab_info_for_logs()
            error_msg = f"tab_id is required for switch_tab action. {tab_info}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if tab_id not in self._tabs:
            tab_info = await self._get_tab_info_for_logs()
            error_msg = f"Tab with ID '{tab_id}' not found. {tab_info}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self._page = self._tabs[tab_id]
        self._cdp_client = await self._page.context.new_cdp_session(self._page)
        self._active_tab_id = tab_id

        # Use CDP to bring the tab to the foreground
        try:
            await self._cdp_client.send("Page.bringToFront")
            logger.info(f"Successfully switched to tab '{tab_id}' and brought it to the foreground")
        except Exception as e:
            logger.warning(f"Failed to bring tab '{tab_id}' to foreground: {str(e)}")

        return tab_id

    async def _close_tab_by_id(self, tab_id):
        """Close the tab with the given ID"""
        if not tab_id:
            raise ValueError("tab_id is required for close_tab action")

        if tab_id not in self._tabs:
            raise ValueError(f"Tab with ID '{tab_id}' not found. Available tabs: {list(self._tabs.keys())}")

        # Close the tab
        await self._tabs[tab_id].close()

        # Remove from tracking
        del self._tabs[tab_id]

        # If we closed the active tab, switch to another tab if available
        if tab_id == self._active_tab_id:
            if self._tabs:
                next_tab_id = next(iter(self._tabs.keys()))
                await self._switch_to_tab(next_tab_id)
            else:
                self._page = None
                self._cdp_client = None
                self._active_tab_id = None

        logger.info(f"Successfully closed tab '{tab_id}'")
        return True

    async def _get_tab_info_for_logs(self):
        """Get a summary of current tabs for error messages"""
        tabs = {}
        for tab_id, page in self._tabs.items():
            try:
                is_active = tab_id == self._active_tab_id
                tabs[tab_id] = {"url": page.url, "active": is_active}
            except (AttributeError, ConnectionError, Exception) as e:
                tabs[tab_id] = {"error": f"Could not retrieve tab info: {str(e)}"}

        return f"Available tabs: {json.dumps(tabs)}"

    async def _safe_navigation(self, page, url):
        try:
            return await page.goto(url)
        except Exception as e:
            error_str = str(e)
            if "ERR_NAME_NOT_RESOLVED" in error_str:
                raise ValueError(
                    f"Could not resolve domain '{url}'. The website might not exist or a network connectivity issue."
                ) from e
            elif "ERR_CONNECTION_REFUSED" in error_str:
                raise ValueError(
                    f"Connection refused for '{url}'. The server might be down or blocking requests."
                ) from e
            elif "ERR_CONNECTION_TIMED_OUT" in error_str:
                raise ValueError(f"Connection timed out for '{url}'. The server might be slow or unreachable.") from e
            elif "ERR_SSL_PROTOCOL_ERROR" in error_str:
                raise ValueError(
                    f"SSL/TLS error when connecting to '{url}'. The site might have an invalid or expired certificate."
                ) from e
            elif "ERR_CERT_" in error_str:
                raise ValueError(
                    f"Certificate error when connecting to '{url}'. The site's security certificate might be invalid."
                ) from e
            else:
                raise

    async def _list_tabs(self):
        """Return a list of all tracked tabs"""
        tab_info = {}
        for tab_id, page in self._tabs.items():
            try:
                url = page.url
                title = await page.title()
                is_active = tab_id == self._active_tab_id
                tab_info[tab_id] = {"url": url, "title": title, "active": is_active}
            except (ConnectionError, RuntimeError, Exception) as e:
                tab_info[tab_id] = {
                    "url": "Error retrieving URL",
                    "title": f"Error: {str(e)}",
                    "active": tab_id == self._active_tab_id,
                }
        return tab_info

    async def _take_screenshot(self, page, args):
        """Take a screenshot and return the path for template formatting"""
        screenshot_path = args.get("path", os.path.join(screenshots_dir, f"screenshot_{int(time.time())}.png"))
        await page.screenshot(path=screenshot_path)
        args["path"] = screenshot_path
        return screenshot_path


# Initialize global browser manager
_playwright_manager = BrowserManager()


def validate_required_param(param_value, param_name, action_name):
    """Validate that a required parameter is provided"""
    if not param_value:
        return [{"text": f"Error: {param_name} required for {action_name}"}]
    return None


@tool
def use_browser(
    url: str = None,
    wait_time: int = int(os.getenv("DEFAULT_WAIT_TIME", 1)),
    action: str = None,
    selector: str = None,
    input_text: str = None,
    script: str = None,
    cdp_method: str = None,
    cdp_params: dict = None,
    launch_options: dict = None,
    actions: list = None,
    args: dict = None,
    key: str = None,
) -> str:
    """
    Interactive browser automation tool powered by Playwright.

    Important Usage Guidelines:
    - Never guess selectors or locators! Always find them first using these steps:
        1. Use get_html to examine the page structure:
        {"action": "get_html"}  # Get full page HTML
        or
        {"action": "get_html", "args": {"selector": "body"}}  # Get body HTML

        2. Use evaluate with JavaScript to find specific elements:
        {"action": "evaluate", "args": {"script": `
            return Array.from(document.querySelectorAll('input, button'))
                .map(el => ({
                    tag: el.tagName,
                    type: el.type,
                    id: el.id,
                    name: el.name,
                    class: el.className,
                    placeholder: el.placeholder,
                    value: el.value
                }))
        `}}

        3. Only after finding the correct selector, use it for actions like click or type

    - For complex operations requiring multiple steps, use the 'actions' parameter
    - For web searches:
        1. Start with Google (https://www.google.com)
        2. First find the search box:
        {"action": "evaluate", "args": {"script": `
            return Array.from(document.querySelectorAll('input'))
                .map(el => ({
                    type: el.type,
                    name: el.name,
                    placeholder: el.placeholder
                }))
        `}}
        3. If CAPTCHA appears, fallback to DuckDuckGo (https://duckduckgo.com)

    Tab Management:
    - Create a new tab with an ID:
      {"action": "new_tab", "args": {"tab_id": "search_tab"}}

    - Switch between tabs (MUST provide tab_id in args):
      use_browser(action="switch_tab", actions=[{"action": "switch_tab", "args": {"tab_id": "main"}}])

      # CORRECT EXAMPLES:
      # Method 1 (recommended): Using the actions parameter
      use_browser(actions=[{"action": "switch_tab", "args": {"tab_id": "main"}}])

      # Method 2: Using single action with args parameter
      use_browser(action="switch_tab", args={"tab_id": "search_tab"})

      # INCORRECT (will fail):
      use_browser(action="switch_tab")  # Missing tab_id

    - Close a specific tab:
      {"action": "close_tab", "args": {"tab_id": "search_tab"}}

    - List all tabs and their status:
      {"action": "list_tabs"}

    - Actions are performed only on the active tab

    Common Multi-Action Patterns:
    1. Form filling (with selector discovery):
        actions=[
            {"action": "navigate", "args": {"url": "form_url"}},
            {"action": "get_html"},  # First get page HTML
            {"action": "evaluate", "args": {"script": `
                return Array.from(document.querySelectorAll('input'))
                    .map(el => ({
                        id: el.id,
                        name: el.name,
                        type: el.type
                    }))
            `}},  # Find input selectors
            {"action": "type", "args": {"selector": "#found-input-id", "text": "value"}}
        ]

    2. Web scraping (with content discovery):
        actions=[
            {"action": "navigate", "args": {"url": "target_url"}},
            {"action": "evaluate", "args": {"script": `
                return {
                    content: document.querySelector('main')?.innerHTML,
                    nextButton: Array.from(document.querySelectorAll('a'))
                        .find(a => a.textContent.includes('Next'))?.outerHTML
                }
            `}},
            {"action": "click", "args": {"selector": "discovered-next-button-selector"}}
        ]

    3. Working with multiple tabs:
        actions=[
            {"action": "navigate", "args": {"url": "https://example.com"}},
            {"action": "new_tab", "args": {"tab_id": "second_tab"}},
            {"action": "navigate", "args": {"url": "https://example.org"}},
            {"action": "switch_tab", "args": {"tab_id": "main"}},
            {"action": "get_html", "args": {"selector": "h1"}}
        ]

    Args:
        url (str, optional): URL to navigate to. Used with 'navigate' action.
        wait_time (int, optional): Time to wait in seconds after performing an action.
            Default is set by DEFAULT_WAIT_TIME env var or 1 second.
        action (str, optional): Single action to perform. Common actions include:
            - navigate: Go to a URL
            - click: Click on an element
            - type: Input text into a field
            - evaluate: Run JavaScript
            - get_text: Get text from an element
            - get_html: Get HTML content
            - screenshot: Take a screenshot
            - new_tab: Create a new browser tab
            - switch_tab: Switch to a different tab (REQUIRES tab_id in args)
            - close_tab: Close a tab
            - list_tabs: List all open tabs
        selector (str, optional): CSS selector to identify page elements. Required for
            actions like click, type, and get_text.
        input_text (str, optional): Text to input into a field. Required for 'type' action.
        script (str, optional): JavaScript code to execute. Required for 'evaluate' action.
        cdp_method (str, optional): Chrome DevTools Protocol method name for 'execute_cdp' action.
        cdp_params (dict, optional): Parameters for CDP method.
        launch_options (dict, optional): Browser launch options. Common options include:
            - headless: Boolean to run browser in headless mode
            - args: List of command-line arguments for the browser
            - persistent_context: Boolean to use persistent browser context
            - user_data_dir: Path to user data directory for persistent context
        actions (list, optional): List of action objects to perform in sequence.
            Each action is a dict with 'action', 'args', and optional 'wait_for' keys.
            Example: [{"action": "switch_tab", "args": {"tab_id": "main"}}]
        args (dict, optional): Dictionary of arguments for the action. Used when specific
            parameters are needed for an action, especially for tab operations.
            Example: {"tab_id": "main"} for switch_tab action.
        key (str, optional): Keyboard key to press for 'press_key' action.

    Returns:
        str: Text description of the action results. For single actions, returns the result text.
            For multiple actions, returns all results concatenated with newlines.
            On error, returns an error message starting with "Error: ".
    """
    strands_dev = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

    if not strands_dev:
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
        # Convert single action to actions list format if not using actions parameter
        if not actions and action:
            # Prepare args dictionary
            action_args = args or {}

            # Add specific parameters to args if provided
            if url:
                action_args["url"] = url
            if input_text:
                action_args["text"] = input_text
            if script:
                action_args["script"] = script
            if selector:
                action_args["selector"] = selector
            if cdp_method:
                action_args["method"] = cdp_method
                if cdp_params:
                    action_args["params"] = cdp_params
            if key:
                action_args["key"] = key
            if launch_options:
                action_args["launchOptions"] = launch_options

            # Special handling for tab_id parameter
            if action == "switch_tab" and "tab_id" not in action_args:
                try:
                    # Only try to get tabs if browser is already initialized
                    if _playwright_manager._page is not None:
                        tabs_list = _playwright_manager._loop.run_until_complete(_playwright_manager._list_tabs())
                        tab_ids = list(tabs_list.keys())
                        return f"Error: tab_id is required for switch_tab action. Available tabs: {tab_ids}"
                    else:
                        return "Error: tab_id is required for switch_tab action. Browser not yet initialized."
                except Exception:
                    return "Error: tab_id is required for switch_tab action. Could not retrieve available tabs."

            # For close_tab action, default to active tab if none specified
            if action == "close_tab" and "tab_id" not in action_args:
                active_tab = _playwright_manager._active_tab_id
                if active_tab:
                    action_args["tab_id"] = active_tab

            actions = [
                {
                    "action": action,
                    "args": action_args,
                    "selector": selector,
                    "wait_for": wait_time * 1000 if wait_time else None,
                }
            ]

        # Create a coroutine that runs all actions sequentially
        async def run_all_actions():
            results = []
            logger.debug(f"Processing {len(actions)} actions: {actions}")  # Debug the actions
            for action_item in actions:
                action_name = action_item.get("action")
                action_args = action_item.get("args", {})
                action_selector = action_item.get("selector")
                action_wait_for = action_item.get("wait_for", wait_time * 1000 if wait_time else None)

                if launch_options:
                    action_args["launchOptions"] = launch_options

                logger.info(f"Executing action: {action_name}")
                logger.debug(f"Action args: {action_args}")  # Debug the args

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

    except Exception as e:
        logger.error(f"Error in use_browser: {str(e)}")
        logger.error("Cleaning up browser due to explicit request or error with non-persistent session")
        _playwright_manager._loop.run_until_complete(_playwright_manager.cleanup())
        return f"Error: {str(e)}"
