import asyncio
import inspect
import json

# Configure logging
import logging
import os
import time  # Added for timestamp in screenshot filenames
from typing import Callable, Dict, List, Optional

import nest_asyncio
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from strands import tool

from strands_tools.utils.user_input import get_user_input

logger = logging.getLogger(__name__)

console = Console()

# Global browser manager instance
_playwright_manager = None


class BrowserApiMethods:
    # Api Method Calls
    async def navigate(page: Page, url: str):
        try:
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            return f"Navigated to {url}"
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

    async def click(page: Page, selector: str):
        await page.click(selector)
        return f"Clicked element: {selector}"

    async def type(page: Page, selector: str, text: str):
        await page.fill(selector, text)
        return f"Typed '{text}' into {selector}"

    async def evaluate(page: Page, script: str):
        result = await page.evaluate(script)
        return f"Evaluation result: {result}"

    async def press_key(page: Page, key: str):
        await page.keyboard.press(key)
        return f"Pressed key: {key}"

    async def get_text(page: Page, selector: str):
        text = await page.text_content(selector)
        return f"Text content: {text}"

    async def get_html(page: Page, selector: str = None):
        if not selector:
            result = await page.content()
        else:
            try:
                await page.wait_for_selector(selector, timeout=5000)
                result = await page.inner_html(selector)
            except PlaywrightTimeoutError as e:
                raise ValueError(
                    f"Element with selector '{selector}' not found on the page. Please verify the selector is correct."
                ) from e
        return (result[:1000] + "..." if len(result) > 1000 else result,)

    async def screenshot(page: Page, path: str = None):
        """Take a screenshot with configurable path from environment variable"""
        screenshots_dir = os.getenv("STRANDS_BROWSER_SCREENSHOTS_DIR", "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)  # Ensure directory exists

        if not path:
            # Generate default filename with timestamp if no path provided
            filename = f"screenshot_{int(time.time())}.png"
            path = os.path.join(screenshots_dir, filename)
        elif not os.path.isabs(path):
            # If relative path provided, make it relative to screenshots directory
            path = os.path.join(screenshots_dir, path)

        await page.screenshot(path=path)
        return f"Screenshot saved as {path}"

    async def refresh(page: Page):
        page.reload()
        page.wait_for_load_state("networkidle")
        return "Page refreshed"

    async def back(page: Page):
        page.go_back()
        page.wait_for_load_state("networkidle")
        return "Navigated back"

    async def forward(page: Page):
        page.go_forward()
        page.wait_for_load_state("networkidle")
        return "Navigated forward"

    async def new_tab(page: Page, browser_manager, tab_id: str = None):
        if tab_id is None:
            tab_id = f"tab_{len(browser_manager._tabs) + 1}"

        if tab_id in browser_manager._tabs:
            return f"Error: Tab with ID {tab_id} already exists"

        new_page = await browser_manager._context.new_page()
        browser_manager._tabs[tab_id] = new_page

        # Switch to the new tab
        await BrowserApiMethods.switch_tab(new_page, browser_manager, tab_id)

        return f"Created new tab with ID: {tab_id}"

    async def switch_tab(page: Page, browser_manager, tab_id: str):
        if not tab_id:
            tab_info = await BrowserApiMethods._get_tab_info_for_logs(browser_manager)
            error_msg = f"tab_id is required for switch_tab action. {tab_info}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if tab_id not in browser_manager._tabs:
            tab_info = await BrowserApiMethods._get_tab_info_for_logs(browser_manager)
            error_msg = f"Tab with ID '{tab_id}' not found. {tab_info}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        browser_manager._page = browser_manager._tabs[tab_id]
        browser_manager._cdp_client = await browser_manager._page.context.new_cdp_session(browser_manager._page)
        browser_manager._active_tab_id = tab_id

        # Use CDP to bring the tab to the foreground
        try:
            await browser_manager._cdp_client.send("Page.bringToFront")
            logger.info(f"Successfully switched to tab '{tab_id}' and brought it to the foreground")
        except Exception as e:
            logger.warning(f"Failed to bring tab '{tab_id}' to foreground: {str(e)}")

        return f"Switched to tab: {tab_id}"

    async def close_tab(page: Page, browser_manager, tab_id: str = None):
        if not tab_id:
            tab_id = browser_manager._active_tab_id

        if tab_id not in browser_manager._tabs:
            raise ValueError(f"Tab with ID '{tab_id}' not found. Available tabs: {list(browser_manager._tabs.keys())}")

        # Close the tab
        await browser_manager._tabs[tab_id].close()

        # Remove from tracking
        del browser_manager._tabs[tab_id]

        # If we closed the active tab, switch to another tab if available
        if tab_id == browser_manager._active_tab_id:
            if browser_manager._tabs:
                next_tab_id = next(iter(browser_manager._tabs.keys()))
                await BrowserApiMethods.switch_tab(page, browser_manager, next_tab_id)
            else:
                browser_manager._page = None
                browser_manager._cdp_client = None
                browser_manager._active_tab_id = None

        logger.info(f"Successfully closed tab '{tab_id}'")
        return f"Closed tab: {tab_id}"

    async def list_tabs(page: Page, browser_manager):
        tabs = await BrowserApiMethods._get_tab_info_for_logs(browser_manager)
        return json.dumps(tabs, indent=2)

    async def get_cookies(page: Page):
        cookies = await page.context.cookies()
        return json.dumps(cookies, indent=2)

    async def set_cookies(page: Page, cookies: List[Dict]):
        await page.context.add_cookies(cookies)
        return "Cookies set successfully"

    async def network_intercept(page: Page, pattern: str):
        await page.route(pattern, lambda route: route.continue_())
        return f"Network interception set for {pattern}"

    async def execute_cdp(page: Page, method: str, params: Dict = None):
        cdp_client = await page.context.new_cdp_session(page)
        result = await cdp_client.send(method, params or {})
        return json.dumps(result, indent=2)

    async def close(page: Page, browser_manager):
        await browser_manager.cleanup()
        return "Browser closed"

    # Api Helper Functions
    async def _get_tab_info_for_logs(self):
        """Get a summary of current tabs for error messages"""
        tabs = {}
        for tab_id, page in self._tabs.items():
            try:
                is_active = tab_id == self._active_tab_id
                tabs[tab_id] = {"url": page.url, "active": is_active}
            except (AttributeError, ConnectionError, Exception) as e:
                tabs[tab_id] = {"error": f"Could not retrieve tab info: {str(e)}"}
        return tabs


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
        self._actions = self._load_actions()
        self._nest_asyncio_applied = False  # Flag to track if nest_asyncio has been applied

    def _load_actions(self) -> Dict[str, Callable]:
        actions = {}
        for name, method in inspect.getmembers(BrowserApiMethods, predicate=inspect.isfunction):
            if not name.startswith("_"):  # Exclude private methods
                actions[name] = method
        return actions

    async def ensure_browser(self, launch_options=None, context_options=None):
        """Initialize browser if not already running."""
        logger.debug("Ensuring browser is running...")

        # Apply nest_asyncio lazily, only when browser is actually needed and only once
        if not self._nest_asyncio_applied:
            nest_asyncio.apply()
            self._nest_asyncio_applied = True
            logger.debug("Applied nest_asyncio for nested event loop support")

        # Ensure required directories exist
        user_data_dir = os.getenv(
            "STRANDS_BROWSER_USER_DATA_DIR", os.path.join(os.path.expanduser("~"), ".browser_automation")
        )
        headless = os.getenv("STRANDS_BROWSER_HEADLESS", "false").lower() == "true"
        width = int(os.getenv("STRANDS_BROWSER_WIDTH", "1280"))
        height = int(os.getenv("STRANDS_BROWSER_HEIGHT", "800"))
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
        max_retries = int(os.getenv("STRANDS_BROWSER_MAX_RETRIES", 3))
        retry_delay = int(os.getenv("STRANDS_BROWSER_RETRY_DELAY", 1))

        async def execute_action():
            if action not in self._actions:
                return [{"text": f"Error: Unknown action {action}"}]

            action_method = self._actions[action]

            # Validate parameters
            sig = inspect.signature(action_method)
            required_params = [p for p in sig.parameters if sig.parameters[p].default == inspect.Parameter.empty]
            for param in required_params:
                if param not in args and param not in ["page", "browser_manager"]:
                    return [{"text": f"Error: Missing required parameter: {param}"}]

            # Execute action
            page, _ = await self.ensure_browser(args.get("launchOptions"))

            # Include self (BrowserManager instance) in the arguments
            action_args = {k: v for k, v in args.items() if k in sig.parameters}
            action_args["page"] = page
            if "browser_manager" in sig.parameters:
                action_args["browser_manager"] = self

            result = await action_method(**action_args)

            return [{"text": str(result)}]

        args = kwargs.get("args", {})

        for attempt in range(max_retries):
            try:
                return await execute_action()
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"Action '{action}' failed after {max_retries} attempts: {str(e)}")
                    return [{"text": f"Error: {str(e)}"}]

                logger.warning(f"Action '{action}' attempt {attempt + 1} failed: {str(e)}")

                # Check for non-retryable errors
                if any(
                    err in str(e).lower()
                    for err in [
                        "could not resolve domain",
                        "connection refused",
                        "ssl/tls error",
                        "certificate error",
                        "protocol error (page.navigate): cannot navigate to invalid url",
                    ]
                ):
                    logger.error(f"Non-retryable error encountered: {str(e)}")
                    return [{"text": f"Error: {str(e)}"}]

                # If it's the evaluate action and there's a JavaScript error, try to fix it
                if action == "evaluate" and "script" in args:
                    error_types = [
                        "SyntaxError",
                        "ReferenceError",
                        "TypeError",
                        "Illegal return",
                        "Unexpected token",
                        "Unexpected end",
                        "is not defined",
                    ]
                    if any(err_type in str(e) for err_type in error_types):
                        fixed_script = await self._fix_javascript_syntax(args["script"], str(e))
                        if fixed_script:
                            args["script"] = fixed_script
                            logger.warning("Attempting retry with fixed JavaScript")
                            continue

                # Exponential backoff
                await asyncio.sleep(retry_delay * (2**attempt))


# Initialize global browser manager
_playwright_manager = BrowserManager()


@tool
def use_browser(
    url: str = None,
    wait_time: int = int(os.getenv("STRANDS_DEFAULT_WAIT_TIME", 1)),
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
            Default is set by STRANDS_DEFAULT_WAIT_TIME env var or 1 second.
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

    logger.debug(f"Tool parameters: {locals()}")
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
