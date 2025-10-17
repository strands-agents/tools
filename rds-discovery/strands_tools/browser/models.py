"""
Pydantic models for Browser tool.

This module contains all the Pydantic models used for type-safe action definitions
with discriminated unions, ensuring required fields are present for each action type.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import BrowserContext, Page
from pydantic import BaseModel, Field


@dataclass
class BrowserSession:
    """Complete browser session state encapsulation."""

    session_name: str
    description: str
    browser: Optional[PlaywrightBrowser] = None  # Browser instance
    context: Optional[BrowserContext] = None  # BrowserContext instance
    page: Optional[Page] = None  # Main Page instance
    tabs: Dict[str, Page] = field(default_factory=dict)  # Dict of tab_id -> Page
    active_tab_id: Optional[str] = None

    async def close(self):
        """Close all session resources."""
        cleanup_errors = []

        # Close browser (this will close all contexts and pages)
        if self.browser:
            try:
                await self.browser.close()
            except Exception as e:
                cleanup_errors.append(f"Error closing browser: {str(e)}")

        # Clear references
        self.browser = None
        self.context = None
        self.page = None
        self.tabs.clear()
        self.active_tab_id = None

        return cleanup_errors

    def get_active_page(self) -> Optional[Page]:
        """Get the currently active page."""
        if self.active_tab_id and self.active_tab_id in self.tabs:
            return self.tabs[self.active_tab_id]
        return self.page

    def add_tab(self, tab_id: str, page: Page) -> None:
        """Add a new tab to the session and updates the active tab."""
        self.tabs[tab_id] = page
        self.active_tab_id = tab_id

    def switch_tab(self, tab_id: str) -> bool:
        """Switch to a different tab. Returns True if successful."""
        if tab_id in self.tabs:
            self.active_tab_id = tab_id
            return True
        return False

    def remove_tab(self, tab_id: str) -> bool:
        """Remove a tab from the session. Returns True if successful."""
        if tab_id in self.tabs:
            del self.tabs[tab_id]
            if self.active_tab_id == tab_id:
                # Switch to another tab if available
                if self.tabs:
                    self.active_tab_id = next(iter(self.tabs.keys()))
                else:
                    self.active_tab_id = None
            return True
        return False


# Session Management Actions
class InitSessionAction(BaseModel):
    """Action for creating a new browser session. Use this as the first step before any browser automation tasks.
    Required before navigating to websites, clicking elements, or performing any browser operations."""

    type: Literal["init_session"] = Field(description="Initialize a new browser session")
    description: str = Field(description="Required description of what this session will be used for")
    session_name: str = Field(
        pattern="^[a-z0-9-]+$", min_length=10, max_length=36, description="Required name to identify the session"
    )


class ListLocalSessionsAction(BaseModel):
    """Action for viewing all active browser sessions. Use this to see what browser sessions are currently
    available for interaction, including their names and descriptions."""

    type: Literal["list_local_sessions"] = Field(description="List all local sessions managed by this tool instance")


# Browser Action Models
class NavigateAction(BaseModel):
    """Action for navigating to a specific URL. Use this to load web pages, visit websites, or change the
    current page location. Must have an active session before using."""

    type: Literal["navigate"] = Field(description="Navigate to a URL")
    session_name: str = Field(description="Required session name from a previous init_session call")
    url: str = Field(description="URL to navigate to")


class ClickAction(BaseModel):
    """Action for clicking on web page elements. Use this to interact with buttons, links, checkboxes, or any
    clickable element. Requires a CSS selector to identify the target element."""

    type: Literal["click"] = Field(description="Click on an element")
    session_name: str = Field(description="Required session name from a previous init_session call")
    selector: str = Field(description="CSS selector for the element to click")


class TypeAction(BaseModel):
    """Action for entering text into input fields. Use this to fill out forms, search boxes, text areas, or any
    text input element. Requires a CSS selector to identify the input field."""

    type: Literal["type"] = Field(description="Type text into an element")
    session_name: str = Field(description="Required session name from a previous init_session call")
    selector: str = Field(description="CSS selector for the element to type into")
    text: str = Field(description="Text to type")


class EvaluateAction(BaseModel):
    """Action for executing JavaScript code in the browser context. Use this to run custom scripts, manipulate DOM
    elements, extract complex data, or perform advanced browser operations that aren't covered by other actions."""

    type: Literal["evaluate"] = Field(description="Execute JavaScript code")
    session_name: str = Field(description="Required session name from a previous init_session call")
    script: str = Field(description="JavaScript code to execute")


class PressKeyAction(BaseModel):
    """Action for simulating keyboard key presses. Use this to submit forms (Enter), navigate between fields (Tab),
    close dialogs (Escape), or trigger keyboard shortcuts. Useful when clicking isn't sufficient."""

    type: Literal["press_key"] = Field(description="Press a keyboard key")
    session_name: str = Field(description="Required session name from a previous init_session call")
    key: str = Field(description="Key to press (e.g., 'Enter', 'Tab', 'Escape')")


class GetTextAction(BaseModel):
    """Action for extracting text content from web page elements. Use this to read visible text from specific
    elements like headings, paragraphs, labels, or any element containing text data you need to capture."""

    type: Literal["get_text"] = Field(description="Get text content from an element")
    session_name: str = Field(description="Required session name from a previous init_session call")
    selector: str = Field(description="CSS selector for the element")


class GetHtmlAction(BaseModel):
    """Action for extracting HTML source code from the page or specific elements. Use this to get the raw HTML
    structure, analyze page markup, or extract complex nested content that text extraction can't capture."""

    type: Literal["get_html"] = Field(description="Get HTML content")
    session_name: str = Field(description="Required session name from a previous init_session call")
    selector: Optional[str] = Field(default=None, description="CSS selector for specific element (optional)")


class ScreenshotAction(BaseModel):
    """Action for capturing visual screenshots of the current page. Use this to document the current state, verify
    visual elements, debug layout issues, or create visual records of web page interactions."""

    type: Literal["screenshot"] = Field(description="Take a screenshot")
    session_name: str = Field(description="Required session name from a previous init_session call")
    path: Optional[str] = Field(default=None, description="Optional path for screenshot file")


class RefreshAction(BaseModel):
    """Action for reloading the current web page. Use this to refresh dynamic content, reset form states, reload
    updated data, or recover from page errors by forcing a fresh page load."""

    type: Literal["refresh"] = Field(description="Refresh the current page")
    session_name: str = Field(description="Required session name from a previous init_session call")


class BackAction(BaseModel):
    """Action for navigating to the previous page in browser history. Use this to return to previously visited
    pages, undo navigation steps, or move backwards through a multi-step process."""

    type: Literal["back"] = Field(description="Navigate back in browser history")
    session_name: str = Field(description="Required session name from a previous init_session call")


class ForwardAction(BaseModel):
    """Action for navigating to the next page in browser history. Use this to move forward through previously
    visited pages after using the back action, or to redo navigation steps."""

    type: Literal["forward"] = Field(description="Navigate forward in browser history")
    session_name: str = Field(description="Required session name from a previous init_session call")


class NewTabAction(BaseModel):
    """Action for creating a new browser tab within the current session. Use this to open additional pages
    simultaneously, compare content across multiple sites, or maintain separate workflows in parallel.
    After using this action, the default tab is automatically switched to the new tab"""

    type: Literal["new_tab"] = Field(description="Create a new browser tab")
    session_name: str = Field(description="Required session name from a previous init_session call")
    tab_id: Optional[str] = Field(default=None, description="Optional ID for the new tab")


class SwitchTabAction(BaseModel):
    """Action for changing focus to a different browser tab. Use this to switch between multiple open tabs,
    continue work on a previously opened page, or alternate between different websites."""

    type: Literal["switch_tab"] = Field(description="Switch to a different tab")
    session_name: str = Field(description="Required session name from a previous init_session call")
    tab_id: str = Field(description="ID of the tab to switch to")


class CloseTabAction(BaseModel):
    """Action for closing a specific browser tab or the currently active tab. Use this to clean up completed
    workflows, free browser resources, or close tabs that are no longer needed."""

    type: Literal["close_tab"] = Field(description="Close a browser tab")
    session_name: str = Field(description="Required session name from a previous init_session call")
    tab_id: Optional[str] = Field(default=None, description="ID of the tab to close (defaults to active tab)")


class ListTabsAction(BaseModel):
    """Action for viewing all open browser tabs in the current session. Use this to see what tabs are available,
    get their IDs for switching, or manage multiple open pages."""

    type: Literal["list_tabs"] = Field(description="List all open browser tabs")
    session_name: str = Field(description="Required session name from a previous init_session call")


class GetCookiesAction(BaseModel):
    """Action for retrieving all cookies from the current page or domain. Use this to inspect authentication tokens,
    session data, user preferences, or any stored cookie information for debugging or data extraction."""

    type: Literal["get_cookies"] = Field(description="Get all cookies for the current page")
    session_name: str = Field(description="Required session name from a previous init_session call")


class SetCookiesAction(BaseModel):
    """Action for setting or modifying cookies on the current page or domain. Use this to simulate user
    authentication, set preferences, maintain session state, or inject specific cookie values for testing purposes."""

    type: Literal["set_cookies"] = Field(description="Set cookies for the current page")
    session_name: str = Field(description="Required session name from a previous init_session call")
    cookies: List[Dict] = Field(description="List of cookie objects to set")


class NetworkInterceptAction(BaseModel):
    """Action for intercepting and monitoring network requests matching a URL pattern. Use this to capture API calls,
    monitor data exchanges, debug network issues, or analyze communication between the browser and servers."""

    type: Literal["network_intercept"] = Field(description="Set up network interception")
    session_name: str = Field(description="Required session name from a previous init_session call")
    pattern: str = Field(description="URL pattern to intercept")


class ExecuteCdpAction(BaseModel):
    """Action for executing Chrome DevTools Protocol commands directly. Use this for advanced browser control,
    performance monitoring, security testing, or accessing low-level browser features not available
    through standard actions."""

    type: Literal["execute_cdp"] = Field(description="Execute Chrome DevTools Protocol command")
    session_name: str = Field(description="Required session name from a previous init_session call")
    method: str = Field(description="CDP method name")
    params: Optional[Dict] = Field(default=None, description="Parameters for the CDP method")


class CloseAction(BaseModel):
    """Action for completely closing the browser and ending the session. Use this to clean up resources, terminate
    automation workflows, or properly shut down the browser when all tasks are completed."""

    type: Literal["close"] = Field(description="Close the browser")
    session_name: str = Field(description="Required session name from a previous init_session call")


class BrowserInput(BaseModel):
    """Input model for browser actions."""

    action: Union[
        InitSessionAction,
        ListLocalSessionsAction,
        NavigateAction,
        ClickAction,
        TypeAction,
        EvaluateAction,
        PressKeyAction,
        GetTextAction,
        GetHtmlAction,
        ScreenshotAction,
        RefreshAction,
        BackAction,
        ForwardAction,
        NewTabAction,
        SwitchTabAction,
        CloseTabAction,
        ListTabsAction,
        GetCookiesAction,
        SetCookiesAction,
        NetworkInterceptAction,
        ExecuteCdpAction,
        CloseAction,
    ] = Field(discriminator="type")
    wait_time: Optional[int] = Field(default=2, description="Time to wait after action in seconds")
