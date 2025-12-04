"""
Browser automation tool with inheritance-based architecture.

This module provides browser automation capabilities through an inheritance-based
architecture similar to the code interpreter tool, where different browser implementations
inherit from a common base class.

Available Browser Implementations:
- LocalChromiumBrowser: Local Chromium browser using Playwright
- AgentCoreBrowser: Remote browser via Bedrock AgentCore

Usage:
    ```python
    from strands import Agent
    from strands_tools.browser import LocalChromiumBrowser

    # Create browser tool with local Chromium
    browser = LocalChromiumBrowser()
    agent = Agent(tools=[browser.browser])

    # Use the browser
    agent.tool.browser(
        browser_input={
            "action": {
                "type": "init_session",
                "description": "Example ession",
                "session_name": "example-session"
            }
        }
    )

    agent.tool.browser(
        browser_input={
            "action": {
                "type": "navigate",
                "url": "https://example.com",
                "session_name": "example-session"
            }
        }
    )

    agent.tool.browser(
        browser_input={
            "action": {
                "type": "close",
                "session_name": "example-session"
            }
        }
    )
    ```
"""

from typing import TYPE_CHECKING

from .browser import Browser

# Type checking imports for static analysis
if TYPE_CHECKING:
    from .agent_core_browser import AgentCoreBrowser
    from .local_chromium_browser import LocalChromiumBrowser

__all__ = [
    # Base class
    "Browser",
    # Browser implementations
    "LocalChromiumBrowser",
    "AgentCoreBrowser",
]


def __getattr__(name: str):
    """
    Lazy load browser implementations only when accessed.

    This defers the import of optional dependencies until actually needed:
    - LocalChromiumBrowser requires playwright (lazy loaded)
    - AgentCoreBrowser requires bedrock_agentcore (lazy loaded)
    """
    if name == "LocalChromiumBrowser":
        from .local_chromium_browser import LocalChromiumBrowser

        return LocalChromiumBrowser
    if name == "AgentCoreBrowser":
        from .agent_core_browser import AgentCoreBrowser

        return AgentCoreBrowser
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
