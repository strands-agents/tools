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
    agent.tool.browser({
        "action": {
            "type": "navigate",
            "url": "https://example.com"
        }
    })
    ```
"""

from .agent_core_browser import AgentCoreBrowser
from .browser import Browser
from .local_chromium_browser import LocalChromiumBrowser

__all__ = [
    # Base class
    "Browser",
    # Browser implementations
    "LocalChromiumBrowser",
    "AgentCoreBrowser",
]
