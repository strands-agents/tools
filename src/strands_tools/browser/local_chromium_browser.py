"""
Local Chromium Browser implementation using Playwright.

This module provides a local Chromium browser implementation that runs
browser instances on the local machine using Playwright.
"""

import logging
import os
from typing import Any, Dict, Optional

from playwright.async_api import Browser as PlaywrightBrowser

from .browser import Browser

logger = logging.getLogger(__name__)


class LocalChromiumBrowser(Browser):
    """Local Chromium browser implementation using Playwright."""

    def __init__(
        self, launch_options: Optional[Dict[str, Any]] = None, context_options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the local Chromium browser.

        Args:
            launch_options: Chromium-specific launch options (headless, args, etc.)
            context_options: Browser context options (viewport, user agent, etc.)
        """
        super().__init__()
        self._launch_options = launch_options or {}
        self._context_options = context_options or {}
        self._default_launch_options: Dict[str, Any] = {}
        self._default_context_options: Dict[str, Any] = {}

    def start_platform(self) -> None:
        """Initialize the local Chromium browser platform with configuration."""
        # Read environment variables
        user_data_dir = os.getenv(
            "STRANDS_BROWSER_USER_DATA_DIR", os.path.join(os.path.expanduser("~"), ".browser_automation")
        )
        headless = os.getenv("STRANDS_BROWSER_HEADLESS", "false").lower() == "true"
        width = int(os.getenv("STRANDS_BROWSER_WIDTH", "1280"))
        height = int(os.getenv("STRANDS_BROWSER_HEIGHT", "800"))

        # Ensure user data directory exists
        os.makedirs(user_data_dir, exist_ok=True)

        # Build default launch options
        self._default_launch_options = {
            "headless": headless,
            "args": [f"--window-size={width},{height}"],
        }
        self._default_launch_options.update(self._launch_options)

        # Build default context options
        self._default_context_options = {"viewport": {"width": width, "height": height}}
        self._default_context_options.update(self._context_options)

    async def create_browser_session(self) -> PlaywrightBrowser:
        """Create a new local Chromium browser instance for a session."""
        if not self._playwright:
            raise RuntimeError("Playwright not initialized")

        # Handle persistent context if specified
        if self._default_launch_options.get("persistent_context"):
            persistent_user_data_dir = self._default_launch_options.get(
                "user_data_dir", os.path.join(os.path.expanduser("~"), ".browser_automation")
            )

            # For persistent context, return the context itself as it acts like a browser
            context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=persistent_user_data_dir,
                **{
                    k: v
                    for k, v in self._default_launch_options.items()
                    if k not in ["persistent_context", "user_data_dir"]
                },
            )
            return context
        else:
            # Regular browser launch
            logger.debug("launching local Chromium session browser with options: %s", self._default_launch_options)
            return await self._playwright.chromium.launch(**self._default_launch_options)

    def close_platform(self) -> None:
        """Close the local Chromium browser. No platform specific changes needed"""
        pass
