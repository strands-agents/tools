"""
Bedrock AgentCore Browser implementation of Browser.

This module provides a Bedrock AgentCore browser implementation that connects to
AWS-hosted browser instances.
"""

import logging
from typing import Dict, Optional

from bedrock_agentcore.tools.browser_client import BrowserClient as AgentCoreBrowserClient
from playwright.async_api import Browser as PlaywrightBrowser

from ..utils.aws_util import resolve_region
from .browser import Browser

logger = logging.getLogger(__name__)

DEFAULT_IDENTIFIER = "aws.browser.v1"


class AgentCoreBrowser(Browser):
    """Bedrock AgentCore browser implementation."""

    def __init__(self, region: Optional[str] = None, identifier: str = DEFAULT_IDENTIFIER, session_timeout: int = 3600):
        """
        Initialize the browser.

        Args:
            region: AWS region for the browser service
            identifier: Browser service identifier
            session_timeout: Session timeout in seconds (default: 3600)
        """
        super().__init__()
        self.region = resolve_region(region)
        self.identifier = identifier
        self.session_timeout = session_timeout
        self._client_dict: Dict[str, AgentCoreBrowserClient] = {}

    def start_platform(self) -> None:
        """Remote platform does not need additional initialization steps."""
        pass

    async def create_browser_session(self) -> PlaywrightBrowser:
        """Create a new browser instance for a session."""
        if not self._playwright:
            raise RuntimeError("Playwright not initialized")

        # Create new browser client for this session
        session_client = AgentCoreBrowserClient(region=self.region)
        session_id = session_client.start(identifier=self.identifier, session_timeout_seconds=self.session_timeout)

        logger.info(f"started Bedrock AgentCore browser session: {session_id}")

        # Get CDP connection details
        cdp_url, cdp_headers = session_client.generate_ws_headers()

        # Connect to Bedrock AgentCore browser over CDP
        browser = await self._playwright.chromium.connect_over_cdp(endpoint_url=cdp_url, headers=cdp_headers)

        return browser

    def close_platform(self) -> None:
        for client in self._client_dict.values():
            try:
                client.stop()
            except Exception as e:
                logger.error(
                    "session=<%s>, exception=<%s> " "| failed to close session , relying on idle timeout to auto close",
                    client.session_id,
                    str(e),
                )
