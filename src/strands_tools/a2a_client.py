"""
A2A (Agent-to-Agent) Protocol Client Tool for Strands Agents.

This tool provides functionality to discover and communicate with A2A-compliant agents

Key Features:
- Agent discovery through agent cards from multiple URLs
- Message sending to specific A2A agents

Warning: when using this collection of tools ensure max_parallel_tools>1 for your Strands Agent.
This is typically set by default since the the Strands Agents use cpu count as the default value.
However, if you see "event loop is already running" errors, you should ensure max_parallel_tools>1.
"""

import asyncio
import atexit
import logging
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import AgentCard, Message, MessageSendParams, Part, Role, SendMessageRequest, TextPart
from strands import tool
from strands.types.tools import AgentTool

DEFAULT_TIMEOUT = 30

logger = logging.getLogger(__name__)


class A2AClientToolProvider:
    """A2A Client tool provider that manages multiple A2A agents and exposes synchronous tools."""

    def __init__(
        self,
        known_agent_urls: list[str] | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Initialize A2A client tool provider.

        Args:
            agent_urls: List of A2A agent URLs to use (defaults to None)
            timeout: Timeout for HTTP operations in seconds (defaults to 30)
        """
        self.timeout = timeout
        self._known_agent_urls: list[str] = known_agent_urls or []
        self._discovered_agents: dict[str, AgentCard] = {}
        self._httpx_client: httpx.AsyncClient | None = None
        self._initial_discovery_done: bool = False

        # Register resource cleanup on program exit
        atexit.register(self.close)

    @property
    def tools(self) -> list[AgentTool]:
        """Extract all @tool decorated methods from this instance."""
        tools = []

        for attr_name in dir(self):
            if attr_name == "tools":
                continue

            attr = getattr(self, attr_name)
            if isinstance(attr, AgentTool):
                tools.append(attr)

        return tools

    def _run_async(self, coro):
        """Handle async-to-sync conversion internally.

        This function requires max_parallel_tools>1 on the Strands Agent.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    async def _ensure_httpx_client(self) -> httpx.AsyncClient:
        """Ensure the shared HTTP client is initialized."""
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(timeout=self.timeout)
        return self._httpx_client

    async def _create_a2a_client(self, url: str) -> A2AClient:
        """Create a new A2A client for the given URL."""
        httpx_client = await self._ensure_httpx_client()
        agent_card = await self._async_discover_agent_card(url)
        logger.info(f"A2AClient created for {url}")
        return A2AClient(httpx_client=httpx_client, agent_card=agent_card)

    async def _create_a2a_card_resolver(self, url: str) -> A2ACardResolver:
        """Create a new A2A client for the given URL."""
        httpx_client = await self._ensure_httpx_client()
        logger.info(f"A2ACardResolver created for {url}")
        return A2ACardResolver(httpx_client=httpx_client, base_url=url)

    async def _discover_known_agents(self) -> None:
        """Discover all agents provided during initialization."""

        async def _discover_agent_with_error_handling(url: str):
            """Helper method to discover an agent with error handling."""
            try:
                await self._async_discover_agent_card(url)
            except Exception as e:
                logger.error(f"Failed to discover agent at {url}: {e}")

        tasks = [_discover_agent_with_error_handling(url) for url in self._known_agent_urls]
        if tasks:
            await asyncio.gather(*tasks)

        self._initial_discovery_done = True

    async def _ensure_discovered_known_agents(self) -> None:
        """Ensure initial discovery of agent URLs from constructor has been done."""
        if not self._initial_discovery_done and self._known_agent_urls:
            await self._discover_known_agents()

    async def _async_discover_agent_card(self, url: str) -> AgentCard:
        """Internal method to discover and cache an agent card."""
        if url in self._discovered_agents:
            return self._discovered_agents[url]

        resolver = await self._create_a2a_card_resolver(url)
        agent_card = await resolver.get_agent_card()
        self._discovered_agents[url] = agent_card
        logger.info(f"Successfully discovered and cached agent card for {url}")

        return agent_card

    @tool
    def discover_agent(self, url: str) -> dict[str, Any]:
        """
        Discover an A2A agent and return its agent card with capabilities.

        This function fetches the agent card from the specified A2A agent URL
        and caches it for future use.

        Args:
            url: The base URL of the A2A agent to discover

        Returns:
            dict: Discovery result including:
                - success: Whether the operation succeeded
                - agent_card: The full agent card data (if successful)
                - error: Error message (if failed)
                - url: The agent URL that was queried
        """
        return self._run_async(self._async_discover_agent_card_tool(url))

    async def _async_discover_agent_card_tool(self, url: str) -> dict[str, Any]:
        """Internal async implementation for discover_agent_card tool."""
        try:
            await self._ensure_discovered_known_agents()
            agent_card = await self._async_discover_agent_card(url)
            return {
                "status": "success",
                "agent_card": agent_card.model_dump(mode="python", exclude_none=True),
                "url": url,
            }
        except Exception as e:
            logger.exception(f"Error discovering agent card for {url}")
            return {
                "status": "error",
                "error": str(e),
                "url": url,
            }

    @tool
    def list_discovered_agents(self) -> dict[str, Any]:
        """
        List all discovered A2A agents and their capabilities.

        Returns:
            dict: Information about all discovered agents including:
                - success: Whether the operation succeeded
                - agents: List of discovered agents with their details
                - total_count: Total number of discovered agents
        """
        return self._run_async(self._async_list_discovered_agents())

    async def _async_list_discovered_agents(self) -> dict[str, Any]:
        """Internal async implementation for list_discovered_agents."""
        try:
            await self._ensure_discovered_known_agents()
            agents = [
                agent_card.model_dump(mode="python", exclude_none=True)
                for agent_card in self._discovered_agents.values()
            ]
            return {
                "status": "success",
                "agents": agents,
                "total_count": len(agents),
            }
        except Exception as e:
            logger.exception("Error listing discovered agents")
            return {
                "status": "error",
                "error": str(e),
                "total_count": 0,
            }

    @tool
    def send_message(self, message_text: str, target_agent_url: str, message_id: str | None = None) -> dict[str, Any]:
        """
        Send a message to a specific A2A agent and return the response.

        Args:
            message_text: The message content to send to the agent
            target_agent_url: The URL of the target A2A agent
            message_id: Optional message ID for tracking (generates UUID if not provided)

        Returns:
            dict: Response data including:
                - success: Whether the message was sent successfully
                - response: The agent's response data (if successful)
                - error: Error message (if failed)
                - message_id: The message ID used
                - target_agent_url: The agent URL that was contacted
        """
        return self._run_async(self._async_send_message(message_text, target_agent_url, message_id))

    async def _async_send_message(
        self, message_text: str, target_agent_url: str, message_id: str | None = None
    ) -> dict[str, Any]:
        """Internal async implementation for send_message."""

        try:
            await self._ensure_discovered_known_agents()
            client = await self._create_a2a_client(target_agent_url)

            if message_id is None:
                message_id = uuid4().hex

            message = Message(
                kind="message",
                role=Role.user,
                parts=[Part(TextPart(kind="text", text=message_text))],
                messageId=message_id,
            )

            request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(message=message))

            logger.info(f"Sending message to {target_agent_url}")
            response = await client.send_message(request)

            return {
                "status": "success",
                "response": response.model_dump(mode="python", exclude_none=True),
                "message_id": message_id,
                "target_agent_url": target_agent_url,
            }

        except Exception as e:
            logger.exception(f"Error sending message to {target_agent_url}")
            return {
                "status": "error",
                "error": str(e),
                "message_id": message_id,
                "target_agent_url": target_agent_url,
            }

    def close(self):
        """Close the HTTP client and clean up resources. Safe to call multiple times."""
        if self._httpx_client is not None:
            logger.debug("Closing HTTP client and cleaning up resources")
            self._run_async(self._async_close())
            self._httpx_client = None
            self._discovered_agents.clear()
            logger.info("A2AClientToolProvider resources cleaned up")
        else:
            logger.debug("A2AClientToolProvider already cleaned up, skipping")

    async def _async_close(self):
        """Internal async method to close the HTTP client."""
        if self._httpx_client is not None:
            await self._httpx_client.aclose()
