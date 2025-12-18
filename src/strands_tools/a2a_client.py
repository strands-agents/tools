"""
A2A (Agent-to-Agent) Protocol Client Tool for Strands Agents.

This tool provides functionality to discover and communicate with A2A-compliant agents

Key Features:
- Agent discovery through agent cards from multiple URLs
- Message sending to specific A2A agents
- Push notification support for real-time task completion alerts
- Custom authentication support via httpx client arguments

Usage Examples:

    Basic usage without authentication:
        >>> provider = A2AClientToolProvider(
        ...     known_agent_urls=["http://agent1.example.com", "http://agent2.example.com"]
        ... )

    With OAuth/Bearer token authentication:
        >>> provider = A2AClientToolProvider(
        ...     known_agent_urls=["http://secure-agent.example.com"],
        ...     httpx_client_args={
        ...         "headers": {"Authorization": "Bearer your-token-here"},
        ...         "timeout": 300
        ...     }
        ... )
"""

import asyncio
import logging
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import AgentCard, Message, Part, PushNotificationConfig, Role, TextPart
from strands import tool
from strands.types.tools import AgentTool

DEFAULT_TIMEOUT = 300  # set request timeout to 5 minutes

logger = logging.getLogger(__name__)


class A2AClientToolProvider:
    """A2A Client tool provider that manages multiple A2A agents and exposes synchronous tools."""

    def __init__(
        self,
        known_agent_urls: list[str] | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        webhook_url: str | None = None,
        webhook_token: str | None = None,
        httpx_client_args: dict[str, Any] | None = None,
    ):
        """
        Initialize A2A client tool provider.

        Args:
            known_agent_urls: List of A2A agent URLs to use (defaults to None)
            timeout: Timeout for HTTP operations in seconds (defaults to 300)
            webhook_url: Optional webhook URL for push notifications
            webhook_token: Optional authentication token for webhook notifications
            httpx_client_args: Optional dictionary of arguments to pass to httpx.AsyncClient
                constructor. This allows custom auth, headers, proxies, etc.
                Example: {"headers": {"Authorization": "Bearer token"}, "timeout": 60}

                Note: To avoid event loop issues in multi-turn conversations,
                a fresh client is created for each async operation using these args.
                This prevents "Event loop is closed" errors when the provider is used
                across multiple asyncio.run() calls.
        """
        self.timeout = timeout
        self._known_agent_urls: list[str] = known_agent_urls or []
        self._discovered_agents: dict[str, AgentCard] = {}

        # Store client args instead of client instance to avoid event loop issues
        self._httpx_client_args: dict[str, Any] = httpx_client_args or {}

        # Set default timeout if not provided in client args
        if "timeout" not in self._httpx_client_args:
            self._httpx_client_args["timeout"] = self.timeout

        self._initial_discovery_done: bool = False

        # Push notification configuration
        self._webhook_url = webhook_url
        self._webhook_token = webhook_token
        self._push_config: PushNotificationConfig | None = None

        if self._webhook_url and self._webhook_token:
            self._push_config = PushNotificationConfig(
                id=f"strands-webhook-{uuid4().hex[:8]}", url=self._webhook_url, token=self._webhook_token
            )

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

    def _get_httpx_client(self) -> httpx.AsyncClient:
        """
        Get a fresh httpx client for the current operation.

        Creates a new client using the stored client args. This prevents event loop
        issues when the provider is used across multiple asyncio.run() calls.

        Similar to the Gemini model provider fix in strands-agents/sdk-python#932,
        we create fresh clients per operation rather than reusing a single instance.
        """
        return httpx.AsyncClient(**self._httpx_client_args)

    def _get_client_factory(self) -> ClientFactory:
        """
        Get a ClientFactory for the current operation.

        Creates a fresh ClientFactory with a fresh httpx client for each call to avoid
        event loop issues when the provider is used across multiple asyncio.run() calls.

        Note: We don't cache the ClientFactory because it contains the httpx client,
        which would cause "Event loop is closed" errors in multi-turn conversations.
        """
        httpx_client = self._get_httpx_client()
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=False,  # Use non-streaming mode for simpler response handling
            push_notification_configs=[self._push_config] if self._push_config else [],
        )
        return ClientFactory(config)

    async def _create_a2a_card_resolver(self, url: str) -> A2ACardResolver:
        """Create a new A2A card resolver for the given URL."""
        httpx_client = self._get_httpx_client()
        logger.info(f"A2ACardResolver created for {url}")
        return A2ACardResolver(httpx_client=httpx_client, base_url=url)

    async def _discover_known_agents(self) -> None:
        """Discover all agents provided during initialization."""

        async def _discover_agent_with_error_handling(url: str):
            """Helper method to discover an agent with error handling."""
            try:
                await self._discover_agent_card(url)
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

    async def _discover_agent_card(self, url: str) -> AgentCard:
        """Internal method to discover and cache an agent card."""
        if url in self._discovered_agents:
            return self._discovered_agents[url]

        resolver = await self._create_a2a_card_resolver(url)
        agent_card = await resolver.get_agent_card()
        self._discovered_agents[url] = agent_card
        logger.info(f"Successfully discovered and cached agent card for {url}")

        return agent_card

    @tool
    async def a2a_discover_agent(self, url: str) -> dict[str, Any]:
        """
        Discover an A2A agent and return its agent card with capabilities.

        This function fetches the agent card from the specified A2A agent URL
        and caches it for future use. Use this when you need to discover a new
        agent that is not in the known agents list.

        Args:
            url: The base URL of the A2A agent to discover

        Returns:
            dict: Discovery result including:
                - success: Whether the operation succeeded
                - agent_card: The full agent card data (if successful)
                - error: Error message (if failed)
                - url: The agent URL that was queried
        """
        return await self._discover_agent_card_tool(url)

    async def _discover_agent_card_tool(self, url: str) -> dict[str, Any]:
        """Internal async implementation for discover_agent_card tool."""
        try:
            await self._ensure_discovered_known_agents()
            agent_card = await self._discover_agent_card(url)
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
    async def a2a_list_discovered_agents(self) -> dict[str, Any]:
        """
        List all discovered A2A agents and their capabilities.

        Returns:
            dict: Information about all discovered agents including:
                - success: Whether the operation succeeded
                - agents: List of discovered agents with their details
                - total_count: Total number of discovered agents
        """
        return await self._list_discovered_agents()

    async def _list_discovered_agents(self) -> dict[str, Any]:
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
    async def a2a_send_message(
        self, message_text: str, target_agent_url: str, message_id: str | None = None
    ) -> dict[str, Any]:
        """
        Send a message to a specific A2A agent and return the response.

        IMPORTANT: If the user provides a specific URL, use it directly. If the user
        refers to an agent by name only, use a2a_list_discovered_agents first to get
        the correct URL. Never guess, generate, or hallucinate URLs.

        Args:
            message_text: The message content to send to the agent
            target_agent_url: The exact URL of the target A2A agent
                (user-provided URL or from a2a_list_discovered_agents)
            message_id: Optional message ID for tracking (generates UUID if not provided)

        Returns:
            dict: Response data including:
                - success: Whether the message was sent successfully
                - response: The agent's response data (if successful)
                - error: Error message (if failed)
                - message_id: The message ID used
                - target_agent_url: The agent URL that was contacted
        """
        return await self._send_message(message_text, target_agent_url, message_id)

    async def _send_message(
        self, message_text: str, target_agent_url: str, message_id: str | None = None
    ) -> dict[str, Any]:
        """Internal async implementation for send_message."""

        try:
            await self._ensure_discovered_known_agents()

            # Get the agent card and create client using factory
            agent_card = await self._discover_agent_card(target_agent_url)
            client_factory = self._get_client_factory()
            client = client_factory.create(agent_card)

            if message_id is None:
                message_id = uuid4().hex

            message = Message(
                kind="message",
                role=Role.user,
                parts=[Part(TextPart(kind="text", text=message_text))],
                message_id=message_id,
            )

            logger.info(f"Sending message to {target_agent_url}")

            # With streaming=False, this will yield exactly one result
            async for event in client.send_message(message):
                if isinstance(event, Message):
                    # Direct message response
                    return {
                        "status": "success",
                        "response": event.model_dump(mode="python", exclude_none=True),
                        "message_id": message_id,
                        "target_agent_url": target_agent_url,
                    }
                elif isinstance(event, tuple) and len(event) == 2:
                    # (Task, UpdateEvent) tuple - extract the task
                    task, update_event = event
                    return {
                        "status": "success",
                        "response": {
                            "task": task.model_dump(mode="python", exclude_none=True),
                            "update": (
                                update_event.model_dump(mode="python", exclude_none=True) if update_event else None
                            ),
                        },
                        "message_id": message_id,
                        "target_agent_url": target_agent_url,
                    }
                else:
                    # Fallback for unexpected response types
                    return {
                        "status": "success",
                        "response": {"raw_response": str(event)},
                        "message_id": message_id,
                        "target_agent_url": target_agent_url,
                    }

            # This should never be reached with streaming=False
            return {
                "status": "error",
                "error": "No response received from agent",
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
