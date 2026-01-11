"""
A2A (Agent-to-Agent) Protocol Client Tool for Strands Agents.

This tool provides functionality to discover and communicate with A2A-compliant agents

Key Features:
- Agent discovery through agent cards from multiple URLs
- Message sending to specific A2A agents
- Context and task ID persistence for multi-turn conversations
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

    Multi-turn conversation with context persistence:
        >>> provider = A2AClientToolProvider(known_agent_urls=["http://agent.example.com"])
        >>> # First message - server returns context_id and task_id
        >>> result1 = provider.a2a_send_message("Start a task", "http://agent.example.com")
        >>> # Second message - automatically includes context_id and task_id
        >>> result2 = provider.a2a_send_message("Continue the task", "http://agent.example.com")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import AgentCard, Message, Part, PushNotificationConfig, Role, TaskState, TextPart
from strands import tool
from strands.types.tools import AgentTool

DEFAULT_TIMEOUT = 300  # set request timeout to 5 minutes

# Terminal task states as defined by A2A protocol
TERMINAL_TASK_STATES = frozenset({TaskState.completed, TaskState.canceled, TaskState.failed, TaskState.rejected})

logger = logging.getLogger(__name__)


@dataclass
class ActiveTask:
    """Represents an active (non-terminal) task for an agent."""

    task_id: str
    state: TaskState


@dataclass
class ConversationState:
    """Tracks conversation state including context_id and active tasks per agent."""

    context_id: str | None = None
    # Maps target_agent_url to active task
    active_tasks: dict[str, ActiveTask] = field(default_factory=dict)


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

        # Conversation state for context_id and task_id persistence
        self._conversation_state = ConversationState()

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

    def _update_conversation_state_from_response(
        self, response_data: dict[str, Any], target_agent_url: str
    ) -> None:
        """
        Update conversation state from server response.

        Extracts context_id and task_id from the response and updates internal state:
        - context_id is stored once and reused for all subsequent messages
        - task_id is tracked per agent and cleared when task reaches terminal state

        Args:
            response_data: The response data from the server
            target_agent_url: The URL of the agent that sent the response
        """
        # Extract context_id from response (only set if not already set)
        if "context_id" in response_data and self._conversation_state.context_id is None:
            self._conversation_state.context_id = response_data["context_id"]
            logger.info(f"Stored context_id: {self._conversation_state.context_id}")

        # Handle task response
        task_data = response_data.get("task")
        if task_data:
            task_id = task_data.get("id")
            task_state_str = task_data.get("status", {}).get("state")

            if task_id and task_state_str:
                try:
                    task_state = TaskState(task_state_str)

                    if task_state in TERMINAL_TASK_STATES:
                        # Remove task from active tracking when terminal
                        if target_agent_url in self._conversation_state.active_tasks:
                            del self._conversation_state.active_tasks[target_agent_url]
                            logger.info(
                                f"Cleared task_id for {target_agent_url} "
                                f"(terminal state: {task_state_str})"
                            )
                    else:
                        # Store/update active task
                        self._conversation_state.active_tasks[target_agent_url] = ActiveTask(
                            task_id=task_id, state=task_state
                        )
                        logger.info(
                            f"Stored task_id {task_id} for {target_agent_url} "
                            f"(state: {task_state_str})"
                        )
                except ValueError:
                    # Unknown task state, log but don't fail
                    logger.warning(f"Unknown task state: {task_state_str}")

    def _get_task_id_for_agent(self, target_agent_url: str, explicit_task_id: str | None) -> str | None:
        """
        Determine the task_id to use for a message.

        Args:
            target_agent_url: The URL of the target agent
            explicit_task_id: Explicitly provided task_id (overrides stored value)

        Returns:
            The task_id to use, or None if no task should be continued
        """
        # Explicit task_id always takes precedence
        if explicit_task_id is not None:
            return explicit_task_id

        # Check for active task for this agent
        active_task = self._conversation_state.active_tasks.get(target_agent_url)
        if active_task:
            return active_task.task_id

        return None

    def reset_conversation(self) -> None:
        """
        Reset conversation state, clearing context_id and all active tasks.

        Call this method to start a fresh conversation session.
        """
        self._conversation_state = ConversationState()
        logger.info("Conversation state reset")

    def get_conversation_state(self) -> dict[str, Any]:
        """
        Get current conversation state for debugging/inspection.

        Returns:
            dict containing context_id and active tasks
        """
        return {
            "context_id": self._conversation_state.context_id,
            "active_tasks": {
                url: {"task_id": task.task_id, "state": task.state.value}
                for url, task in self._conversation_state.active_tasks.items()
            },
        }

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
        self,
        message_text: str,
        target_agent_url: str,
        message_id: str | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Send a message to a specific A2A agent and return the response.

        This method automatically persists and reuses context_id and task_id across
        multiple messages to enable multi-turn conversations. The server-returned
        context_id is stored on the first response and included in all subsequent
        requests. Task IDs are tracked per agent until they reach a terminal state
        (completed, canceled, failed, rejected).

        IMPORTANT: If the user provides a specific URL, use it directly. If the user
        refers to an agent by name only, use a2a_list_discovered_agents first to get
        the correct URL. Never guess, generate, or hallucinate URLs.

        Args:
            message_text: The message content to send to the agent
            target_agent_url: The exact URL of the target A2A agent
                (user-provided URL or from a2a_list_discovered_agents)
            message_id: Optional message ID for tracking (generates UUID if not provided)
            context_id: Optional context ID to override the stored context_id.
                If not provided, uses the context_id from previous responses.
            task_id: Optional task ID to continue a specific task.
                If not provided, automatically uses the active task_id for this agent
                (if one exists and is in a non-terminal state).

        Returns:
            dict: Response data including:
                - success: Whether the message was sent successfully
                - response: The agent's response data (if successful)
                - error: Error message (if failed)
                - message_id: The message ID used
                - target_agent_url: The agent URL that was contacted
                - context_id: The context_id used (if any)
                - task_id: The task_id used (if any)
        """
        return await self._send_message(message_text, target_agent_url, message_id, context_id, task_id)

    async def _send_message(
        self,
        message_text: str,
        target_agent_url: str,
        message_id: str | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
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

            # Determine context_id to use (explicit > stored)
            effective_context_id = context_id if context_id is not None else self._conversation_state.context_id

            # Determine task_id to use (explicit > active task for this agent)
            effective_task_id = self._get_task_id_for_agent(target_agent_url, task_id)

            message = Message(
                kind="message",
                role=Role.user,
                parts=[Part(TextPart(kind="text", text=message_text))],
                message_id=message_id,
                context_id=effective_context_id,
                task_id=effective_task_id,
            )

            logger.info(
                f"Sending message to {target_agent_url} "
                f"(context_id={effective_context_id}, task_id={effective_task_id})"
            )

            # With streaming=False, this will yield exactly one result
            async for event in client.send_message(message):
                if isinstance(event, Message):
                    # Direct message response
                    response_data = event.model_dump(mode="python", exclude_none=True)
                    self._update_conversation_state_from_response(response_data, target_agent_url)
                    return {
                        "status": "success",
                        "response": response_data,
                        "message_id": message_id,
                        "target_agent_url": target_agent_url,
                        "context_id": effective_context_id,
                        "task_id": effective_task_id,
                    }
                elif isinstance(event, tuple) and len(event) == 2:
                    # (Task, UpdateEvent) tuple - extract the task
                    task, update_event = event
                    response_data = {
                        "task": task.model_dump(mode="python", exclude_none=True),
                        "update": (
                            update_event.model_dump(mode="python", exclude_none=True) if update_event else None
                        ),
                    }
                    # Update conversation state from task response
                    self._update_conversation_state_from_response(response_data, target_agent_url)
                    return {
                        "status": "success",
                        "response": response_data,
                        "message_id": message_id,
                        "target_agent_url": target_agent_url,
                        "context_id": effective_context_id,
                        "task_id": effective_task_id,
                    }
                else:
                    # Fallback for unexpected response types
                    return {
                        "status": "success",
                        "response": {"raw_response": str(event)},
                        "message_id": message_id,
                        "target_agent_url": target_agent_url,
                        "context_id": effective_context_id,
                        "task_id": effective_task_id,
                    }

            # This should never be reached with streaming=False
            return {
                "status": "error",
                "error": "No response received from agent",
                "message_id": message_id,
                "target_agent_url": target_agent_url,
                "context_id": effective_context_id,
                "task_id": effective_task_id,
            }

        except Exception as e:
            logger.exception(f"Error sending message to {target_agent_url}")
            return {
                "status": "error",
                "error": str(e),
                "message_id": message_id,
                "target_agent_url": target_agent_url,
            }

    @tool
    async def a2a_reset_conversation(self) -> dict[str, Any]:
        """
        Reset the conversation state, clearing context_id and all active tasks.

        Call this tool to start a fresh conversation session with A2A agents.
        This is useful when you want to start a new conversation thread rather
        than continuing an existing one.

        Returns:
            dict: Result including:
                - status: "success" or "error"
                - message: Confirmation message
        """
        try:
            self.reset_conversation()
            return {
                "status": "success",
                "message": "Conversation state reset successfully",
            }
        except Exception as e:
            logger.exception("Error resetting conversation state")
            return {
                "status": "error",
                "error": str(e),
            }

    @tool
    async def a2a_get_conversation_state(self) -> dict[str, Any]:
        """
        Get the current conversation state including context_id and active tasks.

        This is useful for debugging or inspecting the current state of multi-turn
        conversations with A2A agents.

        Returns:
            dict: Current conversation state including:
                - status: "success" or "error"
                - context_id: The current context_id (or None)
                - active_tasks: Dict mapping agent URLs to their active task info
        """
        try:
            state = self.get_conversation_state()
            return {
                "status": "success",
                **state,
            }
        except Exception as e:
            logger.exception("Error getting conversation state")
            return {
                "status": "error",
                "error": str(e),
            }
