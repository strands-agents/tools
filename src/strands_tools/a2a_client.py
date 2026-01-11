"""
A2A (Agent-to-Agent) Protocol Client Tool for Strands Agents.

This tool provides functionality to discover and communicate with A2A-compliant agents

Key Features:
- Agent discovery through agent cards from multiple URLs
- Message sending to specific A2A agents
- Push notification support for real-time task completion alerts
- Custom authentication support via httpx client arguments
- Context and task ID persistence for multi-turn conversations

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
        >>> # First message - context_id and task_id will be returned in response
        >>> result1 = await provider.a2a_send_message("Start a task", "http://agent.example.com")
        >>> # Second message - context_id is automatically reused for the same agent
        >>> result2 = await provider.a2a_send_message("Continue the task", "http://agent.example.com")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import AgentCard, Message, Part, PushNotificationConfig, Role, TextPart
from strands import tool
from strands.types.tools import AgentTool

DEFAULT_TIMEOUT = 300  # set request timeout to 5 minutes

# Terminal task states - tasks in these states should not be continued
TERMINAL_TASK_STATES = {"completed", "canceled", "failed"}

logger = logging.getLogger(__name__)


@dataclass
class ActiveTask:
    """Represents an active (non-terminal) task for an agent."""

    task_id: str
    state: str
    context_id: str


@dataclass
class ConversationState:
    """Tracks conversation state for a target agent URL."""

    context_id: str | None = None
    active_tasks: dict[str, ActiveTask] = field(default_factory=dict)  # task_id -> ActiveTask


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

        # Conversation state tracking for context_id and task_id persistence
        # Key: target_agent_url, Value: ConversationState
        self._conversation_states: dict[str, ConversationState] = {}

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

    def _get_conversation_state(self, target_agent_url: str) -> ConversationState:
        """Get or create conversation state for a target agent URL."""
        if target_agent_url not in self._conversation_states:
            self._conversation_states[target_agent_url] = ConversationState()
        return self._conversation_states[target_agent_url]

    def _update_conversation_state(
        self,
        target_agent_url: str,
        context_id: str | None = None,
        task_id: str | None = None,
        task_state: str | None = None,
    ) -> None:
        """
        Update conversation state from server response.

        Args:
            target_agent_url: The agent URL this state belongs to
            context_id: The context ID from the response (if any)
            task_id: The task ID from the response (if any)
            task_state: The task state from the response (if any)
        """
        state = self._get_conversation_state(target_agent_url)

        # Store context_id if we don't have one yet
        if context_id and not state.context_id:
            state.context_id = context_id
            logger.debug(f"Stored context_id={context_id} for {target_agent_url}")

        # Track task state
        if task_id and task_state:
            if task_state in TERMINAL_TASK_STATES:
                # Remove task from active tracking when it reaches a terminal state
                if task_id in state.active_tasks:
                    del state.active_tasks[task_id]
                    logger.debug(f"Removed terminal task {task_id} (state={task_state}) for {target_agent_url}")
            else:
                # Update or add active task
                if state.context_id:
                    state.active_tasks[task_id] = ActiveTask(
                        task_id=task_id, state=task_state, context_id=state.context_id
                    )
                    logger.debug(f"Updated active task {task_id} (state={task_state}) for {target_agent_url}")

    def _get_task_id_for_continuation(self, target_agent_url: str) -> str | None:
        """
        Get the task_id to use for continuing a conversation.

        Returns:
            The task_id if there's exactly one active non-terminal task, None otherwise.
        """
        state = self._get_conversation_state(target_agent_url)
        if len(state.active_tasks) == 1:
            return next(iter(state.active_tasks.values())).task_id
        return None

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

        IMPORTANT: If the user provides a specific URL, use it directly. If the user
        refers to an agent by name only, use a2a_list_discovered_agents first to get
        the correct URL. Never guess, generate, or hallucinate URLs.

        For multi-turn conversations:
        - context_id: Automatically persisted from the first response and reused
          in subsequent messages to the same agent. Can be explicitly overridden.
        - task_id: If there's exactly one active non-terminal task for the agent,
          it will be automatically reused. Can be explicitly provided to continue
          a specific task, or omitted to create a new task.

        Args:
            message_text: The message content to send to the agent
            target_agent_url: The exact URL of the target A2A agent
                (user-provided URL or from a2a_list_discovered_agents)
            message_id: Optional message ID for tracking (generates UUID if not provided)
            context_id: Optional context ID for continuing a conversation.
                If not provided, uses the persisted context_id from previous
                interactions with this agent (if any).
            task_id: Optional task ID for continuing a specific task.
                If not provided and there's exactly one active task for this agent,
                that task_id will be used automatically.

        Returns:
            dict: Response data including:
                - success: Whether the message was sent successfully
                - response: The agent's response data (if successful)
                - error: Error message (if failed)
                - message_id: The message ID used
                - target_agent_url: The agent URL that was contacted
                - context_id: The context ID used/returned (for conversation continuity)
                - task_id: The task ID used/returned (if applicable)
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

            # Resolve context_id: explicit > persisted
            effective_context_id = context_id
            if effective_context_id is None:
                state = self._get_conversation_state(target_agent_url)
                effective_context_id = state.context_id

            # Resolve task_id: explicit > auto-continuation
            effective_task_id = task_id
            if effective_task_id is None:
                effective_task_id = self._get_task_id_for_continuation(target_agent_url)

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
                response_context_id = None
                response_task_id = None
                response_task_state = None

                if isinstance(event, Message):
                    # Direct message response
                    response_data = event.model_dump(mode="python", exclude_none=True)
                    response_context_id = getattr(event, "context_id", None)
                    response_task_id = getattr(event, "task_id", None)

                    # Update conversation state from response
                    self._update_conversation_state(
                        target_agent_url, response_context_id, response_task_id, response_task_state
                    )

                    return {
                        "status": "success",
                        "response": response_data,
                        "message_id": message_id,
                        "target_agent_url": target_agent_url,
                        "context_id": response_context_id or effective_context_id,
                        "task_id": response_task_id or effective_task_id,
                    }
                elif isinstance(event, tuple) and len(event) == 2:
                    # (Task, UpdateEvent) tuple - extract the task
                    task, update_event = event
                    task_data = task.model_dump(mode="python", exclude_none=True)

                    # Extract IDs and state from task
                    response_context_id = getattr(task, "context_id", None)
                    response_task_id = getattr(task, "id", None)
                    task_status = getattr(task, "status", None)
                    if task_status:
                        response_task_state = getattr(task_status, "state", None)
                        if hasattr(response_task_state, "value"):
                            response_task_state = response_task_state.value

                    # Update conversation state from response
                    self._update_conversation_state(
                        target_agent_url, response_context_id, response_task_id, response_task_state
                    )

                    return {
                        "status": "success",
                        "response": {
                            "task": task_data,
                            "update": (
                                update_event.model_dump(mode="python", exclude_none=True) if update_event else None
                            ),
                        },
                        "message_id": message_id,
                        "target_agent_url": target_agent_url,
                        "context_id": response_context_id or effective_context_id,
                        "task_id": response_task_id or effective_task_id,
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
                "context_id": context_id,
                "task_id": task_id,
            }

    @tool
    async def a2a_get_conversation_state(self, target_agent_url: str) -> dict[str, Any]:
        """
        Get the current conversation state for a target agent.

        This returns the persisted context_id and active tasks for the specified
        agent URL, useful for debugging or understanding the conversation state.

        Args:
            target_agent_url: The URL of the target A2A agent

        Returns:
            dict: Conversation state including:
                - context_id: The persisted context ID (if any)
                - active_tasks: List of active (non-terminal) tasks
                - target_agent_url: The agent URL queried
        """
        state = self._get_conversation_state(target_agent_url)
        return {
            "status": "success",
            "context_id": state.context_id,
            "active_tasks": [
                {"task_id": task.task_id, "state": task.state, "context_id": task.context_id}
                for task in state.active_tasks.values()
            ],
            "target_agent_url": target_agent_url,
        }

    @tool
    async def a2a_clear_conversation_state(self, target_agent_url: str) -> dict[str, Any]:
        """
        Clear the conversation state for a target agent.

        This removes the persisted context_id and all active tasks for the
        specified agent URL, effectively starting a new conversation.

        Args:
            target_agent_url: The URL of the target A2A agent

        Returns:
            dict: Result of the operation including:
                - status: "success" if cleared successfully
                - target_agent_url: The agent URL that was cleared
        """
        if target_agent_url in self._conversation_states:
            del self._conversation_states[target_agent_url]
            logger.info(f"Cleared conversation state for {target_agent_url}")

        return {
            "status": "success",
            "target_agent_url": target_agent_url,
        }
