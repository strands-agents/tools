"""
Agent Registry Tool Provider for Strands Agents.

Provides tools for discovering and searching agents through a centralized registry.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.types import AgentCard, Message, Part, Role, TextPart, TransportProtocol
from strands import tool
from strands.types.tools import AgentTool

logger = logging.getLogger(__name__)


@dataclass
class AgentSearchCriteria:
    required_skills: List[str] = None
    preferred_skills: List[str] = None
    min_version: str = None
    max_response_time_ms: int = None
    regions: List[str] = None
    exclude_agents: List[str] = None


class AgentRegistryToolProvider:
    """Agent Registry tool provider with both discovery and communication capabilities."""

    def __init__(
        self,
        registry_url: str = "http://localhost:8000",
        timeout: int = 30,
        agent_auth=None,
        transports: Dict[str, Callable] = None,
        default_preferred_transport: str = TransportProtocol.http_json,
    ):
        self.registry_url = registry_url
        self.timeout = timeout
        self.agent_auth = agent_auth
        self.transports = transports or {}
        self._httpx_client: httpx.AsyncClient | None = None
        self._client_factory: ClientFactory | None = None
        self._request_id = 0
        self._agent_cache: Dict[str, AgentCard] = {}
        self._default_preferred_transport = default_preferred_transport

        logger.info(
            f"Initialized AgentRegistryToolProvider with registry_url={registry_url}, "
            f"timeout={timeout}s, transports={list(self.transports.keys())}"
        )

    @property
    def tools(self) -> List[AgentTool]:
        """Extract all @tool decorated methods from this instance."""
        tools = []
        for attr_name in dir(self):
            if attr_name == "tools":
                continue
            attr = getattr(self, attr_name)
            if isinstance(attr, AgentTool):
                tools.append(attr)
        return tools

    async def _ensure_httpx_client(self) -> httpx.AsyncClient:
        """Ensure the shared HTTP client is initialized."""
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(timeout=self.timeout)
        return self._httpx_client

    async def _ensure_client_factory(self, agent_card: AgentCard = None) -> ClientFactory:
        """Ensure the ClientFactory is initialized with proper auth."""
        httpx_client = await self._ensure_httpx_client()

        # Set authentication if agent card and auth class are provided
        if agent_card and self.agent_auth:
            logger.info(
                f"🔐 Applying {self.agent_auth.__name__} authentication for agent {agent_card.name} at {agent_card.url}"
            )
            httpx_client.auth = self.agent_auth(agent_card)
        else:
            logger.warning(f"⚠️ No authentication configured for agent {agent_card.name if agent_card else 'unknown'}")

        # Build supported transports list from registered transports
        supported_transports = [TransportProtocol.http_json, TransportProtocol.jsonrpc] + list(self.transports.keys())

        config = ClientConfig(httpx_client=httpx_client, streaming=False, supported_transports=supported_transports)

        client_factory = ClientFactory(config)

        # Register all configured transports
        for transport_name, transport_factory in self.transports.items():
            client_factory.register(transport_name, transport_factory)

        return client_factory

    async def _get_agent_card_from_registry(self, agent_name: str) -> Optional[AgentCard]:
        """Get agent card from registry and convert to A2A AgentCard."""
        logger.info(f"Getting agent card for {agent_name} from registry")
        try:
            result = await self._jsonrpc_request("get_agent", {"agent_id": agent_name})
            if result.get("found"):
                agent_data = result.get("agent_card")
                if agent_data:
                    logger.info(f"Successfully retrieved agent card for {agent_name}")
                    return AgentCard(**agent_data)

            logger.warning(f"No agent card found for {agent_name}")
            return None

        except Exception as e:
            error_msg = f"Failed to get agent card for {agent_name}: {e}"
            logger.error(error_msg)
            print(error_msg)
            return None

    async def _send_message_to_agent_direct(
        self, agent_data: dict, message_text: str, message_id: str = None
    ) -> Dict[str, Any]:
        """Send message to agent using agent data directly (bypassing registry lookup)."""
        agent_name = agent_data.get("name", "unknown")
        logger.info(f"Sending message directly to agent {agent_name}: {message_text[:100]}...")
        try:
            # Ensure agent data has all required AgentCard fields
            agent_card_data = {
                "name": agent_data.get("name", "unknown"),
                "description": agent_data.get("description", ""),
                "url": agent_data.get("url", ""),
                "version": agent_data.get("version", "1.0.0"),
                "protocol_version": agent_data.get("protocol_version", "0.3.0"),
                "preferred_transport": agent_data.get("preferred_transport", self._default_preferred_transport),
                "skills": agent_data.get("skills", []),
                "capabilities": agent_data.get("capabilities", {}),
                "default_input_modes": agent_data.get("default_input_modes", ["application/json"]),
                "default_output_modes": agent_data.get("default_output_modes", ["application/json"]),
            }

            # Convert agent data to AgentCard format
            agent_card = AgentCard(**agent_card_data)

            # Create A2A client and send message with proper auth
            logger.info(
                f"🔧 Creating A2A client for agent {agent_name} with transport: {agent_card.preferred_transport}"
            )
            client_factory = await self._ensure_client_factory(agent_card)
            client = client_factory.create(agent_card)
            logger.info(f"🔧 Created client type: {type(client).__name__}")

            if message_id is None:
                message_id = uuid4().hex

            message = Message(
                kind="message",
                role=Role.user,
                parts=[Part(TextPart(kind="text", text=message_text))],
                message_id=message_id,
            )

            # Send message and get response - collect all events first
            logger.debug(f"Sending message {message_id} to agent {agent_name}")
            events = []
            async for event in client.send_message(message):
                logger.info(f"Received event type: {type(event)} from agent {agent_name}")
                events.append(event)

            # Process collected events
            for event in events:
                logger.info(f"Processing event type: {type(event)} from agent {agent_name}")
                logger.info(f"Event content: {event}")
                if isinstance(event, Message):
                    logger.info(f"Received message response from agent {agent_name}")
                    return {
                        "status": "success",
                        "response": event.model_dump(mode="python", exclude_none=True),
                        "agent_name": agent_name,
                        "message_id": message_id,
                    }
                elif isinstance(event, tuple) and len(event) == 2:
                    task, update_event = event
                    logger.info(f"Received task response from agent {agent_name}")
                    return {
                        "status": "success",
                        "response": {
                            "task": task.model_dump(mode="python", exclude_none=True),
                            "update": update_event.model_dump(mode="python", exclude_none=True)
                            if update_event
                            else None,
                        },
                        "agent_name": agent_name,
                        "message_id": message_id,
                    }
                elif isinstance(event, dict):
                    logger.info(f"Received dict response from agent {agent_name}")
                    return {"status": "success", "response": event, "agent_name": agent_name, "message_id": message_id}
                else:
                    logger.warning(f"Unknown event type {type(event)}: {event}")

            first_event_type = type(events[0]).__name__ if events else "None"
            logger.warning(
                f"No response received from agent {agent_name} - collected {len(events)} events, "
                f"first event type: {first_event_type}"
            )
            return {
                "status": "error",
                "error": "No response received from agent",
                "agent_name": agent_name,
                "message_id": message_id,
            }

        except Exception as e:
            logger.exception(f"Error sending message to agent {agent_name}")
            return {"status": "error", "error": str(e), "agent_name": agent_name, "message_id": message_id}

    @tool
    async def registry_send_message_to_agent(
        self, agent_name: str, message_text: str, message_id: str = None
    ) -> dict[str, Any]:
        """
        Send a message to an agent found via registry.

        Args:
            agent_name: Name of the agent in the registry
            message_text: Message to send
            message_id: Optional message ID

        Returns:
            dict: Response from the agent
        """
        logger.info(f"Sending message to agent {agent_name}: {message_text[:100]}...")
        try:
            # Get agent card from registry
            agent_data = await self._get_agent_card_from_registry(agent_name)
            if not agent_data:
                error_msg = f"Agent {agent_name} not found in registry"
                logger.error(error_msg)
                print(error_msg)
                return {"status": "error", "error": f"Agent {agent_name} not found in registry: {error_msg}"}

            # Convert AgentCard to dict if needed
            agent_dict = agent_data.model_dump() if isinstance(agent_data, AgentCard) else agent_data
            return await self._send_message_to_agent_direct(agent_dict, message_text, message_id)

        except Exception as e:
            logger.exception(f"Error sending message to agent {agent_name}")
            return {"status": "error", "error": str(e), "agent_name": agent_name, "message_id": message_id or "unknown"}

    @tool
    async def registry_find_and_message_agent(self, required_skills: List[str], message_text: str) -> Dict[str, Any]:
        """
        Find the best agent for a task and send it a message.

        Args:
            required_skills: Skills the agent must have
            message_text: Message to send to the selected agent
            task_description: Optional task description for better matching

        Returns:
            dict: Combined discovery and messaging result
        """
        # First find the best agent
        best_agent_result = await self.registry_find_best_agent_for_task(required_skills)

        if best_agent_result["status"] != "success" or not best_agent_result["best_agent"]:
            return {"status": "error", "error": "No suitable agent found", "required_skills": required_skills}

        # Then send message to that agent using the agent data we already have
        agent_data = best_agent_result["best_agent"]
        message_result = await self._send_message_to_agent_direct(agent_data, message_text)

        return {
            "status": "success",
            "discovery_result": best_agent_result,
            "message_result": message_result,
            "selected_agent": agent_data["name"],
        }

    def _next_id(self):
        """Generate next JSON-RPC request ID."""
        self._request_id += 1
        return self._request_id

    async def _jsonrpc_request(self, method: str, params: dict = None) -> dict:
        """Make a JSON-RPC 2.0 request."""
        logger.debug(f"Making JSON-RPC request: {method} with params: {params}")
        client = await self._ensure_httpx_client()
        payload = {"jsonrpc": "2.0", "method": method, "id": self._next_id()}
        if params:
            payload["params"] = params

        try:
            response = await client.post(
                f"{self.registry_url}/jsonrpc", json=payload, headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()

            logger.debug(f"JSON-RPC response for {method}: status={response.status_code}")

            if "error" in result:
                error_msg = f"JSON-RPC Error for {method}: {result['error']}"
                logger.error(error_msg)
                print(error_msg)
                raise Exception(f"JSON-RPC Error: {result['error']}")

            return result.get("result", {})
        except Exception as e:
            error_msg = f"Failed JSON-RPC request {method}: {e}"
            logger.error(error_msg)
            logger.exception(f"Full stack trace for {method} failure:")
            print(error_msg)
            raise

    @tool
    async def registry_find_agents_by_skill(self, skill_id: str) -> Dict[str, Any]:
        """
        Find all agents that have a specific skill.

        Args:
            skill_id: The skill identifier to search for

        Returns:
            dict: Search results including agents list and metadata
        """
        try:
            result = await self._jsonrpc_request("search_agents", {"query": skill_id})
            return {
                "status": "success",
                "agents": result.get("agents", []),
                "skill_searched": skill_id,
                "total_count": len(result.get("agents", [])),
            }
        except Exception as e:
            error_msg = f"Error finding agents by skill {skill_id}: {e}"
            logger.exception(error_msg)
            print(error_msg)
            return {"status": "error", "error": str(e), "skill_searched": skill_id}

    @tool
    async def registry_get_all_agents(self) -> Dict[str, Any]:
        """
        Get all registered agents from the registry.

        Returns:
            dict: All registered agents with their capabilities
        """
        try:
            result = await self._jsonrpc_request("list_agents")
            agents = result.get("agents", [])
            return {"status": "success", "agents": agents, "total_count": len(agents)}
        except Exception as e:
            error_msg = f"Error getting all agents: {e}"
            logger.exception(error_msg)
            print(error_msg)
            return {"status": "error", "error": str(e)}

    @tool
    async def registry_find_best_agent_for_task(self, required_skills: List[str]) -> Dict[str, Any]:
        """
        Find the best agent for a specific task based on required skills.

        Args:
            required_skills: List of skills the agent must have

        Returns:
            dict: Best matching agent or None if no match found
        """
        logger.info(f"Finding best agent for required skills: {required_skills}")
        try:
            # Get all agents first
            all_agents_result = await self._jsonrpc_request("list_agents")
            all_agents = all_agents_result.get("agents", [])
            logger.debug(f"Found {len(all_agents)} total agents in registry")

            # Filter agents that have all required skills
            compatible_agents = []
            for agent in all_agents:
                # Handle both 'id' and 'name' fields for backward compatibility
                agent_skills = set()
                for skill in agent.get("skills", []):
                    if "id" in skill:
                        agent_skills.add(skill["id"])
                    if "name" in skill:
                        agent_skills.add(skill["name"])

                if all(skill in agent_skills for skill in required_skills):
                    compatible_agents.append(agent)
                    logger.debug(f"Agent {agent.get('name')} is compatible with required skills")
                else:
                    logger.debug(
                        f"Agent {agent.get('name')} skills {agent_skills} don't match required {required_skills}"
                    )

            logger.info(f"Found {len(compatible_agents)} compatible agents")
            if not compatible_agents:
                logger.warning(f"No agents found with all required skills: {required_skills}")
                return {
                    "status": "success",
                    "best_agent": None,
                    "message": "No agents found with all required skills",
                    "required_skills": required_skills,
                }

            # Simple ranking by skill count
            best_agent = max(compatible_agents, key=lambda x: len(x.get("skills", [])))
            logger.info(
                f"Selected best agent: {best_agent.get('name')} with {len(best_agent.get('skills', []))} skills"
            )

            return {
                "status": "success",
                "best_agent": best_agent,
                "required_skills": required_skills,
                "total_compatible": len(compatible_agents),
            }

        except Exception as e:
            error_msg = f"Error finding best agent for task: {e}"
            logger.exception(error_msg)
            print(error_msg)
            return {"status": "error", "error": str(e), "required_skills": required_skills}

    @tool
    async def registry_find_similar_agents(self, reference_agent_id: str) -> Dict[str, Any]:
        """
        Find agents similar to a reference agent based on skill overlap.

        Args:
            reference_agent_id: The ID of the reference agent

        Returns:
            dict: List of similar agents with similarity scores
        """
        try:
            # Get reference agent
            ref_result = await self._jsonrpc_request("get_agent", {"agent_id": reference_agent_id})
            reference_agent = ref_result.get("agent_card")

            if not reference_agent:
                return {"status": "error", "error": f"Reference agent {reference_agent_id} not found"}

            # Handle both 'id' and 'name' fields for skills
            reference_skills = set()
            for skill in reference_agent.get("skills", []):
                if "id" in skill:
                    reference_skills.add(skill["id"])
                if "name" in skill:
                    reference_skills.add(skill["name"])

            # Get all agents and calculate similarity
            all_agents_result = await self._jsonrpc_request("list_agents")
            all_agents = all_agents_result.get("agents", [])

            similar_agents = []
            for agent in all_agents:
                if agent["name"] == reference_agent_id:
                    continue

                # Handle both 'id' and 'name' fields for skills
                agent_skills = set()
                for skill in agent.get("skills", []):
                    if "id" in skill:
                        agent_skills.add(skill["id"])
                    if "name" in skill:
                        agent_skills.add(skill["name"])
                overlap = len(reference_skills.intersection(agent_skills))

                if overlap > 0:
                    similarity_score = overlap / len(reference_skills.union(agent_skills))
                    agent_copy = agent.copy()
                    agent_copy["similarity_score"] = similarity_score
                    similar_agents.append(agent_copy)

            # Sort by similarity
            similar_agents.sort(key=lambda x: x["similarity_score"], reverse=True)

            return {
                "status": "success",
                "similar_agents": similar_agents,
                "reference_agent": reference_agent_id,
                "total_found": len(similar_agents),
            }

        except Exception as e:
            error_msg = f"Error finding similar agents to {reference_agent_id}: {e}"
            logger.exception(error_msg)
            print(error_msg)
            return {"status": "error", "error": str(e), "reference_agent": reference_agent_id}
