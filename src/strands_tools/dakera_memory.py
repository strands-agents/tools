"""
Tool for managing agent memories using Dakera (store, retrieve, get, update, delete).

This module provides persistent, decay-weighted vector memory for Strands agents,
backed by a self-hosted Dakera memory server. It mirrors the structure of the
``mem0_memory`` tool so it slots naturally into the Strands tools ecosystem.

Dakera is a self-hosted AI agent memory server. Run it locally with the
``dakera-ai/dakera-deploy`` docker-compose stack (server + MinIO); the REST API
listens on port 3000 by default. See https://github.com/dakera-ai/dakera-deploy.

Key Features:
------------
1. Memory Management:
   - store: Add a new memory for an agent, with importance and metadata
   - retrieve: Semantic (decay-weighted) recall across an agent's memories
   - get: Fetch a specific memory by its ID
   - update: Update the content/metadata of an existing memory
   - delete: Remove a memory by its ID

2. Safety Features:
   - User confirmation for mutative operations (store, update, delete)
   - Content previews before storage
   - Warning messages before deletion
   - BYPASS_TOOL_CONSENT mode for bypassing confirmations in tests

3. Advanced Capabilities:
   - Decay-weighted, access-aware ranking (recall favors important, fresh memories)
   - Importance scoring (0.0-1.0) and typed memories (episodic/semantic/procedural)
   - Structured metadata storage
   - Rich output formatting

Configuration (environment variables):
   - DAKERA_BASE_URL: Dakera server URL (default: http://localhost:3000)
   - DAKERA_API_KEY:  API key for the Dakera server (dk-...)

Usage Examples:
--------------
```python
from strands import Agent
from strands_tools import dakera_memory

agent = Agent(tools=[dakera_memory])

# Store a memory
agent.tool.dakera_memory(
    action="store",
    agent_id="alex",
    content="Important information to remember",
    importance=0.8,
    metadata={"category": "meeting_notes"},
)

# Retrieve memories with decay-weighted semantic search
agent.tool.dakera_memory(
    action="retrieve",
    agent_id="alex",
    query="meeting information",
    top_k=5,
)
```
"""

import json
import logging
import os
from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from strands.types.tools import ToolResult, ToolResultContent, ToolUse

from strands_tools.utils import console_util

logger = logging.getLogger(__name__)
console = console_util.create()

TOOL_SPEC = {
    "name": "dakera_memory",
    "description": (
        "Persistent, decay-weighted memory for agents, backed by a self-hosted Dakera server.\n\n"
        "Actions:\n"
        "- store: Store a new memory (requires agent_id and content)\n"
        "- retrieve: Decay-weighted semantic search (requires agent_id and query)\n"
        "- get: Get a memory by ID (requires agent_id and memory_id)\n"
        "- update: Update a memory's content/metadata (requires agent_id and memory_id)\n"
        "- delete: Delete a memory by ID (requires agent_id and memory_id)\n\n"
        "Configure the server via DAKERA_BASE_URL (default http://localhost:3000) and DAKERA_API_KEY."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform (store, retrieve, get, update, delete)",
                    "enum": ["store", "retrieve", "get", "update", "delete"],
                },
                "agent_id": {
                    "type": "string",
                    "description": "Agent identifier that owns the memories (required for all actions)",
                },
                "content": {
                    "type": "string",
                    "description": "Memory content (required for store; optional for update)",
                },
                "memory_id": {
                    "type": "string",
                    "description": "Memory ID (required for get, update, delete actions)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (required for retrieve action)",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return for retrieve (default: 5)",
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score 0.0-1.0 for store action",
                },
                "memory_type": {
                    "type": "string",
                    "description": "Memory type for store/update (episodic, semantic, procedural, working)",
                    "enum": ["episodic", "semantic", "procedural", "working"],
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata to store with the memory",
                },
            },
            "required": ["action", "agent_id"],
        }
    },
}


class DakeraServiceClient:
    """Thin wrapper around the Dakera Python SDK for the memory tool."""

    def __init__(self) -> None:
        """Initialize the Dakera client from environment configuration.

        Reads DAKERA_BASE_URL (default http://localhost:3000) and DAKERA_API_KEY.
        """
        try:
            from dakera import DakeraClient
        except ImportError as err:
            raise ImportError(
                "The dakera package is required for the dakera_memory tool. "
                "Install it with: pip install 'strands-agents-tools[dakera-memory]'"
            ) from err

        base_url = os.environ.get("DAKERA_BASE_URL", "http://localhost:3000")
        api_key = os.environ.get("DAKERA_API_KEY")
        self.client = DakeraClient(base_url=base_url, api_key=api_key)

    def store_memory(
        self,
        agent_id: str,
        content: str,
        importance: float | None = None,
        memory_type: str = "episodic",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a memory for an agent."""
        return self.client.store_memory(
            agent_id=agent_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata,
        )

    def get_memory(self, agent_id: str, memory_id: str) -> dict[str, Any]:
        """Get a memory by ID."""
        return self.client.get_memory(agent_id, memory_id)

    def update_memory(
        self,
        agent_id: str,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        memory_type: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing memory."""
        return self.client.update_memory(
            agent_id=agent_id,
            memory_id=memory_id,
            content=content,
            metadata=metadata,
            memory_type=memory_type,
        )

    def search_memories(self, agent_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Decay-weighted semantic recall for an agent."""
        response = self.client.recall(agent_id=agent_id, query=query, top_k=top_k)
        memories = getattr(response, "memories", response)
        return [_memory_to_dict(m) for m in memories]

    def delete_memory(self, agent_id: str, memory_id: str) -> dict[str, Any]:
        """Delete a memory by ID."""
        return self.client.forget(agent_id, memory_id)


def _memory_to_dict(memory: Any) -> dict[str, Any]:
    """Normalize an SDK memory object (or dict) into a plain dict."""
    if isinstance(memory, dict):
        return memory
    return {
        "id": getattr(memory, "id", None),
        "content": getattr(memory, "content", None),
        "importance": getattr(memory, "importance", None),
        "score": getattr(memory, "score", None),
        "memory_type": getattr(memory, "memory_type", None),
        "metadata": getattr(memory, "metadata", None),
        "created_at": getattr(memory, "created_at", None),
    }


def format_store_response(memory: dict[str, Any]) -> Panel:
    """Format a store response."""
    content = [
        "✅ Memory stored successfully:",
        f"🔑 Memory ID: {memory.get('id', 'unknown')}",
        f"📊 Importance: {memory.get('importance', 'default')}",
    ]
    text = memory.get("content")
    if text:
        preview = text[:100] + "..." if len(text) > 100 else text
        content.append(f"\n📄 Content: {preview}")
    return Panel("\n".join(content), title="[bold green]Memory Stored", border_style="green")


def format_get_response(memory: dict[str, Any]) -> Panel:
    """Format a get/update response."""
    result = [
        "✅ Memory retrieved successfully:",
        f"🔑 Memory ID: {memory.get('id', 'unknown')}",
        f"📊 Importance: {memory.get('importance', 'Unknown')}",
        f"🕒 Created: {memory.get('created_at', 'Unknown')}",
    ]
    metadata = memory.get("metadata")
    if metadata:
        result.append(f"📋 Metadata: {json.dumps(metadata, indent=2)}")
    result.append(f"\n📄 Memory: {memory.get('content', 'No content available')}")
    return Panel("\n".join(result), title="[bold green]Memory Retrieved", border_style="green")


def format_retrieve_response(memories: list[dict[str, Any]]) -> Panel:
    """Format a retrieve (semantic search) response."""
    if not memories:
        return Panel(
            "No memories found matching the query.",
            title="[bold yellow]No Matches",
            border_style="yellow",
        )

    table = Table(title="Search Results", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Memory", style="yellow", width=50)
    table.add_column("Score", style="green")
    table.add_column("Importance", style="blue")

    for memory in memories:
        content = memory.get("content") or "No content available"
        preview = content[:100] + "..." if len(content) > 100 else content
        table.add_row(
            str(memory.get("id", "unknown")),
            preview,
            str(memory.get("score", "N/A")),
            str(memory.get("importance", "N/A")),
        )

    return Panel(table, title="[bold green]Search Results", border_style="green")


def format_delete_response(memory_id: str) -> Panel:
    """Format a delete response."""
    content = [
        "✅ Memory deleted successfully:",
        f"🔑 Memory ID: {memory_id}",
    ]
    return Panel("\n".join(content), title="[bold green]Memory Deleted", border_style="green")


def dakera_memory(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Persistent decay-weighted memory management for agents, backed by Dakera.

    Args:
        tool: ToolUse object containing the input fields (action, agent_id, content,
            memory_id, query, top_k, importance, memory_type, metadata).
        **kwargs: Additional keyword arguments.

    Returns:
        ToolResult containing status and response content.
    """
    tool_use_id = tool.get("toolUseId", "default-id")
    try:
        tool_input = tool.get("input", {})

        action = tool_input.get("action")
        if not action:
            raise ValueError("action parameter is required")

        agent_id = tool_input.get("agent_id")
        if not agent_id:
            raise ValueError("agent_id parameter is required")

        client = DakeraServiceClient()
        bypass_consent = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

        mutative_actions = {"store", "update", "delete"}
        needs_confirmation = action in mutative_actions and not bypass_consent

        if needs_confirmation:
            if action == "store":
                if not tool_input.get("content"):
                    raise ValueError("content is required for store action")
                preview = tool_input["content"][:15000]
                console.print(
                    Panel(preview, title=f"[bold green]Memory for agent {agent_id}", border_style="green")
                )
            elif action in {"update", "delete"}:
                if not tool_input.get("memory_id"):
                    raise ValueError(f"memory_id is required for {action} action")
                console.print(
                    Panel(
                        f"Memory ID: {tool_input['memory_id']}",
                        title=f"[bold red]⚠️ Memory to be {action}d",
                        border_style="red",
                    )
                )

        if action == "store":
            if not tool_input.get("content"):
                raise ValueError("content is required for store action")
            memory = client.store_memory(
                agent_id=agent_id,
                content=tool_input["content"],
                importance=tool_input.get("importance"),
                memory_type=tool_input.get("memory_type", "episodic"),
                metadata=tool_input.get("metadata"),
            )
            console.print(format_store_response(memory))
            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(memory, indent=2, default=str))],
            )

        if action == "retrieve":
            if not tool_input.get("query"):
                raise ValueError("query is required for retrieve action")
            memories = client.search_memories(
                agent_id=agent_id,
                query=tool_input["query"],
                top_k=tool_input.get("top_k", 5),
            )
            console.print(format_retrieve_response(memories))
            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(memories, indent=2, default=str))],
            )

        if action == "get":
            if not tool_input.get("memory_id"):
                raise ValueError("memory_id is required for get action")
            memory = client.get_memory(agent_id, tool_input["memory_id"])
            console.print(format_get_response(memory))
            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(memory, indent=2, default=str))],
            )

        if action == "update":
            if not tool_input.get("memory_id"):
                raise ValueError("memory_id is required for update action")
            memory = client.update_memory(
                agent_id=agent_id,
                memory_id=tool_input["memory_id"],
                content=tool_input.get("content"),
                metadata=tool_input.get("metadata"),
                memory_type=tool_input.get("memory_type"),
            )
            console.print(format_get_response(memory))
            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(memory, indent=2, default=str))],
            )

        if action == "delete":
            if not tool_input.get("memory_id"):
                raise ValueError("memory_id is required for delete action")
            client.delete_memory(agent_id, tool_input["memory_id"])
            console.print(format_delete_response(tool_input["memory_id"]))
            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=f"Memory {tool_input['memory_id']} deleted successfully")],
            )

        raise ValueError(f"Invalid action: {action}")

    except Exception as e:
        console.print(
            Panel(Text(str(e), style="red"), title="❌ Memory Operation Error", border_style="red")
        )
        return ToolResult(
            toolUseId=tool_use_id,
            status="error",
            content=[ToolResultContent(text=f"Error: {str(e)}")],
        )
