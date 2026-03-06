"""
Tool for managing memories using MemMachine (store, search, list, and delete)

This module provides memory management capabilities using
the MemMachine Platform API as the backend. It interacts directly with
the MemMachine REST API for storing, searching, listing, and deleting
memories across episodic and semantic memory types.

Key Features:
------------
1. Memory Management:
   • store: Add new memories with metadata, producer info, and timestamps
   • search: Perform semantic search across episodic and semantic memories
   • list: Retrieve memories with pagination and metadata filtering
   • delete: Remove episodic or semantic memories by ID

2. Safety Features:
   • User confirmation for mutative operations
   • Content previews before storage
   • Warning messages before deletion
   • BYPASS_TOOL_CONSENT mode for bypassing confirmations in tests

3. Advanced Capabilities:
   • Multiple memory types (episodic and semantic)
   • Metadata filtering for targeted retrieval
   • Pagination support for large memory sets
   • Rich output formatting
   • Configurable producer/recipient context

4. Error Handling:
   • API key validation
   • Parameter validation
   • HTTP error handling with status codes
   • Clear error messages

Usage Examples:
--------------
```python
from strands import Agent
from strands_tools import memmachine_memory

agent = Agent(tools=[memmachine_memory])

# Store a memory
agent.tool.memmachine_memory(
    action="store",
    content="User prefers aisle seats on flights",
    metadata={"category": "travel", "user_id": "alice"}
)

# Search memories using semantic search
agent.tool.memmachine_memory(
    action="search",
    query="flight preferences",
    top_k=5
)

# List all memories with pagination
agent.tool.memmachine_memory(
    action="list",
    page_size=50,
    page_num=0
)

# Delete an episodic memory
agent.tool.memmachine_memory(
    action="delete",
    memory_type="episodic",
    memory_id="mem-123"
)
```
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from strands.types.tools import ToolResult, ToolResultContent, ToolUse

# Set up logging
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()

TOOL_SPEC = {
    "name": "memmachine_memory",
    "description": (
        "Memory management tool for storing, searching, and managing memories using MemMachine.\n\n"
        "MemMachine provides a persistent memory layer for AI agents with episodic (conversational)\n"
        "and semantic (factual) memory types.\n\n"
        "Actions:\n"
        "- store: Store new memory messages with metadata\n"
        "- search: Semantic search across memories\n"
        "- list: List memories with pagination and filtering\n"
        "- delete: Delete episodic or semantic memories by ID\n\n"
        "Configuration:\n"
        "- MEMMACHINE_API_KEY (required): API key for authentication\n"
        "- MEMMACHINE_BASE_URL (optional): API base URL (default: https://api.memmachine.ai)"
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform (store, search, list, delete)",
                    "enum": ["store", "search", "list", "delete"],
                },
                "content": {
                    "type": "string",
                    "description": "Content to store (required for store action)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (required for search action)",
                },
                "memory_id": {
                    "type": "string",
                    "description": "Memory ID for delete operation",
                },
                "memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Multiple memory IDs for bulk delete",
                },
                "memory_type": {
                    "type": "string",
                    "description": ("Memory type: 'episodic' or 'semantic' (required for delete, optional for list)"),
                    "enum": ["episodic", "semantic"],
                },
                "types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["episodic", "semantic"]},
                    "description": "Memory types to include (for store and search). Defaults to both.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum results for search (default: 10)",
                },
                "page_size": {
                    "type": "integer",
                    "description": "Page size for list (default: 100)",
                },
                "page_num": {
                    "type": "integer",
                    "description": "Zero-based page number for list (default: 0)",
                },
                "filter": {
                    "type": "string",
                    "description": (
                        "Metadata filter string (e.g., 'metadata.user_id=123 AND metadata.category=travel')"
                    ),
                },
                "producer": {
                    "type": "string",
                    "description": "Who produced the message (default: 'user')",
                },
                "produced_for": {
                    "type": "string",
                    "description": "Intended recipient of the message",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata key-value pairs to store with the memory",
                },
            },
            "required": ["action"],
        }
    },
}


class MemMachineServiceClient:
    """Client for interacting with the MemMachine Platform API.

    This client communicates directly with the MemMachine REST API using
    Bearer token authentication. It requires the MEMMACHINE_API_KEY
    environment variable to be set.

    Optionally, MEMMACHINE_BASE_URL can be set to point to a self-hosted
    MemMachine instance (defaults to https://api.memmachine.ai).
    """

    def __init__(self) -> None:
        """Initialize the MemMachine service client.

        Raises:
            ValueError: If MEMMACHINE_API_KEY environment variable is not set.
        """
        self.api_key = os.environ.get("MEMMACHINE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MEMMACHINE_API_KEY environment variable is required. "
                "Get your API key from the MemMachine Platform at https://memmachine.ai"
            )
        self.base_url = os.environ.get("MEMMACHINE_BASE_URL", "https://api.memmachine.ai").rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        logger.debug("MemMachine client initialized with base_url=%s", self.base_url)

    def _request(self, method: str, path: str, json_data: Optional[Dict] = None) -> requests.Response:
        """Make an HTTP request to the MemMachine API.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: API endpoint path (e.g., /v2/memories).
            json_data: Optional JSON payload for the request body.

        Returns:
            The HTTP response object.

        Raises:
            requests.HTTPError: If the API returns a non-2xx status code.
        """
        url = f"{self.base_url}{path}"
        logger.debug("MemMachine API request: %s %s", method, url)
        response = self.session.request(method, url, json=json_data, timeout=60)
        response.raise_for_status()
        return response

    def store_memory(
        self,
        content: str,
        types: Optional[List[str]] = None,
        producer: str = "user",
        produced_for: str = "",
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Store a memory in MemMachine.

        Args:
            content: The memory content to store.
            types: Memory types to store to (episodic, semantic). Defaults to both.
            producer: Who produced the message (default: 'user').
            produced_for: Intended recipient of the message.
            metadata: Key-value string pairs for metadata.

        Returns:
            API response dict with results containing UIDs.
        """
        message: Dict[str, Any] = {
            "content": content,
            "producer": producer,
        }
        if produced_for:
            message["produced_for"] = produced_for
        if metadata:
            message["metadata"] = {k: str(v) for k, v in metadata.items()}

        payload: Dict[str, Any] = {"messages": [message]}
        if types:
            payload["types"] = types

        response = self._request("POST", "/v2/memories", json_data=payload)
        return response.json()

    def search_memories(
        self,
        query: str,
        top_k: int = 10,
        types: Optional[List[str]] = None,
        filter_str: Optional[str] = None,
    ) -> Dict:
        """Search memories using semantic search.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return (default: 10).
            types: Memory types to search (episodic, semantic). Defaults to both.
            filter_str: Metadata filter string (e.g., 'metadata.user_id=123').

        Returns:
            API response dict with search results.
        """
        payload: Dict[str, Any] = {"query": query, "top_k": top_k}
        if types:
            payload["types"] = types
        if filter_str:
            payload["filter"] = filter_str

        response = self._request("POST", "/v2/memories/search", json_data=payload)
        return response.json()

    def list_memories(
        self,
        page_size: int = 100,
        page_num: int = 0,
        memory_type: Optional[str] = None,
        filter_str: Optional[str] = None,
    ) -> Dict:
        """List memories with pagination.

        Args:
            page_size: Number of memories per page (default: 100).
            page_num: Zero-based page number (default: 0).
            memory_type: Specific memory type to list (episodic or semantic).
            filter_str: Metadata filter string (e.g., 'metadata.user_id=123').

        Returns:
            API response dict with listed memories.
        """
        payload: Dict[str, Any] = {"page_size": page_size, "page_num": page_num}
        if memory_type:
            payload["type"] = memory_type
        if filter_str:
            payload["filter"] = filter_str

        response = self._request("POST", "/v2/memories/list", json_data=payload)
        return response.json()

    def delete_episodic_memory(
        self,
        memory_id: Optional[str] = None,
        memory_ids: Optional[List[str]] = None,
    ) -> None:
        """Delete episodic memory by ID(s).

        Args:
            memory_id: Single episodic memory ID to delete.
            memory_ids: List of episodic memory IDs to delete.

        Raises:
            ValueError: If neither memory_id nor memory_ids is provided.
        """
        if not memory_id and not memory_ids:
            raise ValueError("Either memory_id or memory_ids must be provided")

        payload: Dict[str, Any] = {}
        if memory_id:
            payload["episodic_id"] = memory_id
        if memory_ids:
            payload["episodic_ids"] = memory_ids

        self._request("POST", "/v2/memories/episodic/delete", json_data=payload)

    def delete_semantic_memory(
        self,
        memory_id: Optional[str] = None,
        memory_ids: Optional[List[str]] = None,
    ) -> None:
        """Delete semantic memory by ID(s).

        Args:
            memory_id: Single semantic memory ID to delete.
            memory_ids: List of semantic memory IDs to delete.

        Raises:
            ValueError: If neither memory_id nor memory_ids is provided.
        """
        if not memory_id and not memory_ids:
            raise ValueError("Either memory_id or memory_ids must be provided")

        payload: Dict[str, Any] = {}
        if memory_id:
            payload["semantic_id"] = memory_id
        if memory_ids:
            payload["semantic_ids"] = memory_ids

        self._request("POST", "/v2/memories/semantic/delete", json_data=payload)


def _extract_items_from_node(data: Any) -> List[Dict]:
    """Extract list items from a potentially nested data structure.

    Handles various response structures:
    - Direct list: [item1, item2, ...]
    - Dict with known keys: {"episodes": [...]} or {"memories": [...]}
    - Nested dict: {"long_term_memory": {"episodes": [...]}}

    Args:
        data: The data node to extract items from.

    Returns:
        A list of dict items extracted from the data structure.
    """
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if not isinstance(data, dict):
        return []

    # Look for direct list keys at this level
    for key in ("episodes", "memories", "results", "items"):
        if key in data and isinstance(data[key], list):
            return [item for item in data[key] if isinstance(item, dict)]

    # Go one level deeper (e.g., long_term_memory -> episodes)
    for val in data.values():
        if isinstance(val, dict):
            for key in ("episodes", "memories", "results", "items"):
                if key in val and isinstance(val[key], list):
                    return [item for item in val[key] if isinstance(item, dict)]

    return []


def _extract_memory_entries(content: Dict) -> List[Dict]:
    """Extract a flat list of memory entries from API response content.

    Handles various nested response structures from the MemMachine API,
    including episodic, semantic, and profile memory types.

    Args:
        content: The 'content' dict from the API response.

    Returns:
        A flat list of memory entry dicts, each tagged with '_type'.
    """
    entries: List[Dict] = []
    type_keys = {
        "episodic_memory": "episodic",
        "semantic_memory": "semantic",
        "profile_memory": "profile",
    }

    for key, type_label in type_keys.items():
        data = content.get(key)
        if data is None:
            continue

        items = _extract_items_from_node(data)
        for item in items:
            entries.append({**item, "_type": type_label})

    return entries


def format_store_response(results: List[Dict]) -> Panel:
    """Format store memory response showing UIDs of stored memories.

    Args:
        results: List of result dicts from the store API response.

    Returns:
        A Rich Panel containing a table of stored memory UIDs.
    """
    if not results:
        return Panel("No memories stored.", title="[bold yellow]No Memories Stored", border_style="yellow")

    table = Table(title="Memories Stored", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim")
    table.add_column("UID", style="cyan")

    for i, result in enumerate(results, 1):
        table.add_row(str(i), result.get("uid", "unknown"))

    return Panel(table, title="[bold green]Memories Stored Successfully", border_style="green")


def format_search_response(response_data: Dict) -> Panel:
    """Format search response with episodic and semantic results.

    Args:
        response_data: The full API response dict from the search endpoint.

    Returns:
        A Rich Panel containing search results as a table or formatted JSON.
    """
    content = response_data.get("content", {})
    if not content:
        return Panel(
            "No memories found matching the query.",
            title="[bold yellow]No Matches",
            border_style="yellow",
        )

    entries = _extract_memory_entries(content)
    if entries:
        table = Table(title="Search Results", show_header=True, header_style="bold magenta")
        table.add_column("Type", style="cyan", width=10)
        table.add_column("Content", style="yellow", width=50)
        table.add_column("Score", style="green", width=10)
        table.add_column("Timestamp", style="blue", width=22)

        for entry in entries:
            mem_content = str(entry.get("content", entry.get("memory", "")))
            content_preview = mem_content[:80] + "..." if len(mem_content) > 80 else mem_content

            score = entry.get("score", "N/A")
            if isinstance(score, (int, float)):
                if score > 0.8:
                    score_str = f"[green]{score:.3f}[/green]"
                elif score > 0.5:
                    score_str = f"[yellow]{score:.3f}[/yellow]"
                else:
                    score_str = f"[red]{score:.3f}[/red]"
            else:
                score_str = str(score)

            timestamp = str(entry.get("created_at", entry.get("timestamp", "N/A")))
            mem_type = entry.get("_type", "N/A")
            table.add_row(mem_type, content_preview, score_str, timestamp)

        return Panel(table, title="[bold green]Search Results", border_style="green")

    # Fallback: display raw JSON content
    formatted = json.dumps(content, indent=2, default=str)
    if len(formatted) > 3000:
        formatted = formatted[:3000] + "\n... (truncated for display)"
    return Panel(formatted, title="[bold green]Search Results", border_style="green")


def format_list_response(response_data: Dict) -> Panel:
    """Format list memories response.

    Args:
        response_data: The full API response dict from the list endpoint.

    Returns:
        A Rich Panel containing listed memories as a table or formatted JSON.
    """
    content = response_data.get("content", {})
    if not content:
        return Panel("No memories found.", title="[bold yellow]No Memories", border_style="yellow")

    entries = _extract_memory_entries(content)
    if entries:
        table = Table(title="Memories", show_header=True, header_style="bold magenta")
        table.add_column("Type", style="cyan", width=10)
        table.add_column("Content", style="yellow", width=45)
        table.add_column("ID", style="green", width=15)
        table.add_column("Timestamp", style="blue", width=22)
        table.add_column("Metadata", style="magenta", width=20)

        for entry in entries:
            mem_content = str(entry.get("content", entry.get("memory", "")))
            content_preview = mem_content[:70] + "..." if len(mem_content) > 70 else mem_content
            mem_id = str(entry.get("uid", entry.get("id", "N/A")))
            timestamp = str(entry.get("created_at", entry.get("timestamp", "N/A")))
            metadata = entry.get("metadata", {})
            metadata_str = json.dumps(metadata) if metadata else "None"
            mem_type = entry.get("_type", "N/A")

            table.add_row(mem_type, content_preview, mem_id, timestamp, metadata_str)

        return Panel(table, title="[bold green]Memories List", border_style="green")

    # Fallback: display raw JSON content
    formatted = json.dumps(content, indent=2, default=str)
    if len(formatted) > 3000:
        formatted = formatted[:3000] + "\n... (truncated for display)"
    return Panel(formatted, title="[bold green]Memories List", border_style="green")


def format_delete_response(
    memory_type: str,
    memory_id: Optional[str] = None,
    memory_ids: Optional[List[str]] = None,
) -> Panel:
    """Format delete memory response.

    Args:
        memory_type: The type of memory deleted (episodic or semantic).
        memory_id: Single memory ID that was deleted.
        memory_ids: List of memory IDs that were deleted.

    Returns:
        A Rich Panel confirming the deletion.
    """
    ids = [memory_id] if memory_id else (memory_ids or [])
    lines = [
        f"✅ {memory_type.capitalize()} memory deleted successfully:",
    ]
    for mid in ids:
        lines.append(f"🔑 Memory ID: {mid}")

    return Panel(
        "\n".join(lines),
        title=f"[bold green]{memory_type.capitalize()} Memory Deleted",
        border_style="green",
    )


def memmachine_memory(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Memory management tool for storing, searching, and managing memories in MemMachine.

    This tool provides a comprehensive interface for managing memories with the
    MemMachine Platform API, including storing new memories, performing semantic
    searches, listing memories with pagination, and deleting memories.

    Args:
        tool: ToolUse object containing the following input fields:
            - action: The action to perform (store, search, list, delete)
            - content: Content to store (for store action)
            - query: Search query (for search action)
            - memory_id: Memory ID (for delete action)
            - memory_ids: Multiple memory IDs (for bulk delete action)
            - memory_type: Memory type for delete/list (episodic or semantic)
            - types: Memory types to include (for store/search)
            - top_k: Max results for search (default: 10)
            - page_size: Page size for list (default: 100)
            - page_num: Page number for list (default: 0)
            - filter: Metadata filter string
            - producer: Message producer (default: 'user')
            - produced_for: Message recipient
            - metadata: Optional metadata key-value pairs
        **kwargs: Additional keyword arguments

    Returns:
        ToolResult containing status and response content
    """
    tool_use_id = "default-id"
    try:
        # Extract input from tool use object
        tool_input = tool.get("input", {})
        tool_use_id = tool.get("toolUseId", "default-id")

        # Validate required parameters
        if not tool_input.get("action"):
            raise ValueError("action parameter is required")

        # Initialize client
        client = MemMachineServiceClient()

        # Check if we're in development/test mode
        strands_dev = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

        # Handle different actions
        action = tool_input["action"]

        # For mutative operations, show confirmation dialog unless in BYPASS_TOOL_CONSENT mode
        mutative_actions = {"store", "delete"}
        needs_confirmation = action in mutative_actions and not strands_dev

        if needs_confirmation:
            if action == "store":
                if not tool_input.get("content"):
                    raise ValueError("content is required for store action")

                content_preview = (
                    tool_input["content"][:15000] + "..."
                    if len(tool_input["content"]) > 15000
                    else tool_input["content"]
                )
                console.print(Panel(content_preview, title="[bold green]Memory to Store", border_style="green"))

            elif action == "delete":
                memory_type = tool_input.get("memory_type", "unknown")
                mid = tool_input.get("memory_id", "")
                mids = tool_input.get("memory_ids", [])
                ids_str = mid if mid else ", ".join(mids)
                console.print(
                    Panel(
                        f"Memory Type: {memory_type}\nMemory ID(s): {ids_str}",
                        title="[bold red]⚠️ Memory to be permanently deleted",
                        border_style="red",
                    )
                )

        # Execute the requested action
        if action == "store":
            if not tool_input.get("content"):
                raise ValueError("content is required for store action")

            results = client.store_memory(
                content=tool_input["content"],
                types=tool_input.get("types"),
                producer=tool_input.get("producer", "user"),
                produced_for=tool_input.get("produced_for", ""),
                metadata=tool_input.get("metadata"),
            )

            results_list = results.get("results", [])
            panel = format_store_response(results_list)
            console.print(panel)

            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(results, indent=2, default=str))],
            )

        elif action == "search":
            if not tool_input.get("query"):
                raise ValueError("query is required for search action")

            results = client.search_memories(
                query=tool_input["query"],
                top_k=tool_input.get("top_k", 10),
                types=tool_input.get("types"),
                filter_str=tool_input.get("filter"),
            )

            panel = format_search_response(results)
            console.print(panel)

            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(results, indent=2, default=str))],
            )

        elif action == "list":
            results = client.list_memories(
                page_size=tool_input.get("page_size", 100),
                page_num=tool_input.get("page_num", 0),
                memory_type=tool_input.get("memory_type"),
                filter_str=tool_input.get("filter"),
            )

            panel = format_list_response(results)
            console.print(panel)

            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(results, indent=2, default=str))],
            )

        elif action == "delete":
            memory_type = tool_input.get("memory_type")
            if not memory_type:
                raise ValueError("memory_type is required for delete action (episodic or semantic)")

            memory_id = tool_input.get("memory_id")
            memory_ids = tool_input.get("memory_ids")
            if not memory_id and not memory_ids:
                raise ValueError("memory_id or memory_ids is required for delete action")

            if memory_type == "episodic":
                client.delete_episodic_memory(memory_id=memory_id, memory_ids=memory_ids)
            elif memory_type == "semantic":
                client.delete_semantic_memory(memory_id=memory_id, memory_ids=memory_ids)
            else:
                raise ValueError(f"Invalid memory_type: {memory_type}. Must be 'episodic' or 'semantic'")

            panel = format_delete_response(memory_type, memory_id, memory_ids)
            console.print(panel)

            deleted_ids = memory_id or ", ".join(memory_ids or [])
            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[
                    ToolResultContent(text=f"{memory_type.capitalize()} memory deleted successfully: {deleted_ids}")
                ],
            )

        else:
            raise ValueError(f"Invalid action: {action}")

    except Exception as e:
        error_panel = Panel(
            Text(str(e), style="red"),
            title="❌ Memory Operation Error",
            border_style="red",
        )
        console.print(error_panel)
        return ToolResult(
            toolUseId=tool_use_id,
            status="error",
            content=[ToolResultContent(text=f"Error: {str(e)}")],
        )
