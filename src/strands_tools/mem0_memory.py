"""
Tool for managing memories using Mem0 (store, delete, list, get, and retrieve)

This module provides comprehensive memory management capabilities using
Mem0 as the backend. It handles all aspects of memory management with
a user-friendly interface and proper error handling.

Key Features:
------------
1. Memory Management:
   • store: Add new memories with automatic ID generation and metadata
   • delete: Remove existing memories using memory IDs
   • list: Retrieve all memories for a user or agent
   • get: Retrieve specific memories by memory ID
   • retrieve: Perform semantic search across all memories

2. Safety Features:
   • User confirmation for mutative operations
   • Content previews before storage
   • Warning messages before deletion
   • BYPASS_TOOL_CONSENT mode for bypassing confirmations in tests

3. Advanced Capabilities:
   • Automatic memory ID generation
   • Structured memory storage with metadata
   • Semantic search with relevance filtering
   • Rich output formatting
   • Support for both user and agent memories
   • Multiple vector database backends (OpenSearch, Mem0 Platform, FAISS)

4. Error Handling:
   • Memory ID validation
   • Parameter validation
   • Graceful API error handling
   • Clear error messages

Usage Examples:
--------------
```python
from strands import Agent
from strands_tools import mem0_memory

agent = Agent(tools=[mem0_memory])

# Store memory in Memory
agent.tool.mem0_memory(
    action="store",
    content="Important information to remember",
    user_id="alex",  # or agent_id="agent1"
    metadata={"category": "meeting_notes"}
)

# Retrieve content using semantic search
agent.tool.mem0_memory(
    action="retrieve",
    query="meeting information",
    user_id="alex"  # or agent_id="agent1"
)

# List all memories
agent.tool.mem0_memory(
    action="list",
    user_id="alex"  # or agent_id="agent1"
)
```
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import boto3
from mem0 import Memory as Mem0Memory
from mem0 import MemoryClient
from opensearchpy import AWSV4SignerAuth, RequestsHttpConnection
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
    "name": "mem0_memory",
    "description": (
        "Memory management tool for storing, retrieving, and managing memories in Mem0.\n\n"
        "Features:\n"
        "1. Store memories with metadata (requires user_id or agent_id)\n"
        "2. Retrieve memories by ID or semantic search (requires user_id or agent_id)\n"
        "3. List all memories for a user/agent (requires user_id or agent_id)\n"
        "4. Delete memories\n"
        "5. Get memory history\n\n"
        "Actions:\n"
        "- store: Store new memory (requires user_id or agent_id)\n"
        "- get: Get memory by ID\n"
        "- list: List all memories (requires user_id or agent_id)\n"
        "- retrieve: Semantic search (requires user_id or agent_id)\n"
        "- delete: Delete memory\n"
        "- history: Get memory history\n\n"
        "Note: Most operations require either user_id or agent_id to be specified. The tool will automatically "
        "attempt to retrieve relevant memories when user_id or agent_id is available."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": ("Action to perform (store, get, list, retrieve, delete, history)"),
                    "enum": ["store", "get", "list", "retrieve", "delete", "history"],
                },
                "content": {
                    "type": "string",
                    "description": "Content to store (required for store action)",
                },
                "memory_id": {
                    "type": "string",
                    "description": "Memory ID (required for get, delete, history actions)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (required for retrieve action)",
                },
                "user_id": {
                    "type": "string",
                    "description": "User ID for the memory operations (required for store, list, retrieve actions)",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Agent ID for the memory operations (required for store, list, retrieve actions)",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata to store with the memory",
                },
            },
            "required": ["action"],
        }
    },
}


class Mem0ServiceClient:
    """Client for interacting with Mem0 service."""

    DEFAULT_CONFIG = {
        "embedder": {
            "provider": os.environ.get("MEM0_EMBEDDER_PROVIDER", "aws_bedrock"),
            "config": {"model": os.environ.get("MEM0_EMBEDDER_MODEL", "amazon.titan-embed-text-v2:0")},
        },
        "llm": {
            "provider": os.environ.get("MEM0_LLM_PROVIDER", "aws_bedrock"),
            "config": {
                "model": os.environ.get("MEM0_LLM_MODEL", "anthropic.claude-3-5-haiku-20241022-v1:0"),
                "temperature": float(os.environ.get("MEM0_LLM_TEMPERATURE", 0.1)),
                "max_tokens": int(os.environ.get("MEM0_LLM_MAX_TOKENS", 2000)),
            },
        },
        "vector_store": {
            "provider": "opensearch",
            "config": {
                "port": 443,
                "collection_name": "mem0_memories",
                "host": os.environ.get("OPENSEARCH_HOST"),
                "embedding_model_dims": 1024,
                "connection_class": RequestsHttpConnection,
                "pool_maxsize": 20,
                "use_ssl": True,
                "verify_certs": True,
            },
        },
    }

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Mem0 service client.

        Args:
            config: Optional configuration dictionary to override defaults.
                   If provided, it will be merged with DEFAULT_CONFIG.

        The client will use one of three backends based on environment variables:
        1. Mem0 Platform if MEM0_API_KEY is set
        2. OpenSearch if OPENSEARCH_HOST is set
        3. FAISS (default) if neither MEM0_API_KEY nor OPENSEARCH_HOST is set
        """
        self.mem0 = self._initialize_client(config)

    def _initialize_client(self, config: Optional[Dict] = None) -> Any:
        """Initialize the appropriate Mem0 client based on environment variables.

        Args:
            config: Optional configuration dictionary to override defaults.

        Returns:
            An initialized Mem0 client (MemoryClient or Mem0Memory instance).
        """
        if os.environ.get("MEM0_API_KEY"):
            logger.debug("Using Mem0 Platform backend (MemoryClient)")
            return MemoryClient()

        if os.environ.get("OPENSEARCH_HOST"):
            logger.debug("Using OpenSearch backend (Mem0Memory with OpenSearch)")
            return self._initialize_opensearch_client(config)

        logger.debug("Using FAISS backend (Mem0Memory with FAISS)")
        return self._initialize_faiss_client(config)

    def _initialize_opensearch_client(self, config: Optional[Dict] = None) -> Mem0Memory:
        """Initialize a Mem0 client with OpenSearch backend.

        Args:
            config: Optional configuration dictionary to override defaults.

        Returns:
            An initialized Mem0Memory instance configured for OpenSearch.
        """
        # Set up AWS region
        self.region = os.environ.get("AWS_REGION", "us-west-2")
        if not os.environ.get("AWS_REGION"):
            os.environ["AWS_REGION"] = self.region

        # Set up AWS credentials
        session = boto3.Session()
        credentials = session.get_credentials()
        auth = AWSV4SignerAuth(credentials, self.region, "aoss")

        # Prepare configuration
        merged_config = self._merge_config(config)
        merged_config["vector_store"]["config"].update({"http_auth": auth, "host": os.environ["OPENSEARCH_HOST"]})

        return Mem0Memory.from_config(config_dict=merged_config)

    def _initialize_faiss_client(self, config: Optional[Dict] = None) -> Mem0Memory:
        """Initialize a Mem0 client with FAISS backend.

        Args:
            config: Optional configuration dictionary to override defaults.

        Returns:
            An initialized Mem0Memory instance configured for FAISS.

        Raises:
            ImportError: If faiss-cpu package is not installed.
        """
        try:
            import faiss  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "The faiss-cpu package is required for using FAISS as the vector store backend for Mem0."
                "Please install it using: pip install faiss-cpu"
            ) from err

        merged_config = self._merge_config(config)
        merged_config["vector_store"] = {
            "provider": "faiss",
            "config": {
                "embedding_model_dims": 1024,
                "path": "/tmp/mem0_384_faiss",
            },
        }

        return Mem0Memory.from_config(config_dict=merged_config)

    def _merge_config(self, config: Optional[Dict] = None) -> Dict:
        """Merge user-provided configuration with default configuration.

        Args:
            config: Optional configuration dictionary to override defaults.

        Returns:
            A merged configuration dictionary.
        """
        merged_config = self.DEFAULT_CONFIG.copy()
        if not config:
            return merged_config

        # Deep merge the configs
        for key, value in config.items():
            if key in merged_config and isinstance(value, dict) and isinstance(merged_config[key], dict):
                merged_config[key].update(value)
            else:
                merged_config[key] = value

        return merged_config

    def store_memory(
        self,
        content: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """Store a memory in Mem0."""
        if not user_id and not agent_id:
            raise ValueError("Either user_id or agent_id must be provided")

        messages = [{"role": "user", "content": content}]
        return self.mem0.add(messages, user_id=user_id, agent_id=agent_id, metadata=metadata)

    def get_memory(self, memory_id: str):
        """Get a memory by ID."""
        return self.mem0.get(memory_id)

    def list_memories(self, user_id: Optional[str] = None, agent_id: Optional[str] = None):
        """List all memories for a user or agent."""
        if not user_id and not agent_id:
            raise ValueError("Either user_id or agent_id must be provided")

        return self.mem0.get_all(user_id=user_id, agent_id=agent_id)

    def search_memories(self, query: str, user_id: Optional[str] = None, agent_id: Optional[str] = None):
        """Search memories using semantic search."""
        if not user_id and not agent_id:
            raise ValueError("Either user_id or agent_id must be provided")

        return self.mem0.search(query=query, user_id=user_id, agent_id=agent_id)

    def delete_memory(self, memory_id: str):
        """Delete a memory by ID."""
        return self.mem0.delete(memory_id)

    def get_memory_history(self, memory_id: str):
        """Get the history of a memory by ID."""
        return self.mem0.history(memory_id)


def format_get_response(memory: Dict) -> Panel:
    """Format get memory response."""
    memory_id = memory.get("id", "unknown")
    content = memory.get("memory", "No content available")
    metadata = memory.get("metadata")
    created_at = memory.get("created_at", "Unknown")
    user_id = memory.get("user_id", "Unknown")

    result = [
        "✅ Memory retrieved successfully:",
        f"🔑 Memory ID: {memory_id}",
        f"👤 User ID: {user_id}",
        f"🕒 Created: {created_at}",
    ]

    if metadata:
        result.append(f"📋 Metadata: {json.dumps(metadata, indent=2)}")

    result.append(f"\n📄 Memory: {content}")

    return Panel("\n".join(result), title="[bold green]Memory Retrieved", border_style="green")


def format_list_response(memories: List[Dict]) -> Panel:
    """Format list memories response."""
    if not memories:
        return Panel("No memories found.", title="[bold yellow]No Memories", border_style="yellow")

    table = Table(title="Memories", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Memory", style="yellow", width=50)
    table.add_column("Created At", style="blue")
    table.add_column("User ID", style="green")
    table.add_column("Metadata", style="magenta")

    for memory in memories:
        memory_id = memory.get("id", "unknown")
        content = memory.get("memory", "No content available")
        created_at = memory.get("created_at", "Unknown")
        user_id = memory.get("user_id", "Unknown")
        metadata = memory.get("metadata", {})

        # Truncate content if too long
        content_preview = content[:100] + "..." if len(content) > 100 else content

        # Format metadata for display
        metadata_str = json.dumps(metadata, indent=2) if metadata else "None"

        table.add_row(memory_id, content_preview, created_at, user_id, metadata_str)

    return Panel(table, title="[bold green]Memories List", border_style="green")


def format_delete_response(memory_id: str) -> Panel:
    """Format delete memory response."""
    content = [
        "✅ Memory deleted successfully:",
        f"🔑 Memory ID: {memory_id}",
    ]
    return Panel("\n".join(content), title="[bold green]Memory Deleted", border_style="green")


def format_retrieve_response(memories: List[Dict]) -> Panel:
    """Format retrieve response."""
    if not memories:
        return Panel("No memories found matching the query.", title="[bold yellow]No Matches", border_style="yellow")

    table = Table(title="Search Results", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Memory", style="yellow", width=50)
    table.add_column("Relevance", style="green")
    table.add_column("Created At", style="blue")
    table.add_column("User ID", style="magenta")
    table.add_column("Metadata", style="white")

    for memory in memories:
        memory_id = memory.get("id", "unknown")
        content = memory.get("memory", "No content available")
        score = memory.get("score", 0)
        created_at = memory.get("created_at", "Unknown")
        user_id = memory.get("user_id", "Unknown")
        metadata = memory.get("metadata", {})

        # Truncate content if too long
        content_preview = content[:100] + "..." if len(content) > 100 else content

        # Format metadata for display
        metadata_str = json.dumps(metadata, indent=2) if metadata else "None"

        # Color code the relevance score
        if score > 0.8:
            score_color = "green"
        elif score > 0.5:
            score_color = "yellow"
        else:
            score_color = "red"

        table.add_row(
            memory_id, content_preview, f"[{score_color}]{score}[/{score_color}]", created_at, user_id, metadata_str
        )

    return Panel(table, title="[bold green]Search Results", border_style="green")


def format_history_response(history: List[Dict]) -> Panel:
    """Format memory history response."""
    if not history:
        return Panel("No history found for this memory.", title="[bold yellow]No History", border_style="yellow")

    table = Table(title="Memory History", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Memory ID", style="green")
    table.add_column("Event", style="yellow")
    table.add_column("Old Memory", style="blue", width=30)
    table.add_column("New Memory", style="blue", width=30)
    table.add_column("Created At", style="magenta")

    for entry in history:
        entry_id = entry.get("id", "unknown")
        memory_id = entry.get("memory_id", "unknown")
        event = entry.get("event", "UNKNOWN")
        old_memory = entry.get("old_memory", "None")
        new_memory = entry.get("new_memory", "None")
        created_at = entry.get("created_at", "Unknown")

        # Truncate memory content if too long
        old_memory_preview = old_memory[:100] + "..." if old_memory and len(old_memory) > 100 else old_memory
        new_memory_preview = new_memory[:100] + "..." if new_memory and len(new_memory) > 100 else new_memory

        table.add_row(entry_id, memory_id, event, old_memory_preview, new_memory_preview, created_at)

    return Panel(table, title="[bold green]Memory History", border_style="green")


def format_store_response(results: List[Dict]) -> Panel:
    """Format store memory response."""
    if not results:
        return Panel("No memories stored.", title="[bold yellow]No Memories Stored", border_style="yellow")

    table = Table(title="Memory Stored", show_header=True, header_style="bold magenta")
    table.add_column("Operation", style="green")
    table.add_column("Content", style="yellow", width=50)

    for memory in results:
        event = memory.get("event")
        text = memory.get("memory")
        # Truncate content if too long
        content_preview = text[:100] + "..." if len(text) > 100 else text
        table.add_row(event, content_preview)

    return Panel(table, title="[bold green]Memory Stored", border_style="green")


def mem0_memory(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Memory management tool for storing, retrieving, and managing memories in Mem0.

    This tool provides a comprehensive interface for managing memories with Mem0,
    including storing new memories, retrieving existing ones, listing all memories,
    performing semantic searches, and managing memory history.

    Args:
        tool: ToolUse object containing the following input fields:
            - action: The action to perform (store, get, list, retrieve, delete, history)
            - content: Content to store (for store action)
            - memory_id: Memory ID (for get, delete, history actions)
            - query: Search query (for retrieve action)
            - user_id: User ID for the memory operations
            - agent_id: Agent ID for the memory operations
            - metadata: Optional metadata to store with the memory
        **kwargs: Additional keyword arguments

    Returns:
        ToolResult containing status and response content
    """
    try:
        # Extract input from tool use object
        tool_input = tool.get("input", {})
        tool_use_id = tool.get("toolUseId", "default-id")

        # Validate required parameters
        if not tool_input.get("action"):
            raise ValueError("action parameter is required")

        # Initialize client
        client = Mem0ServiceClient()

        # Check if we're in development mode
        strands_dev = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

        # Handle different actions
        action = tool_input["action"]

        # For mutative operations, show confirmation dialog unless in BYPASS_TOOL_CONSENT mode
        mutative_actions = {"store", "delete"}
        needs_confirmation = action in mutative_actions and not strands_dev

        if needs_confirmation:
            if action == "store":
                # Validate content
                if not tool_input.get("content"):
                    raise ValueError("content is required for store action")

                # Preview what will be stored
                content_preview = (
                    tool_input["content"][:15000] + "..."
                    if len(tool_input["content"]) > 15000
                    else tool_input["content"]
                )
                preview_title = (
                    f"Memory for {'user ' + tool_input.get('user_id', '')}"
                    if tool_input.get("user_id")
                    else f"agent {tool_input.get('agent_id', '')}"
                )

                console.print(Panel(content_preview, title=f"[bold green]{preview_title}", border_style="green"))

            elif action == "delete":
                # Validate memory_id
                if not tool_input.get("memory_id"):
                    raise ValueError("memory_id is required for delete action")

                # Try to get memory info first for better context
                try:
                    memory = client.get_memory(tool_input["memory_id"])
                    metadata = memory.get("metadata", {})

                    console.print(
                        Panel(
                            (
                                f"Memory ID: {tool_input['memory_id']}\n"
                                f"Metadata: {json.dumps(metadata) if metadata else 'None'}"
                            ),
                            title="[bold red]⚠️ Memory to be permanently deleted",
                            border_style="red",
                        )
                    )
                except Exception:
                    # Fall back to basic info if we can't get memory details
                    console.print(
                        Panel(
                            f"Memory ID: {tool_input['memory_id']}",
                            title="[bold red]⚠️ Memory to be permanently deleted",
                            border_style="red",
                        )
                    )

        # Execute the requested action
        if action == "store":
            if not tool_input.get("content"):
                raise ValueError("content is required for store action")

            results = client.store_memory(
                tool_input["content"],
                tool_input.get("user_id"),
                tool_input.get("agent_id"),
                tool_input.get("metadata"),
            )

            # Normalize to list
            results_list = results if isinstance(results, list) else results.get("results", [])
            if results_list:
                panel = format_store_response(results_list)
                console.print(panel)
            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(results_list, indent=2))],
            )

        elif action == "get":
            if not tool_input.get("memory_id"):
                raise ValueError("memory_id is required for get action")

            memory = client.get_memory(tool_input["memory_id"])
            panel = format_get_response(memory)
            console.print(panel)
            return ToolResult(
                toolUseId=tool_use_id, status="success", content=[ToolResultContent(text=json.dumps(memory, indent=2))]
            )

        elif action == "list":
            memories = client.list_memories(tool_input.get("user_id"), tool_input.get("agent_id"))
            # Normalize to list
            results_list = memories if isinstance(memories, list) else memories.get("results", [])
            panel = format_list_response(results_list)
            console.print(panel)
            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(results_list, indent=2))],
            )

        elif action == "retrieve":
            if not tool_input.get("query"):
                raise ValueError("query is required for retrieve action")

            memories = client.search_memories(
                tool_input["query"],
                tool_input.get("user_id"),
                tool_input.get("agent_id"),
            )
            # Normalize to list
            results_list = memories if isinstance(memories, list) else memories.get("results", [])
            panel = format_retrieve_response(results_list)
            console.print(panel)
            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=json.dumps(results_list, indent=2))],
            )

        elif action == "delete":
            if not tool_input.get("memory_id"):
                raise ValueError("memory_id is required for delete action")

            client.delete_memory(tool_input["memory_id"])
            panel = format_delete_response(tool_input["memory_id"])
            console.print(panel)
            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=f"Memory {tool_input['memory_id']} deleted successfully")],
            )

        elif action == "history":
            if not tool_input.get("memory_id"):
                raise ValueError("memory_id is required for history action")

            history = client.get_memory_history(tool_input["memory_id"])
            panel = format_history_response(history)
            console.print(panel)
            return ToolResult(
                toolUseId=tool_use_id, status="success", content=[ToolResultContent(text=json.dumps(history, indent=2))]
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
        return ToolResult(toolUseId=tool_use_id, status="error", content=[ToolResultContent(text=f"Error: {str(e)}")])
