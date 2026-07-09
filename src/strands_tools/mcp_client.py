"""MCP Client Tool for Strands Agents.

⚠️ SECURITY WARNING: This tool allows agents to autonomously connect to external
MCP servers and dynamically load remote tools. This poses security risks as agents
can potentially connect to malicious servers and execute untrusted code. Use with
caution in production environments.

This tool provides a high-level interface for dynamically connecting to any MCP server
and loading remote tools at runtime. This is different from the static MCP server
implementation in the Strands SDK (see https://github.com/strands-agents/docs/blob/main/docs/user-guide/concepts/tools/mcp-tools.md).

Key differences from SDK's MCP implementation:
- This tool enables connections to new MCP servers at runtime
- Can autonomously discover and load external tools from untrusted sources
- Tools are loaded into the agent's registry and can be called directly
- Connections persist across multiple tool invocations
- Supports multiple concurrent connections to different MCP servers

It leverages the Strands SDK's MCPClient for robust connection management
and implements a per-operation connection pattern for stability.
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from threading import Lock
from typing import Any, Dict, List, Optional

from mcp import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from strands import tool
from strands.tools.mcp import MCPClient
from strands.types.tools import AgentTool, ToolGenerator, ToolSpec, ToolUse

logger = logging.getLogger(__name__)

# Default timeout for MCP operations - can be overridden via environment variable
DEFAULT_MCP_TIMEOUT = float(os.environ.get("STRANDS_MCP_TIMEOUT", "30.0"))


class MCPTool(AgentTool):
    """Wrapper class for dynamically loaded MCP tools that extends AgentTool.

    This class wraps MCP tools loaded through mcp_client and ensures proper
    connection management using the `with mcp_client:` context pattern used throughout
    the dynamic MCP client. It handles both sync and async tool execution while
    maintaining connection health and error handling.
    """

    def __init__(self, mcp_tool, connection_id: str):
        """Initialize MCPTool wrapper.

        Args:
            mcp_tool: The underlying MCP tool instance from the SDK
            connection_id: ID of the connection this tool belongs to
        """
        super().__init__()
        self._mcp_tool = mcp_tool
        self._connection_id = connection_id
        logger.debug(f"MCPTool wrapper created for tool '{mcp_tool.tool_name}' on connection '{connection_id}'")

    @property
    def tool_name(self) -> str:
        """Get the name of the tool."""
        return self._mcp_tool.tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        """Get the specification of the tool."""
        return self._mcp_tool.tool_spec

    @property
    def tool_type(self) -> str:
        """Get the type of the tool."""
        return "mcp_dynamic"

    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        """Stream the MCP tool execution with proper connection management.

        This method uses the same `with mcp_client:` context pattern as other
        operations in mcp_client to ensure proper connection management
        and error handling.

        Args:
            tool_use: The tool use request containing tool ID and parameters.
            invocation_state: Context for the tool invocation, including agent state.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Tool events with the last being the tool result.
        """
        logger.debug(
            f"MCPTool executing tool '{self.tool_name}' on connection '{self._connection_id}' "
            f"with tool_use_id '{tool_use['toolUseId']}'"
        )

        # Get connection info
        config = _get_connection(self._connection_id)
        if not config:
            error_result = {
                "toolUseId": tool_use["toolUseId"],
                "status": "error",
                "content": [{"text": f"Connection '{self._connection_id}' not found"}],
            }
            yield error_result
            return

        if not config.is_active:
            error_result = {
                "toolUseId": tool_use["toolUseId"],
                "status": "error",
                "content": [{"text": f"Connection '{self._connection_id}' is not active"}],
            }
            yield error_result
            return

        try:
            # Use the same context pattern as other operations in mcp_client
            with config.mcp_client:
                result = await config.mcp_client.call_tool_async(
                    tool_use_id=tool_use["toolUseId"],
                    name=self.tool_name,
                    arguments=tool_use["input"],
                )
                yield result

        except Exception as e:
            logger.error(f"Error executing MCP tool '{self.tool_name}': {e}", exc_info=True)

            # Mark connection as unhealthy if it fails
            with _CONNECTION_LOCK:
                config.is_active = False
                config.last_error = str(e)

            error_result = {
                "toolUseId": tool_use["toolUseId"],
                "status": "error",
                "content": [{"text": f"Failed to execute tool '{self.tool_name}': {str(e)}"}],
            }
            yield error_result

    def get_display_properties(self) -> dict[str, str]:
        """Get properties to display in UI representations of this tool."""
        base_props = super().get_display_properties()
        base_props["Connection ID"] = self._connection_id
        return base_props


@dataclass
class ConnectionInfo:
    """Information about an MCP connection."""

    connection_id: str
    mcp_client: MCPClient
    transport: str
    url: str
    register_time: float
    is_active: bool = True
    last_error: Optional[str] = None
    loaded_tool_names: List[str] = None

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.loaded_tool_names is None:
            self.loaded_tool_names = []


# Thread-safe connection storage
_connections: Dict[str, ConnectionInfo] = {}
_CONNECTION_LOCK = Lock()


def _get_connection(connection_id: str) -> Optional[ConnectionInfo]:
    """Get a connection by ID with thread safety."""
    with _CONNECTION_LOCK:
        return _connections.get(connection_id)


def _validate_connection(connection_id: str, check_active: bool = False) -> Optional[Dict[str, Any]]:
    """Validate that a connection exists and optionally check if it's active."""
    if not connection_id:
        return {"status": "error", "content": [{"text": "connection_id is required"}]}

    config = _get_connection(connection_id)
    if not config:
        return {"status": "error", "content": [{"text": f"Connection '{connection_id}' not found"}]}

    if check_active and not config.is_active:
        return {"status": "error", "content": [{"text": f"Connection '{connection_id}' is not active"}]}

    return None


def _create_transport_callable(transport: str, **params):
    """Create a transport callable based on the transport type and parameters."""
    if transport == "stdio":
        command = params.get("command")
        if not command:
            raise ValueError("command is required for stdio transport")
        args = params.get("args", [])
        env = params.get("env")
        stdio_params = {"command": command, "args": args}
        if env:
            stdio_params["env"] = env
        return lambda: stdio_client(StdioServerParameters(**stdio_params))

    elif transport == "sse":
        server_url = params.get("server_url")
        if not server_url:
            raise ValueError("server_url is required for SSE transport")
        return lambda: sse_client(server_url)

    elif transport == "streamable_http":
        server_url = params.get("server_url")
        if not server_url:
            raise ValueError("server_url is required for streamable HTTP transport")

        # Build streamable HTTP parameters
        http_params = {"url": server_url}
        if params.get("headers"):
            http_params["headers"] = params["headers"]
        if params.get("timeout"):
            http_params["timeout"] = timedelta(seconds=params["timeout"])
        if params.get("sse_read_timeout"):
            http_params["sse_read_timeout"] = timedelta(seconds=params["sse_read_timeout"])
        if params.get("terminate_on_close") is not None:
            http_params["terminate_on_close"] = params["terminate_on_close"]
        if params.get("auth"):
            http_params["auth"] = params["auth"]

        return lambda: streamablehttp_client(**http_params)

    else:
        raise ValueError(f"Unsupported transport: {transport}. Supported: stdio, sse, streamable_http")


@tool
def mcp_client(
    action: str,
    server_config: Optional[Dict[str, Any]] = None,
    connection_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_args: Optional[Dict[str, Any]] = None,
    # Additional parameters that can be passed directly
    transport: Optional[str] = None,
    command: Optional[str] = None,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    server_url: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
    # New streamable HTTP parameters
    headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    sse_read_timeout: Optional[float] = None,
    terminate_on_close: Optional[bool] = None,
    auth: Optional[Any] = None,
    agent: Optional[Any] = None,  # Agent instance passed by SDK
) -> Dict[str, Any]:
    """
    MCP client tool for autonomously connecting to external MCP servers.

    ⚠️ SECURITY WARNING: This tool enables agents to autonomously connect to external
    MCP servers and dynamically load remote tools at runtime. This can pose significant
    security risks as agents may connect to malicious servers or execute untrusted code.

    Key Security Considerations:
    - Agents can connect to ANY MCP server URL or command provided
    - External tools are loaded directly into the agent's tool registry
    - Loaded tools can execute arbitrary code with agent's permissions
    - Connections persist and can be reused across multiple operations

    This is different from the static MCP server configuration in the Strands SDK
    (see https://github.com/strands-agents/docs/blob/main/docs/user-guide/concepts/tools/mcp-tools.md)
    which uses pre-configured, trusted MCP servers.

    Supports multiple actions for comprehensive MCP server management:
    - connect: Establish connection to an MCP server
    - list_tools: List available tools from a connected server
    - disconnect: Close connection to an MCP server
    - call_tool: Directly invoke a tool on a connected server
    - list_connections: Show all active MCP connections
    - load_tools: Load MCP tools into agent's tool registry for direct access

    Args:
        action: The action to perform (connect, list_tools, disconnect, call_tool, list_connections)
        server_config: Configuration for MCP server connection (optional, can use direct parameters)
        connection_id: Identifier for the MCP connection
        tool_name: Name of tool to call (for call_tool action)
        tool_args: Arguments to pass to tool (for call_tool action)
        transport: Transport type (stdio, sse, or streamable_http) - can be passed directly instead of in server_config
        command: Command for stdio transport - can be passed directly
        args: Arguments for stdio command - can be passed directly
        env: Environment variables for stdio command - can be passed directly
        server_url: URL for SSE or streamable_http transport - can be passed directly
        arguments: Alternative to tool_args for tool arguments
        headers: HTTP headers for streamable_http transport (optional)
        timeout: Timeout in seconds for HTTP operations in streamable_http transport (default: 30)
        sse_read_timeout: SSE read timeout in seconds for streamable_http transport (default: 300)
        terminate_on_close: Whether to terminate connection on close for streamable_http transport (default: True)
        auth: Authentication object for streamable_http transport (httpx.Auth compatible)

    Returns:
        Dict with the result of the operation

    Examples:
        # Connect to custom stdio server with direct parameters
        mcp_client(
            action="connect",
            connection_id="my_server",
            transport="stdio",
            command="python",
            args=["my_server.py"]
        )

        # Connect to streamable HTTP server
        mcp_client(
            action="connect",
            connection_id="http_server",
            transport="streamable_http",
            server_url="https://example.com/mcp",
            headers={"Authorization": "Bearer token"},
            timeout=60
        )

        # Call a tool directly with parameters
        mcp_client(
            action="call_tool",
            connection_id="my_server",
            tool_name="calculator",
            tool_args={"x": 10, "y": 20}
        )
    """

    try:
        # Prepare parameters for action handlers
        params = {
            "action": action,
            "connection_id": connection_id,
            "tool_name": tool_name,
            "tool_args": tool_args or arguments,  # Support both parameter names
            "agent": agent,  # Pass agent instance to handlers
        }

        # Handle server configuration - merge direct parameters with server_config
        if action == "connect":
            if server_config is None:
                server_config = {}

            # Direct parameters override server_config
            if transport is not None:
                params["transport"] = transport
            elif "transport" in server_config:
                params["transport"] = server_config["transport"]

            if command is not None:
                params["command"] = command
            elif "command" in server_config:
                params["command"] = server_config["command"]

            if args is not None:
                params["args"] = args
            elif "args" in server_config:
                params["args"] = server_config["args"]

            if server_url is not None:
                params["server_url"] = server_url
            elif "server_url" in server_config:
                params["server_url"] = server_config["server_url"]

            if env is not None:
                params["env"] = env
            elif "env" in server_config:
                params["env"] = server_config["env"]

            # Streamable HTTP specific parameters
            if headers is not None:
                params["headers"] = headers
            elif "headers" in server_config:
                params["headers"] = server_config["headers"]

            if timeout is not None:
                params["timeout"] = timeout
            elif "timeout" in server_config:
                params["timeout"] = server_config["timeout"]

            if sse_read_timeout is not None:
                params["sse_read_timeout"] = sse_read_timeout
            elif "sse_read_timeout" in server_config:
                params["sse_read_timeout"] = server_config["sse_read_timeout"]

            if terminate_on_close is not None:
                params["terminate_on_close"] = terminate_on_close
            elif "terminate_on_close" in server_config:
                params["terminate_on_close"] = server_config["terminate_on_close"]

            if auth is not None:
                params["auth"] = auth
            elif "auth" in server_config:
                params["auth"] = server_config["auth"]

        # Process the action
        if action == "connect":
            return _connect_to_server(params)
        elif action == "disconnect":
            return _disconnect_from_server(params)
        elif action == "list_connections":
            return _list_active_connections(params)
        elif action == "list_tools":
            return _list_server_tools(params)
        elif action == "call_tool":
            return _call_server_tool(params)
        elif action == "load_tools":
            return _load_tools_to_agent(params)
        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Available actions: "
                        "connect, disconnect, list_connections, list_tools, call_tool, load_tools"
                    }
                ],
            }

    except Exception as e:
        logger.error(f"Error in mcp_client: {e}", exc_info=True)
        return {"status": "error", "content": [{"text": f"Error in mcp_client: {str(e)}"}]}


def _connect_to_server(params: Dict[str, Any]) -> Dict[str, Any]:
    """Connect to an MCP server using SDK's MCPClient."""
    connection_id = params.get("connection_id")
    if not connection_id:
        return {"status": "error", "content": [{"text": "connection_id is required for connect action"}]}

    transport = params.get("transport", "stdio")

    # Check if connection already exists
    with _CONNECTION_LOCK:
        if connection_id in _connections and _connections[connection_id].is_active:
            return {
                "status": "error",
                "content": [{"text": f"Connection '{connection_id}' already exists and is active"}],
            }

    try:
        # Create transport callable using the SDK pattern
        params_copy = params.copy()
        params_copy.pop("transport", None)  # Remove transport to avoid duplicate parameter
        transport_callable = _create_transport_callable(transport, **params_copy)

        # Create MCPClient using SDK
        mcp_client = MCPClient(transport_callable)

        # Test the connection by listing tools using the context manager
        # The context manager handles starting and stopping the client
        with mcp_client:
            tools = mcp_client.list_tools_sync()
            tool_count = len(tools)

        # At this point, the client has been initialized and tested
        # The connection is ready for future use

        # Store connection info
        url = params.get("server_url", f"{params.get('command', '')} {' '.join(params.get('args', []))}")
        connection_info = ConnectionInfo(
            connection_id=connection_id,
            mcp_client=mcp_client,
            transport=transport,
            url=url,
            register_time=time.time(),
            is_active=True,
        )

        with _CONNECTION_LOCK:
            _connections[connection_id] = connection_info

        connection_result = {
            "message": f"Connected to MCP server '{connection_id}'",
            "connection_id": connection_id,
            "transport": transport,
            "tools_count": tool_count,
            "available_tools": [tool.tool_name for tool in tools],
        }

        return {
            "status": "success",
            "content": [{"text": f"Connected to MCP server '{connection_id}'"}, {"json": connection_result}],
        }

    except Exception as e:
        logger.error(f"Connection failed: {e}", exc_info=True)
        return {"status": "error", "content": [{"text": f"Connection failed: {str(e)}"}]}


def _disconnect_from_server(params: Dict[str, Any]) -> Dict[str, Any]:
    """Disconnect from an MCP server and clean up loaded tools."""
    connection_id = params.get("connection_id")
    agent = params.get("agent")
    error_result = _validate_connection(connection_id)
    if error_result:
        return error_result

    try:
        with _CONNECTION_LOCK:
            config = _connections[connection_id]
            loaded_tools = config.loaded_tool_names.copy()

            # Remove connection
            del _connections[connection_id]

        # Clean up loaded tools from agent if agent is provided
        cleanup_result = {"cleaned_tools": [], "failed_tools": []}
        if agent and loaded_tools:
            cleanup_result = _clean_up_tools_from_agent(agent, connection_id, loaded_tools)

        disconnect_result = {
            "message": f"Disconnected from MCP server '{connection_id}'",
            "connection_id": connection_id,
            "was_active": config.is_active,
        }

        if cleanup_result["cleaned_tools"]:
            disconnect_result["cleaned_tools"] = cleanup_result["cleaned_tools"]
            disconnect_result["cleaned_tools_count"] = len(cleanup_result["cleaned_tools"])

        if cleanup_result["failed_tools"]:
            disconnect_result["failed_to_clean_tools"] = cleanup_result["failed_tools"]
            disconnect_result["failed_tools_count"] = len(cleanup_result["failed_tools"])

        if loaded_tools and not agent:
            disconnect_result["loaded_tools_info"] = (
                f"Note: No agent provided, {len(loaded_tools)} tools loaded could not be cleaned up: "
                f"{', '.join(loaded_tools)}"
            )

        return {
            "status": "success",
            "content": [{"text": f"Disconnected from MCP server '{connection_id}'"}, {"json": disconnect_result}],
        }
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Disconnect failed: {str(e)}"}]}


def _list_active_connections(params: Dict[str, Any]) -> Dict[str, Any]:
    """List all active MCP connections."""
    with _CONNECTION_LOCK:
        connections_info = []
        for conn_id, config in _connections.items():
            connections_info.append(
                {
                    "connection_id": conn_id,
                    "transport": config.transport,
                    "url": config.url,
                    "is_active": config.is_active,
                    "registered_at": config.register_time,
                    "last_error": config.last_error,
                    "loaded_tools_count": len(config.loaded_tool_names),
                }
            )

        connections_result = {"total_connections": len(_connections), "connections": connections_info}

        return {
            "status": "success",
            "content": [{"text": f"Found {len(_connections)} MCP connections"}, {"json": connections_result}],
        }


def _list_server_tools(params: Dict[str, Any]) -> Dict[str, Any]:
    """List available tools from a connected MCP server."""
    connection_id = params.get("connection_id")
    error_result = _validate_connection(connection_id, check_active=True)
    if error_result:
        return error_result

    try:
        config = _get_connection(connection_id)
        with config.mcp_client:
            tools = config.mcp_client.list_tools_sync()

        tools_info = []
        for tool in tools:
            tool_spec = tool.tool_spec
            tools_info.append(
                {
                    "name": tool.tool_name,
                    "description": tool_spec.get("description", ""),
                    "input_schema": tool_spec.get("inputSchema", {}),
                }
            )

        tools_result = {"connection_id": connection_id, "tools_count": len(tools), "tools": tools_info}

        return {
            "status": "success",
            "content": [{"text": f"Found {len(tools)} tools on MCP server '{connection_id}'"}, {"json": tools_result}],
        }
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Failed to list tools: {str(e)}"}]}


def _call_server_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Call a tool on a connected MCP server."""
    connection_id = params.get("connection_id")
    tool_name = params.get("tool_name")

    if not tool_name:
        return {"status": "error", "content": [{"text": "tool_name is required for call_tool action"}]}

    error_result = _validate_connection(connection_id, check_active=True)
    if error_result:
        return error_result

    try:
        config = _get_connection(connection_id)
        tool_args = params.get("tool_args", {})

        with config.mcp_client:
            # Use SDK's call_tool_sync which returns proper ToolResult
            return config.mcp_client.call_tool_sync(
                tool_use_id=f"mcp_{connection_id}_{tool_name}", name=tool_name, arguments=tool_args
            )
    except Exception as e:
        return {"status": "error", "content": [{"text": f"Failed to call tool: {str(e)}"}]}


def _clean_up_tools_from_agent(agent, connection_id: str, tool_names: List[str]) -> Dict[str, Any]:
    """Clean up tools loaded from a specific connection from the agent's tool registry."""
    if not agent or not hasattr(agent, "tool_registry") or not hasattr(agent.tool_registry, "unregister_tool"):
        return {
            "cleaned_tools": [],
            "failed_tools": tool_names if tool_names else [],
            "error": "Agent does not support tool unregistration",
        }

    cleaned_tools = []
    failed_tools = []

    for tool_name in tool_names:
        try:
            agent.tool_registry.unregister_tool(tool_name)
            cleaned_tools.append(tool_name)
        except Exception as e:
            failed_tools.append(f"{tool_name} ({str(e)})")

    return {"cleaned_tools": cleaned_tools, "failed_tools": failed_tools}


def _load_tools_to_agent(params: Dict[str, Any]) -> Dict[str, Any]:
    """Load MCP tools into agent's tool registry using MCPTool wrapper."""
    connection_id = params.get("connection_id")
    agent = params.get("agent")

    if not agent:
        return {"status": "error", "content": [{"text": "agent instance is required for load_tools action"}]}

    error_result = _validate_connection(connection_id, check_active=True)
    if error_result:
        return error_result

    # Check if agent has tool_registry
    if not hasattr(agent, "tool_registry") or not hasattr(agent.tool_registry, "register_tool"):
        return {
            "status": "error",
            "content": [
                {"text": "Agent does not have a tool registry. Make sure you're using a compatible Strands agent."}
            ],
        }

    try:
        config = _get_connection(connection_id)

        with config.mcp_client:
            # Use SDK's list_tools_sync which returns MCPAgentTool instances
            tools = config.mcp_client.list_tools_sync()

        loaded_tools = []
        skipped_tools = []

        for tool in tools:
            try:
                # Wrap the MCP tool with our MCPTool class that handles context management
                wrapped_tool = MCPTool(tool, connection_id)

                # Register the wrapped tool with the agent
                logger.info(f"Loading MCP tool [{tool.tool_name}] wrapped in MCPTool")
                agent.tool_registry.register_tool(wrapped_tool)
                loaded_tools.append(tool.tool_name)

            except Exception as e:
                skipped_tools.append({"name": tool.tool_name, "error": str(e)})

        # Update loaded tools list
        with _CONNECTION_LOCK:
            config.loaded_tool_names.extend(loaded_tools)

        load_result = {
            "message": f"Loaded {len(loaded_tools)} tools from MCP server '{connection_id}'",
            "connection_id": connection_id,
            "loaded_tools": loaded_tools,
            "tool_count": len(loaded_tools),  # Add this field for test compatibility
            "total_loaded_tools": len(config.loaded_tool_names),
        }

        if skipped_tools:
            load_result["skipped_tools"] = skipped_tools

        return {
            "status": "success",
            "content": [
                {"text": f"Loaded {len(loaded_tools)} tools from MCP server '{connection_id}'"},
                {"json": load_result},
            ],
        }

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Failed to load tools: {str(e)}"}]}
