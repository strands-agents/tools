"""MCP Client Tool for Strands Agents.

This tool provides a high-level interface for connecting to any MCP server.

It leverages the Strands SDK's MCPClient for robust connection management
and implements a per-operation connection pattern for stability.
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from threading import Lock
from typing import Any, Dict, List, Optional

from mcp import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from strands import tool
from strands.tools.mcp import MCPClient, MCPTransport
from strands.types.tools import AgentTool, ToolResult, ToolSpec, ToolUse

logger = logging.getLogger(__name__)

# Suppress warnings from MCP SDK about unknown notification types
# This is particularly useful for test servers that send non-standard notifications
# Users can override this by setting their own logging configuration
logging.getLogger("mcp.shared.session").setLevel(logging.ERROR)

# Default timeout for MCP operations - can be overridden via environment variable
DEFAULT_MCP_TIMEOUT = float(os.environ.get("STRANDS_MCP_TIMEOUT", "30.0"))


@dataclass
class ConnectionInfo:
    """Simple connection information storage."""

    transport: str
    register_time: float
    url: str
    tool_count: Optional[int] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    server_url: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    is_active: bool = False
    last_error: Optional[str] = None
    loaded_tool_names: List[str] = None
    mcp_client: Optional[MCPClient] = None
    timeout: float = DEFAULT_MCP_TIMEOUT
    # New streamable HTTP specific fields
    headers: Optional[Dict[str, Any]] = None
    auth: Optional[Any] = None
    sse_read_timeout: float = 300.0  # 5 minutes default
    terminate_on_close: bool = True

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.loaded_tool_names is None:
            self.loaded_tool_names = []


# Thread-safe connection storage
_connections: Dict[str, ConnectionInfo] = {}
_CONNECTION_LOCK = Lock()


def _get_connection(connection_id: str) -> Optional[ConnectionInfo]:
    """Get a connection by ID with thread safety.

    Args:
        connection_id: The connection ID to retrieve

    Returns:
        ConnectionInfo if found, None otherwise
    """
    with _CONNECTION_LOCK:
        return _connections.get(connection_id)


def _validate_connection(connection_id: str, check_active: bool = False) -> Dict[str, Any]:
    """Validate that a connection exists and optionally check if it's active.

    Args:
        connection_id: The connection ID to validate
        check_active: Whether to also check if the connection is active

    Returns:
        Dict with error message if validation fails, None otherwise
    """
    if not connection_id:
        return {"status": "error", "content": [{"text": "connection_id is required"}]}

    config = _get_connection(connection_id)
    if not config:
        return {"status": "error", "content": [{"text": f"Connection '{connection_id}' not found"}]}

    if check_active and not config.is_active:
        return {"status": "error", "content": [{"text": f"Connection '{connection_id}' is not active"}]}

    if check_active and not config.mcp_client:
        return {"status": "error", "content": [{"text": f"Connection '{connection_id}' has no active MCP client"}]}

    return None


def _update_connection_status(connection_id: str, is_active: bool, last_error: Optional[str] = None):
    """Update connection status with thread safety.

    Args:
        connection_id: The connection ID to update
        is_active: New active status
        last_error: Optional error message
    """
    with _CONNECTION_LOCK:
        if connection_id in _connections:
            _connections[connection_id].is_active = is_active
            _connections[connection_id].last_error = last_error


class MCPToolWrapper(AgentTool):
    """Wrapper that adapts MCP tools to the AgentTool interface."""

    def __init__(
        self,
        connection_id: str,
        tool_name: str,
        tool_spec: ToolSpec,
        mcp_client: Any,  # Required: direct MCP client reference
        name_prefix: bool = True,
    ):
        """Initialize the MCP tool wrapper."""
        super().__init__()
        self.connection_id = connection_id
        self.original_tool_name = tool_name
        self.mcp_client = mcp_client  # Direct client reference

        # Create prefixed name to avoid conflicts
        if name_prefix:
            # Sanitize connection_id and tool_name to comply with tool name pattern
            sanitized_conn_id = connection_id.replace("-", "_").replace(".", "_")
            sanitized_tool_name = tool_name.replace("-", "_").replace(".", "_")
            self._name = f"mcp_{sanitized_conn_id}_{sanitized_tool_name}"
        else:
            # Still sanitize tool name even without prefix
            self._name = tool_name.replace("-", "_").replace(".", "_")

        # Store the tool spec with updated name
        self._tool_spec = tool_spec.copy()
        self._tool_spec["name"] = self._name

        # Mark as dynamic tool
        self.mark_dynamic()

        logger.debug(
            "Created MCP tool wrapper: connection_id=%s, tool_name=%s, wrapped_name=%s",
            connection_id,
            tool_name,
            self._name,
        )

    @property
    def tool_name(self) -> str:
        """Get the wrapped tool name."""
        return self._name

    @property
    def name(self) -> str:
        """Get the wrapped tool name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the tool description."""
        return self._tool_spec.get("description", "")

    @property
    def tool_spec(self) -> ToolSpec:
        """Get the tool specification."""
        return self._tool_spec

    @property
    def tool_type(self) -> str:
        """Get the tool type."""
        return "mcp"

    @property
    def supports_hot_reload(self) -> bool:
        """MCP tools don't support hot reload."""
        return False

    def invoke(self, tool: ToolUse, *args: Any, **kwargs: Any) -> ToolResult:
        """Execute the MCP tool through the connection."""
        tool_use_id = tool.get("toolUseId", "unknown")

        try:
            # Extract the input parameters
            tool_input = tool.get("input", {})

            logger.debug(
                "Invoking MCP tool: connection_id=%s, tool_name=%s, input=%s",
                self.connection_id,
                self.original_tool_name,
                tool_input,
            )

            # Call the tool using direct client
            result = self.mcp_client.call_tool_sync(
                tool_use_id=f"mcp_{self.connection_id}_{self.original_tool_name}_{uuid.uuid4().hex[:8]}",
                name=self.original_tool_name,
                arguments=tool_input,
            )

            # Extract content from SDK's ToolResult format
            if result.get("status") == "error":
                error_msg = "Unknown error"
                if result.get("content"):
                    error_msg = result["content"][0].get("text", error_msg)
                raise Exception(error_msg)

            # Extract content
            content = result.get("content", [])
            if len(content) == 1 and "text" in content[0]:
                result = content[0]["text"]
            else:
                result = content

            # Format the result according to Strands expectations
            if isinstance(result, dict) and "error" in result:
                return {"toolUseId": tool_use_id, "status": "error", "content": [{"text": result["error"]}]}

            # Convert MCP result to Strands format
            content = []
            if isinstance(result, str):
                content.append({"text": result})
            elif isinstance(result, list):
                # Handle list results - could be content items or other data
                if all(isinstance(item, dict) and ("text" in item or "image" in item) for item in result):
                    # Already in content format
                    content = result
                else:
                    # Convert list to JSON representation with error handling
                    try:
                        content.append({"text": json.dumps(result, indent=2)})
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Failed to serialize list result: {e}")
                        content.append({"text": f"[List with {len(result)} items - serialization failed]"})
            elif isinstance(result, dict):
                # Handle structured results
                if "content" in result:
                    # Already in expected format
                    content = result["content"]
                else:
                    # Convert dict to text representation with error handling
                    try:
                        content.append({"text": json.dumps(result, indent=2)})
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Failed to serialize dict result: {e}")
                        # Try to extract meaningful information from the dict
                        if "error" in result:
                            content.append({"text": f"Error: {result['error']}"})
                        else:
                            content.append({"text": f"[Dictionary result - serialization failed: {str(e)}]"})
            elif isinstance(result, bytes):
                # Handle binary data
                try:
                    # Try to decode as UTF-8 text
                    text_content = result.decode("utf-8")
                    content.append({"text": text_content})
                except UnicodeDecodeError:
                    # If it's not text, indicate it's binary data
                    content.append({"text": f"[Binary data: {len(result)} bytes]"})
            elif result is None:
                content.append({"text": "null"})
            else:
                # For other types, convert to string safely
                content.append({"text": str(result)})

            return {"toolUseId": tool_use_id, "status": "success", "content": content}

        except Exception as e:
            logger.error(
                "Error invoking MCP tool: connection_id=%s, tool_name=%s, error=%s",
                self.connection_id,
                self.original_tool_name,
                str(e),
            )
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error calling MCP tool: {str(e)}"}],
            }

    def get_display_properties(self) -> Dict[str, str]:
        """Get display properties for UI representation."""
        return {
            "connection_id": self.connection_id,
            "original_name": self.original_tool_name,
            "wrapped_name": self._name,
            "type": "mcp",
        }


def create_mcp_tool_wrapper(
    connection_id: str,
    tool_info: Dict[str, Any],
    mcp_client: Any,  # Required MCP client
    name_prefix: bool = False,
) -> MCPToolWrapper:
    """Factory function to create MCP tool wrappers."""
    # Extract tool details
    tool_name = tool_info["name"]
    description = tool_info.get("description", f"MCP tool: {tool_name}")

    # Build tool spec in Strands format
    tool_spec: ToolSpec = {
        "name": tool_name,
        "description": description,
        "inputSchema": tool_info.get("inputSchema", {"json": {"type": "object", "properties": {}, "required": []}}),
    }

    return MCPToolWrapper(
        connection_id=connection_id,
        tool_name=tool_name,
        tool_spec=tool_spec,
        mcp_client=mcp_client,  # Pass the MCP client
        name_prefix=name_prefix,
    )


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
    MCP client tool for connecting to any MCP server with simplified configuration.

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


def _create_transport_callable(config: ConnectionInfo) -> MCPTransport:
    """Create a transport callable based on the connection configuration.

    Args:
        config: Connection configuration

    Returns:
        MCPTransport callable

    Raises:
        ValueError: If transport type is unsupported or required parameters are missing
    """
    if config.transport == "stdio":
        if not config.command:
            raise ValueError("command is required for stdio transport")
        params = {"command": config.command, "args": config.args or []}
        if config.env:
            params["env"] = config.env

        return lambda: stdio_client(StdioServerParameters(**params))
    elif config.transport == "sse":
        if not config.server_url:
            raise ValueError("server_url is required for SSE transport")
        return lambda: sse_client(config.server_url)
    elif config.transport == "streamable_http":
        if not config.server_url:
            raise ValueError("server_url is required for streamable HTTP transport")

        # Create streamable HTTP client with all parameters
        return lambda: streamablehttp_client(
            url=config.server_url,
            headers=config.headers,
            timeout=timedelta(seconds=config.timeout),
            sse_read_timeout=timedelta(seconds=config.sse_read_timeout),
            terminate_on_close=config.terminate_on_close,
            auth=config.auth,
        )
    else:
        raise ValueError(f"Unsupported transport: {config.transport}. Supported: stdio, sse, streamable_http")


def _connect_to_server(params: Dict[str, Any]) -> Dict[str, Any]:
    """Connect to an MCP server.

    Args:
        params: Parameters including connection_id, transport, and transport-specific config

    Returns:
        Connection result
    """
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

    # Create connection configuration
    config = ConnectionInfo(
        transport=transport,
        register_time=time.time(),
        url="",  # Will be populated based on transport
    )

    try:
        if transport == "stdio":
            command = params.get("command")
            if not command:
                raise ValueError("command is required for stdio transport")

            args = params.get("args", [])
            config.command = command
            config.args = args
            config.env = params.get("env")  # Add environment variables
            config.url = f"{command} {' '.join(args)}"

        elif transport == "sse":
            server_url = params.get("server_url")
            if not server_url:
                raise ValueError("server_url is required for SSE transport")

            config.server_url = server_url
            config.url = server_url

        elif transport == "streamable_http":
            server_url = params.get("server_url")
            if not server_url:
                raise ValueError("server_url is required for streamable HTTP transport")

            config.server_url = server_url
            config.url = server_url
            config.headers = params.get("headers")
            config.auth = params.get("auth")

            # Set timeout parameters with defaults
            config.timeout = params.get("timeout", DEFAULT_MCP_TIMEOUT)
            config.sse_read_timeout = params.get("sse_read_timeout", 300.0)
            config.terminate_on_close = params.get("terminate_on_close", True)

        else:
            raise ValueError(f"Unsupported transport: {transport}")

        # Create transport callable using SDK's MCPTransport type
        transport_callable = _create_transport_callable(config)

        # Test the connection by creating a temporary client using SDK's MCPClient
        mcp_client = MCPClient(transport_callable)

        # Start the client (for Option 1, we keep the client alive)
        mcp_client.start()

        # Test the connection by listing tools
        tools = mcp_client.list_tools_sync()

        config.tool_count = len(tools)
        config.is_active = True
        config.mcp_client = mcp_client  # Store the client in the config

        # Register the connection
        with _CONNECTION_LOCK:
            _connections[connection_id] = config

        return {
            "status": "success",
            "message": f"Connected to MCP server '{connection_id}'",
            "connection_id": connection_id,
            "transport": transport,
            "tools_count": len(tools),
            "available_tools": [tool.tool_name for tool in tools],
        }

    except Exception as e:
        config.is_active = False
        config.last_error = str(e)

        # Still register the failed connection for debugging
        with _CONNECTION_LOCK:
            _connections[connection_id] = config

        return {
            "status": "error",
            "content": [{"text": f"Connection test failed: {str(e)}"}],
        }


def _disconnect_from_server(params: Dict[str, Any]) -> Dict[str, Any]:
    """Disconnect from an MCP server.

    Args:
        params: Parameters including connection_id

    Returns:
        Disconnection result
    """
    connection_id = params.get("connection_id")
    error_result = _validate_connection(connection_id)
    if error_result:
        return error_result

    with _CONNECTION_LOCK:
        config = _connections[connection_id]
        loaded_tools = config.loaded_tool_names.copy()

        # Clean up client if it exists
        if config.mcp_client:
            try:
                config.mcp_client.stop(None, None, None)  # Clean shutdown
                logger.info(f"Cleaned up MCP client for connection '{connection_id}'")
            except Exception as e:
                logger.error(f"Error cleaning up MCP client for '{connection_id}': {e}")

        # Remove the connection
        del _connections[connection_id]

    result = {
        "status": "success",
        "message": f"Disconnected from MCP server '{connection_id}'",
        "connection_id": connection_id,
        "was_active": config.is_active,
    }

    if loaded_tools:
        result["loaded_tools_info"] = (
            f"Note: {len(loaded_tools)} tools loaded from this connection remain in the agent: "
            f"{', '.join(loaded_tools)}"
        )

    return result


def _list_active_connections(params: Dict[str, Any]) -> Dict[str, Any]:
    """List all registered MCP connections.

    Args:
        params: Parameters (unused for this action)

    Returns:
        Connection list
    """
    with _CONNECTION_LOCK:
        connections_info = []
        for conn_id, config in _connections.items():
            connections_info.append(
                {
                    "connection_id": conn_id,
                    "transport": config.transport,
                    "url": config.url,
                    "is_active": config.is_active,
                    "tools_count": config.tool_count if config.tool_count is not None else 0,
                    "last_error": config.last_error,
                    "registered_at": config.register_time,
                }
            )

        return {"status": "success", "total_connections": len(_connections), "connections": connections_info}


def _list_server_tools(params: Dict[str, Any]) -> Dict[str, Any]:
    """List available tools from a connected MCP server.

    Args:
        params: Parameters including connection_id

    Returns:
        Tools list
    """
    connection_id = params.get("connection_id")
    error_result = _validate_connection(connection_id, check_active=True)
    if error_result:
        return error_result

    config = _get_connection(connection_id)

    try:
        # Use the stored client
        tools = config.mcp_client.list_tools_sync()

        # Update tool count in config
        config.tool_count = len(tools)
        _update_connection_status(connection_id, True, None)

        # Format tool information
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

        return {"status": "success", "connection_id": connection_id, "tools_count": len(tools), "tools": tools_info}
    except Exception as e:
        # Update connection status
        _update_connection_status(connection_id, False, str(e))
        return {"status": "error", "content": [{"text": f"Failed to list tools: {str(e)}"}]}


def _call_server_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Call a tool on a connected MCP server.

    Args:
        params: Parameters including connection_id, tool_name, and tool arguments

    Returns:
        Tool execution result
    """
    connection_id = params.get("connection_id")
    tool_name = params.get("tool_name")

    if not tool_name:
        return {"status": "error", "content": [{"text": "tool_name is required for call_tool action"}]}

    error_result = _validate_connection(connection_id, check_active=True)
    if error_result:
        return error_result

    config = _get_connection(connection_id)

    # Get tool arguments - support both 'arguments' dict and direct parameters
    tool_args = params.get("tool_args", {})
    # If no explicit arguments provided, use empty dict instead of collecting extra params

    try:
        # Use the stored client
        result = config.mcp_client.call_tool_sync(
            tool_use_id=f"mcp_{connection_id}_{tool_name}_{uuid.uuid4().hex[:8]}", name=tool_name, arguments=tool_args
        )

        # Update connection status
        _update_connection_status(connection_id, True, None)

        # The SDK's ToolResult has status, toolUseId, and content
        formatted_result = {"status": result["status"], "content": result.get("content", [])}

        return {
            "status": "success",
            "connection_id": connection_id,
            "tool_name": tool_name,
            "tool_arguments": tool_args,
            "tool_result": formatted_result,
        }

    except Exception as e:
        # Update connection status
        _update_connection_status(connection_id, False, str(e))
        return {
            "status": "error",
            "content": [{"text": f"Failed to call tool: {str(e)}"}],
        }


def _load_tools_to_agent(params: Dict[str, Any]) -> Dict[str, Any]:
    """Load tools from an MCP server into the agent's tool registry.

    Args:
        params: Parameters including connection_id and optional agent instance

    Returns:
        Loading result
    """
    connection_id = params.get("connection_id")
    error_result = _validate_connection(connection_id, check_active=True)
    if error_result:
        return error_result

    agent = params.get("agent")
    if not agent:
        return {
            "status": "error",
            "content": [
                {"text": "Agent instance not available. Make sure this tool is being called by a Strands agent."}
            ],
        }

    # Check if agent has tool_registry
    if not hasattr(agent, "tool_registry") or not hasattr(agent.tool_registry, "register_tool"):
        return {
            "status": "error",
            "content": [
                {"text": "Agent does not have a tool registry. Make sure you're using a compatible Strands agent."}
            ],
        }

    config = _get_connection(connection_id)

    try:
        # Get tools from the stored client
        tools = config.mcp_client.list_tools_sync()

        loaded_tools = []
        skipped_tools = []

        for tool in tools:
            tool_info = {
                "name": tool.tool_name,
                "description": tool.tool_spec.get("description", f"MCP tool: {tool.tool_name}"),
                "inputSchema": tool.tool_spec.get(
                    "inputSchema", {"json": {"type": "object", "properties": {}, "required": []}}
                ),
            }

            # Create wrapper for this tool with direct client access
            wrapper = create_mcp_tool_wrapper(
                connection_id=connection_id,
                tool_info=tool_info,
                mcp_client=config.mcp_client,  # Pass the client for direct access
                name_prefix=True,  # Prefix with mcp_<connection_id>_
            )

            try:
                # Register the tool with the agent
                agent.tool_registry.register_tool(wrapper)
                loaded_tools.append(wrapper.tool_name)

                # Track this tool in the config
                with _CONNECTION_LOCK:
                    if connection_id in _connections:
                        _connections[connection_id].loaded_tool_names.append(wrapper.tool_name)

                logger.info(f"Loaded MCP tool: {wrapper.tool_name} (original: {tool.tool_name})")

            except Exception as e:
                logger.warning(f"Failed to register tool {tool.tool_name}: {e}")
                skipped_tools.append({"name": tool.tool_name, "error": str(e)})

        result = {
            "status": "success",
            "message": f"Loaded {len(loaded_tools)} tools from connection '{connection_id}'",
            "connection_id": connection_id,
            "loaded_tools": loaded_tools,
            "tool_count": len(loaded_tools),
        }

        if skipped_tools:
            result["skipped_tools"] = skipped_tools

        return result

    except Exception as e:
        logger.error(f"Failed to load tools: {e}", exc_info=True)
        return {"status": "error", "content": [{"text": f"Failed to load tools: {str(e)}"}]}
