"""MCP Client Tool for AI Agents.

A tool that allows AI agents to connect to MCP servers and use their available tools.
Acts as a client interface that simplifies connecting to and interacting with any
MCP-compatible server.

The main interface is the use_mcp() function, which handles all interactions including:
- Connecting to MCP servers
- Listing available tools
- Calling specific tools
- Managing connections
"""

import inspect
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from threading import Lock
from typing import Any, Dict, List, Optional, Union

from mcp import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from strands import tool
from strands.tools.mcp import MCPClient, MCPTransport

logger = logging.getLogger(__name__)

# Suppress warnings from MCP SDK about unknown notification types
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
    mcp_client: Optional[MCPClient] = None
    timeout: float = DEFAULT_MCP_TIMEOUT
    # Streamable HTTP specific fields
    headers: Optional[Dict[str, Any]] = None
    auth: Optional[Any] = None
    sse_read_timeout: float = 300.0  # 5 minutes default
    terminate_on_close: bool = True


class ConnectionManager:
    """Thread-safe manager for MCP connections."""
    
    def __init__(self):
        """Initialize the connection manager."""
        self._connections: Dict[str, ConnectionInfo] = {}
        self._lock = Lock()
    
    def get_connection(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get a connection by ID with thread safety."""
        with self._lock:
            return self._connections.get(connection_id)
    
    def validate_connection(self, connection_id: str, check_active: bool = False) -> Dict[str, Any]:
        """Validate that a connection exists and optionally check if it's active."""
        if not connection_id:
            return {"status": "error", "content": [{"text": "connection_id is required"}]}

        config = self.get_connection(connection_id)
        if not config:
            return {"status": "error", "content": [{"text": f"Connection '{connection_id}' not found"}]}

        if check_active and not config.is_active:
            return {"status": "error", "content": [{"text": f"Connection '{connection_id}' is not active"}]}

        if check_active and not config.mcp_client:
            return {"status": "error", "content": [{"text": f"Connection '{connection_id}' has no active MCP client"}]}

        return None  # No error
    
    def update_status(self, connection_id: str, is_active: bool, last_error: Optional[str] = None) -> None:
        """Update connection status with thread safety."""
        with self._lock:
            if connection_id in self._connections:
                self._connections[connection_id].is_active = is_active
                self._connections[connection_id].last_error = last_error
    
    def add_connection(self, connection_id: str, config: ConnectionInfo) -> None:
        """Add or update a connection with thread safety."""
        with self._lock:
            self._connections[connection_id] = config
    
    def remove_connection(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Remove a connection with thread safety and return it."""
        with self._lock:
            if connection_id in self._connections:
                connection = self._connections[connection_id]
                del self._connections[connection_id]
                return connection
            return None
    
    def connection_exists(self, connection_id: str) -> bool:
        """Check if a connection exists."""
        with self._lock:
            return connection_id in self._connections
    
    def list_connections(self) -> List[Dict[str, Any]]:
        """List all connections with thread safety."""
        with self._lock:
            connections_info = []
            for conn_id, config in self._connections.items():
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
            return connections_info


class UseMCPMethods:
    """Class that handles MCP server operations."""
    
    def __init__(self):
        """Initialize the MCP methods handler."""
        self._connection_manager = ConnectionManager()
    
    def _create_transport_callable(self, config: ConnectionInfo) -> MCPTransport:
        """Create a transport callable based on the connection configuration."""
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
    
    def connect(self, 
                connection_id: str, 
                transport: Optional[str] = "stdio",
                command: Optional[str] = None,
                args: Optional[List[str]] = None,
                env: Optional[Dict[str, str]] = None,
                server_url: Optional[str] = None,
                headers: Optional[Dict[str, Any]] = None,
                timeout: Optional[float] = None,
                sse_read_timeout: Optional[float] = None,
                terminate_on_close: Optional[bool] = None,
                auth: Optional[Any] = None) -> Dict[str, Any]:
        """Connect to an MCP server."""
        if not connection_id:
            return {"status": "error", "content": [{"text": "connection_id is required for connect action"}]}

        # Check if connection already exists and is active
        if self._connection_manager.connection_exists(connection_id):
            config = self._connection_manager.get_connection(connection_id)
            if config and config.is_active:
                return {
                    "status": "success",
                    "message": f"Connection '{connection_id}' already exists and is active",
                    "connection_id": connection_id,
                    "transport": config.transport,
                    "tools_count": config.tool_count or 0,
                }

        # Create connection configuration
        config = ConnectionInfo(
            transport=transport,
            register_time=time.time(),
            url="",  # Will be populated based on transport
        )

        try:
            if transport == "stdio":
                if not command:
                    raise ValueError("command is required for stdio transport")

                config.command = command
                config.args = args or []
                config.env = env  # Add environment variables
                config.url = f"{command} {' '.join(config.args)}"

            elif transport == "sse":
                if not server_url:
                    raise ValueError("server_url is required for SSE transport")

                config.server_url = server_url
                config.url = server_url

            elif transport == "streamable_http":
                if not server_url:
                    raise ValueError("server_url is required for streamable HTTP transport")

                config.server_url = server_url
                config.url = server_url
                config.headers = headers
                config.auth = auth

                # Set timeout parameters with defaults
                config.timeout = timeout if timeout is not None else DEFAULT_MCP_TIMEOUT
                config.sse_read_timeout = sse_read_timeout if sse_read_timeout is not None else 300.0
                config.terminate_on_close = terminate_on_close if terminate_on_close is not None else True

            else:
                raise ValueError(f"Unsupported transport: {transport}")

            # Create transport callable
            transport_callable = self._create_transport_callable(config)

            # Test the connection by creating a client
            mcp_client = MCPClient(transport_callable)

            # Start the client
            mcp_client.start()

            # Test the connection by listing tools
            tools = mcp_client.list_tools_sync()

            config.tool_count = len(tools)
            config.is_active = True
            config.mcp_client = mcp_client  # Store the client in the config

            # Register the connection
            self._connection_manager.add_connection(connection_id, config)

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
            self._connection_manager.add_connection(connection_id, config)

            return {
                "status": "error",
                "content": [{"text": f"Connection test failed: {str(e)}"}],
            }
    
    def disconnect(self, connection_id: str) -> Dict[str, Any]:
        """Disconnect from an MCP server."""
        error_result = self._connection_manager.validate_connection(connection_id)
        if error_result:
            return error_result

        config = self._connection_manager.get_connection(connection_id)
        was_active = config.is_active if config else False

        # Clean up client if it exists
        if config and config.mcp_client:
            try:
                config.mcp_client.stop(None, None, None)  # Clean shutdown
                logger.info(f"Cleaned up MCP client for connection '{connection_id}'")
            except Exception as e:
                logger.error(f"Error cleaning up MCP client for '{connection_id}': {e}")

        # Remove the connection
        self._connection_manager.remove_connection(connection_id)

        return {
            "status": "success",
            "message": f"Disconnected from MCP server '{connection_id}'",
            "connection_id": connection_id,
            "was_active": was_active,
        }
    
    def connections(self) -> Dict[str, Any]:
        """List all active connections."""
        connections_info = self._connection_manager.list_connections()
        return {
            "status": "success", 
            "total_connections": len(connections_info), 
            "connections": connections_info
        }
    
    def list(self, connection_id: str) -> Dict[str, Any]:
        """List available tools from a connected MCP server."""
        error_result = self._connection_manager.validate_connection(connection_id, check_active=True)
        if error_result:
            return error_result

        config = self._connection_manager.get_connection(connection_id)

        try:
            # Use the stored client
            tools = config.mcp_client.list_tools_sync()

            # Update tool count in config
            config.tool_count = len(tools)
            self._connection_manager.update_status(connection_id, True, None)

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

            return {
                "status": "success", 
                "connection_id": connection_id, 
                "tools_count": len(tools), 
                "tools": tools_info
            }
        except Exception as e:
            # Update connection status
            self._connection_manager.update_status(connection_id, False, str(e))
            return {"status": "error", "content": [{"text": f"Failed to list tools: {str(e)}"}]}
    
    def call(self, connection_id: str, tool_name: str, tool_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call a tool on a connected MCP server."""
        if not tool_name:
            return {"status": "error", "content": [{"text": "tool_name is required for call_tool action"}]}

        error_result = self._connection_manager.validate_connection(connection_id, check_active=True)
        if error_result:
            return error_result

        config = self._connection_manager.get_connection(connection_id)
        tool_args = tool_args or {}

        try:
            # Use the stored client
            result = config.mcp_client.call_tool_sync(
                tool_use_id=f"mcp_{connection_id}_{tool_name}_{uuid.uuid4().hex[:8]}", 
                name=tool_name, 
                arguments=tool_args
            )

            # Update connection status
            self._connection_manager.update_status(connection_id, True, None)

            # Format the result
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
            self._connection_manager.update_status(connection_id, False, str(e))
            return {
                "status": "error",
                "content": [{"text": f"Failed to call tool: {str(e)}"}],
            }


# Singleton instance of the MCP methods handler
_mcp_methods = UseMCPMethods()


@tool
def use_mcp(
    action: str,
    connection_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_args: Optional[Dict[str, Any]] = None,
    transport: Optional[str] = None,
    # Transport-specific parameters
    # stdio parameters
    command: Optional[str] = None,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    # SSE and HTTP parameters
    server_url: Optional[str] = None,
    # HTTP-specific parameters
    headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    sse_read_timeout: Optional[float] = None,
    terminate_on_close: Optional[bool] = None,
    auth: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Tool for AI agents to connect to MCP (Model Context Protocols) servers and call tools provided by those servers.
    
    This tool simplifies the process of connecting to different types of MCP servers
    (stdio, SSE, HTTP) and managing tools they provide.
    
    Key Actions:
    -----------
    - connect: Create a new connection to an MCP server
    - disconnect: Close a connection to an MCP server
    - list: Show all available tools on a connected server
    - call: Execute a tool on a connected server
    - connections: List all active connections
    
    Connection Types:
    ---------------
    1. stdio: Connect to local process (Python scripts, CLI tools)
       Required: command
       Optional: args, env
       
    2. sse: Connect to Server-Sent Events API
       Required: server_url
       
    3. streamable_http: Connect to HTTP API
       Required: server_url
       Optional: headers, auth, timeout, sse_read_timeout, terminate_on_close
    
    Args:
        action: Operation to perform (connect, disconnect, list, call, connections)
        connection_id: Identifier for the MCP connection
        tool_name: Name of remote tool to call (required for 'call' action)
        tool_args: Arguments to pass to the remote tool (for 'call' action)
        transport: Connection type: "stdio", "sse", or "streamable_http" (required for 'connect')
        
        # stdio transport parameters
        command: Process command to execute (required for stdio transport)
        args: Arguments for the command (optional for stdio transport)
        env: Environment variables for the command (optional for stdio transport)
        
        # SSE and HTTP parameters
        server_url: URL for the SSE or HTTP server (required for sse/streamable_http transport)
        
        # HTTP-specific parameters
        headers: HTTP headers (optional for streamable_http)
        timeout: Connection timeout in seconds (optional for streamable_http)
        sse_read_timeout: SSE read timeout in seconds (optional for streamable_http)
        terminate_on_close: Whether to terminate on close (optional for streamable_http)
        auth: Authentication object (optional for streamable_http)
    
    Returns:
        Dict containing the result of the operation with structured information.
    
    Examples:
        # Connect to a Python script MCP server
        use_mcp(
            action="connect",
            connection_id="python_tools",
            transport="stdio",
            command="python",
            args=["-m", "my_mcp_server"]
        )
        
        # List available tools on the server
        use_mcp(
            action="list",
            connection_id="python_tools"
        )
        
        # Call a tool on the server
        use_mcp(
            action="call",
            connection_id="python_tools",
            tool_name="calculate",
            tool_args={"operation": "add", "x": 10, "y": 20}
        )
        
        # List all active connections
        use_mcp(action="connections")
        
        # Disconnect from a server
        use_mcp(
            action="disconnect", 
            connection_id="python_tools"
        )
    """
    
    try:
        # Create method parameters dictionary from all provided arguments
        method_params = {
            "connection_id": connection_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "transport": transport,
            "command": command,
            "args": args,
            "env": env,
            "server_url": server_url,
            "headers": headers,
            "timeout": timeout,
            "sse_read_timeout": sse_read_timeout,
            "terminate_on_close": terminate_on_close,
            "auth": auth,
        }
        
        # Remove None values
        method_params = {k: v for k, v in method_params.items() if v is not None}
        
        # Get the method corresponding to the action
        method = getattr(_mcp_methods, action, None)
        
        if method:
            # Get method signature to only pass valid parameters
            sig = inspect.signature(method)
            valid_params = {k: v for k, v in method_params.items() if k in sig.parameters}
            return method(**valid_params)
        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Available actions: connect, disconnect, connections, list, call"
                    }
                ],
            }
            
    except Exception as e:
        logger.error(f"Error in use_mcp: {e}", exc_info=True)
        return {"status": "error", "content": [{"text": f"Error in use_mcp: {str(e)}"}]}