"""
TCP tool for Strands Agent to function as both server and client.

This module provides TCP server and client functionality for Strands Agents,
allowing them to communicate over TCP/IP networks. The tool runs server operations
in background threads, enabling concurrent communication without blocking the main agent.

Key Features:
1. TCP Server: Listen for incoming connections and process them with an agent
2. TCP Client: Connect to remote TCP servers and exchange messages
3. Background Processing: Server runs in a background thread
4. Per-Connection Agents: Creates a fresh agent for each client connection

Usage with Strands Agent:

```python
from strands import Agent
from strands_tools import tcp

agent = Agent(tools=[tcp])

# Start a TCP server
result = agent.tool.tcp(
    action="start_server",
    host="127.0.0.1",
    port=8000,
    system_prompt="You are a helpful TCP server assistant."
)

# Connect to a TCP server as client
result = agent.tool.tcp(
    action="client_send",
    host="127.0.0.1",
    port=8000,
    message="Hello, server!"
)

# Stop the TCP server
result = agent.tool.tcp(
    action="stop_server",
    port=8000
)
```

See the tcp function docstring for more details on configuration options and parameters.
"""

import logging
import socket
import threading
import time
from typing import Any, Dict, Optional

from strands import Agent
from strands.types.tools import ToolResult, ToolUse

logger = logging.getLogger(__name__)

# Global registry to store server threads
SERVER_THREADS: Dict[int, Dict[str, Any]] = {}

TOOL_SPEC = {
    "name": "tcp",
    "description": "Create and manage TCP servers and clients for network communication with connection handling",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start_server", "stop_server", "get_status", "client_send"],
                    "description": "Action to perform with the TCP tool",
                },
                "host": {"type": "string", "description": "Host address for server or client connection"},
                "port": {"type": "integer", "description": "Port number for server or client connection"},
                "system_prompt": {"type": "string", "description": "System prompt for the server agent"},
                "message": {
                    "type": "string",
                    "description": "Message to send to the TCP server (for client_send action)",
                },
                "timeout": {"type": "integer", "description": "Connection timeout in seconds (default: 90)"},
                "buffer_size": {
                    "type": "integer",
                    "description": "Size of the message buffer in bytes (default: 4096)",
                },
                "max_connections": {
                    "type": "integer",
                    "description": "Maximum number of concurrent connections (default: 5)",
                },
            },
            "required": ["action"],
        }
    },
}


def handle_client(
    client_socket: socket.socket,
    client_address: tuple,
    system_prompt: str,
    buffer_size: int,
    parent_tools: list = None,
    trace_attributes: dict = None,
) -> None:
    """
    Handle a client connection in the TCP server.

    Args:
        client_socket: The socket for the client connection
        client_address: The address of the client
        system_prompt: System prompt for creating a new agent for this connection
        buffer_size: Size of the message buffer
        parent_tools: Tools inherited from the parent agent
        trace_attributes: Trace attributes from the parent agent
    """
    logger.info(f"Connection established with {client_address}")

    # Create a fresh agent instance for this client connection
    connection_agent = Agent(
        messages=[], tools=parent_tools or [], system_prompt=system_prompt, trace_attributes=trace_attributes or {}
    )

    try:
        # Send welcome message
        welcome_msg = "Welcome to Strands TCP Server! Send a message or 'exit' to close the connection.\n"
        client_socket.sendall(welcome_msg.encode())

        while True:
            # Receive data from the client
            data = client_socket.recv(buffer_size)

            if not data:
                logger.info(f"Client {client_address} disconnected")
                break

            message = data.decode().strip()
            logger.info(f"Received from {client_address}: {message}")

            if message.lower() == "exit":
                client_socket.sendall("Connection closed by client request.\n".encode())
                logger.info(f"Client {client_address} requested to exit")
                break

            # Process the message with the connection-specific agent
            response = connection_agent(message)
            response_text = str(response)

            # Send the response back to the client
            client_socket.sendall((response_text + "\n").encode())

    except Exception as e:
        logger.error(f"Error handling client {client_address}: {e}")
    finally:
        client_socket.close()
        logger.info(f"Connection with {client_address} closed")


def run_server(
    host: str,
    port: int,
    system_prompt: str,
    max_connections: int,
    buffer_size: int,
    parent_agent: Optional[Agent] = None,
) -> None:
    """
    Run a TCP server that processes client requests with per-connection Strands agents.

    Args:
        host: Host address to bind the server
        port: Port number to bind the server
        system_prompt: System prompt for the server agents
        max_connections: Maximum number of concurrent connections
        buffer_size: Size of the message buffer
        parent_agent: Parent agent to inherit tools from
    """
    # Store server state
    SERVER_THREADS[port]["running"] = True
    SERVER_THREADS[port]["connections"] = 0
    SERVER_THREADS[port]["start_time"] = time.time()

    # Get tools and trace attributes from parent agent
    parent_tools = []
    trace_attributes = {}
    if parent_agent:
        parent_tools = list(parent_agent.tool_registry.registry.values())
        trace_attributes = parent_agent.trace_attributes

    # Create server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((host, port))
        server_socket.listen(max_connections)
        logger.info(f"TCP Server listening on {host}:{port}")

        SERVER_THREADS[port]["socket"] = server_socket

        while SERVER_THREADS[port]["running"]:
            # Set a timeout to check periodically if the server should stop
            server_socket.settimeout(1.0)

            try:
                # Accept client connection
                client_socket, client_address = server_socket.accept()
                SERVER_THREADS[port]["connections"] += 1

                # Handle client in a new thread with a fresh agent
                client_thread = threading.Thread(
                    target=handle_client,
                    args=(client_socket, client_address, system_prompt, buffer_size, parent_tools, trace_attributes),
                )
                client_thread.daemon = True
                client_thread.start()

            except socket.timeout:
                # This is expected due to the timeout, allows checking if server should stop
                pass
            except Exception as e:
                if SERVER_THREADS[port]["running"]:
                    logger.error(f"Error accepting connection: {e}")

    except Exception as e:
        logger.error(f"Server error on {host}:{port}: {e}")
    finally:
        try:
            server_socket.close()
        except Exception:
            pass
        logger.info(f"TCP Server on {host}:{port} stopped")
        SERVER_THREADS[port]["running"] = False


def tcp(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Manage TCP server and client operations for network communication with improved
    connection handling (creates a new agent per connection).

    This function provides TCP server and client functionality for Strands agents,
    allowing them to communicate over TCP/IP networks. Servers run in background
    threads with a new, fresh agent instance for each client connection.

    How It Works:
    ------------
    1. Server Mode:
       - Starts a TCP server in a background thread
       - Creates a dedicated agent for EACH client connection
       - Inherits tools from the parent agent
       - Processes client messages and returns responses

    2. Client Mode:
       - Connects to a TCP server
       - Sends messages and receives responses
       - Maintains stateless connections (no persistent sessions)

    3. Management:
       - Track server status and statistics
       - Stop servers gracefully
       - Monitor connections and performance

    Common Use Cases:
    ---------------
    - Network service automation
    - Inter-agent communication
    - Remote command and control
    - API gateway implementation
    - IoT device management

    Args:
        tool (ToolUse): Tool use object containing:
            action (str): Action to perform (start_server, stop_server, get_status, client_send)
            host (str): Host address
            port (int): Port number
            system_prompt (str): System prompt for server agent (for start_server)
            message (str): Message to send (for client_send)
            timeout (int): Connection timeout in seconds
            buffer_size (int): Size of message buffer in bytes
            max_connections (int): Maximum concurrent connections
        **kwargs (Any): Additional keyword arguments

    Returns:
        ToolResult: Dictionary containing status and response content

    Notes:
        - Server instances persist until explicitly stopped
        - Each client connection gets its own agent instance
        - Connection agents inherit tools from the parent agent
        - Client connections are stateless
    """
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    parent_agent = kwargs.get("agent")

    action = tool_input["action"]

    if action == "start_server":
        host = tool_input.get("host", "127.0.0.1")
        port = tool_input.get("port", 8000)
        system_prompt = tool_input.get("system_prompt", "You are a helpful TCP server assistant.")
        max_connections = tool_input.get("max_connections", 5)
        buffer_size = tool_input.get("buffer_size", 4096)

        # Check if server already running on this port
        if port in SERVER_THREADS and SERVER_THREADS[port].get("running", False):
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error: TCP Server already running on port {port}"}],
            }

        # Create server thread
        SERVER_THREADS[port] = {"running": False}
        server_thread = threading.Thread(
            target=run_server, args=(host, port, system_prompt, max_connections, buffer_size, parent_agent)
        )
        server_thread.daemon = True
        server_thread.start()

        # Wait briefly to ensure server starts
        time.sleep(0.5)

        if not SERVER_THREADS[port].get("running", False):
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error: Failed to start TCP Server on {host}:{port}"}],
            }

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {"text": f"TCP Server started successfully on {host}:{port}"},
                {"text": f"System prompt: {system_prompt}"},
                {"text": "Server creates a new agent instance for each connection"},
            ],
        }

    elif action == "stop_server":
        port = tool_input.get("port", 8000)

        if port not in SERVER_THREADS or not SERVER_THREADS[port].get("running", False):
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error: No TCP Server running on port {port}"}],
            }

        # Stop the server
        SERVER_THREADS[port]["running"] = False

        # Close socket if it exists
        if "socket" in SERVER_THREADS[port]:
            try:
                SERVER_THREADS[port]["socket"].close()
            except Exception:
                pass

        # Wait briefly to ensure server stops
        time.sleep(1.0)

        connections = SERVER_THREADS[port].get("connections", 0)
        uptime = time.time() - SERVER_THREADS[port].get("start_time", time.time())

        # Clean up server thread data
        del SERVER_THREADS[port]

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {"text": f"TCP Server on port {port} stopped successfully"},
                {"text": f"Statistics: {connections} connections handled, uptime {uptime:.2f} seconds"},
            ],
        }

    elif action == "get_status":
        if not SERVER_THREADS:
            return {"toolUseId": tool_use_id, "status": "success", "content": [{"text": "No TCP Servers running"}]}

        status_info = []
        for port, data in SERVER_THREADS.items():
            if data.get("running", False):
                uptime = time.time() - data.get("start_time", time.time())
                connections = data.get("connections", 0)
                status_info.append(f"Port {port}: Running - {connections} connections, uptime {uptime:.2f}s")
            else:
                status_info.append(f"Port {port}: Stopped")

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": "TCP Server Status:"}, {"text": "\n".join(status_info)}],
        }

    elif action == "client_send":
        host = tool_input.get("host", "127.0.0.1")
        port = tool_input.get("port", 8000)
        message = tool_input.get("message", "")
        timeout = tool_input.get("timeout", 90)
        buffer_size = tool_input.get("buffer_size", 4096)

        if not message:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "Error: No message provided for client_send action"}],
            }

        # Create client socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(timeout)

        try:
            # Connect to server
            client_socket.connect((host, port))

            # Receive welcome message
            _welcome = client_socket.recv(buffer_size).decode()

            # Send message to server
            client_socket.sendall(message.encode())

            # Receive response
            response = client_socket.recv(buffer_size).decode()

            # Send exit message and close connection
            client_socket.sendall("exit".encode())
            client_socket.close()

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [
                    {"text": f"Connected to {host}:{port} successfully"},
                    {"text": f"Sent message: {message}"},
                    {"text": "Response received:"},
                    {"text": response},
                ],
            }

        except socket.timeout:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error: Connection to {host}:{port} timed out after {timeout} seconds"}],
            }
        except ConnectionRefusedError:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error: Connection to {host}:{port} refused - no server running on that port"}],
            }
        except Exception as e:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error connecting to {host}:{port}: {str(e)}"}],
            }
        finally:
            try:
                client_socket.close()
            except Exception:
                pass

    else:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [
                {
                    "text": f"Error: Unknown action '{action}'. Supported actions are: "
                    f"start_server, stop_server, get_status, client_send"
                }
            ],
        }
