"""
Tests for the Dynamic MCP client tool.

These tests directly call the mcp_client function rather than going through
the Agent interface for simpler and more focused testing.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands_tools.mcp_client import ConnectionInfo, MCPTool, _connections, mcp_client


@pytest.fixture
def mock_mcp_client():
    """Mock the MCPClient class for testing."""
    with patch("strands_tools.mcp_client.MCPClient") as mock_client_class:
        # Create a mock instance
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        # Setup context manager behavior
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=None)

        # Setup default tool list
        mock_tool1 = MagicMock()
        mock_tool1.tool_name = "test_tool"
        mock_tool1.tool_spec = {
            "description": "A test tool",
            "inputSchema": {"json": {"type": "object", "properties": {"param": {"type": "string"}}}},
        }

        mock_instance.list_tools_sync.return_value = [mock_tool1]

        # Setup tool call response
        mock_instance.call_tool_sync.return_value = {
            "status": "success",
            "toolUseId": "test-id",
            "content": [{"text": "Tool executed successfully"}],
        }

        yield mock_client_class, mock_instance


@pytest.fixture
def mock_stdio_client():
    """Mock stdio_client for testing."""
    with patch("strands_tools.mcp_client.stdio_client") as mock_stdio:
        mock_stdio.return_value = MagicMock()
        yield mock_stdio


@pytest.fixture
def mock_sse_client():
    """Mock sse_client for testing."""
    with patch("strands_tools.mcp_client.sse_client") as mock_sse:
        mock_sse.return_value = MagicMock()
        yield mock_sse


@pytest.fixture
def mock_streamablehttp_client():
    """Mock streamablehttp_client for testing."""
    with patch("strands_tools.mcp_client.streamablehttp_client") as mock_streamable:
        mock_streamable.return_value = MagicMock()
        yield mock_streamable


@pytest.fixture(autouse=True)
def reset_connections():
    """Reset the connections dictionary between tests."""
    # Clear all connections
    _connections.clear()

    yield _connections


class TestMCPClientConnect:
    """Test connection-related functionality."""

    def test_connect_stdio_transport(self, mock_mcp_client, mock_stdio_client):
        """Test connecting to an MCP server via stdio transport."""
        result = mcp_client(
            action="connect", connection_id="test_server", transport="stdio", command="python", args=["server.py"]
        )

        assert result["status"] == "success"
        # Text message
        assert "Connected to MCP server 'test_server'" in result["content"][0]["text"]

        # Structured data
        connection_data = result["content"][1]["json"]
        assert "Connected to MCP server 'test_server'" in connection_data["message"]
        assert connection_data["connection_id"] == "test_server"
        assert connection_data["transport"] == "stdio"
        assert connection_data["tools_count"] == 1
        assert connection_data["available_tools"] == ["test_tool"]

        # Verify MCPClient was created correctly
        mock_client_class, mock_instance = mock_mcp_client
        mock_client_class.assert_called_once()
        mock_instance.list_tools_sync.assert_called_once()

    def test_connect_streamable_http_transport(self, mock_mcp_client, mock_streamablehttp_client):
        """Test connecting to an MCP server via streamable HTTP transport."""
        result = mcp_client(
            action="connect",
            connection_id="http_server",
            transport="streamable_http",
            server_url="https://example.com/mcp",
            headers={"Authorization": "Bearer token123"},
            timeout=60,
            sse_read_timeout=180,
        )

        assert result["status"] == "success"
        # Text message
        assert "Connected to MCP server 'http_server'" in result["content"][0]["text"]

        # Structured data
        connection_data = result["content"][1]["json"]
        assert "Connected to MCP server 'http_server'" in connection_data["message"]
        assert connection_data["connection_id"] == "http_server"
        assert connection_data["transport"] == "streamable_http"
        assert connection_data["tools_count"] == 1
        assert connection_data["available_tools"] == ["test_tool"]

        # Verify MCPClient was created correctly
        mock_client_class, mock_instance = mock_mcp_client
        mock_client_class.assert_called_once()
        mock_instance.list_tools_sync.assert_called_once()

    def test_connect_streamable_http_minimal_params(self, mock_mcp_client, mock_streamablehttp_client):
        """Test connecting to streamable HTTP server with minimal parameters."""
        result = mcp_client(
            action="connect",
            connection_id="simple_http",
            transport="streamable_http",
            server_url="https://api.example.com/mcp",
        )

        assert result["status"] == "success"
        # Access structured data from new ToolResult format
        connection_data = result["content"][1]["json"]
        assert connection_data["transport"] == "streamable_http"

    def test_connect_streamable_http_missing_url(self):
        """Test connecting to streamable HTTP without server_url."""
        result = mcp_client(action="connect", connection_id="test", transport="streamable_http")
        assert result["status"] == "error"
        assert "server_url is required for streamable HTTP transport" in result["content"][0]["text"]

    def test_connect_streamable_http_with_auth(self, mock_mcp_client, mock_streamablehttp_client):
        """Test connecting to streamable HTTP server with authentication."""
        # Mock httpx auth object
        mock_auth = MagicMock()

        result = mcp_client(
            action="connect",
            connection_id="auth_http_server",
            transport="streamable_http",
            server_url="https://secure.example.com/mcp",
            auth=mock_auth,
            headers={"User-Agent": "Test-Client/1.0"},
        )

        assert result["status"] == "success"
        # Access structured data from new ToolResult format
        connection_data = result["content"][1]["json"]
        assert connection_data["connection_id"] == "auth_http_server"

    def test_connect_streamable_http_server_config(self, mock_mcp_client, mock_streamablehttp_client):
        """Test connecting using server_config with streamable HTTP parameters."""
        result = mcp_client(
            action="connect",
            connection_id="config_http_server",
            server_config={
                "transport": "streamable_http",
                "server_url": "https://config.example.com/mcp",
                "headers": {"X-API-Key": "secret123"},
                "timeout": 45,
                "sse_read_timeout": 240,
                "terminate_on_close": False,
            },
        )

        assert result["status"] == "success"
        # Access structured data from new ToolResult format
        connection_data = result["content"][1]["json"]
        assert connection_data["connection_id"] == "config_http_server"

    def test_connect_sse_transport(self, mock_mcp_client, mock_sse_client):
        """Test connecting to an MCP server via SSE transport."""
        result = mcp_client(
            action="connect", connection_id="sse_server", transport="sse", server_url="http://localhost:8080/mcp"
        )

        assert result["status"] == "success"
        # Access structured data from new ToolResult format
        connection_data = result["content"][1]["json"]
        assert "Connected to MCP server 'sse_server'" in connection_data["message"]
        assert connection_data["connection_id"] == "sse_server"
        assert connection_data["transport"] == "sse"

    def test_connect_unsupported_transport(self):
        """Test connecting with an unsupported transport type."""
        result = mcp_client(action="connect", connection_id="test", transport="unsupported_transport")
        assert result["status"] == "error"
        assert "Connection failed" in result["content"][0]["text"]
        assert "Unsupported transport: unsupported_transport" in result["content"][0]["text"]

    def test_connect_missing_required_params(self):
        """Test connecting with missing required parameters."""
        # Missing connection_id
        result = mcp_client(action="connect", transport="stdio", command="python")
        assert result["status"] == "error"
        assert "connection_id is required" in result["content"][0]["text"]

        # Missing command for stdio
        result = mcp_client(action="connect", connection_id="test", transport="stdio")
        assert result["status"] == "error"
        assert "command is required for stdio transport" in result["content"][0]["text"]

        # Missing server_url for SSE
        result = mcp_client(action="connect", connection_id="test", transport="sse")
        assert result["status"] == "error"
        assert "server_url is required for SSE transport" in result["content"][0]["text"]

        # Missing server_url for streamable HTTP
        result = mcp_client(action="connect", connection_id="test", transport="streamable_http")
        assert result["status"] == "error"
        assert "server_url is required for streamable HTTP transport" in result["content"][0]["text"]

    def test_connect_duplicate_connection(self, mock_mcp_client, mock_stdio_client):
        """Test connecting with an existing connection ID."""
        # First connection
        mcp_client(
            action="connect", connection_id="test_server", transport="stdio", command="python", args=["server.py"]
        )

        # Try to connect again with same ID
        result = mcp_client(
            action="connect", connection_id="test_server", transport="stdio", command="python", args=["server.py"]
        )

        assert result["status"] == "error"
        assert "already exists and is active" in result["content"][0]["text"]

    def test_connect_with_server_config(self, mock_mcp_client, mock_stdio_client):
        """Test connecting using server_config parameter."""
        result = mcp_client(
            action="connect",
            connection_id="test_server",
            server_config={"transport": "stdio", "command": "python", "args": ["server.py"]},
        )

        assert result["status"] == "success"
        # Access structured data from new ToolResult format
        connection_data = result["content"][1]["json"]
        assert connection_data["connection_id"] == "test_server"

    def test_connect_with_environment_variables(self, mock_mcp_client, mock_stdio_client):
        """Test connecting with environment variables (parameters passed but not stored)."""
        result = mcp_client(
            action="connect",
            connection_id="test_server_with_env",
            transport="stdio",
            command="python",
            args=["server.py"],
            env={"API_KEY": "test-key-123", "DEBUG": "true"},
        )

        assert result["status"] == "success"
        # Access structured data from new ToolResult format
        connection_data = result["content"][1]["json"]
        assert connection_data["connection_id"] == "test_server_with_env"

    def test_connect_with_env_in_server_config(self, mock_mcp_client, mock_stdio_client):
        """Test connecting with environment variables in server_config (parameters passed but not stored)."""
        result = mcp_client(
            action="connect",
            connection_id="test_server_config_env",
            server_config={
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "server-perplexity-ask"],
                "env": {"PERPLEXITY_API_KEY": "pplx-test-key"},
            },
        )

        assert result["status"] == "success"
        # Access structured data from new ToolResult format
        connection_data = result["content"][1]["json"]
        assert connection_data["connection_id"] == "test_server_config_env"

    def test_connect_failure(self, mock_mcp_client):
        """Test handling connection failure."""
        # Make list_tools_sync raise an exception
        _, mock_instance = mock_mcp_client
        mock_instance.list_tools_sync.side_effect = Exception("Connection failed")

        result = mcp_client(
            action="connect",
            connection_id="failing_server",
            transport="stdio",
            command="python",
            args=["failing_server.py"],
        )

        assert result["status"] == "error"
        assert "Connection failed" in result["content"][0]["text"]


class TestMCPClientDisconnect:
    """Test disconnection functionality."""

    def test_disconnect_active_connection(self, mock_mcp_client, mock_stdio_client):
        """Test disconnecting from an active connection."""
        # First connect
        mcp_client(
            action="connect", connection_id="test_server", transport="stdio", command="python", args=["server.py"]
        )

        # Then disconnect
        result = mcp_client(action="disconnect", connection_id="test_server")

        assert result["status"] == "success"
        # Text message
        assert "Disconnected from MCP server 'test_server'" in result["content"][0]["text"]

        # Structured data
        disconnect_data = result["content"][1]["json"]
        assert "Disconnected from MCP server 'test_server'" in disconnect_data["message"]
        assert disconnect_data["was_active"] is True

    def test_disconnect_nonexistent_connection(self):
        """Test disconnecting from a non-existent connection."""
        result = mcp_client(action="disconnect", connection_id="nonexistent")

        assert result["status"] == "error"
        assert "Connection 'nonexistent' not found" in result["content"][0]["text"]

    def test_disconnect_missing_connection_id(self):
        """Test disconnecting without providing connection_id."""
        result = mcp_client(action="disconnect")

        assert result["status"] == "error"
        assert "connection_id is required" in result["content"][0]["text"]

    def test_disconnect_with_loaded_tools(self, mock_mcp_client, mock_stdio_client, reset_connections):
        """Test disconnecting from a connection with loaded tools."""
        # Connect
        mcp_client(
            action="connect", connection_id="test_server", transport="stdio", command="python", args=["server.py"]
        )

        # Track some loaded tools by updating the connection info
        if "test_server" in _connections:
            _connections["test_server"].loaded_tool_names = ["tool1", "tool2"]

        # Disconnect
        result = mcp_client(action="disconnect", connection_id="test_server")

        assert result["status"] == "success"
        # Access structured data from new ToolResult format
        disconnect_data = result["content"][1]["json"]
        assert "loaded_tools_info" in disconnect_data
        assert "2 tools loaded" in disconnect_data["loaded_tools_info"]
        assert "tool1" in disconnect_data["loaded_tools_info"]


class TestMCPClientListConnections:
    """Test listing connections functionality."""

    def test_list_empty_connections(self):
        """Test listing connections when none exist."""
        result = mcp_client(action="list_connections")

        assert result["status"] == "success"
        # Text message
        assert "Found 0 MCP connections" in result["content"][0]["text"]

        # Structured data
        connections_data = result["content"][1]["json"]
        assert connections_data["total_connections"] == 0
        assert connections_data["connections"] == []

    def test_list_multiple_connections(
        self, mock_mcp_client, mock_stdio_client, mock_sse_client, mock_streamablehttp_client
    ):
        """Test listing multiple connections."""
        # Create multiple connections
        mcp_client(
            action="connect", connection_id="stdio_server", transport="stdio", command="python", args=["server1.py"]
        )

        mcp_client(
            action="connect", connection_id="sse_server", transport="sse", server_url="http://localhost:8080/mcp"
        )

        mcp_client(
            action="connect",
            connection_id="http_server",
            transport="streamable_http",
            server_url="https://example.com/mcp",
        )

        # List connections
        result = mcp_client(action="list_connections")

        assert result["status"] == "success"
        # Access structured data from new ToolResult format
        connections_data = result["content"][1]["json"]
        assert connections_data["total_connections"] == 3
        assert len(connections_data["connections"]) == 3

        # Verify connection details
        conn_ids = [conn["connection_id"] for conn in connections_data["connections"]]
        assert "stdio_server" in conn_ids
        assert "sse_server" in conn_ids
        assert "http_server" in conn_ids

        # Check stdio connection
        stdio_conn = next(c for c in connections_data["connections"] if c["connection_id"] == "stdio_server")
        assert stdio_conn["transport"] == "stdio"
        assert stdio_conn["is_active"] is True

        # Check SSE connection
        sse_conn = next(c for c in connections_data["connections"] if c["connection_id"] == "sse_server")
        assert sse_conn["transport"] == "sse"
        assert sse_conn["url"] == "http://localhost:8080/mcp"

        # Check streamable HTTP connection
        http_conn = next(c for c in connections_data["connections"] if c["connection_id"] == "http_server")
        assert http_conn["transport"] == "streamable_http"
        assert http_conn["url"] == "https://example.com/mcp"


class TestMCPClientListTools:
    """Test listing tools functionality."""

    def test_list_tools_success(self, mock_mcp_client, mock_stdio_client):
        """Test listing tools from a connected server."""
        # Connect first
        mcp_client(
            action="connect", connection_id="test_server", transport="stdio", command="python", args=["server.py"]
        )

        # List tools
        result = mcp_client(action="list_tools", connection_id="test_server")

        assert result["status"] == "success"
        # Text message
        assert "Found 1 tools on MCP server 'test_server'" in result["content"][0]["text"]

        # Structured data
        tools_data = result["content"][1]["json"]
        assert tools_data["connection_id"] == "test_server"
        assert tools_data["tools_count"] == 1
        assert len(tools_data["tools"]) == 1
        assert tools_data["tools"][0]["name"] == "test_tool"
        assert tools_data["tools"][0]["description"] == "A test tool"

    def test_list_tools_nonexistent_connection(self):
        """Test listing tools from a non-existent connection."""
        result = mcp_client(action="list_tools", connection_id="nonexistent")

        assert result["status"] == "error"
        assert "Connection 'nonexistent' not found" in result["content"][0]["text"]

    def test_list_tools_missing_connection_id(self):
        """Test listing tools without providing connection_id."""
        result = mcp_client(action="list_tools")

        assert result["status"] == "error"
        assert "connection_id is required" in result["content"][0]["text"]

    def test_list_tools_connection_failure(self, mock_mcp_client, mock_stdio_client):
        """Test handling errors when listing tools."""
        # Connect first
        mcp_client(
            action="connect", connection_id="test_server", transport="stdio", command="python", args=["server.py"]
        )

        # Make list_tools_sync fail
        _, mock_instance = mock_mcp_client
        mock_instance.list_tools_sync.side_effect = Exception("Server error")

        # Try to list tools
        result = mcp_client(action="list_tools", connection_id="test_server")

        assert result["status"] == "error"
        assert "Failed to list tools" in result["content"][0]["text"]
        assert "Server error" in result["content"][0]["text"]


class TestMCPClientCallTool:
    """Test calling tools functionality."""

    def test_call_tool_success(self, mock_mcp_client, mock_stdio_client):
        """Test successfully calling a tool."""
        # Connect first
        mcp_client(
            action="connect", connection_id="test_server", transport="stdio", command="python", args=["server.py"]
        )

        # Call a tool
        result = mcp_client(
            action="call_tool", connection_id="test_server", tool_name="test_tool", tool_args={"param": "value"}
        )

        assert result["status"] == "success"
        assert result["toolUseId"] == "test-id"
        assert result["content"][0]["text"] == "Tool executed successfully"

    def test_call_tool_with_direct_params(self, mock_mcp_client, mock_stdio_client):
        """Test calling a tool with parameters passed directly - they should be explicitly provided in tool_args."""
        # Connect first
        mcp_client(
            action="connect", connection_id="test_server", transport="stdio", command="python", args=["server.py"]
        )

        # Call tool with direct parameters - should now use tool_args explicitly
        result = mcp_client(
            action="call_tool",
            connection_id="test_server",
            tool_name="test_tool",
            tool_args={"param": "direct_value", "another_param": 123},
        )

        assert result["status"] == "success"
        # Verify the SDK call was made with the correct arguments
        _, mock_instance = mock_mcp_client
        mock_instance.call_tool_sync.assert_called_with(
            tool_use_id="mcp_test_server_test_tool",
            name="test_tool",
            arguments={"param": "direct_value", "another_param": 123},
        )

    def test_call_tool_missing_params(self):
        """Test calling a tool with missing parameters."""
        # Missing connection_id
        result = mcp_client(action="call_tool", tool_name="test_tool")
        assert result["status"] == "error"
        assert "connection_id is required" in result["content"][0]["text"]

        # Missing tool_name
        result = mcp_client(action="call_tool", connection_id="test_server")
        assert result["status"] == "error"
        assert "tool_name is required" in result["content"][0]["text"]

    def test_call_tool_error(self, mock_mcp_client, mock_stdio_client):
        """Test handling errors when calling a tool."""
        # Connect first
        mcp_client(
            action="connect", connection_id="test_server", transport="stdio", command="python", args=["server.py"]
        )

        # Make call_tool_sync fail
        _, mock_instance = mock_mcp_client
        mock_instance.call_tool_sync.side_effect = Exception("Tool execution failed")

        # Try to call tool
        result = mcp_client(action="call_tool", connection_id="test_server", tool_name="test_tool")

        assert result["status"] == "error"
        assert "Failed to call tool" in result["content"][0]["text"]
        assert "Tool execution failed" in result["content"][0]["text"]


class TestMCPClientLoadTools:
    """Test loading tools functionality."""

    def test_load_tools_success(self, mock_mcp_client, mock_stdio_client):
        """Test successfully loading tools into agent with MCPTool wrapper."""
        # Mock agent's tool_registry
        mock_agent = MagicMock()
        mock_registry = MagicMock()
        mock_registry.register_tool = MagicMock()
        mock_agent.tool_registry = mock_registry

        # Connect first
        mcp_client(
            action="connect", connection_id="test_server", transport="stdio", command="python", args=["server.py"]
        )

        # Load tools
        result = mcp_client(action="load_tools", connection_id="test_server", agent=mock_agent)

        assert result["status"] == "success"
        # Text message
        assert "Loaded 1 tools from MCP server 'test_server'" in result["content"][0]["text"]

        # Structured data
        load_data = result["content"][1]["json"]
        assert "Loaded 1 tools" in load_data["message"]
        assert load_data["loaded_tools"] == ["test_tool"]
        assert load_data["tool_count"] == 1

        # Verify tool was registered as MCPTool wrapper
        mock_registry.register_tool.assert_called_once()
        registered_tool = mock_registry.register_tool.call_args[0][0]
        assert isinstance(registered_tool, MCPTool)
        assert registered_tool.tool_name == "test_tool"
        assert registered_tool._connection_id == "test_server"

    def test_load_tools_no_agent(self, mock_mcp_client, mock_stdio_client):
        """Test loading tools without agent instance."""
        # Connect first to ensure we get to the agent check
        mcp_client(
            action="connect", connection_id="test_server", transport="stdio", command="python", args=["server.py"]
        )

        result = mcp_client(action="load_tools", connection_id="test_server")

        assert result["status"] == "error"
        # The current implementation expects this specific message
        assert "agent instance is required" in result["content"][0]["text"]

    def test_load_tools_no_registry(self, mock_mcp_client, mock_stdio_client):
        """Test loading tools with agent that has no tool registry."""
        # Create a mock agent without tool_registry
        mock_agent = MagicMock(spec=[])  # No attributes

        # Connect first
        mcp_client(
            action="connect", connection_id="test_server", transport="stdio", command="python", args=["server.py"]
        )

        # Try to load tools
        result = mcp_client(action="load_tools", connection_id="test_server", agent=mock_agent)

        assert result["status"] == "error"
        assert "Agent does not have a tool registry" in result["content"][0]["text"]

    def test_load_tools_inactive_connection(self, reset_connections):
        """Test loading tools from an inactive connection."""
        # Manually register an inactive connection
        mock_client = MagicMock()
        config = ConnectionInfo(
            connection_id="inactive_server",
            mcp_client=mock_client,
            transport="stdio",
            url="python server.py",
            register_time=0,
            is_active=False,
        )
        _connections["inactive_server"] = config

        # Mock agent
        mock_agent = MagicMock()

        # Try to load tools
        result = mcp_client(action="load_tools", connection_id="inactive_server", agent=mock_agent)

        assert result["status"] == "error"
        assert "Connection 'inactive_server' is not active" in result["content"][0]["text"]

    def test_load_tools_with_errors(self, mock_mcp_client, mock_stdio_client):
        """Test loading tools with some registration failures."""
        # Setup multiple tools
        mock_tool1 = MagicMock()
        mock_tool1.tool_name = "tool1"
        mock_tool1.tool_spec = {"description": "Tool 1"}

        mock_tool2 = MagicMock()
        mock_tool2.tool_name = "tool2"
        mock_tool2.tool_spec = {"description": "Tool 2"}

        _, mock_instance = mock_mcp_client
        mock_instance.list_tools_sync.return_value = [mock_tool1, mock_tool2]

        mock_agent = MagicMock()
        mock_registry = MagicMock()
        mock_registry.register_tool.side_effect = [None, Exception("Registration failed")]
        mock_agent.tool_registry = mock_registry

        # Connect
        mcp_client(
            action="connect", connection_id="test_server", transport="stdio", command="python", args=["server.py"]
        )

        # Load tools
        result = mcp_client(action="load_tools", connection_id="test_server", agent=mock_agent)

        assert result["status"] == "success"
        # Access structured data from new ToolResult format
        load_data = result["content"][1]["json"]
        assert "Loaded 1 tools" in load_data["message"]
        assert load_data["loaded_tools"] == ["tool1"]
        assert "skipped_tools" in load_data
        assert load_data["skipped_tools"][0]["name"] == "tool2"
        assert "Registration failed" in load_data["skipped_tools"][0]["error"]


class TestMCPToolClass:
    """Test the MCPTool wrapper class."""

    @pytest.fixture
    def mock_mcp_tool(self):
        """Create a mock MCP tool."""
        mock_tool = MagicMock()
        mock_tool.tool_name = "test_tool"
        mock_tool.tool_spec = {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {"json": {"type": "object", "properties": {"param": {"type": "string"}}}},
        }
        return mock_tool

    @pytest.fixture
    def mock_connection_config(self, mock_mcp_client):
        """Create a mock connection config."""
        mock_client_class, mock_instance = mock_mcp_client
        connection_info = ConnectionInfo(
            connection_id="test_connection",
            mcp_client=mock_instance,
            transport="stdio",
            url="python server.py",
            register_time=0,
            is_active=True,
        )
        _connections["test_connection"] = connection_info
        return connection_info

    def test_mcp_tool_initialization(self, mock_mcp_tool):
        """Test MCPTool initialization."""
        mcp_tool = MCPTool(mock_mcp_tool, "test_connection")

        assert mcp_tool.tool_name == "test_tool"
        assert mcp_tool.tool_spec == mock_mcp_tool.tool_spec
        assert mcp_tool.tool_type == "mcp_dynamic"
        assert mcp_tool._connection_id == "test_connection"
        assert mcp_tool._mcp_tool == mock_mcp_tool

    def test_mcp_tool_display_properties(self, mock_mcp_tool):
        """Test MCPTool display properties."""
        mcp_tool = MCPTool(mock_mcp_tool, "test_connection")
        props = mcp_tool.get_display_properties()

        assert props["Name"] == "test_tool"
        assert props["Type"] == "mcp_dynamic"
        assert props["Connection ID"] == "test_connection"

    @pytest.mark.asyncio
    async def test_mcp_tool_stream_success(self, mock_mcp_tool, mock_connection_config):
        """Test successful MCPTool stream execution."""
        mcp_tool = MCPTool(mock_mcp_tool, "test_connection")

        # Mock the async call_tool_async method
        mock_result = {
            "toolUseId": "test-tool-use-id",
            "status": "success",
            "content": [{"text": "Tool executed successfully"}],
        }
        mock_connection_config.mcp_client.call_tool_async = AsyncMock(return_value=mock_result)

        tool_use = {"toolUseId": "test-tool-use-id", "name": "test_tool", "input": {"param": "test_value"}}

        # Execute the stream method
        results = []
        async for result in mcp_tool.stream(tool_use, {}):
            results.append(result)

        assert len(results) == 1
        assert results[0] == mock_result

        # Verify the call_tool_async was called correctly
        mock_connection_config.mcp_client.call_tool_async.assert_called_once_with(
            tool_use_id="test-tool-use-id", name="test_tool", arguments={"param": "test_value"}
        )

    @pytest.mark.asyncio
    async def test_mcp_tool_stream_connection_not_found(self, mock_mcp_tool):
        """Test MCPTool stream when connection is not found."""
        mcp_tool = MCPTool(mock_mcp_tool, "nonexistent_connection")

        tool_use = {"toolUseId": "test-tool-use-id", "name": "test_tool", "input": {"param": "test_value"}}

        # Execute the stream method
        results = []
        async for result in mcp_tool.stream(tool_use, {}):
            results.append(result)

        assert len(results) == 1
        assert results[0]["toolUseId"] == "test-tool-use-id"
        assert results[0]["status"] == "error"
        assert "Connection 'nonexistent_connection' not found" in results[0]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_mcp_tool_stream_connection_inactive(self, mock_mcp_tool, mock_connection_config):
        """Test MCPTool stream when connection is inactive."""
        # Make connection inactive
        mock_connection_config.is_active = False

        mcp_tool = MCPTool(mock_mcp_tool, "test_connection")

        tool_use = {"toolUseId": "test-tool-use-id", "name": "test_tool", "input": {"param": "test_value"}}

        # Execute the stream method
        results = []
        async for result in mcp_tool.stream(tool_use, {}):
            results.append(result)

        assert len(results) == 1
        assert results[0]["toolUseId"] == "test-tool-use-id"
        assert results[0]["status"] == "error"
        assert "Connection 'test_connection' is not active" in results[0]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_mcp_tool_stream_execution_error(self, mock_mcp_tool, mock_connection_config):
        """Test MCPTool stream when tool execution fails."""
        mcp_tool = MCPTool(mock_mcp_tool, "test_connection")

        # Mock the async call_tool_async method to raise an exception
        mock_connection_config.mcp_client.call_tool_async = AsyncMock(side_effect=Exception("Tool execution failed"))

        tool_use = {"toolUseId": "test-tool-use-id", "name": "test_tool", "input": {"param": "test_value"}}

        # Execute the stream method
        results = []
        async for result in mcp_tool.stream(tool_use, {}):
            results.append(result)

        assert len(results) == 1
        assert results[0]["toolUseId"] == "test-tool-use-id"
        assert results[0]["status"] == "error"
        assert "Failed to execute tool 'test_tool'" in results[0]["content"][0]["text"]
        assert "Tool execution failed" in results[0]["content"][0]["text"]

        # Verify connection was marked as inactive
        assert not mock_connection_config.is_active
        assert mock_connection_config.last_error == "Tool execution failed"

    @pytest.mark.asyncio
    async def test_mcp_tool_stream_uses_context_manager(self, mock_mcp_tool, mock_connection_config):
        """Test that MCPTool stream uses the context manager pattern."""
        mcp_tool = MCPTool(mock_mcp_tool, "test_connection")

        # Mock the async call_tool_async method
        mock_result = {"toolUseId": "test-tool-use-id", "status": "success", "content": [{"text": "Success"}]}
        mock_connection_config.mcp_client.call_tool_async = AsyncMock(return_value=mock_result)

        tool_use = {"toolUseId": "test-tool-use-id", "name": "test_tool", "input": {"param": "test_value"}}

        # Execute the stream method
        results = []
        async for result in mcp_tool.stream(tool_use, {}):
            results.append(result)

        # Verify that the context manager was used
        mock_connection_config.mcp_client.__enter__.assert_called_once()
        mock_connection_config.mcp_client.__exit__.assert_called_once()


class TestMCPClientInvalidAction:
    """Test handling of invalid actions."""

    def test_invalid_action(self):
        """Test calling with an invalid action."""
        result = mcp_client(action="invalid_action", connection_id="test")

        assert result["status"] == "error"
        assert "Unknown action: invalid_action" in result["content"][0]["text"]
        assert "Available actions:" in result["content"][0]["text"]


class TestBinaryDataHandling:
    """Test binary data handling through direct tool calls."""

    def test_call_server_tool_with_binary_data(self, mock_mcp_client, mock_stdio_client):
        """Test calling a tool that returns binary data directly."""
        # Mock the MCP client to return binary data
        _, mock_instance = mock_mcp_client
        mock_result = {
            "status": "success",
            "toolUseId": "test-tool-use-id",
            "content": [
                {"type": "text", "text": "Generated image"},
                {"type": "image", "data": "[BASE64_DATA]iVBORw0KGgoAAAANSUhE..."},
                {"type": "text", "text": "Raw bytes"},
            ],
        }
        mock_instance.call_tool_sync.return_value = mock_result

        # Connect first
        mcp_client(
            action="connect", connection_id="binary_tool_test", transport="stdio", command="python", args=["server.py"]
        )

        # Call the tool that returns binary data
        result = mcp_client(
            action="call_tool",
            connection_id="binary_tool_test",
            tool_name="generate_image",
            tool_args={"width": 100, "height": 100},
        )

        # Verify the result - it should return the mock result directly
        assert result["status"] == "success"
        assert result["toolUseId"] == "test-tool-use-id"
        assert len(result["content"]) == 3


class TestMCPClientCleanup:
    """Test cleanup functionality for MCP client tools."""

    def test_disconnect_cleans_up_tools(self, mock_mcp_client, mock_stdio_client):
        """Test that disconnecting cleans up loaded tools."""
        # Create a mock agent
        mock_agent = MagicMock()
        mock_registry = MagicMock()
        mock_registry.unregister_tool = MagicMock()
        mock_agent.tool_registry = mock_registry

        # Connect first
        mcp_client(
            action="connect", connection_id="test_connection", transport="stdio", command="python", args=["server.py"]
        )

        # Manually set some loaded tools on the connection
        if "test_connection" in _connections:
            _connections["test_connection"].loaded_tool_names = ["test_tool", "another_tool"]

        # Call disconnect with the agent
        result = mcp_client(action="disconnect", connection_id="test_connection", agent=mock_agent)

        # Check that unregister_tool was called for each loaded tool
        assert mock_agent.tool_registry.unregister_tool.call_count == 2
        mock_agent.tool_registry.unregister_tool.assert_any_call("test_tool")
        mock_agent.tool_registry.unregister_tool.assert_any_call("another_tool")

        # Check the result
        assert result["status"] == "success"
        # Access structured data from new ToolResult format
        disconnect_data = result["content"][1]["json"]
        assert "cleaned_tools" in disconnect_data
        assert len(disconnect_data["cleaned_tools"]) == 2
        assert "test_connection" not in _connections

    def test_call_tool_cleans_up_on_connection_failure(self, mock_mcp_client):
        """Test that call_tool cleans up tools when connection fails during call."""
        # Create a mock agent
        mock_agent = MagicMock()
        mock_registry = MagicMock()
        mock_registry.unregister_tool = MagicMock()
        mock_agent.tool_registry = mock_registry

        # Setup mock client that fails on tool call
        _, mock_instance = mock_mcp_client
        mock_instance.call_tool_sync.side_effect = Exception("Connection failed during call")

        # Connect first
        mcp_client(
            action="connect",
            connection_id="failing_call_connection",
            transport="stdio",
            command="python",
            args=["server.py"],
        )

        # Manually set some loaded tools
        if "failing_call_connection" in _connections:
            _connections["failing_call_connection"].loaded_tool_names = ["test_tool"]

        # Call the tool (should fail and trigger cleanup)
        result = mcp_client(
            action="call_tool", connection_id="failing_call_connection", tool_name="test_tool", agent=mock_agent
        )

        # Check the result indicates failure
        assert result["status"] == "error"
        assert "Failed to call tool" in result["content"][0]["text"]


class TestMCPClientIntegration:
    """Integration tests for full workflows."""

    def test_full_workflow(self, mock_mcp_client, mock_stdio_client):
        """Test a complete workflow: connect, list tools, call tool, disconnect."""
        # 1. Connect
        connect_result = mcp_client(
            action="connect", connection_id="workflow_server", transport="stdio", command="python", args=["server.py"]
        )
        assert connect_result["status"] == "success"

        # 2. List connections
        list_result = mcp_client(action="list_connections")
        assert list_result["status"] == "success"
        # Access structured data from new ToolResult format
        connections_data = list_result["content"][1]["json"]
        assert connections_data["total_connections"] == 1

        # 3. List tools
        tools_result = mcp_client(action="list_tools", connection_id="workflow_server")
        assert tools_result["status"] == "success"
        # Access structured data from new ToolResult format
        tools_data = tools_result["content"][1]["json"]
        assert tools_data["tools_count"] == 1

        # 4. Call a tool
        call_result = mcp_client(
            action="call_tool", connection_id="workflow_server", tool_name="test_tool", tool_args={"param": "test"}
        )
        assert call_result["status"] == "success"

        # 5. Disconnect
        disconnect_result = mcp_client(action="disconnect", connection_id="workflow_server")
        assert disconnect_result["status"] == "success"

        # 6. Verify connection is gone
        final_list = mcp_client(action="list_connections")
        final_connections_data = final_list["content"][1]["json"]
        assert final_connections_data["total_connections"] == 0


# The MCPToolWrapper class functionality is now tested through the MCPTool class tests above


# Configuration tests are handled within other test classes

# Result handling is tested through MCPTool class tests above
