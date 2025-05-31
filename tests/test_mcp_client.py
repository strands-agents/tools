"""
Tests for the MCP client tool.

These tests directly call the mcp_client function rather than going through
the Agent interface for simpler and more focused testing.
"""

from unittest.mock import MagicMock, patch

import pytest
from strands_tools.mcp_client import ConnectionInfo, _connections, create_mcp_tool_wrapper, mcp_client


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
        assert "Connected to MCP server 'test_server'" in result["message"]
        assert result["connection_id"] == "test_server"
        assert result["transport"] == "stdio"
        assert result["tools_count"] == 1
        assert result["available_tools"] == ["test_tool"]

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
        assert "Connected to MCP server 'http_server'" in result["message"]
        assert result["connection_id"] == "http_server"
        assert result["transport"] == "streamable_http"
        assert result["tools_count"] == 1
        assert result["available_tools"] == ["test_tool"]

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
        assert result["transport"] == "streamable_http"

    def test_connect_streamable_http_missing_url(self):
        """Test connecting to streamable HTTP without server_url."""
        result = mcp_client(action="connect", connection_id="test", transport="streamable_http")
        assert result["status"] == "error"
        assert "server_url is required for streamable HTTP transport" in result["content"][0]["text"]

    def test_connect_streamable_http_with_auth(self, mock_mcp_client, mock_streamablehttp_client):
        """Test connecting to streamable HTTP server with authentication."""
        from unittest.mock import MagicMock

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
        assert result["connection_id"] == "auth_http_server"

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
        assert result["connection_id"] == "config_http_server"

    def test_connect_sse_transport(self, mock_mcp_client, mock_sse_client):
        """Test connecting to an MCP server via SSE transport."""
        result = mcp_client(
            action="connect", connection_id="sse_server", transport="sse", server_url="http://localhost:8080/mcp"
        )

        assert result["status"] == "success"
        assert "Connected to MCP server 'sse_server'" in result["message"]
        assert result["connection_id"] == "sse_server"
        assert result["transport"] == "sse"

    def test_connect_unsupported_transport(self):
        """Test connecting with an unsupported transport type."""
        result = mcp_client(action="connect", connection_id="test", transport="unsupported_transport")
        assert result["status"] == "error"
        assert "Connection test failed" in result["content"][0]["text"]
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
        assert result["connection_id"] == "test_server"

    def test_connect_with_environment_variables(self, mock_mcp_client):
        """Test connecting with environment variables."""
        with patch("strands_tools.mcp_client._create_transport_callable") as mock_create_transport:
            # Set up the mock to return a callable
            mock_transport = MagicMock()
            mock_create_transport.return_value = mock_transport

            result = mcp_client(
                action="connect",
                connection_id="test_server_with_env",
                transport="stdio",
                command="python",
                args=["server.py"],
                env={"API_KEY": "test-key-123", "DEBUG": "true"},
            )

            assert result["status"] == "success"
            assert result["connection_id"] == "test_server_with_env"

            # Check that _create_transport_callable was called with the right config
            mock_create_transport.assert_called_once()
            config = mock_create_transport.call_args[0][0]
            assert config.transport == "stdio"
            assert config.command == "python"
            assert config.args == ["server.py"]
            assert config.env == {"API_KEY": "test-key-123", "DEBUG": "true"}

            # Check that the connection info includes environment variables
            from strands_tools.mcp_client import _connections

            conn_info = _connections.get("test_server_with_env")
            assert conn_info is not None
            assert conn_info.env == {"API_KEY": "test-key-123", "DEBUG": "true"}

    def test_connect_with_env_in_server_config(self, mock_mcp_client):
        """Test connecting with environment variables in server_config."""
        with patch("strands_tools.mcp_client._create_transport_callable") as mock_create_transport:
            # Set up the mock to return a callable
            mock_transport = MagicMock()
            mock_create_transport.return_value = mock_transport

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
            assert result["connection_id"] == "test_server_config_env"

            # Check that _create_transport_callable was called with the right config
            mock_create_transport.assert_called_once()
            config = mock_create_transport.call_args[0][0]
            assert config.transport == "stdio"
            assert config.command == "npx"
            assert config.args == ["-y", "server-perplexity-ask"]
            assert config.env == {"PERPLEXITY_API_KEY": "pplx-test-key"}

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
        assert "Connection test failed" in result["content"][0]["text"]
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
        assert "Disconnected from MCP server 'test_server'" in result["message"]
        assert result["was_active"] is True

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
            _connections["test_server"].loaded_tool_names = ["mcp_test_server_tool1", "mcp_test_server_tool2"]

        # Disconnect
        result = mcp_client(action="disconnect", connection_id="test_server")

        assert result["status"] == "success"
        assert "loaded_tools_info" in result
        assert "2 tools loaded" in result["loaded_tools_info"]
        assert "mcp_test_server_tool1" in result["loaded_tools_info"]


class TestMCPClientListConnections:
    """Test listing connections functionality."""

    def test_list_empty_connections(self):
        """Test listing connections when none exist."""
        result = mcp_client(action="list_connections")

        assert result["status"] == "success"
        assert result["total_connections"] == 0
        assert result["connections"] == []

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
        assert result["total_connections"] == 3
        assert len(result["connections"]) == 3

        # Verify connection details
        conn_ids = [conn["connection_id"] for conn in result["connections"]]
        assert "stdio_server" in conn_ids
        assert "sse_server" in conn_ids
        assert "http_server" in conn_ids

        # Check stdio connection
        stdio_conn = next(c for c in result["connections"] if c["connection_id"] == "stdio_server")
        assert stdio_conn["transport"] == "stdio"
        assert stdio_conn["is_active"] is True
        assert stdio_conn["tools_count"] == 1

        # Check SSE connection
        sse_conn = next(c for c in result["connections"] if c["connection_id"] == "sse_server")
        assert sse_conn["transport"] == "sse"
        assert sse_conn["url"] == "http://localhost:8080/mcp"

        # Check streamable HTTP connection
        http_conn = next(c for c in result["connections"] if c["connection_id"] == "http_server")
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
        assert result["connection_id"] == "test_server"
        assert result["tools_count"] == 1
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "test_tool"
        assert result["tools"][0]["description"] == "A test tool"

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
        assert result["connection_id"] == "test_server"
        assert result["tool_name"] == "test_tool"
        assert result["tool_arguments"] == {"param": "value"}
        assert result["tool_result"]["status"] == "success"
        assert result["tool_result"]["content"][0]["text"] == "Tool executed successfully"

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
        # Tool arguments should contain the explicitly provided values
        assert result["tool_arguments"]["param"] == "direct_value"
        assert result["tool_arguments"]["another_param"] == 123

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
        """Test successfully loading tools into agent."""
        # Mock agent's tool_registry
        from unittest.mock import MagicMock

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
        assert "Loaded 1 tools" in result["message"]
        assert result["loaded_tools"] == ["mcp_test_server_test_tool"]
        assert result["tool_count"] == 1

        # Verify tool was registered
        mock_registry.register_tool.assert_called_once()

    def test_load_tools_no_agent(self, mock_mcp_client, mock_stdio_client):
        """Test loading tools without agent instance."""
        # Connect first to ensure we get to the agent check
        mcp_client(
            action="connect", connection_id="test_server", transport="stdio", command="python", args=["server.py"]
        )

        result = mcp_client(action="load_tools", connection_id="test_server")

        assert result["status"] == "error"
        # For direct call without agent passed as parameter
        is_agent_param_required = "agent parameter is required" in result["content"][0]["text"]
        is_agent_instance_available = "Agent instance not available" in result["content"][0]["text"]
        assert is_agent_param_required or is_agent_instance_available

    def test_load_tools_no_registry(self, mock_mcp_client, mock_stdio_client):
        """Test loading tools with agent that has no tool registry."""
        # Create a mock agent without tool_registry
        from unittest.mock import MagicMock

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
        config = ConnectionInfo(
            transport="stdio",
            command="python",
            args=["server.py"],
            register_time=0,
            url="python server.py",
            is_active=False,
        )
        _connections["inactive_server"] = config

        # Mock agent
        from unittest.mock import MagicMock

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
        assert "Loaded 1 tools" in result["message"]
        assert result["loaded_tools"] == ["mcp_test_server_tool1"]
        assert "skipped_tools" in result
        assert result["skipped_tools"][0]["name"] == "tool2"
        assert "Registration failed" in result["skipped_tools"][0]["error"]


class TestMCPClientInvalidAction:
    """Test handling of invalid actions."""

    def test_invalid_action(self):
        """Test calling with an invalid action."""
        result = mcp_client(action="invalid_action", connection_id="test")

        assert result["status"] == "error"
        assert "Unknown action: invalid_action" in result["content"][0]["text"]
        assert "Available actions:" in result["content"][0]["text"]


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
        assert list_result["total_connections"] == 1

        # 3. List tools
        tools_result = mcp_client(action="list_tools", connection_id="workflow_server")
        assert tools_result["status"] == "success"
        assert tools_result["tools_count"] == 1

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
        assert final_list["total_connections"] == 0


class TestMCPToolWrapper:
    """Test the MCPToolWrapper class."""

    def test_wrapper_creation(self):
        """Test creating a tool wrapper."""
        mock_client = MagicMock()
        tool_info = {
            "name": "test_tool",
            "description": "A test tool for unit testing",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "First parameter"},
                        "param2": {"type": "number", "description": "Second parameter"},
                    },
                    "required": ["param1"],
                }
            },
        }

        wrapper = create_mcp_tool_wrapper(
            connection_id="test_conn",
            tool_info=tool_info,
            mcp_client=mock_client,
            name_prefix=False,
        )

        assert wrapper.name == "test_tool"
        assert wrapper.tool_name == "test_tool"
        assert wrapper.description == "A test tool for unit testing"
        assert wrapper.connection_id == "test_conn"
        assert wrapper.original_tool_name == "test_tool"
        assert wrapper.tool_type == "mcp"
        assert wrapper.supports_hot_reload is False

    def test_wrapper_with_name_prefix(self):
        """Test creating a tool wrapper with name prefix."""
        mock_client = MagicMock()
        tool_info = {
            "name": "my-tool",
            "description": "Test tool",
        }

        wrapper = create_mcp_tool_wrapper(
            connection_id="test-conn",
            tool_info=tool_info,
            mcp_client=mock_client,
            name_prefix=True,
        )

        # Check that dashes are replaced with underscores
        assert wrapper.name == "mcp_test_conn_my_tool"
        assert wrapper.original_tool_name == "my-tool"

    def test_wrapper_invoke_success(self):
        """Test successfully invoking a wrapped tool."""
        mock_client = MagicMock()
        mock_client.call_tool_sync.return_value = {
            "status": "success",
            "content": [{"text": "Tool executed successfully"}],
        }

        tool_info = {"name": "test_tool", "description": "Test tool"}
        wrapper = create_mcp_tool_wrapper(
            connection_id="test_conn",
            tool_info=tool_info,
            mcp_client=mock_client,
        )

        # Create a tool use request
        tool_use = {"toolUseId": "test-id-123", "input": {"param1": "value1", "param2": 42}}

        result = wrapper.invoke(tool_use)

        assert result["status"] == "success"
        assert result["toolUseId"] == "test-id-123"
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "Tool executed successfully"

        # Verify the client was called correctly
        mock_client.call_tool_sync.assert_called_once()
        call_args = mock_client.call_tool_sync.call_args[1]
        assert call_args["name"] == "test_tool"
        assert call_args["arguments"] == {"param1": "value1", "param2": 42}

    def test_wrapper_invoke_error(self):
        """Test handling errors when invoking a wrapped tool."""
        mock_client = MagicMock()
        mock_client.call_tool_sync.return_value = {"status": "error", "content": [{"text": "Tool execution failed"}]}

        tool_info = {"name": "test_tool", "description": "Test tool"}
        wrapper = create_mcp_tool_wrapper(
            connection_id="test_conn",
            tool_info=tool_info,
            mcp_client=mock_client,
        )

        tool_use = {"toolUseId": "test-id-123", "input": {}}
        result = wrapper.invoke(tool_use)

        assert result["status"] == "error"
        assert result["toolUseId"] == "test-id-123"
        assert "Tool execution failed" in result["content"][0]["text"]

    def test_wrapper_invoke_exception(self):
        """Test handling exceptions when invoking a wrapped tool."""
        mock_client = MagicMock()
        mock_client.call_tool_sync.side_effect = Exception("Connection lost")

        tool_info = {"name": "test_tool", "description": "Test tool"}
        wrapper = create_mcp_tool_wrapper(
            connection_id="test_conn",
            tool_info=tool_info,
            mcp_client=mock_client,
        )

        tool_use = {"toolUseId": "test-id-123", "input": {}}
        result = wrapper.invoke(tool_use)

        assert result["status"] == "error"
        assert result["toolUseId"] == "test-id-123"
        assert "Error calling MCP tool: Connection lost" in result["content"][0]["text"]

    def test_wrapper_display_properties(self):
        """Test getting display properties from wrapper."""
        mock_client = MagicMock()
        tool_info = {"name": "test_tool", "description": "Test tool"}
        wrapper = create_mcp_tool_wrapper(
            connection_id="test_conn",
            tool_info=tool_info,
            mcp_client=mock_client,
            name_prefix=True,
        )

        props = wrapper.get_display_properties()
        assert props["connection_id"] == "test_conn"
        assert props["original_name"] == "test_tool"
        assert props["wrapped_name"] == "mcp_test_conn_test_tool"
        assert props["type"] == "mcp"
