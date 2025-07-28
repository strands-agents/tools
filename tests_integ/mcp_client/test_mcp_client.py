"""
Integration tests for the Dynamic MCP Client tool.

These tests use a MockMCP server (FastMCP) to validate end-to-end functionality 
of the mcp_client tool with Strands agents.
"""

import os
from unittest.mock import patch

import pytest
from strands import Agent
from strands_tools import mcp_client

from .test_helpers import parse_tool_result, stdio_mcp_server, sse_mcp_server


@pytest.fixture(autouse=True)
def bypass_tool_consent():
    """Bypass tool consent for integration tests."""
    with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
        yield


@pytest.fixture
def agent():
    """Create an Agent instance configured with dynamic MCP client tool."""
    return Agent(tools=[mcp_client])


class TestMCPClientBasicFunctionality:
    """Test basic MCP client functionality that doesn't require real servers."""

    def test_list_connections_empty(self, agent):
        """Test listing connections when none exist."""
        result = agent.tool.mcp_client(action="list_connections")
        
        assert result["status"] == "success"
        # Text message
        assert "Found 0 MCP connections" in result["content"][0]["text"]
        
        # Structured data
        connections_data = result["content"][1]["json"]
        assert connections_data["total_connections"] == 0
        assert connections_data["connections"] == []

    def test_invalid_action(self, agent):
        """Test calling with an invalid action."""
        result = agent.tool.mcp_client(action="invalid_action", connection_id="test")

        assert result["status"] == "error"
        assert "Unknown action: invalid_action" in result["content"][0]["text"]
        assert "Available actions:" in result["content"][0]["text"]


class TestStdioMCPServerIntegration:
    """Test integration with stdio-based MCP servers using MockMCP."""
    
    def test_stdio_server_basic_connection(self, agent, stdio_mcp_server):
        """Test basic connection to a stdio MCP server."""
        result = agent.tool.mcp_client(
            action="connect",
            connection_id="stdio_test_server",
            transport="stdio", 
            command="python",
            args=[stdio_mcp_server]
        )
        
        assert result["status"] == "success"
        # Access structured data from new ToolResult format
        connection_data = result["content"][1]["json"]
        assert connection_data["connection_id"] == "stdio_test_server"
        assert connection_data["transport"] == "stdio"
        assert connection_data["tools_count"] > 0
        assert "echo_tool" in connection_data["available_tools"]
        
        # Cleanup
        agent.tool.mcp_client(action="disconnect", connection_id="stdio_test_server")
    
    def test_stdio_server_list_tools(self, agent, stdio_mcp_server):
        """Test listing tools from a stdio MCP server."""
        # Connect first
        agent.tool.mcp_client(
            action="connect",
            connection_id="stdio_list_test",
            transport="stdio",
            command="python", 
            args=[stdio_mcp_server]
        )
        
        # List tools
        result = agent.tool.mcp_client(
            action="list_tools",
            connection_id="stdio_list_test"
        )
        
        assert result["status"] == "success"
        # Access structured data from new ToolResult format
        tools_data = result["content"][1]["json"]
        assert tools_data["tools_count"] >= 2  # echo_tool and add_numbers
        assert len(tools_data["tools"]) >= 2
        
        # Check that we have our expected tools
        tool_names = [tool["name"] for tool in tools_data["tools"]]
        assert "echo_tool" in tool_names
        assert "add_numbers" in tool_names
        
        # Check tool structure
        echo_tool = next(tool for tool in tools_data["tools"] if tool["name"] == "echo_tool")
        assert "description" in echo_tool
        assert "input_schema" in echo_tool
        # FastMCP tools have a nested json structure in their input schema
        assert "json" in echo_tool["input_schema"]
        assert echo_tool["input_schema"]["json"]["type"] == "object"
        
        # Cleanup
        agent.tool.mcp_client(action="disconnect", connection_id="stdio_list_test")
    
    def test_stdio_server_call_tool(self, agent, stdio_mcp_server):
        """Test calling a tool on a stdio MCP server."""
        # Connect first
        agent.tool.mcp_client(
            action="connect",
            connection_id="stdio_call_test",
            transport="stdio",
            command="python",
            args=[stdio_mcp_server]
        )
        
        # Call echo_tool
        result = agent.tool.mcp_client(
            action="call_tool",
            connection_id="stdio_call_test",
            tool_name="echo_tool",
            tool_args={"message": "integration test"}
        )
        
        assert result["status"] == "success"
        # With the simplified implementation, result is now returned directly from SDK
        assert result["toolUseId"] is not None
        
        # Check that the echo worked
        content = result["content"]
        assert len(content) > 0
        assert "Echo: integration test" in content[0]["text"]
        
        # Test add_numbers tool
        result = agent.tool.mcp_client(
            action="call_tool",
            connection_id="stdio_call_test",
            tool_name="add_numbers",
            tool_args={"a": 5, "b": 3}
        )
        
        assert result["status"] == "success"
        # With the simplified implementation, result is now returned directly from SDK
        assert result["toolUseId"] is not None
        
        # Cleanup
        agent.tool.mcp_client(action="disconnect", connection_id="stdio_call_test")
    
    def test_stdio_server_with_environment_variables(self, agent, stdio_mcp_server):
        """Test stdio server with custom environment variables."""
        result = agent.tool.mcp_client(
            action="connect",
            connection_id="stdio_env_test",
            transport="stdio",
            command="python",
            args=[stdio_mcp_server],
            env={"TEST_VAR": "integration_test_value", "DEBUG": "true"}
        )
        
        assert result["status"] == "success"
        # Access structured data from new ToolResult format
        connection_data = result["content"][1]["json"]
        assert connection_data["connection_id"] == "stdio_env_test"
        
        # Cleanup
        agent.tool.mcp_client(action="disconnect", connection_id="stdio_env_test")


class TestSSEMCPServerIntegration:
    """Test integration with SSE-based MCP servers using MockMCP."""
    
    def test_sse_server_basic_connection(self, agent, sse_mcp_server):
        """Test basic connection to an SSE MCP server."""
        result = agent.tool.mcp_client(
            action="connect",
            connection_id="sse_test_server",
            transport="sse",
            server_url=sse_mcp_server
        )
        
        assert result["status"] == "success"
        # Access structured data from new ToolResult format
        connection_data = result["content"][1]["json"]
        assert connection_data["connection_id"] == "sse_test_server"
        assert connection_data["transport"] == "sse"
        assert connection_data["tools_count"] > 0
        assert "echo_tool" in connection_data["available_tools"]
        
        # Cleanup
        agent.tool.mcp_client(action="disconnect", connection_id="sse_test_server")
    
    def test_sse_server_reliability(self, agent, sse_mcp_server):
        """Test SSE connection stability and basic operations."""
        # Connect
        connect_result = agent.tool.mcp_client(
            action="connect",
            connection_id="sse_reliability_test",
            transport="sse",
            server_url=sse_mcp_server
        )
        assert connect_result["status"] == "success"
        
        # Multiple operations to test stability
        for i in range(3):
            list_result = agent.tool.mcp_client(
                action="list_tools",
                connection_id="sse_reliability_test"
            )
            assert list_result["status"] == "success"
            
            call_result = agent.tool.mcp_client(
                action="call_tool",
                connection_id="sse_reliability_test", 
                tool_name="echo_tool",
                tool_args={"message": f"test_{i}"}
            )
            assert call_result["status"] == "success"
        
        # Cleanup
        agent.tool.mcp_client(action="disconnect", connection_id="sse_reliability_test")


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows with real MCP servers."""
    
    def test_full_workflow_stdio_server(self, agent, stdio_mcp_server):
        """Test complete workflow: connect → list tools → call tool → load tools → disconnect."""
        connection_id = "full_workflow_test"
        
        # 1. Connect
        connect_result = agent.tool.mcp_client(
            action="connect",
            connection_id=connection_id,
            transport="stdio",
            command="python",
            args=[stdio_mcp_server]
        )
        assert connect_result["status"] == "success"
        
        # 2. List connections
        list_conn_result = agent.tool.mcp_client(action="list_connections")
        assert list_conn_result["status"] == "success"
        # Access structured data from new ToolResult format
        connections_data = list_conn_result["content"][1]["json"]
        assert connections_data["total_connections"] >= 1
        
        # Find our connection
        our_conn = next(
            (c for c in connections_data["connections"] if c["connection_id"] == connection_id),
            None
        )
        assert our_conn is not None
        assert our_conn["is_active"] is True
        
        # 3. List tools
        list_tools_result = agent.tool.mcp_client(
            action="list_tools",
            connection_id=connection_id
        )
        assert list_tools_result["status"] == "success"
        # Access structured data from new ToolResult format
        tools_data = list_tools_result["content"][1]["json"]
        assert tools_data["tools_count"] >= 2
        
        # 4. Call a tool
        call_result = agent.tool.mcp_client(
            action="call_tool",
            connection_id=connection_id,
            tool_name="echo_tool",
            tool_args={"message": "full workflow test"}
        )
        assert call_result["status"] == "success"
        
        # 5. Load tools into agent
        load_result = agent.tool.mcp_client(
            action="load_tools",
            connection_id=connection_id,
            agent=agent
        )
        assert load_result["status"] == "success"
        # Access structured data from new ToolResult format
        load_data = load_result["content"][1]["json"]
        assert load_data["tool_count"] >= 2
        assert "echo_tool" in load_data["loaded_tools"]
        
        # 6. Disconnect
        disconnect_result = agent.tool.mcp_client(
            action="disconnect",
            connection_id=connection_id,
            agent=agent  # Pass agent for tool cleanup
        )
        assert disconnect_result["status"] == "success"
        
        # 7. Verify connection is gone
        final_list = agent.tool.mcp_client(action="list_connections")
        final_connections_data = final_list["content"][1]["json"]
        remaining_connections = [
            c for c in final_connections_data["connections"] 
            if c["connection_id"] == connection_id
        ]
        assert len(remaining_connections) == 0
    
    def test_tool_loading_and_execution_through_agent(self, agent, stdio_mcp_server):
        """Test loading MCP tools into agent and executing them via agent calls."""
        connection_id = "tool_loading_test"
        
        # Connect and load tools
        agent.tool.mcp_client(
            action="connect",
            connection_id=connection_id,
            transport="stdio",
            command="python",
            args=[stdio_mcp_server]
        )
        
        load_result = agent.tool.mcp_client(
            action="load_tools",
            connection_id=connection_id,
            agent=agent
        )
        assert load_result["status"] == "success"
        # Access structured data from new ToolResult format
        load_data = load_result["content"][1]["json"]
        assert load_data["tool_count"] >= 2
        
        # The tools should now be in the agent's registry
        tool_names = load_data["loaded_tools"]
        assert len(tool_names) >= 2
        assert "echo_tool" in tool_names
        assert "add_numbers" in tool_names
        
        # 🚀 NEW: Actually test calling the loaded tools through the agent interface
        try:
            # Test calling echo_tool directly through the agent
            echo_result = agent.tool.echo_tool(message="test from agent interface")
            assert echo_result["status"] == "success"
            assert "Echo: test from agent interface" in echo_result["content"][0]["text"]
            
            # Test calling add_numbers directly through the agent
            add_result = agent.tool.add_numbers(a=7, b=13)
            
            # The result should be in proper Strands format 
            assert add_result["status"] == "success"
            assert add_result["content"][0]["text"] == "20"  # 7 + 13 = 20
            
        except AttributeError as e:
            # This would indicate the tools weren't properly loaded into the agent
            assert False, f"Loaded tools not accessible through agent.tool interface: {e}"
        
        # Cleanup
        agent.tool.mcp_client(
            action="disconnect",
            connection_id=connection_id,
            agent=agent
        )


class TestErrorHandlingAndReliability:
    """Test error handling and reliability scenarios."""
    
    def test_connection_to_nonexistent_server(self, agent):
        """Test connecting to a non-existent server."""
        result = agent.tool.mcp_client(
            action="connect",
            connection_id="nonexistent_test",
            transport="stdio",
            command="nonexistent_command",
            args=["fake_server.py"]
        )
        
        assert result["status"] == "error"
        assert "Connection failed" in result["content"][0]["text"]
    
    def test_invalid_server_url_http(self, agent):
        """Test connecting to invalid HTTP server URL."""
        result = agent.tool.mcp_client(
            action="connect",
            connection_id="invalid_http_test",
            transport="streamable_http",
            server_url="http://localhost:99999/invalid"
        )
        
        assert result["status"] == "error"
        assert "Connection failed" in result["content"][0]["text"]
    
    def test_invalid_server_url_sse(self, agent):
        """Test connecting to invalid SSE server URL."""
        result = agent.tool.mcp_client(
            action="connect",
            connection_id="invalid_sse_test", 
            transport="sse",
            server_url="http://localhost:99999/invalid"
        )
        
        assert result["status"] == "error"
        assert "Connection failed" in result["content"][0]["text"]
    
    def test_missing_required_parameters(self, agent):
        """Test error handling for missing required parameters."""
        # Missing connection_id for connect
        result = agent.tool.mcp_client(
            action="connect",
            transport="stdio",
            command="python"
        )
        assert result["status"] == "error"
        assert "connection_id is required" in result["content"][0]["text"]
        
        # Missing server_url for HTTP
        result = agent.tool.mcp_client(
            action="connect",
            connection_id="test",
            transport="streamable_http"
        )
        assert result["status"] == "error"
        assert "server_url is required" in result["content"][0]["text"]
        
        # Missing tool_name for call_tool
        result = agent.tool.mcp_client(
            action="call_tool",
            connection_id="test"
        )
        assert result["status"] == "error"
        assert "tool_name is required" in result["content"][0]["text"]

    def test_operations_on_nonexistent_connections(self, agent):
        """Test operations on connections that don't exist."""
        # Try to list tools from non-existent connection
        result = agent.tool.mcp_client(
            action="list_tools",
            connection_id="nonexistent_connection"
        )
        assert result["status"] == "error"
        assert "Connection 'nonexistent_connection' not found" in result["content"][0]["text"]
        
        # Try to call tool on non-existent connection
        result = agent.tool.mcp_client(
            action="call_tool",
            connection_id="nonexistent_connection",
            tool_name="test_tool"
        )
        assert result["status"] == "error"
        assert "Connection 'nonexistent_connection' not found" in result["content"][0]["text"]
        
        # Try to disconnect from non-existent connection
        result = agent.tool.mcp_client(
            action="disconnect",
            connection_id="nonexistent_connection"
        )
        assert result["status"] == "error"
        assert "Connection 'nonexistent_connection' not found" in result["content"][0]["text"]

    def test_load_tools_without_agent(self, agent):
        """Test load_tools action without providing agent instance."""
        result = agent.tool.mcp_client(
            action="load_tools",
            connection_id="test_connection"
            # Note: agent parameter is not provided
        )
        assert result["status"] == "error"
        assert "agent instance is required" in result["content"][0]["text"]


class TestConfigurationAndParameterHandling:
    """Test configuration and parameter handling."""
    
    def test_transport_parameter_validation(self, agent):
        """Test validation of transport parameters."""
        # Test unsupported transport (should fail during connection)
        result = agent.tool.mcp_client(
            action="connect",
            connection_id="test",
            transport="unsupported_transport"
        )
        assert result["status"] == "error"
        assert "Connection failed" in result["content"][0]["text"]
    
    def test_server_config_vs_direct_parameters(self, agent):
        """Test that direct parameters override server_config."""
        # This should fail because command is missing, but it tests parameter precedence
        result = agent.tool.mcp_client(
            action="connect",
            connection_id="test",
            server_config={"transport": "sse", "server_url": "http://example.com"},
            transport="stdio"  # This should override the server_config transport
        )
        assert result["status"] == "error"
        # Should complain about missing command (stdio transport) not server_url (sse transport)
        assert "command is required for stdio transport" in result["content"][0]["text"]