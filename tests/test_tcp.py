"""
Tests for the TCP tool using the Agent interface.
"""

import socket
import time

import pytest
from strands import Agent
from strands_tools import tcp


@pytest.fixture
def agent():
    """Create an agent with the tcp tool loaded."""
    return Agent(tools=[tcp])


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def get_free_port():
    """Get a free port to use for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestTcpTool:
    """Tests for the TCP tool."""

    def test_start_stop_server(self, agent):
        """Test starting and stopping a TCP server."""
        port = get_free_port()

        # Start server
        start_result = agent.tool.tcp(
            action="start_server", host="127.0.0.1", port=port, system_prompt="Test TCP Server"
        )

        assert "TCP Server started successfully" in extract_result_text(start_result)

        # Check server status
        status_result = agent.tool.tcp(action="get_status")
        status_text = "\n".join([item["text"] for item in status_result["content"]])
        assert f"Port {port}: Running" in status_text

        # Stop server
        stop_result = agent.tool.tcp(action="stop_server", port=port)

        assert "TCP Server on port" in extract_result_text(stop_result)
        assert "stopped successfully" in extract_result_text(stop_result)

        # Verify server is stopped
        status_result = agent.tool.tcp(action="get_status")
        status_text = "\n".join([item["text"] for item in status_result["content"]])
        assert f"Port {port}" not in status_text or f"Port {port}: Stopped" in status_text

    def test_server_client_communication(self, agent):
        """Test communication between server and client."""
        port = get_free_port()

        # Start server
        agent.tool.tcp(
            action="start_server",
            host="127.0.0.1",
            port=port,
            system_prompt="You are a test server. Always respond with 'TEST RESPONSE'.",
        )

        # Allow server time to start
        time.sleep(0.5)

        # Send message as client
        client_result = agent.tool.tcp(action="client_send", host="127.0.0.1", port=port, message="Hello, server!")

        # Check client received response
        result_text = "\n".join([item["text"] for item in client_result["content"]])
        assert "Connected to 127.0.0.1" in result_text
        assert "Sent message: Hello, server!" in result_text
        assert "Response received:" in result_text

        # Stop server
        agent.tool.tcp(action="stop_server", port=port)

    def test_start_server_twice_error(self, agent):
        """Test that trying to start a server on the same port twice gives an error."""
        port = get_free_port()

        # Start server
        agent.tool.tcp(action="start_server", host="127.0.0.1", port=port)

        # Try to start again on same port
        result = agent.tool.tcp(action="start_server", host="127.0.0.1", port=port)

        assert result["status"] == "error"
        assert "already running" in result["content"][0]["text"]

        # Stop server
        agent.tool.tcp(action="stop_server", port=port)

    def test_stop_nonexistent_server(self, agent):
        """Test stopping a server that doesn't exist."""
        # Use a likely unused port
        port = 65432

        result = agent.tool.tcp(action="stop_server", port=port)

        assert result["status"] == "error"
        assert "No TCP Server running" in result["content"][0]["text"]

    def test_client_nonexistent_server(self, agent):
        """Test connecting to a server that doesn't exist."""
        # Use a likely unused port
        port = 65433

        result = agent.tool.tcp(action="client_send", host="127.0.0.1", port=port, message="Hello?")

        assert result["status"] == "error"
        assert "Connection" in result["content"][0]["text"]
        assert "refused" in result["content"][0]["text"]

    def test_direct_invocation(self):
        """Test direct invocation of the TCP tool."""
        # Create a tool use dictionary similar to how the agent would call it
        tool_use = {"toolUseId": "test-tool-use-id", "input": {"action": "get_status"}}

        # Call the tcp function directly
        result = tcp.tcp(tool=tool_use)

        # Verify the result has the expected structure
        assert result["toolUseId"] == "test-tool-use-id"
        assert result["status"] == "success"

    def test_invalid_action(self, agent):
        """Test providing an invalid action."""
        result = agent.tool.tcp(action="invalid_action")

        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    def test_no_message_for_client_send(self, agent):
        """Test client_send without a message."""
        result = agent.tool.tcp(action="client_send", host="127.0.0.1", port=12345)

        assert result["status"] == "error"
        assert "No message provided" in result["content"][0]["text"]
