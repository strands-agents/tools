"""Test helpers for MCP client integration tests."""

import ast
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

from .mock_mcp_server import MockMCPServer


def find_free_port() -> int:
    """Find and return an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_server(host: str, port: int, timeout: int = 10) -> bool:
    """Wait for a server to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.1)
    return False


class StdioMCPServerManager:
    """Manager for stdio-based MCP server processes."""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.script_path = Path(__file__).parent / "mock_mcp_server.py"
        
    def start(self) -> str:
        """Start the stdio MCP server and return the script path."""
        # The script path is what we need for stdio connections
        return str(self.script_path.absolute())
        
    def stop(self):
        """Stop the stdio MCP server."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None


class SSEMCPServerManager:
    """Manager for SSE-based MCP server processes."""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.script_path = Path(__file__).parent / "mock_mcp_server.py"
        
    def start(self) -> str:
        """Start the SSE MCP server and return the server URL."""
        self.port = find_free_port()
        
        # Start the server process
        self.process = subprocess.Popen([
            sys.executable, 
            str(self.script_path), 
            "sse", 
            str(self.port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to be ready
        if not wait_for_server("localhost", self.port, timeout=10):
            self.stop()
            raise RuntimeError(f"SSE MCP server failed to start on port {self.port}")
            
        return f"http://localhost:{self.port}/sse"
        
    def stop(self):
        """Stop the SSE MCP server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None


@pytest.fixture(scope="session")
def stdio_mcp_server():
    """Fixture providing a stdio MCP server for testing."""
    manager = StdioMCPServerManager()
    server_script = manager.start()
    yield server_script
    manager.stop()


@pytest.fixture(scope="session")
def sse_mcp_server():
    """Fixture providing an SSE MCP server for testing.""" 
    manager = SSEMCPServerManager()
    try:
        server_url = manager.start()
        yield server_url
    finally:
        manager.stop()


def parse_tool_result(result):
    """Parse the tool result from agent.tool calls that may return serialized data.
    
    Agent calls return: {'toolUseId': ..., 'status': 'success', 'content': [{'text': '...'}]}
    Some tools may serialize complex data in the content[0]['text'] field.
    
    - Text message in content[0]['text']
    - Structured data in content[1]['json']
    
    This helper is kept for backward compatibility with other tools that may still
    serialize their results, but should not be needed for properly implemented tools.
    
    Example:
    - mcp_client calls: Use results directly (no parsing needed)
    - Other tools that serialize: May still need parse_tool_result()
    - Loaded MCP tools: Use results directly (no parsing needed)
    """
    if result.get('status') != 'success':
        return result
    
    try:
        text = result['content'][0]['text']
        # Try JSON parsing first
        try:
            actual_result = json.loads(text)
            return actual_result
        except json.JSONDecodeError:
            # Try evaluating as Python literal (safe eval for dict/list/etc)
            actual_result = ast.literal_eval(text)
            return actual_result
    except (KeyError, IndexError, ValueError, SyntaxError):
        return result