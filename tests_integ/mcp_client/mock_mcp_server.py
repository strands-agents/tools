"""Mock MCP Server for integration testing using FastMCP.

This server provides a simple echo tool for testing the dynamic MCP client.
It supports both stdio and SSE transports.
"""

import asyncio
import logging
import os
import sys
import threading
import time
from typing import Any, Dict

from mcp.server import FastMCP
from mcp.types import TextContent

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# Create FastMCP server
app = FastMCP("MockMCPServer")


@app.tool()
def echo_tool(message: str) -> str:
    """Echo the provided message back to the caller.
    
    Args:
        message: The message to echo back
        
    Returns:
        The echoed message with a prefix
    """
    return f"Echo: {message}"


@app.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The sum of a and b
    """
    return a + b


class MockMCPServer:
    """Mock MCP Server that can run in stdio or SSE mode."""
    
    def __init__(self):
        self.server_task = None
        self.server_thread = None
        
    async def run_stdio_async(self):
        """Run the server in stdio mode."""
        logger.info("Starting MockMCP server in stdio mode")
        await app.run_stdio_async()
        
    async def run_sse_async(self, host: str = "localhost", port: int = 8000):
        """Run the server in SSE mode using uvicorn."""
        logger.info(f"Starting MockMCP server in SSE mode on {host}:{port}")
        
        # Get the SSE app from FastMCP
        sse_app = app.sse_app()
        
        # Use uvicorn to run the SSE app
        import uvicorn
        config = uvicorn.Config(
            app=sse_app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    def run_stdio_sync(self):
        """Run the server in stdio mode synchronously."""
        try:
            asyncio.run(self.run_stdio_async())
        except KeyboardInterrupt:
            logger.info("MockMCP stdio server stopped")
        except Exception as e:
            logger.error(f"MockMCP stdio server error: {e}")
            
    def run_sse_sync(self, host: str = "localhost", port: int = 8000):
        """Run the server in SSE mode synchronously."""
        try:
            asyncio.run(self.run_sse_async(host=host, port=port))
        except KeyboardInterrupt:
            logger.info("MockMCP SSE server stopped")
        except Exception as e:
            logger.error(f"MockMCP SSE server error: {e}")
            
    def start_stdio_in_thread(self):
        """Start the stdio server in a separate thread."""
        self.server_thread = threading.Thread(target=self.run_stdio_sync, daemon=True)
        self.server_thread.start()
        return self.server_thread
        
    def start_sse_in_thread(self, host: str = "localhost", port: int = 8000):
        """Start the SSE server in a separate thread."""
        self.server_thread = threading.Thread(
            target=self.run_sse_sync, 
            args=(host, port), 
            daemon=True
        )
        self.server_thread.start()
        return self.server_thread


def main():
    """Main entry point for running the server."""
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        # SSE mode with optional port
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        server = MockMCPServer()
        server.run_sse_sync(port=port)
    else:
        # Default to stdio mode
        server = MockMCPServer()
        server.run_stdio_sync()


if __name__ == "__main__":
    main()