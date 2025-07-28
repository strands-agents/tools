"""Integration tests for HTTP request functionality using Strands Agent.

This module contains integration tests that verify the HTTP request tool's ability
to handle various HTTP methods (GET, POST, PUT) and operations like downloading
files and interacting with AWS S3 services.
"""

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from strands import Agent
from strands_tools import http_request

os.environ["BYPASS_TOOL_CONSENT"] = "true"


@pytest.fixture
def agent():
    """Create an Agent instance configured with HTTP request tool."""
    return Agent(tools=[http_request])


@pytest.fixture(scope="module")
def http_server():
    """Create a local HTTP server for testing."""
    host = "localhost"
    port = _find_free_port()
    server = HTTPServer((host, port), LocalHttpRequestHandler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    if not _wait_for_server(host, port):
        server.shutdown()
        pytest.fail(f"Server failed to start on {host}:{port}")

    server.test_port = port
    yield server

    server.shutdown()
    server.server_close()


def test_get_http_request(agent, http_server):
    """Test GET request functionality using local test server."""
    port = http_server.test_port
    response = agent(f"Send a GET request to http://localhost:{port}/repo and show me the number of stars")
    assert "stars" in str(response).lower() and ("99999" in str(response).lower() or "99,999" in str(response).lower())


def test_post_http_request(agent, http_server):
    """Test POST request functionality using local test server."""
    port = http_server.test_port
    response = agent(f'Send a POST request to http://localhost:{port} with JSON body {{"foo": "bar"}}')
    assert "foo" in str(response).lower() and "bar" in str(response).lower()


def test_update_http_request(agent, http_server):
    """Test PUT request functionality using local test server."""
    port = http_server.test_port
    response = agent(f'Send a PUT request to http://localhost:{port} with JSON body {{"update": true}}')
    assert "update" in str(response).lower() and "true" in str(response).lower()


def test_download_http_request(agent):
    """Verify that the agent can use tool to download a file from a URL"""
    response = agent("Download the PNG logo from https://www.python.org/static/community_logos/python-logo.png")
    assert "successfully" in str(response).lower() or "image" in str(response).lower(), str(response)


@pytest.mark.skipif("AWS_ACCESS_KEY_ID" not in os.environ, reason="ACCESS_KEY_ID environment variable missing")
def test_list_s3_bucket_http_request(agent):
    """Test AWS S3 bucket listing functionality via HTTP requests."""
    region = os.getenv("AWS_REGION", "us-east-1")
    response = agent(f"List all S3 buckets in region {region}.")
    assert "s3 buckets" in str(response).lower(), str(response)


# --Helper function & Class--


class LocalHttpRequestHandler(BaseHTTPRequestHandler):
    """A simplified HTTP request handler for local testing."""

    def _send_json_response(self, status_code: int, payload: dict) -> None:
        """Send a JSON response with the given status and payload."""
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/repo":
            response_data = {"name": "sdk-python", "stars": 99999}
            self._send_json_response(200, response_data)
        else:
            self._send_json_response(404, {"error": "Not Found"})

    def _handle_request_with_body(self, success_key: str) -> None:
        """Handle requests with a JSON body (for POST, PUT, etc.)."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}
            response_payload = {success_key: data, "status": "success"}
            self._send_json_response(200, response_payload)
        except json.JSONDecodeError:
            self._send_json_response(400, {"error": "Invalid JSON"})
        except Exception as e:
            self._send_json_response(500, {"error": str(e)})

    def do_POST(self) -> None:
        """Handle POST requests by reusing the body handler."""
        self._handle_request_with_body("received_data")

    def do_PUT(self) -> None:
        """Handle PUT requests by reusing the body handler."""
        self._handle_request_with_body("updated_data")


def _find_free_port() -> int:
    """Find and return an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout: int = 5) -> bool:
    """Wait for the local test server to become available."""
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return True
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(0.1)
    return False
