from __future__ import annotations

import http.server
import json
import os
import socketserver
import threading
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent
from strands_tools import (
    environment,
    file_read,
    file_write,
    http_request,
    python_repl,
    use_llm,
)


# To bypass the interactive confirmation prompt for this test.
@patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"})
def test_file_write_then_read(tmp_path):
    """
    Write a file with file_write and immediately read it back with file_read.
    """
    file_path = tmp_path / "hello.txt"
    content = "Hello Integration!"

    write_tool_use = {"toolUseId": "write_file_123", "input": {"path": str(file_path), "content": content}}
    write_res = file_write.file_write(tool=write_tool_use)
    assert write_res["status"] == "success"

    read_tool_use = {"toolUseId": "read_file_123", "input": {"path": str(file_path), "mode": "view"}}
    read_res = file_read.file_read(tool=read_tool_use)
    assert read_res["status"] == "success"
    assert content in read_res["content"][0]["text"]


@patch("strands_tools.utils.user_input.get_user_input", return_value="y")
def test_environment_set_then_get(mock_get_user_input):
    """
    Round-trip an environment variable using the environment tool.

    This test uses a real Agent instance and mocks the user input
    to bypass the interactive confirmation prompt.
    """
    agent = Agent(tools=[environment])

    var_name, value = "STRANDS_INTEGRATION_TEST_VAR", "42"

    # Set& Get an environment variable using the agent's tool interface
    set_res = agent.tool.environment(action="set", name=var_name, value=value)
    assert set_res["status"] == "success"

    get_res = agent.tool.environment(action="get", name=var_name)
    assert get_res["status"] == "success"

    assert value in str(get_res["content"][0]["text"])

    del os.environ[var_name]


def test_python_repl_state_persistence():
    """Verify that state created in one python_repl call is available in the next."""
    agent = Agent(tools=[python_repl])

    res1 = agent.tool.python_repl(code="x = 6 * 7", interactive=False, non_interactive_mode=True)
    assert res1["status"] == "success"

    res2 = agent.tool.python_repl(code="print(x)", interactive=False, non_interactive_mode=True)
    assert res2["status"] == "success"

    combined_output = json.dumps(res2["content"])
    assert "42" in combined_output


# This class is for test local http_server
class _SimpleJSONHandler(http.server.BaseHTTPRequestHandler):
    """Very small HTTP server that always returns a JSON payload."""

    def do_GET(self) -> None:  # noqa: N802  (method name dictated by BaseHTTPRequestHandler)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true, "msg": "hello"}')

    # Silence the default noisy logging
    def log_message(self, *_args, **_kwargs):  # noqa: D401
        pass


@pytest.fixture(scope="module")
def local_http_server() -> Iterator[str]:
    """
    Spin up a throw-away local HTTP server on an ephemeral port and
    return the base URL for requests.
    """
    with socketserver.TCPServer(("127.0.0.1", 0), _SimpleJSONHandler) as httpd:
        host, port = httpd.server_address
        url = f"http://{host}:{port}"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield url
        finally:
            httpd.shutdown()
            thread.join()


def test_http_request_against_local_server(local_http_server):
    """Real HTTP round-trip to a local server using http_request."""
    agent = Agent(tools=[http_request])

    res = agent.tool.http_request(method="GET", url=f"{local_http_server}/")
    assert res["status"] == "success"

    # The response body is part of the tool's content list; find it and check it.
    body_found = False
    for item in res["content"]:
        if item.get("text", "").lower().startswith("body:"):
            assert '"ok": true' in item["text"]
            body_found = True
            break
    assert body_found, "Response body not found in tool output"


@patch("strands.agent.agent.Agent.__call__")
def test_use_llm_with_mocked_model(mock_agent_call):
    """
    Exercise use_llm with a mocked model to avoid real network calls.
    """
    # The tool expects the agent to return an object with attributes like '.metrics',
    # not just a raw string. We create a MagicMock to simulate this object.
    mock_response = MagicMock()
    mock_response.metrics = {}  # Provide the 'metrics' attribute that the tool checks.
    mock_response.__str__.return_value = "Hello"

    mock_agent_call.return_value = mock_response

    agent = Agent(tools=[use_llm])
    res = agent.tool.use_llm(
        prompt="Reply with exactly the word 'Hello'.",
        system_prompt="You are a minimal echo-bot that always answers with 'Hello'.",
    )

    assert res["status"] == "success"
    assert "hello" in json.dumps(res["content"]).lower()
