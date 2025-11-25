from unittest.mock import MagicMock

import pytest

from strands_tools import batch


@pytest.fixture
def mock_agent():
    """Fixture to create a mock agent with tools."""
    agent = MagicMock()

    # Create a mock tool registry that mimics the real agent's tool access pattern
    mock_tool_registry = MagicMock()
    mock_tool_registry.registry = {
        "http_request": MagicMock(return_value={"status": "success", "result": {"ip": "127.0.0.1"}}),
        "use_aws": MagicMock(return_value={"status": "success", "result": {"buckets": ["bucket1", "bucket2"]}}),
        "error_tool": MagicMock(side_effect=Exception("Tool execution failed")),
    }
    agent.tool_registry = mock_tool_registry

    # Create a custom mock tool object that properly handles getattr
    class MockTool:
        def __init__(self):
            self.http_request = mock_tool_registry.registry["http_request"]
            self.use_aws = mock_tool_registry.registry["use_aws"]
            self.error_tool = mock_tool_registry.registry["error_tool"]

        def __getattr__(self, name):
            # Return None for non-existent tools (this will make callable() return False)
            return None

    agent.tool = MockTool()

    return agent


def test_batch_success(mock_agent):
    """Test successful execution of multiple tools."""
    mock_tool = {"toolUseId": "mock_tool_id"}
    invocations = [
        {"name": "http_request", "arguments": {"method": "GET", "url": "https://api.ipify.org?format=json"}},
        {"name": "use_aws", "arguments": {"service_name": "s3", "operation_name": "list_buckets"}},
    ]

    result = batch.batch(tool=mock_tool, agent=mock_agent, invocations=invocations)

    assert result["toolUseId"] == "mock_tool_id"
    assert result["status"] == "success"
    assert len(result["content"]) == 2

    # Check the summary text
    assert "Batch execution completed with 2 tool(s):" in result["content"][0]["text"]
    assert "✓ http_request: Success" in result["content"][0]["text"]
    assert "✓ use_aws: Success" in result["content"][0]["text"]

    # Check the JSON results
    json_content = result["content"][1]["json"]
    assert json_content["batch_summary"]["total_tools"] == 2
    assert json_content["batch_summary"]["successful"] == 2
    assert json_content["batch_summary"]["failed"] == 0

    results = json_content["results"]
    assert len(results) == 2
    assert results[0]["name"] == "http_request"
    assert results[0]["status"] == "success"
    assert results[0]["result"]["result"]["ip"] == "127.0.0.1"
    assert results[1]["name"] == "use_aws"
    assert results[1]["status"] == "success"
    assert results[1]["result"]["result"]["buckets"] == ["bucket1", "bucket2"]


def test_batch_missing_tool(mock_agent):
    """Test behavior when a tool is not found."""
    mock_tool = {"toolUseId": "mock_tool_id"}
    invocations = [
        {"name": "non_existent_tool", "arguments": {}},
    ]

    result = batch.batch(tool=mock_tool, agent=mock_agent, invocations=invocations)

    assert result["toolUseId"] == "mock_tool_id"
    assert result["status"] == "success"
    assert len(result["content"]) == 2

    # Check the summary text
    assert "Batch execution completed with 1 tool(s):" in result["content"][0]["text"]
    assert "✗ non_existent_tool: Error" in result["content"][0]["text"]

    # Check the JSON results
    json_content = result["content"][1]["json"]
    assert json_content["batch_summary"]["total_tools"] == 1
    assert json_content["batch_summary"]["successful"] == 0
    assert json_content["batch_summary"]["failed"] == 1

    results = json_content["results"]
    assert len(results) == 1
    assert results[0]["name"] == "non_existent_tool"
    assert results[0]["status"] == "error"
    assert "not found in agent" in results[0]["error"]


def test_batch_tool_error(mock_agent):
    """Test behavior when a tool raises an exception."""
    mock_tool = {"toolUseId": "mock_tool_id"}
    invocations = [
        {"name": "error_tool", "arguments": {}},
    ]

    result = batch.batch(tool=mock_tool, agent=mock_agent, invocations=invocations)

    assert result["toolUseId"] == "mock_tool_id"
    assert result["status"] == "success"
    assert len(result["content"]) == 2

    # Check the summary text
    assert "Batch execution completed with 1 tool(s):" in result["content"][0]["text"]
    assert "✗ error_tool: Error" in result["content"][0]["text"]

    # Check the JSON results
    json_content = result["content"][1]["json"]
    assert json_content["batch_summary"]["total_tools"] == 1
    assert json_content["batch_summary"]["successful"] == 0
    assert json_content["batch_summary"]["failed"] == 1

    results = json_content["results"]
    assert len(results) == 1
    assert results[0]["name"] == "error_tool"
    assert results[0]["status"] == "error"
    assert "Tool execution failed" in results[0]["error"]
    assert "traceback" in results[0]


def test_batch_no_invocations(mock_agent):
    """Test behavior when no invocations are provided."""
    mock_tool = {"toolUseId": "mock_tool_id"}
    invocations = []

    result = batch.batch(tool=mock_tool, agent=mock_agent, invocations=invocations)

    assert result["toolUseId"] == "mock_tool_id"
    assert result["status"] == "success"
    assert len(result["content"]) == 2

    # Check the summary text
    assert "Batch execution completed with 0 tool(s):" in result["content"][0]["text"]

    # Check the JSON results
    json_content = result["content"][1]["json"]
    assert json_content["batch_summary"]["total_tools"] == 0
    assert json_content["batch_summary"]["successful"] == 0
    assert json_content["batch_summary"]["failed"] == 0
    assert len(json_content["results"]) == 0


def test_batch_top_level_error(mock_agent):
    """Test behavior when a top-level exception occurs."""
    mock_tool = {"toolUseId": "mock_tool_id"}

    # Simulate an error in the agent
    mock_agent.tool = None  # This will cause an AttributeError when accessing tools

    result = batch.batch(tool=mock_tool, agent=mock_agent, invocations=[])

    assert result["toolUseId"] == "mock_tool_id"
    assert result["status"] == "error"  # Expect 'error' status
    assert "Agent does not have a valid 'tool' attribute." in result["content"][0]["text"]
