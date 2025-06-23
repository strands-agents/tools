from unittest.mock import MagicMock

import pytest
from strands_tools import batch


@pytest.fixture
def mock_agent():
    """Fixture to create a mock agent with tools."""
    agent = MagicMock()
    agent.tool.http_request = MagicMock(return_value={"status": "success", "result": {"ip": "127.0.0.1"}})
    agent.tool.use_aws = MagicMock(return_value={"status": "success", "result": {"buckets": ["bucket1", "bucket2"]}})
    agent.tool.error_tool = MagicMock(side_effect=Exception("Tool execution failed"))
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
    assert result["content"][0]["json"]["name"] == "http_request"
    assert result["content"][0]["json"]["status"] == "success"
    assert result["content"][0]["json"]["result"]["result"]["ip"] == "127.0.0.1"
    assert result["content"][1]["json"]["name"] == "use_aws"
    assert result["content"][1]["json"]["status"] == "success"
    assert result["content"][1]["json"]["result"]["result"]["buckets"] == ["bucket1", "bucket2"]


def test_batch_missing_tool(mock_agent):
    """Test behavior when a tool is not found."""
    mock_tool = {"toolUseId": "mock_tool_id"}
    invocations = [
        {"name": "non_existent_tool", "arguments": {}},
    ]

    result = batch.batch(tool=mock_tool, agent=mock_agent, invocations=invocations)

    assert result["toolUseId"] == "mock_tool_id"
    assert result["status"] == "success"
    assert len(result["content"]) == 1
    assert result["content"][0]["toolUseId"] == "mock_tool_id"
    assert result["content"][0]["status"] == "error"
    assert "Tool missing" in result["content"][0]["content"][0]["text"]


def test_batch_tool_error(mock_agent):
    """Test behavior when a tool raises an exception."""
    mock_tool = {"toolUseId": "mock_tool_id"}
    invocations = [
        {"name": "error_tool", "arguments": {}},
    ]

    result = batch.batch(tool=mock_tool, agent=mock_agent, invocations=invocations)

    assert result["toolUseId"] == "mock_tool_id"
    assert result["status"] == "success"
    assert len(result["content"]) == 1
    assert result["content"][0]["toolUseId"] == "mock_tool_id"
    assert result["content"][0]["status"] == "error"
    assert "Error in batch tool" in result["content"][0]["content"][0]["text"]


def test_batch_no_invocations(mock_agent):
    """Test behavior when no invocations are provided."""
    mock_tool = {"toolUseId": "mock_tool_id"}
    invocations = []

    result = batch.batch(tool=mock_tool, agent=mock_agent, invocations=invocations)

    assert result["toolUseId"] == "mock_tool_id"
    assert result["status"] == "success"
    assert len(result["content"]) == 0


def test_batch_top_level_error(mock_agent):
    """Test behavior when a top-level exception occurs."""
    mock_tool = {"toolUseId": "mock_tool_id"}

    # Simulate an error in the agent
    mock_agent.tool = None  # This will cause an AttributeError when accessing tools

    result = batch.batch(tool=mock_tool, agent=mock_agent, invocations=[])

    assert result["toolUseId"] == "mock_tool_id"
    assert result["status"] == "error"  # Expect 'error' status
    assert "Agent does not have a valid 'tool' attribute." in result["content"][0]["text"]
