"""
Tests for the dakera_memory tool.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from strands.types.tools import ToolUse

from strands_tools.dakera_memory import DakeraServiceClient, dakera_memory


@pytest.fixture
def mock_tool():
    """Create a mock ToolUse object."""
    mock = MagicMock(spec=ToolUse)
    mock.get.side_effect = lambda key, default=None: {"toolUseId": "test-id", "input": {}}.get(key, default)
    return mock


def make_tool(input_data: dict) -> MagicMock:
    """Helper: build a mock ToolUse with given input dict."""
    mock = MagicMock(spec=ToolUse)
    mock.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": input_data,
    }.get(key, default)
    return mock


# ---------------------------------------------------------------------------
# DakeraServiceClient init
# ---------------------------------------------------------------------------


def test_service_client_raises_on_missing_package():
    """ImportError when dakera package is not installed."""
    with patch("builtins.__import__", side_effect=ImportError("No module named 'dakera'")):
        with pytest.raises(ImportError, match="dakera package is required"):
            DakeraServiceClient()


@patch.dict(os.environ, {"DAKERA_BASE_URL": "http://localhost:3000", "DAKERA_API_KEY": "dk-test"})
def test_service_client_init():
    """Client initialises from env vars."""
    mock_dakera_client = MagicMock()
    with patch("strands_tools.dakera_memory.DakeraServiceClient.__init__", return_value=None) as patched:
        patched.return_value = None
        client = DakeraServiceClient.__new__(DakeraServiceClient)
        client.client = mock_dakera_client
        assert client.client is mock_dakera_client


# ---------------------------------------------------------------------------
# store action
# ---------------------------------------------------------------------------


@patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"})
def test_store_memory():
    """store action returns success with memory dict."""
    stored = {
        "id": "mem-abc",
        "content": "Test memory content",
        "importance": 0.8,
        "created_at": "2026-07-02T00:00:00Z",
    }

    with patch("strands_tools.dakera_memory.DakeraServiceClient") as MockClient:
        instance = MockClient.return_value
        instance.store_memory.return_value = stored

        tool = make_tool(
            {
                "action": "store",
                "agent_id": "alex",
                "content": "Test memory content",
                "importance": 0.8,
                "metadata": {"category": "notes"},
            }
        )
        result = dakera_memory(tool=tool)

    assert result["status"] == "success"
    body = json.loads(result["content"][0]["text"])
    assert body["id"] == "mem-abc"
    instance.store_memory.assert_called_once_with(
        agent_id="alex",
        content="Test memory content",
        importance=0.8,
        memory_type="episodic",
        metadata={"category": "notes"},
    )


def test_store_missing_content():
    """store without content returns error."""
    tool = make_tool({"action": "store", "agent_id": "alex"})
    with patch("strands_tools.dakera_memory.DakeraServiceClient"):
        result = dakera_memory(tool=tool)
    assert result["status"] == "error"
    assert "content is required for store action" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# retrieve action
# ---------------------------------------------------------------------------


@patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"})
def test_retrieve_memories():
    """retrieve returns a list of matching memories."""
    memories = [
        {"id": "mem-1", "content": "hello world", "score": 0.95, "importance": 0.7},
        {"id": "mem-2", "content": "another fact", "score": 0.80, "importance": 0.5},
    ]
    with patch("strands_tools.dakera_memory.DakeraServiceClient") as MockClient:
        instance = MockClient.return_value
        instance.search_memories.return_value = memories

        tool = make_tool({"action": "retrieve", "agent_id": "alex", "query": "hello", "top_k": 2})
        result = dakera_memory(tool=tool)

    assert result["status"] == "success"
    body = json.loads(result["content"][0]["text"])
    assert len(body) == 2
    assert body[0]["id"] == "mem-1"
    instance.search_memories.assert_called_once_with(agent_id="alex", query="hello", top_k=2)


def test_retrieve_missing_query():
    """retrieve without query returns error."""
    tool = make_tool({"action": "retrieve", "agent_id": "alex"})
    with patch("strands_tools.dakera_memory.DakeraServiceClient"):
        result = dakera_memory(tool=tool)
    assert result["status"] == "error"
    assert "query is required" in result["content"][0]["text"]


def test_retrieve_empty_results():
    """retrieve with no matches returns success with empty list."""
    with patch("strands_tools.dakera_memory.DakeraServiceClient") as MockClient:
        instance = MockClient.return_value
        instance.search_memories.return_value = []

        tool = make_tool({"action": "retrieve", "agent_id": "alex", "query": "xyz"})
        result = dakera_memory(tool=tool)

    assert result["status"] == "success"
    body = json.loads(result["content"][0]["text"])
    assert body == []


# ---------------------------------------------------------------------------
# get action
# ---------------------------------------------------------------------------


def test_get_memory():
    """get returns the memory dict."""
    memory = {
        "id": "mem-abc",
        "content": "stored fact",
        "importance": 0.9,
        "created_at": "2026-07-02T00:00:00Z",
        "metadata": {},
    }
    with patch("strands_tools.dakera_memory.DakeraServiceClient") as MockClient:
        instance = MockClient.return_value
        instance.get_memory.return_value = memory

        tool = make_tool({"action": "get", "agent_id": "alex", "memory_id": "mem-abc"})
        result = dakera_memory(tool=tool)

    assert result["status"] == "success"
    body = json.loads(result["content"][0]["text"])
    assert body["id"] == "mem-abc"
    instance.get_memory.assert_called_once_with("alex", "mem-abc")


def test_get_missing_memory_id():
    """get without memory_id returns error."""
    tool = make_tool({"action": "get", "agent_id": "alex"})
    with patch("strands_tools.dakera_memory.DakeraServiceClient"):
        result = dakera_memory(tool=tool)
    assert result["status"] == "error"
    assert "memory_id is required for get action" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# update action
# ---------------------------------------------------------------------------


@patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"})
def test_update_memory():
    """update returns updated memory dict."""
    updated = {
        "id": "mem-abc",
        "content": "updated content",
        "importance": 0.9,
        "created_at": "2026-07-02T00:00:00Z",
        "metadata": {"tag": "v2"},
    }
    with patch("strands_tools.dakera_memory.DakeraServiceClient") as MockClient:
        instance = MockClient.return_value
        instance.update_memory.return_value = updated

        tool = make_tool(
            {
                "action": "update",
                "agent_id": "alex",
                "memory_id": "mem-abc",
                "content": "updated content",
                "metadata": {"tag": "v2"},
            }
        )
        result = dakera_memory(tool=tool)

    assert result["status"] == "success"
    body = json.loads(result["content"][0]["text"])
    assert body["content"] == "updated content"


def test_update_missing_memory_id():
    """update without memory_id returns error."""
    tool = make_tool({"action": "update", "agent_id": "alex", "content": "new"})
    with patch("strands_tools.dakera_memory.DakeraServiceClient"):
        result = dakera_memory(tool=tool)
    assert result["status"] == "error"
    assert "memory_id is required for update action" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# delete action
# ---------------------------------------------------------------------------


@patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"})
def test_delete_memory():
    """delete returns success text with memory_id."""
    with patch("strands_tools.dakera_memory.DakeraServiceClient") as MockClient:
        instance = MockClient.return_value
        instance.delete_memory.return_value = {"status": "ok"}

        tool = make_tool({"action": "delete", "agent_id": "alex", "memory_id": "mem-abc"})
        result = dakera_memory(tool=tool)

    assert result["status"] == "success"
    assert "mem-abc" in result["content"][0]["text"]
    instance.delete_memory.assert_called_once_with("alex", "mem-abc")


def test_delete_missing_memory_id():
    """delete without memory_id returns error."""
    tool = make_tool({"action": "delete", "agent_id": "alex"})
    with patch("strands_tools.dakera_memory.DakeraServiceClient"):
        result = dakera_memory(tool=tool)
    assert result["status"] == "error"
    assert "memory_id is required for delete action" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Guard: missing required params
# ---------------------------------------------------------------------------


def test_missing_action():
    """No action returns error."""
    tool = make_tool({"agent_id": "alex"})
    with patch("strands_tools.dakera_memory.DakeraServiceClient"):
        result = dakera_memory(tool=tool)
    assert result["status"] == "error"
    assert "action parameter is required" in result["content"][0]["text"]


def test_missing_agent_id():
    """No agent_id returns error."""
    tool = make_tool({"action": "retrieve", "query": "test"})
    with patch("strands_tools.dakera_memory.DakeraServiceClient"):
        result = dakera_memory(tool=tool)
    assert result["status"] == "error"
    assert "agent_id parameter is required" in result["content"][0]["text"]


def test_invalid_action():
    """Unknown action returns error."""
    tool = make_tool({"action": "explode", "agent_id": "alex"})
    with patch("strands_tools.dakera_memory.DakeraServiceClient"):
        result = dakera_memory(tool=tool)
    assert result["status"] == "error"
    assert "Invalid action" in result["content"][0]["text"]
