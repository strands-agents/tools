"""
Tests for the MemMachine memory tool using the Agent interface.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from strands.types.tools import ToolUse

from strands_tools import memmachine_memory
from strands_tools.memmachine_memory import MemMachineServiceClient


@pytest.fixture
def mock_tool():
    """Create a mock tool use object that properly mocks the tool interface."""
    mock = MagicMock(spec=ToolUse)
    mock.get = MagicMock()
    mock.get.return_value = {}
    mock.get.side_effect = lambda key, default=None: {"toolUseId": "test-id", "input": {}}.get(key, default)
    return mock


@pytest.fixture
def mock_memmachine_service_client():
    """Create a mock MemMachine service client."""
    client = MagicMock(spec=MemMachineServiceClient)
    return client


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_store_memory(mock_client_class, mock_tool):
    """Test store memory functionality."""
    mock_client = MagicMock()
    mock_client.store_memory.return_value = {"results": [{"uid": "mem-123"}]}
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "store",
            "content": "User prefers aisle seats on flights",
            "metadata": {"category": "travel"},
        },
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    result_data = json.loads(result["content"][0]["text"])
    assert "results" in result_data
    assert result_data["results"][0]["uid"] == "mem-123"


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_store_memory_with_options(mock_client_class, mock_tool):
    """Test store memory with producer, produced_for, types, and metadata."""
    mock_client = MagicMock()
    mock_client.store_memory.return_value = {"results": [{"uid": "mem-456"}]}
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "store",
            "content": "Meeting at 10 AM tomorrow",
            "producer": "assistant",
            "produced_for": "alice",
            "types": ["episodic"],
            "metadata": {"type": "reminder", "priority": "high"},
        },
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    mock_client.store_memory.assert_called_once_with(
        content="Meeting at 10 AM tomorrow",
        types=["episodic"],
        producer="assistant",
        produced_for="alice",
        metadata={"type": "reminder", "priority": "high"},
    )


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_search_memories(mock_client_class, mock_tool):
    """Test search memories functionality."""
    mock_client = MagicMock()
    mock_client.search_memories.return_value = {
        "status": 0,
        "content": {
            "episodic_memory": {
                "long_term_memory": {
                    "episodes": [
                        {
                            "content": "User prefers aisle seats on flights",
                            "score": 0.95,
                            "created_at": "2024-03-20T10:00:00Z",
                        }
                    ]
                }
            }
        },
    }
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "search",
            "query": "flight preferences",
            "top_k": 5,
        },
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    result_data = json.loads(result["content"][0]["text"])
    assert "content" in result_data
    assert "episodic_memory" in result_data["content"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_search_memories_with_filter(mock_client_class, mock_tool):
    """Test search memories with filter and types."""
    mock_client = MagicMock()
    mock_client.search_memories.return_value = {
        "status": 0,
        "content": {"semantic_memory": {"memories": [{"content": "Prefers Python", "score": 0.88}]}},
    }
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "search",
            "query": "programming preferences",
            "types": ["semantic"],
            "filter": "metadata.user_id=alice",
        },
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    mock_client.search_memories.assert_called_once_with(
        query="programming preferences",
        top_k=10,
        types=["semantic"],
        filter_str="metadata.user_id=alice",
    )


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_search_memories_empty_results(mock_client_class, mock_tool):
    """Test search memories with no results."""
    mock_client = MagicMock()
    mock_client.search_memories.return_value = {
        "status": 0,
        "content": {},
    }
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "search",
            "query": "nonexistent topic",
        },
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_list_memories(mock_client_class, mock_tool):
    """Test list memories functionality."""
    mock_client = MagicMock()
    mock_client.list_memories.return_value = {
        "status": 0,
        "content": {
            "episodic_memory": [
                {
                    "uid": "mem-123",
                    "content": "User prefers aisle seats",
                    "created_at": "2024-03-20T10:00:00Z",
                    "metadata": {"category": "travel"},
                }
            ]
        },
    }
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "list",
            "page_size": 50,
        },
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    result_data = json.loads(result["content"][0]["text"])
    assert "content" in result_data


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_list_memories_with_filter_and_type(mock_client_class, mock_tool):
    """Test list memories with memory_type filter and metadata filter."""
    mock_client = MagicMock()
    mock_client.list_memories.return_value = {
        "status": 0,
        "content": {},
    }
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "list",
            "memory_type": "episodic",
            "filter": "metadata.user_id=alice",
            "page_size": 25,
            "page_num": 2,
        },
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    mock_client.list_memories.assert_called_once_with(
        page_size=25,
        page_num=2,
        memory_type="episodic",
        filter_str="metadata.user_id=alice",
    )


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key", "BYPASS_TOOL_CONSENT": "true"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_delete_episodic_memory(mock_client_class, mock_tool):
    """Test delete episodic memory functionality."""
    mock_client = MagicMock()
    mock_client.delete_episodic_memory.return_value = None
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "delete",
            "memory_type": "episodic",
            "memory_id": "mem-123",
        },
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    assert "mem-123" in result["content"][0]["text"]
    mock_client.delete_episodic_memory.assert_called_once_with(memory_id="mem-123", memory_ids=None)


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key", "BYPASS_TOOL_CONSENT": "true"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_delete_semantic_memory(mock_client_class, mock_tool):
    """Test delete semantic memory functionality."""
    mock_client = MagicMock()
    mock_client.delete_semantic_memory.return_value = None
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "delete",
            "memory_type": "semantic",
            "memory_id": "sem-456",
        },
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    assert "sem-456" in result["content"][0]["text"]
    mock_client.delete_semantic_memory.assert_called_once_with(memory_id="sem-456", memory_ids=None)


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key", "BYPASS_TOOL_CONSENT": "true"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_bulk_delete_episodic(mock_client_class, mock_tool):
    """Test bulk delete episodic memories."""
    mock_client = MagicMock()
    mock_client.delete_episodic_memory.return_value = None
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "delete",
            "memory_type": "episodic",
            "memory_ids": ["mem-1", "mem-2", "mem-3"],
        },
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    mock_client.delete_episodic_memory.assert_called_once_with(memory_id=None, memory_ids=["mem-1", "mem-2", "mem-3"])


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_invalid_action(mock_client_class, mock_tool):
    """Test invalid action."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "invalid"},
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "Invalid action" in result["content"][0]["text"]


def test_missing_api_key(mock_tool, monkeypatch):
    """Test missing API key raises error."""
    monkeypatch.delenv("MEMMACHINE_API_KEY", raising=False)

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "list"},
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "MEMMACHINE_API_KEY" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_missing_content_for_store(mock_client_class, mock_tool):
    """Test missing content for store action."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "store"},
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "content is required" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_missing_query_for_search(mock_client_class, mock_tool):
    """Test missing query for search action."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "search"},
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "query is required" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_missing_memory_type_for_delete(mock_client_class, mock_tool):
    """Test missing memory_type for delete action."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "delete", "memory_id": "mem-123"},
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "memory_type is required" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_missing_memory_id_for_delete(mock_client_class, mock_tool):
    """Test missing memory_id and memory_ids for delete action."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "delete", "memory_type": "episodic"},
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "memory_id or memory_ids is required" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_invalid_memory_type_for_delete(mock_client_class, mock_tool):
    """Test invalid memory_type for delete action."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "delete",
            "memory_type": "invalid_type",
            "memory_id": "mem-123",
        },
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "Invalid memory_type" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
def test_client_initialization():
    """Test MemMachineServiceClient initialization with default settings."""
    client = MemMachineServiceClient()
    assert client.api_key == "test-api-key"
    assert client.base_url == "https://api.memmachine.ai"


@patch.dict(
    os.environ,
    {"MEMMACHINE_API_KEY": "test-api-key", "MEMMACHINE_BASE_URL": "http://localhost:8080"},
)
def test_client_custom_base_url():
    """Test MemMachineServiceClient with custom base URL."""
    client = MemMachineServiceClient()
    assert client.base_url == "http://localhost:8080"


@patch.dict(
    os.environ,
    {"MEMMACHINE_API_KEY": "test-api-key", "MEMMACHINE_BASE_URL": "http://localhost:8080/"},
)
def test_client_base_url_trailing_slash():
    """Test MemMachineServiceClient strips trailing slash from base URL."""
    client = MemMachineServiceClient()
    assert client.base_url == "http://localhost:8080"


def test_client_missing_api_key(monkeypatch):
    """Test MemMachineServiceClient raises ValueError without API key."""
    monkeypatch.delenv("MEMMACHINE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MEMMACHINE_API_KEY"):
        MemMachineServiceClient()


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_api_error_handling(mock_client_class, mock_tool):
    """Test API error handling."""
    mock_client = MagicMock()
    mock_client.search_memories.side_effect = Exception("API connection failed")
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "search",
            "query": "test query",
        },
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "API connection failed" in result["content"][0]["text"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_store_empty_results(mock_client_class, mock_tool):
    """Test store memory with empty results."""
    mock_client = MagicMock()
    mock_client.store_memory.return_value = {"results": []}
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "store",
            "content": "Test content",
        },
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_search_with_flat_episodic_list(mock_client_class, mock_tool):
    """Test search with flat episodic memory list (non-nested structure)."""
    mock_client = MagicMock()
    mock_client.search_memories.return_value = {
        "status": 0,
        "content": {
            "episodic_memory": [
                {"content": "Memory 1", "score": 0.9},
                {"content": "Memory 2", "score": 0.7},
            ]
        },
    }
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "search",
            "query": "test",
        },
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    result_data = json.loads(result["content"][0]["text"])
    assert "episodic_memory" in result_data["content"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
@patch("strands_tools.memmachine_memory.MemMachineServiceClient")
def test_search_with_semantic_memories(mock_client_class, mock_tool):
    """Test search with semantic memory results."""
    mock_client = MagicMock()
    mock_client.search_memories.return_value = {
        "status": 0,
        "content": {"semantic_memory": {"memories": [{"content": "User likes Python", "score": 0.92}]}},
    }
    mock_client_class.return_value = mock_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "search",
            "query": "programming",
        },
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "success"
    result_data = json.loads(result["content"][0]["text"])
    assert "semantic_memory" in result_data["content"]


@patch.dict(os.environ, {"MEMMACHINE_API_KEY": "test-api-key"})
def test_missing_action(mock_tool):
    """Test missing action parameter."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {},
    }.get(key, default)

    result = memmachine_memory.memmachine_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "action parameter is required" in result["content"][0]["text"]
