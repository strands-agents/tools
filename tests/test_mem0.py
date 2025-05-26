"""
Tests for the memory tool using the Agent interface.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent
from strands.types.tools import ToolUse
from strands_tools import mem0_memory
from strands_tools.mem0_memory import Mem0ServiceClient


@pytest.fixture
def agent():
    """Create an agent with the memory tool loaded."""
    return Agent(tools=[mem0_memory])


@pytest.fixture
def mock_mem0_service_client():
    """Create a mock mem0 service client."""
    client = MagicMock(spec=Mem0ServiceClient)
    return client


@pytest.fixture
def mock_tool():
    """Create a mock tool use object that properly mocks the tool interface."""
    mock = MagicMock(spec=ToolUse)
    # Set up the get method to behave like a dictionary get
    mock.get = MagicMock()
    mock.get.return_value = {}
    # Set a default tool use ID
    mock.get.side_effect = lambda key, default=None: {"toolUseId": "test-id", "input": {}}.get(key, default)
    return mock


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        content = result["content"][0]
        # Handle different response formats
        if isinstance(content, dict):
            if "text" in content:
                return content["text"]
            # Return the first key-value pair if it's a memory object
            elif "id" in content and "memory" in content:
                return content["memory"]
    return str(result)


@patch.dict(os.environ, {"OPENSEARCH_HOST": "test.opensearch.amazonaws.com"})
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_store_memory(mock_opensearch, mock_mem0_client, mock_mem0_service_client, mock_tool):
    """Test store memory functionality."""
    # Setup mocks
    mock_mem0_client.return_value = mock_mem0_service_client

    # Configure the mock_tool
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "store",
            "content": "Test memory content",
            "user_id": "test_user",
            "metadata": {"category": "test"},
        },
    }.get(key, default)

    # Mock data
    store_response = {
        "results": [
            {
                "event": "store",
                "memory": "Test memory content",
                "id": "mem123",
                "created_at": "2024-03-20T10:00:00Z",
            }
        ]
    }

    # Configure mocks
    mock_mem0_service_client.store_memory.return_value = store_response

    # Call the memory function
    result = mem0_memory.mem0_memory(tool=mock_tool)

    # Assertions
    assert result["status"] == "success"
    assert "Successfully stored" in str(result["content"][0]["text"])

    # Verify correct functions were called - check just that it was called with the right values
    mock_mem0_service_client.store_memory.assert_called_once()
    call_args, call_kwargs = mock_mem0_service_client.store_memory.call_args
    assert call_args[0] == "Test memory content"
    assert call_args[1] == "test_user" or call_kwargs.get("user_id") == "test_user"
    assert call_args[3] == {"category": "test"} or call_kwargs.get("metadata") == {"category": "test"}


@patch.dict(os.environ, {"OPENSEARCH_HOST": "test.opensearch.amazonaws.com"})
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_get_memory(mock_opensearch, mock_mem0_client, mock_mem0_service_client, mock_tool):
    """Test get memory functionality."""
    # Setup mocks
    mock_mem0_client.return_value = mock_mem0_service_client

    # Configure the mock_tool
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "get", "memory_id": "mem123"},
    }.get(key, default)

    # Mock data
    get_response = {
        "id": "mem123",
        "memory": "Test memory content",
        "created_at": "2024-03-20T10:00:00Z",
        "user_id": "test_user",
        "metadata": {"category": "test"},
    }

    # Configure mocks
    mock_mem0_service_client.get_memory.return_value = get_response

    # Call the memory function
    result = mem0_memory.mem0_memory(tool=mock_tool)

    # Assertions
    assert result["status"] == "success"
    assert isinstance(result["content"], list)
    assert len(result["content"]) > 0
    assert "text" in result["content"][0]
    memory = json.loads(result["content"][0]["text"])
    assert memory["id"] == "mem123"
    assert memory["memory"] == "Test memory content"
    assert memory["user_id"] == "test_user"
    assert memory["metadata"] == {"category": "test"}


@patch.dict(os.environ, {"OPENSEARCH_HOST": "test.opensearch.amazonaws.com"})
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_list_memories(mock_opensearch, mock_mem0_client, mock_mem0_service_client, mock_tool):
    """Test list memories functionality."""
    # Setup mocks
    mock_mem0_client.return_value = mock_mem0_service_client

    # Configure the mock_tool
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "list", "user_id": "test_user"},
    }.get(key, default)

    # Mock data for list_memories response - the memory.py expects this format
    list_response = {
        "results": [
            {
                "id": "mem123",
                "memory": "Test memory content",
                "created_at": "2024-03-20T10:00:00Z",
                "user_id": "test_user",
                "metadata": {"category": "test"},
            }
        ]
    }

    # Configure mocks
    mock_mem0_service_client.list_memories.return_value = list_response

    # Call the memory function
    result = mem0_memory.mem0_memory(tool=mock_tool)

    # Assertions
    assert result["status"] == "success"
    assert isinstance(result["content"], list)
    assert len(result["content"]) > 0
    assert "text" in result["content"][0]
    # Parse the JSON string in text
    memories = json.loads(result["content"][0]["text"])
    assert isinstance(memories, list)
    assert len(memories) > 0
    assert "id" in memories[0]
    assert memories[0]["id"] == "mem123"


@patch.dict(os.environ, {"OPENSEARCH_HOST": "test.opensearch.amazonaws.com"})
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_retrieve_memories(mock_opensearch, mock_mem0_client, mock_mem0_service_client, mock_tool):
    """Test retrieve memories functionality."""
    # Setup mocks
    mock_mem0_client.return_value = mock_mem0_service_client

    # Configure the mock_tool
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "retrieve", "query": "test query", "user_id": "test_user"},
    }.get(key, default)

    # Mock data for search_memories response - the memory.py expects this format
    retrieve_response = {
        "results": [
            {
                "id": "mem123",
                "memory": "Test memory content",
                "score": 0.85,
                "created_at": "2024-03-20T10:00:00Z",
                "user_id": "test_user",
                "metadata": {"category": "test"},
            }
        ]
    }

    # Configure mocks
    mock_mem0_service_client.search_memories.return_value = retrieve_response

    # Call the memory function
    result = mem0_memory.mem0_memory(tool=mock_tool)

    # Assertions
    assert result["status"] == "success"
    assert isinstance(result["content"], list)
    assert len(result["content"]) > 0
    assert "text" in result["content"][0]
    # Parse the JSON string in text
    memories = json.loads(result["content"][0]["text"])
    assert isinstance(memories, list)
    assert len(memories) > 0
    assert "id" in memories[0]
    assert memories[0]["id"] == "mem123"


@patch.dict(os.environ, {"OPENSEARCH_HOST": "test.opensearch.amazonaws.com", "BYPASS_TOOL_CONSENT": "true"})
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_delete_memory(mock_opensearch, mock_mem0_client, mock_mem0_service_client, mock_tool):
    """Test delete memory functionality with BYPASS_TOOL_CONSENT mode enabled."""
    # Setup mocks
    mock_mem0_client.return_value = mock_mem0_service_client

    # Configure the mock_tool
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "delete", "memory_id": "mem123"},
    }.get(key, default)

    # Configure mocks
    mock_mem0_service_client.delete_memory.return_value = {"status": "success"}

    # Call the memory function
    result = mem0_memory.mem0_memory(tool=mock_tool)

    # Assertions
    assert result["status"] == "success"
    assert "Memory mem123 deleted successfully" in str(result["content"][0]["text"])

    # Verify correct functions were called
    mock_mem0_service_client.delete_memory.assert_called_once()
    call_args = mock_mem0_service_client.delete_memory.call_args[0]
    assert call_args[0] == "mem123"


@patch.dict(os.environ, {"OPENSEARCH_HOST": "test.opensearch.amazonaws.com"})
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_get_memory_history(mock_opensearch, mock_mem0_client, mock_mem0_service_client, mock_tool):
    """Test get memory history functionality."""
    # Setup mocks
    mock_mem0_client.return_value = mock_mem0_service_client

    # Configure the mock_tool
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "history", "memory_id": "mem123"},
    }.get(key, default)

    # Mock data
    history_response = [
        {
            "id": "hist123",
            "memory_id": "mem123",
            "event": "store",
            "old_memory": None,
            "new_memory": "Test memory content",
            "created_at": "2024-03-20T10:00:00Z",
        }
    ]

    # Configure mocks
    mock_mem0_service_client.get_memory_history.return_value = history_response

    # Call the memory function
    result = mem0_memory.mem0_memory(tool=mock_tool)

    # Assertions
    assert result["status"] == "success"
    assert isinstance(result["content"], list)
    assert len(result["content"]) > 0
    assert "text" in result["content"][0]
    # Parse the JSON string in text
    history = json.loads(result["content"][0]["text"])
    assert isinstance(history, list)
    assert len(history) > 0
    assert "id" in history[0]
    assert history[0]["id"] == "hist123"


@patch.dict(os.environ, {"OPENSEARCH_HOST": "test.opensearch.amazonaws.com"})
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_invalid_action(mock_opensearch, mock_mem0_client, mock_tool):
    """Test invalid action."""
    # Configure the mock_tool
    mock_tool.get.side_effect = lambda key, default=None: {"toolUseId": "test-id", "input": {"action": "invalid"}}.get(
        key, default
    )

    result = mem0_memory.mem0_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "Invalid action" in str(result["content"][0]["text"])


@patch.dict(os.environ, {})
def test_missing_opensearch_host(mock_tool):
    """Test missing OpenSearch host."""
    # Configure the mock_tool
    mock_tool.get.side_effect = lambda key, default=None: {"toolUseId": "test-id", "input": {"action": "list"}}.get(
        key, default
    )

    result = mem0_memory.mem0_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "OPENSEARCH_HOST environment variable is required" in str(result["content"][0]["text"])


@patch.dict(os.environ, {"OPENSEARCH_HOST": "test.opensearch.amazonaws.com"})
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_action_specific_missing_params(mock_opensearch, mock_mem0_client, mock_tool):
    """Test missing action-specific parameters."""
    # Setup mock
    mock_client = MagicMock()
    mock_mem0_client.return_value = mock_client

    # Test missing content for store action
    mock_tool.get.side_effect = lambda key, default=None: {"toolUseId": "test-id", "input": {"action": "store"}}.get(
        key, default
    )
    store_result = mem0_memory.mem0_memory(tool=mock_tool)
    assert store_result["status"] == "error"
    assert "content is required for store action" in str(store_result["content"][0]["text"])

    # Test missing memory_id for delete action
    mock_tool.get.side_effect = lambda key, default=None: {"toolUseId": "test-id", "input": {"action": "delete"}}.get(
        key, default
    )
    delete_result = mem0_memory.mem0_memory(tool=mock_tool)
    assert delete_result["status"] == "error"
    assert "memory_id is required for delete action" in str(delete_result["content"][0]["text"])

    # Test missing memory_id for get action
    mock_tool.get.side_effect = lambda key, default=None: {"toolUseId": "test-id", "input": {"action": "get"}}.get(
        key, default
    )
    get_result = mem0_memory.mem0_memory(tool=mock_tool)
    assert get_result["status"] == "error"
    assert "memory_id is required for get action" in str(get_result["content"][0]["text"])

    # Test missing query for retrieve action
    mock_tool.get.side_effect = lambda key, default=None: {"toolUseId": "test-id", "input": {"action": "retrieve"}}.get(
        key, default
    )
    retrieve_result = mem0_memory.mem0_memory(tool=mock_tool)
    assert retrieve_result["status"] == "error"
    assert "query is required for retrieve action" in str(retrieve_result["content"][0]["text"])


@patch("boto3.Session")
@patch("strands_tools.mem0_memory.Mem0Memory")
@patch("opensearchpy.OpenSearch")
def test_mem0_service_client_init(mock_opensearch, mock_mem0_memory, mock_session):
    """Test Mem0ServiceClient initialization."""
    # Mock session and credentials
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_session.return_value.get_credentials.return_value = mock_credentials

    # Test with default parameters
    with patch.dict(os.environ, {"OPENSEARCH_HOST": "test.opensearch.amazonaws.com"}):
        client = Mem0ServiceClient()
        assert client.region == os.environ.get("AWS_REGION", "us-west-2")

    # Test with custom config
    custom_config = {
        "embedder": {"provider": "custom", "config": {"model": "custom-model"}},
        "llm": {"provider": "custom", "config": {"model": "custom-model"}},
    }
    with patch.dict(os.environ, {"OPENSEARCH_HOST": "test.opensearch.amazonaws.com"}):
        custom_client = Mem0ServiceClient(config=custom_config)
        assert custom_client.mem0 is not None
