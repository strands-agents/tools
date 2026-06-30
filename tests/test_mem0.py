"""
Tests for the memory tool using the Agent interface.
"""

import builtins
import json
import os
import sys
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


@patch.dict(
    os.environ,
    {
        "MEM0_LLM_PROVIDER": "openai",
        "MEM0_LLM_MODEL": "gpt-4o",
        "MEM0_LLM_TEMPERATURE": "0.2",
        "MEM0_LLM_MAX_TOKENS": "4000",
        "MEM0_EMBEDDER_PROVIDER": "openai",
        "MEM0_EMBEDDER_MODEL": "text-embedding-3-large",
        "OPENSEARCH_HOST": "test.opensearch.amazonaws.com",
    },
)
@patch("strands_tools.mem0_memory.Mem0Memory")
@patch("strands_tools.mem0_memory.boto3.Session")
def test_store_memory(mock_boto3_session, mock_mem0_memory, mock_tool):
    """Test store memory functionality."""
    # Setup mock AWS credentials
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test_access_key"
    mock_credentials.secret_key = "test_secret_key"
    mock_credentials.token = "test_token"
    mock_session = MagicMock()
    mock_session.get_credentials.return_value = mock_credentials
    mock_boto3_session.return_value = mock_session

    # Setup mock client
    mock_client = MagicMock()
    mock_client.add.return_value = [
        {
            "event": "store",
            "memory": "Test memory content",
            "id": "mem123",
            "created_at": "2024-03-20T10:00:00Z",
        }
    ]
    mock_mem0_memory.from_config.return_value = mock_client

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

    # Call the memory function
    result = mem0_memory.mem0_memory(tool=mock_tool)

    # Assertions
    assert result["status"] == "success"
    assert result["content"][0]["text"] == json.dumps(
        [
            {
                "event": "store",
                "memory": "Test memory content",
                "id": "mem123",
                "created_at": "2024-03-20T10:00:00Z",
            }
        ],
        indent=2,
    )


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
    """Test missing OpenSearch host defaults to FAISS."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "list", "user_id": "test-user"},
    }.get(key, default)

    real_import = builtins.__import__

    def fail_faiss(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("No module named 'faiss'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fail_faiss):
        result = mem0_memory.mem0_memory(tool=mock_tool)
        assert result["status"] == "error"
        assert "The faiss-cpu package is required" in str(result["content"][0]["text"])


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

    # Test with default parameters (OpenSearch)
    with patch.dict(os.environ, {"OPENSEARCH_HOST": "test.opensearch.amazonaws.com"}):
        client = Mem0ServiceClient()
        assert client.region == os.environ.get("AWS_REGION", "us-west-2")

    # Test with conflict scenario
    with patch.dict(
        os.environ,
        {
            "OPENSEARCH_HOST": "test.opensearch.amazonaws.com",
            "NEPTUNE_ANALYTICS_GRAPH_IDENTIFIER": "g-5aaaaa1234",
        },
    ):
        with pytest.raises(RuntimeError):
            Mem0ServiceClient()

    # Test with Neptune Analytics for both vector and graph
    with patch.dict(
        os.environ,
        {
            "NEPTUNE_ANALYTICS_GRAPH_IDENTIFIER": "g-5aaaaa1234",
        },
    ):
        client = Mem0ServiceClient()
        assert client.mem0 is not None

    # Test with Neptune Database with OpenSearch
    with patch.dict(
        os.environ,
        {
            "OPENSEARCH_HOST": "test.opensearch.amazonaws.com",
            "NEPTUNE_DATABASE_ENDPOINT": "xxx.us-west-2.neptune.amazonaws.com",
        },
    ):
        client = Mem0ServiceClient()
        assert client.region == os.environ.get("AWS_REGION", "us-west-2")
        assert client.mem0 is not None

    # Test with custom config (OpenSearch)
    custom_config = {
        "embedder": {"provider": "custom", "config": {"model": "custom-model"}},
        "llm": {"provider": "custom", "config": {"model": "custom-model"}},
    }
    with patch.dict(os.environ, {"OPENSEARCH_HOST": "test.opensearch.amazonaws.com"}):
        custom_client = Mem0ServiceClient(config=custom_config)
        assert custom_client.mem0 is not None

    # Test with Mem0 Platform
    with patch.dict(os.environ, {"MEM0_API_KEY": "test-api-key"}):
        with patch("strands_tools.mem0_memory.MemoryClient") as mock_memory_client:
            mock_client = MagicMock()
            mock_client._validate_api_key.return_value = "test@example.com"
            mock_memory_client.return_value = mock_client
            platform_client = Mem0ServiceClient()
            assert platform_client.mem0 is not None


@patch.dict(os.environ, {"MEM0_API_KEY": "test-api-key"})
@patch("strands_tools.mem0_memory.MemoryClient")
def test_mem0_platform_client(mock_memory_client, mock_tool):
    """Test Mem0 Platform client functionality."""
    # Setup mock client
    mock_client = MagicMock()
    mock_client.add.return_value = [
        {
            "event": "store",
            "memory": "Test memory content",
            "id": "mem123",
            "created_at": "2024-03-20T10:00:00Z",
        }
    ]
    mock_memory_client.return_value = mock_client

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

    # Call the memory function
    result = mem0_memory.mem0_memory(tool=mock_tool)

    # Assertions
    assert result["status"] == "success"
    assert "Test memory content" in str(result["content"][0]["text"])


@patch.dict(os.environ, {})
@patch("strands_tools.mem0_memory.Mem0Memory")
def test_faiss_client(mock_mem0_memory, mock_tool):
    """Test FAISS client functionality."""
    # Inject a mock faiss module into sys.modules
    sys.modules["faiss"] = MagicMock()
    # Setup mock client
    mock_client = MagicMock()
    # Return a real list of dicts, not MagicMock objects
    mock_client.add.return_value = [
        {
            "event": "store",
            "memory": "Test memory content",
            "id": "mem123",
            "created_at": "2024-03-20T10:00:00Z",
        }
    ]
    mock_mem0_memory.from_config.return_value = mock_client

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

    # Call the memory function
    result = mem0_memory.mem0_memory(tool=mock_tool)

    # Assertions
    assert result["status"] == "success"
    assert "Test memory content" in str(result["content"][0]["text"])


# Tests for PostgreSQL/PGVector backend functionality


@patch.dict(
    os.environ,
    {
        "POSTGRESQL_HOST": "localhost",
        "POSTGRESQL_PORT": "5432",
        "POSTGRESQL_USER": "test_user",
        "POSTGRESQL_PASSWORD": "test_password",
        "DB_NAME": "test_db",
        "DB_COLLECTION_NAME": "test_collection",
    },
)
@patch("strands_tools.mem0_memory.Mem0Memory")
@patch("boto3.Session")
def test_postgresql_backend_initialization(mock_session, mock_mem0_memory):
    """Test PGVector backend initializes correctly with required environment variables."""
    # Mock session and credentials
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_session.return_value.get_credentials.return_value = mock_credentials

    mock_client = MagicMock()
    mock_mem0_memory.from_config.return_value = mock_client

    Mem0ServiceClient()

    # Verify that from_config was called
    assert mock_mem0_memory.from_config.called
    call_args = mock_mem0_memory.from_config.call_args
    config = call_args[1]["config_dict"]

    # Verify PGVector configuration
    assert "vector_store" in config
    assert config["vector_store"]["provider"] == "pgvector"
    assert config["vector_store"]["config"]["host"] == "localhost"
    assert config["vector_store"]["config"]["port"] == 5432
    assert config["vector_store"]["config"]["user"] == "test_user"
    assert config["vector_store"]["config"]["password"] == "test_password"
    assert config["vector_store"]["config"]["dbname"] == "test_db"
    assert config["vector_store"]["config"]["collection_name"] == "test_collection"


@patch.dict(
    os.environ,
    {
        "POSTGRESQL_HOST": "localhost",
    },
)
@patch("strands_tools.mem0_memory.Mem0Memory")
@patch("boto3.Session")
def test_postgresql_backend_with_defaults(mock_session, mock_mem0_memory):
    """Test PGVector backend uses default values when optional env vars are not set."""
    # Mock session and credentials
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_session.return_value.get_credentials.return_value = mock_credentials

    mock_client = MagicMock()
    mock_mem0_memory.from_config.return_value = mock_client

    Mem0ServiceClient()

    # Verify that from_config was called
    assert mock_mem0_memory.from_config.called
    call_args = mock_mem0_memory.from_config.call_args
    config = call_args[1]["config_dict"]

    # Verify PGVector configuration with defaults
    assert "vector_store" in config
    assert config["vector_store"]["provider"] == "pgvector"
    assert config["vector_store"]["config"]["host"] == "localhost"
    assert config["vector_store"]["config"]["port"] == 5432  # Default
    assert config["vector_store"]["config"]["dbname"] == "postgres"  # Default
    assert config["vector_store"]["config"]["collection_name"] == "mem0_memories"  # Default


@patch("strands_tools.mem0_memory.Mem0Memory")
@patch("boto3.Session")
def test_append_pgvector_config_method(mock_session, mock_mem0_memory):
    """Test the _append_pgvector_config method directly."""
    # Mock session and credentials
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_session.return_value.get_credentials.return_value = mock_credentials

    with patch.dict(
        os.environ,
        {
            "POSTGRESQL_HOST": "pg.example.com",
            "POSTGRESQL_PORT": "5433",
            "POSTGRESQL_USER": "admin",
            "POSTGRESQL_PASSWORD": "secret",
            "DB_NAME": "custom_db",
            "DB_COLLECTION_NAME": "custom_collection",
        },
    ):
        client = Mem0ServiceClient()

        # Create a test config
        test_config = {
            "llm": {"provider": "test", "config": {"model": "test-model"}},
            "embedder": {"provider": "test", "config": {"model": "test-embedder"}},
        }

        # Call the method
        result = client._append_pgvector_config(test_config)

        # Verify the result
        assert result["vector_store"]["provider"] == "pgvector"
        assert result["vector_store"]["config"]["host"] == "pg.example.com"
        assert result["vector_store"]["config"]["port"] == 5433
        assert result["vector_store"]["config"]["user"] == "admin"
        assert result["vector_store"]["config"]["password"] == "secret"
        assert result["vector_store"]["config"]["dbname"] == "custom_db"
        assert result["vector_store"]["config"]["collection_name"] == "custom_collection"


@patch.dict(
    os.environ,
    {
        "POSTGRESQL_HOST": "localhost",
        "POSTGRESQL_USER": "test_user",
        "POSTGRESQL_PASSWORD": "test_password",
    },
)
@patch("strands_tools.mem0_memory.Mem0Memory")
@patch("boto3.Session")
def test_postgresql_with_custom_config(mock_session, mock_mem0_memory):
    """Test PGVector backend merges custom config with PostgreSQL config."""
    # Mock session and credentials
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_session.return_value.get_credentials.return_value = mock_credentials

    mock_client = MagicMock()
    mock_mem0_memory.from_config.return_value = mock_client

    custom_config = {
        "embedder": {"provider": "custom", "config": {"model": "custom-model"}},
        "llm": {"provider": "custom", "config": {"model": "custom-llm"}},
    }

    Mem0ServiceClient(config=custom_config)

    # Verify that from_config was called
    assert mock_mem0_memory.from_config.called
    call_args = mock_mem0_memory.from_config.call_args
    config = call_args[1]["config_dict"]

    # Verify both custom config and PGVector config are present
    assert config["embedder"]["provider"] == "custom"
    assert config["llm"]["provider"] == "custom"
    assert config["llm"]["config"]["model"] == "custom-llm"
    assert config["vector_store"]["provider"] == "pgvector"
    assert config["vector_store"]["config"]["host"] == "localhost"
