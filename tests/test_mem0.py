"""
Tests for the mem0_memory tool.

Covers the standalone module-level function (single-tenant, identity from the
environment) and the Mem0MemoryTool class (multi-tenant, identity bound at
construction). Also includes IDOR regression guards ensuring the tenant
identity is never an LLM-controllable tool parameter.
"""

import builtins
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from strands.types.tools import ToolUse

from strands_tools import mem0_memory
from strands_tools.mem0_memory import Mem0MemoryTool, Mem0ServiceClient


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


@pytest.fixture
def mock_mem0_service_client():
    """Create a mock mem0 service client."""
    client = MagicMock(spec=Mem0ServiceClient)
    return client


# --- Security Tests (IDOR guards) ---


def test_identity_params_not_in_standalone_tool_spec():
    """IDOR guard: the standalone tool spec must not expose tenant identity params.

    Regression test for the IDOR where an LLM-supplied user_id/agent_id could read, write, or delete
    another tenant's memories. Identity must come from the environment, never the LLM.
    """
    properties = mem0_memory.TOOL_SPEC["inputSchema"]["json"]["properties"]
    for forbidden in ["user_id", "agent_id"]:
        assert forbidden not in properties, f"'{forbidden}' must not be an LLM-controllable tool parameter"


def test_identity_params_not_in_class_tool_spec():
    """IDOR guard: Mem0MemoryTool.mem0_memory must not expose tenant identity to the agent.

    Identity is bound at construction; the LLM cannot select a different tenant key.
    """
    tool = Mem0MemoryTool(user_id="test_user")
    properties = tool.mem0_memory.TOOL_SPEC["inputSchema"]["json"]["properties"]
    for forbidden in ["user_id", "agent_id"]:
        assert forbidden not in properties


# --- Standalone function (identity from environment) ---


@patch.dict(
    os.environ,
    {
        "MEM0_USER_ID": "test_user",
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
    """Test store uses the environment-configured user_id, not one from tool input."""
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

    # Configure the mock_tool (no user_id/agent_id - those come from the environment)
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "store",
            "content": "Test memory content",
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
    # The environment-configured identity was used, not anything from tool input
    mock_client.add.assert_called_once()
    assert mock_client.add.call_args[1]["user_id"] == "test_user"


@patch.dict(os.environ, {"MEM0_USER_ID": "test_user", "OPENSEARCH_HOST": "test.opensearch.amazonaws.com"})
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

    # Mock data - owned by the environment-configured user
    get_response = {
        "id": "mem123",
        "memory": "Test memory content",
        "created_at": "2024-03-20T10:00:00Z",
        "user_id": "test_user",
        "metadata": {"category": "test"},
    }
    mock_mem0_service_client.get_memory.return_value = get_response

    # Call the memory function
    result = mem0_memory.mem0_memory(tool=mock_tool)

    # Assertions
    assert result["status"] == "success"
    memory = json.loads(result["content"][0]["text"])
    assert memory["id"] == "mem123"
    assert memory["memory"] == "Test memory content"
    assert memory["user_id"] == "test_user"


@patch.dict(os.environ, {"MEM0_USER_ID": "test_user", "OPENSEARCH_HOST": "test.opensearch.amazonaws.com"})
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_list_memories(mock_opensearch, mock_mem0_client, mock_mem0_service_client, mock_tool):
    """Test list memories functionality uses the bound identity."""
    # Setup mocks
    mock_mem0_client.return_value = mock_mem0_service_client

    # Configure the mock_tool
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "list"},
    }.get(key, default)

    # Mock data for list_memories response
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
    mock_mem0_service_client.list_memories.return_value = list_response

    # Call the memory function
    result = mem0_memory.mem0_memory(tool=mock_tool)

    # Assertions
    assert result["status"] == "success"
    memories = json.loads(result["content"][0]["text"])
    assert memories[0]["id"] == "mem123"
    # The environment-configured identity was used
    mock_mem0_service_client.list_memories.assert_called_once_with("test_user", None)


@patch.dict(os.environ, {"MEM0_USER_ID": "test_user", "OPENSEARCH_HOST": "test.opensearch.amazonaws.com"})
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_retrieve_memories(mock_opensearch, mock_mem0_client, mock_mem0_service_client, mock_tool):
    """Test retrieve memories functionality uses the bound identity."""
    # Setup mocks
    mock_mem0_client.return_value = mock_mem0_service_client

    # Configure the mock_tool
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "retrieve", "query": "test query"},
    }.get(key, default)

    # Mock data for search_memories response
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
    mock_mem0_service_client.search_memories.return_value = retrieve_response

    # Call the memory function
    result = mem0_memory.mem0_memory(tool=mock_tool)

    # Assertions
    assert result["status"] == "success"
    memories = json.loads(result["content"][0]["text"])
    assert memories[0]["id"] == "mem123"
    # The environment-configured identity was used
    mock_mem0_service_client.search_memories.assert_called_once_with("test query", "test_user", None)


@patch.dict(
    os.environ,
    {"MEM0_USER_ID": "test_user", "OPENSEARCH_HOST": "test.opensearch.amazonaws.com", "BYPASS_TOOL_CONSENT": "true"},
)
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

    # Configure mocks - memory owned by the bound user
    mock_mem0_service_client.get_memory.return_value = {"id": "mem123", "user_id": "test_user"}
    mock_mem0_service_client.delete_memory.return_value = {"status": "success"}

    # Call the memory function
    result = mem0_memory.mem0_memory(tool=mock_tool)

    # Assertions
    assert result["status"] == "success"
    assert "Memory mem123 deleted successfully" in str(result["content"][0]["text"])

    # Verify correct functions were called
    mock_mem0_service_client.delete_memory.assert_called_once_with("mem123")


@patch.dict(os.environ, {"MEM0_USER_ID": "test_user", "OPENSEARCH_HOST": "test.opensearch.amazonaws.com"})
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
    mock_mem0_service_client.get_memory.return_value = {"id": "mem123", "user_id": "test_user"}
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
    mock_mem0_service_client.get_memory_history.return_value = history_response

    # Call the memory function
    result = mem0_memory.mem0_memory(tool=mock_tool)

    # Assertions
    assert result["status"] == "success"
    history = json.loads(result["content"][0]["text"])
    assert history[0]["id"] == "hist123"


def test_standalone_requires_identity_env_var(mock_tool):
    """Test the standalone function errors when neither MEM0_USER_ID nor MEM0_AGENT_ID is set."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "list"},
    }.get(key, default)

    with patch.dict(os.environ, {}, clear=True):
        result = mem0_memory.mem0_memory(tool=mock_tool)
        assert result["status"] == "error"
        assert "MEM0_USER_ID or MEM0_AGENT_ID" in result["content"][0]["text"]


def test_invalid_identity_env_var(mock_tool):
    """Test that an invalid MEM0_USER_ID env var is rejected.

    The agent cannot supply an identity, so it cannot reach this via the tool signature. An operator
    can still misconfigure the environment variable, so validation runs on that value.
    """
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "list"},
    }.get(key, default)

    invalid_identities = [
        "user name",  # Space
        "user@domain",  # @ symbol
        "user/path",  # Forward slash
        "user:name",  # Colon
        "a" * 129,  # Too long (over 128 chars)
        "",  # Empty
        "   ",  # Whitespace only
    ]

    for invalid_identity in invalid_identities:
        with patch.dict(os.environ, {"MEM0_USER_ID": invalid_identity}, clear=True):
            result = mem0_memory.mem0_memory(tool=mock_tool)
            assert result["status"] == "error", f"Invalid identity '{invalid_identity}' should be rejected"


@patch.dict(os.environ, {"MEM0_USER_ID": "bound_user", "OPENSEARCH_HOST": "test.opensearch.amazonaws.com"})
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_get_denies_cross_tenant(mock_opensearch, mock_mem0_client, mock_mem0_service_client, mock_tool):
    """Test get denies access to a memory owned by a different tenant."""
    mock_mem0_client.return_value = mock_mem0_service_client

    # Memory belongs to a different user
    mock_mem0_service_client.get_memory.return_value = {
        "id": "mem123",
        "memory": "Secret content",
        "user_id": "other_user",
    }

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "get", "memory_id": "mem123"},
    }.get(key, default)

    result = mem0_memory.mem0_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "Access denied" in result["content"][0]["text"]


@patch.dict(
    os.environ,
    {"MEM0_USER_ID": "bound_user", "OPENSEARCH_HOST": "test.opensearch.amazonaws.com", "BYPASS_TOOL_CONSENT": "true"},
)
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_delete_denies_cross_tenant(mock_opensearch, mock_mem0_client, mock_mem0_service_client, mock_tool):
    """Test delete denies removing a memory owned by a different tenant."""
    mock_mem0_client.return_value = mock_mem0_service_client

    # Memory belongs to a different user
    mock_mem0_service_client.get_memory.return_value = {"id": "mem123", "user_id": "other_user"}

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "delete", "memory_id": "mem123"},
    }.get(key, default)

    result = mem0_memory.mem0_memory(tool=mock_tool)

    assert result["status"] == "error"
    assert "Access denied" in result["content"][0]["text"]
    mock_mem0_service_client.delete_memory.assert_not_called()


@patch.dict(os.environ, {"MEM0_USER_ID": "test_user", "OPENSEARCH_HOST": "test.opensearch.amazonaws.com"})
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


@patch.dict(os.environ, {"MEM0_USER_ID": "test_user", "OPENSEARCH_HOST": "test.opensearch.amazonaws.com"})
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_action_specific_missing_params(mock_opensearch, mock_mem0_client, mock_tool):
    """Test missing action-specific parameters."""
    # Setup mock
    mock_mem0_client.return_value = MagicMock()

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


@patch.dict(os.environ, {"MEM0_USER_ID": "test-user"})
def test_missing_opensearch_host(mock_tool):
    """Test missing OpenSearch host defaults to FAISS."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "list"},
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


@patch("boto3.Session")
@patch("strands_tools.mem0_memory.Mem0Memory")
@patch("opensearchpy.OpenSearch")
def test_mem0_service_client_init(mock_opensearch, mock_mem0_memory, mock_session):
    """Test Mem0ServiceClient initialization across backends."""
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
    with patch.dict(os.environ, {"NEPTUNE_ANALYTICS_GRAPH_IDENTIFIER": "g-5aaaaa1234"}):
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


@patch.dict(os.environ, {"MEM0_USER_ID": "test_user", "MEM0_API_KEY": "test-api-key"})
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
            "metadata": {"category": "test"},
        },
    }.get(key, default)

    # Call the memory function
    result = mem0_memory.mem0_memory(tool=mock_tool)

    # Assertions
    assert result["status"] == "success"
    assert "Test memory content" in str(result["content"][0]["text"])


@patch.dict(os.environ, {"MEM0_USER_ID": "test_user"})
@patch("strands_tools.mem0_memory.Mem0Memory")
def test_faiss_client(mock_mem0_memory, mock_tool):
    """Test FAISS client functionality."""
    # Inject a mock faiss module into sys.modules
    sys.modules["faiss"] = MagicMock()
    try:
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
                "metadata": {"category": "test"},
            },
        }.get(key, default)

        # Call the memory function
        result = mem0_memory.mem0_memory(tool=mock_tool)

        # Assertions
        assert result["status"] == "success"
        assert "Test memory content" in str(result["content"][0]["text"])
    finally:
        del sys.modules["faiss"]


# --- Mem0MemoryTool class (per-principal binding) ---


def test_class_requires_identity():
    """Constructing the class without a user_id or agent_id raises."""
    with pytest.raises(ValueError, match="Either user_id or agent_id"):
        Mem0MemoryTool()


def test_class_rejects_invalid_identity():
    """The class validates the bound identity at construction."""
    with pytest.raises(ValueError, match="contains invalid characters"):
        Mem0MemoryTool(user_id="user name")


@patch.dict(os.environ, {"OPENSEARCH_HOST": "test.opensearch.amazonaws.com", "BYPASS_TOOL_CONSENT": "true"})
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_class_binds_identity(mock_opensearch, mock_mem0_client, mock_mem0_service_client, mock_tool):
    """The class uses its constructor identity, not any agent-supplied value."""
    mock_mem0_client.return_value = mock_mem0_service_client
    mock_mem0_service_client.list_memories.return_value = {"results": []}

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "list"},
    }.get(key, default)

    tool = Mem0MemoryTool(user_id="user_alice")
    result = tool.mem0_memory(tool=mock_tool)

    assert result["status"] == "success"
    # Query used the bound identity
    mock_mem0_service_client.list_memories.assert_called_once_with("user_alice", None)


@patch.dict(os.environ, {"OPENSEARCH_HOST": "test.opensearch.amazonaws.com", "BYPASS_TOOL_CONSENT": "true"})
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_class_record_uses_bound_identity(mock_opensearch, mock_mem0_client, mock_mem0_service_client, mock_tool):
    """A record stored through the class is attributed to the bound identity."""
    mock_mem0_client.return_value = mock_mem0_service_client
    mock_mem0_service_client.store_memory.return_value = [{"event": "store", "memory": "Bob's secret", "id": "mem123"}]

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "store", "content": "Bob's secret"},
    }.get(key, default)

    tool = Mem0MemoryTool(user_id="user_bob")
    result = tool.mem0_memory(tool=mock_tool)

    assert result["status"] == "success"
    # The record was attributed to the bound identity
    mock_mem0_service_client.store_memory.assert_called_once_with("Bob's secret", "user_bob", None, None)


@patch.dict(os.environ, {"OPENSEARCH_HOST": "test.opensearch.amazonaws.com", "BYPASS_TOOL_CONSENT": "true"})
@patch("strands_tools.mem0_memory.Mem0ServiceClient")
@patch("opensearchpy.OpenSearch")
def test_class_uses_agent_id(mock_opensearch, mock_mem0_client, mock_mem0_service_client, mock_tool):
    """Binding an agent_id (instead of user_id) routes operations by agent_id."""
    mock_mem0_client.return_value = mock_mem0_service_client
    mock_mem0_service_client.list_memories.return_value = {"results": []}

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "list"},
    }.get(key, default)

    tool = Mem0MemoryTool(agent_id="my_agent")
    result = tool.mem0_memory(tool=mock_tool)

    assert result["status"] == "success"
    mock_mem0_service_client.list_memories.assert_called_once_with(None, "my_agent")
