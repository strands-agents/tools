"""
Tests for the elasticsearch_memory tool.
"""

import inspect
import json
import os
from unittest import mock
from unittest.mock import MagicMock

import pytest
from strands import Agent

from src.strands_tools.elasticsearch_memory import ElasticsearchMemoryTool, elasticsearch_memory

ES_ENV_VARS = {
    "ELASTICSEARCH_CLOUD_ID": "test-cloud-id",
    "ELASTICSEARCH_API_KEY": "test-api-key",
    "ELASTICSEARCH_INDEX_NAME": "test_index",
    "ELASTICSEARCH_NAMESPACE": "test_namespace",
    "AWS_REGION": "us-east-1",
}

ES_URL_ENV_VARS = {
    "ELASTICSEARCH_URL": "https://test-cluster.es.region.aws.elastic.cloud:443",
    "ELASTICSEARCH_API_KEY": "test-api-key",
    "ELASTICSEARCH_INDEX_NAME": "test_index",
    "ELASTICSEARCH_NAMESPACE": "test_namespace",
    "AWS_REGION": "us-east-1",
}


@pytest.fixture(autouse=True)
def set_es_env_vars():
    """Auto-set ES environment variables for all tests."""
    with mock.patch.dict(os.environ, ES_ENV_VARS):
        yield


@pytest.fixture
def mock_elasticsearch_client():
    """Mock Elasticsearch client to avoid actual connections."""
    with mock.patch("src.strands_tools.elasticsearch_memory.Elasticsearch") as mock_es:
        # Create mock client instance
        mock_client = MagicMock()
        mock_es.return_value = mock_client

        # Configure ping to return True (successful connection)
        mock_client.ping.return_value = True

        # Configure indices.exists to return False initially (index doesn't exist)
        mock_client.indices.exists.return_value = False

        # Configure indices.create to return success
        mock_client.indices.create.return_value = {"acknowledged": True}

        yield {
            "elasticsearch_class": mock_es,
            "client": mock_client,
        }


@pytest.fixture
def mock_bedrock_client():
    """Mock Amazon Bedrock client for embeddings."""
    with mock.patch("boto3.client") as mock_boto_client:
        # Create mock bedrock runtime client
        mock_bedrock = MagicMock()

        # Configure boto3.client to return our mock for bedrock-runtime
        def client_side_effect(service, **kwargs):
            if service == "bedrock-runtime":
                return mock_bedrock
            return MagicMock()

        mock_boto_client.side_effect = client_side_effect

        # Configure embedding response
        mock_response = MagicMock()
        mock_response.__getitem__.return_value.read.return_value = json.dumps(
            {
                "embedding": [0.1] * 1024  # Mock 1024-dimensional embedding (Titan v2)
            }
        ).encode()
        mock_bedrock.invoke_model.return_value = mock_response

        yield {
            "boto_client": mock_boto_client,
            "bedrock": mock_bedrock,
        }


@pytest.fixture
def agent(mock_elasticsearch_client, mock_bedrock_client):
    """Create an agent with the direct elasticsearch_memory tool."""
    return Agent(tools=[elasticsearch_memory])


@pytest.fixture
def tool_config():
    """Configuration for constructing an ElasticsearchMemoryTool in tests."""
    return {
        "cloud_id": "test-cloud-id",
        "api_key": "test-api-key",
        "index_name": "test_index",
        "namespace": "test_namespace",
        "region": "us-east-1",
    }


# --- Security Tests ---


def test_credential_and_namespace_params_not_in_standalone_signature():
    """IDOR guard: the standalone tool must not expose connection, credential, or namespace params.

    Regression test for the IDOR where an LLM-supplied namespace/api_key/cloud_id could redirect the
    memory layer at another tenant or cluster (or authenticate as a different principal). These must
    be environment-only.
    """
    param_names = set(inspect.signature(elasticsearch_memory).parameters.keys())
    for forbidden in ["es_url", "cloud_id", "api_key", "index_name", "namespace", "embedding_model", "region"]:
        assert forbidden not in param_names, f"'{forbidden}' must not be an LLM-controllable tool parameter"


def test_namespace_params_not_in_class_tool_signature():
    """IDOR guard: ElasticsearchMemoryTool.elasticsearch_memory must not expose namespace/creds."""
    param_names = set(inspect.signature(ElasticsearchMemoryTool.elasticsearch_memory).parameters.keys())
    for forbidden in ["es_url", "cloud_id", "api_key", "index_name", "namespace"]:
        assert forbidden not in param_names


def test_missing_required_params(mock_elasticsearch_client, mock_bedrock_client):
    """Test tool with missing required environment variables."""
    agent = Agent(tools=[elasticsearch_memory])

    # Test missing both cloud_id and es_url (no env vars set)
    with mock.patch.dict(os.environ, {"ELASTICSEARCH_API_KEY": "test-api-key"}, clear=True):
        result = agent.tool.elasticsearch_memory(action="record", content="test")
        assert result["status"] == "error"
        assert "Either cloud_id or es_url is required" in result["content"][0]["text"]

    # Test missing api_key
    with mock.patch.dict(os.environ, {"ELASTICSEARCH_CLOUD_ID": "test-cloud-id"}, clear=True):
        result = agent.tool.elasticsearch_memory(action="record", content="test")
        assert result["status"] == "error"
        assert "api_key is required" in result["content"][0]["text"]


def test_connection_failure(mock_elasticsearch_client, mock_bedrock_client):
    """Test tool with connection failure."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure ping to return False (connection failure)
    mock_elasticsearch_client["client"].ping.return_value = False

    with mock.patch.dict(os.environ, ES_ENV_VARS):
        result = agent.tool.elasticsearch_memory(action="record", content="test")

    assert result["status"] == "error"
    assert "Unable to connect to Elasticsearch cluster" in result["content"][0]["text"]


def test_index_creation(mock_elasticsearch_client, mock_bedrock_client):
    """Test that index is created with proper mappings."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure mock responses
    mock_elasticsearch_client["client"].index.return_value = {"result": "created", "_id": "test_memory_id"}

    with mock.patch.dict(os.environ, ES_ENV_VARS):
        agent.tool.elasticsearch_memory(action="record", content="Test content")

    # Verify index creation was called
    mock_elasticsearch_client["client"].indices.create.assert_called_once()

    # Get the call arguments
    call_args = mock_elasticsearch_client["client"].indices.create.call_args
    assert call_args[1]["index"] == "test_index"

    # Verify mapping structure
    mapping = call_args[1]["body"]
    assert "mappings" in mapping
    assert "properties" in mapping["mappings"]

    properties = mapping["mappings"]["properties"]
    assert "content" in properties
    assert "embedding" in properties
    assert "namespace" in properties
    assert "memory_id" in properties
    assert "timestamp" in properties
    assert "metadata" in properties

    # Verify embedding field configuration
    embedding_config = properties["embedding"]
    assert embedding_config["type"] == "dense_vector"
    assert embedding_config["dims"] == 1024  # Titan v2 returns 1024 dimensions
    assert embedding_config["similarity"] == "cosine"


def test_record_memory(mock_elasticsearch_client, mock_bedrock_client):
    """Test recording a memory."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure mock responses
    mock_elasticsearch_client["client"].index.return_value = {"result": "created", "_id": "test_memory_id"}

    # Call the tool
    with mock.patch.dict(os.environ, ES_ENV_VARS):
        result = agent.tool.elasticsearch_memory(
            action="record", content="Test memory content", metadata={"category": "test"}
        )

    # Verify success response
    assert result["status"] == "success"
    assert "Memory stored successfully" in result["content"][0]["text"]

    # Verify Elasticsearch index was called
    mock_elasticsearch_client["client"].index.assert_called_once()

    # Verify embedding generation was called
    mock_bedrock_client["bedrock"].invoke_model.assert_called_once()


def test_retrieve_memories(mock_elasticsearch_client, mock_bedrock_client):
    """Test retrieving memories with semantic search."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure mock search response
    mock_elasticsearch_client["client"].search.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "memory_id": "mem_123",
                        "content": "Test content",
                        "timestamp": "2023-01-01T00:00:00Z",
                        "metadata": {},
                    },
                    "_score": 0.95,
                }
            ],
            "total": {"value": 1},
            "max_score": 0.95,
        }
    }

    # Call the tool
    result = agent.tool.elasticsearch_memory(action="retrieve", query="test query", max_results=5)

    # Verify success response
    assert result["status"] == "success"
    assert "Memories retrieved successfully" in result["content"][0]["text"]

    # Verify search was called with k-NN query
    mock_elasticsearch_client["client"].search.assert_called_once()
    search_call = mock_elasticsearch_client["client"].search.call_args[1]
    assert "knn" in search_call["body"]
    assert search_call["body"]["knn"]["field"] == "embedding"

    # Verify embedding generation for query
    mock_bedrock_client["bedrock"].invoke_model.assert_called_once()


def test_list_memories(mock_elasticsearch_client, mock_bedrock_client):
    """Test listing all memories."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure mock search response
    mock_elasticsearch_client["client"].search.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "memory_id": "mem_123",
                        "content": "Test content 1",
                        "timestamp": "2023-01-01T00:00:00Z",
                        "metadata": {},
                    }
                },
                {
                    "_source": {
                        "memory_id": "mem_456",
                        "content": "Test content 2",
                        "timestamp": "2023-01-02T00:00:00Z",
                        "metadata": {},
                    }
                },
            ],
            "total": {"value": 2},
        }
    }

    # Call the tool
    result = agent.tool.elasticsearch_memory(action="list", max_results=10)

    # Verify success response
    assert result["status"] == "success"
    assert "Memories listed successfully" in result["content"][0]["text"]

    # Verify search was called with proper query
    mock_elasticsearch_client["client"].search.assert_called_once()
    search_call = mock_elasticsearch_client["client"].search.call_args[1]
    assert search_call["body"]["query"]["term"]["namespace"] == "test_namespace"
    assert search_call["body"]["sort"] == [{"timestamp": {"order": "desc"}}]


def test_get_memory(mock_elasticsearch_client, mock_bedrock_client):
    """Test getting a specific memory by ID."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure mock search response (uses search for namespace enforcement)
    mock_elasticsearch_client["client"].search.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "memory_id": "mem_123",
                        "content": "Test content",
                        "timestamp": "2023-01-01T00:00:00Z",
                        "metadata": {"category": "test"},
                        "namespace": "test_namespace",
                    }
                }
            ],
            "total": {"value": 1},
        }
    }

    # Call the tool
    result = agent.tool.elasticsearch_memory(action="get", memory_id="mem_123")

    # Verify success response
    assert result["status"] == "success"
    assert "Memory retrieved successfully" in result["content"][0]["text"]

    # Verify search was called with both memory_id and namespace for security
    mock_elasticsearch_client["client"].search.assert_called_once()
    search_call = mock_elasticsearch_client["client"].search.call_args[1]
    query = search_call["body"]["query"]["bool"]["must"]
    assert {"term": {"memory_id": "mem_123"}} in query
    assert {"term": {"namespace": "test_namespace"}} in query


def test_delete_memory(mock_elasticsearch_client, mock_bedrock_client):
    """Test deleting a memory."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure mock delete_by_query response (atomic delete with namespace constraint)
    mock_elasticsearch_client["client"].delete_by_query.return_value = {"deleted": 1}

    # Call the tool
    result = agent.tool.elasticsearch_memory(action="delete", memory_id="mem_123")

    # Verify success response
    assert result["status"] == "success"
    assert "Memory deleted successfully: mem_123" in result["content"][0]["text"]

    # Verify delete_by_query was called with both memory_id and namespace
    mock_elasticsearch_client["client"].delete_by_query.assert_called_once()
    call_args = mock_elasticsearch_client["client"].delete_by_query.call_args[1]
    query = call_args["body"]["query"]["bool"]["must"]
    assert {"term": {"memory_id": "mem_123"}} in query
    assert {"term": {"namespace": "test_namespace"}} in query


def test_unsupported_action(mock_elasticsearch_client, mock_bedrock_client):
    """Test tool with an unsupported action."""
    agent = Agent(tools=[elasticsearch_memory])

    result = agent.tool.elasticsearch_memory(action="unsupported_action")

    # Verify error response
    assert result["status"] == "error"
    assert "is not supported" in result["content"][0]["text"]
    assert "record" in result["content"][0]["text"]
    assert "retrieve" in result["content"][0]["text"]


def test_missing_required_parameters(mock_elasticsearch_client, mock_bedrock_client):
    """Test tool with missing required parameters."""
    agent = Agent(tools=[elasticsearch_memory])

    # Test record action without content
    result = agent.tool.elasticsearch_memory(action="record")

    # Verify error response
    assert result["status"] == "error"
    assert "parameters are required" in result["content"][0]["text"]
    assert "content" in result["content"][0]["text"]

    # Test retrieve action without query
    result = agent.tool.elasticsearch_memory(action="retrieve")

    # Verify error response
    assert result["status"] == "error"
    assert "parameters are required" in result["content"][0]["text"]
    assert "query" in result["content"][0]["text"]

    # Test get action without memory_id
    result = agent.tool.elasticsearch_memory(action="get")

    # Verify error response
    assert result["status"] == "error"
    assert "parameters are required" in result["content"][0]["text"]
    assert "memory_id" in result["content"][0]["text"]


def test_elasticsearch_api_error_handling(mock_elasticsearch_client, mock_bedrock_client):
    """Test handling of Elasticsearch API errors."""
    agent = Agent(tools=[elasticsearch_memory])

    # Set up mock to raise an exception
    mock_elasticsearch_client["client"].index.side_effect = Exception("Elasticsearch error")

    # Call the tool
    result = agent.tool.elasticsearch_memory(action="record", content="Test content")

    # Verify error response
    assert result["status"] == "error"
    assert "API error" in result["content"][0]["text"]
    assert "Elasticsearch error" in result["content"][0]["text"]


def test_bedrock_api_error_handling(mock_elasticsearch_client, mock_bedrock_client):
    """Test handling of Bedrock API errors."""
    agent = Agent(tools=[elasticsearch_memory])

    # Set up mock to raise an exception
    mock_bedrock_client["bedrock"].invoke_model.side_effect = Exception("Bedrock error")

    # Call the tool
    result = agent.tool.elasticsearch_memory(action="record", content="Test content")

    # Verify error response
    assert result["status"] == "error"
    assert "API error" in result["content"][0]["text"]
    assert "Embedding generation failed" in result["content"][0]["text"]


def test_memory_not_found(mock_elasticsearch_client, mock_bedrock_client):
    """Test handling when memory is not found."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure mock search to return empty results (memory not found in namespace)
    mock_elasticsearch_client["client"].search.return_value = {
        "hits": {
            "hits": [],
            "total": {"value": 0},
        }
    }

    # Call the tool
    result = agent.tool.elasticsearch_memory(action="get", memory_id="nonexistent")

    # Verify error response
    assert result["status"] == "error"
    assert "Memory nonexistent not found in namespace test_namespace" in result["content"][0]["text"]


def test_namespace_filtering(mock_elasticsearch_client, mock_bedrock_client):
    """Test that memories are properly filtered by the environment-configured namespace."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure mock search to return empty results (memory not in this namespace)
    mock_elasticsearch_client["client"].search.return_value = {
        "hits": {
            "hits": [],
            "total": {"value": 0},
        }
    }

    # Call the tool
    result = agent.tool.elasticsearch_memory(action="get", memory_id="mem_123")

    # Verify error response
    assert result["status"] == "error"
    assert "not found in namespace test_namespace" in result["content"][0]["text"]

    # Verify search was called with both memory_id and namespace
    mock_elasticsearch_client["client"].search.assert_called_once()
    search_call = mock_elasticsearch_client["client"].search.call_args[1]
    query = search_call["body"]["query"]["bool"]["must"]
    assert {"term": {"memory_id": "mem_123"}} in query
    assert {"term": {"namespace": "test_namespace"}} in query


def test_pagination_support(mock_elasticsearch_client, mock_bedrock_client):
    """Test pagination support in list and retrieve operations."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure mock search response with pagination
    mock_elasticsearch_client["client"].search.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "memory_id": "mem_123",
                        "content": "Test content",
                        "timestamp": "2023-01-01T00:00:00Z",
                        "metadata": {},
                    }
                }
            ],
            "total": {"value": 20},  # More results available
        }
    }

    # Test list with pagination
    agent.tool.elasticsearch_memory(action="list", max_results=5, next_token="10")

    # Verify search was called with correct offset
    search_call = mock_elasticsearch_client["client"].search.call_args[1]
    assert search_call["body"]["from"] == 10
    assert search_call["body"]["size"] == 5


def test_environment_variable_defaults(mock_elasticsearch_client, mock_bedrock_client):
    """Test that environment variables are used for configuration."""
    agent = Agent(tools=[elasticsearch_memory])

    with mock.patch.dict(
        os.environ,
        {
            "ELASTICSEARCH_CLOUD_ID": "env-cloud-id",
            "ELASTICSEARCH_API_KEY": "env-api-key",
            "ELASTICSEARCH_INDEX_NAME": "env_index",
            "ELASTICSEARCH_NAMESPACE": "env_namespace",
            "ELASTICSEARCH_EMBEDDING_MODEL": "env_model",
            "AWS_REGION": "env_region",
        },
    ):
        # Configure mock responses
        mock_elasticsearch_client["client"].index.return_value = {"result": "created", "_id": "test_memory_id"}

        # Call tool without explicit parameters (should use env vars)
        result = agent.tool.elasticsearch_memory(action="record", content="Test content")

        # Verify success (means env vars were used correctly)
        assert result["status"] == "success"
        assert "Memory stored successfully" in result["content"][0]["text"]

        # Verify the stored document used the environment-configured namespace and index
        index_call = mock_elasticsearch_client["client"].index.call_args[1]
        assert index_call["index"] == "env_index"
        assert index_call["body"]["namespace"] == "env_namespace"


def test_es_url_connection(mock_elasticsearch_client, mock_bedrock_client):
    """Test using ELASTICSEARCH_URL instead of ELASTICSEARCH_CLOUD_ID for connection."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure mock responses
    mock_elasticsearch_client["client"].index.return_value = {"result": "created", "_id": "test_memory_id"}

    # Override env vars to use URL instead of cloud_id
    with mock.patch.dict(os.environ, ES_URL_ENV_VARS, clear=True):
        result = agent.tool.elasticsearch_memory(action="record", content="Test memory content")

    # Verify success response
    assert result["status"] == "success"
    assert "Memory stored successfully" in result["content"][0]["text"]

    # Verify Elasticsearch was initialized with URL
    mock_elasticsearch_client["elasticsearch_class"].assert_called_once()
    call_args = mock_elasticsearch_client["elasticsearch_class"].call_args[1]
    assert call_args["hosts"] == ["https://test-cluster.es.region.aws.elastic.cloud:443"]
    assert call_args["api_key"] == "test-api-key"


def test_custom_embedding_model(mock_elasticsearch_client, mock_bedrock_client):
    """Test using custom embedding model from the environment."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure mock responses
    mock_elasticsearch_client["client"].index.return_value = {"result": "created", "_id": "test_memory_id"}

    # Call tool with custom embedding model via env var
    with mock.patch.dict(os.environ, {"ELASTICSEARCH_EMBEDDING_MODEL": "amazon.titan-embed-text-v1:0"}):
        result = agent.tool.elasticsearch_memory(action="record", content="Test memory content")

    # Verify success response
    assert result["status"] == "success"
    assert "Memory stored successfully" in result["content"][0]["text"]

    # Verify Bedrock was called with custom model
    mock_bedrock_client["bedrock"].invoke_model.assert_called_once()
    call_args = mock_bedrock_client["bedrock"].invoke_model.call_args
    assert call_args[1]["modelId"] == "amazon.titan-embed-text-v1:0"


def test_batch_operations(mock_elasticsearch_client, mock_bedrock_client):
    """Test storing multiple related memories in batch."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure mock responses
    mock_elasticsearch_client["client"].index.return_value = {"result": "created", "_id": "test_memory_id"}

    # Store multiple related memories
    memories = ["User likes Italian food", "User is allergic to nuts", "User prefers evening meetings"]

    results = []
    for content in memories:
        result = agent.tool.elasticsearch_memory(
            action="record",
            content=content,
            metadata={"batch": "user_preferences", "category": "preferences"},
        )
        results.append(result)

    # Verify all operations succeeded
    for result in results:
        assert result["status"] == "success"
        assert "Memory stored successfully" in result["content"][0]["text"]

    # Verify correct number of calls were made
    assert mock_elasticsearch_client["client"].index.call_count == len(memories)


def test_error_handling_scenarios(mock_elasticsearch_client, mock_bedrock_client):
    """Test comprehensive error handling scenarios."""
    agent = Agent(tools=[elasticsearch_memory])

    # Test connection errors
    mock_elasticsearch_client["client"].ping.return_value = False
    result = agent.tool.elasticsearch_memory(action="record", content="test")
    assert result["status"] == "error"
    assert "Unable to connect to Elasticsearch cluster" in result["content"][0]["text"]

    # Reset ping to return True for subsequent tests
    mock_elasticsearch_client["client"].ping.return_value = True

    # Test Elasticsearch API errors
    mock_elasticsearch_client["client"].index.side_effect = Exception("Elasticsearch connection failed")
    result = agent.tool.elasticsearch_memory(action="record", content="test")
    assert result["status"] == "error"
    assert "API error" in result["content"][0]["text"]

    # Reset side effect
    mock_elasticsearch_client["client"].index.side_effect = None

    # Test Bedrock API errors
    mock_bedrock_client["bedrock"].invoke_model.side_effect = Exception("Bedrock access denied")
    result = agent.tool.elasticsearch_memory(action="record", content="test")
    assert result["status"] == "error"
    assert "Embedding generation failed" in result["content"][0]["text"]


def test_metadata_usage_scenarios(mock_elasticsearch_client, mock_bedrock_client):
    """Test various metadata usage patterns."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure mock responses
    mock_elasticsearch_client["client"].index.return_value = {"result": "created", "_id": "test_memory_id"}

    # Test structured metadata
    structured_metadata = {
        "type": "deadline",
        "project": "project_alpha",
        "priority": "high",
        "due_date": "2024-02-01",
        "assigned_to": ["alice", "bob"],
    }

    result = agent.tool.elasticsearch_memory(
        action="record", content="Important project deadline", metadata=structured_metadata
    )

    assert result["status"] == "success"
    assert "Memory stored successfully" in result["content"][0]["text"]

    # Verify the index call included metadata
    mock_elasticsearch_client["client"].index.assert_called()
    call_args = mock_elasticsearch_client["client"].index.call_args[1]
    assert call_args["body"]["metadata"] == structured_metadata


def test_performance_scenarios(mock_elasticsearch_client, mock_bedrock_client):
    """Test performance-related scenarios like pagination."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure mock search response with pagination
    mock_elasticsearch_client["client"].search.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "memory_id": f"mem_{i}",
                        "content": f"Test content {i}",
                        "timestamp": "2023-01-01T00:00:00Z",
                        "metadata": {},
                    }
                }
                for i in range(5)
            ],
            "total": {"value": 25},  # More results available
        }
    }

    # Test pagination with next_token
    result = agent.tool.elasticsearch_memory(action="list", max_results=5, next_token="10")

    assert result["status"] == "success"
    assert "Memories listed successfully" in result["content"][0]["text"]

    # Verify pagination parameters were used
    search_call = mock_elasticsearch_client["client"].search.call_args[1]
    assert search_call["body"]["from"] == 10
    assert search_call["body"]["size"] == 5


def test_troubleshooting_scenarios(mock_elasticsearch_client, mock_bedrock_client):
    """Test troubleshooting scenarios mentioned in documentation."""
    agent = Agent(tools=[elasticsearch_memory])

    # Test index creation failure
    mock_elasticsearch_client["client"].indices.create.side_effect = Exception("Index creation failed")
    result = agent.tool.elasticsearch_memory(action="record", content="test")
    assert result["status"] == "error"
    assert "Failed to create index" in result["content"][0]["text"]

    # Reset side effect
    mock_elasticsearch_client["client"].indices.create.side_effect = None

    # Test authentication errors (simulated by connection failure)
    mock_elasticsearch_client["client"].ping.return_value = False
    result = agent.tool.elasticsearch_memory(action="record", content="test")
    assert result["status"] == "error"
    assert "Unable to connect to Elasticsearch cluster" in result["content"][0]["text"]


def test_invalid_namespace_env_var(mock_elasticsearch_client, mock_bedrock_client):
    """Test that an invalid ELASTICSEARCH_NAMESPACE env var is rejected.

    The agent cannot supply a namespace, so injection cannot reach it via the tool signature. An
    operator can still misconfigure the environment variable, so validation runs on that value.
    """
    agent = Agent(tools=[elasticsearch_memory])

    invalid_namespaces = [
        "user.name",  # Dots not allowed
        "user@domain",  # @ symbol
        "user$name",  # $ symbol
        "user name",  # Space
        "user/path",  # Forward slash
        "user:name",  # Colon
        "a" * 65,  # Too long (over 64 chars)
        "",  # Empty
        "   ",  # Whitespace only
    ]

    for invalid_namespace in invalid_namespaces:
        with mock.patch.dict(os.environ, {"ELASTICSEARCH_NAMESPACE": invalid_namespace}):
            result = agent.tool.elasticsearch_memory(action="list")
            assert result["status"] == "error", f"Invalid namespace '{invalid_namespace}' should be rejected"
            assert "Invalid namespace" in result["content"][0]["text"]


def test_valid_namespaces_accepted(mock_elasticsearch_client, mock_bedrock_client):
    """Test that valid namespaces (via env var) are accepted."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure mock responses
    mock_elasticsearch_client["client"].search.return_value = {
        "hits": {
            "hits": [],
            "total": {"value": 0},
        }
    }

    valid_namespaces = [
        "default",
        "user_123",
        "tenant-abc",
        "MyNamespace",
        "a",
        "A" * 64,  # Max length
    ]

    for valid_namespace in valid_namespaces:
        with mock.patch.dict(os.environ, {"ELASTICSEARCH_NAMESPACE": valid_namespace}):
            result = agent.tool.elasticsearch_memory(action="list")
            assert result["status"] == "success", f"Valid namespace '{valid_namespace}' should be accepted"


def test_delete_memory_namespace_enforcement(mock_elasticsearch_client, mock_bedrock_client):
    """Test that delete enforces namespace atomically (no TOCTOU)."""
    agent = Agent(tools=[elasticsearch_memory])

    # Configure delete_by_query to return 0 deleted (memory not in namespace)
    mock_elasticsearch_client["client"].delete_by_query.return_value = {"deleted": 0}

    result = agent.tool.elasticsearch_memory(action="delete", memory_id="mem_123")

    # Should fail because memory not found in the requested namespace
    assert result["status"] == "error"
    assert "not found in namespace test_namespace" in result["content"][0]["text"]

    # Verify delete_by_query was called with namespace constraint
    mock_elasticsearch_client["client"].delete_by_query.assert_called_once()
    call_args = mock_elasticsearch_client["client"].delete_by_query.call_args[1]
    query = call_args["body"]["query"]["bool"]["must"]
    assert {"term": {"memory_id": "mem_123"}} in query
    assert {"term": {"namespace": "test_namespace"}} in query


# --- ElasticsearchMemoryTool class (per-principal binding) ---


def test_class_requires_api_key(mock_elasticsearch_client, mock_bedrock_client):
    """Constructing the class without an api_key raises."""
    from src.strands_tools.elasticsearch_memory import ElasticsearchValidationError

    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ElasticsearchValidationError, match="api_key is required"):
            ElasticsearchMemoryTool(cloud_id="test-cloud-id")


def test_class_requires_connection(mock_elasticsearch_client, mock_bedrock_client):
    """Constructing the class without cloud_id or es_url raises."""
    from src.strands_tools.elasticsearch_memory import ElasticsearchValidationError

    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ElasticsearchValidationError, match="Either cloud_id or es_url is required"):
            ElasticsearchMemoryTool(api_key="test-api-key")


def test_class_rejects_invalid_namespace(mock_elasticsearch_client, mock_bedrock_client, tool_config):
    """The class validates the bound namespace at construction."""
    from src.strands_tools.elasticsearch_memory import ElasticsearchValidationError

    bad_config = {**tool_config, "namespace": "user$name"}
    with pytest.raises(ElasticsearchValidationError, match="Invalid namespace"):
        ElasticsearchMemoryTool(**bad_config)


def test_class_binds_namespace(mock_elasticsearch_client, mock_bedrock_client, tool_config):
    """The class uses its constructor namespace, not any agent-supplied value."""
    mock_elasticsearch_client["client"].search.return_value = {"hits": {"hits": [], "total": {"value": 0}}}

    tool = ElasticsearchMemoryTool(**{**tool_config, "namespace": "user_alice"})
    agent = Agent(tools=[tool.elasticsearch_memory])

    result = agent.tool.elasticsearch_memory(action="get", memory_id="mem_123")

    assert result["status"] == "error"
    assert "not found in namespace user_alice" in result["content"][0]["text"]

    # Query used the bound namespace
    search_call = mock_elasticsearch_client["client"].search.call_args[1]
    query = search_call["body"]["query"]["bool"]["must"]
    assert {"term": {"namespace": "user_alice"}} in query


def test_class_record_uses_bound_namespace(mock_elasticsearch_client, mock_bedrock_client, tool_config):
    """A record stored through the class lands in the bound namespace and index."""
    mock_elasticsearch_client["client"].index.return_value = {"result": "created", "_id": "test_memory_id"}

    tool = ElasticsearchMemoryTool(**{**tool_config, "namespace": "user_bob", "index_name": "bob_index"})
    agent = Agent(tools=[tool.elasticsearch_memory])

    result = agent.tool.elasticsearch_memory(action="record", content="Bob's secret")

    assert result["status"] == "success"

    # The stored document carries the bound namespace and index
    index_call = mock_elasticsearch_client["client"].index.call_args[1]
    assert index_call["index"] == "bob_index"
    assert index_call["body"]["namespace"] == "user_bob"


def test_class_namespace_falls_back_to_env(mock_elasticsearch_client, mock_bedrock_client):
    """When namespace is not passed to the constructor, it falls back to ELASTICSEARCH_NAMESPACE."""
    mock_elasticsearch_client["client"].search.return_value = {"hits": {"hits": [], "total": {"value": 0}}}

    with mock.patch.dict(os.environ, {**ES_ENV_VARS, "ELASTICSEARCH_NAMESPACE": "env_tenant"}):
        tool = ElasticsearchMemoryTool(cloud_id="test-cloud-id", api_key="test-api-key")
        agent = Agent(tools=[tool.elasticsearch_memory])
        agent.tool.elasticsearch_memory(action="get", memory_id="mem_123")

    search_call = mock_elasticsearch_client["client"].search.call_args[1]
    query = search_call["body"]["query"]["bool"]["must"]
    assert {"term": {"namespace": "env_tenant"}} in query
