"""
Tests for the elasticsearch_memory tool.
"""

import json
import os
from unittest import mock
from unittest.mock import MagicMock

import pytest
from strands import Agent

from src.strands_tools.elasticsearch_memory import elasticsearch_memory


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
def config():
    """Configuration parameters for testing."""
    return {
        "cloud_id": "test-cloud-id",
        "api_key": "test-api-key",
        "index_name": "test_index",
        "namespace": "test_namespace",
        "region": "us-east-1"
    }


def test_missing_required_params(mock_elasticsearch_client, mock_bedrock_client):
    """Test tool with missing required parameters."""
    # Test missing both cloud_id and es_url
    result = elasticsearch_memory(action="record", content="test", api_key="test-api-key")
    assert result["status"] == "error"
    assert "Either cloud_id or es_url is required" in result["content"][0]["text"]

    # Test missing api_key
    result = elasticsearch_memory(action="record", content="test", cloud_id="test-cloud-id")
    assert result["status"] == "error"
    assert "api_key is required" in result["content"][0]["text"]


def test_connection_failure(mock_elasticsearch_client, mock_bedrock_client):
    """Test tool with connection failure."""
    # Configure ping to return False (connection failure)
    mock_elasticsearch_client["client"].ping.return_value = False

    result = elasticsearch_memory(
        action="record",
        content="test",
        cloud_id="test-cloud-id",
        api_key="test-api-key"
    )

    assert result["status"] == "error"
    assert "Unable to connect to Elasticsearch cluster" in result["content"][0]["text"]


def test_index_creation(mock_elasticsearch_client, mock_bedrock_client, config):
    """Test that index is created with proper mappings."""
    # Configure mock responses
    mock_elasticsearch_client["client"].index.return_value = {"result": "created", "_id": "test_memory_id"}

    elasticsearch_memory(
        action="record",
        content="Test content",
        **config
    )

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


def test_record_memory(mock_elasticsearch_client, mock_bedrock_client, config):
    """Test recording a memory."""
    # Configure mock responses
    mock_elasticsearch_client["client"].index.return_value = {"result": "created", "_id": "test_memory_id"}

    # Call the tool
    result = elasticsearch_memory(
        action="record",
        content="Test memory content",
        metadata={"category": "test"},
        **config
    )

    # Verify success response
    assert result["status"] == "success"
    assert "Memory stored successfully" in result["content"][0]["text"]

    # Verify Elasticsearch index was called
    mock_elasticsearch_client["client"].index.assert_called_once()

    # Verify embedding generation was called
    mock_bedrock_client["bedrock"].invoke_model.assert_called_once()


def test_retrieve_memories(mock_elasticsearch_client, mock_bedrock_client, config):
    """Test retrieving memories with semantic search."""
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
    result = elasticsearch_memory(
        action="retrieve",
        query="test query",
        max_results=5,
        **config
    )

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


def test_list_memories(mock_elasticsearch_client, mock_bedrock_client, config):
    """Test listing all memories."""
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
    result = elasticsearch_memory(
        action="list",
        max_results=10,
        **config
    )

    # Verify success response
    assert result["status"] == "success"
    assert "Memories listed successfully" in result["content"][0]["text"]

    # Verify search was called with proper query
    mock_elasticsearch_client["client"].search.assert_called_once()
    search_call = mock_elasticsearch_client["client"].search.call_args[1]
    assert search_call["body"]["query"]["term"]["namespace"] == "test_namespace"
    assert search_call["body"]["sort"] == [{"timestamp": {"order": "desc"}}]


def test_get_memory(mock_elasticsearch_client, mock_bedrock_client, config):
    """Test getting a specific memory by ID."""
    # Configure mock get response
    mock_elasticsearch_client["client"].get.return_value = {
        "_source": {
            "memory_id": "mem_123",
            "content": "Test content",
            "timestamp": "2023-01-01T00:00:00Z",
            "metadata": {"category": "test"},
            "namespace": "test_namespace",
        }
    }

    # Call the tool
    result = elasticsearch_memory(
        action="get",
        memory_id="mem_123",
        **config
    )

    # Verify success response
    assert result["status"] == "success"
    assert "Memory retrieved successfully" in result["content"][0]["text"]

    # Verify get was called
    mock_elasticsearch_client["client"].get.assert_called_once_with(index="test_index", id="mem_123")


def test_delete_memory(mock_elasticsearch_client, mock_bedrock_client, config):
    """Test deleting a memory."""
    # Configure mock responses
    mock_elasticsearch_client["client"].get.return_value = {
        "_source": {
            "memory_id": "mem_123",
            "content": "Test content",
            "timestamp": "2023-01-01T00:00:00Z",
            "metadata": {},
            "namespace": "test_namespace",
        }
    }
    mock_elasticsearch_client["client"].delete.return_value = {"result": "deleted"}

    # Call the tool
    result = elasticsearch_memory(
        action="delete",
        memory_id="mem_123",
        **config
    )

    # Verify success response
    assert result["status"] == "success"
    assert "Memory deleted successfully: mem_123" in result["content"][0]["text"]

    # Verify delete was called
    mock_elasticsearch_client["client"].delete.assert_called_once_with(index="test_index", id="mem_123")


def test_unsupported_action(mock_elasticsearch_client, mock_bedrock_client, config):
    """Test tool with an unsupported action."""
    result = elasticsearch_memory(action="unsupported_action", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "is not supported" in result["content"][0]["text"]
    assert "record" in result["content"][0]["text"]
    assert "retrieve" in result["content"][0]["text"]


def test_missing_required_parameters(mock_elasticsearch_client, mock_bedrock_client, config):
    """Test tool with missing required parameters."""
    # Test record action without content
    result = elasticsearch_memory(action="record", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "parameters are required" in result["content"][0]["text"]
    assert "content" in result["content"][0]["text"]

    # Test retrieve action without query
    result = elasticsearch_memory(action="retrieve", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "parameters are required" in result["content"][0]["text"]
    assert "query" in result["content"][0]["text"]

    # Test get action without memory_id
    result = elasticsearch_memory(action="get", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "parameters are required" in result["content"][0]["text"]
    assert "memory_id" in result["content"][0]["text"]


def test_elasticsearch_api_error_handling(mock_elasticsearch_client, mock_bedrock_client, config):
    """Test handling of Elasticsearch API errors."""
    # Set up mock to raise an exception
    mock_elasticsearch_client["client"].index.side_effect = Exception("Elasticsearch error")

    # Call the tool
    result = elasticsearch_memory(action="record", content="Test content", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "API error" in result["content"][0]["text"]
    assert "Elasticsearch error" in result["content"][0]["text"]


def test_bedrock_api_error_handling(mock_elasticsearch_client, mock_bedrock_client, config):
    """Test handling of Bedrock API errors."""
    # Set up mock to raise an exception
    mock_bedrock_client["bedrock"].invoke_model.side_effect = Exception("Bedrock error")

    # Call the tool
    result = elasticsearch_memory(action="record", content="Test content", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "API error" in result["content"][0]["text"]
    assert "Embedding generation failed" in result["content"][0]["text"]


def test_memory_not_found(mock_elasticsearch_client, mock_bedrock_client, config):
    """Test handling when memory is not found."""
    from elasticsearch import NotFoundError

    # Configure mock to raise NotFoundError
    mock_elasticsearch_client["client"].get.side_effect = NotFoundError("404", "not_found_exception", {})

    # Call the tool
    result = elasticsearch_memory(action="get", memory_id="nonexistent", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "Memory nonexistent not found" in result["content"][0]["text"]


def test_namespace_validation(mock_elasticsearch_client, mock_bedrock_client, config):
    """Test that memories are properly filtered by namespace."""
    # Configure mock get response with wrong namespace
    mock_elasticsearch_client["client"].get.return_value = {
        "_source": {"memory_id": "mem_123", "content": "Test content", "namespace": "wrong_namespace"}
    }

    # Call the tool
    result = elasticsearch_memory(action="get", memory_id="mem_123", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "not found in namespace test_namespace" in result["content"][0]["text"]


def test_pagination_support(mock_elasticsearch_client, mock_bedrock_client, config):
    """Test pagination support in list and retrieve operations."""
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
    elasticsearch_memory(action="list", max_results=5, next_token="10", **config)

    # Verify search was called with correct offset
    search_call = mock_elasticsearch_client["client"].search.call_args[1]
    assert search_call["body"]["from"] == 10
    assert search_call["body"]["size"] == 5


def test_environment_variable_defaults(mock_elasticsearch_client, mock_bedrock_client):
    """Test that environment variables are used for defaults."""
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
        result = elasticsearch_memory(action="record", content="Test content")

        # Verify success (means env vars were used correctly)
        assert result["status"] == "success"
        assert "Memory stored successfully" in result["content"][0]["text"]


def test_direct_tool_usage(mock_elasticsearch_client, mock_bedrock_client):
    """Test using the elasticsearch_memory tool directly with an agent."""
    # Configure mock responses
    mock_elasticsearch_client["client"].index.return_value = {"result": "created", "_id": "test_memory_id"}

    # Create agent with direct tool usage - this demonstrates the new pattern
    agent = Agent(tools=[elasticsearch_memory])

    # Test calling the tool directly with configuration parameters
    result = elasticsearch_memory(
        action="record",
        content="Test memory content",
        cloud_id="test-cloud-id",
        api_key="test-api-key",
        index_name="test_index",
        namespace="test_namespace"
    )

    # Verify success response
    assert result["status"] == "success"
    assert "Memory stored successfully" in result["content"][0]["text"]

    # Verify Elasticsearch index was called
    mock_elasticsearch_client["client"].index.assert_called_once()

    # Verify embedding generation was called
    mock_bedrock_client["bedrock"].invoke_model.assert_called_once()


def test_es_url_connection(mock_elasticsearch_client, mock_bedrock_client):
    """Test using es_url instead of cloud_id for connection."""
    # Configure mock responses
    mock_elasticsearch_client["client"].index.return_value = {"result": "created", "_id": "test_memory_id"}

    # Call tool with es_url instead of cloud_id
    result = elasticsearch_memory(
        action="record",
        content="Test memory content",
        es_url="https://test-cluster.es.region.aws.elastic.cloud:443",
        api_key="test-api-key",
        index_name="test_index",
        namespace="test_namespace"
    )

    # Verify success response
    assert result["status"] == "success"
    assert "Memory stored successfully" in result["content"][0]["text"]

    # Verify Elasticsearch was initialized with URL
    mock_elasticsearch_client["elasticsearch_class"].assert_called_once()
    call_args = mock_elasticsearch_client["elasticsearch_class"].call_args[1]
    assert call_args["hosts"] == ["https://test-cluster.es.region.aws.elastic.cloud:443"]
    assert call_args["api_key"] == "test-api-key"
