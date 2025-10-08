"""
Tests for the elasticsearch_memory tool.
"""

import json
import os
from unittest import mock
from unittest.mock import MagicMock

import pytest
from strands import Agent

from src.strands_tools.elasticsearch_memory import ElasticsearchMemoryToolProvider


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
def provider(mock_elasticsearch_client, mock_bedrock_client):
    """Create a provider instance with mocked clients."""
    provider = ElasticsearchMemoryToolProvider(
        cloud_id="test-cloud-id", api_key="test-api-key", index_name="test_index", namespace="test_namespace"
    )
    return provider


@pytest.fixture
def agent(provider):
    """Create an agent with the provider's tools."""
    return Agent(tools=provider.tools)


def test_initialization(mock_elasticsearch_client, mock_bedrock_client):
    """Test provider initialization with default and custom parameters."""
    # Test with required parameters
    provider = ElasticsearchMemoryToolProvider(cloud_id="test-cloud-id", api_key="test-api-key")

    assert provider.cloud_id == "test-cloud-id"
    assert provider.api_key == "test-api-key"
    assert provider.index_name == "strands_memory"  # default
    assert provider.namespace == "default"  # default

    # Test with custom parameters
    provider = ElasticsearchMemoryToolProvider(
        cloud_id="test-cloud-id",
        api_key="test-api-key",
        index_name="custom_index",
        namespace="custom_namespace",
        embedding_model="custom-model",
        region="us-east-1",
    )

    assert provider.index_name == "custom_index"
    assert provider.namespace == "custom_namespace"
    assert provider.embedding_model == "custom-model"
    assert provider.region == "us-east-1"


def test_initialization_missing_required_params():
    """Test initialization with missing required parameters."""
    # Test missing both cloud_id and es_url
    with pytest.raises(ValueError, match="Either cloud_id or es_url is required"):
        ElasticsearchMemoryToolProvider(api_key="test-api-key")

    # Test missing api_key
    with pytest.raises(ValueError, match="api_key is required"):
        ElasticsearchMemoryToolProvider(cloud_id="test-cloud-id")

    # Test empty cloud_id (caught by "Either cloud_id or es_url is required")
    with pytest.raises(ValueError, match="Either cloud_id or es_url is required"):
        ElasticsearchMemoryToolProvider(cloud_id="", api_key="test-api-key")

    # Test empty api_key
    with pytest.raises(ValueError, match="api_key is required"):
        ElasticsearchMemoryToolProvider(cloud_id="test-cloud-id", api_key="")


def test_initialization_connection_failure():
    """Test initialization with connection failure."""
    from src.strands_tools.elasticsearch_memory import ElasticsearchConnectionError
    
    with mock.patch("src.strands_tools.elasticsearch_memory.Elasticsearch") as mock_es:
        mock_client = MagicMock()
        mock_es.return_value = mock_client
        mock_client.ping.return_value = False  # Connection failure

        with pytest.raises(ElasticsearchConnectionError, match="Failed to initialize Elasticsearch client"):
            ElasticsearchMemoryToolProvider(cloud_id="test-cloud-id", api_key="test-api-key")


def test_index_creation(mock_elasticsearch_client, mock_bedrock_client):
    """Test that index is created with proper mappings."""
    ElasticsearchMemoryToolProvider(
        cloud_id="test-cloud-id", api_key="test-api-key", index_name="test_index"
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


def test_record_memory(provider, mock_elasticsearch_client, mock_bedrock_client):
    """Test recording a memory."""
    # Configure mock responses
    mock_elasticsearch_client["client"].index.return_value = {"result": "created", "_id": "test_memory_id"}

    # Call the method
    result = provider.elasticsearch_memory(
        action="record", content="Test memory content", metadata={"category": "test"}
    )

    # Verify success response
    assert result["status"] == "success"
    assert "Memory stored successfully" in result["content"][0]["text"]

    # Verify Elasticsearch index was called
    mock_elasticsearch_client["client"].index.assert_called_once()

    # Verify embedding generation was called
    mock_bedrock_client["bedrock"].invoke_model.assert_called_once()


def test_retrieve_memories(provider, mock_elasticsearch_client, mock_bedrock_client):
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

    # Call the method
    result = provider.elasticsearch_memory(action="retrieve", query="test query", max_results=5)

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


def test_list_memories(provider, mock_elasticsearch_client):
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

    # Call the method
    result = provider.elasticsearch_memory(action="list", max_results=10)

    # Verify success response
    assert result["status"] == "success"
    assert "Memories listed successfully" in result["content"][0]["text"]

    # Verify search was called with proper query
    mock_elasticsearch_client["client"].search.assert_called_once()
    search_call = mock_elasticsearch_client["client"].search.call_args[1]
    assert search_call["body"]["query"]["term"]["namespace"] == "test_namespace"
    assert search_call["body"]["sort"] == [{"timestamp": {"order": "desc"}}]


def test_get_memory(provider, mock_elasticsearch_client):
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

    # Call the method
    result = provider.elasticsearch_memory(action="get", memory_id="mem_123")

    # Verify success response
    assert result["status"] == "success"
    assert "Memory retrieved successfully" in result["content"][0]["text"]

    # Verify get was called
    mock_elasticsearch_client["client"].get.assert_called_once_with(index="test_index", id="mem_123")


def test_delete_memory(provider, mock_elasticsearch_client):
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

    # Call the method
    result = provider.elasticsearch_memory(action="delete", memory_id="mem_123")

    # Verify success response
    assert result["status"] == "success"
    assert "Memory deleted successfully: mem_123" in result["content"][0]["text"]

    # Verify delete was called
    mock_elasticsearch_client["client"].delete.assert_called_once_with(index="test_index", id="mem_123")


def test_unsupported_action(provider):
    """Test elasticsearch_memory method with an unsupported action."""
    result = provider.elasticsearch_memory(action="unsupported_action")

    # Verify error response
    assert result["status"] == "error"
    assert "is not supported" in result["content"][0]["text"]
    assert "record" in result["content"][0]["text"]
    assert "retrieve" in result["content"][0]["text"]


def test_missing_required_parameters(provider):
    """Test elasticsearch_memory method with missing required parameters."""
    # Test record action without content
    result = provider.elasticsearch_memory(action="record")

    # Verify error response
    assert result["status"] == "error"
    assert "parameters are required" in result["content"][0]["text"]
    assert "content" in result["content"][0]["text"]

    # Test retrieve action without query
    result = provider.elasticsearch_memory(action="retrieve")

    # Verify error response
    assert result["status"] == "error"
    assert "parameters are required" in result["content"][0]["text"]
    assert "query" in result["content"][0]["text"]

    # Test get action without memory_id
    result = provider.elasticsearch_memory(action="get")

    # Verify error response
    assert result["status"] == "error"
    assert "parameters are required" in result["content"][0]["text"]
    assert "memory_id" in result["content"][0]["text"]


def test_elasticsearch_api_error_handling(provider, mock_elasticsearch_client):
    """Test handling of Elasticsearch API errors."""
    # Set up mock to raise an exception
    mock_elasticsearch_client["client"].index.side_effect = Exception("Elasticsearch error")

    # Call the method
    result = provider.elasticsearch_memory(action="record", content="Test content")

    # Verify error response
    assert result["status"] == "error"
    assert "API error" in result["content"][0]["text"]
    assert "Elasticsearch error" in result["content"][0]["text"]


def test_bedrock_api_error_handling(provider, mock_bedrock_client):
    """Test handling of Bedrock API errors."""
    # Set up mock to raise an exception
    mock_bedrock_client["bedrock"].invoke_model.side_effect = Exception("Bedrock error")

    # Call the method
    result = provider.elasticsearch_memory(action="record", content="Test content")

    # Verify error response
    assert result["status"] == "error"
    assert "API error" in result["content"][0]["text"]
    assert "Embedding generation failed" in result["content"][0]["text"]


def test_memory_not_found(provider, mock_elasticsearch_client):
    """Test handling when memory is not found."""
    from elasticsearch import NotFoundError

    # Configure mock to raise NotFoundError
    mock_elasticsearch_client["client"].get.side_effect = NotFoundError("404", "not_found_exception", {})

    # Call the method
    result = provider.elasticsearch_memory(action="get", memory_id="nonexistent")

    # Verify error response
    assert result["status"] == "error"
    assert "Memory nonexistent not found" in result["content"][0]["text"]


def test_namespace_validation(provider, mock_elasticsearch_client):
    """Test that memories are properly filtered by namespace."""
    # Configure mock get response with wrong namespace
    mock_elasticsearch_client["client"].get.return_value = {
        "_source": {"memory_id": "mem_123", "content": "Test content", "namespace": "wrong_namespace"}
    }

    # Call the method
    result = provider.elasticsearch_memory(action="get", memory_id="mem_123")

    # Verify error response
    assert result["status"] == "error"
    assert "not found in namespace test_namespace" in result["content"][0]["text"]


def test_pagination_support(provider, mock_elasticsearch_client):
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
    provider.elasticsearch_memory(action="list", max_results=5, next_token="10")

    # Verify search was called with correct offset
    search_call = mock_elasticsearch_client["client"].search.call_args[1]
    assert search_call["body"]["from"] == 10
    assert search_call["body"]["size"] == 5


def test_environment_variable_defaults():
    """Test that environment variables are used for defaults."""
    with mock.patch.dict(
        os.environ,
        {
            "ELASTICSEARCH_INDEX_NAME": "env_index",
            "ELASTICSEARCH_NAMESPACE": "env_namespace",
            "ELASTICSEARCH_EMBEDDING_MODEL": "env_model",
            "AWS_REGION": "env_region",
        },
    ):
        with mock.patch("src.strands_tools.elasticsearch_memory.Elasticsearch") as mock_es:
            with mock.patch("boto3.client"):
                mock_client = MagicMock()
                mock_es.return_value = mock_client
                mock_client.ping.return_value = True
                mock_client.indices.exists.return_value = True

                provider = ElasticsearchMemoryToolProvider(cloud_id="test-cloud-id", api_key="test-api-key")

                assert provider.index_name == "env_index"
                assert provider.namespace == "env_namespace"
                assert provider.embedding_model == "env_model"
                assert provider.region == "env_region"


def test_memory_id_generation(provider):
    """Test that memory IDs are generated correctly."""
    memory_id = provider._generate_memory_id()

    # Verify format
    assert memory_id.startswith("mem_")
    parts = memory_id.split("_")
    assert len(parts) == 3
    assert parts[0] == "mem"
    assert parts[1].isdigit()  # timestamp
    assert len(parts[2]) == 8  # UUID fragment


def test_embedding_generation(provider, mock_bedrock_client):
    """Test embedding generation."""
    embedding = provider._generate_embedding("test text")

    # Verify embedding is returned
    assert isinstance(embedding, list)
    assert len(embedding) == 1024  # Titan v2 returns 1024 dimensions
    assert all(isinstance(x, float) for x in embedding)

    # Verify Bedrock was called correctly
    mock_bedrock_client["bedrock"].invoke_model.assert_called_once()
    call_args = mock_bedrock_client["bedrock"].invoke_model.call_args[1]
    assert call_args["modelId"] == "amazon.titan-embed-text-v2:0"

    # Verify request body
    body = json.loads(call_args["body"])
    assert body["inputText"] == "test text"
