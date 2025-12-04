"""
Tests for the mongodb_memory tool.
"""

import json
import os
from unittest import mock
from unittest.mock import MagicMock

import pytest
from strands import Agent

from src.strands_tools.mongodb_memory import mongodb_memory


@pytest.fixture(autouse=True)
def mock_mongodb_client():
    """Mock MongoDB client to avoid actual connections."""
    with mock.patch("src.strands_tools.mongodb_memory.MongoClient") as mock_mongo:
        # Create mock client instance
        mock_client = MagicMock()
        mock_mongo.return_value = mock_client

        # Configure admin.command to return success (ping test)
        mock_client.admin.command.return_value = {"ok": 1}

        # Create mock database and collection
        mock_database = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value = mock_database
        mock_database.__getitem__.return_value = mock_collection

        # Configure collection methods
        mock_collection.list_search_indexes.return_value = []
        mock_collection.create_search_index.return_value = None

        # Configure count_documents to return an integer
        mock_collection.count_documents.return_value = 0

        # Configure aggregate to return empty list by default
        mock_collection.aggregate.return_value = []

        # Configure find to return empty cursor
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.skip.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_cursor.__iter__.return_value = []
        mock_collection.find.return_value = mock_cursor

        yield {
            "mongo_class": mock_mongo,
            "client": mock_client,
            "database": mock_database,
            "collection": mock_collection,
        }


@pytest.fixture(autouse=True)
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
def agent(mock_mongodb_client, mock_bedrock_client):
    """Create an agent with the direct mongodb_memory tool."""
    return Agent(tools=[mongodb_memory])


@pytest.fixture
def config():
    """Configuration parameters for testing."""
    return {
        "cluster_uri": "mongodb+srv://test:test@cluster.mongodb.net/",
        "database_name": "test_db",
        "collection_name": "test_collection",
        "namespace": "test_namespace",
        "region": "us-east-1",
    }


def test_missing_required_params(mock_mongodb_client, mock_bedrock_client):
    """Test tool with missing required parameters."""
    agent = Agent(tools=[mongodb_memory])

    # Test missing cluster_uri
    result = agent.tool.mongodb_memory(action="record", content="test")
    assert result["status"] == "error"
    assert "cluster_uri is required for MongoDB Memory Tool" in result["content"][0]["text"]


def test_connection_failure(mock_mongodb_client, mock_bedrock_client):
    """Test tool with connection failure."""
    agent = Agent(tools=[mongodb_memory])

    # Configure admin.command to raise ConnectionFailure
    from pymongo.errors import ConnectionFailure

    mock_mongodb_client["client"].admin.command.side_effect = ConnectionFailure("Connection failed")

    result = agent.tool.mongodb_memory(
        action="record", content="test", cluster_uri="mongodb+srv://test:test@cluster.mongodb.net/"
    )

    assert result["status"] == "error"
    assert "Unable to connect to MongoDB cluster" in result["content"][0]["text"]


def test_vector_index_creation(mock_mongodb_client, mock_bedrock_client, config):
    """Test that vector search index is created with proper configuration."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock responses
    mock_mongodb_client["collection"].insert_one.return_value = MagicMock(inserted_id="test_id")

    agent.tool.mongodb_memory(action="record", content="Test content", **config)

    # Verify index creation was called for record (it shouldn't be)
    # Index creation only happens for retrieve operations
    mock_mongodb_client["collection"].create_search_index.assert_not_called()

    # Test retrieve action which should create index
    mock_mongodb_client["collection"].aggregate.return_value = []
    agent.tool.mongodb_memory(action="retrieve", query="test query", **config)

    # Verify index creation was called
    mock_mongodb_client["collection"].create_search_index.assert_called_once()

    # Get the call arguments
    call_args = mock_mongodb_client["collection"].create_search_index.call_args[0][0]
    assert call_args["name"] == "vector_index"
    assert call_args["definition"]["mappings"]["fields"]["embedding"]["type"] == "knnVector"
    assert call_args["definition"]["mappings"]["fields"]["embedding"]["dimensions"] == 1024
    assert call_args["definition"]["mappings"]["fields"]["embedding"]["similarity"] == "cosine"
    assert call_args["definition"]["mappings"]["fields"]["namespace"]["type"] == "string"


def test_record_memory(mock_mongodb_client, mock_bedrock_client, config):
    """Test recording a memory."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock responses
    mock_result = MagicMock()
    mock_result.inserted_id = "test_object_id"
    mock_mongodb_client["collection"].insert_one.return_value = mock_result

    # Call the tool
    result = agent.tool.mongodb_memory(
        action="record", content="Test memory content", metadata={"category": "test"}, **config
    )

    # Verify success response
    assert result["status"] == "success"
    assert "text" in result["content"][0]
    assert "Memory stored successfully" in result["content"][0]["text"]
    # Get JSON from second content item
    assert "json" in result["content"][1]
    response_data = result["content"][1]["json"]
    assert "memory_id" in response_data
    assert response_data["content"] == "Test memory content"

    # Verify MongoDB insert was called
    mock_mongodb_client["collection"].insert_one.assert_called_once()

    # Verify embedding generation was called
    mock_bedrock_client["bedrock"].invoke_model.assert_called_once()


def test_retrieve_memories(mock_mongodb_client, mock_bedrock_client, config):
    """Test retrieving memories with semantic search."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock search response
    mock_mongodb_client["collection"].aggregate.return_value = [
        {
            "memory_id": "mem_123",
            "content": "Test content",
            "timestamp": "2023-01-01T00:00:00Z",
            "metadata": {},
            "score": 0.95,
        }
    ]

    # Call the tool
    result = agent.tool.mongodb_memory(action="retrieve", query="test query", max_results=5, **config)

    # Verify success response
    assert result["status"] == "success"
    assert "text" in result["content"][0]
    assert "Memories retrieved successfully" in result["content"][0]["text"]
    # Get JSON from second content item
    assert "json" in result["content"][1]
    response_data = result["content"][1]["json"]
    assert "memories" in response_data
    assert len(response_data["memories"]) >= 0

    # Verify aggregate was called with vector search pipeline
    mock_mongodb_client["collection"].aggregate.assert_called()
    call_args = mock_mongodb_client["collection"].aggregate.call_args[0][0]
    assert "$vectorSearch" in call_args[0]
    assert call_args[0]["$vectorSearch"]["path"] == "embedding"

    # Verify embedding generation for query
    mock_bedrock_client["bedrock"].invoke_model.assert_called_once()


def test_list_memories(mock_mongodb_client, mock_bedrock_client, config):
    """Test listing all memories."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock find response
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.skip.return_value = mock_cursor
    mock_cursor.limit.return_value = mock_cursor
    mock_cursor.__iter__.return_value = [
        {
            "memory_id": "mem_123",
            "content": "Test content 1",
            "timestamp": "2023-01-01T00:00:00Z",
            "metadata": {},
        },
        {
            "memory_id": "mem_456",
            "content": "Test content 2",
            "timestamp": "2023-01-02T00:00:00Z",
            "metadata": {},
        },
    ]

    mock_mongodb_client["collection"].find.return_value = mock_cursor
    mock_mongodb_client["collection"].count_documents.return_value = 2

    # Call the tool
    result = agent.tool.mongodb_memory(action="list", max_results=10, **config)

    # Verify success response
    assert result["status"] == "success"
    assert "text" in result["content"][0]
    assert "Memories listed successfully" in result["content"][0]["text"]
    # Get JSON from second content item
    assert "json" in result["content"][1]
    response_data = result["content"][1]["json"]
    assert "memories" in response_data
    assert "total" in response_data

    # Verify find was called with proper query
    mock_mongodb_client["collection"].find.assert_called_once()
    call_args = mock_mongodb_client["collection"].find.call_args[0]
    assert call_args[0] == {"namespace": "test_namespace"}


def test_get_memory(mock_mongodb_client, mock_bedrock_client, config):
    """Test getting a specific memory by ID."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock find_one response
    mock_mongodb_client["collection"].find_one.return_value = {
        "memory_id": "mem_123",
        "content": "Test content",
        "timestamp": "2023-01-01T00:00:00Z",
        "metadata": {"category": "test"},
        "namespace": "test_namespace",
    }

    # Call the tool
    result = agent.tool.mongodb_memory(action="get", memory_id="mem_123", **config)

    # Verify success response
    assert result["status"] == "success"
    assert "text" in result["content"][0]
    assert "Memory retrieved successfully" in result["content"][0]["text"]
    # Get JSON from second content item
    assert "json" in result["content"][1]
    response_data = result["content"][1]["json"]
    assert "memory_id" in response_data
    assert response_data["memory_id"] == "mem_123"

    # Verify find_one was called with both memory_id and namespace for security
    mock_mongodb_client["collection"].find_one.assert_called_once()
    call_args = mock_mongodb_client["collection"].find_one.call_args[0]
    assert call_args[0] == {"memory_id": "mem_123", "namespace": "test_namespace"}


def test_delete_memory(mock_mongodb_client, mock_bedrock_client, config):
    """Test deleting a memory."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock responses
    mock_mongodb_client["collection"].find_one.return_value = {
        "memory_id": "mem_123",
        "content": "Test content",
        "timestamp": "2023-01-01T00:00:00Z",
        "metadata": {},
        "namespace": "test_namespace",
    }

    mock_delete_result = MagicMock()
    mock_delete_result.deleted_count = 1
    mock_mongodb_client["collection"].delete_one.return_value = mock_delete_result

    # Call the tool
    result = agent.tool.mongodb_memory(action="delete", memory_id="mem_123", **config)

    # Verify success response
    assert result["status"] == "success"
    assert "text" in result["content"][0]
    assert "Memory deleted successfully: mem_123" in result["content"][0]["text"]

    # Verify delete was called
    mock_mongodb_client["collection"].delete_one.assert_called_once()
    call_args = mock_mongodb_client["collection"].delete_one.call_args[0]
    assert call_args[0] == {"memory_id": "mem_123", "namespace": "test_namespace"}


def test_unsupported_action(mock_mongodb_client, mock_bedrock_client, config):
    """Test tool with an unsupported action."""
    agent = Agent(tools=[mongodb_memory])

    result = agent.tool.mongodb_memory(action="unsupported_action", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "is not supported" in result["content"][0]["text"]
    assert "record" in result["content"][0]["text"]
    assert "retrieve" in result["content"][0]["text"]


def test_missing_required_parameters(mock_mongodb_client, mock_bedrock_client, config):
    """Test tool with missing required parameters."""
    agent = Agent(tools=[mongodb_memory])

    # Test record action without content
    result = agent.tool.mongodb_memory(action="record", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "parameters are required" in result["content"][0]["text"]
    assert "content" in result["content"][0]["text"]

    # Test retrieve action without query
    result = agent.tool.mongodb_memory(action="retrieve", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "parameters are required" in result["content"][0]["text"]
    assert "query" in result["content"][0]["text"]

    # Test get action without memory_id
    result = agent.tool.mongodb_memory(action="get", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "parameters are required" in result["content"][0]["text"]
    assert "memory_id" in result["content"][0]["text"]


def test_mongodb_api_error_handling(mock_mongodb_client, mock_bedrock_client, config):
    """Test handling of MongoDB API errors."""
    agent = Agent(tools=[mongodb_memory])

    # Set up mock to raise an exception
    mock_mongodb_client["collection"].insert_one.side_effect = Exception("MongoDB error")

    # Call the tool
    result = agent.tool.mongodb_memory(action="record", content="Test content", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "API error" in result["content"][0]["text"]
    assert "MongoDB error" in result["content"][0]["text"]


def test_bedrock_api_error_handling(mock_mongodb_client, mock_bedrock_client, config):
    """Test handling of Bedrock API errors."""
    agent = Agent(tools=[mongodb_memory])

    # Set up mock to raise an exception
    mock_bedrock_client["bedrock"].invoke_model.side_effect = Exception("Bedrock error")

    # Call the tool
    result = agent.tool.mongodb_memory(action="record", content="Test content", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "API error" in result["content"][0]["text"]
    assert "Embedding generation failed" in result["content"][0]["text"]


def test_memory_not_found(mock_mongodb_client, mock_bedrock_client, config):
    """Test handling when memory is not found."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock to return None (not found)
    mock_mongodb_client["collection"].find_one.return_value = None

    # Call the tool
    result = agent.tool.mongodb_memory(action="get", memory_id="nonexistent", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "Memory nonexistent not found" in result["content"][0]["text"]


def test_namespace_validation(mock_mongodb_client, mock_bedrock_client, config):
    """Test that memories are properly filtered by namespace."""
    agent = Agent(tools=[mongodb_memory])

    mock_mongodb_client["collection"].find_one.return_value = None

    result = agent.tool.mongodb_memory(action="get", memory_id="mem_123", **config)

    # Verify error response
    assert result["status"] == "error"
    assert "Memory mem_123 not found in namespace test_namespace" in result["content"][0]["text"]

    mock_mongodb_client["collection"].find_one.assert_called_once()
    call_args = mock_mongodb_client["collection"].find_one.call_args[0]
    assert call_args[0] == {"memory_id": "mem_123", "namespace": "test_namespace"}


def test_pagination_support(mock_mongodb_client, mock_bedrock_client, config):
    """Test pagination support in list and retrieve operations."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock find response with pagination
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.skip.return_value = mock_cursor
    mock_cursor.limit.return_value = mock_cursor
    mock_cursor.__iter__.return_value = [
        {
            "memory_id": "mem_123",
            "content": "Test content",
            "timestamp": "2023-01-01T00:00:00Z",
            "metadata": {},
        }
    ]

    mock_mongodb_client["collection"].find.return_value = mock_cursor
    mock_mongodb_client["collection"].count_documents.return_value = 20  # More results available

    # Test list with pagination
    agent.tool.mongodb_memory(action="list", max_results=5, next_token="10", **config)

    # Verify skip was called with correct offset
    mock_cursor.skip.assert_called_with(10)
    mock_cursor.limit.assert_called_with(5)


def test_environment_variable_defaults(mock_mongodb_client, mock_bedrock_client):
    """Test that environment variables are used for defaults."""
    agent = Agent(tools=[mongodb_memory])

    with mock.patch.dict(
        os.environ,
        {
            "MONGODB_ATLAS_CLUSTER_URI": "mongodb+srv://env:env@cluster.mongodb.net/",
            "MONGODB_DATABASE_NAME": "env_db",
            "MONGODB_COLLECTION_NAME": "env_collection",
            "MONGODB_NAMESPACE": "env_namespace",
            "MONGODB_EMBEDDING_MODEL": "env_model",
            "AWS_REGION": "env_region",
        },
    ):
        # Configure mock responses
        mock_result = MagicMock()
        mock_result.inserted_id = "test_id"
        mock_mongodb_client["collection"].insert_one.return_value = mock_result

        # Call tool without explicit parameters (should use env vars)
        result = agent.tool.mongodb_memory(action="record", content="Test content")

        # Verify success (means env vars were used correctly)
        assert result["status"] == "success"
        assert "text" in result["content"][0]
        assert "Memory stored successfully" in result["content"][0]["text"]
        # Get JSON from second content item
        assert "json" in result["content"][1]
        response_data = result["content"][1]["json"]
        assert "memory_id" in response_data


def test_agent_tool_usage(mock_mongodb_client, mock_bedrock_client):
    """Test using the mongodb_memory tool through agent.tool pattern."""
    # Configure mock responses
    mock_result = MagicMock()
    mock_result.inserted_id = "test_id"
    mock_mongodb_client["collection"].insert_one.return_value = mock_result

    # Create agent with direct tool usage - this demonstrates the standard pattern
    agent = Agent(tools=[mongodb_memory])

    # Test calling the tool through agent.tool with configuration parameters
    result = agent.tool.mongodb_memory(
        action="record",
        content="Test memory content",
        cluster_uri="mongodb+srv://test:test@cluster.mongodb.net/",
        database_name="test_db",
        collection_name="test_collection",
        namespace="test_namespace",
    )

    # Verify success response
    assert result["status"] == "success"
    assert "Memory stored successfully" in result["content"][0]["text"]

    # Verify MongoDB insert was called
    mock_mongodb_client["collection"].insert_one.assert_called_once()

    # Verify embedding generation was called
    mock_bedrock_client["bedrock"].invoke_model.assert_called_once()


def test_custom_embedding_model(mock_mongodb_client, mock_bedrock_client, config):
    """Test using custom embedding model."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock responses
    mock_result = MagicMock()
    mock_result.inserted_id = "test_id"
    mock_mongodb_client["collection"].insert_one.return_value = mock_result

    # Call tool with custom embedding model
    result = agent.tool.mongodb_memory(
        action="record", content="Test memory content", embedding_model="amazon.titan-embed-text-v1:0", **config
    )

    # Verify success response
    assert result["status"] == "success"
    assert "Memory stored successfully" in result["content"][0]["text"]

    # Verify Bedrock was called with custom model
    mock_bedrock_client["bedrock"].invoke_model.assert_called_once()
    call_args = mock_bedrock_client["bedrock"].invoke_model.call_args
    assert call_args[1]["modelId"] == "amazon.titan-embed-text-v1:0"


def test_multiple_namespaces(mock_mongodb_client, mock_bedrock_client, config):
    """Test using different namespaces for data isolation."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock responses
    mock_result = MagicMock()
    mock_result.inserted_id = "test_id"
    mock_mongodb_client["collection"].insert_one.return_value = mock_result

    # Store memory in user namespace
    result1 = agent.tool.mongodb_memory(
        action="record",
        content="Alice likes Italian food",
        namespace="user_alice",
        **{k: v for k, v in config.items() if k != "namespace"},
    )

    # Store memory in system namespace
    result2 = agent.tool.mongodb_memory(
        action="record",
        content="System maintenance scheduled",
        namespace="system_global",
        **{k: v for k, v in config.items() if k != "namespace"},
    )

    # Verify both operations succeeded
    assert result1["status"] == "success"
    assert result2["status"] == "success"

    # Verify both calls were made
    assert mock_mongodb_client["collection"].insert_one.call_count == 2


def test_configuration_dictionary_pattern(mock_mongodb_client, mock_bedrock_client):
    """Test using configuration dictionary for cleaner code."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock responses
    mock_result = MagicMock()
    mock_result.inserted_id = "test_id"
    mock_mongodb_client["collection"].insert_one.return_value = mock_result

    mock_mongodb_client["collection"].aggregate.return_value = [
        {
            "memory_id": "mem_123",
            "content": "Test content",
            "timestamp": "2023-01-01T00:00:00Z",
            "metadata": {},
            "score": 0.95,
        }
    ]

    # Create configuration dictionary
    config = {
        "cluster_uri": "mongodb+srv://test:test@cluster.mongodb.net/",
        "database_name": "memories_db",
        "collection_name": "memories",
        "namespace": "user_123",
        "region": "us-east-1",
    }

    # Store memory using config dictionary
    result1 = agent.tool.mongodb_memory(action="record", content="User prefers vegetarian pizza", **config)

    # Search memories using config dictionary
    result2 = agent.tool.mongodb_memory(action="retrieve", query="food preferences", max_results=5, **config)

    # Verify both operations succeeded
    assert result1["status"] == "success"
    assert result2["status"] == "success"
    assert "Memory stored successfully" in result1["content"][0]["text"]
    assert "Memories retrieved successfully" in result2["content"][0]["text"]


def test_batch_operations(mock_mongodb_client, mock_bedrock_client, config):
    """Test storing multiple related memories in batch."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock responses
    mock_result = MagicMock()
    mock_result.inserted_id = "test_id"
    mock_mongodb_client["collection"].insert_one.return_value = mock_result

    # Store multiple related memories
    memories = ["User likes Italian food", "User is allergic to nuts", "User prefers evening meetings"]

    results = []
    for content in memories:
        result = agent.tool.mongodb_memory(
            action="record",
            content=content,
            metadata={"batch": "user_preferences", "category": "preferences"},
            **config,
        )
        results.append(result)

    # Verify all operations succeeded
    for result in results:
        assert result["status"] == "success"
        assert "Memory stored successfully" in result["content"][0]["text"]

    # Verify correct number of calls were made
    assert mock_mongodb_client["collection"].insert_one.call_count == len(memories)


def test_error_handling_scenarios(mock_mongodb_client, mock_bedrock_client, config):
    """Test comprehensive error handling scenarios."""
    agent = Agent(tools=[mongodb_memory])

    # Test connection errors
    from pymongo.errors import ConnectionFailure

    mock_mongodb_client["client"].admin.command.side_effect = ConnectionFailure("Connection failed")
    result = agent.tool.mongodb_memory(action="record", content="test", **config)
    assert result["status"] == "error"
    assert "Unable to connect to MongoDB cluster" in result["content"][0]["text"]

    # Reset admin.command to return success for subsequent tests
    mock_mongodb_client["client"].admin.command.side_effect = None
    mock_mongodb_client["client"].admin.command.return_value = {"ok": 1}

    # Test MongoDB API errors
    mock_mongodb_client["collection"].insert_one.side_effect = Exception("MongoDB connection failed")
    result = agent.tool.mongodb_memory(action="record", content="test", **config)
    assert result["status"] == "error"
    assert "API error" in result["content"][0]["text"]

    # Reset side effect
    mock_mongodb_client["collection"].insert_one.side_effect = None

    # Test Bedrock API errors
    mock_bedrock_client["bedrock"].invoke_model.side_effect = Exception("Bedrock access denied")
    result = agent.tool.mongodb_memory(action="record", content="test", **config)
    assert result["status"] == "error"
    assert "Embedding generation failed" in result["content"][0]["text"]


def test_metadata_usage_scenarios(mock_mongodb_client, mock_bedrock_client, config):
    """Test various metadata usage patterns."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock responses
    mock_result = MagicMock()
    mock_result.inserted_id = "test_id"
    mock_mongodb_client["collection"].insert_one.return_value = mock_result

    # Test structured metadata
    structured_metadata = {
        "type": "deadline",
        "project": "project_alpha",
        "priority": "high",
        "due_date": "2024-02-01",
        "assigned_to": ["alice", "bob"],
    }

    result = agent.tool.mongodb_memory(
        action="record", content="Important project deadline", metadata=structured_metadata, **config
    )

    assert result["status"] == "success"
    assert "Memory stored successfully" in result["content"][0]["text"]

    # Verify the insert call included metadata
    mock_mongodb_client["collection"].insert_one.assert_called()
    call_args = mock_mongodb_client["collection"].insert_one.call_args[0][0]
    assert call_args["metadata"] == structured_metadata


def test_performance_scenarios(mock_mongodb_client, mock_bedrock_client, config):
    """Test performance-related scenarios like pagination."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock find response with pagination
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.skip.return_value = mock_cursor
    mock_cursor.limit.return_value = mock_cursor
    mock_cursor.__iter__.return_value = [
        {
            "memory_id": f"mem_{i}",
            "content": f"Test content {i}",
            "timestamp": "2023-01-01T00:00:00Z",
            "metadata": {},
        }
        for i in range(5)
    ]

    mock_mongodb_client["collection"].find.return_value = mock_cursor
    mock_mongodb_client["collection"].count_documents.return_value = 25  # More results available

    # Test pagination with next_token
    result = agent.tool.mongodb_memory(action="list", max_results=5, next_token="10", **config)

    assert result["status"] == "success"
    assert "Memories listed successfully" in result["content"][0]["text"]

    # Verify pagination parameters were used
    mock_cursor.skip.assert_called_with(10)
    mock_cursor.limit.assert_called_with(5)


def test_security_scenarios(mock_mongodb_client, mock_bedrock_client):
    """Test security-related scenarios like namespace isolation."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock find_one response to return None (not found)
    # This simulates the new behavior where we query with both memory_id and namespace
    mock_mongodb_client["collection"].find_one.return_value = None

    # Test namespace validation
    result = agent.tool.mongodb_memory(
        action="get",
        memory_id="mem_123",
        cluster_uri="mongodb+srv://test:test@cluster.mongodb.net/",
        database_name="test_db",
        collection_name="test_collection",
        namespace="correct_namespace",
    )

    assert result["status"] == "error"
    assert "Memory mem_123 not found in namespace correct_namespace" in result["content"][0]["text"]


def test_troubleshooting_scenarios(mock_mongodb_client, mock_bedrock_client, config):
    """Test troubleshooting scenarios mentioned in documentation."""
    agent = Agent(tools=[mongodb_memory])

    # Test index creation failure - now it should succeed with warning, not error
    mock_mongodb_client["collection"].create_search_index.side_effect = Exception("Index creation failed")
    mock_mongodb_client["collection"].aggregate.return_value = []
    result = agent.tool.mongodb_memory(action="retrieve", query="test", **config)
    assert result["status"] == "success"  # Should succeed despite index creation failure

    # Reset side effect
    mock_mongodb_client["collection"].create_search_index.side_effect = None

    # Test authentication errors (simulated by connection failure)
    from pymongo.errors import ConnectionFailure

    mock_mongodb_client["client"].admin.command.side_effect = ConnectionFailure("Authentication failed")
    result = agent.tool.mongodb_memory(action="record", content="test", **config)
    assert result["status"] == "error"
    assert "Unable to connect to MongoDB cluster" in result["content"][0]["text"]


def test_nosql_injection_prevention(mock_mongodb_client, mock_bedrock_client, config):
    """Test that NoSQL injection attempts are blocked."""
    agent = Agent(tools=[mongodb_memory])

    # Test the specific PoC attack: namespace={"$ne": ""}
    malicious_namespace = {"$ne": ""}

    # Remove namespace from config to avoid conflict
    test_config = {k: v for k, v in config.items() if k != "namespace"}

    # Test with list action (most common attack vector)
    result = agent.tool.mongodb_memory(action="list", namespace=malicious_namespace, **test_config)

    # Should be blocked - either by Pydantic validation or our custom validation
    assert result["status"] == "error"
    error_text = result["content"][0]["text"]
    assert "Invalid namespace" in error_text or "Input should be a valid string" in error_text, (
        f"Expected validation error, got: {error_text}"
    )

    # Test other MongoDB operators
    other_injection_attempts = [
        {"$gt": ""},
        {"$regex": ".*"},
        {"$exists": True},
        {"$in": ["tenant1", "tenant2"]},
    ]

    for injection_payload in other_injection_attempts:
        result = agent.tool.mongodb_memory(action="list", namespace=injection_payload, **test_config)

        assert result["status"] == "error", f"Injection {injection_payload} should be blocked"
        error_text = result["content"][0]["text"]
        assert "Invalid namespace" in error_text or "Input should be a valid string" in error_text, (
            f"Expected validation error for {injection_payload}"
        )


def test_namespace_validation_strict_rules(mock_mongodb_client, mock_bedrock_client, config):
    """Test strict namespace validation rules."""
    agent = Agent(tools=[mongodb_memory])

    # Remove namespace from config to avoid conflict
    test_config = {k: v for k, v in config.items() if k != "namespace"}

    # Test invalid characters (should be rejected)
    invalid_namespaces = [
        "user.name",  # Dots not allowed in strict mode
        "user@domain",  # @ symbol
        "user$name",  # $ symbol (MongoDB operator prefix)
        "user name",  # Space
        "user/path",  # Forward slash
        "user:name",  # Colon
        "a" * 65,  # Too long (over 64 chars)
        "",  # Empty
        "   ",  # Whitespace only
    ]

    for invalid_namespace in invalid_namespaces:
        result = agent.tool.mongodb_memory(action="list", namespace=invalid_namespace, **test_config)

        assert result["status"] == "error", f"Invalid namespace '{invalid_namespace}' should be rejected"
        error_text = result["content"][0]["text"]
        assert "Invalid namespace" in error_text, f"Expected validation error for '{invalid_namespace}'"


def test_vector_search_pipeline_structure(mock_mongodb_client, mock_bedrock_client, config):
    """Test that the vector search pipeline is structured correctly."""
    agent = Agent(tools=[mongodb_memory])

    # Configure mock aggregate response
    mock_mongodb_client["collection"].aggregate.return_value = [
        {
            "memory_id": "mem_123",
            "content": "Test content",
            "timestamp": "2023-01-01T00:00:00Z",
            "metadata": {},
            "score": 0.95,
        }
    ]

    # Call retrieve action
    agent.tool.mongodb_memory(action="retrieve", query="test query", **config)

    # Verify aggregate was called
    mock_mongodb_client["collection"].aggregate.assert_called()

    # Get the pipeline structure - there should be two calls to aggregate
    # First call is the main search pipeline, second is for total count
    aggregate_calls = mock_mongodb_client["collection"].aggregate.call_args_list
    assert len(aggregate_calls) >= 1

    # Get the first (main search) pipeline
    main_pipeline = aggregate_calls[0][0][0]

    # Verify pipeline structure
    assert len(main_pipeline) == 5  # Should have vectorSearch, skip, limit, addFields, project stages
    assert "$vectorSearch" in main_pipeline[0]
    assert "$skip" in main_pipeline[1]
    assert "$limit" in main_pipeline[2]
    assert "$addFields" in main_pipeline[3]
    assert "$project" in main_pipeline[4]

    # Verify vectorSearch configuration
    vector_search = main_pipeline[0]["$vectorSearch"]
    assert vector_search["index"] == "vector_index"
    assert vector_search["path"] == "embedding"
