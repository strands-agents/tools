"""
Simplified tests for the agent_core_memory tool.
"""

import os
from unittest import mock

import pytest
from botocore.config import Config as BotocoreConfig
from strands import Agent

from src.strands_tools.agent_core_memory import AgentCoreMemoryToolProvider


@pytest.fixture
def mock_boto3_client():
    """Mock boto3.client to avoid actual AWS calls."""
    with mock.patch("boto3.client") as mock_client:
        # Create mock client
        mock_bedrock_agent_core = mock.MagicMock()

        # Configure boto3.client to return our mock
        mock_client.side_effect = (
            lambda service, **kwargs: mock_bedrock_agent_core if service == "bedrock-agentcore" else None
        )

        yield {
            "client": mock_client,
            "bedrock_agent_core": mock_bedrock_agent_core,
        }


@pytest.fixture
def provider(mock_boto3_client):
    """Create a provider instance with mocked clients."""
    provider = AgentCoreMemoryToolProvider(
        memory_id="test-memory-id",
        actor_id="test-actor-id",
        session_id="test-session-id",
        namespace="test-namespace",
    )
    return provider


@pytest.fixture
def agent(provider):
    """Create an agent with the provider's tools."""
    return Agent(tools=provider.tools)


def test_initialization(mock_boto3_client):
    """Test provider initialization with default and custom parameters."""
    # Test with required parameters
    provider = AgentCoreMemoryToolProvider(
        memory_id="test-memory-id", actor_id="test-actor-id", session_id="test-session-id", namespace="test-namespace"
    )

    assert provider.memory_id == "test-memory-id"
    assert provider.region == os.environ.get("AWS_REGION", "us-west-2")
    assert provider.actor_id == "test-actor-id"
    assert provider.session_id == "test-session-id"
    assert provider.namespace == "test-namespace"

    # Test with custom region
    provider = AgentCoreMemoryToolProvider(
        memory_id="test-memory-id",
        region="us-east-1",
        actor_id="test-actor-id",
        session_id="test-session-id",
        namespace="test-namespace",
    )

    assert provider.memory_id == "test-memory-id"
    assert provider.region == "us-east-1"
    assert provider.actor_id == "test-actor-id"
    assert provider.session_id == "test-session-id"
    assert provider.namespace == "test-namespace"

    # Test with boto client config
    boto_config = BotocoreConfig(region_name="us-west-1", user_agent_extra="test-agent")
    provider = AgentCoreMemoryToolProvider(
        memory_id="test-memory-id",
        actor_id="test-actor-id",
        session_id="test-session-id",
        namespace="test-namespace",
        boto_client_config=boto_config,
    )

    # Verify client config was properly merged
    assert "test-agent" in provider.client_config.user_agent_extra
    assert "strands-agents-memory" in provider.client_config.user_agent_extra


def test_initialization_missing_required_params():
    """Test initialization with missing required parameters."""
    # Test missing memory_id
    with pytest.raises((ValueError, TypeError)):
        AgentCoreMemoryToolProvider(actor_id="test-actor-id", session_id="test-session-id", namespace="test-namespace")

    # Test missing actor_id
    with pytest.raises((ValueError, TypeError)):
        AgentCoreMemoryToolProvider(
            memory_id="test-memory-id", session_id="test-session-id", namespace="test-namespace"
        )

    # Test missing session_id
    with pytest.raises((ValueError, TypeError)):
        AgentCoreMemoryToolProvider(memory_id="test-memory-id", actor_id="test-actor-id", namespace="test-namespace")

    # Test missing namespace
    with pytest.raises((ValueError, TypeError)):
        AgentCoreMemoryToolProvider(memory_id="test-memory-id", actor_id="test-actor-id", session_id="test-session-id")


def test_initialization_with_region(mock_boto3_client):
    """Test that provider can be initialized with a specific region."""
    # Initialize with us-east-1 region
    provider = AgentCoreMemoryToolProvider(
        memory_id="test-memory-id",
        actor_id="test-actor-id",
        session_id="test-session-id",
        namespace="test-namespace",
        region="us-east-1",
    )

    # The client is already initialized in the constructor
    # No need to access it to trigger initialization

    # Verify correct endpoint URL was used
    mock_boto3_client["client"].assert_called_with(
        "bedrock-agentcore",
        region_name="us-east-1",
        config=provider.client_config,
    )


def test_record_memory_response_filtering(provider, mock_boto3_client):
    """Test that record action filters out metadata from the response."""
    # Set up mock response with both relevant data and metadata
    event_data = {
        "eventId": "test-event-id",
        "memoryId": "test-memory-id",
        "actorId": "test-actor-id",
        "sessionId": "test-session-id",
    }
    mock_boto3_client["bedrock_agent_core"].create_event.return_value = {
        "event": event_data,
        "ResponseMetadata": {"RequestId": "test-request-id"},
    }

    # Call the method
    result = provider.agent_core_memory(action="record", content="Hello")

    # Verify only event data is included in the response
    assert result["status"] == "success"
    assert "test-event-id" in result["content"][0]["text"]
    assert "RequestId" not in result["content"][0]["text"]


def test_retrieve_memory_response_filtering(provider, mock_boto3_client):
    """Test that retrieve action filters out metadata from the response."""
    # Set up mock response with both relevant data and metadata
    memory_records = [
        {"memoryRecordId": "record-1", "content": {"text": "Memory 1"}},
        {"memoryRecordId": "record-2", "content": {"text": "Memory 2"}},
    ]
    mock_boto3_client["bedrock_agent_core"].retrieve_memory_records.return_value = {
        "memoryRecordSummaries": memory_records,
        "nextToken": "next-page",
        "ResponseMetadata": {"RequestId": "test-request-id"},
    }

    # Call the method
    result = provider.agent_core_memory(action="retrieve", query="test query")

    # Verify only relevant data is included in the response
    assert result["status"] == "success"
    assert "record-1" in result["content"][0]["text"]
    assert "record-2" in result["content"][0]["text"]
    assert "next-page" in result["content"][0]["text"]
    assert "RequestId" not in result["content"][0]["text"]


def test_list_memories_response_filtering(provider, mock_boto3_client):
    """Test that list action filters out metadata from the response."""
    # Set up mock response with both relevant data and metadata
    memory_records = [
        {"memoryRecordId": "record-1", "content": {"text": "Memory 1"}},
        {"memoryRecordId": "record-2", "content": {"text": "Memory 2"}},
    ]
    mock_boto3_client["bedrock_agent_core"].list_memory_records.return_value = {
        "memoryRecordSummaries": memory_records,
        "nextToken": "next-page",
        "ResponseMetadata": {"RequestId": "test-request-id"},
    }

    # Call the method
    result = provider.agent_core_memory(action="list")

    # Verify only relevant data is included in the response
    assert result["status"] == "success"
    assert "record-1" in result["content"][0]["text"]
    assert "record-2" in result["content"][0]["text"]
    assert "next-page" in result["content"][0]["text"]
    assert "RequestId" not in result["content"][0]["text"]


def test_get_memory_response_filtering(provider, mock_boto3_client):
    """Test that get action filters out metadata from the response."""
    # Set up mock response with both relevant data and metadata
    memory_record = {"memoryRecordId": "test-record-id", "content": {"text": "Memory content"}}
    mock_boto3_client["bedrock_agent_core"].get_memory_record.return_value = {
        "memoryRecord": memory_record,
        "ResponseMetadata": {"RequestId": "test-request-id"},
    }

    # Call the method
    result = provider.agent_core_memory(action="get", memory_record_id="test-record-id")

    # Verify only relevant data is included in the response
    assert result["status"] == "success"
    assert "test-record-id" in result["content"][0]["text"]
    assert "Memory content" in result["content"][0]["text"]
    assert "RequestId" not in result["content"][0]["text"]


def test_delete_memory_response_filtering(provider, mock_boto3_client):
    """Test that delete action filters out metadata from the response."""
    # Set up mock response with both relevant data and metadata
    mock_boto3_client["bedrock_agent_core"].delete_memory_record.return_value = {
        "memoryRecordId": "test-record-id",
        "ResponseMetadata": {"RequestId": "test-request-id"},
    }

    # Call the method
    result = provider.agent_core_memory(action="delete", memory_record_id="test-record-id")

    # Verify only relevant data is included in the response
    assert result["status"] == "success"
    assert "test-record-id" in result["content"][0]["text"]
    assert "RequestId" not in result["content"][0]["text"]


def test_unsupported_action(provider):
    """Test agent_core_memory method with an unsupported action."""
    # Call the method with an unsupported action
    result = provider.agent_core_memory(action="UnsupportedAction")

    # Verify error response
    assert result["status"] == "error"
    assert "is not supported" in result["content"][0]["text"]
    assert "record" in result["content"][0]["text"]
    assert "retrieve" in result["content"][0]["text"]
    assert "list" in result["content"][0]["text"]
    assert "get" in result["content"][0]["text"]
    assert "delete" in result["content"][0]["text"]


def test_missing_required_parameters(provider):
    """Test agent_core_memory method with missing required parameters."""
    # Call record action without content
    result = provider.agent_core_memory(action="record")

    # Verify error response
    assert result["status"] == "error"
    assert "parameters are required" in result["content"][0]["text"]
    assert "content" in result["content"][0]["text"]


def test_api_error_handling(provider, mock_boto3_client):
    """Test handling of API errors."""
    # Set up mock to raise an exception
    mock_boto3_client["bedrock_agent_core"].create_event.side_effect = Exception("API error")

    # Call the method
    result = provider.agent_core_memory(action="record", content="Hello")

    # Verify error response
    assert result["status"] == "error"
    assert "API error" in result["content"][0]["text"]


def test_boto3_session_support(mock_boto3_client):
    """Test that provider can use a custom boto3 Session."""
    # Create a mock boto3 Session
    mock_session = mock.MagicMock()
    mock_session_client = mock.MagicMock()
    mock_session.client.return_value = mock_session_client

    # Initialize provider with the mock session
    provider = AgentCoreMemoryToolProvider(
        memory_id="test-memory-id",
        actor_id="test-actor-id",
        session_id="test-session-id",
        namespace="test-namespace",
        boto_session=mock_session,
    )

    # The client is already initialized in the constructor
    client = provider.bedrock_agent_core_client

    # Verify the session's client method was called with correct parameters
    mock_session.client.assert_called_with(
        "bedrock-agentcore",
        region_name=provider.region,
        config=provider.client_config,
    )

    # Verify that boto3.client was not called directly
    mock_boto3_client["client"].assert_not_called()

    # Verify that the client is the one returned by the session
    assert client == mock_session_client
