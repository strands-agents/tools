"""Tests for the MemoryServiceClient class in memory.py."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from botocore.config import Config as BotocoreConfig

from strands_tools.memory import MemoryServiceClient


@pytest.fixture
def mock_boto3_session():
    """Create a mock boto3 session."""
    session = MagicMock()

    # Mock agent client
    agent_client = MagicMock()
    session.client.return_value = agent_client

    return session


@patch("boto3.Session")
def test_client_init_default(mock_session):
    """Test client initialization with default parameters."""
    # Create session mock
    session_instance = MagicMock()
    mock_session.return_value = session_instance

    # Initialize client
    client = MemoryServiceClient()

    # Verify default region
    assert client.region == os.environ.get("AWS_REGION", "us-west-2")
    assert client.profile_name is None

    # Verify session was created
    mock_session.assert_called_once()


@patch("boto3.Session")
def test_client_init_custom_region(mock_session):
    """Test client initialization with custom region."""
    # Create session mock
    session_instance = MagicMock()
    mock_session.return_value = session_instance

    # Initialize client
    client = MemoryServiceClient(region="us-east-1")

    # Verify custom region
    assert client.region == "us-east-1"
    assert client.profile_name is None


@patch("boto3.Session")
def test_client_init_custom_profile(mock_session):
    """Test client initialization with custom profile."""
    # Create session mock
    session_instance = MagicMock()
    mock_session.return_value = session_instance

    # Initialize client
    client = MemoryServiceClient(profile_name="test-profile")

    # Verify profile
    assert client.region == os.environ.get("AWS_REGION", "us-west-2")
    assert client.profile_name == "test-profile"

    # Verify session was created with profile
    mock_session.assert_called_once_with(profile_name="test-profile")


@patch("boto3.Session")
def test_agent_client_property(mock_session):
    """Test the agent_client property."""
    # Create session mock
    session_instance = MagicMock()
    agent_client = MagicMock()
    session_instance.client.return_value = agent_client
    mock_session.return_value = session_instance

    # Initialize client
    client = MemoryServiceClient()

    # Access the property
    result = client.agent_client

    # Verify the client was called with the correct arguments
    session_instance.client.assert_called_once()
    args, kwargs = session_instance.client.call_args

    assert args[0] == "bedrock-agent"
    assert kwargs["region_name"] == client.region
    assert "config" in kwargs
    config = kwargs["config"]
    assert isinstance(config, BotocoreConfig)
    assert config.user_agent_extra == "strands-agents-memory"

    # Verify same client is returned on second access
    session_instance.client.reset_mock()
    result2 = client.agent_client
    assert result is result2

    # Verify client was not created again
    session_instance.client.assert_not_called()


@patch("boto3.Session")
def test_runtime_client_property(mock_session):
    """Test the runtime_client property."""
    # Create session mock
    session_instance = MagicMock()
    runtime_client = MagicMock()
    session_instance.client.return_value = runtime_client
    mock_session.return_value = session_instance

    # Initialize client
    client = MemoryServiceClient()

    # Access the property
    result = client.runtime_client

    # Verify the client was called with the correct arguments
    session_instance.client.assert_called_once()
    args, kwargs = session_instance.client.call_args

    assert args[0] == "bedrock-agent-runtime"
    assert kwargs["region_name"] == client.region
    assert "config" in kwargs
    config = kwargs["config"]
    assert isinstance(config, BotocoreConfig)
    assert config.user_agent_extra == "strands-agents-memory"

    # Verify same client is returned on second access
    session_instance.client.reset_mock()
    result2 = client.runtime_client
    assert result is result2

    # Verify client was not created again
    session_instance.client.assert_not_called()


@patch("boto3.Session")
def test_get_data_source_id(mock_session):
    """Test get_data_source_id method."""
    # Create session mock
    session_instance = MagicMock()
    agent_client = MagicMock()
    session_instance.client.return_value = agent_client
    mock_session.return_value = session_instance

    # Mock response
    data_sources = {"dataSourceSummaries": [{"dataSourceId": "ds123"}]}
    agent_client.list_data_sources.return_value = data_sources

    # Initialize client
    client = MemoryServiceClient()

    # Call method
    result = client.get_data_source_id("kb123")

    # Verify response
    assert result == "ds123"

    # Verify API call
    agent_client.list_data_sources.assert_called_once_with(knowledgeBaseId="kb123")


@patch("boto3.Session")
def test_get_data_source_id_no_sources(mock_session):
    """Test get_data_source_id method with no data sources."""
    # Create session mock
    session_instance = MagicMock()
    agent_client = MagicMock()
    session_instance.client.return_value = agent_client
    mock_session.return_value = session_instance

    # Mock empty response
    agent_client.list_data_sources.return_value = {"dataSourceSummaries": []}

    # Initialize client
    client = MemoryServiceClient()

    # Call method and verify exception
    with pytest.raises(ValueError, match=r"No data sources found"):
        client.get_data_source_id("kb123")


@patch("boto3.Session")
def test_list_documents_with_defaults(mock_session):
    """Test list_documents method with default parameters."""
    # Create session mock
    session_instance = MagicMock()
    agent_client = MagicMock()
    session_instance.client.return_value = agent_client
    mock_session.return_value = session_instance

    # Mock get_data_source_id
    agent_client.list_data_sources.return_value = {"dataSourceSummaries": [{"dataSourceId": "ds123"}]}

    # Mock get_data_source for _detect_data_source_type method
    agent_client.get_data_source.return_value = {"dataSource": {"dataSourceConfiguration": {"type": "CUSTOM"}}}

    # Initialize client
    client = MemoryServiceClient()

    # Call method
    client.list_documents("kb123")

    # Verify API call
    agent_client.list_knowledge_base_documents.assert_called_once_with(knowledgeBaseId="kb123", dataSourceId="ds123")


@patch("boto3.Session")
def test_list_documents_with_params(mock_session):
    """Test list_documents method with all parameters."""
    # Create session mock
    session_instance = MagicMock()
    agent_client = MagicMock()
    session_instance.client.return_value = agent_client
    mock_session.return_value = session_instance

    # Initialize client
    client = MemoryServiceClient()

    # Call method
    client.list_documents("kb123", "ds456", 10, "token123")

    # Verify API call
    agent_client.list_knowledge_base_documents.assert_called_once_with(
        knowledgeBaseId="kb123", dataSourceId="ds456", maxResults=10, nextToken="token123"
    )


@patch("boto3.Session")
def test_get_document(mock_session):
    """Test get_document method."""
    # Create session mock
    session_instance = MagicMock()
    agent_client = MagicMock()
    session_instance.client.return_value = agent_client
    mock_session.return_value = session_instance

    # Mock get_data_source_id
    agent_client.list_data_sources.return_value = {"dataSourceSummaries": [{"dataSourceId": "ds123"}]}

    # Mock get_data_source for _detect_data_source_type method
    agent_client.get_data_source.return_value = {"dataSource": {"dataSourceConfiguration": {"type": "CUSTOM"}}}

    # Initialize client
    client = MemoryServiceClient()

    # Call method
    client.get_document("kb123", None, "doc123")

    # Verify API call
    agent_client.get_knowledge_base_documents.assert_called_once_with(
        knowledgeBaseId="kb123",
        dataSourceId="ds123",
        documentIdentifiers=[{"dataSourceType": "CUSTOM", "custom": {"id": "doc123"}}],
    )


@patch("boto3.Session")
def test_store_document(mock_session):
    """Test store_document method."""
    # Create session mock
    session_instance = MagicMock()
    agent_client = MagicMock()
    session_instance.client.return_value = agent_client
    mock_session.return_value = session_instance

    # Mock get_data_source_id
    agent_client.list_data_sources.return_value = {"dataSourceSummaries": [{"dataSourceId": "ds123"}]}

    # Mock get_data_source for _detect_data_source_type method
    agent_client.get_data_source.return_value = {"dataSource": {"dataSourceConfiguration": {"type": "CUSTOM"}}}

    # Mock get_data_source for _detect_data_source_type method
    agent_client.get_data_source.return_value = {"dataSource": {"dataSourceConfiguration": {"type": "CUSTOM"}}}

    # Mock ingest response
    agent_client.ingest_knowledge_base_documents.return_value = {"status": "success"}

    # Initialize client
    client = MemoryServiceClient()

    # Call method
    response, doc_id, doc_title = client.store_document("kb123", None, "test content", "Test Title")

    # Verify response
    assert response == {"status": "success"}
    assert "memory_" in doc_id  # Verify ID format
    assert doc_title == "Test Title"

    # Verify API call structure
    call_args = agent_client.ingest_knowledge_base_documents.call_args[1]
    assert call_args["knowledgeBaseId"] == "kb123"
    assert call_args["dataSourceId"] == "ds123"
    assert len(call_args["documents"]) == 1

    # Verify document content
    doc = call_args["documents"][0]
    assert doc["content"]["dataSourceType"] == "CUSTOM"
    assert doc["content"]["custom"]["sourceType"] == "IN_LINE"

    # Verify content format
    content_json = doc["content"]["custom"]["inlineContent"]["textContent"]["data"]
    content_data = json.loads(content_json)
    assert content_data["title"] == "Test Title"
    assert content_data["action"] == "store"
    assert content_data["content"] == "test content"


@patch("boto3.Session")
def test_store_document_no_title(mock_session):
    """Test store_document method with auto-generated title."""
    # Create session mock
    session_instance = MagicMock()
    agent_client = MagicMock()
    session_instance.client.return_value = agent_client
    mock_session.return_value = session_instance

    # Mock get_data_source_id
    agent_client.list_data_sources.return_value = {"dataSourceSummaries": [{"dataSourceId": "ds123"}]}

    # Mock get_data_source for _detect_data_source_type method
    agent_client.get_data_source.return_value = {"dataSource": {"dataSourceConfiguration": {"type": "CUSTOM"}}}

    # Mock get_data_source for _detect_data_source_type method
    agent_client.get_data_source.return_value = {"dataSource": {"dataSourceConfiguration": {"type": "CUSTOM"}}}

    # Mock ingest response
    agent_client.ingest_knowledge_base_documents.return_value = {"status": "success"}

    # Initialize client
    client = MemoryServiceClient()

    # Call method without title
    response, doc_id, doc_title = client.store_document("kb123", None, "test content")

    # Verify title format
    assert "Strands Memory" in doc_title

    # Verify API call structure
    call_args = agent_client.ingest_knowledge_base_documents.call_args[1]

    # Verify document content
    doc = call_args["documents"][0]

    # Verify content format
    content_json = doc["content"]["custom"]["inlineContent"]["textContent"]["data"]
    content_data = json.loads(content_json)
    assert content_data["title"] == doc_title


@patch("boto3.Session")
def test_delete_document(mock_session):
    """Test delete_document method."""
    # Create session mock
    session_instance = MagicMock()
    agent_client = MagicMock()
    session_instance.client.return_value = agent_client
    mock_session.return_value = session_instance

    # Mock get_data_source_id
    agent_client.list_data_sources.return_value = {"dataSourceSummaries": [{"dataSourceId": "ds123"}]}

    # Mock get_data_source for _detect_data_source_type method
    agent_client.get_data_source.return_value = {"dataSource": {"dataSourceConfiguration": {"type": "CUSTOM"}}}

    # Mock get_data_source for _detect_data_source_type method
    agent_client.get_data_source.return_value = {"dataSource": {"dataSourceConfiguration": {"type": "CUSTOM"}}}

    # Mock delete response
    agent_client.delete_knowledge_base_documents.return_value = {"status": "success"}

    # Initialize client
    client = MemoryServiceClient()

    # Call method
    response = client.delete_document("kb123", None, "doc123")

    # Verify response
    assert response == {"status": "success"}

    # Verify API call
    agent_client.delete_knowledge_base_documents.assert_called_once_with(
        knowledgeBaseId="kb123",
        dataSourceId="ds123",
        documentIdentifiers=[{"dataSourceType": "CUSTOM", "custom": {"id": "doc123"}}],
    )


@patch("boto3.Session")
def test_retrieve(mock_session):
    """Test retrieve method."""
    # Create session mock
    session_instance = MagicMock()
    runtime_client = MagicMock()
    session_instance.client.return_value = runtime_client
    mock_session.return_value = session_instance

    # Mock retrieve response
    runtime_client.retrieve.return_value = {"retrievalResults": []}

    # Initialize client
    client = MemoryServiceClient()

    # Call method
    result = client.retrieve("kb123", "test query", 10)

    # Verify response
    assert result == {"retrievalResults": []}

    # Verify API call
    runtime_client.retrieve.assert_called_once_with(
        retrievalQuery={"text": "test query"},
        knowledgeBaseId="kb123",
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": 10},
        },
    )


@patch("boto3.Session")
def test_retrieve_with_token(mock_session):
    """Test retrieve method with pagination token."""
    # Create session mock
    session_instance = MagicMock()
    runtime_client = MagicMock()
    session_instance.client.return_value = runtime_client
    mock_session.return_value = session_instance

    # Initialize client
    client = MemoryServiceClient()

    # Call method
    client.retrieve("kb123", "test query", 10, "token123")

    # Verify API call includes token
    call_args = runtime_client.retrieve.call_args[1]
    assert call_args["nextToken"] == "token123"
