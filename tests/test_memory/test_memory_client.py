"""
Tests for the MemoryServiceClient class in memory.py.
"""

import json
import os
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from strands_tools.memory import MemoryServiceClient


@pytest.fixture
def mock_boto3_session():
    """Create a mock boto3 session."""
    session = MagicMock()

    # Mock agent client
    agent_client = MagicMock()
    session.client.return_value = agent_client

    return session


def test_client_init_default():
    """Test client initialization with default parameters."""
    # Initialize client
    client = MemoryServiceClient()

    # Verify default region
    assert client.region == os.environ.get("AWS_REGION", "us-west-2")
    assert client.profile_name is None
    assert client.session is None


def test_client_init_custom_region():
    """Test client initialization with custom region."""
    # Initialize client
    client = MemoryServiceClient(region="us-east-1")

    # Verify custom region
    assert client.region == "us-east-1"
    assert client.profile_name is None
    assert client.session is None


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
    assert client.session == session_instance


def test_client_init_with_session():
    """Test client initialization with provided session."""
    # Create a mock session
    mock_session = MagicMock()

    # Initialize client with session
    client = MemoryServiceClient(session=mock_session)

    # Verify session is stored
    assert client.session == mock_session
    assert client.region == os.environ.get("AWS_REGION", "us-west-2")
    assert client.profile_name is None


def test_client_init_with_session_and_region():
    """Test client initialization with both session and custom region."""
    # Create a mock session
    mock_session = MagicMock()

    # Initialize client with session and region
    client = MemoryServiceClient(session=mock_session, region="eu-west-1")

    # Verify both are stored correctly
    assert client.session == mock_session
    assert client.region == "eu-west-1"
    assert client.profile_name is None


@patch("boto3.Session")
def test_client_init_profile_overrides_session(mock_session):
    """Test that profile_name creates a new session even if session is provided."""
    # Create session mocks
    provided_session = MagicMock()
    created_session = MagicMock()
    mock_session.return_value = created_session

    # Initialize client with both session and profile
    client = MemoryServiceClient(session=provided_session, profile_name="test-profile")

    # Verify profile created a new session
    mock_session.assert_called_once_with(profile_name="test-profile")
    assert client.session == created_session
    assert client.profile_name == "test-profile"


def test_agent_client_property_with_no_session():
    """Test the agent_client property without provided session."""
    # Create a mock client
    client = MemoryServiceClient()

    # Create mocks
    mock_session = MagicMock()
    mock_agent_client = MagicMock()
    mock_session.client.return_value = mock_agent_client

    # Mock the property to simulate lazy session creation
    with patch("boto3.Session", return_value=mock_session):
        with patch.object(type(client), "agent_client", new_callable=PropertyMock) as mock_property:
            # Configure the property to return the mock client
            mock_property.return_value = mock_agent_client

            # Access the property
            result = client.agent_client

            # Verify the mock was returned
            assert result == mock_agent_client


def test_agent_client_property_with_session():
    """Test the agent_client property with provided session."""
    # Create mock session and client
    mock_session = MagicMock()
    agent_client = MagicMock()
    mock_session.client.return_value = agent_client

    # Initialize client with session
    client = MemoryServiceClient(session=mock_session)

    # Mock the property
    with patch.object(type(client), "agent_client", new_callable=PropertyMock) as mock_property:
        # Configure the property to return the mock client
        mock_property.return_value = agent_client

        # Access the property
        result = client.agent_client

        # Verify the mock was returned
        assert result == agent_client


def test_runtime_client_property_with_no_session():
    """Test the runtime_client property without provided session."""
    # Create a mock client
    client = MemoryServiceClient()

    # Create mocks
    mock_session = MagicMock()
    mock_runtime_client = MagicMock()
    mock_session.client.return_value = mock_runtime_client

    # Mock the property to simulate lazy session creation
    with patch("boto3.Session", return_value=mock_session):
        with patch.object(type(client), "runtime_client", new_callable=PropertyMock) as mock_property:
            # Configure the property to return the mock client
            mock_property.return_value = mock_runtime_client

            # Access the property
            result = client.runtime_client

            # Verify the mock was returned
            assert result == mock_runtime_client


def test_runtime_client_property_with_session():
    """Test the runtime_client property with provided session."""
    # Create mock session and client
    mock_session = MagicMock()
    runtime_client = MagicMock()
    mock_session.client.return_value = runtime_client

    # Initialize client with session
    client = MemoryServiceClient(session=mock_session)

    # Mock the property
    with patch.object(type(client), "runtime_client", new_callable=PropertyMock) as mock_property:
        # Configure the property to return the mock client
        mock_property.return_value = runtime_client

        # Access the property
        result = client.runtime_client

        # Verify the mock was returned
        assert result == runtime_client


def test_get_data_source_id():
    """Test get_data_source_id method."""
    # Create client
    client = MemoryServiceClient()

    # Mock agent client
    mock_agent_client = MagicMock()
    mock_agent_client.list_data_sources.return_value = {"dataSourceSummaries": [{"dataSourceId": "ds123"}]}

    # Mock the agent_client property
    with patch.object(type(client), "agent_client", new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_agent_client

        # Call method
        result = client.get_data_source_id("kb123")

        # Verify response
        assert result == "ds123"

        # Verify API call
        mock_agent_client.list_data_sources.assert_called_once_with(knowledgeBaseId="kb123")


def test_get_data_source_id_no_sources():
    """Test get_data_source_id method with no data sources."""
    # Create client
    client = MemoryServiceClient()

    # Mock agent client with empty response
    mock_agent_client = MagicMock()
    mock_agent_client.list_data_sources.return_value = {"dataSourceSummaries": []}

    # Mock the agent_client property
    with patch.object(type(client), "agent_client", new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_agent_client

        # Call method and verify exception
        with pytest.raises(ValueError, match=r"No data sources found"):
            client.get_data_source_id("kb123")


def test_list_documents_with_defaults():
    """Test list_documents method with default parameters."""
    # Create client
    client = MemoryServiceClient()

    # Mock agent client
    mock_agent_client = MagicMock()
    mock_agent_client.list_data_sources.return_value = {"dataSourceSummaries": [{"dataSourceId": "ds123"}]}
    mock_agent_client.list_knowledge_base_documents.return_value = {"documents": []}

    # Mock the agent_client property
    with patch.object(type(client), "agent_client", new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_agent_client

        # Call method
        client.list_documents("kb123")

        # Verify API call
        mock_agent_client.list_knowledge_base_documents.assert_called_once_with(
            knowledgeBaseId="kb123", dataSourceId="ds123"
        )


def test_list_documents_with_params():
    """Test list_documents method with all parameters."""
    # Create client
    client = MemoryServiceClient()

    # Mock agent client
    mock_agent_client = MagicMock()
    mock_agent_client.list_knowledge_base_documents.return_value = {"documents": []}

    # Mock the agent_client property
    with patch.object(type(client), "agent_client", new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_agent_client

        # Call method
        client.list_documents("kb123", "ds456", 10, "token123")

        # Verify API call
        mock_agent_client.list_knowledge_base_documents.assert_called_once_with(
            knowledgeBaseId="kb123",
            dataSourceId="ds456",
            maxResults=10,
            nextToken="token123",
        )


def test_get_document():
    """Test get_document method."""
    # Create client
    client = MemoryServiceClient()

    # Mock agent client
    mock_agent_client = MagicMock()
    mock_agent_client.list_data_sources.return_value = {"dataSourceSummaries": [{"dataSourceId": "ds123"}]}
    mock_agent_client.get_knowledge_base_documents.return_value = {"documentDetails": []}

    # Mock the agent_client property
    with patch.object(type(client), "agent_client", new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_agent_client

        # Call method
        client.get_document("kb123", None, "doc123")

        # Verify API call
        mock_agent_client.get_knowledge_base_documents.assert_called_once_with(
            knowledgeBaseId="kb123",
            dataSourceId="ds123",
            documentIdentifiers=[{"dataSourceType": "CUSTOM", "custom": {"id": "doc123"}}],
        )


def test_store_document():
    """Test store_document method."""
    # Create client
    client = MemoryServiceClient()

    # Mock agent client
    mock_agent_client = MagicMock()
    mock_agent_client.list_data_sources.return_value = {"dataSourceSummaries": [{"dataSourceId": "ds123"}]}
    mock_agent_client.ingest_knowledge_base_documents.return_value = {"status": "success"}

    # Mock the agent_client property
    with patch.object(type(client), "agent_client", new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_agent_client

        # Call method
        response, doc_id, doc_title = client.store_document("kb123", None, "test content", "Test Title")

        # Verify response
        assert response == {"status": "success"}
        assert "memory_" in doc_id  # Verify ID format
        assert doc_title == "Test Title"

        # Verify API call structure
        call_args = mock_agent_client.ingest_knowledge_base_documents.call_args[1]
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


def test_store_document_no_title():
    """Test store_document method with auto-generated title."""
    # Create client
    client = MemoryServiceClient()

    # Mock agent client
    mock_agent_client = MagicMock()
    mock_agent_client.list_data_sources.return_value = {"dataSourceSummaries": [{"dataSourceId": "ds123"}]}
    mock_agent_client.ingest_knowledge_base_documents.return_value = {"status": "success"}

    # Mock the agent_client property
    with patch.object(type(client), "agent_client", new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_agent_client

        # Call method without title
        response, doc_id, doc_title = client.store_document("kb123", None, "test content")

        # Verify title format
        assert "Strands Memory" in doc_title

        # Verify API call structure
        call_args = mock_agent_client.ingest_knowledge_base_documents.call_args[1]

        # Verify document content
        doc = call_args["documents"][0]

        # Verify content format
        content_json = doc["content"]["custom"]["inlineContent"]["textContent"]["data"]
        content_data = json.loads(content_json)
        assert content_data["title"] == doc_title


def test_delete_document():
    """Test delete_document method."""
    # Create client
    client = MemoryServiceClient()

    # Mock agent client
    mock_agent_client = MagicMock()
    mock_agent_client.list_data_sources.return_value = {"dataSourceSummaries": [{"dataSourceId": "ds123"}]}
    mock_agent_client.delete_knowledge_base_documents.return_value = {"status": "success"}

    # Mock the agent_client property
    with patch.object(type(client), "agent_client", new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_agent_client

        # Call method
        response = client.delete_document("kb123", None, "doc123")

        # Verify response
        assert response == {"status": "success"}

        # Verify API call
        mock_agent_client.delete_knowledge_base_documents.assert_called_once_with(
            knowledgeBaseId="kb123",
            dataSourceId="ds123",
            documentIdentifiers=[{"dataSourceType": "CUSTOM", "custom": {"id": "doc123"}}],
        )


def test_retrieve():
    """Test retrieve method."""
    # Create client
    client = MemoryServiceClient()

    # Mock runtime client
    mock_runtime_client = MagicMock()
    mock_runtime_client.retrieve.return_value = {"retrievalResults": []}

    # Mock the runtime_client property
    with patch.object(type(client), "runtime_client", new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_runtime_client

        # Call method
        result = client.retrieve("kb123", "test query", 10)

        # Verify response
        assert result == {"retrievalResults": []}

        # Verify API call
        mock_runtime_client.retrieve.assert_called_once_with(
            retrievalQuery={"text": "test query"},
            knowledgeBaseId="kb123",
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 10},
            },
        )


def test_retrieve_with_token():
    """Test retrieve method with pagination token."""
    # Create client
    client = MemoryServiceClient()

    # Mock runtime client
    mock_runtime_client = MagicMock()
    mock_runtime_client.retrieve.return_value = {"retrievalResults": []}

    # Mock the runtime_client property
    with patch.object(type(client), "runtime_client", new_callable=PropertyMock) as mock_property:
        mock_property.return_value = mock_runtime_client

        # Call method
        client.retrieve("kb123", "test query", 10, "token123")

        # Verify API call includes token
        call_args = mock_runtime_client.retrieve.call_args[1]
        assert call_args["nextToken"] == "token123"


def test_all_methods_with_provided_session():
    """Test that all methods work correctly with a provided session."""
    # Create mock session with both clients
    mock_session = MagicMock()
    agent_client = MagicMock()
    runtime_client = MagicMock()

    # Configure session to return appropriate client
    def get_client(service_name, **kwargs):
        if service_name == "bedrock-agent":
            return agent_client
        elif service_name == "bedrock-agent-runtime":
            return runtime_client

    mock_session.client.side_effect = get_client

    # Mock responses
    agent_client.list_data_sources.return_value = {"dataSourceSummaries": [{"dataSourceId": "ds123"}]}
    agent_client.list_knowledge_base_documents.return_value = {"documents": []}
    agent_client.ingest_knowledge_base_documents.return_value = {"status": "success"}
    runtime_client.retrieve.return_value = {"retrievalResults": []}

    # Initialize client with session
    client = MemoryServiceClient(session=mock_session, region="us-east-1")

    # Mock the properties to return the correct clients
    with patch.object(type(client), "agent_client", new_callable=PropertyMock) as mock_agent_prop:
        with patch.object(type(client), "runtime_client", new_callable=PropertyMock) as mock_runtime_prop:
            mock_agent_prop.return_value = agent_client
            mock_runtime_prop.return_value = runtime_client

            # Test various operations
            client.list_documents("kb123")
            client.store_document("kb123", None, "content", "title")
            client.retrieve("kb123", "query")

            # Verify methods were called
            assert agent_client.list_data_sources.called
            assert agent_client.list_knowledge_base_documents.called
            assert runtime_client.retrieve.called
