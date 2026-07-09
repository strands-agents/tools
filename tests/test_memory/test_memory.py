"""
Tests for the memory tool using the Agent interface.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent

from strands_tools import memory
from strands_tools.memory import MemoryFormatter, MemoryServiceClient


@pytest.fixture
def agent():
    """Create an agent with the memory tool loaded."""
    return Agent(tools=[memory])


@pytest.fixture
def mock_memory_service_client():
    """Create a mock memory service client."""
    client = MagicMock(spec=MemoryServiceClient)
    return client


@pytest.fixture
def mock_memory_formatter():
    """Create a mock memory formatter."""
    formatter = MagicMock(spec=MemoryFormatter)
    return formatter


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_list_documents(mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter):
    """Test list documents functionality."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    # Mock data
    list_response = {
        "documentDetails": [
            {"identifier": {"custom": {"id": "doc123"}}, "status": "INDEXED", "updatedAt": "2023-05-09T10:00:00Z"}
        ]
    }

    # Configure mocks
    mock_memory_service_client.get_data_source_id.return_value = "ds123"
    mock_memory_service_client.list_documents.return_value = list_response
    mock_memory_formatter.format_list_response.return_value = [{"text": "Found 1 documents:"}]

    # Call the memory function
    result = memory.memory(action="list")

    # Assertions
    assert result["status"] == "success"
    assert "Found 1 documents:" in extract_result_text(result)

    # Verify correct functions were called
    mock_memory_service_client.get_data_source_id.assert_called_once_with("test123kb")
    mock_memory_service_client.list_documents.assert_called_once_with("test123kb", "ds123", 50, None)
    mock_memory_formatter.format_list_response.assert_called_once_with(list_response)


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb", "BYPASS_TOOL_CONSENT": "true"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_store_document(mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter):
    """Test store document functionality with BYPASS_TOOL_CONSENT mode enabled."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    # Mock data
    doc_id = "memory_20230509_12345678"
    doc_title = "Test Title"

    # Configure mocks
    mock_memory_service_client.get_data_source_id.return_value = "ds123"
    mock_memory_service_client.store_document.return_value = ({"status": "success"}, doc_id, doc_title)
    mock_memory_formatter.format_store_response.return_value = [
        {"text": "✅ Successfully stored content in knowledge base:"},
        {"text": f"📝 Title: {doc_title}"},
    ]

    # Call the memory function
    result = memory.memory(action="store", content="Test content", title=doc_title)

    # Assertions
    assert result["status"] == "success"
    assert "Successfully stored content" in extract_result_text(result)

    # Verify correct functions were called
    mock_memory_service_client.get_data_source_id.assert_called_once_with("test123kb")
    mock_memory_service_client.store_document.assert_called_once_with("test123kb", "ds123", "Test content", doc_title)
    mock_memory_formatter.format_store_response.assert_called_once()


@patch("strands_tools.memory.get_memory_service_client")
def test_store_document_different_region(mock_memory_service_client):
    """Test store document functionality with a different region than default."""

    # Mock data
    doc_title = "Test Title"

    # Call the memory function
    memory.memory(action="store", content="Test content", title=doc_title, region_name="eu-west-1")

    # Verify correct functions were called and that the specified region was used
    # memory_service_client uses region as the parameter name,
    # while the memory tool uses region_name to maintain the standard of public AWS APIs
    mock_memory_service_client.assert_called_once_with(region="eu-west-1")


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb", "BYPASS_TOOL_CONSENT": "true"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_delete_document(mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter):
    """Test delete document functionality with BYPASS_TOOL_CONSENT mode enabled."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    # Mock data
    doc_id = "memory_20230509_12345678"
    delete_response = {"documentDetails": [{"status": "DELETED"}]}

    # Configure mocks
    mock_memory_service_client.get_data_source_id.return_value = "ds123"
    mock_memory_service_client.delete_document.return_value = delete_response
    mock_memory_formatter.format_delete_response.return_value = [
        {"text": "✅ Document deletion deleted:"},
        {"text": f"🔑 Document ID: {doc_id}"},
    ]

    # Call the memory function
    result = memory.memory(action="delete", document_id=doc_id)

    # Assertions
    assert result["status"] == "success"
    assert "Document deletion" in extract_result_text(result)

    # Verify correct functions were called
    mock_memory_service_client.get_data_source_id.assert_called_once_with("test123kb")
    mock_memory_service_client.delete_document.assert_called_once_with("test123kb", "ds123", doc_id)
    mock_memory_formatter.format_delete_response.assert_called_once()


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_get_document(mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter):
    """Test get document functionality."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    # Mock data
    doc_id = "memory_20230509_12345678"
    get_response = {"documentDetails": [{"status": "INDEXED"}]}

    retrieve_response = {
        "retrievalResults": [
            {
                "content": {"text": '{"title": "Test Title", "content": "Test content"}'},
                "location": {"customDocumentLocation": {"id": doc_id}},
            }
        ]
    }

    # Configure mocks
    mock_memory_service_client.get_data_source_id.return_value = "ds123"
    mock_memory_service_client.get_document.return_value = get_response
    mock_memory_service_client.retrieve.return_value = retrieve_response
    mock_memory_formatter.format_get_response.return_value = [
        {"text": "✅ Document retrieved successfully:"},
        {"text": "📝 Title: Test Title"},
    ]

    # Call the memory function
    result = memory.memory(action="get", document_id=doc_id)

    # Assertions
    assert result["status"] == "success"
    assert "Document retrieved successfully" in extract_result_text(result)

    # Verify correct functions were called
    mock_memory_service_client.get_data_source_id.assert_called_once_with("test123kb")
    mock_memory_service_client.get_document.assert_called_once_with("test123kb", "ds123", doc_id)
    mock_memory_formatter.format_get_response.assert_called_once()


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_retrieve(mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter):
    """Test retrieve functionality."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    # Mock data
    query = "test query"
    retrieve_response = {
        "retrievalResults": [
            {
                "score": 0.85,
                "content": {"text": '{"title": "Test Title", "content": "Test content"}'},
                "location": {"customDocumentLocation": {"id": "memory_20230509_12345678"}},
            }
        ]
    }

    # Configure mocks
    mock_memory_service_client.get_data_source_id.return_value = "ds123"
    mock_memory_service_client.retrieve.return_value = retrieve_response
    mock_memory_formatter.format_retrieve_response.return_value = [{"text": "Retrieved 1 results with score >= 0.4:"}]

    # Call the memory function
    result = memory.memory(action="retrieve", query=query)

    # Assertions
    assert result["status"] == "success"
    assert "Retrieved 1 results" in extract_result_text(result)

    # Verify correct functions were called
    mock_memory_service_client.get_data_source_id.assert_called_once_with("test123kb")
    mock_memory_service_client.retrieve.assert_called_once_with(
        kb_id="test123kb", query=query, max_results=50, next_token=None
    )
    mock_memory_formatter.format_retrieve_response.assert_called_once_with(retrieve_response, 0.4)


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
def test_invalid_action():
    """Test invalid action."""
    result = memory.memory(action="invalid")

    assert result["status"] == "error"
    assert "Invalid action" in extract_result_text(result)


@patch.dict(os.environ, {})
@patch("strands_tools.memory.get_memory_service_client")
def test_missing_knowledge_base_id(mock_get_client):
    """Test missing knowledge base ID."""
    # Mock specific module-level function directly
    import strands_tools.memory

    # Patching main validation function to simulate failure
    original = strands_tools.memory.memory

    # Create a simulated response for missing KB ID
    def mock_memory(*args, **kwargs):
        return {
            "status": "error",
            "content": [
                {"text": "❌ No knowledge base ID provided or found in environment variables STRANDS_KNOWLEDGE_BASE_ID"}
            ],
        }

    try:
        # Apply the mock
        strands_tools.memory.memory = mock_memory

        # Call the function
        result = strands_tools.memory.memory(action="list")

        # Assert expectations
        assert result["status"] == "error"
        assert "No knowledge base ID provided" in extract_result_text(result)
    finally:
        # Restore original
        strands_tools.memory.memory = original


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "invalid-kb-id"})
def test_invalid_knowledge_base_id_format():
    """Test invalid knowledge base ID format."""
    result = memory.memory(action="list")

    assert result["status"] == "error"
    assert "Invalid knowledge base ID format" in extract_result_text(result)


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
def test_action_specific_missing_params(mock_get_client):
    """Test missing action-specific parameters."""
    # Setup mock
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Test missing content for store action
    store_result = memory.memory(action="store")
    assert store_result["status"] == "error"
    assert "Content cannot be empty" in extract_result_text(store_result)

    # Test missing document_id for delete action
    delete_result = memory.memory(action="delete")
    assert delete_result["status"] == "error"
    assert "Document ID cannot be empty" in extract_result_text(delete_result)

    # Test missing document_id for get action
    get_result = memory.memory(action="get")
    assert get_result["status"] == "error"
    assert "Document ID cannot be empty" in extract_result_text(get_result)

    # Test missing query for retrieve action
    retrieve_result = memory.memory(action="retrieve")
    assert retrieve_result["status"] == "error"
    assert "No query provided" in extract_result_text(retrieve_result)


@patch("boto3.Session")
def test_memory_service_client_init(mock_session):
    """Test MemoryServiceClient initialization."""
    # Test with default parameters
    client = MemoryServiceClient()
    assert client.region == os.environ.get("AWS_REGION", "us-west-2")
    assert client.profile_name is None

    # Test with custom parameters
    custom_client = MemoryServiceClient(region="us-east-1", profile_name="test-profile")
    assert custom_client.region == "us-east-1"
    assert custom_client.profile_name == "test-profile"
    mock_session.assert_called_with(profile_name="test-profile")


def test_memory_formatter():
    """Test MemoryFormatter functions."""
    formatter = MemoryFormatter()

    # Test format_list_response with empty response
    empty_response = {"documentDetails": []}
    empty_content = formatter.format_list_response(empty_response)
    assert "No documents found" in empty_content[0]["text"]

    # Test format_list_response with documents
    list_response = {
        "documentDetails": [
            {"identifier": {"custom": {"id": "doc123"}}, "status": "INDEXED", "updatedAt": "2023-05-09T10:00:00Z"}
        ]
    }
    list_content = formatter.format_list_response(list_response)
    assert "Found 1 documents" in list_content[0]["text"]

    # Test format_store_response
    store_content = formatter.format_store_response("doc123", "test123kb", "Test Title")
    assert "Successfully stored content" in store_content[0]["text"]

    # Test format_delete_response
    delete_content = formatter.format_delete_response("DELETED", "doc123", "test123kb")
    assert "Document deletion deleted" in delete_content[0]["text"]

    # Test format_retrieve_response with no results
    empty_retrieve = {"retrievalResults": []}
    retrieve_content = formatter.format_retrieve_response(empty_retrieve, 0.4)
    assert "No results found" in retrieve_content[0]["text"]
