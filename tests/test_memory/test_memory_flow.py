"""
Tests for complex user confirmation flows, retrieval logic, and error handling in memory.py.

This test file specifically targets areas with lower test coverage identified in the journal:
1. Complex user confirmation paths (lines 667-740)
2. Edge cases in document retrieval (lines 813-831)
3. Additional error handling scenarios (lines 855-918)
4. Complex query handling (lines 938-959)
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from botocore.config import Config as BotocoreConfig

from strands_tools import memory
from strands_tools.memory import MemoryFormatter, MemoryServiceClient


@pytest.fixture
def mock_memory_service_client():
    """Create a mock memory service client."""
    client = MagicMock(spec=MemoryServiceClient)
    # Set up default behaviors
    client.get_data_source_id.return_value = "test_ds_id"
    return client


@pytest.fixture
def mock_memory_formatter():
    """Create a mock memory formatter."""
    formatter = MagicMock(spec=MemoryFormatter)
    return formatter


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
@patch("strands_tools.memory.get_user_input")
def test_store_with_user_confirmation(
    mock_get_user_input, mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter
):
    """Test store with user confirmation flow."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter
    mock_get_user_input.return_value = "y"  # User confirms

    # Configure store operation mocks
    doc_id = "memory_20230509_12345678"
    doc_title = "Test Title"
    mock_memory_service_client.store_document.return_value = ({"status": "success"}, doc_id, doc_title)
    mock_memory_formatter.format_store_response.return_value = [
        {"text": "✅ Successfully stored content in knowledge base:"},
        {"text": f"📝 Title: {doc_title}"},
    ]

    # Call memory function with store action (which should trigger confirmation flow)
    with patch.dict(
        os.environ, {"BYPASS_TOOL_CONSENT": "false"}
    ):  # Ensure BYPASS_TOOL_CONSENT mode is off to trigger confirmation
        result = memory.memory(action="store", content="Test content", title=doc_title)

    # Assertions
    assert result["status"] == "success"
    mock_get_user_input.assert_called_once()
    mock_memory_service_client.store_document.assert_called_once()
    mock_memory_formatter.format_store_response.assert_called_once()


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
@patch("strands_tools.memory.get_user_input")
def test_store_with_user_cancellation(
    mock_get_user_input, mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter
):
    """Test store with user cancellation flow."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter
    mock_get_user_input.side_effect = ["n", "Changed my mind"]  # User cancels and provides reason

    # Call memory function with store action
    with patch.dict(
        os.environ, {"BYPASS_TOOL_CONSENT": "false"}
    ):  # Ensure BYPASS_TOOL_CONSENT mode is off to trigger confirmation
        result = memory.memory(action="store", content="Test content", title="Test Title")

    # Assertions
    assert result["status"] == "error"
    assert "Operation cancelled by the user" in result["content"][0]["text"]
    assert "Changed my mind" in result["content"][0]["text"]
    assert mock_get_user_input.call_count == 2
    mock_memory_service_client.store_document.assert_not_called()


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
@patch("strands_tools.memory.get_user_input")
def test_delete_with_document_preview(
    mock_get_user_input, mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter
):
    """Test delete with document preview and confirmation flow."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter
    mock_get_user_input.return_value = "y"  # User confirms

    doc_id = "memory_20230509_12345678"

    # Setup for document preview before deletion
    get_response = {"documentDetails": [{"status": "INDEXED"}]}

    # Setup for title retrieval
    retrieve_response = {
        "retrievalResults": [
            {
                "content": {"text": '{"title": "Test Document", "content": "Test content"}'},
                "location": {"customDocumentLocation": {"id": doc_id}},
            }
        ]
    }

    # Setup for delete operation
    delete_response = {"documentDetails": [{"status": "DELETED"}]}

    # Configure mocks
    mock_memory_service_client.get_document.return_value = get_response
    mock_memory_service_client.retrieve.return_value = retrieve_response
    mock_memory_service_client.delete_document.return_value = delete_response
    mock_memory_formatter.format_delete_response.return_value = [
        {"text": "✅ Document deletion deleted:"},
        {"text": f"🔑 Document ID: {doc_id}"},
    ]

    # Call memory function with delete action
    with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "false"}):  # Ensure BYPASS_TOOL_CONSENT mode is off
        result = memory.memory(action="delete", document_id=doc_id)

    # Assertions
    assert result["status"] == "success"
    mock_get_user_input.assert_called_once()
    mock_memory_service_client.get_document.assert_called_once()
    mock_memory_service_client.retrieve.assert_called_once()
    mock_memory_service_client.delete_document.assert_called_once()
    mock_memory_formatter.format_delete_response.assert_called_once()


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
@patch("strands_tools.memory.get_user_input")
def test_delete_with_error_in_preview(
    mock_get_user_input, mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter
):
    """Test delete with error during document preview but confirmation flow continues."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter
    mock_get_user_input.return_value = "y"  # User confirms

    doc_id = "memory_20230509_12345678"

    # Make get_document raise an exception to simulate error in preview
    mock_memory_service_client.get_document.side_effect = Exception("Error getting document")

    # Setup for delete operation
    delete_response = {"documentDetails": [{"status": "DELETED"}]}

    # Configure mocks
    mock_memory_service_client.delete_document.return_value = delete_response
    mock_memory_formatter.format_delete_response.return_value = [
        {"text": "✅ Document deletion deleted:"},
        {"text": f"🔑 Document ID: {doc_id}"},
    ]

    # Call memory function with delete action
    with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "false"}):  # Ensure BYPASS_TOOL_CONSENT mode is off
        result = memory.memory(action="delete", document_id=doc_id)

    # Assertions
    assert result["status"] == "success"
    mock_get_user_input.assert_called_once()
    mock_memory_service_client.get_document.assert_called_once()
    mock_memory_service_client.delete_document.assert_called_once()
    mock_memory_formatter.format_delete_response.assert_called_once()


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_get_document_with_retries(
    mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter
):
    """Test get document with retries when not yet indexed."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    doc_id = "memory_20230509_12345678"

    # First get_document call returns not indexed
    not_indexed_response = {
        "documentDetails": [
            {
                "status": "PROCESSING"  # Not INDEXED yet
            }
        ]
    }

    # Second get_document call returns indexed
    indexed_response = {
        "documentDetails": [
            {
                "status": "INDEXED"  # Now indexed
            }
        ]
    }

    # Setup for retrieve response
    retrieve_response = {
        "retrievalResults": [
            {
                "content": {"text": '{"title": "Test Document", "content": "Test content"}'},
                "location": {"customDocumentLocation": {"id": doc_id}},
            }
        ]
    }

    # Configure mocks for retry behavior
    mock_memory_service_client.get_document.side_effect = [not_indexed_response, indexed_response]
    mock_memory_service_client.retrieve.return_value = retrieve_response
    mock_memory_formatter.format_get_response.return_value = [
        {"text": "✅ Document retrieved successfully:"},
        {"text": "📝 Title: Test Document"},
    ]

    # Mock time.sleep to avoid actual waiting during test
    with patch("time.sleep"):
        result = memory.memory(action="get", document_id=doc_id)

    # Assertions
    assert result["status"] == "success"
    assert mock_memory_service_client.get_document.call_count == 2  # Called twice due to retry
    mock_memory_service_client.retrieve.assert_called_once()
    mock_memory_formatter.format_get_response.assert_called_once()


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_get_document_still_not_indexed_after_retries(
    mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter
):
    """Test get document that's still not indexed after max retries."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    doc_id = "memory_20230509_12345678"

    # All get_document calls return not indexed
    not_indexed_response = {
        "documentDetails": [
            {
                "status": "PROCESSING"  # Never becomes INDEXED
            }
        ]
    }

    # Configure mock to always return not indexed
    mock_memory_service_client.get_document.return_value = not_indexed_response

    # Mock time.sleep to avoid actual waiting during test
    with patch("time.sleep"):
        result = memory.memory(action="get", document_id=doc_id)

    # Assertions
    assert result["status"] == "error"
    assert "Document is not indexed" in result["content"][0]["text"]
    assert mock_memory_service_client.get_document.call_count == 4  # Initial call + 3 retries


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_get_document_with_first_retrieve_failing(
    mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter
):
    """Test get document with first retrieve query failing but alternative succeeding."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    doc_id = "memory_20230509_12345678"

    # get_document returns indexed
    indexed_response = {"documentDetails": [{"status": "INDEXED"}]}

    # First retrieve returns no results
    empty_retrieve_response = {"retrievalResults": []}

    # Second retrieve returns results
    success_retrieve_response = {
        "retrievalResults": [
            {
                "content": {"text": '{"title": "Test Document", "content": "Test content"}'},
                "location": {"customDocumentLocation": {"id": doc_id}},
            }
        ]
    }

    # Configure mocks
    mock_memory_service_client.get_document.return_value = indexed_response
    mock_memory_service_client.retrieve.side_effect = [empty_retrieve_response, success_retrieve_response]
    mock_memory_formatter.format_get_response.return_value = [
        {"text": "✅ Document retrieved successfully:"},
        {"text": "📝 Title: Test Document"},
    ]

    result = memory.memory(action="get", document_id=doc_id)

    # Assertions
    assert result["status"] == "success"
    assert mock_memory_service_client.get_document.call_count == 1
    assert mock_memory_service_client.retrieve.call_count == 2  # First fails, second succeeds
    mock_memory_formatter.format_get_response.assert_called_once()

    # Check that the second retrieve call used the document_id as query
    assert mock_memory_service_client.retrieve.call_args_list[1][1]["query"] == doc_id


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_get_document_with_all_retrieves_failing(
    mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter
):
    """Test get document with all retrieve attempts failing."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    doc_id = "memory_20230509_12345678"

    # get_document returns indexed
    indexed_response = {"documentDetails": [{"status": "INDEXED"}]}

    # All retrieve attempts return no results
    empty_retrieve_response = {"retrievalResults": []}

    # Configure mocks
    mock_memory_service_client.get_document.return_value = indexed_response
    mock_memory_service_client.retrieve.return_value = empty_retrieve_response

    result = memory.memory(action="get", document_id=doc_id)

    # Assertions
    assert result["status"] == "error"
    assert "Document found but content could not be retrieved" in result["content"][0]["text"]
    assert mock_memory_service_client.get_document.call_count == 1
    # The implementation actually makes 3 retrieve attempts (documentId: prefix, raw ID, and alt_query)
    assert mock_memory_service_client.retrieve.call_count == 3


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_get_document_with_second_retrieve_returning_multiple_results(
    mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter
):
    """Test get document with second retrieve returning multiple results including the correct one."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    doc_id = "memory_20230509_12345678"

    # get_document returns indexed
    indexed_response = {"documentDetails": [{"status": "INDEXED"}]}

    # First retrieve returns no results
    empty_retrieve_response = {"retrievalResults": []}

    # Second retrieve returns multiple results, one with matching ID
    multi_retrieve_response = {
        "retrievalResults": [
            {"content": {"text": "Some other document"}, "location": {"customDocumentLocation": {"id": "wrong_id"}}},
            {
                "content": {"text": '{"title": "Test Document", "content": "Test content"}'},
                "location": {"customDocumentLocation": {"id": doc_id}},
            },
        ]
    }

    # Configure mocks
    mock_memory_service_client.get_document.return_value = indexed_response
    mock_memory_service_client.retrieve.side_effect = [empty_retrieve_response, multi_retrieve_response]
    mock_memory_formatter.format_get_response.return_value = [
        {"text": "✅ Document retrieved successfully:"},
        {"text": "📝 Title: Test Document"},
    ]

    result = memory.memory(action="get", document_id=doc_id)

    # Assertions
    assert result["status"] == "success"
    assert mock_memory_service_client.retrieve.call_count == 2
    mock_memory_formatter.format_get_response.assert_called_once()


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_get_document_with_non_json_content(
    mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter
):
    """Test get document with non-JSON content in the result."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    doc_id = "memory_20230509_12345678"

    # get_document returns indexed
    indexed_response = {"documentDetails": [{"status": "INDEXED"}]}

    # Retrieve returns raw text instead of JSON
    retrieve_response = {
        "retrievalResults": [
            {
                "content": {"text": "This is plain text content, not JSON"},
                "location": {"customDocumentLocation": {"id": doc_id}},
            }
        ]
    }

    # Configure mocks
    mock_memory_service_client.get_document.return_value = indexed_response
    mock_memory_service_client.retrieve.return_value = retrieve_response

    result = memory.memory(action="get", document_id=doc_id)

    # Assertions
    assert result["status"] == "success"
    assert "Document retrieved successfully" in result["content"][0]["text"]
    # Find the content element that contains the plain text content
    content_texts = [item["text"] for item in result["content"]]
    content_text = "\n".join(content_texts)
    assert "This is plain text content" in content_text
    assert mock_memory_service_client.get_document.call_count == 1
    assert mock_memory_service_client.retrieve.call_count == 1


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_retrieve_with_validation_error(
    mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter
):
    """Test retrieve with validation error in AWS API call."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    # Configure retrieve to raise ValidationException
    mock_memory_service_client.retrieve.side_effect = Exception("ValidationException: The provided knowledgeBaseId")

    result = memory.memory(action="retrieve", query="test query")

    # Assertions
    assert result["status"] == "error"
    assert "Invalid knowledge base ID format" in result["content"][0]["text"]
    mock_memory_service_client.retrieve.assert_called_once()


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_retrieve_with_generic_error(
    mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter
):
    """Test retrieve with generic error in AWS API call."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    # Configure retrieve to raise generic Exception
    mock_memory_service_client.retrieve.side_effect = Exception("Some other AWS error")

    result = memory.memory(action="retrieve", query="test query")

    # Assertions
    assert result["status"] == "error"
    assert "Error during retrieval" in result["content"][0]["text"]
    mock_memory_service_client.retrieve.assert_called_once()


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
@patch("strands_tools.memory.get_memory_formatter")
def test_retrieve_with_results_and_pagination(
    mock_get_formatter, mock_get_client, mock_memory_service_client, mock_memory_formatter
):
    """Test retrieve with results and pagination token."""
    # Setup mocks
    mock_get_client.return_value = mock_memory_service_client
    mock_get_formatter.return_value = mock_memory_formatter

    # Configure retrieve response with nextToken
    retrieve_response = {
        "retrievalResults": [
            {
                "score": 0.85,
                "content": {"text": '{"title": "Test Title", "content": "Test content"}'},
                "location": {"customDocumentLocation": {"id": "doc123"}},
            }
        ],
        "nextToken": "pagination_token_123",
    }

    mock_memory_service_client.retrieve.return_value = retrieve_response
    mock_memory_formatter.format_retrieve_response.return_value = [
        {"text": "Retrieved 1 results with score >= 0.4:"},
        {"text": "➡️ More results available. Use next_token parameter to continue."},
        {"text": "next_token: pagination_token_123"},
    ]

    result = memory.memory(action="retrieve", query="test query", min_score=0.4)

    # Assertions
    assert result["status"] == "success"
    assert "Retrieved 1 results" in result["content"][0]["text"]
    assert "More results available" in result["content"][1]["text"]
    assert "pagination_token_123" in result["content"][2]["text"]
    mock_memory_service_client.retrieve.assert_called_once()
    mock_memory_formatter.format_retrieve_response.assert_called_once_with(retrieve_response, 0.4)


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
def test_memory_formatter_with_various_inputs():
    """Test MemoryFormatter functions with various input types."""
    formatter = MemoryFormatter()

    # Test format_list_response with next_token
    list_response_with_token = {
        "documentDetails": [
            {"identifier": {"custom": {"id": "doc123"}}, "status": "INDEXED", "updatedAt": "2023-05-09T10:00:00Z"}
        ],
        "nextToken": "next_page_token",
    }
    list_result = formatter.format_list_response(list_response_with_token)
    assert "Found 1 documents" in list_result[0]["text"]
    assert "More results available" in list_result[1]["text"]

    # Test format_list_response with S3 identifier
    s3_list_response = {
        "documentDetails": [
            {"identifier": {"s3": {"uri": "s3://bucket/key"}}, "status": "INDEXED", "updatedAt": "2023-05-09T10:00:00Z"}
        ]
    }
    s3_result = formatter.format_list_response(s3_list_response)
    assert "s3://bucket/key" in s3_result[0]["text"]

    # Test format_retrieve_response with score filtering
    retrieve_response = {
        "retrievalResults": [
            {
                "score": 0.9,
                "content": {"text": '{"title": "High Score", "content": "High score content"}'},
                "location": {"customDocumentLocation": {"id": "doc1"}},
            },
            {
                "score": 0.3,  # Below threshold
                "content": {"text": '{"title": "Low Score", "content": "Low score content"}'},
                "location": {"customDocumentLocation": {"id": "doc2"}},
            },
        ]
    }

    retrieve_result = formatter.format_retrieve_response(retrieve_response, 0.5)
    assert "Retrieved 1 results" in retrieve_result[0]["text"]
    assert "High Score" in retrieve_result[0]["text"]
    assert "Low Score" not in retrieve_result[0]["text"]

    # Test format_retrieve_response with non-JSON content
    non_json_retrieve = {
        "retrievalResults": [
            {
                "score": 0.9,
                "content": {"text": "Plain text content, not JSON"},
                "location": {"customDocumentLocation": {"id": "doc1"}},
            }
        ]
    }

    non_json_result = formatter.format_retrieve_response(non_json_retrieve, 0.5)
    assert "Retrieved 1 results" in non_json_result[0]["text"]
    assert "Plain text content" in non_json_result[0]["text"]


@patch("boto3.Session")
def test_memory_service_client_lazy_loading(mock_session):
    """Test MemoryServiceClient lazy loading of clients."""
    # Create a client
    client = MemoryServiceClient(region="us-east-1")

    # Set up mock session and clients
    mock_agent_client = MagicMock()
    mock_runtime_client = MagicMock()

    # Configure the mock session to return our mock clients
    mock_session_instance = MagicMock()
    mock_session.return_value = mock_session_instance
    mock_session_instance.client.side_effect = lambda service, region_name, config=None: {
        "bedrock-agent": mock_agent_client,
        "bedrock-agent-runtime": mock_runtime_client,
    }[service]

    # Replace the real session with our mock
    client.session = mock_session_instance

    # Access the agent_client property (should trigger lazy loading)
    retrieved_agent_client = client.agent_client
    assert retrieved_agent_client == mock_agent_client
    call_args = mock_session_instance.client.call_args
    assert call_args[0] == ("bedrock-agent",)
    assert call_args[1]["region_name"] == "us-east-1"
    assert "config" in call_args[1]
    config = call_args[1]["config"]
    assert isinstance(config, BotocoreConfig)
    assert config.user_agent_extra == "strands-agents-memory"

    # Access again (should use cached client)
    mock_session_instance.client.reset_mock()
    retrieved_agent_client_again = client.agent_client
    assert retrieved_agent_client_again == mock_agent_client
    mock_session_instance.client.assert_not_called()

    # Access runtime_client property (should trigger lazy loading)
    retrieved_runtime_client = client.runtime_client
    assert retrieved_runtime_client == mock_runtime_client
    call_args = mock_session_instance.client.call_args
    assert call_args[0] == ("bedrock-agent-runtime",)
    assert call_args[1]["region_name"] == "us-east-1"
    assert "config" in call_args[1]
    config = call_args[1]["config"]
    assert isinstance(config, BotocoreConfig)
    assert config.user_agent_extra == "strands-agents-memory"
