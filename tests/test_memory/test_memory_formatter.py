"""
Tests for the MemoryFormatter class in memory.py.
"""

import json

import pytest

from strands_tools.memory import MemoryFormatter


@pytest.fixture
def formatter():
    """Create a MemoryFormatter instance."""
    return MemoryFormatter()


def test_format_list_response_empty(formatter):
    """Test formatting an empty list response."""
    empty_response = {"documentDetails": []}
    result = formatter.format_list_response(empty_response)

    assert len(result) == 1
    assert "No documents found" in result[0]["text"]


def test_format_list_response_with_documents(formatter):
    """Test formatting a list response with documents."""
    response = {
        "documentDetails": [
            {"identifier": {"custom": {"id": "doc123"}}, "status": "INDEXED", "updatedAt": "2023-05-09T10:00:00Z"},
            {"identifier": {"custom": {"id": "doc456"}}, "status": "INGESTING", "updatedAt": "2023-05-09T11:00:00Z"},
        ]
    }

    result = formatter.format_list_response(response)

    assert len(result) == 1
    assert "Found 2 documents" in result[0]["text"]
    assert "doc123" in result[0]["text"]
    assert "doc456" in result[0]["text"]
    assert "INDEXED" in result[0]["text"]
    assert "INGESTING" in result[0]["text"]


def test_format_list_response_with_s3_identifier(formatter):
    """Test formatting a list response with S3 identifiers."""
    response = {
        "documentDetails": [
            {
                "identifier": {"s3": {"uri": "s3://bucket/file.txt"}},
                "status": "INDEXED",
                "updatedAt": "2023-05-09T10:00:00Z",
            }
        ]
    }

    result = formatter.format_list_response(response)

    assert len(result) == 1
    assert "Found 1 documents" in result[0]["text"]
    assert "s3://bucket/file.txt" in result[0]["text"]


def test_format_list_response_with_pagination(formatter):
    """Test formatting a list response with pagination token."""
    response = {
        "documentDetails": [
            {"identifier": {"custom": {"id": "doc123"}}, "status": "INDEXED", "updatedAt": "2023-05-09T10:00:00Z"}
        ],
        "nextToken": "token123",
    }

    result = formatter.format_list_response(response)

    assert len(result) == 3
    assert "Found 1 documents" in result[0]["text"]
    assert "More results available" in result[1]["text"]
    assert "token123" in result[2]["text"]


def test_format_get_response(formatter):
    """Test formatting a get document response."""
    document_id = "doc123"
    kb_id = "kb456"
    content_data = {"title": "Test Document", "content": "This is test content"}

    result = formatter.format_get_response(document_id, kb_id, content_data)

    assert len(result) == 5
    assert "Document retrieved successfully" in result[0]["text"]
    assert "Test Document" in result[1]["text"]
    assert document_id in result[2]["text"]
    assert kb_id in result[3]["text"]
    assert "This is test content" in result[4]["text"]


def test_format_store_response(formatter):
    """Test formatting a store document response."""
    doc_id = "doc123"
    kb_id = "kb456"
    title = "Test Document"

    result = formatter.format_store_response(doc_id, kb_id, title)

    assert len(result) == 4
    assert "Successfully stored content" in result[0]["text"]
    assert title in result[1]["text"]
    assert doc_id in result[2]["text"]
    assert kb_id in result[3]["text"]


def test_format_delete_response_success(formatter):
    """Test formatting a successful delete document response."""
    status = "DELETED"
    doc_id = "doc123"
    kb_id = "kb456"

    result = formatter.format_delete_response(status, doc_id, kb_id)

    assert len(result) == 3
    assert "Document deletion deleted" in result[0]["text"]
    assert doc_id in result[1]["text"]
    assert kb_id in result[2]["text"]


def test_format_delete_response_in_progress(formatter):
    """Test formatting an in-progress delete document response."""
    status = "DELETING"
    doc_id = "doc123"
    kb_id = "kb456"

    result = formatter.format_delete_response(status, doc_id, kb_id)

    assert len(result) == 3
    assert "Document deletion deleting" in result[0]["text"]


def test_format_delete_response_failure(formatter):
    """Test formatting a failed delete document response."""
    status = "FAILED"
    doc_id = "doc123"
    kb_id = "kb456"

    result = formatter.format_delete_response(status, doc_id, kb_id)

    assert len(result) == 3
    assert "Document deletion failed" in result[0]["text"]


def test_format_retrieve_response_empty(formatter):
    """Test formatting an empty retrieve response."""
    response = {"retrievalResults": []}

    result = formatter.format_retrieve_response(response)

    assert len(result) == 1
    assert "No results found" in result[0]["text"]


def test_format_retrieve_response_with_results(formatter):
    """Test formatting a retrieve response with results."""
    response = {
        "retrievalResults": [
            {
                "score": 0.95,
                "location": {"customDocumentLocation": {"id": "doc123"}},
                "content": {"text": "This is test content"},
            },
            {
                "score": 0.85,
                "location": {"customDocumentLocation": {"id": "doc456"}},
                "content": {"text": "This is more content"},
            },
        ]
    }

    result = formatter.format_retrieve_response(response)

    assert len(result) == 1
    assert "Retrieved 2 results" in result[0]["text"]
    assert "0.9500" in result[0]["text"]
    assert "0.8500" in result[0]["text"]
    assert "doc123" in result[0]["text"]
    assert "doc456" in result[0]["text"]


def test_format_retrieve_response_with_score_filter(formatter):
    """Test formatting a retrieve response with score filtering."""
    response = {
        "retrievalResults": [
            {
                "score": 0.95,
                "location": {"customDocumentLocation": {"id": "doc123"}},
                "content": {"text": "This is test content"},
            },
            {
                "score": 0.45,
                "location": {"customDocumentLocation": {"id": "doc456"}},
                "content": {"text": "This is more content"},
            },
        ]
    }

    # Filter out results below 0.5
    result = formatter.format_retrieve_response(response, 0.5)

    assert len(result) == 1
    assert "Retrieved 1 results" in result[0]["text"]
    assert "0.9500" in result[0]["text"]
    assert "doc123" in result[0]["text"]
    assert "doc456" not in result[0]["text"]


def test_format_retrieve_response_with_json_content(formatter):
    """Test formatting a retrieve response with JSON content."""
    json_content = json.dumps({"title": "Test Document", "content": "This is test content"})

    response = {
        "retrievalResults": [
            {"score": 0.95, "location": {"customDocumentLocation": {"id": "doc123"}}, "content": {"text": json_content}}
        ]
    }

    result = formatter.format_retrieve_response(response)

    assert len(result) == 1
    assert "Test Document" in result[0]["text"]


def test_format_retrieve_response_with_pagination(formatter):
    """Test formatting a retrieve response with pagination."""
    response = {
        "retrievalResults": [
            {
                "score": 0.95,
                "location": {"customDocumentLocation": {"id": "doc123"}},
                "content": {"text": "This is test content"},
            }
        ],
        "nextToken": "token123",
    }

    result = formatter.format_retrieve_response(response)

    assert len(result) > 1
    assert "More results available" in result[1]["text"]
    assert "token123" in result[2]["text"]
