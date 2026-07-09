"""
Tests for error handling in memory.py.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from strands_tools.memory import memory


@pytest.fixture
def mock_memory_service_client():
    """Create a mock memory service client."""
    client = MagicMock()

    # Set up default behavior
    client.get_data_source_id.return_value = "ds123"

    return client


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
def test_invalid_kb_id_format(mock_get_client):
    """Test error handling for invalid knowledge base ID format."""
    # Test with hyphenated KB ID (invalid format)
    result = memory(action="list", STRANDS_KNOWLEDGE_BASE_ID="invalid-kb-id")

    assert result["status"] == "error"
    assert "Invalid knowledge base ID format" in result["content"][0]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
def test_data_source_error(mock_get_client):
    """Test error handling when getting data source ID fails."""
    # Setup mock to raise exception
    mock_client = MagicMock()
    mock_client.get_data_source_id.side_effect = Exception("API Error")
    mock_get_client.return_value = mock_client

    result = memory(action="list")

    assert result["status"] == "error"
    assert "Failed to get data source ID" in result["content"][0]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
def test_store_empty_content(mock_get_client):
    """Test error handling for empty content in store operation."""
    result = memory(action="store", content="")

    assert result["status"] == "error"
    assert "Content cannot be empty" in result["content"][0]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
def test_delete_missing_id(mock_get_client):
    """Test error handling for missing document ID in delete operation."""
    result = memory(action="delete")

    assert result["status"] == "error"
    assert "Document ID cannot be empty" in result["content"][0]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
def test_get_missing_id(mock_get_client):
    """Test error handling for missing document ID in get operation."""
    result = memory(action="get")

    assert result["status"] == "error"
    assert "Document ID cannot be empty" in result["content"][0]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
def test_retrieve_missing_query(mock_get_client):
    """Test error handling for missing query in retrieve operation."""
    result = memory(action="retrieve")

    assert result["status"] == "error"
    assert "No query provided" in result["content"][0]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
def test_retrieve_invalid_min_score(mock_get_client):
    """Test error handling for invalid min_score in retrieve operation."""
    result = memory(
        action="retrieve",
        query="test query",
        min_score=1.5,  # Invalid: should be between 0.0 and 1.0
    )

    assert result["status"] == "error"
    assert "min_score must be between 0.0 and 1.0" in result["content"][0]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
def test_retrieve_invalid_max_results(mock_get_client):
    """Test error handling for invalid max_results in retrieve operation."""
    result = memory(
        action="retrieve",
        query="test query",
        max_results=1500,  # Invalid: should be between 1 and 1000
    )

    assert result["status"] == "error"
    assert "max_results must be between 1 and 1000" in result["content"][0]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
def test_list_invalid_max_results(mock_get_client):
    """Test error handling for invalid max_results in list operation."""
    result = memory(
        action="list",
        max_results=1500,  # Invalid: should be between 1 and 1000
    )

    assert result["status"] == "error"
    assert "max_results must be between 1 and 1000" in result["content"][0]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb", "BYPASS_TOOL_CONSENT": "true"})
@patch("strands_tools.memory.get_memory_service_client")
def test_store_api_error(mock_get_client):
    """Test error handling when store operation fails."""
    # Setup mock to raise exception during store
    mock_client = MagicMock()
    mock_client.get_data_source_id.return_value = "ds123"
    mock_client.store_document.side_effect = Exception("API Error during store")
    mock_get_client.return_value = mock_client

    result = memory(action="store", content="Test content", title="Test Title")

    assert result["status"] == "error"
    assert "Error during store operation" in result["content"][0]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb", "BYPASS_TOOL_CONSENT": "true"})
@patch("strands_tools.memory.get_memory_service_client")
def test_delete_api_error(mock_get_client):
    """Test error handling when delete operation fails."""
    # Setup mock to raise exception during delete
    mock_client = MagicMock()
    mock_client.get_data_source_id.return_value = "ds123"
    mock_client.delete_document.side_effect = Exception("API Error during delete")
    mock_get_client.return_value = mock_client

    result = memory(action="delete", document_id="doc123")

    assert result["status"] == "error"
    assert "Error during delete operation" in result["content"][0]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
def test_get_api_error(mock_get_client):
    """Test error handling when get operation fails."""
    # Setup mock to raise exception during get
    mock_client = MagicMock()
    mock_client.get_data_source_id.return_value = "ds123"
    mock_client.get_document.side_effect = Exception("API Error during get")
    mock_get_client.return_value = mock_client

    result = memory(action="get", document_id="doc123")

    assert result["status"] == "error"
    assert "Error retrieving document" in result["content"][0]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
def test_retrieve_api_error(mock_get_client):
    """Test error handling when retrieve operation fails."""
    # Setup mock to raise exception during retrieve
    mock_client = MagicMock()
    mock_client.get_data_source_id.return_value = "ds123"
    mock_client.retrieve.side_effect = Exception("API Error during retrieve")
    mock_get_client.return_value = mock_client

    result = memory(action="retrieve", query="test query")

    assert result["status"] == "error"
    assert "Error during retrieval" in result["content"][0]["text"]


@patch.dict(os.environ, {"STRANDS_KNOWLEDGE_BASE_ID": "test123kb"})
@patch("strands_tools.memory.get_memory_service_client")
def test_retrieve_validation_error(mock_get_client):
    """Test error handling when retrieve encounters a validation error."""
    # Setup mock to raise specific exception
    mock_client = MagicMock()
    mock_client.get_data_source_id.return_value = "ds123"

    # Create an exception with a specific error message pattern
    exception = Exception("ValidationException: Invalid knowledgeBaseId")
    mock_client.retrieve.side_effect = exception
    mock_get_client.return_value = mock_client

    result = memory(action="retrieve", query="test query")

    assert result["status"] == "error"
    assert "Invalid knowledge base ID" in result["content"][0]["text"]
