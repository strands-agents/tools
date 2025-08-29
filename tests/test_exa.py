"""
Tests for the Exa tools.
"""

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

from strands_tools import exa


@pytest.fixture
def mock_aiohttp_response():
    """Create a mock aiohttp response for search."""
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "requestId": "b5947044c4b78efa9552a7c89b306d95",
        "resolvedSearchType": "neural",
        "searchType": "auto",
        "results": [
            {
                "title": "A Comprehensive Overview of Large Language Models",
                "url": "https://arxiv.org/pdf/2307.06435.pdf",
                "publishedDate": "2023-11-16T01:36:32.547Z",
                "author": "Humza Naveed, University of Engineering and Technology (UET), Lahore, Pakistan",
                "id": "https://arxiv.org/abs/2307.06435",
                "image": "https://arxiv.org/pdf/2307.06435.pdf/page_1.png",
                "favicon": "https://arxiv.org/favicon.ico",
                "text": "Abstract Large Language Models (LLMs) have recently demonstrated remarkable capabilities...",
                "summary": "This overview paper on Large Language Models (LLMs) highlights key developments...",
            }
        ],
        "context": "Formatted context string...",
        "costDollars": {
            "total": 0.005,
            "breakDown": [
                {
                    "search": 0.005,
                    "contents": 0,
                    "breakdown": {
                        "keywordSearch": 0,
                        "neuralSearch": 0.005,
                        "contentText": 0,
                        "contentHighlight": 0,
                        "contentSummary": 0,
                    },
                }
            ],
        },
    }
    return mock_response


@pytest.fixture
def mock_contents_response():
    """Create a mock contents response."""
    return {
        "requestId": "e492118ccdedcba5088bfc4357a8a125",
        "results": [
            {
                "title": "A Comprehensive Overview of Large Language Models",
                "url": "https://arxiv.org/pdf/2307.06435.pdf",
                "publishedDate": "2023-11-16T01:36:32.547Z",
                "author": "Humza Naveed, University of Engineering and Technology (UET), Lahore, Pakistan",
                "id": "https://arxiv.org/abs/2307.06435",
                "text": "Abstract Large Language Models (LLMs) have recently demonstrated remarkable capabilities...",
                "summary": "This overview paper on Large Language Models (LLMs) highlights key developments...",
            }
        ],
        "context": "Formatted context string...",
        "statuses": [{"id": "https://arxiv.org/pdf/2307.06435.pdf", "status": "success", "error": None}],
        "costDollars": {
            "total": 0.001,
            "breakDown": [
                {
                    "search": 0,
                    "contents": 0.001,
                    "breakdown": {"contentText": 0.001, "contentHighlight": 0, "contentSummary": 0},
                }
            ],
        },
    }


# Tests for exa_search


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_exa_search_success(mock_aiohttp_response):
    """Test successful exa_search."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    # This test is skipped for now as it requires complex async context manager mocking
    pass


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_exa_search_with_all_parameters(mock_aiohttp_response):
    """Test exa_search with all parameters."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


def test_exa_search_missing_api_key():
    """Test exa_search with missing API key."""
    with patch.dict(os.environ, {}, clear=True):
        result = asyncio.run(exa.exa_search(query="test query"))

        assert result["status"] == "error"
        assert "EXA_API_KEY environment variable is required" in result["content"][0]["text"]


def test_exa_search_empty_query():
    """Test exa_search with empty query."""
    with patch.dict(os.environ, {"EXA_API_KEY": "test-api-key"}):
        result = asyncio.run(exa.exa_search(query=""))

        assert result["status"] == "error"
        assert "Query parameter is required and cannot be empty" in result["content"][0]["text"]


def test_exa_search_invalid_num_results():
    """Test exa_search with invalid num_results."""
    with patch.dict(os.environ, {"EXA_API_KEY": "test-api-key"}):
        result = asyncio.run(exa.exa_search(query="test", num_results=150))

        assert result["status"] == "error"
        assert "num_results must be between 1 and 100" in result["content"][0]["text"]


def test_exa_search_zero_num_results():
    """Test exa_search with zero num_results."""
    with patch.dict(os.environ, {"EXA_API_KEY": "test-api-key"}):
        result = asyncio.run(exa.exa_search(query="test", num_results=0))

        assert result["status"] == "error"
        assert "num_results must be between 1 and 100" in result["content"][0]["text"]


def test_exa_search_invalid_start_date():
    """Test exa_search with invalid start_date format."""
    with patch.dict(os.environ, {"EXA_API_KEY": "test-api-key"}):
        result = asyncio.run(exa.exa_search(query="test", start_published_date="invalid-date"))

        assert result["status"] == "error"
        assert "Invalid date format for start_published_date" in result["content"][0]["text"]


def test_exa_search_invalid_end_date():
    """Test exa_search with invalid end_date format."""
    with patch.dict(os.environ, {"EXA_API_KEY": "test-api-key"}):
        result = asyncio.run(exa.exa_search(query="test", end_published_date="2024-13-01"))

        assert result["status"] == "error"
        assert "Invalid date format for end_published_date" in result["content"][0]["text"]


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_exa_search_connection_error():
    """Test exa_search with connection error."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_exa_search_timeout():
    """Test exa_search with timeout."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_exa_search_json_parse_error():
    """Test exa_search with JSON parse error."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


# Tests for exa_get_contents


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_exa_get_contents_success(mock_contents_response):
    """Test successful exa_get_contents."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_exa_get_contents_with_all_parameters(mock_contents_response):
    """Test exa_get_contents with all parameters."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


def test_exa_get_contents_no_urls():
    """Test exa_get_contents with no URLs."""
    with patch.dict(os.environ, {"EXA_API_KEY": "test-api-key"}):
        result = asyncio.run(exa.exa_get_contents(urls=[]))

        assert result["status"] == "error"
        assert "At least one URL must be provided" in result["content"][0]["text"]


def test_exa_get_contents_missing_api_key():
    """Test exa_get_contents with missing API key."""
    with patch.dict(os.environ, {}, clear=True):
        result = asyncio.run(exa.exa_get_contents(urls=["https://example.com"]))

        assert result["status"] == "error"
        assert "EXA_API_KEY environment variable is required" in result["content"][0]["text"]


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_exa_get_contents_connection_error():
    """Test exa_get_contents with connection error."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_exa_get_contents_timeout():
    """Test exa_get_contents with timeout."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


# Tests for utility functions


def test_get_api_key_success():
    """Test _get_api_key with valid API key."""
    with patch.dict(os.environ, {"EXA_API_KEY": "test-api-key"}):
        api_key = exa._get_api_key()
        assert api_key == "test-api-key"


def test_get_api_key_missing():
    """Test _get_api_key with missing API key."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            exa._get_api_key()
        assert "EXA_API_KEY environment variable is required" in str(exc_info.value)


def test_format_search_response():
    """Test format_search_response function."""
    data = {
        "requestId": "test-request-123",
        "searchType": "auto",
        "resolvedSearchType": "neural",
        "results": [
            {
                "title": "Test Result",
                "url": "https://example.com",
                "author": "Test Author",
                "publishedDate": "2023-11-16T01:36:32.547Z",
                "text": "This is test content for the search result...",
                "summary": "Test summary",
            }
        ],
        "context": "Formatted context string",
        "costDollars": {"total": 0.005},
    }

    panel = exa.format_search_response(data)
    assert panel.title == "[bold blue]Exa Search Results"
    assert "test-request-123" in panel.renderable
    assert "Test Result" in panel.renderable


def test_format_contents_response():
    """Test format_contents_response function."""
    data = {
        "requestId": "test-request-456",
        "results": [
            {
                "title": "Test Content",
                "url": "https://example.com",
                "text": "This is test content...",
                "summary": "Test summary",
            }
        ],
        "statuses": [{"id": "https://example.com", "status": "success", "error": None}],
        "context": "Formatted context string",
        "costDollars": {"total": 0.001},
    }

    panel = exa.format_contents_response(data)
    assert panel.title == "[bold blue]Exa Contents Results"
    assert "test-request-456" in panel.renderable
    assert "Successfully retrieved: 1 URLs" in panel.renderable


def test_format_contents_response_with_errors():
    """Test format_contents_response with failed results."""
    data = {
        "requestId": "test-request-789",
        "results": [],
        "statuses": [
            {"id": "https://example.com", "status": "error", "error": {"tag": "CRAWL_NOT_FOUND", "httpStatusCode": 404}}
        ],
        "costDollars": {"total": 0.001},
    }

    panel = exa.format_contents_response(data)
    assert panel.title == "[bold blue]Exa Contents Results"
    assert "Failed retrievals: 1 URLs" in panel.renderable
    assert "CRAWL_NOT_FOUND" in panel.renderable
