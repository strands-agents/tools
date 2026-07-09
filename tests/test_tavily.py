"""
Tests for the Tavily tools.
"""

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

from strands_tools import tavily


@pytest.fixture
def mock_aiohttp_response():
    """Create a mock aiohttp response."""
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "query": "test query",
        "results": [
            {
                "title": "Test Result",
                "url": "https://www.tavily.com",
                "content": "Test content",
                "score": 0.9,
                "raw_content": "Raw test content",
                "favicon": "https://www.tavily.com/favicon.ico",
            }
        ],
        "answer": "Test answer",
        "images": ["https://www.tavily.com/image.jpg"],
    }
    return mock_response


@pytest.fixture
def mock_extract_response():
    """Create a mock extract response."""
    return {
        "results": [
            {
                "url": "https://www.tavily.com",
                "raw_content": "Extracted content from the page",
                "images": ["https://www.tavily.com/image.jpg"],
                "favicon": "https://www.tavily.com/favicon.ico",
            }
        ],
        "failed_results": [],
    }


@pytest.fixture
def mock_crawl_response():
    """Create a mock crawl response."""
    return {
        "base_url": "https://www.tavily.com",
        "response_time": 1.5,
        "results": [
            {
                "url": "https://www.tavily.com/page1",
                "raw_content": "Content from page 1",
                "favicon": "https://www.tavily.com/favicon.ico",
            },
            {
                "url": "https://www.tavily.com/page2",
                "raw_content": "Content from page 2",
                "favicon": "https://www.tavily.com/favicon.ico",
            },
        ],
    }


@pytest.fixture
def mock_map_response():
    """Create a mock map response."""
    return {
        "base_url": "https://www.tavily.com",
        "response_time": 1.2,
        "results": [
            "https://www.tavily.com/page1",
            "https://www.tavily.com/page2",
            "https://www.tavily.com/about",
            "https://www.tavily.com/contact",
        ],
    }


# Tests for tavily_search


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_tavily_search_success(mock_aiohttp_response):
    """Test successful tavily_search."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    # This test is skipped for now as it requires complex async context manager mocking
    pass


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_tavily_search_with_all_parameters(mock_aiohttp_response):
    """Test tavily_search with all parameters."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


def test_tavily_search_missing_api_key():
    """Test tavily_search with missing API key."""
    with patch.dict(os.environ, {}, clear=True):
        result = asyncio.run(tavily.tavily_search(query="test query"))

        assert result["status"] == "error"
        assert "TAVILY_API_KEY environment variable is required" in result["content"][0]["text"]


def test_tavily_search_empty_query():
    """Test tavily_search with empty query."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = asyncio.run(tavily.tavily_search(query=""))

        assert result["status"] == "error"
        assert "Query parameter is required and cannot be empty" in result["content"][0]["text"]


def test_tavily_search_invalid_max_results():
    """Test tavily_search with invalid max_results."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = asyncio.run(tavily.tavily_search(query="test", max_results=25))

        assert result["status"] == "error"
        assert "max_results must be between 0 and 20" in result["content"][0]["text"]


def test_tavily_search_invalid_chunks_per_source():
    """Test tavily_search with invalid chunks_per_source."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = asyncio.run(tavily.tavily_search(query="test", chunks_per_source=5))

        assert result["status"] == "error"
        assert "chunks_per_source must be between 1 and 3" in result["content"][0]["text"]


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_tavily_search_connection_error():
    """Test tavily_search with connection error."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_tavily_search_timeout():
    """Test tavily_search with timeout."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_tavily_search_json_parse_error():
    """Test tavily_search with JSON parse error."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


# Tests for tavily_extract


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_tavily_extract_success(mock_extract_response):
    """Test successful tavily_extract."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_tavily_extract_multiple_urls(mock_extract_response):
    """Test tavily_extract with multiple URLs."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


def test_tavily_extract_no_urls():
    """Test tavily_extract with no URLs."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = asyncio.run(tavily.tavily_extract(urls=[]))

        assert result["status"] == "error"
        assert "At least one URL must be provided" in result["content"][0]["text"]


# Tests for tavily_crawl


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_tavily_crawl_success(mock_crawl_response):
    """Test successful tavily_crawl."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_tavily_crawl_with_all_parameters(mock_crawl_response):
    """Test tavily_crawl with all parameters."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


def test_tavily_crawl_empty_url():
    """Test tavily_crawl with empty URL."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = asyncio.run(tavily.tavily_crawl(url=""))

        assert result["status"] == "error"
        assert "URL parameter is required and cannot be empty" in result["content"][0]["text"]


def test_tavily_crawl_invalid_depth():
    """Test tavily_crawl with invalid max_depth."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = asyncio.run(tavily.tavily_crawl(url="https://www.tavily.com", max_depth=0))

        assert result["status"] == "error"
        assert "max_depth must be at least 1" in result["content"][0]["text"]


def test_tavily_crawl_invalid_breadth():
    """Test tavily_crawl with invalid max_breadth."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = asyncio.run(tavily.tavily_crawl(url="https://www.tavily.com", max_breadth=0))

        assert result["status"] == "error"
        assert "max_breadth must be at least 1" in result["content"][0]["text"]


def test_tavily_crawl_invalid_limit():
    """Test tavily_crawl with invalid limit."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = asyncio.run(tavily.tavily_crawl(url="https://www.tavily.com", limit=0))

        assert result["status"] == "error"
        assert "limit must be at least 1" in result["content"][0]["text"]


# Tests for tavily_map


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_tavily_map_success(mock_map_response):
    """Test successful tavily_map."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


@pytest.mark.skip(reason="Complex async HTTP mocking needs aioresponses library")
def test_tavily_map_with_parameters(mock_map_response):
    """Test tavily_map with parameters."""
    # TODO: Implement with aioresponses library for proper async HTTP mocking
    pass


# Tests for utility functions


def test_get_api_key_success():
    """Test _get_api_key with valid API key."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        api_key = tavily._get_api_key()
        assert api_key == "test-api-key"


def test_get_api_key_missing():
    """Test _get_api_key with missing API key."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            tavily._get_api_key()
        assert "TAVILY_API_KEY environment variable is required" in str(exc_info.value)


def test_format_search_response():
    """Test format_search_response function."""
    data = {
        "query": "test query",
        "results": [
            {
                "title": "Test Result",
                "url": "https://www.tavily.com",
                "content": "Test content",
                "score": 0.9,
                "raw_content": "Raw test content",
                "favicon": "https://www.tavily.com/favicon.ico",
            }
        ],
        "answer": "Test answer",
        "images": ["https://www.tavily.com/image.jpg"],
    }

    panel = tavily.format_search_response(data)
    assert panel.title == "[bold cyan]Tavily Search Results"
    assert "test query" in panel.renderable


def test_format_extract_response():
    """Test format_extract_response function."""
    data = {
        "results": [
            {
                "url": "https://www.tavily.com",
                "raw_content": "Extracted content",
                "images": ["https://www.tavily.com/image.jpg"],
                "favicon": "https://www.tavily.com/favicon.ico",
            }
        ],
        "failed_results": [],
    }

    panel = tavily.format_extract_response(data)
    assert panel.title == "[bold cyan]Tavily Extract Results"
    assert "Successfully extracted: 1 URLs" in panel.renderable


def test_format_crawl_response():
    """Test format_crawl_response function."""
    data = {
        "base_url": "https://www.tavily.com",
        "response_time": 1.5,
        "results": [
            {
                "url": "https://www.tavily.com/page1",
                "raw_content": "Content from page 1",
                "favicon": "https://www.tavily.com/favicon.ico",
            }
        ],
    }

    panel = tavily.format_crawl_response(data)
    assert panel.title == "[bold cyan]]Tavily Crawl Results"
    assert "https://www.tavily.com" in panel.renderable


def test_format_map_response():
    """Test format_map_response function."""
    data = {
        "base_url": "https://www.tavily.com",
        "response_time": 1.2,
        "results": ["https://www.tavily.com/page1", "https://www.tavily.com/page2"],
    }

    panel = tavily.format_map_response(data)
    assert panel.title == "[bold cyan]Tavily Map Results"
    assert "URLs Discovered: 2" in panel.renderable
