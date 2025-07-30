"""
Tests for the Tavily tools.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests
from strands_tools import tavily


@pytest.fixture
def mock_requests_response():
    """Create a mock requests response."""
    mock_response = MagicMock()
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


@patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
@patch("strands_tools.tavily.requests.post")
def test_tavily_search_success(mock_post, mock_requests_response):
    """Test successful tavily_search."""
    mock_post.return_value = mock_requests_response

    result = tavily.tavily_search(query="test query", max_results=5)

    assert result["status"] == "success"
    assert "content" in result
    response_data = eval(result["content"][0]["text"])
    assert response_data["query"] == "test query"
    assert len(response_data["results"]) == 1
    assert response_data["results"][0]["title"] == "Test Result"

    # Verify API call
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://api.tavily.com/search"
    assert call_args[1]["json"]["query"] == "test query"
    assert call_args[1]["json"]["max_results"] == 5


@patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
@patch("strands_tools.tavily.requests.post")
def test_tavily_search_with_all_parameters(mock_post, mock_requests_response):
    """Test tavily_search with all parameters."""
    mock_post.return_value = mock_requests_response

    result = tavily.tavily_search(
        query="test query",
        search_depth="advanced",
        topic="news",
        max_results=10,
        auto_parameters=True,
        chunks_per_source=2,
        time_range="week",
        days=7,
        include_answer=True,
        include_raw_content="markdown",
        include_images=True,
        include_image_descriptions=True,
        include_favicon=True,
        include_domains=["example.com"],
        exclude_domains=["spam.com"],
        country="united states",
    )

    assert result["status"] == "success"

    # Verify all parameters were passed
    call_args = mock_post.call_args
    payload = call_args[1]["json"]
    assert payload["search_depth"] == "advanced"
    assert payload["topic"] == "news"
    assert payload["max_results"] == 10
    assert payload["auto_parameters"] is True
    assert payload["chunks_per_source"] == 2
    assert payload["time_range"] == "week"
    assert payload["days"] == 7
    assert payload["include_answer"] is True
    assert payload["include_raw_content"] == "markdown"
    assert payload["include_images"] is True
    assert payload["include_image_descriptions"] is True
    assert payload["include_favicon"] is True
    assert payload["include_domains"] == ["example.com"]
    assert payload["exclude_domains"] == ["spam.com"]
    assert payload["country"] == "united states"


def test_tavily_search_missing_api_key():
    """Test tavily_search with missing API key."""
    with patch.dict(os.environ, {}, clear=True):
        result = tavily.tavily_search(query="test query")

        assert result["status"] == "error"
        assert "TAVILY_API_KEY environment variable is required" in result["content"][0]["text"]


def test_tavily_search_empty_query():
    """Test tavily_search with empty query."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = tavily.tavily_search(query="")

        assert result["status"] == "error"
        assert "Query parameter is required and cannot be empty" in result["content"][0]["text"]


def test_tavily_search_invalid_max_results():
    """Test tavily_search with invalid max_results."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = tavily.tavily_search(query="test", max_results=25)

        assert result["status"] == "error"
        assert "max_results must be between 0 and 20" in result["content"][0]["text"]


def test_tavily_search_invalid_chunks_per_source():
    """Test tavily_search with invalid chunks_per_source."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = tavily.tavily_search(query="test", chunks_per_source=5)

        assert result["status"] == "error"
        assert "chunks_per_source must be between 1 and 3" in result["content"][0]["text"]


@patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
@patch("strands_tools.tavily.requests.post")
def test_tavily_search_connection_error(mock_post):
    """Test tavily_search with connection error."""
    mock_post.side_effect = requests.exceptions.ConnectionError()

    result = tavily.tavily_search(query="test query")

    assert result["status"] == "error"
    assert "Connection error" in result["content"][0]["text"]


@patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
@patch("strands_tools.tavily.requests.post")
def test_tavily_search_timeout(mock_post):
    """Test tavily_search with timeout."""
    mock_post.side_effect = requests.exceptions.Timeout()

    result = tavily.tavily_search(query="test query")

    assert result["status"] == "error"
    assert "Request timeout" in result["content"][0]["text"]


@patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
@patch("strands_tools.tavily.requests.post")
def test_tavily_search_json_parse_error(mock_post):
    """Test tavily_search with JSON parse error."""
    mock_response = MagicMock()
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_post.return_value = mock_response

    result = tavily.tavily_search(query="test query")

    assert result["status"] == "error"
    assert "Failed to parse API response" in result["content"][0]["text"]


# Tests for tavily_extract


@patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
@patch("requests.post")
def test_tavily_extract_success(mock_post, mock_extract_response):
    """Test successful tavily_extract."""
    mock_response = MagicMock()
    mock_response.json.return_value = mock_extract_response
    mock_post.return_value = mock_response

    result = tavily.tavily_extract(urls=["https://www.tavily.com"])

    assert result["status"] == "success"
    assert "content" in result
    response_data = eval(result["content"][0]["text"])
    assert response_data["results"][0]["url"] == "https://www.tavily.com"

    # Verify API call
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]["json"]["urls"] == ["https://www.tavily.com"]


@patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
@patch("strands_tools.tavily.requests.post")
def test_tavily_extract_multiple_urls(mock_post, mock_extract_response):
    """Test tavily_extract with multiple URLs."""
    mock_response = MagicMock()
    mock_response.json.return_value = mock_extract_response
    mock_post.return_value = mock_response

    urls = ["https://www.tavily.com", "https://test.com"]
    result = tavily.tavily_extract(
        urls=urls, extract_depth="advanced", format="markdown", include_images=True, include_favicon=True
    )

    assert result["status"] == "success"

    # Verify parameters
    call_args = mock_post.call_args
    payload = call_args[1]["json"]
    assert payload["urls"] == urls
    assert payload["extract_depth"] == "advanced"
    assert payload["format"] == "markdown"
    assert payload["include_images"] is True
    assert payload["include_favicon"] is True


def test_tavily_extract_no_urls():
    """Test tavily_extract with no URLs."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = tavily.tavily_extract(urls=[])

        assert result["status"] == "error"
        assert "At least one URL must be provided" in result["content"][0]["text"]


# Tests for tavily_crawl


@patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
@patch("strands_tools.tavily.requests.post")
def test_tavily_crawl_success(mock_post, mock_crawl_response):
    """Test successful tavily_crawl."""
    mock_response = MagicMock()
    mock_response.json.return_value = mock_crawl_response
    mock_post.return_value = mock_response

    result = tavily.tavily_crawl(url="https://www.tavily.com")

    assert result["status"] == "success"
    assert "content" in result
    response_data = eval(result["content"][0]["text"])
    assert response_data["base_url"] == "https://www.tavily.com"
    assert len(response_data["results"]) == 2

    # Verify API call
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://api.tavily.com/crawl"
    assert call_args[1]["json"]["url"] == "https://www.tavily.com"


@patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
@patch("strands_tools.tavily.requests.post")
def test_tavily_crawl_with_all_parameters(mock_post, mock_crawl_response):
    """Test tavily_crawl with all parameters."""
    mock_response = MagicMock()
    mock_response.json.return_value = mock_crawl_response
    mock_post.return_value = mock_response

    result = tavily.tavily_crawl(
        url="https://www.tavily.com",
        max_depth=3,
        max_breadth=10,
        limit=50,
        instructions="Focus on documentation pages",
        select_paths=["/docs/*"],
        select_domains=["www.tavily.com"],
        exclude_paths=["/admin/*"],
        exclude_domains=["ads.www.tavily.com"],
        allow_external=False,
        include_images=True,
        categories=["Documentation", "Blog"],
        extract_depth="advanced",
        format="markdown",
        include_favicon=True,
    )

    assert result["status"] == "success"

    # Verify all parameters were passed
    call_args = mock_post.call_args
    payload = call_args[1]["json"]
    assert payload["max_depth"] == 3
    assert payload["max_breadth"] == 10
    assert payload["limit"] == 50
    assert payload["instructions"] == "Focus on documentation pages"
    assert payload["select_paths"] == ["/docs/*"]
    assert payload["select_domains"] == ["www.tavily.com"]
    assert payload["exclude_paths"] == ["/admin/*"]
    assert payload["exclude_domains"] == ["ads.www.tavily.com"]
    assert payload["allow_external"] is False
    assert payload["include_images"] is True
    assert payload["categories"] == ["Documentation", "Blog"]
    assert payload["extract_depth"] == "advanced"
    assert payload["format"] == "markdown"
    assert payload["include_favicon"] is True


def test_tavily_crawl_empty_url():
    """Test tavily_crawl with empty URL."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = tavily.tavily_crawl(url="")

        assert result["status"] == "error"
        assert "URL parameter is required and cannot be empty" in result["content"][0]["text"]


def test_tavily_crawl_invalid_depth():
    """Test tavily_crawl with invalid max_depth."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = tavily.tavily_crawl(url="https://www.tavily.com", max_depth=0)

        assert result["status"] == "error"
        assert "max_depth must be at least 1" in result["content"][0]["text"]


def test_tavily_crawl_invalid_breadth():
    """Test tavily_crawl with invalid max_breadth."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = tavily.tavily_crawl(url="https://www.tavily.com", max_breadth=0)

        assert result["status"] == "error"
        assert "max_breadth must be at least 1" in result["content"][0]["text"]


def test_tavily_crawl_invalid_limit():
    """Test tavily_crawl with invalid limit."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"}):
        result = tavily.tavily_crawl(url="https://www.tavily.com", limit=0)

        assert result["status"] == "error"
        assert "limit must be at least 1" in result["content"][0]["text"]


# Tests for tavily_map


@patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
@patch("strands_tools.tavily.requests.post")
def test_tavily_map_success(mock_post, mock_map_response):
    """Test successful tavily_map."""
    mock_response = MagicMock()
    mock_response.json.return_value = mock_map_response
    mock_post.return_value = mock_response

    result = tavily.tavily_map(url="https://www.tavily.com")

    assert result["status"] == "success"
    assert "content" in result
    response_data = eval(result["content"][0]["text"])
    assert response_data["base_url"] == "https://www.tavily.com"
    assert len(response_data["results"]) == 4

    # Verify API call
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://api.tavily.com/map"
    assert call_args[1]["json"]["url"] == "https://www.tavily.com"


@patch.dict(os.environ, {"TAVILY_API_KEY": "test-api-key"})
@patch("strands_tools.tavily.requests.post")
def test_tavily_map_with_parameters(mock_post, mock_map_response):
    """Test tavily_map with parameters."""
    mock_response = MagicMock()
    mock_response.json.return_value = mock_map_response
    mock_post.return_value = mock_response

    result = tavily.tavily_map(
        url="https://www.tavily.com",
        max_depth=2,
        max_breadth=5,
        limit=25,
        instructions="Map documentation structure",
        select_paths=["/docs/*"],
        select_domains=["www.tavily.com"],
        exclude_paths=["/private/*"],
        exclude_domains=["cdn.www.tavily.com"],
        allow_external=True,
        categories=["Documentation", "About"],
    )

    assert result["status"] == "success"

    # Verify parameters
    call_args = mock_post.call_args
    payload = call_args[1]["json"]
    assert payload["max_depth"] == 2
    assert payload["max_breadth"] == 5
    assert payload["limit"] == 25
    assert payload["instructions"] == "Map documentation structure"
    assert payload["select_paths"] == ["/docs/*"]
    assert payload["select_domains"] == ["www.tavily.com"]
    assert payload["exclude_paths"] == ["/private/*"]
    assert payload["exclude_domains"] == ["cdn.www.tavily.com"]
    assert payload["allow_external"] is True
    assert payload["categories"] == ["Documentation", "About"]


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
