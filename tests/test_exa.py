"""
Tests for the Exa tools.
"""

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

from strands_tools import exa


class AsyncContextManager:
    """Helper to wrap a value as an async context manager for mocking aiohttp."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_aiohttp_response():
    """Create a mock aiohttp response for search."""
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "requestId": "b5947044c4b78efa9552a7c89b306d95",
        "resolvedSearchType": "auto",
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
                "highlights": [
                    "Large Language Models (LLMs) have recently demonstrated remarkable capabilities",
                    "This survey provides a comprehensive overview of recent advances in LLMs",
                ],
                "highlightScores": [0.95, 0.88],
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
                        "search": 0.005,
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
                "highlights": [
                    "Large Language Models (LLMs) have recently demonstrated remarkable capabilities",
                    "This survey provides a comprehensive overview of recent advances in LLMs",
                ],
                "highlightScores": [0.95, 0.88],
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
        "resolvedSearchType": "auto",
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


def test_format_search_response_with_highlights():
    """Test format_search_response renders highlights."""
    data = {
        "requestId": "test-hl-search",
        "searchType": "auto",
        "resolvedSearchType": "auto",
        "results": [
            {
                "title": "Highlights Test",
                "url": "https://example.com",
                "author": "Author",
                "publishedDate": "2024-01-01",
                "highlights": [
                    "First key excerpt from the page",
                    "Second key excerpt from the page",
                ],
            }
        ],
        "costDollars": {"total": 0.005},
    }

    panel = exa.format_search_response(data)
    rendered = panel.renderable
    assert "Highlights:" in rendered
    assert "First key excerpt from the page" in rendered
    assert "Second key excerpt from the page" in rendered


def test_format_contents_response_with_highlights():
    """Test format_contents_response renders highlights."""
    data = {
        "requestId": "test-hl-contents",
        "results": [
            {
                "title": "Highlights Test",
                "url": "https://example.com",
                "highlights": [
                    "Key excerpt from contents",
                ],
            }
        ],
        "statuses": [{"id": "https://example.com", "status": "success", "error": None}],
        "costDollars": {"total": 0.001},
    }

    panel = exa.format_contents_response(data)
    rendered = panel.renderable
    assert "Highlights:" in rendered
    assert "Key excerpt from contents" in rendered


def _make_mock_session(captured, response_data):
    """Build a mock aiohttp session that captures the payload and headers from post() calls."""
    mock_resp = AsyncMock()
    mock_resp.json.return_value = response_data

    mock_post_ctx = AsyncMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=False)

    def fake_post(url, json=None, headers=None):
        captured["url"] = url
        captured["payload"] = json
        captured["headers"] = headers
        return mock_post_ctx

    mock_session = AsyncMock()
    mock_session.post = fake_post

    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_session_ctx


SEARCH_RESPONSE = {
    "requestId": "test",
    "searchType": "auto",
    "resolvedSearchType": "auto",
    "results": [],
    "costDollars": {"total": 0},
}

CONTENTS_RESPONSE = {
    "requestId": "test",
    "results": [],
    "statuses": [],
    "costDollars": {"total": 0},
}


def test_exa_search_payload_wires_highlights_and_max_age_hours():
    """Test that exa_search correctly wires highlights and maxAgeHours into the contents dict."""
    captured = {}
    mock_session_ctx = _make_mock_session(captured, SEARCH_RESPONSE)

    with (
        patch.dict(os.environ, {"EXA_API_KEY": "test-key"}),
        patch("aiohttp.ClientSession", return_value=mock_session_ctx),
    ):
        asyncio.run(exa.exa_search(query="test", highlights=True, max_age_hours=24))

    contents = captured["payload"]["contents"]
    assert contents["highlights"] is True
    assert contents["maxAgeHours"] == 24


def test_exa_search_payload_wires_highlights_dict():
    """Test that exa_search passes a highlights dict through to the contents dict."""
    captured = {}
    mock_session_ctx = _make_mock_session(captured, SEARCH_RESPONSE)

    with (
        patch.dict(os.environ, {"EXA_API_KEY": "test-key"}),
        patch("aiohttp.ClientSession", return_value=mock_session_ctx),
    ):
        asyncio.run(exa.exa_search(query="test", highlights={"maxCharacters": 4000}))

    contents = captured["payload"]["contents"]
    assert contents["highlights"] == {"maxCharacters": 4000}


def test_exa_get_contents_payload_wires_highlights_and_max_age_hours():
    """Test that exa_get_contents correctly wires highlights and maxAgeHours into the flat payload."""
    captured = {}
    mock_session_ctx = _make_mock_session(captured, CONTENTS_RESPONSE)

    with (
        patch.dict(os.environ, {"EXA_API_KEY": "test-key"}),
        patch("aiohttp.ClientSession", return_value=mock_session_ctx),
    ):
        asyncio.run(exa.exa_get_contents(urls=["https://example.com"], highlights=True, max_age_hours=48))

    payload = captured["payload"]
    assert payload["highlights"] is True
    assert payload["maxAgeHours"] == 48
    assert "contents" not in payload


def test_exa_search_payload_includes_integration_header():
    """Test that exa_search sends the x-exa-integration header."""
    captured = {}
    mock_session_ctx = _make_mock_session(captured, SEARCH_RESPONSE)

    with (
        patch.dict(os.environ, {"EXA_API_KEY": "test-key"}),
        patch("aiohttp.ClientSession", return_value=mock_session_ctx),
    ):
        asyncio.run(exa.exa_search(query="test"))

    assert captured["headers"]["x-exa-integration"] == "aws-strands-agent"


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
