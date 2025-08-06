"""
Simple tests for Tavily tools (both sync and async versions).

These tests verify that the tools return the expected data structure and handle basic cases.
Run with: python test_tavily_tools.py
Or with pytest: pytest test_tavily_tools.py
"""

import asyncio
import os
import pytest
from unittest.mock import patch, MagicMock

from strands_tools.tavily import (
    tavily_search, tavily_extract, tavily_crawl, tavily_map,
    tavily_search_async, tavily_extract_async, tavily_crawl_async, tavily_map_async
)


def test_api_key_required():
    """Test that tools fail gracefully when API key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove TAVILY_API_KEY if it exists
        if 'TAVILY_API_KEY' in os.environ:
            del os.environ['TAVILY_API_KEY']
        
        result = tavily_search("test query")
        
        assert result["status"] == "error"
        assert "TAVILY_API_KEY environment variable is required" in result["content"][0]["text"]


def test_tavily_search_validation():
    """Test tavily_search parameter validation."""
    # Test empty query
    result = tavily_search("")
    assert result["status"] == "error"
    assert "Query parameter is required" in result["content"][0]["text"]
    
    # Test invalid max_results
    result = tavily_search("test", max_results=25)
    assert result["status"] == "error"
    assert "max_results must be between 0 and 20" in result["content"][0]["text"]
    
    # Test invalid chunks_per_source
    result = tavily_search("test", chunks_per_source=5)
    assert result["status"] == "error"
    assert "chunks_per_source must be between 1 and 3" in result["content"][0]["text"]


def test_tavily_extract_validation():
    """Test tavily_extract parameter validation."""
    # Test empty URLs
    result = tavily_extract([])
    assert result["status"] == "error"
    assert "At least one URL must be provided" in result["content"][0]["text"]
    
    result = tavily_extract("")
    assert result["status"] == "error"
    assert "At least one URL must be provided" in result["content"][0]["text"]


def test_tavily_crawl_validation():
    """Test tavily_crawl parameter validation."""
    # Test empty URL
    result = tavily_crawl("")
    assert result["status"] == "error"
    assert "URL parameter is required" in result["content"][0]["text"]
    
    # Test invalid max_depth
    result = tavily_crawl("https://example.com", max_depth=0)
    assert result["status"] == "error"
    assert "max_depth must be at least 1" in result["content"][0]["text"]
    
    # Test invalid max_breadth
    result = tavily_crawl("https://example.com", max_breadth=0)
    assert result["status"] == "error"
    assert "max_breadth must be at least 1" in result["content"][0]["text"]
    
    # Test invalid limit
    result = tavily_crawl("https://example.com", limit=0)
    assert result["status"] == "error"
    assert "limit must be at least 1" in result["content"][0]["text"]


def test_tavily_map_validation():
    """Test tavily_map parameter validation."""
    # Test empty URL
    result = tavily_map("")
    assert result["status"] == "error"
    assert "URL parameter is required" in result["content"][0]["text"]
    
    # Test invalid max_depth
    result = tavily_map("https://example.com", max_depth=0)
    assert result["status"] == "error"
    assert "max_depth must be at least 1" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_tavily_search_async_validation():
    """Test tavily_search_async parameter validation."""
    # Test empty query
    result = await tavily_search_async("")
    assert result["status"] == "error"
    assert "Query parameter is required" in result["content"][0]["text"]
    
    # Test invalid max_results
    result = await tavily_search_async("test", max_results=25)
    assert result["status"] == "error"
    assert "max_results must be between 0 and 20" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_tavily_extract_async_validation():
    """Test tavily_extract_async parameter validation."""
    # Test empty URLs
    result = await tavily_extract_async([])
    assert result["status"] == "error"
    assert "At least one URL must be provided" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_tavily_crawl_async_validation():
    """Test tavily_crawl_async parameter validation."""
    # Test empty URL
    result = await tavily_crawl_async("")
    assert result["status"] == "error"
    assert "URL parameter is required" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_tavily_map_async_validation():
    """Test tavily_map_async parameter validation."""
    # Test empty URL
    result = await tavily_map_async("")
    assert result["status"] == "error"
    assert "URL parameter is required" in result["content"][0]["text"]


# Mock tests for actual API calls (when API key is available)

@patch('strands_tools.tavily.requests.post')
@patch('strands_tools.tavily._get_api_key')
def test_tavily_search_success(mock_get_api_key, mock_post):
    """Test successful tavily_search call."""
    mock_get_api_key.return_value = "test_api_key"
    
    # Mock successful API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "query": "test query",
        "results": [
            {
                "title": "Test Result",
                "url": "https://example.com",
                "content": "Test content",
                "score": 0.95
            }
        ],
        "response_time": 1.23
    }
    mock_post.return_value = mock_response
    
    result = tavily_search("test query")
    
    assert result["status"] == "success"
    assert "content" in result
    assert len(result["content"]) == 1
    
    # Verify API was called with correct parameters
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]["json"]["query"] == "test query"
    assert "Authorization" in call_args[1]["headers"]


@patch('strands_tools.tavily.requests.post')
@patch('strands_tools.tavily._get_api_key')
def test_tavily_extract_success(mock_get_api_key, mock_post):
    """Test successful tavily_extract call."""
    mock_get_api_key.return_value = "test_api_key"
    
    # Mock successful API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {
                "url": "https://example.com",
                "raw_content": "Extracted content here..."
            }
        ],
        "failed_results": [],
        "response_time": 1.5
    }
    mock_post.return_value = mock_response
    
    result = tavily_extract("https://example.com")
    
    assert result["status"] == "success"
    assert "content" in result
    
    # Verify API was called
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]["json"]["urls"] == "https://example.com"


@patch('strands_tools.tavily.requests.post')
@patch('strands_tools.tavily._get_api_key')
def test_tavily_crawl_success(mock_get_api_key, mock_post):
    """Test successful tavily_crawl call."""
    mock_get_api_key.return_value = "test_api_key"
    
    # Mock successful API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "base_url": "example.com",
        "results": [
            {
                "url": "https://example.com/page1",
                "raw_content": "Page 1 content"
            },
            {
                "url": "https://example.com/page2", 
                "raw_content": "Page 2 content"
            }
        ],
        "response_time": 2.1
    }
    mock_post.return_value = mock_response
    
    result = tavily_crawl("https://example.com")
    
    assert result["status"] == "success"
    assert "content" in result
    
    # Verify API was called
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]["json"]["url"] == "https://example.com"


@patch('strands_tools.tavily.requests.post')
@patch('strands_tools.tavily._get_api_key')
def test_tavily_map_success(mock_get_api_key, mock_post):
    """Test successful tavily_map call."""
    mock_get_api_key.return_value = "test_api_key"
    
    # Mock successful API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "base_url": "example.com",
        "results": [
            "https://example.com/",
            "https://example.com/about",
            "https://example.com/contact"
        ],
        "response_time": 1.8
    }
    mock_post.return_value = mock_response
    
    result = tavily_map("https://example.com")
    
    assert result["status"] == "success"
    assert "content" in result
    
    # Verify API was called
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[1]["json"]["url"] == "https://example.com"


# Async mock tests

@patch('strands_tools.tavily.aiohttp.ClientSession.post')
@patch('strands_tools.tavily._get_api_key')
@pytest.mark.asyncio
async def test_tavily_search_async_success(mock_get_api_key, mock_post):
    """Test successful tavily_search_async call."""
    mock_get_api_key.return_value = "test_api_key"
    
    # Mock successful async API response
    mock_response = MagicMock()
    mock_response.json = AsyncMock(return_value={
        "query": "test query",
        "results": [
            {
                "title": "Test Result",
                "url": "https://example.com",
                "content": "Test content",
                "score": 0.95
            }
        ],
        "response_time": 1.23
    })
    
    # Mock the context manager
    mock_post.return_value.__aenter__.return_value = mock_response
    
    result = await tavily_search_async("test query")
    
    assert result["status"] == "success"
    assert "content" in result


@patch('strands_tools.tavily.aiohttp.ClientSession.post')
@patch('strands_tools.tavily._get_api_key')
@pytest.mark.asyncio
async def test_tavily_extract_async_success(mock_get_api_key, mock_post):
    """Test successful tavily_extract_async call."""
    mock_get_api_key.return_value = "test_api_key"
    
    # Mock successful async API response
    mock_response = MagicMock()
    mock_response.json = AsyncMock(return_value={
        "results": [
            {
                "url": "https://example.com",
                "raw_content": "Extracted content here..."
            }
        ],
        "failed_results": [],
        "response_time": 1.5
    })
    
    # Mock the context manager
    mock_post.return_value.__aenter__.return_value = mock_response
    
    result = await tavily_extract_async("https://example.com")
    
    assert result["status"] == "success"
    assert "content" in result


class AsyncMock(MagicMock):
    """Helper class for async mocking."""
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


def test_response_structure():
    """Test that all tools return consistent response structure."""
    
    # Test error responses have consistent structure
    result = tavily_search("")
    assert "status" in result
    assert "content" in result
    assert isinstance(result["content"], list)
    assert "text" in result["content"][0]
    
    result = tavily_extract([])
    assert "status" in result
    assert "content" in result
    assert isinstance(result["content"], list)
    
    result = tavily_crawl("")
    assert "status" in result
    assert "content" in result
    assert isinstance(result["content"], list)
    
    result = tavily_map("")
    assert "status" in result
    assert "content" in result
    assert isinstance(result["content"], list)


@pytest.mark.asyncio
async def test_async_response_structure():
    """Test that all async tools return consistent response structure."""
    
    # Test error responses have consistent structure
    result = await tavily_search_async("")
    assert "status" in result
    assert "content" in result
    assert isinstance(result["content"], list)
    assert "text" in result["content"][0]
    
    result = await tavily_extract_async([])
    assert "status" in result
    assert "content" in result
    assert isinstance(result["content"], list)
    
    result = await tavily_crawl_async("")
    assert "status" in result
    assert "content" in result
    assert isinstance(result["content"], list)
    
    result = await tavily_map_async("")
    assert "status" in result
    assert "content" in result
    assert isinstance(result["content"], list)


def run_simple_tests():
    """Run basic tests that don't require pytest."""
    print("Running simple Tavily tools tests...")
    
    # Test parameter validation
    print("✓ Testing parameter validation...")
    test_tavily_search_validation()
    test_tavily_extract_validation()
    test_tavily_crawl_validation()
    test_tavily_map_validation()
    
    print("✓ Testing response structure...")
    test_response_structure()
    
    print("✓ Testing API key requirement...")
    test_api_key_required()
    
    print("All basic tests passed!")


async def run_async_tests():
    """Run async tests that don't require pytest."""
    print("Running async Tavily tools tests...")
    
    print("✓ Testing async parameter validation...")
    await test_tavily_search_async_validation()
    await test_tavily_extract_async_validation()
    await test_tavily_crawl_async_validation()
    await test_tavily_map_async_validation()
    
    print("✓ Testing async response structure...")
    await test_async_response_structure()
    
    print("All async tests passed!")


if __name__ == "__main__":
    # Run simple tests
    run_simple_tests()
    
    # Run async tests
    asyncio.run(run_async_tests())
    
    print("\n🎉 All tests completed successfully!")
    print("\nTo run with pytest (for more detailed testing):")
    print("pip install pytest pytest-asyncio")
    print("pytest test_tavily_tools.py -v") 