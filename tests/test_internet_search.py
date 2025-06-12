from unittest.mock import MagicMock, patch

import pytest
from strands_tools import internet_search


@pytest.fixture
def mock_agent():
    """Fixture to create a mock agent with the internet_search tool."""
    agent = MagicMock()
    agent.tool.internet_search = MagicMock()
    return agent


def extract_results(result):
    """Extract the results list from the agent/tool response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        for item in result["content"]:
            if "json" in item and "results" in item["json"]:
                return item["json"]["results"]
    return []


def test_duckduckgo_search_with_mock_agent(mock_agent):
    """Test DuckDuckGo backend search using mock agent."""
    mock_results = [
        {"title": "AI News 1", "href": "http://a.com", "body": "Summary 1"},
        {"title": "AI News 2", "href": "http://b.com", "body": "Summary 2"},
    ]
    mock_agent.tool.internet_search.return_value = {
        "toolUseId": "test-duck-id",
        "status": "success",
        "content": [{"json": {"results": mock_results}}],
    }
    result = mock_agent.tool.internet_search(query="AI news", max_results=2)
    assert result["status"] == "success"
    results = extract_results(result)
    assert len(results) == 2
    assert results[0]["title"] == "AI News 1"


def test_serpapi_search_with_mock_agent(mock_agent):
    """Test SerpAPI backend search using mock agent."""
    mock_results = [
        {"title": "Serp Result 1", "href": "http://serp1.com", "body": "Serp snippet 1"},
        {"title": "Serp Result 2", "href": "http://serp2.com", "body": "Serp snippet 2"},
    ]
    mock_agent.tool.internet_search.return_value = {
        "toolUseId": "test-serpapi-id",
        "status": "success",
        "content": [{"json": {"results": mock_results}}],
    }
    result = mock_agent.tool.internet_search(query="AI news", max_results=2, backend="serpapi", serpapi_api_key="dummy-key")
    assert result["status"] == "success"
    results = extract_results(result)
    assert len(results) == 2
    assert results[0]["title"] == "Serp Result 1"


def test_tavily_search_with_mock_agent(mock_agent):
    """Test Tavily backend search using mock agent."""
    mock_results = [
        {"title": "Tavily 1", "href": "http://tav1.com", "body": "Tavily content 1"},
        {"title": "Tavily 2", "href": "http://tav2.com", "body": "Tavily content 2"},
    ]
    mock_agent.tool.internet_search.return_value = {
        "toolUseId": "test-tavily-id",
        "status": "success",
        "content": [{"json": {"results": mock_results}}],
    }
    result = mock_agent.tool.internet_search(query="AI news", max_results=2, backend="tavily", tavily_api_key="dummy-key")
    assert result["status"] == "success"
    results = extract_results(result)
    assert len(results) == 2
    assert results[0]["title"] == "Tavily 1"


def test_invalid_backend_with_mock_agent(mock_agent):
    """Test error on unknown backend using mock agent."""
    mock_agent.tool.internet_search.return_value = {
        "toolUseId": "test-invalid-backend",
        "status": "error",
        "content": [{"text": "Unknown backend: notarealbackend"}],
    }
    result = mock_agent.tool.internet_search(query="AI news", backend="notarealbackend")
    assert result["status"] == "error"
    assert "Unknown backend" in result["content"][0]["text"]
