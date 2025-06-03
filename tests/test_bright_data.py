"""
Tests for the Bright Data tool using the Agent interface.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent
from strands.types.tools import ToolUse
from strands_tools import bright_data
from strands_tools.bright_data import BrightDataClient


@pytest.fixture
def agent():
    """Create an agent with the bright_data tool loaded."""
    return Agent(tools=[bright_data])


@pytest.fixture
def mock_bright_data_client():
    """Create a mock Bright Data client."""
    client = MagicMock(spec=BrightDataClient)
    return client


@pytest.fixture
def mock_tool():
    """Create a mock tool use object that properly mocks the tool interface."""
    mock = MagicMock(spec=ToolUse)
    mock.get = MagicMock()
    mock.get.return_value = {}
    mock.get.side_effect = lambda key, default=None: {"toolUseId": "test-id", "input": {}}.get(key, default)
    return mock


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        content = result["content"][0]
        if isinstance(content, dict) and "text" in content:
            return content["text"]
    return str(result)


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
@patch("strands_tools.bright_data.BrightDataClient")
def test_scrape_as_markdown(mock_bright_data_client_class, mock_bright_data_client, mock_tool):
    """Test scrape_as_markdown functionality."""
    mock_bright_data_client_class.return_value = mock_bright_data_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "scrape_as_markdown",
            "url": "https://example.com",
            "zone": "unblocker",
        },
    }.get(key, default)

    markdown_content = "# Example Website\n\nThis is example content."

    mock_bright_data_client.scrape_as_markdown.return_value = markdown_content

    result = bright_data.bright_data(tool=mock_tool)

    assert result["status"] == "success"
    assert result["content"][0]["text"] == markdown_content

    mock_bright_data_client.scrape_as_markdown.assert_called_once_with("https://example.com", "unblocker")


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
@patch("strands_tools.bright_data.BrightDataClient")
def test_get_screenshot(mock_bright_data_client_class, mock_bright_data_client, mock_tool):
    """Test get_screenshot functionality."""
    mock_bright_data_client_class.return_value = mock_bright_data_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "get_screenshot",
            "url": "https://example.com",
            "output_path": "/tmp/screenshot.png",
            "zone": "test_zone",
        },
    }.get(key, default)

    mock_bright_data_client.get_screenshot.return_value = "/tmp/screenshot.png"

    result = bright_data.bright_data(tool=mock_tool)

    assert result["status"] == "success"
    assert "Screenshot saved to /tmp/screenshot.png" in result["content"][0]["text"]

    mock_bright_data_client.get_screenshot.assert_called_once_with(
        "https://example.com", "/tmp/screenshot.png", "test_zone"
    )


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
@patch("strands_tools.bright_data.BrightDataClient")
def test_search_engine(mock_bright_data_client_class, mock_bright_data_client, mock_tool):
    """Test search_engine functionality."""
    mock_bright_data_client_class.return_value = mock_bright_data_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "search_engine",
            "query": "test query",
            "engine": "google",
            "language": "en",
            "country_code": "us",
            "search_type": "images",
            "start": 0,
            "num_results": 10,
            "location": "New York",
            "device": "mobile",
            "return_json": False,
            "zone": "test_zone",
        },
    }.get(key, default)

    search_results = "# Search Results\n\n1. Result 1\n2. Result 2"

    mock_bright_data_client.search_engine.return_value = search_results

    result = bright_data.bright_data(tool=mock_tool)

    assert result["status"] == "success"
    assert result["content"][0]["text"] == search_results

    mock_bright_data_client.search_engine.assert_called_once()
    call_kwargs = mock_bright_data_client.search_engine.call_args[1]
    assert call_kwargs["query"] == "test query"
    assert call_kwargs["engine"] == "google"
    assert call_kwargs["language"] == "en"
    assert call_kwargs["country_code"] == "us"
    assert call_kwargs["search_type"] == "images"
    assert call_kwargs["start"] == 0
    assert call_kwargs["num_results"] == 10
    assert call_kwargs["location"] == "New York"
    assert call_kwargs["device"] == "mobile"
    assert call_kwargs["return_json"] is False
    assert call_kwargs["zone"] == "test_zone"


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
@patch("strands_tools.bright_data.BrightDataClient")
def test_web_data_feed(mock_bright_data_client_class, mock_bright_data_client, mock_tool):
    """Test web_data_feed functionality."""
    mock_bright_data_client_class.return_value = mock_bright_data_client

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {
            "action": "web_data_feed",
            "source_type": "amazon_product",
            "url": "https://www.amazon.com/product-url",
            "num_of_reviews": 5,
            "timeout": 300,
            "polling_interval": 2,
        },
    }.get(key, default)

    amazon_data = {
        "title": "Test Product",
        "price": "29.99",
        "rating": 4.5,
        "reviews_count": 1024,
    }

    mock_bright_data_client.web_data_feed.return_value = amazon_data

    result = bright_data.bright_data(tool=mock_tool)

    assert result["status"] == "success"
    assert json.loads(result["content"][0]["text"]) == amazon_data

    mock_bright_data_client.web_data_feed.assert_called_once_with(
        source_type="amazon_product",
        url="https://www.amazon.com/product-url",
        num_of_reviews=5,
        timeout=300,
        polling_interval=2,
    )


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
@patch("strands_tools.bright_data.BrightDataClient")
def test_missing_required_parameters(mock_bright_data_client_class, mock_tool):
    """Test missing required parameters for different actions."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "scrape_as_markdown"},
    }.get(key, default)

    result = bright_data.bright_data(tool=mock_tool)
    assert result["status"] == "error"
    assert "url is required for scrape_as_markdown action" in result["content"][0]["text"]

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "get_screenshot"},
    }.get(key, default)

    result = bright_data.bright_data(tool=mock_tool)
    assert result["status"] == "error"
    assert "url is required for get_screenshot action" in result["content"][0]["text"]

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "get_screenshot", "url": "https://example.com"},
    }.get(key, default)

    result = bright_data.bright_data(tool=mock_tool)
    assert result["status"] == "error"
    assert "output_path is required for get_screenshot action" in result["content"][0]["text"]

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "search_engine"},
    }.get(key, default)

    result = bright_data.bright_data(tool=mock_tool)
    assert result["status"] == "error"
    assert "query is required for search_engine action" in result["content"][0]["text"]

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "web_data_feed", "url": "https://example.com"},
    }.get(key, default)

    result = bright_data.bright_data(tool=mock_tool)
    assert result["status"] == "error"
    assert "source_type is required for web_data_feed action" in result["content"][0]["text"]

    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "web_data_feed", "source_type": "amazon_product"},
    }.get(key, default)

    result = bright_data.bright_data(tool=mock_tool)
    assert result["status"] == "error"
    assert "url is required for web_data_feed action" in result["content"][0]["text"]


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
@patch("strands_tools.bright_data.BrightDataClient")
def test_invalid_action(mock_bright_data_client_class, mock_tool):
    """Test invalid action."""
    mock_tool.get.side_effect = lambda key, default=None: {"toolUseId": "test-id", "input": {"action": "invalid"}}.get(
        key, default
    )

    result = bright_data.bright_data(tool=mock_tool)

    assert result["status"] == "error"
    assert "Invalid action: invalid" in result["content"][0]["text"]


@patch.dict(os.environ, {})
def test_missing_api_key(mock_tool):
    """Test missing Bright Data API key."""
    mock_tool.get.side_effect = lambda key, default=None: {
        "toolUseId": "test-id",
        "input": {"action": "scrape_as_markdown", "url": "https://example.com", "zone": "unblocker"},
    }.get(key, default)

    result = bright_data.bright_data(tool=mock_tool)

    assert result["status"] == "error"
    assert "BRIGHTDATA_API_KEY environment variable is required" in result["content"][0]["text"]


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
def test_bright_data_client_methods():
    """Test BrightDataClient class methods directly."""
    client = BrightDataClient(api_key="test_api_key", zone="test_zone", verbose=True)

    assert client.api_key == "test_api_key"
    assert client.zone == "test_zone"
    assert client.verbose is True

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Test content"
        mock_post.return_value = mock_response

        payload = {"url": "https://example.com", "zone": "test_zone"}
        result = client.make_request(payload)
        assert result == "Test content"
        mock_post.assert_called_with(
            "https://api.brightdata.com/request",
            headers={"Content-Type": "application/json", "Authorization": "Bearer test_api_key"},
            data=json.dumps(payload),
        )

    with patch.object(client, "make_request", return_value="# Markdown Content") as mock_make_request:
        result = client.scrape_as_markdown("https://example.com")
        assert result == "# Markdown Content"
        mock_make_request.assert_called_with(
            {"url": "https://example.com", "zone": "test_zone", "format": "raw", "data_format": "markdown"}
        )

    encoded = client.encode_query("test query")
    assert encoded == "test%20query"


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
def test_bright_data_client_failed_request():
    """Test BrightDataClient handling of failed requests."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        mock_post.return_value = mock_response

        client = BrightDataClient()

        payload = {"url": "https://example.com", "zone": "unlocker"}

        with pytest.raises(Exception) as excinfo:
            client.make_request(payload)

        assert "Failed to scrape: 403 - Forbidden" in str(excinfo.value)


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
def test_web_data_feed_timeout():
    """Test web_data_feed timeout handling."""
    with patch("requests.post") as mock_post, patch("requests.get") as mock_get, patch("time.sleep"):
        trigger_response = MagicMock()
        trigger_response.status_code = 200
        trigger_response.json.return_value = {"snapshot_id": "test_snapshot_id"}
        mock_post.return_value = trigger_response

        snapshot_response = MagicMock()
        snapshot_response.status_code = 200
        snapshot_response.json.return_value = {"status": "running"}
        mock_get.return_value = snapshot_response

        client = BrightDataClient(verbose=True)

        with pytest.raises(TimeoutError) as excinfo:
            client.web_data_feed(
                source_type="amazon_product", url="https://www.amazon.com/product-url", timeout=5, polling_interval=1
            )

        assert "Timeout after 5 seconds waiting for amazon_product data" in str(excinfo.value)
