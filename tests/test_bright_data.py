"""
Tests for the Bright Data tool using the tool decorator interface.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent

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


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
@patch("strands_tools.bright_data.BrightDataClient")
def test_scrape_as_markdown(mock_bright_data_client_class, mock_bright_data_client):
    """Test scrape_as_markdown functionality."""
    mock_bright_data_client_class.return_value = mock_bright_data_client

    markdown_content = "# Example Website\n\nThis is example content."
    mock_bright_data_client.scrape_as_markdown.return_value = markdown_content

    result = bright_data.bright_data(action="scrape_as_markdown", url="https://example.com", zone="unblocker")

    assert result == markdown_content
    mock_bright_data_client.scrape_as_markdown.assert_called_once_with("https://example.com", "unblocker")


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
@patch("strands_tools.bright_data.BrightDataClient")
def test_get_screenshot(mock_bright_data_client_class, mock_bright_data_client):
    """Test get_screenshot functionality."""
    mock_bright_data_client_class.return_value = mock_bright_data_client

    mock_bright_data_client.get_screenshot.return_value = "/tmp/screenshot.png"

    result = bright_data.bright_data(
        action="get_screenshot", url="https://example.com", output_path="/tmp/screenshot.png", zone="test_zone"
    )

    assert "Screenshot saved to /tmp/screenshot.png" in result
    mock_bright_data_client.get_screenshot.assert_called_once_with(
        "https://example.com", "/tmp/screenshot.png", "test_zone"
    )


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
@patch("strands_tools.bright_data.BrightDataClient")
def test_search_engine(mock_bright_data_client_class, mock_bright_data_client):
    """Test search_engine functionality."""
    mock_bright_data_client_class.return_value = mock_bright_data_client

    search_results = "# Search Results\n\n1. Result 1\n2. Result 2"
    mock_bright_data_client.search_engine.return_value = search_results

    result = bright_data.bright_data(
        action="search_engine",
        query="test query",
        engine="google",
        language="en",
        country_code="us",
        search_type="images",
        start=0,
        num_results=10,
        location="New York",
        device="mobile",
        return_json=False,
        zone="test_zone",
    )

    assert result == search_results
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
def test_web_data_feed(mock_bright_data_client_class, mock_bright_data_client):
    """Test web_data_feed functionality."""
    mock_bright_data_client_class.return_value = mock_bright_data_client

    amazon_data = {
        "title": "Test Product",
        "price": "29.99",
        "rating": 4.5,
        "reviews_count": 1024,
    }
    mock_bright_data_client.web_data_feed.return_value = amazon_data

    result = bright_data.bright_data(
        action="web_data_feed",
        source_type="amazon_product",
        url="https://www.amazon.com/product-url",
        num_of_reviews=5,
        timeout=300,
        polling_interval=2,
    )

    assert json.loads(result) == amazon_data
    mock_bright_data_client.web_data_feed.assert_called_once_with(
        source_type="amazon_product",
        url="https://www.amazon.com/product-url",
        num_of_reviews=5,
        timeout=300,
        polling_interval=2,
    )


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
def test_missing_required_parameters():
    """Test missing required parameters for different actions."""

    # Test missing URL for scrape_as_markdown
    with pytest.raises(ValueError) as exc_info:
        bright_data.bright_data(action="scrape_as_markdown")
    assert "url is required for scrape_as_markdown action" in str(exc_info.value)

    # Test missing URL for get_screenshot
    with pytest.raises(ValueError) as exc_info:
        bright_data.bright_data(action="get_screenshot")
    assert "url is required for get_screenshot action" in str(exc_info.value)

    # Test missing output_path for get_screenshot
    with pytest.raises(ValueError) as exc_info:
        bright_data.bright_data(action="get_screenshot", url="https://example.com")
    assert "output_path is required for get_screenshot action" in str(exc_info.value)

    # Test missing query for search_engine
    with pytest.raises(ValueError) as exc_info:
        bright_data.bright_data(action="search_engine")
    assert "query is required for search_engine action" in str(exc_info.value)

    # Test missing source_type for web_data_feed
    with pytest.raises(ValueError) as exc_info:
        bright_data.bright_data(action="web_data_feed", url="https://example.com")
    assert "source_type is required for web_data_feed action" in str(exc_info.value)

    # Test missing URL for web_data_feed
    with pytest.raises(ValueError) as exc_info:
        bright_data.bright_data(action="web_data_feed", source_type="amazon_product")
    assert "url is required for web_data_feed action" in str(exc_info.value)


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
def test_invalid_action():
    """Test invalid action."""
    with pytest.raises(ValueError) as exc_info:
        bright_data.bright_data(action="invalid")
    assert "Invalid action: invalid" in str(exc_info.value)


@patch.dict(os.environ, {})
def test_missing_api_key():
    """Test missing Bright Data API key."""
    with pytest.raises(ValueError) as exc_info:
        bright_data.bright_data(action="scrape_as_markdown", url="https://example.com")
    assert "BRIGHTDATA_API_KEY environment variable is required" in str(exc_info.value)


def test_missing_action():
    """Test missing action parameter."""
    with patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"}):
        with pytest.raises(ValueError) as exc_info:
            bright_data.bright_data(action="")
        assert "action parameter is required" in str(exc_info.value)


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


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
def test_search_engine_with_defaults():
    """Test search_engine with default parameters."""
    with patch("strands_tools.bright_data.BrightDataClient") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.search_engine.return_value = "search results"

        result = bright_data.bright_data(action="search_engine", query="test query")

        assert result == "search results"
        # Verify default values are used
        call_kwargs = mock_client.search_engine.call_args[1]
        assert call_kwargs["engine"] == "google"
        assert call_kwargs["num_results"] == 10
        assert call_kwargs["return_json"] is False


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
def test_zone_environment_variable():
    """Test that BRIGHTDATA_ZONE environment variable is used when zone is not provided."""
    with patch.dict(os.environ, {"BRIGHTDATA_ZONE": "custom_zone"}):
        with patch("strands_tools.bright_data.BrightDataClient") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.scrape_as_markdown.return_value = "content"

            bright_data.bright_data(action="scrape_as_markdown", url="https://example.com")

            # Verify the client was created with the custom zone
            mock_client_class.assert_called_with(verbose=True, zone="custom_zone")


@patch.dict(os.environ, {"BRIGHTDATA_API_KEY": "test_api_key"})
def test_zone_default_fallback():
    """Test that default zone is used when no zone is provided and no environment variable is set."""
    with patch("strands_tools.bright_data.BrightDataClient") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.scrape_as_markdown.return_value = "content"

        bright_data.bright_data(action="scrape_as_markdown", url="https://example.com")

        # Verify the client was created with the default zone
        mock_client_class.assert_called_with(verbose=True, zone="web_unlocker1")
