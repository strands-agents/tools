"""Tests for the Apify tools."""

import json
from unittest.mock import MagicMock, patch

import pytest

from strands_tools import apify
from strands_tools.apify import (
    ApifyToolClient,
    apify_get_dataset_items,
    apify_run_actor,
    apify_run_actor_and_get_dataset,
    apify_scrape_url,
)

MOCK_ACTOR_RUN = {
    "id": "run-HG7ml5fB1hCp8YEBA",
    "actId": "janedoe~my-scraper",
    "userId": "user-abc123",
    "startedAt": "2026-03-15T14:30:00.000Z",
    "finishedAt": "2026-03-15T14:35:22.000Z",
    "status": "SUCCEEDED",
    "statusMessage": "Actor finished successfully",
    "defaultDatasetId": "dataset-WkC9gct8rq1uR5vDZ",
    "defaultKeyValueStoreId": "kvs-Xb3A8gct8rq1uR5vD",
    "buildNumber": "1.2.3",
}

MOCK_FAILED_RUN = {
    **MOCK_ACTOR_RUN,
    "status": "FAILED",
    "statusMessage": "Actor failed with an error",
}

MOCK_TIMED_OUT_RUN = {
    **MOCK_ACTOR_RUN,
    "status": "TIMED-OUT",
    "statusMessage": "Actor run timed out",
}

MOCK_DATASET_ITEMS = [
    {"url": "https://example.com/product/1", "title": "Widget A", "price": 19.99, "currency": "USD"},
    {"url": "https://example.com/product/2", "title": "Widget B", "price": 29.99, "currency": "USD"},
    {"url": "https://example.com/product/3", "title": "Widget C", "price": 39.99, "currency": "EUR"},
]

MOCK_SCRAPED_ITEM = {
    "url": "https://example.com",
    "markdown": "# Example Domain\n\nThis domain is for use in illustrative examples.",
    "text": "Example Domain. This domain is for use in illustrative examples.",
}


@pytest.fixture
def mock_apify_client():
    """Create a mock ApifyClient with pre-configured responses."""
    client = MagicMock()

    mock_actor = MagicMock()
    mock_actor.call.return_value = MOCK_ACTOR_RUN
    client.actor.return_value = mock_actor

    mock_dataset = MagicMock()
    mock_list_result = MagicMock()
    mock_list_result.items = MOCK_DATASET_ITEMS
    mock_dataset.list_items.return_value = mock_list_result
    client.dataset.return_value = mock_dataset

    return client


@pytest.fixture
def mock_apify_env(monkeypatch):
    """Set required Apify environment variables."""
    monkeypatch.setenv("APIFY_API_TOKEN", "test-token-12345")


# --- Module import ---


def test_apify_module_is_importable():
    """Verify that the apify tool module can be imported from strands_tools."""
    assert apify is not None
    assert apify.__name__ == "strands_tools.apify"


# --- ApifyToolClient ---


def test_client_missing_token(monkeypatch):
    """ApifyToolClient raises ValueError when APIFY_API_TOKEN is not set."""
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="APIFY_API_TOKEN"):
        ApifyToolClient()


def test_client_uses_env_token(mock_apify_env):
    """ApifyToolClient passes the env token to ApifyClient."""
    with patch("strands_tools.apify.ApifyClient") as MockClient:
        ApifyToolClient()
        MockClient.assert_called_once_with(
            "test-token-12345",
            headers={"x-apify-integration-platform": "strands-agents"},
        )


# --- apify_run_actor ---


def test_run_actor_success(mock_apify_env, mock_apify_client):
    """Successful Actor Run returns JSON with run metadata."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_actor(actor_id="janedoe/my-scraper", run_input={"url": "https://example.com"})

    data = json.loads(result)
    assert data["run_id"] == "run-HG7ml5fB1hCp8YEBA"
    assert data["status"] == "SUCCEEDED"
    assert data["dataset_id"] == "dataset-WkC9gct8rq1uR5vDZ"
    assert "started_at" in data
    assert "finished_at" in data
    mock_apify_client.actor.assert_called_once_with("janedoe/my-scraper")


def test_run_actor_with_memory(mock_apify_env, mock_apify_client):
    """Actor Run passes memory_mbytes when provided."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        apify_run_actor(actor_id="janedoe/my-scraper", memory_mbytes=512)

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs["memory_mbytes"] == 512


def test_run_actor_failure(mock_apify_env, mock_apify_client):
    """Actor Run raises RuntimeError when Actor fails."""
    mock_apify_client.actor.return_value.call.return_value = MOCK_FAILED_RUN

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        with pytest.raises(RuntimeError, match="FAILED"):
            apify_run_actor(actor_id="janedoe/my-scraper")


def test_run_actor_timeout(mock_apify_env, mock_apify_client):
    """Actor Run raises RuntimeError when Actor times out."""
    mock_apify_client.actor.return_value.call.return_value = MOCK_TIMED_OUT_RUN

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        with pytest.raises(RuntimeError, match="TIMED-OUT"):
            apify_run_actor(actor_id="janedoe/my-scraper")


def test_run_actor_api_exception(mock_apify_env, mock_apify_client):
    """Actor Run re-raises exceptions from the Apify client."""
    mock_apify_client.actor.return_value.call.side_effect = Exception("Connection failed")

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        with pytest.raises(Exception, match="Connection failed"):
            apify_run_actor(actor_id="janedoe/my-scraper")


# --- apify_get_dataset_items ---


def test_get_dataset_items_success(mock_apify_env, mock_apify_client):
    """Successful dataset retrieval returns JSON array of items."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_get_dataset_items(dataset_id="dataset-WkC9gct8rq1uR5vDZ")

    items = json.loads(result)
    assert len(items) == 3
    assert items[0]["title"] == "Widget A"
    assert items[2]["currency"] == "EUR"
    mock_apify_client.dataset.assert_called_once_with("dataset-WkC9gct8rq1uR5vDZ")


def test_get_dataset_items_with_pagination(mock_apify_env, mock_apify_client):
    """Dataset retrieval passes limit and offset."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        apify_get_dataset_items(dataset_id="dataset-xyz", limit=50, offset=10)

    mock_apify_client.dataset.return_value.list_items.assert_called_once_with(limit=50, offset=10)


def test_get_dataset_items_empty(mock_apify_env, mock_apify_client):
    """Empty dataset returns an empty JSON array."""
    mock_list_result = MagicMock()
    mock_list_result.items = []
    mock_apify_client.dataset.return_value.list_items.return_value = mock_list_result

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_get_dataset_items(dataset_id="dataset-empty")

    items = json.loads(result)
    assert items == []


# --- apify_run_actor_and_get_dataset ---


def test_run_actor_and_get_dataset_success(mock_apify_env, mock_apify_client):
    """Combined run + dataset fetch returns run metadata and items."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_actor_and_get_dataset(
            actor_id="janedoe/my-scraper",
            run_input={"url": "https://example.com"},
            dataset_items_limit=50,
        )

    data = json.loads(result)
    assert data["run_id"] == "run-HG7ml5fB1hCp8YEBA"
    assert data["status"] == "SUCCEEDED"
    assert data["dataset_id"] == "dataset-WkC9gct8rq1uR5vDZ"
    assert len(data["items"]) == 3
    assert data["items"][0]["title"] == "Widget A"


def test_run_actor_and_get_dataset_actor_failure(mock_apify_env, mock_apify_client):
    """Combined tool raises when the Actor fails."""
    mock_apify_client.actor.return_value.call.return_value = MOCK_FAILED_RUN

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        with pytest.raises(RuntimeError, match="FAILED"):
            apify_run_actor_and_get_dataset(actor_id="janedoe/my-scraper")


# --- apify_scrape_url ---


def test_scrape_url_success(mock_apify_env, mock_apify_client):
    """Scrape URL returns markdown content from the crawled page."""
    mock_list_result = MagicMock()
    mock_list_result.items = [MOCK_SCRAPED_ITEM]
    mock_apify_client.dataset.return_value.list_items.return_value = mock_list_result

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_scrape_url(url="https://example.com")

    assert "Example Domain" in result
    mock_apify_client.actor.assert_called_once_with("apify/website-content-crawler")


def test_scrape_url_no_content(mock_apify_env, mock_apify_client):
    """Scrape URL raises when no content is returned."""
    mock_list_result = MagicMock()
    mock_list_result.items = []
    mock_apify_client.dataset.return_value.list_items.return_value = mock_list_result

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        with pytest.raises(RuntimeError, match="No content returned"):
            apify_scrape_url(url="https://example.com")


def test_scrape_url_crawler_failure(mock_apify_env, mock_apify_client):
    """Scrape URL raises when the crawler Actor fails."""
    mock_apify_client.actor.return_value.call.return_value = MOCK_FAILED_RUN

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        with pytest.raises(RuntimeError, match="FAILED"):
            apify_scrape_url(url="https://example.com")


def test_scrape_url_falls_back_to_text(mock_apify_env, mock_apify_client):
    """Scrape URL falls back to text field when markdown is absent."""
    item_without_markdown = {"url": "https://example.com", "text": "Plain text content"}
    mock_list_result = MagicMock()
    mock_list_result.items = [item_without_markdown]
    mock_apify_client.dataset.return_value.list_items.return_value = mock_list_result

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_scrape_url(url="https://example.com")

    assert result == "Plain text content"


# --- Dependency guard ---


def test_missing_apify_client_run_actor(mock_apify_env):
    """apify_run_actor raises ImportError when apify-client is not installed."""
    with patch("strands_tools.apify.HAS_APIFY_CLIENT", False):
        with pytest.raises(ImportError, match="apify-client"):
            apify_run_actor(actor_id="test/actor")


def test_missing_apify_client_get_dataset(mock_apify_env):
    """apify_get_dataset_items raises ImportError when apify-client is not installed."""
    with patch("strands_tools.apify.HAS_APIFY_CLIENT", False):
        with pytest.raises(ImportError, match="apify-client"):
            apify_get_dataset_items(dataset_id="dataset-123")


def test_missing_apify_client_run_and_get(mock_apify_env):
    """apify_run_actor_and_get_dataset raises ImportError when apify-client is not installed."""
    with patch("strands_tools.apify.HAS_APIFY_CLIENT", False):
        with pytest.raises(ImportError, match="apify-client"):
            apify_run_actor_and_get_dataset(actor_id="test/actor")


def test_missing_apify_client_scrape_url(mock_apify_env):
    """apify_scrape_url raises ImportError when apify-client is not installed."""
    with patch("strands_tools.apify.HAS_APIFY_CLIENT", False):
        with pytest.raises(ImportError, match="apify-client"):
            apify_scrape_url(url="https://example.com")


# --- Missing token from tool entry points ---


def test_run_actor_missing_token(monkeypatch):
    """apify_run_actor raises ValueError when APIFY_API_TOKEN is missing."""
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="APIFY_API_TOKEN"):
        apify_run_actor(actor_id="test/actor")


def test_scrape_url_missing_token(monkeypatch):
    """apify_scrape_url raises ValueError when APIFY_API_TOKEN is missing."""
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="APIFY_API_TOKEN"):
        apify_scrape_url(url="https://example.com")
