"""Tests for the Apify tools."""

import json
from unittest.mock import MagicMock, patch

import pytest

from strands_tools import apify
from strands_tools.apify import (
    ApifyToolClient,
    apify_ecommerce_scraper,
    apify_get_dataset_items,
    apify_google_places_scraper,
    apify_google_search_scraper,
    apify_run_actor,
    apify_run_actor_and_get_dataset,
    apify_run_task,
    apify_run_task_and_get_dataset,
    apify_scrape_url,
    apify_website_content_crawler,
    apify_youtube_scraper,
)

MOCK_ACTOR_RUN = {
    "id": "run-HG7ml5fB1hCp8YEBA",
    "actId": "actor~my-scraper",
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


def _make_apify_api_error(status_code: int, message: str) -> Exception:
    """Create an ApifyApiError instance for testing without calling its real __init__."""
    from apify_client.errors import ApifyApiError

    error = ApifyApiError.__new__(ApifyApiError)
    Exception.__init__(error, message)
    error.status_code = status_code
    error.message = message
    return error


@pytest.fixture
def mock_apify_client():
    """Create a mock ApifyClient with pre-configured responses."""
    client = MagicMock()

    mock_actor = MagicMock()
    mock_actor.call.return_value = MOCK_ACTOR_RUN
    client.actor.return_value = mock_actor

    mock_task = MagicMock()
    mock_task.call.return_value = MOCK_ACTOR_RUN
    client.task.return_value = mock_task

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
    """Verify that the apify module can be imported from strands_tools."""
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
    """Successful Actor run returns structured result with run metadata."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_actor(actor_id="actor/my-scraper", run_input={"url": "https://example.com"})

    assert result["status"] == "success"
    data = json.loads(result["content"][0]["text"])
    assert data["run_id"] == "run-HG7ml5fB1hCp8YEBA"
    assert data["status"] == "SUCCEEDED"
    assert data["dataset_id"] == "dataset-WkC9gct8rq1uR5vDZ"
    assert "started_at" in data
    assert "finished_at" in data
    mock_apify_client.actor.assert_called_once_with("actor/my-scraper")


def test_run_actor_default_input(mock_apify_env, mock_apify_client):
    """Actor run defaults run_input to empty dict when not provided."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_actor(actor_id="actor/my-scraper")

    assert result["status"] == "success"
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs["run_input"] == {}


def test_run_actor_with_memory(mock_apify_env, mock_apify_client):
    """Actor run passes memory_mbytes when provided."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        apify_run_actor(actor_id="actor/my-scraper", memory_mbytes=512)

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs["memory_mbytes"] == 512


def test_run_actor_failure(mock_apify_env, mock_apify_client):
    """Actor run returns error dict when Actor fails."""
    mock_apify_client.actor.return_value.call.return_value = MOCK_FAILED_RUN

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_actor(actor_id="actor/my-scraper")

    assert result["status"] == "error"
    assert "FAILED" in result["content"][0]["text"]


def test_run_actor_timeout(mock_apify_env, mock_apify_client):
    """Actor run returns error dict when Actor times out."""
    mock_apify_client.actor.return_value.call.return_value = MOCK_TIMED_OUT_RUN

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_actor(actor_id="actor/my-scraper")

    assert result["status"] == "error"
    assert "TIMED-OUT" in result["content"][0]["text"]


def test_run_actor_api_exception(mock_apify_env, mock_apify_client):
    """Actor run returns error dict on API exceptions."""
    mock_apify_client.actor.return_value.call.side_effect = Exception("Connection failed")

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_actor(actor_id="actor/my-scraper")

    assert result["status"] == "error"
    assert "Connection failed" in result["content"][0]["text"]


def test_run_actor_apify_api_error_401(mock_apify_env, mock_apify_client):
    """Actor run returns friendly message for 401 authentication errors."""
    error = _make_apify_api_error(401, "Unauthorized")
    mock_apify_client.actor.return_value.call.side_effect = error

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_actor(actor_id="actor/my-scraper")

    assert result["status"] == "error"
    assert "Authentication failed" in result["content"][0]["text"]


def test_run_actor_apify_api_error_404(mock_apify_env, mock_apify_client):
    """Actor run returns friendly message for 404 not-found errors."""
    error = _make_apify_api_error(404, "Actor not found")
    mock_apify_client.actor.return_value.call.side_effect = error

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_actor(actor_id="actor/nonexistent")

    assert result["status"] == "error"
    assert "Resource not found" in result["content"][0]["text"]


# --- apify_get_dataset_items ---


def test_get_dataset_items_success(mock_apify_env, mock_apify_client):
    """Successful dataset retrieval returns structured result with items."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_get_dataset_items(dataset_id="dataset-WkC9gct8rq1uR5vDZ")

    assert result["status"] == "success"
    items = json.loads(result["content"][0]["text"])
    assert len(items) == 3
    assert items[0]["title"] == "Widget A"
    assert items[2]["currency"] == "EUR"
    mock_apify_client.dataset.assert_called_once_with("dataset-WkC9gct8rq1uR5vDZ")


def test_get_dataset_items_with_pagination(mock_apify_env, mock_apify_client):
    """dataset retrieval passes limit and offset."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        apify_get_dataset_items(dataset_id="dataset-xyz", limit=50, offset=10)

    mock_apify_client.dataset.return_value.list_items.assert_called_once_with(limit=50, offset=10)


def test_get_dataset_items_empty(mock_apify_env, mock_apify_client):
    """Empty dataset returns a structured result with empty JSON array."""
    mock_list_result = MagicMock()
    mock_list_result.items = []
    mock_apify_client.dataset.return_value.list_items.return_value = mock_list_result

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_get_dataset_items(dataset_id="dataset-empty")

    assert result["status"] == "success"
    items = json.loads(result["content"][0]["text"])
    assert items == []


# --- apify_run_actor_and_get_dataset ---


def test_run_actor_and_get_dataset_success(mock_apify_env, mock_apify_client):
    """Combined run + dataset fetch returns structured result with metadata and items."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_actor_and_get_dataset(
            actor_id="actor/my-scraper",
            run_input={"url": "https://example.com"},
            dataset_items_limit=50,
        )

    assert result["status"] == "success"
    data = json.loads(result["content"][0]["text"])
    assert data["run_id"] == "run-HG7ml5fB1hCp8YEBA"
    assert data["status"] == "SUCCEEDED"
    assert data["dataset_id"] == "dataset-WkC9gct8rq1uR5vDZ"
    assert len(data["items"]) == 3
    assert data["items"][0]["title"] == "Widget A"


def test_run_actor_and_get_dataset_actor_failure(mock_apify_env, mock_apify_client):
    """Combined tool returns error dict when the Actor fails."""
    mock_apify_client.actor.return_value.call.return_value = MOCK_FAILED_RUN

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_actor_and_get_dataset(actor_id="actor/my-scraper")

    assert result["status"] == "error"
    assert "FAILED" in result["content"][0]["text"]


# --- apify_run_task ---


def test_run_task_success(mock_apify_env, mock_apify_client):
    """Successful task run returns structured result with run metadata."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_task(task_id="user~my-task", task_input={"query": "test"})

    assert result["status"] == "success"
    data = json.loads(result["content"][0]["text"])
    assert data["run_id"] == "run-HG7ml5fB1hCp8YEBA"
    assert data["status"] == "SUCCEEDED"
    assert data["dataset_id"] == "dataset-WkC9gct8rq1uR5vDZ"
    mock_apify_client.task.assert_called_once_with("user~my-task")


def test_run_task_no_input(mock_apify_env, mock_apify_client):
    """task run omits task_input kwarg when not provided."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_task(task_id="user~my-task")

    assert result["status"] == "success"
    call_kwargs = mock_apify_client.task.return_value.call.call_args.kwargs
    assert "task_input" not in call_kwargs


def test_run_task_with_memory(mock_apify_env, mock_apify_client):
    """task run passes memory_mbytes when provided."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        apify_run_task(task_id="user~my-task", memory_mbytes=1024)

    call_kwargs = mock_apify_client.task.return_value.call.call_args.kwargs
    assert call_kwargs["memory_mbytes"] == 1024


def test_run_task_failure(mock_apify_env, mock_apify_client):
    """task run returns error dict when task fails."""
    mock_apify_client.task.return_value.call.return_value = MOCK_FAILED_RUN

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_task(task_id="user~my-task")

    assert result["status"] == "error"
    assert "FAILED" in result["content"][0]["text"]


def test_run_task_none_response(mock_apify_env, mock_apify_client):
    """task run returns error dict when TaskClient.call() returns None."""
    mock_apify_client.task.return_value.call.return_value = None

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_task(task_id="user~my-task")

    assert result["status"] == "error"
    assert "no run data" in result["content"][0]["text"]


def test_run_task_apify_api_error_401(mock_apify_env, mock_apify_client):
    """task run returns friendly message for 401 authentication errors."""
    error = _make_apify_api_error(401, "Unauthorized")
    mock_apify_client.task.return_value.call.side_effect = error

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_task(task_id="user~my-task")

    assert result["status"] == "error"
    assert "Authentication failed" in result["content"][0]["text"]


# --- apify_run_task_and_get_dataset ---


def test_run_task_and_get_dataset_success(mock_apify_env, mock_apify_client):
    """Combined task run + dataset fetch returns structured result with metadata and items."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_task_and_get_dataset(
            task_id="user~my-task",
            task_input={"query": "test"},
            dataset_items_limit=50,
        )

    assert result["status"] == "success"
    data = json.loads(result["content"][0]["text"])
    assert data["run_id"] == "run-HG7ml5fB1hCp8YEBA"
    assert len(data["items"]) == 3
    assert data["items"][0]["title"] == "Widget A"


def test_run_task_and_get_dataset_task_failure(mock_apify_env, mock_apify_client):
    """Combined task tool returns error dict when the task fails."""
    mock_apify_client.task.return_value.call.return_value = MOCK_FAILED_RUN

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_run_task_and_get_dataset(task_id="user~my-task")

    assert result["status"] == "error"
    assert "FAILED" in result["content"][0]["text"]


# --- apify_scrape_url ---


def test_scrape_url_success(mock_apify_env, mock_apify_client):
    """Scrape URL returns structured result with markdown content."""
    mock_list_result = MagicMock()
    mock_list_result.items = [MOCK_SCRAPED_ITEM]
    mock_apify_client.dataset.return_value.list_items.return_value = mock_list_result

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_scrape_url(url="https://example.com")

    assert result["status"] == "success"
    assert "Example Domain" in result["content"][0]["text"]
    mock_apify_client.actor.assert_called_once_with("apify/website-content-crawler")


def test_scrape_url_no_content(mock_apify_env, mock_apify_client):
    """Scrape URL returns error dict when no content is returned."""
    mock_list_result = MagicMock()
    mock_list_result.items = []
    mock_apify_client.dataset.return_value.list_items.return_value = mock_list_result

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_scrape_url(url="https://example.com")

    assert result["status"] == "error"
    assert "No content returned" in result["content"][0]["text"]


def test_scrape_url_crawler_failure(mock_apify_env, mock_apify_client):
    """Scrape URL returns error dict when the crawler Actor fails."""
    mock_apify_client.actor.return_value.call.return_value = MOCK_FAILED_RUN

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_scrape_url(url="https://example.com")

    assert result["status"] == "error"
    assert "FAILED" in result["content"][0]["text"]


def test_scrape_url_falls_back_to_text(mock_apify_env, mock_apify_client):
    """Scrape URL falls back to text field when markdown is absent."""
    item_without_markdown = {"url": "https://example.com", "text": "Plain text content"}
    mock_list_result = MagicMock()
    mock_list_result.items = [item_without_markdown]
    mock_apify_client.dataset.return_value.list_items.return_value = mock_list_result

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_scrape_url(url="https://example.com")

    assert result["status"] == "success"
    assert result["content"][0]["text"] == "Plain text content"


def test_scrape_url_invalid_url_scheme(mock_apify_env):
    """apify_scrape_url returns error for invalid URL scheme."""
    result = apify_scrape_url(url="ftp://example.com")

    assert result["status"] == "error"
    assert "Invalid URL scheme" in result["content"][0]["text"]


def test_scrape_url_missing_scheme(mock_apify_env):
    """apify_scrape_url returns error for URL without http/https scheme."""
    result = apify_scrape_url(url="example.com")

    assert result["status"] == "error"
    assert "Invalid URL scheme" in result["content"][0]["text"]


# --- Parameter validation ---


def test_run_actor_empty_actor_id(mock_apify_env):
    """apify_run_actor returns error for whitespace-only actor_id."""
    result = apify_run_actor(actor_id="   ")

    assert result["status"] == "error"
    assert "actor_id" in result["content"][0]["text"]


def test_run_actor_zero_timeout(mock_apify_env):
    """apify_run_actor returns error for non-positive timeout_secs."""
    result = apify_run_actor(actor_id="actor/valid", timeout_secs=0)

    assert result["status"] == "error"
    assert "timeout_secs" in result["content"][0]["text"]


def test_run_actor_negative_timeout(mock_apify_env):
    """apify_run_actor returns error for negative timeout_secs."""
    result = apify_run_actor(actor_id="actor/valid", timeout_secs=-5)

    assert result["status"] == "error"
    assert "timeout_secs" in result["content"][0]["text"]


def test_run_actor_zero_memory(mock_apify_env):
    """apify_run_actor returns error for non-positive memory_mbytes."""
    result = apify_run_actor(actor_id="actor/valid", memory_mbytes=0)

    assert result["status"] == "error"
    assert "memory_mbytes" in result["content"][0]["text"]


def test_run_task_empty_task_id(mock_apify_env):
    """apify_run_task returns error for whitespace-only task_id."""
    result = apify_run_task(task_id="  ")

    assert result["status"] == "error"
    assert "task_id" in result["content"][0]["text"]


def test_run_task_zero_timeout(mock_apify_env):
    """apify_run_task returns error for non-positive timeout_secs."""
    result = apify_run_task(task_id="user~my-task", timeout_secs=0)

    assert result["status"] == "error"
    assert "timeout_secs" in result["content"][0]["text"]


def test_run_task_zero_memory(mock_apify_env):
    """apify_run_task returns error for non-positive memory_mbytes."""
    result = apify_run_task(task_id="user~my-task", memory_mbytes=0)

    assert result["status"] == "error"
    assert "memory_mbytes" in result["content"][0]["text"]


def test_get_dataset_items_empty_dataset_id(mock_apify_env):
    """apify_get_dataset_items returns error for whitespace-only dataset_id."""
    result = apify_get_dataset_items(dataset_id="  ")

    assert result["status"] == "error"
    assert "dataset_id" in result["content"][0]["text"]


def test_get_dataset_items_zero_limit(mock_apify_env):
    """apify_get_dataset_items returns error for non-positive limit."""
    result = apify_get_dataset_items(dataset_id="dataset-abc", limit=0)

    assert result["status"] == "error"
    assert "limit" in result["content"][0]["text"]


def test_get_dataset_items_negative_offset(mock_apify_env):
    """apify_get_dataset_items returns error for negative offset."""
    result = apify_get_dataset_items(dataset_id="dataset-abc", offset=-1)

    assert result["status"] == "error"
    assert "offset" in result["content"][0]["text"]


def test_run_actor_and_get_dataset_zero_dataset_limit(mock_apify_env):
    """apify_run_actor_and_get_dataset returns error for non-positive dataset_items_limit."""
    result = apify_run_actor_and_get_dataset(actor_id="actor/valid", dataset_items_limit=0)

    assert result["status"] == "error"
    assert "dataset_items_limit" in result["content"][0]["text"]


def test_run_actor_and_get_dataset_negative_dataset_offset(mock_apify_env):
    """apify_run_actor_and_get_dataset returns error for negative dataset_items_offset."""
    result = apify_run_actor_and_get_dataset(actor_id="actor/valid", dataset_items_offset=-1)

    assert result["status"] == "error"
    assert "dataset_items_offset" in result["content"][0]["text"]


def test_run_task_and_get_dataset_zero_dataset_limit(mock_apify_env):
    """apify_run_task_and_get_dataset returns error for non-positive dataset_items_limit."""
    result = apify_run_task_and_get_dataset(task_id="user~my-task", dataset_items_limit=0)

    assert result["status"] == "error"
    assert "dataset_items_limit" in result["content"][0]["text"]


def test_run_task_and_get_dataset_negative_dataset_offset(mock_apify_env):
    """apify_run_task_and_get_dataset returns error for negative dataset_items_offset."""
    result = apify_run_task_and_get_dataset(task_id="user~my-task", dataset_items_offset=-1)

    assert result["status"] == "error"
    assert "dataset_items_offset" in result["content"][0]["text"]


def test_scrape_url_zero_timeout(mock_apify_env):
    """apify_scrape_url returns error for non-positive timeout_secs."""
    result = apify_scrape_url(url="https://example.com", timeout_secs=0)

    assert result["status"] == "error"
    assert "timeout_secs" in result["content"][0]["text"]


def test_scrape_url_invalid_crawler_type(mock_apify_env):
    """apify_scrape_url returns error for unsupported crawler_type."""
    result = apify_scrape_url(url="https://example.com", crawler_type="invalid")

    assert result["status"] == "error"
    assert "crawler_type" in result["content"][0]["text"]


def test_scrape_url_missing_domain(mock_apify_env):
    """apify_scrape_url returns error for URL with no domain."""
    result = apify_scrape_url(url="https://")

    assert result["status"] == "error"
    assert "domain" in result["content"][0]["text"].lower()


# --- Dependency guard ---


def test_missing_apify_client_run_actor(mock_apify_env):
    """apify_run_actor returns error dict when apify-client is not installed."""
    with patch("strands_tools.apify.HAS_APIFY_CLIENT", False):
        result = apify_run_actor(actor_id="test/actor")

    assert result["status"] == "error"
    assert "apify-client" in result["content"][0]["text"]


def test_missing_apify_client_get_dataset(mock_apify_env):
    """apify_get_dataset_items returns error dict when apify-client is not installed."""
    with patch("strands_tools.apify.HAS_APIFY_CLIENT", False):
        result = apify_get_dataset_items(dataset_id="dataset-123")

    assert result["status"] == "error"
    assert "apify-client" in result["content"][0]["text"]


def test_missing_apify_client_run_and_get(mock_apify_env):
    """apify_run_actor_and_get_dataset returns error dict when apify-client is not installed."""
    with patch("strands_tools.apify.HAS_APIFY_CLIENT", False):
        result = apify_run_actor_and_get_dataset(actor_id="test/actor")

    assert result["status"] == "error"
    assert "apify-client" in result["content"][0]["text"]


def test_missing_apify_client_run_task(mock_apify_env):
    """apify_run_task returns error dict when apify-client is not installed."""
    with patch("strands_tools.apify.HAS_APIFY_CLIENT", False):
        result = apify_run_task(task_id="user~my-task")

    assert result["status"] == "error"
    assert "apify-client" in result["content"][0]["text"]


def test_missing_apify_client_run_task_and_get(mock_apify_env):
    """apify_run_task_and_get_dataset returns error dict when apify-client is not installed."""
    with patch("strands_tools.apify.HAS_APIFY_CLIENT", False):
        result = apify_run_task_and_get_dataset(task_id="user~my-task")

    assert result["status"] == "error"
    assert "apify-client" in result["content"][0]["text"]


def test_missing_apify_client_scrape_url(mock_apify_env):
    """apify_scrape_url returns error dict when apify-client is not installed."""
    with patch("strands_tools.apify.HAS_APIFY_CLIENT", False):
        result = apify_scrape_url(url="https://example.com")

    assert result["status"] == "error"
    assert "apify-client" in result["content"][0]["text"]


# --- Missing token from tool entry points ---


def test_run_actor_missing_token(monkeypatch):
    """apify_run_actor returns error dict when APIFY_API_TOKEN is missing."""
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    result = apify_run_actor(actor_id="test/actor")

    assert result["status"] == "error"
    assert "APIFY_API_TOKEN" in result["content"][0]["text"]


def test_get_dataset_items_missing_token(monkeypatch):
    """apify_get_dataset_items returns error dict when APIFY_API_TOKEN is missing."""
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    result = apify_get_dataset_items(dataset_id="dataset-123")

    assert result["status"] == "error"
    assert "APIFY_API_TOKEN" in result["content"][0]["text"]


def test_run_actor_and_get_dataset_missing_token(monkeypatch):
    """apify_run_actor_and_get_dataset returns error dict when APIFY_API_TOKEN is missing."""
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    result = apify_run_actor_and_get_dataset(actor_id="test/actor")

    assert result["status"] == "error"
    assert "APIFY_API_TOKEN" in result["content"][0]["text"]


def test_run_task_missing_token(monkeypatch):
    """apify_run_task returns error dict when APIFY_API_TOKEN is missing."""
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    result = apify_run_task(task_id="user~my-task")

    assert result["status"] == "error"
    assert "APIFY_API_TOKEN" in result["content"][0]["text"]


def test_run_task_and_get_dataset_missing_token(monkeypatch):
    """apify_run_task_and_get_dataset returns error dict when APIFY_API_TOKEN is missing."""
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    result = apify_run_task_and_get_dataset(task_id="user~my-task")

    assert result["status"] == "error"
    assert "APIFY_API_TOKEN" in result["content"][0]["text"]


def test_scrape_url_missing_token(monkeypatch):
    """apify_scrape_url returns error dict when APIFY_API_TOKEN is missing."""
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    result = apify_scrape_url(url="https://example.com")

    assert result["status"] == "error"
    assert "APIFY_API_TOKEN" in result["content"][0]["text"]


# --- apify_google_search_scraper ---


def test_google_search_scraper_success(mock_apify_env, mock_apify_client):
    """Google Search Scraper returns structured results with correct input mapping."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_google_search_scraper(search_query="best AI frameworks", results_limit=5)

    assert result["status"] == "success"
    data = json.loads(result["content"][0]["text"])
    assert data["run_id"] == "run-HG7ml5fB1hCp8YEBA"
    assert len(data["items"]) == 3

    mock_apify_client.actor.assert_called_once_with("apify/google-search-scraper")
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["queries"] == "best AI frameworks"
    assert run_input["maxPagesPerQuery"] == 1
    assert "resultsPerPage" not in run_input


def test_google_search_scraper_multi_page(mock_apify_env, mock_apify_client):
    """Google Search Scraper calculates correct page count when results_limit exceeds 10."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        apify_google_search_scraper(search_query="AI", results_limit=25)

    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["maxPagesPerQuery"] == 3
    assert "resultsPerPage" not in run_input


def test_google_search_scraper_optional_params(mock_apify_env, mock_apify_client):
    """Google Search Scraper includes optional country and language codes when provided."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        apify_google_search_scraper(search_query="AI", results_limit=10, country_code="de", language_code="de")

    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["countryCode"] == "de"
    assert run_input["languageCode"] == "de"


def test_google_search_scraper_optional_params_omitted(mock_apify_env, mock_apify_client):
    """Google Search Scraper omits optional fields when not provided."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        apify_google_search_scraper(search_query="AI")

    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert "countryCode" not in run_input
    assert "languageCode" not in run_input


def test_google_search_scraper_missing_dependency(mock_apify_env):
    """Google Search Scraper returns error when apify-client is not installed."""
    with patch("strands_tools.apify.HAS_APIFY_CLIENT", False):
        result = apify_google_search_scraper(search_query="test")

    assert result["status"] == "error"
    assert "apify-client" in result["content"][0]["text"]


def test_google_search_scraper_missing_token(monkeypatch):
    """Google Search Scraper returns error when APIFY_API_TOKEN is missing."""
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    result = apify_google_search_scraper(search_query="test")

    assert result["status"] == "error"
    assert "APIFY_API_TOKEN" in result["content"][0]["text"]


def test_google_search_scraper_actor_failure(mock_apify_env, mock_apify_client):
    """Google Search Scraper returns error when Actor fails."""
    mock_apify_client.actor.return_value.call.return_value = MOCK_FAILED_RUN

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_google_search_scraper(search_query="test")

    assert result["status"] == "error"
    assert "FAILED" in result["content"][0]["text"]


# --- apify_google_places_scraper ---


def test_google_places_scraper_success(mock_apify_env, mock_apify_client):
    """Google Places Scraper returns structured results with correct input mapping."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_google_places_scraper(search_query="restaurants in Prague", results_limit=10)

    assert result["status"] == "success"
    data = json.loads(result["content"][0]["text"])
    assert data["run_id"] == "run-HG7ml5fB1hCp8YEBA"

    mock_apify_client.actor.assert_called_once_with("compass/crawler-google-places")
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["searchStringsArray"] == ["restaurants in Prague"]
    assert run_input["maxCrawledPlacesPerSearch"] == 10
    assert run_input["maxReviews"] == 0


def test_google_places_scraper_with_reviews(mock_apify_env, mock_apify_client):
    """Google Places Scraper sets maxReviews when include_reviews is True."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        apify_google_places_scraper(search_query="hotels in Berlin", include_reviews=True, max_reviews=10)

    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["maxReviews"] == 10


def test_google_places_scraper_reviews_disabled(mock_apify_env, mock_apify_client):
    """Google Places Scraper sets maxReviews to 0 when include_reviews is False."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        apify_google_places_scraper(search_query="cafes", include_reviews=False, max_reviews=10)

    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["maxReviews"] == 0


def test_google_places_scraper_optional_language(mock_apify_env, mock_apify_client):
    """Google Places Scraper includes language when provided."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        apify_google_places_scraper(search_query="cafes", language="de")

    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["language"] == "de"


# --- apify_youtube_scraper ---


def test_youtube_scraper_search_query(mock_apify_env, mock_apify_client):
    """YouTube Scraper returns results when given a search query."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_youtube_scraper(search_query="python tutorial", results_limit=5)

    assert result["status"] == "success"
    mock_apify_client.actor.assert_called_once_with("streamers/youtube-scraper")
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["searchQueries"] == ["python tutorial"]
    assert run_input["maxResults"] == 5
    assert "startUrls" not in run_input


def test_youtube_scraper_urls(mock_apify_env, mock_apify_client):
    """YouTube Scraper returns results when given specific URLs."""
    urls = ["https://www.youtube.com/watch?v=abc123"]
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_youtube_scraper(urls=urls)

    assert result["status"] == "success"
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["startUrls"] == [{"url": "https://www.youtube.com/watch?v=abc123"}]
    assert "searchQueries" not in run_input


def test_youtube_scraper_both_query_and_urls(mock_apify_env, mock_apify_client):
    """YouTube Scraper accepts both search_query and urls simultaneously."""
    urls = ["https://www.youtube.com/watch?v=abc123"]
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_youtube_scraper(search_query="python", urls=urls)

    assert result["status"] == "success"
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["searchQueries"] == ["python"]
    assert run_input["startUrls"] == [{"url": "https://www.youtube.com/watch?v=abc123"}]


def test_youtube_scraper_no_input(mock_apify_env):
    """YouTube Scraper returns error when neither search_query nor urls is provided."""
    result = apify_youtube_scraper()

    assert result["status"] == "error"
    assert "search_query" in result["content"][0]["text"]


# --- apify_website_content_crawler ---


def test_website_content_crawler_success(mock_apify_env, mock_apify_client):
    """Website Content Crawler returns results with correct input mapping."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_website_content_crawler(start_url="https://docs.example.com", max_pages=5, max_depth=3)

    assert result["status"] == "success"
    mock_apify_client.actor.assert_called_once_with("apify/website-content-crawler")
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["startUrls"] == [{"url": "https://docs.example.com"}]
    assert run_input["maxCrawlPages"] == 5
    assert run_input["maxCrawlDepth"] == 3
    assert run_input["proxyConfiguration"] == {"useApifyProxy": True}


def test_website_content_crawler_defaults(mock_apify_env, mock_apify_client):
    """Website Content Crawler uses correct defaults for max_pages and max_depth."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        apify_website_content_crawler(start_url="https://example.com")

    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["maxCrawlPages"] == 10
    assert run_input["maxCrawlDepth"] == 2


def test_website_content_crawler_invalid_url(mock_apify_env):
    """Website Content Crawler returns error for invalid URL."""
    result = apify_website_content_crawler(start_url="not-a-url")

    assert result["status"] == "error"
    assert "Invalid URL" in result["content"][0]["text"]


# --- apify_ecommerce_scraper ---


def test_ecommerce_scraper_success(mock_apify_env, mock_apify_client):
    """E-commerce Scraper returns results with correct input mapping for product URL."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_ecommerce_scraper(url="https://www.amazon.com/dp/B0TEST", results_limit=10)

    assert result["status"] == "success"
    data = json.loads(result["content"][0]["text"])
    assert data["run_id"] == "run-HG7ml5fB1hCp8YEBA"

    mock_apify_client.actor.assert_called_once_with("apify/e-commerce-scraping-tool")
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["detailsUrls"] == [{"url": "https://www.amazon.com/dp/B0TEST"}]
    assert "listingUrls" not in run_input
    assert run_input["maxProductResults"] == 10


def test_ecommerce_scraper_listing_url(mock_apify_env, mock_apify_client):
    """E-commerce Scraper uses listingUrls when url_type is 'listing'."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_ecommerce_scraper(
            url="https://www.amazon.com/s?k=headphones", url_type="listing", results_limit=10
        )

    assert result["status"] == "success"
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["listingUrls"] == [{"url": "https://www.amazon.com/s?k=headphones"}]
    assert "detailsUrls" not in run_input


def test_ecommerce_scraper_invalid_url_type(mock_apify_env):
    """E-commerce Scraper returns error for invalid url_type."""
    result = apify_ecommerce_scraper(url="https://www.amazon.com/dp/B0TEST", url_type="invalid")

    assert result["status"] == "error"
    assert "url_type" in result["content"][0]["text"]


def test_ecommerce_scraper_invalid_url(mock_apify_env):
    """E-commerce Scraper returns error for invalid URL."""
    result = apify_ecommerce_scraper(url="not-a-url")

    assert result["status"] == "error"
    assert "Invalid URL" in result["content"][0]["text"]


def test_ecommerce_scraper_actor_failure(mock_apify_env, mock_apify_client):
    """E-commerce Scraper returns error when Actor fails."""
    mock_apify_client.actor.return_value.call.return_value = MOCK_FAILED_RUN

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_ecommerce_scraper(url="https://www.amazon.com/dp/B0TEST")

    assert result["status"] == "error"
    assert "FAILED" in result["content"][0]["text"]
