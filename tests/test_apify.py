"""Tests for the Apify tools."""

import json
from unittest.mock import MagicMock, patch

import pytest

from strands_tools import apify
from strands_tools.apify import (
    ApifyToolClient,
    _extract_linkedin_username,
    apify_facebook_posts_scraper,
    apify_get_dataset_items,
    apify_instagram_scraper,
    apify_linkedin_profile_detail,
    apify_linkedin_profile_posts,
    apify_linkedin_profile_search,
    apify_run_actor,
    apify_run_actor_and_get_dataset,
    apify_run_task,
    apify_run_task_and_get_dataset,
    apify_scrape_url,
    apify_tiktok_scraper,
    apify_twitter_scraper,
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


# --- _extract_linkedin_username ---


def test_extract_linkedin_username_from_url():
    """Extracts username from a standard LinkedIn profile URL."""
    assert _extract_linkedin_username("https://www.linkedin.com/in/neal-mohan") == "neal-mohan"


def test_extract_linkedin_username_from_url_trailing_slash():
    """Extracts username from a LinkedIn URL with trailing slash."""
    assert _extract_linkedin_username("https://www.linkedin.com/in/neal-mohan/") == "neal-mohan"


def test_extract_linkedin_username_bare():
    """Passes through a bare username unchanged."""
    assert _extract_linkedin_username("neal-mohan") == "neal-mohan"


def test_extract_linkedin_username_non_profile_url():
    """Non-/in/ LinkedIn URL is returned as-is."""
    assert (
        _extract_linkedin_username("https://www.linkedin.com/company/apify") == "https://www.linkedin.com/company/apify"
    )


# --- apify_instagram_scraper ---


def test_instagram_scraper_search_success(mock_apify_env, mock_apify_client):
    """Instagram scraper with search query maps to 'search' field with resultsType and searchLimit."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_instagram_scraper(search_query="apify", results_limit=10, search_type="user")

    assert result["status"] == "success"
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    run_input = call_kwargs["run_input"]
    assert run_input["search"] == "apify"
    assert run_input["searchType"] == "user"
    assert run_input["resultsLimit"] == 10
    assert run_input["resultsType"] == "posts"
    assert run_input["searchLimit"] == 10
    mock_apify_client.actor.assert_called_once_with("apify/instagram-scraper")


def test_instagram_scraper_with_urls(mock_apify_env, mock_apify_client):
    """Instagram scraper with explicit URLs maps to 'directUrls'."""
    urls = ["https://www.instagram.com/apify/"]
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_instagram_scraper(urls=urls, results_limit=5)

    assert result["status"] == "success"
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    run_input = call_kwargs["run_input"]
    assert run_input["directUrls"] == urls
    assert run_input["resultsType"] == "posts"
    assert "search" not in run_input


def test_instagram_scraper_results_type(mock_apify_env, mock_apify_client):
    """Instagram scraper passes results_type to Actor input."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_instagram_scraper(search_query="apify", results_type="comments")

    assert result["status"] == "success"
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["resultsType"] == "comments"


def test_instagram_scraper_invalid_results_type(mock_apify_env):
    """Instagram scraper returns error for invalid results_type."""
    result = apify_instagram_scraper(search_query="apify", results_type="invalid")

    assert result["status"] == "error"
    assert "results_type" in result["content"][0]["text"]


def test_instagram_scraper_invalid_search_type(mock_apify_env):
    """Instagram scraper returns error for invalid search_type."""
    result = apify_instagram_scraper(search_query="apify", search_type="invalid")

    assert result["status"] == "error"
    assert "search_type" in result["content"][0]["text"]


def test_instagram_scraper_url_in_search_query(mock_apify_env, mock_apify_client):
    """Instagram scraper routes URL-like search_query to 'directUrls'."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_instagram_scraper(search_query="https://www.instagram.com/apify/")

    assert result["status"] == "success"
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    run_input = call_kwargs["run_input"]
    assert run_input["directUrls"] == ["https://www.instagram.com/apify/"]
    assert "search" not in run_input


def test_instagram_scraper_missing_params(mock_apify_env):
    """Instagram scraper returns error when neither search_query nor urls provided."""
    result = apify_instagram_scraper()

    assert result["status"] == "error"
    assert "search_query" in result["content"][0]["text"] or "urls" in result["content"][0]["text"]


# --- apify_linkedin_profile_posts ---


def test_linkedin_profile_posts_success(mock_apify_env, mock_apify_client):
    """LinkedIn profile posts maps profile URL to username and results_limit to limit."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_linkedin_profile_posts(
            profile_url="https://www.linkedin.com/in/neal-mohan",
            results_limit=15,
        )

    assert result["status"] == "success"
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    run_input = call_kwargs["run_input"]
    assert run_input["username"] == "neal-mohan"
    assert run_input["limit"] == 15
    mock_apify_client.actor.assert_called_once_with("apimaestro/linkedin-profile-posts")


def test_linkedin_profile_posts_caps_limit(mock_apify_env, mock_apify_client):
    """LinkedIn profile posts caps the limit at 100 per Actor constraint."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        apify_linkedin_profile_posts(profile_url="neal-mohan", results_limit=200)

    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs["run_input"]["limit"] == 100


# --- apify_linkedin_profile_search ---


def test_linkedin_profile_search_success(mock_apify_env, mock_apify_client):
    """LinkedIn profile search maps search_query to searchQuery and results_limit to maxItems."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_linkedin_profile_search(search_query="software engineer SF", results_limit=25)

    assert result["status"] == "success"
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    run_input = call_kwargs["run_input"]
    assert run_input["searchQuery"] == "software engineer SF"
    assert run_input["maxItems"] == 25
    assert run_input["profileScraperMode"] == "Short"
    assert "locations" not in run_input
    assert "currentJobTitles" not in run_input
    mock_apify_client.actor.assert_called_once_with("harvestapi/linkedin-profile-search")


def test_linkedin_profile_search_with_filters(mock_apify_env, mock_apify_client):
    """LinkedIn profile search passes locations and job title filters."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_linkedin_profile_search(
            search_query="engineer",
            locations=["San Francisco", "New York"],
            current_job_titles=["Software Engineer"],
            profile_scraper_mode="Full",
        )

    assert result["status"] == "success"
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["locations"] == ["San Francisco", "New York"]
    assert run_input["currentJobTitles"] == ["Software Engineer"]
    assert run_input["profileScraperMode"] == "Full"


def test_linkedin_profile_search_invalid_mode(mock_apify_env):
    """LinkedIn profile search returns error for invalid profile_scraper_mode."""
    result = apify_linkedin_profile_search(search_query="test", profile_scraper_mode="Invalid")

    assert result["status"] == "error"
    assert "profile_scraper_mode" in result["content"][0]["text"]


# --- apify_linkedin_profile_detail ---


def test_linkedin_profile_detail_success(mock_apify_env, mock_apify_client):
    """LinkedIn profile detail maps profile URL to username field."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_linkedin_profile_detail(profile_url="https://www.linkedin.com/in/neal-mohan")

    assert result["status"] == "success"
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    run_input = call_kwargs["run_input"]
    assert run_input["username"] == "neal-mohan"
    assert run_input["includeEmail"] is False
    mock_apify_client.actor.assert_called_once_with("apimaestro/linkedin-profile-detail")


def test_linkedin_profile_detail_bare_username(mock_apify_env, mock_apify_client):
    """LinkedIn profile detail accepts a bare username."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_linkedin_profile_detail(profile_url="neal-mohan")

    assert result["status"] == "success"
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs["run_input"]["username"] == "neal-mohan"


def test_linkedin_profile_detail_include_email(mock_apify_env, mock_apify_client):
    """LinkedIn profile detail passes includeEmail when requested."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_linkedin_profile_detail(profile_url="neal-mohan", include_email=True)

    assert result["status"] == "success"
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["includeEmail"] is True


# --- apify_twitter_scraper ---


def test_twitter_scraper_search_success(mock_apify_env, mock_apify_client):
    """Twitter scraper maps search_query to searchTerms array with sort and maxItems."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_twitter_scraper(search_query="from:NASA", results_limit=30)

    assert result["status"] == "success"
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    run_input = call_kwargs["run_input"]
    assert run_input["searchTerms"] == ["from:NASA"]
    assert run_input["maxItems"] == 30
    assert run_input["sort"] == "Latest"
    assert "tweetLanguage" not in run_input
    mock_apify_client.actor.assert_called_once_with("apidojo/twitter-scraper-lite")


def test_twitter_scraper_with_urls(mock_apify_env, mock_apify_client):
    """Twitter scraper maps urls to startUrls as objects."""
    tweet_urls = ["https://x.com/elonmusk/status/1728108619189874825"]
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_twitter_scraper(urls=tweet_urls)

    assert result["status"] == "success"
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs["run_input"]["startUrls"] == [{"url": "https://x.com/elonmusk/status/1728108619189874825"}]


def test_twitter_scraper_with_handles(mock_apify_env, mock_apify_client):
    """Twitter scraper passes twitterHandles when provided."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_twitter_scraper(twitter_handles=["NASA", "elonmusk"])

    assert result["status"] == "success"
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["twitterHandles"] == ["NASA", "elonmusk"]


def test_twitter_scraper_sort_and_language(mock_apify_env, mock_apify_client):
    """Twitter scraper passes sort and tweet_language parameters."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_twitter_scraper(search_query="AI", sort="Top", tweet_language="en")

    assert result["status"] == "success"
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["sort"] == "Top"
    assert run_input["tweetLanguage"] == "en"


def test_twitter_scraper_invalid_sort(mock_apify_env):
    """Twitter scraper returns error for invalid sort option."""
    result = apify_twitter_scraper(search_query="test", sort="Invalid")

    assert result["status"] == "error"
    assert "sort" in result["content"][0]["text"]


def test_twitter_scraper_missing_params(mock_apify_env):
    """Twitter scraper returns error when no input provided."""
    result = apify_twitter_scraper()

    assert result["status"] == "error"
    assert "search_query" in result["content"][0]["text"] or "urls" in result["content"][0]["text"]


# --- apify_tiktok_scraper ---


def test_tiktok_scraper_search_success(mock_apify_env, mock_apify_client):
    """TikTok scraper maps search_query to searchQueries and results_limit to resultsPerPage."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_tiktok_scraper(search_query="cooking", results_limit=15)

    assert result["status"] == "success"
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    run_input = call_kwargs["run_input"]
    assert run_input["searchQueries"] == ["cooking"]
    assert run_input["resultsPerPage"] == 15
    mock_apify_client.actor.assert_called_once_with("clockworks/tiktok-scraper")


def test_tiktok_scraper_with_urls(mock_apify_env, mock_apify_client):
    """TikTok scraper maps urls to postURLs."""
    post_urls = ["https://www.tiktok.com/@user/video/123"]
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_tiktok_scraper(urls=post_urls)

    assert result["status"] == "success"
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    assert call_kwargs["run_input"]["postURLs"] == post_urls


def test_tiktok_scraper_with_hashtags(mock_apify_env, mock_apify_client):
    """TikTok scraper passes hashtags to the dedicated hashtags input."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_tiktok_scraper(hashtags=["fyp", "cooking"])

    assert result["status"] == "success"
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["hashtags"] == ["fyp", "cooking"]
    assert "searchQueries" not in run_input


def test_tiktok_scraper_with_profiles(mock_apify_env, mock_apify_client):
    """TikTok scraper passes profiles to the dedicated profiles input."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_tiktok_scraper(profiles=["charlidamelio", "khaby.lame"])

    assert result["status"] == "success"
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["profiles"] == ["charlidamelio", "khaby.lame"]
    assert "searchQueries" not in run_input


def test_tiktok_scraper_missing_params(mock_apify_env):
    """TikTok scraper returns error when no input provided."""
    result = apify_tiktok_scraper()

    assert result["status"] == "error"
    assert "search_query" in result["content"][0]["text"] or "hashtags" in result["content"][0]["text"]


# --- apify_facebook_posts_scraper ---


def test_facebook_posts_scraper_success(mock_apify_env, mock_apify_client):
    """Facebook posts scraper maps page_url to startUrls and results_limit to resultsLimit."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_facebook_posts_scraper(
            page_url="https://www.facebook.com/apify",
            results_limit=10,
        )

    assert result["status"] == "success"
    call_kwargs = mock_apify_client.actor.return_value.call.call_args.kwargs
    run_input = call_kwargs["run_input"]
    assert run_input["startUrls"] == [{"url": "https://www.facebook.com/apify"}]
    assert run_input["resultsLimit"] == 10
    assert "onlyPostsNewerThan" not in run_input
    mock_apify_client.actor.assert_called_once_with("apify/facebook-posts-scraper")


def test_facebook_posts_scraper_with_date_filter(mock_apify_env, mock_apify_client):
    """Facebook posts scraper passes onlyPostsNewerThan when provided."""
    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_facebook_posts_scraper(
            page_url="https://www.facebook.com/apify",
            only_posts_newer_than="2024-01-01",
        )

    assert result["status"] == "success"
    run_input = mock_apify_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["onlyPostsNewerThan"] == "2024-01-01"


# --- Social media: dependency and token guards ---


def test_social_media_missing_dependency(mock_apify_env):
    """Social media tools return error when apify-client is not installed."""
    with patch("strands_tools.apify.HAS_APIFY_CLIENT", False):
        result = apify_instagram_scraper(search_query="test")

    assert result["status"] == "error"
    assert "apify-client" in result["content"][0]["text"]


def test_social_media_missing_token(monkeypatch):
    """Social media tools return error when APIFY_API_TOKEN is missing."""
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    result = apify_twitter_scraper(search_query="test")

    assert result["status"] == "error"
    assert "APIFY_API_TOKEN" in result["content"][0]["text"]


def test_social_media_actor_failure(mock_apify_env, mock_apify_client):
    """Social media tools return error when the underlying Actor run fails."""
    mock_apify_client.actor.return_value.call.return_value = MOCK_FAILED_RUN

    with patch("strands_tools.apify.ApifyClient", return_value=mock_apify_client):
        result = apify_facebook_posts_scraper(page_url="https://www.facebook.com/apify")

    assert result["status"] == "error"
    assert "FAILED" in result["content"][0]["text"]
