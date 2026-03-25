"""Apify platform tools for Strands Agents.

This module provides web scraping, data extraction, automation, and social media
scraping capabilities using the Apify platform. It lets you run any Actor, task,
fetch dataset results, scrape individual URLs, and scrape popular social media
platforms.

Available Tools:
---------------
Core tools:
- apify_run_actor: Run any Apify Actor with custom input
- apify_get_dataset_items: Fetch items from an Apify dataset with pagination
- apify_run_actor_and_get_dataset: Run an Actor and fetch results in one step
- apify_run_task: Run a saved Actor task with optional input overrides
- apify_run_task_and_get_dataset: Run a task and fetch results in one step
- apify_scrape_url: Scrape a single URL and return content as Markdown

Social media tools:
- apify_instagram_scraper: Scrape Instagram profiles, posts, or hashtags
- apify_linkedin_profile_posts: Scrape posts from a LinkedIn profile
- apify_linkedin_profile_search: Search for LinkedIn profiles by keywords
- apify_linkedin_profile_detail: Get detailed LinkedIn profile information
- apify_twitter_scraper: Scrape tweets from Twitter/X
- apify_tiktok_scraper: Scrape TikTok videos, profiles, or hashtags
- apify_facebook_posts_scraper: Scrape posts from Facebook pages

Setup Requirements:
------------------
1. Create an Apify account at https://apify.com
2. Obtain your API token: Apify Console > Settings > API & Integrations > Personal API tokens
3. Install the optional dependency: pip install strands-agents-tools[apify]
4. Set the environment variable:
   APIFY_API_TOKEN=your_api_token_here

Usage Examples:
--------------
Register all core tools at once via the preset list:

```python
from strands import Agent
from strands_tools.apify import APIFY_CORE_TOOLS

agent = Agent(tools=APIFY_CORE_TOOLS)
```

Register all social media tools:

```python
from strands_tools.apify import APIFY_SOCIAL_TOOLS

agent = Agent(tools=APIFY_SOCIAL_TOOLS)
```

Or combine both, or pick individual tools:

```python
from strands import Agent
from strands_tools.apify import APIFY_CORE_TOOLS
from strands_tools import apify

agent = Agent(tools=[
    *APIFY_CORE_TOOLS,
    apify.apify_instagram_scraper,
    apify.apify_twitter_scraper,
])
```
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from rich.panel import Panel
from rich.text import Text
from strands import tool

from strands_tools.utils import console_util

logger = logging.getLogger(__name__)
console = console_util.create()

try:
    from apify_client import ApifyClient
    from apify_client.errors import ApifyApiError

    HAS_APIFY_CLIENT = True
except ImportError:
    HAS_APIFY_CLIENT = False

WEBSITE_CONTENT_CRAWLER = "apify/website-content-crawler"
TRACKING_HEADER = {"x-apify-integration-platform": "strands-agents"}
ERROR_PANEL_TITLE = "[bold red]Apify Error[/bold red]"
DEFAULT_TIMEOUT_SECS = 300
DEFAULT_SCRAPE_TIMEOUT_SECS = 120
DEFAULT_DATASET_ITEMS_LIMIT = 100
VALID_CRAWLER_TYPES = ("playwright:adaptive", "playwright:firefox", "cheerio")


# --- Helper functions ---


def _check_dependency() -> None:
    """Raise ImportError if apify-client is not installed."""
    if not HAS_APIFY_CLIENT:
        raise ImportError("apify-client package is required. Install with: pip install strands-agents-tools[apify]")


def _format_error(e: Exception) -> str:
    """Map exceptions to user-friendly error messages, with special handling for ApifyApiError."""
    if HAS_APIFY_CLIENT and isinstance(e, ApifyApiError):
        status_code = getattr(e, "status_code", None)
        msg = getattr(e, "message", str(e))
        match status_code:
            case 400:
                return f"Invalid request: {msg}"
            case 401:
                return "Authentication failed. Verify your APIFY_API_TOKEN is valid."
            case 402:
                return "Insufficient Apify plan credits or subscription limits exceeded."
            case 404:
                return f"Resource not found: {msg}"
            case 408:
                return f"Actor run timed out: {msg}"
            case 429:
                return (
                    "Rate limit exceeded. The Apify client retries automatically; "
                    "if this persists, reduce request frequency."
                )
            case _:
                return f"Apify API error ({status_code}): {msg}"
    return str(e)


def _error_result(e: Exception, tool_name: str) -> Dict[str, Any]:
    """Build a structured error response and display an error panel."""
    message = _format_error(e)
    logger.error("%s failed: %s", tool_name, message)
    console.print(Panel(Text(message, style="red"), title=ERROR_PANEL_TITLE, border_style="red"))
    return {"status": "error", "content": [{"text": message}]}


def _success_result(text: str, panel_body: str, panel_title: str) -> Dict[str, Any]:
    """Build a structured success response and display a success panel."""
    console.print(Panel(panel_body, title=f"[bold cyan]{panel_title}[/bold cyan]", border_style="green"))
    return {"status": "success", "content": [{"text": text}]}


class ApifyToolClient:
    """Helper class encapsulating Apify API interactions via apify-client."""

    def __init__(self) -> None:
        token = os.getenv("APIFY_API_TOKEN", "")
        if not token:
            raise ValueError(
                "APIFY_API_TOKEN environment variable is not set. "
                "Get your token at https://console.apify.com/account/integrations"
            )
        self.client: "ApifyClient" = ApifyClient(token, headers=TRACKING_HEADER)

    @staticmethod
    def _check_run_status(actor_run: Dict[str, Any], label: str) -> None:
        """Raise RuntimeError if the Actor run did not succeed."""
        status = actor_run.get("status", "UNKNOWN")
        if status != "SUCCEEDED":
            run_id = actor_run.get("id", "N/A")
            raise RuntimeError(f"{label} finished with status {status}. Run ID: {run_id}")

    @staticmethod
    def _validate_url(url: str) -> None:
        """Raise ValueError if the URL does not have a valid HTTP(S) scheme and domain."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme '{parsed.scheme}'. Only http and https URLs are supported.")
        if not parsed.netloc:
            raise ValueError(f"Invalid URL '{url}'. A domain is required.")

    @staticmethod
    def _validate_identifier(value: str, name: str) -> None:
        """Raise ValueError if a required string identifier is empty or whitespace-only."""
        if not value.strip():
            raise ValueError(f"'{name}' must be a non-empty string.")

    @staticmethod
    def _validate_positive(value: int, name: str) -> None:
        """Raise ValueError if the value is not a positive integer (> 0)."""
        if value <= 0:
            raise ValueError(f"'{name}' must be a positive integer, got {value}.")

    @staticmethod
    def _validate_non_negative(value: int, name: str) -> None:
        """Raise ValueError if the value is negative."""
        if value < 0:
            raise ValueError(f"'{name}' must be a non-negative integer, got {value}.")

    def run_actor(
        self,
        actor_id: str,
        run_input: Optional[Dict[str, Any]] = None,
        timeout_secs: int = DEFAULT_TIMEOUT_SECS,
        memory_mbytes: Optional[int] = None,
        build: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run an Apify Actor synchronously and return run metadata."""
        self._validate_identifier(actor_id, "actor_id")
        self._validate_positive(timeout_secs, "timeout_secs")
        if memory_mbytes is not None:
            self._validate_positive(memory_mbytes, "memory_mbytes")

        call_kwargs: Dict[str, Any] = {
            "run_input": run_input or {},
            "timeout_secs": timeout_secs,
            "logger": None,  # Suppress verbose apify-client logging not useful to end users
        }
        if memory_mbytes is not None:
            call_kwargs["memory_mbytes"] = memory_mbytes
        if build is not None:
            call_kwargs["build"] = build

        actor_run = self.client.actor(actor_id).call(**call_kwargs)
        if actor_run is None:
            raise RuntimeError(f"Actor {actor_id} returned no run data (possible wait timeout).")
        self._check_run_status(actor_run, f"Actor {actor_id}")

        return {
            "run_id": actor_run.get("id"),
            "status": actor_run.get("status"),
            "dataset_id": actor_run.get("defaultDatasetId"),
            "started_at": actor_run.get("startedAt"),
            "finished_at": actor_run.get("finishedAt"),
        }

    def get_dataset_items(
        self,
        dataset_id: str,
        limit: int = DEFAULT_DATASET_ITEMS_LIMIT,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Fetch items from an Apify dataset."""
        self._validate_identifier(dataset_id, "dataset_id")
        self._validate_positive(limit, "limit")
        self._validate_non_negative(offset, "offset")

        result = self.client.dataset(dataset_id).list_items(limit=limit, offset=offset)
        return list(result.items)

    def run_actor_and_get_dataset(
        self,
        actor_id: str,
        run_input: Optional[Dict[str, Any]] = None,
        timeout_secs: int = DEFAULT_TIMEOUT_SECS,
        memory_mbytes: Optional[int] = None,
        build: Optional[str] = None,
        dataset_items_limit: int = DEFAULT_DATASET_ITEMS_LIMIT,
        dataset_items_offset: int = 0,
    ) -> Dict[str, Any]:
        """Run an Actor synchronously, then fetch its default dataset items."""
        self._validate_positive(dataset_items_limit, "dataset_items_limit")
        self._validate_non_negative(dataset_items_offset, "dataset_items_offset")

        run_metadata = self.run_actor(
            actor_id=actor_id,
            run_input=run_input,
            timeout_secs=timeout_secs,
            memory_mbytes=memory_mbytes,
            build=build,
        )
        dataset_id = run_metadata["dataset_id"]
        if not dataset_id:
            raise RuntimeError(f"Actor {actor_id} run has no default dataset.")
        items = self.get_dataset_items(dataset_id=dataset_id, limit=dataset_items_limit, offset=dataset_items_offset)
        return {**run_metadata, "items": items}

    def run_task(
        self,
        task_id: str,
        task_input: Optional[Dict[str, Any]] = None,
        timeout_secs: int = DEFAULT_TIMEOUT_SECS,
        memory_mbytes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run an Apify task synchronously and return run metadata."""
        self._validate_identifier(task_id, "task_id")
        self._validate_positive(timeout_secs, "timeout_secs")
        if memory_mbytes is not None:
            self._validate_positive(memory_mbytes, "memory_mbytes")

        call_kwargs: Dict[str, Any] = {"timeout_secs": timeout_secs}
        if task_input is not None:
            call_kwargs["task_input"] = task_input
        if memory_mbytes is not None:
            call_kwargs["memory_mbytes"] = memory_mbytes

        task_run = self.client.task(task_id).call(**call_kwargs)
        if task_run is None:
            raise RuntimeError(f"Task {task_id} returned no run data (possible wait timeout).")
        self._check_run_status(task_run, f"Task {task_id}")

        return {
            "run_id": task_run.get("id"),
            "status": task_run.get("status"),
            "dataset_id": task_run.get("defaultDatasetId"),
            "started_at": task_run.get("startedAt"),
            "finished_at": task_run.get("finishedAt"),
        }

    def run_task_and_get_dataset(
        self,
        task_id: str,
        task_input: Optional[Dict[str, Any]] = None,
        timeout_secs: int = DEFAULT_TIMEOUT_SECS,
        memory_mbytes: Optional[int] = None,
        dataset_items_limit: int = DEFAULT_DATASET_ITEMS_LIMIT,
        dataset_items_offset: int = 0,
    ) -> Dict[str, Any]:
        """Run a task synchronously, then fetch its default dataset items."""
        self._validate_positive(dataset_items_limit, "dataset_items_limit")
        self._validate_non_negative(dataset_items_offset, "dataset_items_offset")

        run_metadata = self.run_task(
            task_id=task_id,
            task_input=task_input,
            timeout_secs=timeout_secs,
            memory_mbytes=memory_mbytes,
        )
        dataset_id = run_metadata["dataset_id"]
        if not dataset_id:
            raise RuntimeError(f"Task {task_id} run has no default dataset.")
        items = self.get_dataset_items(dataset_id=dataset_id, limit=dataset_items_limit, offset=dataset_items_offset)
        return {**run_metadata, "items": items}

    def scrape_url(
        self,
        url: str,
        timeout_secs: int = DEFAULT_SCRAPE_TIMEOUT_SECS,
        crawler_type: str = "cheerio",
    ) -> str:
        """Scrape a single URL using Website Content Crawler and return markdown."""
        self._validate_url(url)
        self._validate_positive(timeout_secs, "timeout_secs")
        if crawler_type not in VALID_CRAWLER_TYPES:
            raise ValueError(
                f"Invalid crawler_type '{crawler_type}'. Must be one of: {', '.join(VALID_CRAWLER_TYPES)}."
            )

        run_input: Dict[str, Any] = {
            "startUrls": [{"url": url}],
            "maxCrawlPages": 1,
            "crawlerType": crawler_type,
        }
        actor_run = self.client.actor(WEBSITE_CONTENT_CRAWLER).call(
            run_input=run_input,
            timeout_secs=timeout_secs,
            logger=None,  # Suppress verbose apify-client logging not useful to end users
        )
        self._check_run_status(actor_run, "Website Content Crawler")

        dataset_id = actor_run.get("defaultDatasetId")
        result = self.client.dataset(dataset_id).list_items(limit=1)
        items = list(result.items)

        if not items:
            raise RuntimeError(f"No content returned for URL: {url}")

        return str(items[0].get("markdown") or items[0].get("text", ""))


# --- Core tool functions ---


@tool
def apify_run_actor(
    actor_id: str,
    run_input: Optional[Dict[str, Any]] = None,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
    memory_mbytes: Optional[int] = None,
    build: Optional[str] = None,
) -> Dict[str, Any]:
    """Run any Apify Actor and return the run metadata as JSON.

    Executes the Actor synchronously - blocks until the Actor run finishes or the timeout
    is reached. Use this when you need to run a specific Actor and then inspect or process
    the results separately.

    Common Actors:
    - "apify/website-content-crawler" - scrape websites and extract content
    - "apify/web-scraper" - general-purpose web scraper
    - "apify/google-search-scraper" - scrape Google search results

    Args:
        actor_id: Actor identifier, e.g. "apify/website-content-crawler" or "username/actor-name".
        run_input: JSON-serializable input for the Actor. Each Actor defines its own input schema.
        timeout_secs: Maximum time in seconds to wait for the Actor run to finish. Defaults to 300.
        memory_mbytes: Memory allocation in MB for the Actor run. Uses Actor default if not set.
        build: Actor build tag or number to run a specific version. Uses latest build if not set.

    Returns:
        Dict with status and content containing run metadata: run_id, status, dataset_id,
        started_at, finished_at.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        result = client.run_actor(
            actor_id=actor_id,
            run_input=run_input,
            timeout_secs=timeout_secs,
            memory_mbytes=memory_mbytes,
            build=build,
        )
        return _success_result(
            text=json.dumps(result, indent=2, default=str),
            panel_body=(
                f"[green]Actor run completed[/green]\n"
                f"Actor: {actor_id}\n"
                f"Run ID: {result['run_id']}\n"
                f"Status: {result['status']}\n"
                f"Dataset ID: {result['dataset_id']}"
            ),
            panel_title="Apify: Run Actor",
        )
    except Exception as e:
        return _error_result(e, "apify_run_actor")


@tool
def apify_get_dataset_items(
    dataset_id: str,
    limit: int = DEFAULT_DATASET_ITEMS_LIMIT,
    offset: int = 0,
) -> Dict[str, Any]:
    """Fetch items from an existing Apify dataset and return them as JSON.

    Use this after running an Actor to retrieve the structured results from its
    default dataset, or to access any dataset by ID.

    Args:
        dataset_id: The Apify dataset ID to fetch items from.
        limit: Maximum number of items to return. Defaults to 100.
        offset: Number of items to skip for pagination. Defaults to 0.

    Returns:
        Dict with status and content containing an array of dataset items.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        items = client.get_dataset_items(dataset_id=dataset_id, limit=limit, offset=offset)
        return _success_result(
            text=json.dumps(items, indent=2, default=str),
            panel_body=(
                f"[green]Dataset items retrieved[/green]\nDataset ID: {dataset_id}\nItems returned: {len(items)}"
            ),
            panel_title="Apify: Dataset Items",
        )
    except Exception as e:
        return _error_result(e, "apify_get_dataset_items")


@tool
def apify_run_actor_and_get_dataset(
    actor_id: str,
    run_input: Optional[Dict[str, Any]] = None,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
    memory_mbytes: Optional[int] = None,
    build: Optional[str] = None,
    dataset_items_limit: int = DEFAULT_DATASET_ITEMS_LIMIT,
    dataset_items_offset: int = 0,
) -> Dict[str, Any]:
    """Run an Apify Actor and fetch its dataset results in one step.

    Convenience tool that combines running an Actor and fetching its default
    dataset items into a single call. Use this when you want both the run metadata and the
    result data without making two separate tool calls.

    Args:
        actor_id: Actor identifier, e.g. "apify/website-content-crawler" or "username/actor-name".
        run_input: JSON-serializable input for the Actor.
        timeout_secs: Maximum time in seconds to wait for the Actor run to finish. Defaults to 300.
        memory_mbytes: Memory allocation in MB for the Actor run.
        build: Actor build tag or number to run a specific version. Uses latest build if not set.
        dataset_items_limit: Maximum number of dataset items to return. Defaults to 100.
        dataset_items_offset: Number of dataset items to skip for pagination. Defaults to 0.

    Returns:
        Dict with status and content containing run metadata (run_id, status, dataset_id,
        started_at, finished_at) plus an "items" array containing the dataset results.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        result = client.run_actor_and_get_dataset(
            actor_id=actor_id,
            run_input=run_input,
            timeout_secs=timeout_secs,
            memory_mbytes=memory_mbytes,
            build=build,
            dataset_items_limit=dataset_items_limit,
            dataset_items_offset=dataset_items_offset,
        )
        return _success_result(
            text=json.dumps(result, indent=2, default=str),
            panel_body=(
                f"[green]Actor run completed with dataset[/green]\n"
                f"Actor: {actor_id}\n"
                f"Run ID: {result['run_id']}\n"
                f"Status: {result['status']}\n"
                f"Dataset ID: {result['dataset_id']}\n"
                f"Items returned: {len(result['items'])}"
            ),
            panel_title="Apify: Run Actor + Dataset",
        )
    except Exception as e:
        return _error_result(e, "apify_run_actor_and_get_dataset")


@tool
def apify_run_task(
    task_id: str,
    task_input: Optional[Dict[str, Any]] = None,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
    memory_mbytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Run an Apify task and return the run metadata as JSON.

    Tasks are saved Actor configurations with preset inputs. Use this when a task
    has already been configured in Apify Console, so you don't need to specify
    the full Actor input every time.

    Args:
        task_id: Task identifier, e.g. "user/my-task" or a task ID string.
        task_input: Optional JSON-serializable input to override the task's default input.
        timeout_secs: Maximum time in seconds to wait for the task run to finish. Defaults to 300.
        memory_mbytes: Memory allocation in MB for the task run. Uses task default if not set.

    Returns:
        Dict with status and content containing run metadata: run_id, status, dataset_id,
        started_at, finished_at.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        result = client.run_task(
            task_id=task_id,
            task_input=task_input,
            timeout_secs=timeout_secs,
            memory_mbytes=memory_mbytes,
        )
        return _success_result(
            text=json.dumps(result, indent=2, default=str),
            panel_body=(
                f"[green]Task run completed[/green]\n"
                f"Task: {task_id}\n"
                f"Run ID: {result['run_id']}\n"
                f"Status: {result['status']}\n"
                f"Dataset ID: {result['dataset_id']}"
            ),
            panel_title="Apify: Run Task",
        )
    except Exception as e:
        return _error_result(e, "apify_run_task")


@tool
def apify_run_task_and_get_dataset(
    task_id: str,
    task_input: Optional[Dict[str, Any]] = None,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
    memory_mbytes: Optional[int] = None,
    dataset_items_limit: int = DEFAULT_DATASET_ITEMS_LIMIT,
    dataset_items_offset: int = 0,
) -> Dict[str, Any]:
    """Run an Apify task and fetch its dataset results in one step.

    Convenience tool that combines running a task and fetching its default
    dataset items into a single call. Use this when you want both the run metadata and the
    result data without making two separate tool calls.

    Args:
        task_id: Task identifier, e.g. "user/my-task" or a task ID string.
        task_input: Optional JSON-serializable input to override the task's default input.
        timeout_secs: Maximum time in seconds to wait for the task run to finish. Defaults to 300.
        memory_mbytes: Memory allocation in MB for the task run.
        dataset_items_limit: Maximum number of dataset items to return. Defaults to 100.
        dataset_items_offset: Number of dataset items to skip for pagination. Defaults to 0.

    Returns:
        Dict with status and content containing run metadata (run_id, status, dataset_id,
        started_at, finished_at) plus an "items" array containing the dataset results.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        result = client.run_task_and_get_dataset(
            task_id=task_id,
            task_input=task_input,
            timeout_secs=timeout_secs,
            memory_mbytes=memory_mbytes,
            dataset_items_limit=dataset_items_limit,
            dataset_items_offset=dataset_items_offset,
        )
        return _success_result(
            text=json.dumps(result, indent=2, default=str),
            panel_body=(
                f"[green]Task run completed with dataset[/green]\n"
                f"Task: {task_id}\n"
                f"Run ID: {result['run_id']}\n"
                f"Status: {result['status']}\n"
                f"Dataset ID: {result['dataset_id']}\n"
                f"Items returned: {len(result['items'])}"
            ),
            panel_title="Apify: Run Task + Dataset",
        )
    except Exception as e:
        return _error_result(e, "apify_run_task_and_get_dataset")


@tool
def apify_scrape_url(
    url: str,
    timeout_secs: int = DEFAULT_SCRAPE_TIMEOUT_SECS,
    crawler_type: str = "cheerio",
) -> Dict[str, Any]:
    """Scrape a single URL and return its content as markdown.

    Uses the Website Content Crawler Actor under the hood, pre-configured for
    fast single-page scraping. This is the simplest way to extract readable content
    from any web page.

    Args:
        url: The URL to scrape, e.g. "https://example.com".
        timeout_secs: Maximum time in seconds to wait for scraping to finish. Defaults to 120.
        crawler_type: Crawler engine to use. One of "cheerio" (fastest, no JS rendering,
            default), "playwright:adaptive" (fast, renders JS if present), or
            "playwright:firefox" (reliable, renders JS, best at avoiding blocking but slower).

    Returns:
        Dict with status and content containing the markdown content of the scraped page.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        content = client.scrape_url(url=url, timeout_secs=timeout_secs, crawler_type=crawler_type)
        return _success_result(
            text=content,
            panel_body=(
                f"[green]URL scraped successfully[/green]\nURL: {url}\nContent length: {len(content)} characters"
            ),
            panel_title="Apify: Scrape URL",
        )
    except Exception as e:
        return _error_result(e, "apify_scrape_url")


# Pre-built list of all core tools for convenient agent registration.
# Usage: Agent(tools=APIFY_CORE_TOOLS)
APIFY_CORE_TOOLS = [
    apify_run_actor,
    apify_get_dataset_items,
    apify_run_actor_and_get_dataset,
    apify_run_task,
    apify_run_task_and_get_dataset,
    apify_scrape_url,
]


# --- Social media tool constants ---

DEFAULT_SOCIAL_MEDIA_RESULTS_LIMIT = 20
INSTAGRAM_SCRAPER = "apify/instagram-scraper"
LINKEDIN_PROFILE_POSTS = "apimaestro/linkedin-profile-posts"
LINKEDIN_PROFILE_SEARCH = "harvestapi/linkedin-profile-search"
LINKEDIN_PROFILE_DETAIL = "apimaestro/linkedin-profile-detail"
TWITTER_SCRAPER = "apidojo/twitter-scraper-lite"
TIKTOK_SCRAPER = "clockworks/tiktok-scraper"
FACEBOOK_POSTS_SCRAPER = "apify/facebook-posts-scraper"
_MISSING_SEARCH_OR_URLS = "Provide at least one of 'search_query' or 'urls'."

VALID_INSTAGRAM_SEARCH_TYPES = ("hashtag", "user", "place")
VALID_INSTAGRAM_RESULTS_TYPES = ("posts", "comments", "details")
VALID_TWITTER_SORT_OPTIONS = ("Latest", "Top")
VALID_LINKEDIN_SCRAPER_MODES = ("Short", "Full")


# --- Social media helper functions ---


def _extract_linkedin_username(profile_url: str) -> str:
    """Extract a LinkedIn username from a profile URL, or return the value as-is if already a username."""
    parsed = urlparse(profile_url)
    if parsed.netloc and "linkedin.com" in parsed.netloc:
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(parts) >= 2 and parts[0] == "in":
            return parts[1]
    return profile_url


def _social_media_result(
    actor_name: str,
    client: ApifyToolClient,
    run_input: Dict[str, Any],
    actor_id: str,
    timeout_secs: int,
    results_limit: int,
) -> Dict[str, Any]:
    """Shared execution logic for social media scraper tools."""
    result = client.run_actor_and_get_dataset(
        actor_id=actor_id,
        run_input=run_input,
        timeout_secs=timeout_secs,
        dataset_items_limit=results_limit,
    )
    return _success_result(
        text=json.dumps(result, indent=2, default=str),
        panel_body=(
            f"[green]{actor_name} completed[/green]\n"
            f"Actor: {actor_id}\n"
            f"Run ID: {result['run_id']}\n"
            f"Status: {result['status']}\n"
            f"Items returned: {len(result['items'])}"
        ),
        panel_title=f"Apify: {actor_name}",
    )


# --- Social media tool functions ---


@tool
def apify_instagram_scraper(
    search_query: Optional[str] = None,
    urls: Optional[List[str]] = None,
    results_type: str = "posts",
    results_limit: int = DEFAULT_SOCIAL_MEDIA_RESULTS_LIMIT,
    search_type: str = "hashtag",
    search_limit: int = 10,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Scrape Instagram profiles, posts, reels, or hashtags.

    Provide either a search query to discover content or direct Instagram URLs to scrape.
    Supports searching by user profile, hashtag, or place.

    Args:
        search_query: Username, hashtag, or keyword to search for on Instagram.
            If the value looks like an Instagram URL it is treated as a direct URL instead.
        urls: One or more Instagram URLs to scrape directly (profiles, posts, reels, etc.).
        results_type: What to scrape from each page: "posts" (default), "comments", or
            "details" (profile metadata only).
        results_limit: Maximum number of items to return per URL or search hit. Defaults to 20.
        search_type: What kind of search to perform: "hashtag" (default), "user", or "place".
            Only used when search_query is a plain keyword (not a URL).
        search_limit: How many search results (hashtags, users, or places) to process.
            Defaults to 10. Each search hit then returns up to results_limit items.
        timeout_secs: Maximum time in seconds to wait for the Actor run. Defaults to 300.

    Returns:
        Dict with status and content containing scraped Instagram data items.
    """
    try:
        _check_dependency()
        if not search_query and not urls:
            raise ValueError(_MISSING_SEARCH_OR_URLS)
        if results_type not in VALID_INSTAGRAM_RESULTS_TYPES:
            raise ValueError(
                f"Invalid results_type '{results_type}'. Must be one of: {', '.join(VALID_INSTAGRAM_RESULTS_TYPES)}."
            )
        if search_type not in VALID_INSTAGRAM_SEARCH_TYPES:
            raise ValueError(
                f"Invalid search_type '{search_type}'. Must be one of: {', '.join(VALID_INSTAGRAM_SEARCH_TYPES)}."
            )

        client = ApifyToolClient()
        run_input: Dict[str, Any] = {
            "resultsType": results_type,
            "resultsLimit": results_limit,
        }

        if urls:
            run_input["directUrls"] = urls
        elif search_query and ("instagram.com" in search_query or search_query.startswith("http")):
            run_input["directUrls"] = [search_query]
        else:
            run_input["search"] = search_query
            run_input["searchType"] = search_type
            run_input["searchLimit"] = search_limit

        return _social_media_result(
            actor_name="Instagram Scraper",
            client=client,
            run_input=run_input,
            actor_id=INSTAGRAM_SCRAPER,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_instagram_scraper")


@tool
def apify_linkedin_profile_posts(
    profile_url: str,
    results_limit: int = DEFAULT_SOCIAL_MEDIA_RESULTS_LIMIT,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Scrape posts from a LinkedIn profile.

    Accepts a LinkedIn profile URL (e.g. "https://www.linkedin.com/in/username") or
    a bare username. Returns the most recent posts from that profile.

    Args:
        profile_url: LinkedIn profile URL or username to scrape posts from.
        results_limit: Maximum number of posts to return (1-100). Defaults to 20.
        timeout_secs: Maximum time in seconds to wait for the Actor run. Defaults to 300.

    Returns:
        Dict with status and content containing scraped LinkedIn post data.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        username = _extract_linkedin_username(profile_url)
        run_input: Dict[str, Any] = {
            "username": username,
            "limit": min(results_limit, 100),
        }
        return _social_media_result(
            actor_name="LinkedIn Profile Posts",
            client=client,
            run_input=run_input,
            actor_id=LINKEDIN_PROFILE_POSTS,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_linkedin_profile_posts")


@tool
def apify_linkedin_profile_search(
    search_query: str,
    results_limit: int = DEFAULT_SOCIAL_MEDIA_RESULTS_LIMIT,
    locations: Optional[List[str]] = None,
    current_job_titles: Optional[List[str]] = None,
    profile_scraper_mode: str = "Short",
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Search for LinkedIn profiles with filters.

    Find people on LinkedIn using a keyword query combined with optional filters
    for location and job title. Returns basic profile data in Short mode or
    full details (experience, education, skills) in Full mode.

    Args:
        search_query: Search keywords such as job titles, skills, or names
            (e.g. "software engineer", "marketing manager"). Supports LinkedIn
            search operators.
        results_limit: Maximum number of profiles to return. Defaults to 20.
        locations: Filter by locations (e.g. ["San Francisco", "New York"]).
            Use full names — LinkedIn may misinterpret abbreviations.
        current_job_titles: Filter by current job titles
            (e.g. ["Software Engineer", "Data Scientist"]).
        profile_scraper_mode: Amount of detail to return. "Short" (default)
            returns basic profile data from search results. "Full" opens each
            profile to scrape complete details including experience and education.
        timeout_secs: Maximum time in seconds to wait for the Actor run. Defaults to 300.

    Returns:
        Dict with status and content containing matched LinkedIn profile data.
    """
    try:
        _check_dependency()
        if profile_scraper_mode not in VALID_LINKEDIN_SCRAPER_MODES:
            raise ValueError(
                f"Invalid profile_scraper_mode '{profile_scraper_mode}'. "
                f"Must be one of: {', '.join(VALID_LINKEDIN_SCRAPER_MODES)}."
            )

        client = ApifyToolClient()
        run_input: Dict[str, Any] = {
            "searchQuery": search_query,
            "maxItems": results_limit,
            "profileScraperMode": profile_scraper_mode,
        }
        if locations is not None:
            run_input["locations"] = locations
        if current_job_titles is not None:
            run_input["currentJobTitles"] = current_job_titles

        return _social_media_result(
            actor_name="LinkedIn Profile Search",
            client=client,
            run_input=run_input,
            actor_id=LINKEDIN_PROFILE_SEARCH,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_linkedin_profile_search")


@tool
def apify_linkedin_profile_detail(
    profile_url: str,
    include_email: bool = False,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Get detailed information from a LinkedIn profile.

    Accepts a LinkedIn profile URL (e.g. "https://www.linkedin.com/in/username") or
    a bare username. Returns full profile details including work experience, education,
    skills, and more. No LinkedIn account or cookies required.

    Args:
        profile_url: LinkedIn profile URL or username to scrape.
        include_email: Whether to include the email address in results if publicly
            available. Defaults to False.
        timeout_secs: Maximum time in seconds to wait for the Actor run. Defaults to 300.

    Returns:
        Dict with status and content containing detailed LinkedIn profile data
        (work experience, education, certifications, location, and optionally email).
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        username = _extract_linkedin_username(profile_url)
        run_input: Dict[str, Any] = {
            "username": username,
            "includeEmail": include_email,
        }
        return _social_media_result(
            actor_name="LinkedIn Profile Detail",
            client=client,
            run_input=run_input,
            actor_id=LINKEDIN_PROFILE_DETAIL,
            timeout_secs=timeout_secs,
            results_limit=DEFAULT_DATASET_ITEMS_LIMIT,
        )
    except Exception as e:
        return _error_result(e, "apify_linkedin_profile_detail")


@tool
def apify_twitter_scraper(
    search_query: Optional[str] = None,
    urls: Optional[List[str]] = None,
    twitter_handles: Optional[List[str]] = None,
    results_limit: int = DEFAULT_SOCIAL_MEDIA_RESULTS_LIMIT,
    sort: str = "Latest",
    tweet_language: Optional[str] = None,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Scrape tweets from Twitter/X by search query, handles, or specific URLs.

    Supports Twitter advanced search syntax (e.g. "from:NASA", "#AI min_faves:100",
    "bitcoin min_faves:1000 min_retweets:100"). Provide at least one of search_query,
    urls, or twitter_handles.

    Args:
        search_query: Search query to find tweets. Supports Twitter advanced search
            operators like "from:user", "#hashtag", "min_faves:N", date ranges with
            "since:YYYY-MM-DD until:YYYY-MM-DD", and boolean operators.
        urls: Specific tweet, profile, search, or list URLs to scrape directly.
        twitter_handles: Twitter handles to scrape (without the @ symbol,
            e.g. ["NASA", "WHO"]).
        results_limit: Maximum number of tweets to return. Defaults to 20.
        sort: Sort order for search results: "Latest" (default, chronological) or
            "Top" (most popular/relevant).
        tweet_language: Restrict tweets to this language. Use an ISO 639-1 code
            (e.g. "en", "es", "de"). Defaults to all languages.
        timeout_secs: Maximum time in seconds to wait for the Actor run. Defaults to 300.

    Returns:
        Dict with status and content containing scraped tweet data.
    """
    try:
        _check_dependency()
        if not search_query and not urls and not twitter_handles:
            raise ValueError("Provide at least one of 'search_query', 'urls', or 'twitter_handles'.")
        if sort not in VALID_TWITTER_SORT_OPTIONS:
            raise ValueError(f"Invalid sort '{sort}'. Must be one of: {', '.join(VALID_TWITTER_SORT_OPTIONS)}.")

        client = ApifyToolClient()
        run_input: Dict[str, Any] = {
            "maxItems": results_limit,
            "sort": sort,
        }

        if search_query:
            run_input["searchTerms"] = [search_query]
        if urls:
            run_input["startUrls"] = [{"url": u} for u in urls]
        if twitter_handles:
            run_input["twitterHandles"] = twitter_handles
        if tweet_language is not None:
            run_input["tweetLanguage"] = tweet_language

        return _social_media_result(
            actor_name="Twitter Scraper",
            client=client,
            run_input=run_input,
            actor_id=TWITTER_SCRAPER,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_twitter_scraper")


@tool
def apify_tiktok_scraper(
    search_query: Optional[str] = None,
    hashtags: Optional[List[str]] = None,
    profiles: Optional[List[str]] = None,
    urls: Optional[List[str]] = None,
    results_limit: int = DEFAULT_SOCIAL_MEDIA_RESULTS_LIMIT,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Scrape TikTok videos by search, hashtag, profile, or direct post URL.

    Use the input that best matches your intent:
    - search_query: keyword search across videos and profiles
    - hashtags: scrape videos tagged with specific hashtags (e.g. ["fyp", "cooking"])
    - profiles: scrape videos from specific users (e.g. ["username1", "username2"])
    - urls: scrape specific TikTok post URLs

    Provide at least one of the above.

    Args:
        search_query: Keyword to search TikTok. Applies to both videos and profiles.
        hashtags: One or more TikTok hashtags (without #) to scrape videos from.
        profiles: One or more TikTok usernames to scrape videos from.
        urls: Specific TikTok post URLs to scrape.
        results_limit: Maximum number of videos per hashtag, profile, or search.
            Defaults to 20.
        timeout_secs: Maximum time in seconds to wait for the Actor run. Defaults to 300.

    Returns:
        Dict with status and content containing scraped TikTok video data.
    """
    try:
        _check_dependency()
        if not search_query and not hashtags and not profiles and not urls:
            raise ValueError("Provide at least one of 'search_query', 'hashtags', 'profiles', or 'urls'.")

        client = ApifyToolClient()
        run_input: Dict[str, Any] = {"resultsPerPage": results_limit}

        if search_query:
            run_input["searchQueries"] = [search_query]
        if hashtags:
            run_input["hashtags"] = hashtags
        if profiles:
            run_input["profiles"] = profiles
        if urls:
            run_input["postURLs"] = urls

        return _social_media_result(
            actor_name="TikTok Scraper",
            client=client,
            run_input=run_input,
            actor_id=TIKTOK_SCRAPER,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_tiktok_scraper")


@tool
def apify_facebook_posts_scraper(
    page_url: str,
    results_limit: int = DEFAULT_SOCIAL_MEDIA_RESULTS_LIMIT,
    only_posts_newer_than: Optional[str] = None,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Scrape posts from a Facebook page or profile.

    Provide a Facebook page or profile URL to scrape its posts. Optionally filter
    to only return recent posts.

    Args:
        page_url: Facebook page or profile URL to scrape posts from.
        results_limit: Maximum number of posts to return. Defaults to 20.
        only_posts_newer_than: Only return posts newer than this date. Use a
            natural-language date string like "2024-01-01", "1 week ago", or
            "3 months ago". Defaults to no date filter.
        timeout_secs: Maximum time in seconds to wait for the Actor run. Defaults to 300.

    Returns:
        Dict with status and content containing scraped Facebook post data.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        run_input: Dict[str, Any] = {
            "startUrls": [{"url": page_url}],
            "resultsLimit": results_limit,
        }
        if only_posts_newer_than is not None:
            run_input["onlyPostsNewerThan"] = only_posts_newer_than

        return _social_media_result(
            actor_name="Facebook Posts Scraper",
            client=client,
            run_input=run_input,
            actor_id=FACEBOOK_POSTS_SCRAPER,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_facebook_posts_scraper")


# Pre-built list of all social media tools for convenient agent registration.
# Usage: Agent(tools=APIFY_SOCIAL_TOOLS)
APIFY_SOCIAL_TOOLS = [
    apify_instagram_scraper,
    apify_linkedin_profile_posts,
    apify_linkedin_profile_search,
    apify_linkedin_profile_detail,
    apify_twitter_scraper,
    apify_tiktok_scraper,
    apify_facebook_posts_scraper,
]


# Pre-built list of all tools for convenient agent registration.
# WARNING: Registering APIFY_ALL_TOOLS will expose every Apify tool to the agent,
# which might overwhelm the LLM or cause the agent to make suboptimal tool choices.
# Prefer registering only the specific tools your use case actually needs.
# Usage: Agent(tools=APIFY_ALL_TOOLS)
APIFY_ALL_TOOLS = APIFY_CORE_TOOLS + APIFY_SOCIAL_TOOLS
