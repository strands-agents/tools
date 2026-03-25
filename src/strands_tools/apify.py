"""Apify platform tools for Strands Agents.

This module provides web scraping, data extraction, and automation capabilities
using the Apify platform. It lets you run any Actor, task, fetch dataset
results, scrape individual URLs, and perform specialized search and crawling.

Available Tools:
---------------
Core:
- apify_run_actor: Run any Apify Actor with custom input
- apify_get_dataset_items: Fetch items from an Apify dataset with pagination
- apify_run_actor_and_get_dataset: Run an Actor and fetch results in one step
- apify_run_task: Run a saved Actor task with optional input overrides
- apify_run_task_and_get_dataset: Run a task and fetch results in one step
- apify_scrape_url: Scrape a single URL and return content as Markdown

Search & Crawling:
- apify_google_search_scraper: Search Google and return structured results
- apify_google_places_scraper: Search Google Maps for businesses and places
- apify_youtube_scraper: Scrape YouTube videos, channels, or search results
- apify_website_content_crawler: Crawl a website and extract content from multiple pages
- apify_ecommerce_scraper: Scrape product data from e-commerce websites

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

Register all search & crawling tools:

```python
from strands import Agent
from strands_tools.apify import APIFY_SEARCH_TOOLS

agent = Agent(tools=APIFY_SEARCH_TOOLS)
```

Register all Apify tools (core + search):

```python
from strands import Agent
from strands_tools.apify import APIFY_ALL_TOOLS

agent = Agent(tools=APIFY_ALL_TOOLS)
```

Or pick individual tools for a smaller LLM tool surface:

```python
from strands import Agent
from strands_tools import apify

agent = Agent(tools=[
    apify.apify_scrape_url,
    apify.apify_run_actor,
    apify.apify_google_search_scraper,
])

# Scrape a single URL
content = agent.tool.apify_scrape_url(url="https://example.com")

# Run an Actor
result = agent.tool.apify_run_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://example.com"}]},
)

# Search Google
results = agent.tool.apify_google_search_scraper(
    search_query="best AI frameworks 2025",
    results_limit=10,
)
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


# --- Tool functions ---


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


APIFY_CORE_TOOLS = [
    apify_run_actor,
    apify_get_dataset_items,
    apify_run_actor_and_get_dataset,
    apify_run_task,
    apify_run_task_and_get_dataset,
    apify_scrape_url,
]


# --- Search & crawling tool constants ---

GOOGLE_SEARCH_SCRAPER_ID = "apify/google-search-scraper"
GOOGLE_PLACES_SCRAPER_ID = "compass/crawler-google-places"
YOUTUBE_SCRAPER_ID = "streamers/youtube-scraper"
ECOMMERCE_SCRAPER_ID = "apify/e-commerce-scraping-tool"
DEFAULT_SEARCH_RESULTS_LIMIT = 20


# --- Search & crawling helpers ---


def _search_crawl_result(
    actor_name: str,
    client: ApifyToolClient,
    run_input: Dict[str, Any],
    actor_id: str,
    timeout_secs: int,
    results_limit: int,
) -> Dict[str, Any]:
    """Run a search/crawling Actor and return formatted results."""
    result = client.run_actor_and_get_dataset(
        actor_id=actor_id,
        run_input=run_input,
        timeout_secs=timeout_secs,
        dataset_items_limit=results_limit,
    )
    return _success_result(
        text=json.dumps(result, indent=2, default=str),
        panel_body=(
            f"[green]{actor_name} completed[/green]\nRun ID: {result['run_id']}\nItems returned: {len(result['items'])}"
        ),
        panel_title=f"Apify: {actor_name}",
    )


# --- Search & crawling tool functions ---


@tool
def apify_google_search_scraper(
    search_query: str,
    results_limit: int = 10,
    country_code: Optional[str] = None,
    language_code: Optional[str] = None,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Search Google and return structured search results.

    Uses the Google Search Scraper Actor to perform a Google search and return
    organic results, ads, People Also Ask, and related queries in a structured format.

    Args:
        search_query: The search query string, e.g. "best AI frameworks 2025".
            Supports advanced Google operators like "site:example.com" or "AI OR ML".
        results_limit: Maximum number of results to return. Google returns ~10 results
            per page, so requesting more triggers additional page scraping. Defaults to 10.
        country_code: Two-letter country code for localized results, e.g. "us", "de".
        language_code: Two-letter language code for the interface, e.g. "en", "de".
        timeout_secs: Maximum time in seconds to wait for the run to finish. Defaults to 300.

    Returns:
        Dict with status and content containing structured Google search results including
        organic results, ads, and People Also Ask data.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        max_pages = max(1, (results_limit + 9) // 10)
        run_input: Dict[str, Any] = {
            "queries": search_query,
            "maxPagesPerQuery": max_pages,
        }
        if country_code is not None:
            run_input["countryCode"] = country_code
        if language_code is not None:
            run_input["languageCode"] = language_code
        return _search_crawl_result(
            actor_name="Google Search Scraper",
            client=client,
            run_input=run_input,
            actor_id=GOOGLE_SEARCH_SCRAPER_ID,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_google_search_scraper")


@tool
def apify_google_places_scraper(
    search_query: str,
    results_limit: int = DEFAULT_SEARCH_RESULTS_LIMIT,
    language: Optional[str] = None,
    include_reviews: bool = False,
    max_reviews: int = 5,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Search Google Maps for businesses and places, optionally including reviews.

    Uses the Google Maps Scraper Actor to find places matching a search query
    and return structured data including name, address, rating, phone, and website.

    Args:
        search_query: Search query for Google Maps, e.g. "restaurants in Prague".
        results_limit: Maximum number of places to return. Defaults to 20.
        language: Language for results, e.g. "en", "de". Defaults to English.
        include_reviews: Whether to include user reviews for each place. Defaults to False.
        max_reviews: Maximum reviews per place when include_reviews is True. Defaults to 5.
        timeout_secs: Maximum time in seconds to wait for the run to finish. Defaults to 300.

    Returns:
        Dict with status and content containing structured Google Maps place data.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        run_input: Dict[str, Any] = {
            "searchStringsArray": [search_query],
            "maxCrawledPlacesPerSearch": results_limit,
            "maxReviews": max_reviews if include_reviews else 0,
        }
        if language is not None:
            run_input["language"] = language
        return _search_crawl_result(
            actor_name="Google Places Scraper",
            client=client,
            run_input=run_input,
            actor_id=GOOGLE_PLACES_SCRAPER_ID,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_google_places_scraper")


@tool
def apify_youtube_scraper(
    search_query: Optional[str] = None,
    urls: Optional[List[str]] = None,
    results_limit: int = DEFAULT_SEARCH_RESULTS_LIMIT,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Scrape YouTube videos, channels, or search results.

    Uses the YouTube Scraper Actor to search YouTube or scrape specific video/channel
    URLs. Provide either a search query, specific URLs, or both.

    Args:
        search_query: YouTube search query, e.g. "python tutorial".
        urls: Specific YouTube video or channel URLs to scrape.
        results_limit: Maximum number of results to return. Defaults to 20.
        timeout_secs: Maximum time in seconds to wait for the run to finish. Defaults to 300.

    Returns:
        Dict with status and content containing structured YouTube video/channel data.
    """
    try:
        _check_dependency()
        if not search_query and not urls:
            raise ValueError("At least one of 'search_query' or 'urls' must be provided.")
        client = ApifyToolClient()
        run_input: Dict[str, Any] = {
            "maxResults": results_limit,
        }
        if search_query is not None:
            run_input["searchQueries"] = [search_query]
        if urls is not None:
            run_input["startUrls"] = [{"url": u} for u in urls]
        return _search_crawl_result(
            actor_name="YouTube Scraper",
            client=client,
            run_input=run_input,
            actor_id=YOUTUBE_SCRAPER_ID,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_youtube_scraper")


@tool
def apify_website_content_crawler(
    start_url: str,
    max_pages: int = 10,
    max_depth: int = 2,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Crawl a website and extract content from multiple pages.

    Uses the Website Content Crawler Actor to perform a multi-page crawl starting
    from the given URL. Returns page content as markdown. This is the extended
    multi-page version — distinct from apify_scrape_url which scrapes a single page.

    Args:
        start_url: The starting URL to crawl, e.g. "https://docs.example.com".
        max_pages: Maximum number of pages to crawl. Defaults to 10.
        max_depth: Maximum crawl depth from the start URL. Defaults to 2.
        timeout_secs: Maximum time in seconds to wait for the run to finish. Defaults to 300.

    Returns:
        Dict with status and content containing crawled page data with markdown content.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        client._validate_url(start_url)
        run_input: Dict[str, Any] = {
            "startUrls": [{"url": start_url}],
            "maxCrawlPages": max_pages,
            "maxCrawlDepth": max_depth,
            "proxyConfiguration": {"useApifyProxy": True},
        }
        return _search_crawl_result(
            actor_name="Website Content Crawler",
            client=client,
            run_input=run_input,
            actor_id=WEBSITE_CONTENT_CRAWLER,
            timeout_secs=timeout_secs,
            results_limit=max_pages,
        )
    except Exception as e:
        return _error_result(e, "apify_website_content_crawler")


VALID_ECOMMERCE_URL_TYPES = ("product", "listing")


@tool
def apify_ecommerce_scraper(
    url: str,
    url_type: str = "product",
    results_limit: int = DEFAULT_SEARCH_RESULTS_LIMIT,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> Dict[str, Any]:
    """Scrape product data from e-commerce websites.

    Uses the E-commerce Scraping Tool Actor to extract structured product data
    (title, price, description, images, etc.) from supported e-commerce platforms
    including Amazon, eBay, Walmart, and others. The Actor auto-detects the site.

    Args:
        url: The URL to scrape.
        url_type: Type of URL being scraped. Use "product" (default) for a direct product
            detail page, or "listing" for a category page or search results page containing
            multiple products.
        results_limit: Maximum number of products to return. Defaults to 20.
        timeout_secs: Maximum time in seconds to wait for the run to finish. Defaults to 300.

    Returns:
        Dict with status and content containing structured product data.
    """
    try:
        _check_dependency()
        client = ApifyToolClient()
        client._validate_url(url)
        if url_type not in VALID_ECOMMERCE_URL_TYPES:
            raise ValueError(f"Invalid url_type '{url_type}'. Must be one of: {', '.join(VALID_ECOMMERCE_URL_TYPES)}.")
        url_field = "listingUrls" if url_type == "listing" else "detailsUrls"
        run_input: Dict[str, Any] = {
            url_field: [{"url": url}],
            "maxProductResults": results_limit,
        }
        return _search_crawl_result(
            actor_name="E-commerce Scraper",
            client=client,
            run_input=run_input,
            actor_id=ECOMMERCE_SCRAPER_ID,
            timeout_secs=timeout_secs,
            results_limit=results_limit,
        )
    except Exception as e:
        return _error_result(e, "apify_ecommerce_scraper")


APIFY_SEARCH_TOOLS = [
    apify_google_search_scraper,
    apify_google_places_scraper,
    apify_youtube_scraper,
    apify_website_content_crawler,
    apify_ecommerce_scraper,
]

APIFY_ALL_TOOLS = APIFY_CORE_TOOLS + APIFY_SEARCH_TOOLS
