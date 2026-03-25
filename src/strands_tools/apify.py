"""Shared base for Apify platform tools.

This module provides the shared infrastructure used by all Apify tool modules
(e.g. apify_core, apify_social). It contains the API client, error handling,
response helpers, and constants. It does NOT contain any @tool functions itself.

Tool modules import from here:
    from strands_tools.apify import ApifyToolClient, _check_dependency, ...

Setup Requirements:
------------------
1. Create an Apify account at https://apify.com
2. Obtain your API token: Apify Console > Settings > API & Integrations > Personal API tokens
3. Install the optional dependency: pip install strands-agents-tools[apify]
4. Set the environment variable:
   APIFY_API_TOKEN=your_api_token_here
"""

import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from rich.panel import Panel
from rich.text import Text

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
