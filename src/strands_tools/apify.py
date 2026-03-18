"""Apify platform integration tool for Strands Agents.

Provides capabilities to run Apify Actors, retrieve Datasets, and scrape URLs
using the Apify platform programmatically.

Available tools:
- apify_run_actor: Run any Apify Actor by ID with arbitrary input
- apify_get_dataset_items: Fetch items from an Apify Dataset
- apify_run_actor_and_get_dataset: Run an Actor and fetch its Dataset results in one step
- apify_scrape_url: Scrape a URL and return its content as markdown

Setup Requirements:
------------------
1. Create an Apify account at https://apify.com
2. Obtain your API token: Apify Console → Settings → API & Integrations → Personal API tokens
3. Install the optional dependency: pip install -e ".[apify]"
4. Set the environment variable:
   APIFY_API_TOKEN=your_api_token_here

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import apify

agent = Agent(tools=[
    apify.apify_run_actor,
    apify.apify_get_dataset_items,
    apify.apify_run_actor_and_get_dataset,
    apify.apify_scrape_url,
])

# Run an Actor
result = agent.tool.apify_run_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://example.com"}]},
)

# Scrape a single URL
content = agent.tool.apify_scrape_url(url="https://example.com")
```

!!!!!!!!!!!!! IMPORTANT: !!!!!!!!!!!!!

Environment Variables:
- APIFY_API_TOKEN: Your Apify API token (required)
  Obtain from https://console.apify.com/account/integrations

Example .env configuration:
    APIFY_API_TOKEN=apify_api_1a2B3cD4eF5gH6iJ7kL8m

!!!!!!!!!!!!! IMPORTANT: !!!!!!!!!!!!!

See the function docstrings for complete parameter documentation.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from rich.panel import Panel
from rich.text import Text
from strands import tool

from strands_tools.utils import console_util

logger = logging.getLogger(__name__)
console = console_util.create()

try:
    from apify_client import ApifyClient

    HAS_APIFY_CLIENT = True
except ImportError:
    HAS_APIFY_CLIENT = False

WEBSITE_CONTENT_CRAWLER = "apify/website-content-crawler"
TRACKING_HEADER = {"x-apify-integration-platform": "strands-agents"}


def _check_dependency() -> None:
    """Raise ImportError if apify-client is not installed."""
    if not HAS_APIFY_CLIENT:
        raise ImportError("apify-client package is required. Install with: pip install strands-agents-tools[apify]")


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

    def run_actor(
        self,
        actor_id: str,
        run_input: Optional[Dict[str, Any]] = None,
        timeout_secs: int = 300,
        memory_mbytes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run an Apify Actor synchronously and return run metadata."""
        call_kwargs: Dict[str, Any] = {
            "run_input": run_input or {},
            "timeout_secs": timeout_secs,
        }
        if memory_mbytes is not None:
            call_kwargs["memory_mbytes"] = memory_mbytes

        actor_run = self.client.actor(actor_id).call(**call_kwargs)

        status = actor_run.get("status", "UNKNOWN")
        if status not in ("SUCCEEDED",):
            raise RuntimeError(f"Actor {actor_id} finished with status {status}. Run ID: {actor_run.get('id', 'N/A')}")

        return {
            "run_id": actor_run.get("id"),
            "status": status,
            "dataset_id": actor_run.get("defaultDatasetId"),
            "started_at": actor_run.get("startedAt"),
            "finished_at": actor_run.get("finishedAt"),
        }

    def get_dataset_items(
        self,
        dataset_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Fetch items from an Apify Dataset."""
        result = self.client.dataset(dataset_id).list_items(limit=limit, offset=offset)
        return list(result.items)

    def run_actor_and_get_dataset(
        self,
        actor_id: str,
        run_input: Optional[Dict[str, Any]] = None,
        timeout_secs: int = 300,
        memory_mbytes: Optional[int] = None,
        dataset_items_limit: int = 100,
    ) -> Dict[str, Any]:
        """Run an Actor synchronously, then fetch its default Dataset items."""
        run_metadata = self.run_actor(
            actor_id=actor_id,
            run_input=run_input,
            timeout_secs=timeout_secs,
            memory_mbytes=memory_mbytes,
        )
        dataset_id = run_metadata["dataset_id"]
        items = self.get_dataset_items(dataset_id=dataset_id, limit=dataset_items_limit)
        return {**run_metadata, "items": items}

    def scrape_url(self, url: str, timeout_secs: int = 120) -> str:
        """Scrape a single URL using Website Content Crawler and return markdown."""
        run_input = {
            "startUrls": [{"url": url}],
            "maxCrawlPages": 1,
        }
        actor_run = self.client.actor(WEBSITE_CONTENT_CRAWLER).call(
            run_input=run_input,
            timeout_secs=timeout_secs,
        )

        status = actor_run.get("status", "UNKNOWN")
        if status not in ("SUCCEEDED",):
            raise RuntimeError(
                f"Website Content Crawler finished with status {status}. Run ID: {actor_run.get('id', 'N/A')}"
            )

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
    timeout_secs: int = 300,
    memory_mbytes: Optional[int] = None,
) -> str:
    """Run any Apify Actor by its ID or name and return the run metadata as JSON.

    Executes the Actor synchronously - blocks until the Actor Run finishes or the timeout
    is reached. Use this when you need to run a specific Actor and then inspect or process
    the results separately.

    Common Actors:
    - "apify/website-content-crawler" - scrape websites and extract content
    - "apify/web-scraper" - general-purpose web scraper
    - "apify/google-search-scraper" - scrape Google search results

    Args:
        actor_id: Actor identifier, e.g. "apify/website-content-crawler" or "username/actor-name".
        run_input: JSON-serializable input for the Actor. Each Actor defines its own input schema.
        timeout_secs: Maximum time in seconds to wait for the Actor Run to finish. Defaults to 300.
        memory_mbytes: Memory allocation in MB for the Actor Run. Uses Actor default if not set.

    Returns:
        JSON string with run metadata: run_id, status, dataset_id, started_at, finished_at.
    """
    _check_dependency()
    try:
        client = ApifyToolClient()
        result = client.run_actor(
            actor_id=actor_id,
            run_input=run_input,
            timeout_secs=timeout_secs,
            memory_mbytes=memory_mbytes,
        )
        panel = Panel(
            f"[green]Actor Run completed[/green]\n"
            f"Actor: {actor_id}\n"
            f"Run ID: {result['run_id']}\n"
            f"Status: {result['status']}\n"
            f"Dataset ID: {result['dataset_id']}",
            title="[bold cyan]Apify: Run Actor[/bold cyan]",
            border_style="green",
        )
        console.print(panel)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        error_panel = Panel(
            Text(str(e), style="red"),
            title="[bold red]Apify Error[/bold red]",
            border_style="red",
        )
        console.print(error_panel)
        raise


@tool
def apify_get_dataset_items(
    dataset_id: str,
    limit: int = 100,
    offset: int = 0,
) -> str:
    """Fetch items from an existing Apify Dataset and return them as JSON.

    Use this after running an Actor to retrieve the structured results from its
    default Dataset, or to access any Dataset by ID.

    Args:
        dataset_id: The Apify Dataset ID to fetch items from.
        limit: Maximum number of items to return. Defaults to 100.
        offset: Number of items to skip for pagination. Defaults to 0.

    Returns:
        JSON string containing an array of Dataset items.
    """
    _check_dependency()
    try:
        client = ApifyToolClient()
        items = client.get_dataset_items(dataset_id=dataset_id, limit=limit, offset=offset)
        panel = Panel(
            f"[green]Dataset items retrieved[/green]\nDataset ID: {dataset_id}\nItems returned: {len(items)}",
            title="[bold cyan]Apify: Dataset Items[/bold cyan]",
            border_style="green",
        )
        console.print(panel)
        return json.dumps(items, indent=2, default=str)
    except Exception as e:
        error_panel = Panel(
            Text(str(e), style="red"),
            title="[bold red]Apify Error[/bold red]",
            border_style="red",
        )
        console.print(error_panel)
        raise


@tool
def apify_run_actor_and_get_dataset(
    actor_id: str,
    run_input: Optional[Dict[str, Any]] = None,
    timeout_secs: int = 300,
    memory_mbytes: Optional[int] = None,
    dataset_items_limit: int = 100,
) -> str:
    """Run an Apify Actor and fetch its Dataset results in one step.

    Convenience tool that combines running an Actor and fetching its default Dataset
    items into a single call. Use this when you want both the run metadata and the
    result data without making two separate tool calls.

    Args:
        actor_id: Actor identifier, e.g. "apify/website-content-crawler" or "username/actor-name".
        run_input: JSON-serializable input for the Actor.
        timeout_secs: Maximum time in seconds to wait for the Actor Run to finish. Defaults to 300.
        memory_mbytes: Memory allocation in MB for the Actor Run.
        dataset_items_limit: Maximum number of Dataset items to return. Defaults to 100.

    Returns:
        JSON string with run metadata (run_id, status, dataset_id, started_at, finished_at)
        plus an "items" array containing the Dataset results.
    """
    _check_dependency()
    try:
        client = ApifyToolClient()
        result = client.run_actor_and_get_dataset(
            actor_id=actor_id,
            run_input=run_input,
            timeout_secs=timeout_secs,
            memory_mbytes=memory_mbytes,
            dataset_items_limit=dataset_items_limit,
        )
        panel = Panel(
            f"[green]Actor Run completed with dataset[/green]\n"
            f"Actor: {actor_id}\n"
            f"Run ID: {result['run_id']}\n"
            f"Status: {result['status']}\n"
            f"Dataset ID: {result['dataset_id']}\n"
            f"Items returned: {len(result['items'])}",
            title="[bold cyan]Apify: Run Actor + Dataset[/bold cyan]",
            border_style="green",
        )
        console.print(panel)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        error_panel = Panel(
            Text(str(e), style="red"),
            title="[bold red]Apify Error[/bold red]",
            border_style="red",
        )
        console.print(error_panel)
        raise


@tool
def apify_scrape_url(
    url: str,
    timeout_secs: int = 120,
) -> str:
    """Scrape a single URL and return its content as markdown.

    Uses the Apify Website Content Crawler Actor under the hood, pre-configured for
    fast single-page scraping. This is the simplest way to extract readable content
    from any web page.

    Args:
        url: The URL to scrape, e.g. "https://example.com".
        timeout_secs: Maximum time in seconds to wait for scraping to finish. Defaults to 120.

    Returns:
        Markdown content of the scraped page as a plain string.
    """
    _check_dependency()
    try:
        client = ApifyToolClient()
        content = client.scrape_url(url=url, timeout_secs=timeout_secs)
        panel = Panel(
            f"[green]URL scraped successfully[/green]\nURL: {url}\nContent length: {len(content)} characters",
            title="[bold cyan]Apify: Scrape URL[/bold cyan]",
            border_style="green",
        )
        console.print(panel)
        return content
    except Exception as e:
        error_panel = Panel(
            Text(str(e), style="red"),
            title="[bold red]Apify Error[/bold red]",
            border_style="red",
        )
        console.print(error_panel)
        raise
