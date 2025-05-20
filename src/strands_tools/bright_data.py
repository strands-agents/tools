"""
Tool for web scraping, searching, and data extraction using Bright Data for Strands Agents

This module provides comprehensive web scraping and data extraction capabilities using
Bright Data as the backend. It handles all aspects of web scraping with a user-friendly
interface and proper error handling.

Key Features:
------------
1. Web Scraping:
   • scrape_as_markdown: Scrape webpage content and return as Markdown
   • get_screenshot: Take screenshots of webpages
   • search_engine: Perform search queries using various search engines
   • web_data_feed: Extract structured data from websites like LinkedIn, Amazon, Instagram, etc.

2. Advanced Capabilities:
   • Support for multiple search engines (Google, Bing, Yandex)
   • Advanced search parameters including language, location, device type
   • Extracting structured data from various websites
   • Screenshot generation for web pages

3. Error Handling:
   • Graceful API error handling
   • Clear error messages
   • Timeout management for web_data_feed

Usage Examples:
--------------
```python
from strands import Agent
from strands_tools import bright_data

agent = Agent(tools=[bright_data])

# Scrape webpage as markdown
agent.tool.bright_data(
    action="scrape_as_markdown",
    url="https://example.com"
)

# Search using Google
agent.tool.bright_data(
    action="search_engine",
    query="climate change solutions",
    engine="google",
    country_code="us",
    language="en"
)

# Extract product data from Amazon
agent.tool.bright_data(
    action="web_data_feed",
    source_type="amazon_product",
    url="https://www.amazon.com/product-url"
)
```
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from strands.types.tools import ToolResult, ToolResultContent, ToolUse

logger = logging.getLogger(__name__)

console = Console()

TOOL_SPEC = {
    "name": "bright_data",
    "description": (
        "Web scraping and data extraction tool powered by Bright Data.\n\n"
        "Features:\n"
        "1. Scrape webpage content as Markdown\n"
        "2. Take screenshots of webpages\n"
        "3. Search using Google, Bing, or Yandex with advanced parameters\n"
        "4. Extract structured data from various websites (LinkedIn, Amazon, Instagram, etc.)\n\n"
        "Actions:\n"
        "- scrape_as_markdown: Scrape webpage content as Markdown\n"
        "- get_screenshot: Take screenshot of a webpage\n"
        "- search_engine: Perform search queries using various search engines\n"
        "- web_data_feed: Extract structured data from websites\n\n"
        "Note: This tool requires a valid Bright Data API key."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "Action to perform (scrape_as_markdown, get_screenshot, search_engine, web_data_feed)"
                    ),
                    "enum": ["scrape_as_markdown", "get_screenshot", "search_engine", "web_data_feed"],
                },
                "url": {
                    "type": "string",
                    "description": "URL to scrape or extract data from (required for"
                    + "scrape_as_markdown, get_screenshot, web_data_feed)",
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to save the screenshot (required for get_screenshot)",
                },
                "zone": {
                    "type": "string",
                    "description": "Bright Data zone to use (optional, overrides default zone)",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (required for search_engine)",
                },
                "engine": {
                    "type": "string",
                    "description": "Search engine to use: google, bing, or yandex (default: google)",
                },
                "language": {
                    "type": "string",
                    "description": "Two-letter language code (hl parameter for Google)",
                },
                "country_code": {
                    "type": "string",
                    "description": "Two-letter country code (gl parameter for Google)",
                },
                "search_type": {
                    "type": "string",
                    "description": "Type of search (images, shopping, news, etc.)",
                },
                "start": {
                    "type": "integer",
                    "description": "Results pagination offset (0=first page, 10=second page)",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 10)",
                },
                "location": {
                    "type": "string",
                    "description": "Location for search results (uule parameter)",
                },
                "device": {
                    "type": "string",
                    "description": "Device type (mobile, ios, android, ipad, android_tablet)",
                },
                "return_json": {
                    "type": "boolean",
                    "description": "Return parsed JSON instead of HTML/Markdown",
                },
                "hotel_dates": {
                    "type": "string",
                    "description": "Check-in and check-out dates (format: YYYY-MM-DD,YYYY-MM-DD)",
                },
                "hotel_occupancy": {
                    "type": "integer",
                    "description": "Number of guests (1-4)",
                },
                "source_type": {
                    "type": "string",
                    "description": "Type of data source for web_data_feed"
                    + "(e.g., 'linkedin_person_profile', 'amazon_product')",
                },
                "num_of_reviews": {
                    "type": "integer",
                    "description": "Number of reviews to retrieve (only for facebook_company_reviews)",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum time in seconds to wait for data retrieval (default: 600)",
                },
                "polling_interval": {
                    "type": "integer",
                    "description": "Time in seconds between polling attempts (default: 1)",
                },
            },
            "required": ["action"],
            "allOf": [
                {
                    "if": {
                        "properties": {"action": {"enum": ["scrape_as_markdown", "get_screenshot", "web_data_feed"]}}
                    },
                    "then": {"required": ["url"]},
                },
                {"if": {"properties": {"action": {"enum": ["get_screenshot"]}}}, "then": {"required": ["output_path"]}},
                {"if": {"properties": {"action": {"enum": ["search_engine"]}}}, "then": {"required": ["query"]}},
                {
                    "if": {"properties": {"action": {"enum": ["web_data_feed"]}}},
                    "then": {"required": ["source_type", "url"]},
                },
            ],
        }
    },
}


class BrightDataClient:
    """Client for interacting with Bright Data API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        zone: str = "unlocker",
        verbose: bool = False,
    ) -> None:
        """
        Initialize with API token and default zone.

        Args:
            api_key (Optional[str]): Your Bright Data API token, defaults to BRIGHTDATA_API_KEY env var
            zone (str): Bright Data zone name
            verbose (bool): Print additional information about requests
        """
        self.api_key = api_key or os.environ.get("BRIGHTDATA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "BRIGHTDATA_API_KEY environment variable is required but not set. "
                "Please set it to your Bright Data API token or provide it as an argument."
            )

        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        self.zone = zone
        self.verbose = verbose
        self.endpoint = "https://api.brightdata.com/request"

    def make_request(self, payload: Dict) -> str:
        """
        Make a request to Bright Data API.

        Args:
            payload (Dict): Request payload

        Returns:
            str: Response text
        """
        if self.verbose:
            print(f"[Bright Data] Request: {payload['url']}")

        response = requests.post(self.endpoint, headers=self.headers, data=json.dumps(payload))

        if response.status_code != 200:
            raise Exception(f"Failed to scrape: {response.status_code} - {response.text}")

        return response.text

    def scrape_as_markdown(self, url: str, zone: Optional[str] = None) -> str:
        """
        Scrape a webpage and return content in Markdown format.

        Args:
            url (str): URL to scrape
            zone (Optional[str]): Override default zone

        Returns:
            str: Scraped content as Markdown
        """
        payload = {"url": url, "zone": zone or self.zone, "format": "raw", "data_format": "markdown"}

        return self.make_request(payload)

    def get_screenshot(self, url: str, output_path: str, zone: Optional[str] = None) -> str:
        """
        Take a screenshot of a webpage.

        Args:
            url (str): URL to screenshot
            output_path (str): Path to save the screenshot
            zone (Optional[str]): Override default zone

        Returns:
            str: Path to saved screenshot
        """
        payload = {"url": url, "zone": zone or self.zone, "format": "raw", "data_format": "screenshot"}

        response = requests.post(self.endpoint, headers=self.headers, data=json.dumps(payload))

        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")

        with open(output_path, "wb") as f:
            f.write(response.content)

        return output_path

    @staticmethod
    def encode_query(query: str) -> str:
        """URL encode a search query."""
        return quote(query)

    def search_engine(
        self,
        query: str,
        engine: str = "google",
        zone: Optional[str] = None,
        language: Optional[str] = None,
        country_code: Optional[str] = None,
        search_type: Optional[str] = None,
        start: Optional[int] = None,
        num_results: Optional[int] = 10,
        location: Optional[str] = None,
        device: Optional[str] = None,
        return_json: bool = False,
        hotel_dates: Optional[str] = None,
        hotel_occupancy: Optional[int] = None,
    ) -> str:
        """
        Search using Google, Bing, or Yandex with advanced parameters and return results in Markdown.

        Args:
            query (str): Search query
            engine (str): Search engine - 'google', 'bing', or 'yandex'
            zone (Optional[str]): Override default zone

            # Google SERP specific parameters
            language (Optional[str]): Two-letter language code (hl parameter)
            country_code (Optional[str]): Two-letter country code (gl parameter)
            search_type (Optional[str]): Type of search (images, shopping, news, etc.)
            start (Optional[int]): Results pagination offset (0=first page, 10=second page)
            num_results (Optional[int]): Number of results to return (default 10)
            location (Optional[str]): Location for search results (uule parameter)
            device (Optional[str]): Device type (mobile, ios, android, ipad, android_tablet)
            return_json (bool): Return parsed JSON instead of HTML/Markdown

            # Hotel search parameters
            hotel_dates (Optional[str]): Check-in and check-out dates (format: YYYY-MM-DD,YYYY-MM-DD)
            hotel_occupancy (Optional[int]): Number of guests (1-4)

        Returns:
            str: Search results as Markdown or JSON
        """
        encoded_query = self.encode_query(query)

        base_urls = {
            "google": f"https://www.google.com/search?q={encoded_query}",
            "bing": f"https://www.bing.com/search?q={encoded_query}",
            "yandex": f"https://yandex.com/search/?text={encoded_query}",
        }

        if engine not in base_urls:
            raise ValueError(f"Unsupported search engine: {engine}. Use 'google', 'bing', or 'yandex'")

        search_url = base_urls[engine]

        if engine == "google":
            params = []

            if language:
                params.append(f"hl={language}")

            if country_code:
                params.append(f"gl={country_code}")

            if search_type:
                if search_type == "jobs":
                    params.append("ibp=htl;jobs")
                else:
                    search_types = {"images": "isch", "shopping": "shop", "news": "nws"}
                    tbm_value = search_types.get(search_type, search_type)
                    params.append(f"tbm={tbm_value}")

            if start is not None:
                params.append(f"start={start}")

            if num_results:
                params.append(f"num={num_results}")

            if location:
                params.append(f"uule={self.encode_query(location)}")

            if device:
                device_value = "1"

                if device in ["ios", "iphone"]:
                    device_value = "ios"
                elif device == "ipad":
                    device_value = "ios_tablet"
                elif device == "android":
                    device_value = "android"
                elif device == "android_tablet":
                    device_value = "android_tablet"

                params.append(f"brd_mobile={device_value}")

            if return_json:
                params.append("brd_json=1")

            if hotel_dates:
                params.append(f"hotel_dates={self.encode_query(hotel_dates)}")

            if hotel_occupancy:
                params.append(f"hotel_occupancy={hotel_occupancy}")

            if params:
                search_url += "&" + "&".join(params)

        payload = {
            "url": search_url,
            "zone": zone or self.zone,
            "format": "raw",
            "data_format": "markdown" if not return_json else "raw",
        }

        return self.make_request(payload)

    def web_data_feed(
        self,
        source_type: str,
        url: str,
        num_of_reviews: Optional[int] = None,
        timeout: int = 600,
        polling_interval: int = 1,
    ) -> Dict:
        """
        Retrieve structured web data from various sources like LinkedIn, Amazon, Instagram, etc.

        Args:
            source_type (str): Type of data source (e.g., 'linkedin_person_profile', 'amazon_product')
            url (str): URL of the web resource to retrieve data from
            num_of_reviews (Optional[int]): Number of reviews to retrieve (only for facebook_company_reviews)
            timeout (int): Maximum time in seconds to wait for data retrieval
            polling_interval (int): Time in seconds between polling attempts

        Returns:
            Dict: Structured data from the requested source
        """
        datasets = {
            "amazon_product": "gd_l7q7dkf244hwjntr0",
            "amazon_product_reviews": "gd_le8e811kzy4ggddlq",
            "linkedin_person_profile": "gd_l1viktl72bvl7bjuj0",
            "linkedin_company_profile": "gd_l1vikfnt1wgvvqz95w",
            "zoominfo_company_profile": "gd_m0ci4a4ivx3j5l6nx",
            "instagram_profiles": "gd_l1vikfch901nx3by4",
            "instagram_posts": "gd_lk5ns7kz21pck8jpis",
            "instagram_reels": "gd_lyclm20il4r5helnj",
            "instagram_comments": "gd_ltppn085pokosxh13",
            "facebook_posts": "gd_lyclm1571iy3mv57zw",
            "facebook_marketplace_listings": "gd_lvt9iwuh6fbcwmx1a",
            "facebook_company_reviews": "gd_m0dtqpiu1mbcyc2g86",
            "x_posts": "gd_lwxkxvnf1cynvib9co",
            "zillow_properties_listing": "gd_lfqkr8wm13ixtbd8f5",
            "booking_hotel_listings": "gd_m5mbdl081229ln6t4a",
            "youtube_videos": "gd_m5mbdl081229ln6t4a",
        }

        if source_type not in datasets:
            valid_sources = ", ".join(datasets.keys())
            raise ValueError(f"Invalid source_type: {source_type}. Valid options are: {valid_sources}")

        dataset_id = datasets[source_type]

        request_data = {"url": url}
        if source_type == "facebook_company_reviews" and num_of_reviews is not None:
            request_data["num_of_reviews"] = str(num_of_reviews)

        trigger_response = requests.post(
            "https://api.brightdata.com/datasets/v3/trigger",
            params={"dataset_id": dataset_id, "include_errors": True},
            headers=self.headers,
            json=[request_data],
        )

        trigger_data = trigger_response.json()
        if not trigger_data.get("snapshot_id"):
            raise Exception("No snapshot ID returned from trigger request")

        snapshot_id = trigger_data["snapshot_id"]
        if self.verbose:
            print(f"[Bright Data] {source_type} triggered with snapshot ID: {snapshot_id}")

        attempts = 0
        max_attempts = timeout

        while attempts < max_attempts:
            try:
                snapshot_response = requests.get(
                    f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
                    params={"format": "json"},
                    headers=self.headers,
                )

                snapshot_data = snapshot_response.json()

                if isinstance(snapshot_data, dict) and snapshot_data.get("status") == "running":
                    if self.verbose:
                        print(
                            f"[Bright Data] Snapshot not ready, polling again (attempt {attempts + 1}/{max_attempts})"
                        )
                    attempts += 1
                    time.sleep(polling_interval)
                    continue

                if self.verbose:
                    print(f"[Bright Data] Data received after {attempts + 1} attempts")

                return snapshot_data

            except Exception as e:
                if self.verbose:
                    print(f"[Bright Data] Polling error: {e!s}")
                attempts += 1
                time.sleep(polling_interval)

        raise TimeoutError(f"Timeout after {max_attempts} seconds waiting for {source_type} data")


def bright_data(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Web scraping and data extraction tool powered by Bright Data.

    This tool provides a comprehensive interface for web scraping and data extraction using
    Bright Data, including scraping web pages as markdown, taking screenshots, performing
    search queries, and extracting structured data from various websites.

    Args:
        tool: ToolUse object containing the following input fields:
            - action: The action to perform (scrape_as_markdown, get_screenshot, search_engine, web_data_feed)
            - url: URL to scrape or extract data from (for scrape_as_markdown, get_screenshot, web_data_feed)
            - output_path: Path to save the screenshot (for get_screenshot)
            - zone: Override default Bright Data zone (optional)
            - query: Search query (for search_engine)
            - engine: Search engine to use (google, bing, yandex)
            - [Various search parameters for search_engine]
            - source_type: Type of data source for web_data_feed
            - [Various parameters for web_data_feed]
        **kwargs: Additional keyword arguments

    Returns:
        ToolResult containing status and response content
    """
    try:
        tool_input = tool.get("input", {})
        tool_use_id = tool.get("toolUseId", "default-id")

        if not tool_input.get("action"):
            raise ValueError("action parameter is required")

        client = BrightDataClient(verbose=True)

        action = tool_input["action"]

        if action == "scrape_as_markdown":
            if not tool_input.get("url"):
                raise ValueError("url is required for scrape_as_markdown action")

            content = client.scrape_as_markdown(tool_input["url"], tool_input.get("zone"))

            return ToolResult(toolUseId=tool_use_id, status="success", content=[ToolResultContent(text=content)])

        elif action == "get_screenshot":
            if not tool_input.get("url"):
                raise ValueError("url is required for get_screenshot action")
            if not tool_input.get("output_path"):
                raise ValueError("output_path is required for get_screenshot action")

            output_path = client.get_screenshot(tool_input["url"], tool_input["output_path"], tool_input.get("zone"))

            return ToolResult(
                toolUseId=tool_use_id,
                status="success",
                content=[ToolResultContent(text=f"Screenshot saved to {output_path}")],
            )

        elif action == "search_engine":
            if not tool_input.get("query"):
                raise ValueError("query is required for search_engine action")

            content = client.search_engine(
                query=tool_input["query"],
                engine=tool_input.get("engine", "google"),
                zone=tool_input.get("zone"),
                language=tool_input.get("language"),
                country_code=tool_input.get("country_code"),
                search_type=tool_input.get("search_type"),
                start=tool_input.get("start"),
                num_results=tool_input.get("num_results", 10),
                location=tool_input.get("location"),
                device=tool_input.get("device"),
                return_json=tool_input.get("return_json", False),
                hotel_dates=tool_input.get("hotel_dates"),
                hotel_occupancy=tool_input.get("hotel_occupancy"),
            )

            return ToolResult(toolUseId=tool_use_id, status="success", content=[ToolResultContent(text=content)])

        elif action == "web_data_feed":
            if not tool_input.get("url"):
                raise ValueError("url is required for web_data_feed action")
            if not tool_input.get("source_type"):
                raise ValueError("source_type is required for web_data_feed action")

            data = client.web_data_feed(
                source_type=tool_input["source_type"],
                url=tool_input["url"],
                num_of_reviews=tool_input.get("num_of_reviews"),
                timeout=tool_input.get("timeout", 600),
                polling_interval=tool_input.get("polling_interval", 1),
            )

            return ToolResult(
                toolUseId=tool_use_id, status="success", content=[ToolResultContent(text=json.dumps(data, indent=2))]
            )

        else:
            raise ValueError(f"Invalid action: {action}")

    except Exception as e:
        error_panel = Panel(
            Text(str(e), style="red"),
            title="❌ Bright Data Operation Error",
            border_style="red",
        )
        console.print(error_panel)
        return ToolResult(toolUseId=tool_use_id, status="error", content=[ToolResultContent(text=f"Error: {str(e)}")])
