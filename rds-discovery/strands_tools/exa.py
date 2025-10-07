"""
Exa Search and Contents tools for intelligent web search and content processing.

This module provides access to Exa's API, which offers neural search capabilities optimized for LLMs and AI agents.
The "auto" mode intelligently combines neural embeddings-based search with traditional keyword search for best results.

Key Features:
- Auto mode that intelligently selects the best search approach (default)
- Neural and keyword search capabilities
- Advanced content filtering and domain management
- Full page content extraction with summaries
- Support for general web search, company info, news, PDFs, GitHub repos, and more
- Date range filtering and domain management
- Live crawling with fallback options
- Subpage crawling and content extraction
- Structured output with JSON schemas

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import exa

agent = Agent(tools=[exa])

# Basic search (auto mode is default and recommended)
result = agent.tool.exa_search(query="Best project management tools", text=True)

# Get contents from specific URLs
result = agent.tool.exa_get_contents(urls=["https://strandsagents.com/"], text=True)
```

!!!!!!!!!!!!! IMPORTANT: !!!!!!!!!!!!!

Environment Variables:
- EXA_API_KEY: Your Exa API key (required)

You can get your Exa API key at https://dashboard.exa.ai/api-keys

!!!!!!!!!!!!! IMPORTANT: !!!!!!!!!!!!!

See the function docstrings for complete parameter documentation.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

import aiohttp
from rich.console import Console
from rich.panel import Panel
from strands import tool

logger = logging.getLogger(__name__)

# Exa API configuration
EXA_API_BASE_URL = "https://api.exa.ai"
EXA_SEARCH_ENDPOINT = "/search"
EXA_CONTENTS_ENDPOINT = "/contents"

# Initialize Rich console
console = Console()


def _get_api_key() -> str:
    """Get Exa API key from environment variables."""
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise ValueError(
            "EXA_API_KEY environment variable is required. Get your free API key at https://dashboard.exa.ai/api-keys"
        )
    return api_key


def format_search_response(data: Dict[str, Any]) -> Panel:
    """Format search response for rich display."""
    request_id = data.get("requestId", "Unknown request ID")
    results = data.get("results", [])
    search_type = data.get("searchType", "Unknown")
    resolved_search_type = data.get("resolvedSearchType", "Unknown")
    context = data.get("context")
    cost = data.get("costDollars", {})

    content = [f"Request ID: {request_id}"]
    content.append(f"Search Type: {search_type} (resolved: {resolved_search_type})")

    if cost:
        total_cost = cost.get("total", 0)
        content.append(f"Cost: ${total_cost:.4f}")

    if results:
        content.append(f"\nResults: {len(results)} found")
        content.append("-" * 50)

        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            author = result.get("author", "No author")
            published_date = result.get("publishedDate", "No date")
            text = result.get("text", "")
            summary = result.get("summary", "")

            content.append(f"\n[{i}] {title}")
            content.append(f"URL: {url}")
            content.append(f"Author: {author}")
            content.append(f"Published: {published_date}")

            if summary:
                content.append(f"Summary: {summary}")

            # Add full text content (length controlled by API maxCharacters parameter)
            if text:
                content.append(f"Content: {text.strip()}")

            # Add separator between results
            if i < len(results):
                content.append("")

    if context:
        content.append(f"\nFormatted Context Available: {len(context)} characters")

    return Panel("\n".join(content), title="[bold blue]Exa Search Results", border_style="blue")


def format_contents_response(data: Dict[str, Any]) -> Panel:
    """Format contents response for rich display."""
    request_id = data.get("requestId", "Unknown request ID")
    results = data.get("results", [])
    statuses = data.get("statuses", [])
    context = data.get("context")
    cost = data.get("costDollars", {})

    content = [f"Request ID: {request_id}"]

    if cost:
        total_cost = cost.get("total", 0)
        content.append(f"Cost: ${total_cost:.4f}")

    successful_results = len([s for s in statuses if s.get("status") == "success"])
    failed_results = len([s for s in statuses if s.get("status") == "error"])

    content.append(f"Successfully retrieved: {successful_results} URLs")
    if failed_results > 0:
        content.append(f"Failed retrievals: {failed_results} URLs")

    if results:
        content.append("-" * 50)

        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "Unknown URL")
            text = result.get("text", "")
            summary = result.get("summary", "")
            subpages = result.get("subpages", [])

            content.append(f"\n[{i}] {title}")
            content.append(f"URL: {url}")

            if summary:
                content.append(f"Summary: {summary}")

            if subpages:
                content.append(f"Subpages: {len(subpages)} found")

            # Add full text content (length controlled by API maxCharacters parameter)
            if text:
                content.append(f"Content: {text.strip()}")

            # Add separator between results
            if i < len(results):
                content.append("")

    if failed_results > 0:
        content.append("\nFailed retrievals:")
        for status in statuses:
            if status.get("status") == "error":
                error_url = status.get("id", "Unknown URL")
                error_info = status.get("error", {})
                error_tag = error_info.get("tag", "Unknown error")
                content.append(f"  • {error_url}: {error_tag}")

    if context:
        content.append(f"\nFormatted Context Available: {len(context)} characters")

    return Panel("\n".join(content), title="[bold blue]Exa Contents Results", border_style="blue")


# Exa Tools


@tool
async def exa_search(
    query: str,
    type: Optional[Literal["keyword", "neural", "fast", "auto"]] = "auto",
    category: Optional[
        Literal["company", "news", "pdf", "github", "personal site", "linkedin profile", "financial report"]
    ] = None,
    user_location: Optional[str] = None,
    num_results: Optional[int] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    start_crawl_date: Optional[str] = None,
    end_crawl_date: Optional[str] = None,
    start_published_date: Optional[str] = None,
    end_published_date: Optional[str] = None,
    include_text: Optional[List[str]] = None,
    exclude_text: Optional[List[str]] = None,
    context: Optional[Union[bool, Dict[str, Any]]] = None,
    moderation: Optional[bool] = None,
    # Contents options
    text: Optional[Union[bool, Dict[str, Any]]] = None,
    summary: Optional[Dict[str, Any]] = None,
    livecrawl: Optional[Literal["never", "fallback", "always", "preferred"]] = None,
    livecrawl_timeout: Optional[int] = None,
    subpages: Optional[int] = None,
    subpage_target: Optional[Union[str, List[str]]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Search the web intelligently using Exa's neural and keyword search capabilities.

    Exa provides advanced web search optimized for LLMs and AI agents. The "auto" mode (default)
    intelligently combines neural embeddings-based search with traditional keyword search to find
    the most relevant results for your query.

    Key Features:
    - Auto mode that intelligently selects the best search approach (default)
    - Neural search using embeddings for semantic understanding
    - Traditional keyword search for exact matches
    - Advanced filtering by domain, date, and content
    - Live crawling with fallback options
    - Rich content extraction with summaries

    Search Types:
    - auto: Intelligently combines neural and keyword approaches (recommended default)
    - neural: Uses embeddings-based model for semantic search
    - keyword: Google-like SERP search for exact matches
    - fast: Streamlined versions of neural and keyword models

    Categories (optional - general web search works best):
    - company: Focus on company websites and information when specifically needed
    - news: News articles and journalism
    - pdf: PDF documents
    - github: GitHub repositories and code
    - personal site: Personal websites and blogs
    - linkedin profile: LinkedIn profiles
    - financial report: Financial and earnings reports

    Args:
        query: The search query string. Examples: "Latest developments in artificial intelligence",
            "Best project management tools"
        type: Search type - "auto" (default, recommended), "neural", "keyword", or "fast"
        category: Optional data category - use sparingly as general search works best.
            Use "company" when specifically looking for company information
        user_location: Two-letter ISO country code (e.g., "US", "UK") for geo-localized results
        num_results: Number of results to return (max 100, default 10)
        include_domains: List of domains to include (e.g., ["github.com", "stackoverflow.com"])
        exclude_domains: List of domains to exclude from results
        start_crawl_date: Include links crawled after this date (ISO 8601 format)
        end_crawl_date: Include links crawled before this date (ISO 8601 format)
        start_published_date: Include links published after this date (ISO 8601 format)
        end_published_date: Include links published before this date (ISO 8601 format)
        include_text: List of strings that must be present in webpage text (max 1 string, up to 5 words)
        exclude_text: List of strings that must not be present in webpage text (max 1 string, up to 5 words)
        context: Format results for LLM context - True/False or object with maxCharacters
        moderation: Enable content moderation to filter unsafe content
        text: Include full page text - True/False or object with maxCharacters and includeHtmlTags.
            Use maxCharacters to control text length instead of relying on default limits
        summary: Generate summaries - object with query and optional schema for structured output
        livecrawl: Live crawling options - "never", "fallback", "always", "preferred"
        livecrawl_timeout: Timeout for live crawling in milliseconds (default 10000)
        subpages: Number of subpages to crawl from each result
        subpage_target: Keywords to find specific subpages (string or array)
        extras: Additional options - object with links (int) and imageLinks (int)

    Returns:
        Dict containing search results with title, URL, content, and metadata.

    Examples:
    --------
    # Basic search (auto mode is default and recommended)
    result = await exa_search(
        query="Best project management software",
        text=True
    )

    # Company-specific search
    result = await exa_search(
        query="Anthropic AI safety research",
        category="company",
        text=True
    )

    # Search with domain filtering and content options
    result = await exa_search(
        query="JavaScript frameworks comparison",
        include_domains=["github.com", "stackoverflow.com"],
        num_results=5,
        text={"maxCharacters": 500},
        summary={"query": "Key features and differences"}
    )

    # News search with date filtering
    result = await exa_search(
        query="AI regulation developments",
        category="news",
        start_published_date="2024-01-01T00:00:00.000Z",
        text=True
    )
    """
    try:
        # Validate parameters
        if not query or not query.strip():
            return {"status": "error", "content": [{"text": "Query parameter is required and cannot be empty"}]}

        # Validate num_results range
        if num_results is not None and not (1 <= num_results <= 100):
            return {"status": "error", "content": [{"text": "num_results must be between 1 and 100"}]}

        # Validate date formats
        if start_published_date is not None:
            try:
                from datetime import datetime

                datetime.fromisoformat(start_published_date.replace("Z", "+00:00"))
            except ValueError:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "Invalid date format for start_published_date. Use ISO 8601 format "
                            "(YYYY-MM-DDTHH:MM:SS.sssZ)"
                        }
                    ],
                }

        if end_published_date is not None:
            try:
                from datetime import datetime

                datetime.fromisoformat(end_published_date.replace("Z", "+00:00"))
            except ValueError:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "Invalid date format for end_published_date. Use ISO 8601 format "
                            "(YYYY-MM-DDTHH:MM:SS.sssZ)"
                        }
                    ],
                }

        # Get API key
        api_key = _get_api_key()

        # Build request payload
        payload = {
            "query": query,
            "type": type or "auto",
            "category": category,
            "userLocation": user_location,
            "numResults": num_results,
            "includeDomains": include_domains,
            "excludeDomains": exclude_domains,
            "startCrawlDate": start_crawl_date,
            "endCrawlDate": end_crawl_date,
            "startPublishedDate": start_published_date,
            "endPublishedDate": end_published_date,
            "includeText": include_text,
            "excludeText": exclude_text,
            "context": context,
            "moderation": moderation,
        }

        # Add contents options if any are specified
        contents = {}
        if text is not None:
            contents["text"] = text
        if summary is not None:
            contents["summary"] = summary
        if livecrawl is not None:
            contents["livecrawl"] = livecrawl
        if livecrawl_timeout is not None:
            contents["livecrawlTimeout"] = livecrawl_timeout
        if subpages is not None:
            contents["subpages"] = subpages
        if subpage_target is not None:
            contents["subpageTarget"] = subpage_target
        if extras is not None:
            contents["extras"] = extras

        if contents:
            payload["contents"] = contents

        # Make API request
        headers = {"x-api-key": api_key, "Content-Type": "application/json"}
        url = f"{EXA_API_BASE_URL}{EXA_SEARCH_ENDPOINT}"

        # Remove None values
        payload = {key: value for key, value in payload.items() if value is not None}

        logger.info(f"Making Exa search request for query: {query}")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                try:
                    data = await response.json()
                except Exception as e:
                    return {"status": "error", "content": [{"text": f"Failed to parse API response: {str(e)}"}]}

        # Format and display response
        panel = format_search_response(data)
        console.print(panel)

        return {"status": "success", "content": [{"text": str(data)}]}

    except asyncio.TimeoutError:
        return {"status": "error", "content": [{"text": "Request timeout. The API request took too long to complete."}]}
    except aiohttp.ClientError:
        return {"status": "error", "content": [{"text": "Connection error. Please check your internet connection."}]}
    except ValueError as e:
        return {"status": "error", "content": [{"text": str(e)}]}
    except Exception as e:
        logger.error(f"Unexpected error in exa_search: {str(e)}")
        return {"status": "error", "content": [{"text": f"Unexpected error: {str(e)}"}]}


@tool
async def exa_get_contents(
    urls: List[str],
    text: Optional[Union[bool, Dict[str, Any]]] = None,
    summary: Optional[Dict[str, Any]] = None,
    livecrawl: Optional[Literal["never", "fallback", "always", "preferred"]] = None,
    livecrawl_timeout: Optional[int] = None,
    subpages: Optional[int] = None,
    subpage_target: Optional[Union[str, List[str]]] = None,
    extras: Optional[Dict[str, Any]] = None,
    context: Optional[Union[bool, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Get full page contents, summaries, and metadata for a list of URLs using Exa.

    This endpoint provides instant results from Exa's cache with automatic live crawling as fallback
    for uncached pages. It's perfect for extracting content from specific URLs you already know about.

    Key Features:
    - Instant cached results with live crawling fallback
    - Full text extraction with optional character limits
    - AI-generated summaries with custom queries
    - Subpage crawling and discovery
    - Rich metadata extraction
    - Structured output options with JSON schemas

    Content Options:
    - Text: Full page content with optional HTML tags and character limits
    - Summary: AI-generated summaries with optional structured schemas
    - Subpages: Crawl and extract content from related pages
    - Extras: Additional links and images from pages

    Args:
        urls: List of URLs to retrieve content from. Can be any valid web URLs.
        text: Text extraction options:
            - True: Extract full text with default settings
            - False: Disable text extraction
            - Object: Advanced options with maxCharacters (controls text length) and includeHtmlTags
        summary: Summary generation options:
            - query: Custom query for summary generation
            - schema: JSON schema for structured summary output
        livecrawl: Live crawling behavior:
            - "never": Only use cached content
            - "fallback": Use cache first, crawl if not available (default)
            - "always": Always perform live crawl
            - "preferred": Try live crawl, fall back to cache if it fails
        livecrawl_timeout: Timeout for live crawling in milliseconds (default 10000)
        subpages: Number of subpages to crawl from each URL
        subpage_target: Keywords to find specific subpages (string or list)
        extras: Extra content options:
            - links: Number of links to extract from each page
            - imageLinks: Number of image URLs to extract
        context: Format results for LLM context - True/False or object with maxCharacters

    Returns:
        Dict containing content results with text, summaries, and metadata.

    Examples:
    --------
    # Simple content retrieval
    result = await exa_get_contents(
        urls=["https://strandsagents.com/"],
        text=True
    )

    # Advanced content extraction with summary
    result = await exa_get_contents(
        urls=["https://en.wikipedia.org/wiki/Artificial_intelligence"],
        text={"maxCharacters": 5000, "includeHtmlTags": False},
        summary={"query": "key points and conclusions"},
        subpages=2,
        extras={"links": 5, "imageLinks": 3}
    )

    # Structured content analysis
    result = await exa_get_contents(
        urls=["https://arxiv.org/abs/2303.08774"],
        summary={
            "query": "main findings and recommendations",
            "schema": {
                "type": "object",
                "properties": {
                    "main_findings": {"type": "string"},
                    "recommendations": {"type": "string"},
                    "conclusion": {"type": "string"}
                }
            }
        }
    )
    """
    try:
        # Validate parameters
        if not urls or len(urls) == 0:
            return {"status": "error", "content": [{"text": "At least one URL must be provided"}]}

        # Get API key
        api_key = _get_api_key()

        # Build request payload
        payload = {
            "urls": urls,
            "text": text,
            "summary": summary,
            "livecrawl": livecrawl,
            "livecrawlTimeout": livecrawl_timeout,
            "subpages": subpages,
            "subpageTarget": subpage_target,
            "extras": extras,
            "context": context,
        }

        # Make API request
        headers = {"x-api-key": api_key, "Content-Type": "application/json"}
        url = f"{EXA_API_BASE_URL}{EXA_CONTENTS_ENDPOINT}"

        # Remove None values
        payload = {key: value for key, value in payload.items() if value is not None}

        url_count = len(urls)
        logger.info(f"Making Exa contents request for {url_count} URLs")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                try:
                    data = await response.json()
                except Exception as e:
                    return {"status": "error", "content": [{"text": f"Failed to parse API response: {str(e)}"}]}

        # Format and display response
        panel = format_contents_response(data)
        console.print(panel)

        return {"status": "success", "content": [{"text": str(data)}]}

    except asyncio.TimeoutError:
        return {"status": "error", "content": [{"text": "Request timeout. The API request took too long to complete."}]}
    except aiohttp.ClientError:
        return {"status": "error", "content": [{"text": "Connection error. Please check your internet connection."}]}
    except ValueError as e:
        return {"status": "error", "content": [{"text": str(e)}]}
    except Exception as e:
        logger.error(f"Unexpected error in exa_get_contents: {str(e)}")
        return {"status": "error", "content": [{"text": f"Unexpected error: {str(e)}"}]}
