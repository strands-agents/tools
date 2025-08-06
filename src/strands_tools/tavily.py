"""
Tavily Search, Extract, Crawl, and Map tools for real-time web search and content processing.

This module provides access to Tavily's API, which is specifically optimized for LLMs and AI agents.
Tavily takes care of searching, scraping, filtering and extracting the most relevant information from online sources.

Key Features:
- Real-time web search optimized for AI agents
- Advanced content filtering and ranking
- Web page content extraction from URLs
- Website crawling with intelligent discovery
- Website structure mapping and discovery
- Support for news and general search topics
- Image search capabilities
- Domain filtering (include/exclude)
- Multiple search depths (basic/advanced)
- Country-specific search boosting
- Date range filtering
Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import tavily

agent = Agent(tools=[tavily])

# Basic search
result = agent.tool.tavily_search(query="What is artificial intelligence?")

# Extract content from URLs
result = agent.tool.tavily_extract(urls=["www.tavily.com"])

# Crawl website starting from base URL
result = agent.tool.tavily_crawl(url="www.tavily.com")

# Map website structure
result = agent.tool.tavily_map(url="www.tavily.com")
```

!!!!!!!!!!!!! IMPORTANT: !!!!!!!!!!!!!

Environment Variables:
- TAVILY_API_KEY: Your Tavily API key (required)

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

# Tavily API configuration
TAVILY_API_BASE_URL = "https://api.tavily.com"
TAVILY_SEARCH_ENDPOINT = "/search"
TAVILY_EXTRACT_ENDPOINT = "/extract"
TAVILY_CRAWL_ENDPOINT = "/crawl"
TAVILY_MAP_ENDPOINT = "/map"

# Initialize Rich console
console = Console()


def _get_api_key() -> str:
    """Get Tavily API key from environment variables."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable is required. " "Get your free API key at https://app.tavily.com"
        )
    return api_key


def format_search_response(data: Dict[str, Any]) -> Panel:
    """Format search response for rich display."""
    query = data.get("query", "Unknown query")
    results = data.get("results", [])
    answer = data.get("answer")
    images = data.get("images", None)

    content = [f"Query: {query}"]

    if answer:
        content.append(f"\nAnswer: {answer}")

    if images:
        content.append(f"\nImages: {len(images)} found")

    if results:
        content.append(f"\nResults: {len(results)} found")
        content.append("-" * 50)

        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            result_content = result.get("content", "No content")
            score = result.get("score", "No score")
            raw_content = result.get("raw_content", None)
            favicon = result.get("favicon", None)

            content.append(f"\n[{i}] {title}")
            content.append(f"URL: {url}")
            content.append(f"Score: {score}")
            content.append(f"Content: {result_content}")

            # Limit raw content to a preview
            if raw_content:
                preview_length = 150
                if len(raw_content) > preview_length:
                    raw_preview = raw_content[:preview_length].strip() + "..."
                else:
                    raw_preview = raw_content.strip()
                content.append(f"Raw Content: {raw_preview}")

            if favicon:
                content.append(f"Favicon: {favicon}")

            # Add separator between results
            if i < len(results):
                content.append("")

    return Panel("\n".join(content), title="[bold cyan]Tavily Search Results", border_style="cyan")


def format_extract_response(data: Dict[str, Any]) -> Panel:
    """Format extraction response for rich display."""
    results = data.get("results", [])
    failed_results = data.get("failed_results", [])

    content = [f"Successfully extracted: {len(results)} URLs"]

    if results:
        content.append("-" * 50)

        for i, result in enumerate(results, 1):
            url = result.get("url", "Unknown URL")
            raw_content = result.get("raw_content", None)
            images = result.get("images", None)
            favicon = result.get("favicon", None)

            content.append(f"\n[{i}] {url}")

            if raw_content:
                preview_length = 150
                if len(raw_content) > preview_length:
                    raw_preview = raw_content[:preview_length].strip() + "..."
                else:
                    raw_preview = raw_content.strip()
                content.append(f"Content: {raw_preview}")

            if images:
                content.append(f"Images: {len(images)} found")

            if favicon:
                content.append(f"Favicon: {favicon}")

            # Add separator between results
            if i < len(results):
                content.append("")

    if failed_results:
        content.append(f"\nFailed extractions: {len(failed_results)}")
        content.append("-" * 30)

        for i, failed in enumerate(failed_results, 1):
            url = failed.get("url", "Unknown URL")
            error = failed.get("error", "Unknown error")
            content.append(f"\n[{i}] {url}")
            content.append(f"Error: {error}")

            # Add separator between failed results
            if i < len(failed_results):
                content.append("")

    return Panel("\n".join(content), title="[bold cyan]Tavily Extract Results", border_style="cyan")


def format_crawl_response(data: Dict[str, Any]) -> Panel:
    """Format crawl response for rich display."""
    base_url = data.get("base_url", "Unknown base URL")
    results = data.get("results", [])
    response_time = data.get("response_time", "Unknown")

    content = [f"Base URL: {base_url}"]
    content.append(f"Response Time: {response_time}s")

    if results:
        content.append(f"\nPages Crawled: {len(results)}")
        content.append("-" * 50)

        for i, result in enumerate(results, 1):
            url = result.get("url", "No URL")
            raw_content = result.get("raw_content", "")
            favicon = result.get("favicon", "")

            content.append(f"\n[{i}] {url}")

            if favicon:
                content.append(f"Favicon: {favicon}")

            # Limit content to a preview
            if raw_content:
                preview_length = 100
                if len(raw_content) > preview_length:
                    content_preview = raw_content[:preview_length].strip() + "..."
                else:
                    content_preview = raw_content.strip()
                content.append(f"Content Preview: {content_preview}")

            # Add separator between results
            if i < len(results):
                content.append("")
    else:
        content.append("\nNo pages found during crawl.")

    return Panel("\n".join(content), title="[bold cyan]]Tavily Crawl Results", border_style="cyan")


def format_map_response(data: Dict[str, Any]) -> Panel:
    """Format map response for rich display."""
    base_url = data.get("base_url", "Unknown base URL")
    results = data.get("results", [])
    response_time = data.get("response_time", "Unknown")

    content = [f"Base URL: {base_url}"]
    content.append(f"Response Time: {response_time}s")

    if results:
        content.append(f"\nURLs Discovered: {len(results)}")
        content.append("-" * 50)

        for i, url in enumerate(results, 1):
            content.append(f"[{i}] {url}")

            # Add separator every 10 URLs for readability
            if i % 10 == 0 and i < len(results):
                content.append("")
    else:
        content.append("\nNo URLs found during mapping.")

    return Panel("\n".join(content), title="[bold cyan]Tavily Map Results", border_style="cyan")


# Tavily Tools


@tool
async def tavily_search(
    query: str,
    search_depth: Optional[Literal["basic", "advanced"]] = None,
    topic: Optional[Literal["general", "news"]] = None,
    max_results: Optional[int] = None,
    auto_parameters: Optional[bool] = None,
    chunks_per_source: Optional[int] = None,
    time_range: Optional[Literal["day", "week", "month", "year", "d", "w", "m", "y"]] = None,
    days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_answer: Optional[Union[bool, Literal["basic", "advanced"]]] = None,
    include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]] = None,
    include_images: Optional[bool] = None,
    include_image_descriptions: Optional[bool] = None,
    include_favicon: Optional[bool] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    country: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search the web for real-time information using Tavily's AI-optimized search engine.

    Tavily is a search engine specifically optimized for LLMs and AI agents. It handles all the
    complexity of searching, scraping, filtering, and extracting the most relevant information
    from online sources in a single API call.

    Key Features:
    - Real-time web search with AI-powered relevance ranking
    - Automatic content extraction and cleaning
    - Support for both general and news search topics
    - Advanced filtering and domain management
    - Image search capabilities with descriptions
    - Date range filtering for temporal queries

    Search Types:
    - general: Broader, general-purpose searches across various sources
    - news: Real-time updates from mainstream media sources

    Search Depth:
    - basic: Provides generic content snippets (1 API credit)
    - advanced: Tailored content snippets with better relevance (2 API credits)

    Args:
        query: The search query to execute with Tavily. This should be a clear, specific question
            or search term. Examples: "What is machine learning?", "Latest news about climate change"
        search_depth: The depth of the search ("basic" or "advanced")
        topic: The category of the search ("general" or "news")
        max_results: Maximum number of search results to return (0-20)
        auto_parameters: When enabled, Tavily automatically configures search parameters based
            on query content and intent. May automatically use advanced search (2 credits)
        chunks_per_source: Number of content chunks per source (1-3). Only available with
            advanced search depth. Chunks are 500-character snippets from each source
        time_range: Filter results by time range ("day", "week", "month", "year" or shorthand "d", "w", "m", "y")
        days: Number of days back from current date to include. Only available with news topic
        start_date: Include results after this date (YYYY-MM-DD format)
        end_date: Include results before this date (YYYY-MM-DD format)
        include_answer: Include an LLM-generated answer (False, True/"basic", or "advanced")
        include_raw_content: Include cleaned HTML content (False, True/"markdown", or "text")
        include_images: Include query-related images in the response
        include_image_descriptions: When include_images is True, also add descriptive text for each image
        include_favicon: Include favicon URLs for each result
        include_domains: List of domains to specifically include in results
        exclude_domains: List of domains to specifically exclude from results
        country: Boost results from specific country (only with general topic).
            Examples: "united states", "canada", "united kingdom"

    Returns:
        Dict containing search results and metadata with status and content fields.
    """

    try:
        # Validate parameters
        if not query or not query.strip():
            return {"status": "error", "content": [{"text": "Query parameter is required and cannot be empty"}]}

        # Validate max_results range
        if max_results is not None and not (0 <= max_results <= 20):
            return {"status": "error", "content": [{"text": "max_results must be between 0 and 20"}]}

        # Validate chunks_per_source range
        if chunks_per_source is not None and not (1 <= chunks_per_source <= 3):
            return {"status": "error", "content": [{"text": "chunks_per_source must be between 1 and 3"}]}

        # Get API key
        api_key = _get_api_key()

        # Build request payload
        payload = {
            "query": query,
            "search_depth": search_depth,
            "topic": topic,
            "max_results": max_results,
            "auto_parameters": auto_parameters,
            "chunks_per_source": chunks_per_source,
            "time_range": time_range,
            "days": days,
            "start_date": start_date,
            "end_date": end_date,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
            "include_image_descriptions": include_image_descriptions,
            "include_favicon": include_favicon,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "country": country,
        }

        # Make API request
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        url = f"{TAVILY_API_BASE_URL}{TAVILY_SEARCH_ENDPOINT}"

        payload = {key: value for key, value in payload.items() if value is not None}

        logger.info(f"Making Tavily search request for query: {query}")

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
        logger.error(f"Unexpected error in tavily_search: {str(e)}")
        return {"status": "error", "content": [{"text": f"Unexpected error: {str(e)}"}]}


@tool
async def tavily_extract(
    urls: Union[str, List[str]],
    extract_depth: Optional[Literal["basic", "advanced"]] = None,
    format: Optional[Literal["markdown", "text"]] = None,
    include_images: Optional[bool] = None,
    include_favicon: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Extract clean, structured content from one or more web pages using Tavily's extraction service.

    Tavily Extract provides high-quality content extraction with advanced processing to remove
    navigation, ads, and other noise, returning clean, readable content optimized for AI processing.

    Key Features:
    - Clean content extraction without ads or navigation
    - Support for multiple URLs in a single request
    - Advanced extraction with tables and embedded content
    - Multiple output formats (markdown, text)
    - Image extraction from pages
    - Favicon URL extraction

    Extract Depth:
    - basic: Standard extraction (1 credit per 5 successful extractions)
    - advanced: Enhanced extraction with tables/embedded content (2 credits per 5)

    Output Formats:
    - markdown: Returns content formatted as markdown (recommended for AI)
    - text: Returns plain text content (may increase latency)

    Args:
        urls: A single URL string or list of URL strings to extract content from
        extract_depth: The depth of the extraction process ("basic" or "advanced")
        format: The format of the extracted content ("markdown" or "text")
        include_images: Whether to include a list of images from the extracted pages
        include_favicon: Whether to include the favicon URL for each result

    Returns:
        Dict containing extraction results and metadata with status and content fields.
    """

    try:
        # Validate parameters
        if not urls:
            return {"status": "error", "content": [{"text": "At least one URL must be provided"}]}

        # Get API key
        api_key = _get_api_key()

        # Build request payload
        payload = {
            "urls": urls,
            "extract_depth": extract_depth,
            "format": format,
            "include_images": include_images,
            "include_favicon": include_favicon,
        }

        # Make API request
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        url = f"{TAVILY_API_BASE_URL}{TAVILY_EXTRACT_ENDPOINT}"

        payload = {key: value for key, value in payload.items() if value is not None}

        url_count = len(urls) if isinstance(urls, list) else 1
        logger.info(f"Making Tavily extract request for {url_count} URLs")

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                try:
                    data = await response.json()
                except Exception as e:
                    return {"status": "error", "content": [{"text": f"Failed to parse API response: {str(e)}"}]}

        # Format and display response
        panel = format_extract_response(data)
        console.print(panel)

        return {"status": "success", "content": [{"text": str(data)}]}

    except asyncio.TimeoutError:
        return {"status": "error", "content": [{"text": "Request timeout. The API request took too long to complete."}]}
    except aiohttp.ClientError:
        return {"status": "error", "content": [{"text": "Connection error. Please check your internet connection."}]}
    except ValueError as e:
        return {"status": "error", "content": [{"text": str(e)}]}
    except Exception as e:
        logger.error(f"Unexpected error in tavily_extract: {str(e)}")
        return {"status": "error", "content": [{"text": f"Unexpected error: {str(e)}"}]}


@tool
async def tavily_crawl(
    url: str,
    max_depth: Optional[int] = None,
    max_breadth: Optional[int] = None,
    limit: Optional[int] = None,
    instructions: Optional[str] = None,
    select_paths: Optional[List[str]] = None,
    select_domains: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    allow_external: Optional[bool] = None,
    include_images: Optional[bool] = None,
    categories: Optional[
        List[
            Literal[
                "Careers", "Blog", "Documentation", "About", "Pricing", "Community", "Developers", "Contact", "Media"
            ]
        ]
    ] = None,
    extract_depth: Optional[Literal["basic", "advanced"]] = None,
    format: Optional[Literal["markdown", "text"]] = None,
    include_favicon: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Crawl multiple pages from a website starting from a base URL using Tavily's crawling service.

    Tavily Crawl is a graph-based website traversal tool that can explore hundreds of paths in parallel
    with built-in extraction and intelligent discovery. This allows comprehensive website exploration
    starting from a single URL.

    Key Features:
    - Graph-based website traversal with parallel exploration
    - Built-in content extraction and cleaning
    - Intelligent discovery of related pages
    - Advanced filtering by paths, domains, and categories
    - Natural language instructions for targeted crawling
    - Support for both basic and advanced extraction depths

    Extraction Depth:
    - basic: Standard extraction (1 credit per 5 successful extractions)
    - advanced: Enhanced extraction with tables/embedded content (2 credits per 5)

    Content Format:
    - markdown: Returns content formatted as markdown (recommended for AI)
    - text: Returns plain text content (may increase latency)

    Args:
        url: The root URL to begin the crawl from. This should be a complete URL including protocol
        max_depth: Maximum depth of the crawl. Defines how far from the base URL the crawler can explore
        max_breadth: Maximum number of links to follow per level of the tree (i.e., per page)
        limit: Total number of links the crawler will process before stopping
        instructions: Natural language instructions for the crawler. When specified, the cost increases
            to 2 API credits per 10 successful pages instead of 1 API credit per 10 pages
        select_paths: List of regex patterns to select only URLs with specific path patterns
        select_domains: List of regex patterns to select crawling to specific domains or subdomains
        exclude_paths: List of regex patterns to exclude URLs with specific path patterns
        exclude_domains: List of regex patterns to exclude specific domains or subdomains from crawling
        allow_external: Whether to allow following links that go to external domains
        include_images: Whether to include images in the crawl results
        categories: List of predefined categories to filter URLs
        extract_depth: The depth of content extraction ("basic" or "advanced")
        format: The format of the extracted content ("markdown" or "text")
        include_favicon: Whether to include the favicon URL for each result

    Returns:
        Dict containing crawl results and metadata with status and content fields.
    """

    try:
        # Validate parameters
        if not url or not url.strip():
            return {"status": "error", "content": [{"text": "URL parameter is required and cannot be empty"}]}

        # Validate numeric parameters
        if max_depth is not None and max_depth < 1:
            return {"status": "error", "content": [{"text": "max_depth must be at least 1"}]}

        if max_breadth is not None and max_breadth < 1:
            return {"status": "error", "content": [{"text": "max_breadth must be at least 1"}]}

        if limit is not None and limit < 1:
            return {"status": "error", "content": [{"text": "limit must be at least 1"}]}

        # Get API key
        api_key = _get_api_key()

        # Build request payload
        payload = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "extract_depth": extract_depth,
            "format": format,
            "include_favicon": include_favicon,
            "include_images": include_images,
            "categories": categories,
            "instructions": instructions,
            "select_paths": select_paths,
            "select_domains": select_domains,
            "exclude_paths": exclude_paths,
            "exclude_domains": exclude_domains,
            "allow_external": allow_external,
        }

        # Make API request
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        api_url = f"{TAVILY_API_BASE_URL}{TAVILY_CRAWL_ENDPOINT}"

        payload = {key: value for key, value in payload.items() if value is not None}

        logger.info(f"Making Tavily crawl request for URL: {url}")

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload, headers=headers) as response:
                try:
                    data = await response.json()
                except Exception as e:
                    return {"status": "error", "content": [{"text": f"Failed to parse API response: {str(e)}"}]}

        # Format and display response
        panel = format_crawl_response(data)
        console.print(panel)

        return {"status": "success", "content": [{"text": str(data)}]}

    except asyncio.TimeoutError:
        return {
            "status": "error",
            "content": [{"text": "Request timeout. The crawl request took too long to complete."}],
        }
    except aiohttp.ClientError:
        return {"status": "error", "content": [{"text": "Connection error. Please check your internet connection."}]}
    except ValueError as e:
        return {"status": "error", "content": [{"text": str(e)}]}
    except Exception as e:
        logger.error(f"Unexpected error in tavily_crawl: {str(e)}")
        return {"status": "error", "content": [{"text": f"Unexpected error: {str(e)}"}]}


@tool
async def tavily_map(
    url: str,
    max_depth: Optional[int] = None,
    max_breadth: Optional[int] = None,
    limit: Optional[int] = None,
    instructions: Optional[str] = None,
    select_paths: Optional[List[str]] = None,
    select_domains: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    allow_external: Optional[bool] = None,
    categories: Optional[
        List[
            Literal[
                "Careers", "Blog", "Documentation", "About", "Pricing", "Community", "Developers", "Contact", "Media"
            ]
        ]
    ] = None,
) -> Dict[str, Any]:
    """
    Map website structure starting from a base URL using Tavily's mapping service.

    Tavily Map traverses websites like a graph and can explore hundreds of paths in parallel
    with intelligent discovery to generate comprehensive site maps. This returns a list of
    discovered URLs without content extraction.

    Key Features:
    - Graph-based website traversal with parallel exploration
    - Intelligent discovery of website structure and pages
    - Advanced filtering by paths, domains, and categories
    - Natural language instructions for targeted mapping
    - URL discovery without content extraction for faster mapping
    - Comprehensive site structure analysis

    Use Cases:
    - Discover all pages on a website
    - Understand website structure and organization
    - Find specific types of pages (documentation, blog posts, etc.)
    - Generate sitemaps for analysis

    Args:
        url: The root URL to begin the mapping from. This should be a complete URL including protocol
        max_depth: Maximum depth of the mapping. Defines how far from the base URL the mapper can explore
        max_breadth: Maximum number of links to follow per level of the tree (i.e., per page)
        limit: Total number of links the mapper will process before stopping
        instructions: Natural language instructions for the mapper
        select_paths: List of regex patterns to select only URLs with specific path patterns
        select_domains: List of regex patterns to select mapping to specific domains or subdomains
        exclude_paths: List of regex patterns to exclude URLs with specific path patterns
        exclude_domains: List of regex patterns to exclude specific domains or subdomains from mapping
        allow_external: Whether to allow following links that go to external domains
        categories: List of predefined categories to filter URLs

    Returns:
        Dict containing map results and metadata with status and content fields.
    """

    try:
        # Validate parameters
        if not url or not url.strip():
            return {"status": "error", "content": [{"text": "URL parameter is required and cannot be empty"}]}

        # Validate numeric parameters
        if max_depth is not None and max_depth < 1:
            return {"status": "error", "content": [{"text": "max_depth must be at least 1"}]}

        if max_breadth is not None and max_breadth < 1:
            return {"status": "error", "content": [{"text": "max_breadth must be at least 1"}]}

        if limit is not None and limit < 1:
            return {"status": "error", "content": [{"text": "limit must be at least 1"}]}

        # Get API key
        api_key = _get_api_key()

        # Build request payload
        payload = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "instructions": instructions,
            "select_paths": select_paths,
            "select_domains": select_domains,
            "exclude_paths": exclude_paths,
            "exclude_domains": exclude_domains,
            "allow_external": allow_external,
            "categories": categories,
        }

        # Make API request
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        api_url = f"{TAVILY_API_BASE_URL}{TAVILY_MAP_ENDPOINT}"

        payload = {key: value for key, value in payload.items() if value is not None}

        logger.info(f"Making Tavily map request for URL: {url}")

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload, headers=headers) as response:
                try:
                    data = await response.json()
                except Exception as e:
                    return {"status": "error", "content": [{"text": f"Failed to parse API response: {str(e)}"}]}

        # Format and display response
        panel = format_map_response(data)
        console.print(panel)

        return {"status": "success", "content": [{"text": str(data)}]}

    except asyncio.TimeoutError:
        return {
            "status": "error",
            "content": [{"text": "Request timeout. The mapping request took too long to complete."}],
        }
    except aiohttp.ClientError:
        return {"status": "error", "content": [{"text": "Connection error. Please check your internet connection."}]}
    except ValueError as e:
        return {"status": "error", "content": [{"text": str(e)}]}
    except Exception as e:
        logger.error(f"Unexpected error in tavily_map: {str(e)}")
        return {"status": "error", "content": [{"text": f"Unexpected error: {str(e)}"}]}
