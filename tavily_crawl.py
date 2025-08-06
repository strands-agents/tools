import logging
import os
from typing import Any, Dict, List, Literal, Optional

import requests
from rich.console import Console
from rich.panel import Panel
from strands import tool

logger = logging.getLogger(__name__)

# Tavily API configuration
TAVILY_API_BASE_URL = "https://api.tavily.com"
TAVILY_CRAWL_ENDPOINT = "/crawl"

# Initialize Rich console
console = Console()


def _get_api_key() -> str:
    """Get Tavily API key from environment variables."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable is required. "
            "Get your free API key at https://app.tavily.com"
        )
    return api_key


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
    
    return Panel("\n".join(content), title="[bold green]Tavily Crawl Results", border_style="green")


@tool
def tavily_crawl(
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
    categories: Optional[List[Literal["Careers", "Blog", "Documentation", "About", "Pricing", "Community", "Developers", "Contact", "Media"]]] = None,
    extract_depth: Optional[Literal["basic", "advanced"]] = None,
    format: Optional[Literal["markdown", "text"]] = None,
    include_favicon: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Crawl multiple pages from a website starting from a base URL using Tavily's crawling service (BETA).
    
    Tavily Crawl is a graph-based website traversal tool that can explore hundreds of paths in parallel 
    with built-in extraction and intelligent discovery. This is a BETA feature that allows comprehensive
    website exploration starting from a single URL.

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
            return {
                "status": "error",
                "content": [{"text": "URL parameter is required and cannot be empty"}]
            }

        # Validate numeric parameters
        if max_depth is not None and max_depth < 1:
            return {
                "status": "error",
                "content": [{"text": "max_depth must be at least 1"}]
            }

        if max_breadth is not None and max_breadth < 1:
            return {
                "status": "error",
                "content": [{"text": "max_breadth must be at least 1"}]
            }

        if limit is not None and limit < 1:
            return {
                "status": "error",
                "content": [{"text": "limit must be at least 1"}]
            }
        
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
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        api_url = f"{TAVILY_API_BASE_URL}{TAVILY_CRAWL_ENDPOINT}"
        
        payload = {key: value for key, value in payload.items() if value is not None}

        logger.info(f"Making Tavily crawl request for URL: {url}")
        response = requests.post(api_url, json=payload, headers=headers) 
        
        # Parse response
        try:
            data = response.json()
        except ValueError as e:
            return {
                "status": "error",
                "content": [{"text": f"Failed to parse API response: {str(e)}"}]
            }

        # Format and display response
        panel = format_crawl_response(data)
        console.print(panel)

        return {
            "status": "success",
            "content": [{"text": str(data)}]
        }
        
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "content": [{"text": "Request timeout. The crawl request took too long to complete."}]
        }
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "content": [{"text": "Connection error. Please check your internet connection."}]
        }
    except ValueError as e:
        return {
            "status": "error",
            "content": [{"text": str(e)}]
        }
    except Exception as e:
        logger.error(f"Unexpected error in tavily_crawl: {str(e)}")
        return {
            "status": "error",
            "content": [{"text": f"Unexpected error: {str(e)}"}]
        } 