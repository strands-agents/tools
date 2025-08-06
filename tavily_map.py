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
TAVILY_MAP_ENDPOINT = "/map"

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
    
    return Panel("\n".join(content), title="[bold blue]Tavily Map Results", border_style="blue")


@tool
def tavily_map(
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
    categories: Optional[List[Literal["Careers", "Blog", "Documentation", "About", "Pricing", "Community", "Developers", "Contact", "Media"]]] = None,
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
            "instructions": instructions,
            "select_paths": select_paths,
            "select_domains": select_domains,
            "exclude_paths": exclude_paths,
            "exclude_domains": exclude_domains,
            "allow_external": allow_external,
            "categories": categories,
        }
        
        # Make API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        api_url = f"{TAVILY_API_BASE_URL}{TAVILY_MAP_ENDPOINT}"

        payload = {key: value for key, value in payload.items() if value is not None}
        
        logger.info(f"Making Tavily map request for URL: {url}")
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
        panel = format_map_response(data)
        console.print(panel)

        return {
            "status": "success",
            "content": [{"text": str(data)}]
        }
        
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "content": [{"text": "Request timeout. The mapping request took too long to complete."}]
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
        logger.error(f"Unexpected error in tavily_map: {str(e)}")
        return {
            "status": "error",
            "content": [{"text": f"Unexpected error: {str(e)}"}]
        } 