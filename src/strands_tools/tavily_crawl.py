import logging
import os
from typing import Any, Dict

import requests
from rich.console import Console
from rich.panel import Panel
from strands.types.tools import ToolResult, ToolResultContent, ToolUse


logger = logging.getLogger(__name__)

# Tavily API configuration
TAVILY_API_BASE_URL = "https://api.tavily.com"
TAVILY_CRAWL_ENDPOINT = "/crawl"

# Initialize Rich console
console = Console()

TOOL_SPEC = {
    "name": "tavily_crawl",
    "description": (
        "Crawl multiple pages from a website starting from a base URL using Tavily's crawling service.\n\n"
        "Tavily Crawl is a graph-based website traversal tool that can explore hundreds of paths in parallel "
        "with built-in extraction and intelligent discovery. This allows comprehensive "
        "website exploration starting from a single URL.\n\n"
        "Key Features:\n"
        "- Graph-based website traversal with parallel exploration\n"
        "- Built-in content extraction and cleaning\n"
        "- Intelligent discovery of related pages\n"
        "- Advanced filtering by paths, domains, and categories\n"
        "- Natural language instructions for targeted crawling\n"
        "- Support for both basic and advanced extraction depths\n\n"
        "Extraction Depth:\n"
        "- basic: Standard extraction (1 credit per 5 successful extractions)\n"
        "- advanced: Enhanced extraction with tables/embedded content (2 credits per 5)\n\n"
        "Content Format:\n"
        "- markdown: Returns content formatted as markdown (recommended for AI)\n"
        "- text: Returns plain text content (may increase latency)"
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The root URL to begin the crawl from. This should be a complete URL including protocol.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth of the crawl. Defines how far from the base URL the crawler can explore.",
                    "minimum": 1,
                },
                "max_breadth": {
                    "type": "integer",
                    "description": "Maximum number of links to follow per level of the tree (i.e., per page).",
                    "minimum": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Total number of links the crawler will process before stopping.",
                    "minimum": 1,
                },
                "instructions": {
                    "type": "string",
                    "description": "Natural language instructions for the crawler",
                },
                "select_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of regex patterns to select only URLs with specific path patterns.",
                },
                "select_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of regex patterns to select crawling to specific domains or subdomains.",
                },
                "exclude_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of regex patterns to exclude URLs with specific path patterns.",
                },
                "exclude_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of regex patterns to exclude specific domains or subdomains from crawling.",
                },
                "allow_external": {
                    "type": "boolean",
                    "description": "Whether to allow following links that go to external domains. Default is False.",
                },
                "include_images": {
                    "type": "boolean",
                    "description": "Whether to include images in the crawl results. Default is False.",
                },
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["Careers", "Blog", "Documentation", "About", "Pricing", "Community", "Developers", "Contact", "Media"]
                    },
                    "description": "List of predefined categories to filter URLs.",
                },
                "extract_depth": {
                    "type": "string",
                    "description": "The depth of content extraction.",
                    "enum": ["basic", "advanced"],
                },
                "format": {
                    "type": "string",
                    "description": "The format of the extracted content.",
                    "enum": ["markdown", "text"],
                },
                "include_favicon": {
                    "type": "boolean",
                    "description": "Whether to include the favicon URL for each result. Default is False.",
                },
            },
            "required": ["url"],
        }
    },
}

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


def tavily_crawl(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Crawl multiple pages from a website starting from a base URL using Tavily's crawling service (BETA).
    
    Tavily Crawl is a graph-based website traversal tool that can explore hundreds of paths in parallel 
    with built-in extraction and intelligent discovery. This is a BETA feature that allows comprehensive
    website exploration starting from a single URL.
    
    Args:
        tool: ToolUse object containing the following input fields:
            - url: The root URL to begin the crawl from. This should be a complete URL including protocol.
            - max_depth: Maximum depth of the crawl. Defines how far from the base URL the crawler can explore.
            - max_breadth: Maximum number of links to follow per level of the tree (i.e., per page).
            - limit: Total number of links the crawler will process before stopping.
            - instructions: Natural language instructions for the crawler. When specified, the cost increases
                to 2 API credits per 10 successful pages instead of 1 API credit per 10 pages.
            - select_paths: List of regex patterns to select only URLs with specific path patterns.
            - select_domains: List of regex patterns to select crawling to specific domains or subdomains.
            - exclude_paths: List of regex patterns to exclude URLs with specific path patterns.
            - exclude_domains: List of regex patterns to exclude specific domains or subdomains from crawling.
            - allow_external: Whether to allow following links that go to external domains. Default is False.
            - include_images: Whether to include images in the crawl results. Default is False.
            - categories: List of predefined categories to filter URLs.
            - extract_depth: The depth of content extraction (basic or advanced).
            - format: The format of the extracted content (markdown or text).
            - include_favicon: Whether to include the favicon URL for each result. Default is False.
        **kwargs: Additional keyword arguments

    Returns:
        ToolResult containing crawl results and metadata:
        {
            "toolUseId": "tool-use-id",
            "status": "success|error",
            "content": [
                {
                    "text": "JSON string containing crawl results"
                }
            ]
        }
    """
    
    try:
        # Extract input from tool use object
        tool_input = tool.get("input", {})
        tool_use_id = tool.get("toolUseId", "default-id")

        # Validate parameters
        url = tool_input.get("url")
        if not url or not url.strip():
            return ToolResult(
                toolUseId=tool_use_id,
                status="error",
                content=[ToolResultContent(text="URL parameter is required and cannot be empty")]
            )
        
        # Get API key
        api_key = _get_api_key()
        
        # Build request payload
        payload = {
            "url": url,
            "max_depth": tool_input.get("max_depth"),
            "max_breadth": tool_input.get("max_breadth"),
            "limit": tool_input.get("limit"),
            "extract_depth": tool_input.get("extract_depth"),
            "format": tool_input.get("format"),
            "include_favicon": tool_input.get("include_favicon"),
            "include_images": tool_input.get("include_images"),
            "categories": tool_input.get("categories"),
            "instructions": tool_input.get("instructions"),
            "select_paths": tool_input.get("select_paths"),
            "select_domains": tool_input.get("select_domains"),
            "exclude_paths": tool_input.get("exclude_paths"),
            "exclude_domains": tool_input.get("exclude_domains"),
            "allow_external": tool_input.get("allow_external"),
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
            return ToolResult(
                toolUseId=tool_use_id,
                status="error",
                content=[ToolResultContent(text=f"Failed to parse API response: {str(e)}")]
            )

        # Format and display response
        panel = format_crawl_response(data)
        console.print(panel)

        return ToolResult(
            toolUseId=tool_use_id,
            status="success",
            content=[ToolResultContent(text=str(data))]
        )
        
    except requests.exceptions.Timeout:
        return ToolResult(
            toolUseId=tool_use_id,
            status="error",
            content=[ToolResultContent(text="Request timeout. The crawl request took too long to complete.")]
        )
    except requests.exceptions.ConnectionError:
        return ToolResult(
            toolUseId=tool_use_id,
            status="error", 
            content=[ToolResultContent(text="Connection error. Please check your internet connection.")]
        )
    except ValueError as e:
        return ToolResult(
            toolUseId=tool_use_id,
            status="error",
            content=[ToolResultContent(text=str(e))]
        )
    except Exception as e:
        logger.error(f"Unexpected error in tavily_crawl: {str(e)}")
        return ToolResult(
            toolUseId=tool_use_id,
            status="error",
            content=[ToolResultContent(text=f"Unexpected error: {str(e)}")]
        ) 