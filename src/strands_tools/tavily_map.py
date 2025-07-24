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
TAVILY_MAP_ENDPOINT = "/map"

# Initialize Rich console
console = Console()

TOOL_SPEC = {
    "name": "tavily_map",
    "description": (
        "Map website structure starting from a base URL using Tavily's mapping service.\n\n"
        "Tavily Map traverses websites like a graph and can explore hundreds of paths in parallel "
        "with intelligent discovery to generate comprehensive site maps. This "
        "returns a list of discovered URLs without content extraction.\n\n"
        "Key Features:\n"
        "- Graph-based website traversal with parallel exploration\n"
        "- Intelligent discovery of website structure and pages\n"
        "- Advanced filtering by paths, domains, and categories\n"
        "- Natural language instructions for targeted mapping\n"
        "- URL discovery without content extraction for faster mapping\n"
        "- Comprehensive site structure analysis\n\n"
        "Use Cases:\n"
        "- Discover all pages on a website\n"
        "- Understand website structure and organization\n"
        "- Find specific types of pages (documentation, blog posts, etc.)\n"
        "- Generate sitemaps for analysis"
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The root URL to begin the mapping from. This should be a complete URL including protocol.",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth of the mapping. Defines how far from the base URL the mapper can explore.",
                    "minimum": 1,
                },
                "max_breadth": {
                    "type": "integer",
                    "description": "Maximum number of links to follow per level of the tree (i.e., per page).",
                    "minimum": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Total number of links the mapper will process before stopping.",
                    "minimum": 1,
                },
                "instructions": {
                    "type": "string",
                    "description": "Natural language instructions for the mapper",
                },
                "select_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of regex patterns to select only URLs with specific path patterns.",
                },
                "select_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of regex patterns to select mapping to specific domains or subdomains.",
                },
                "exclude_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of regex patterns to exclude URLs with specific path patterns.",
                },
                "exclude_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of regex patterns to exclude specific domains or subdomains from mapping.",
                },
                "allow_external": {
                    "type": "boolean",
                    "description": "Whether to allow following links that go to external domains. Default is False.",
                },
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["Careers", "Blog", "Documentation", "About", "Pricing", "Community", "Developers", "Contact", "Media"]
                    },
                    "description": "List of predefined categories to filter URLs.",
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


def tavily_map(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Map website structure starting from a base URL using Tavily's mapping service.
    
    Tavily Map traverses websites like a graph and can explore hundreds of paths in parallel 
    with intelligent discovery to generate comprehensive site maps. This returns a list of 
    discovered URLs without content extraction.
    
    Args:
        tool: ToolUse object containing the following input fields:
            - url: The root URL to begin the mapping from. This should be a complete URL including protocol.
            - max_depth: Maximum depth of the mapping. Defines how far from the base URL the mapper can explore.
            - max_breadth: Maximum number of links to follow per level of the tree (i.e., per page).
            - limit: Total number of links the mapper will process before stopping.
            - instructions: Natural language instructions for the mapper.
            - select_paths: List of regex patterns to select only URLs with specific path patterns.
            - select_domains: List of regex patterns to select mapping to specific domains or subdomains.
            - exclude_paths: List of regex patterns to exclude URLs with specific path patterns.
            - exclude_domains: List of regex patterns to exclude specific domains or subdomains from mapping.
            - allow_external: Whether to allow following links that go to external domains. Default is False.
            - categories: List of predefined categories to filter URLs.
        **kwargs: Additional keyword arguments

    Returns:
        ToolResult containing map results and metadata:
        {
            "toolUseId": "tool-use-id",
            "status": "success|error",
            "content": [
                {
                    "text": "JSON string containing map results"
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
            "instructions": tool_input.get("instructions"),
            "select_paths": tool_input.get("select_paths"),
            "select_domains": tool_input.get("select_domains"),
            "exclude_paths": tool_input.get("exclude_paths"),
            "exclude_domains": tool_input.get("exclude_domains"),
            "allow_external": tool_input.get("allow_external"),
            "categories": tool_input.get("categories"),
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
            return ToolResult(
                toolUseId=tool_use_id,
                status="error",
                content=[ToolResultContent(text=f"Failed to parse API response: {str(e)}")]
            )

        # Format and display response
        panel = format_map_response(data)
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
            content=[ToolResultContent(text="Request timeout. The mapping request took too long to complete.")]
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
        logger.error(f"Unexpected error in tavily_map: {str(e)}")
        return ToolResult(
            toolUseId=tool_use_id,
            status="error",
            content=[ToolResultContent(text=f"Unexpected error: {str(e)}")]
        ) 