import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

import requests
from rich.console import Console
from rich.panel import Panel
from strands import tool

logger = logging.getLogger(__name__)

# Tavily API configuration
TAVILY_API_BASE_URL = "https://api.tavily.com"
TAVILY_EXTRACT_ENDPOINT = "/extract"

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


@tool
def tavily_extract(
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
            return {
                "status": "error",
                "content": [{"text": "At least one URL must be provided"}]
            }

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
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        url = f"{TAVILY_API_BASE_URL}{TAVILY_EXTRACT_ENDPOINT}"

        payload = {key: value for key, value in payload.items() if value is not None}
        
        url_count = len(urls) if isinstance(urls, list) else 1
        logger.info(f"Making Tavily extract request for {url_count} URLs")
        response = requests.post(url, json=payload, headers=headers)

        # Parse response
        try:
            data = response.json()
        except ValueError as e:
            return {
                "status": "error",
                "content": [{"text": f"Failed to parse API response: {str(e)}"}]
            }

        # Format and display response
        panel = format_extract_response(data)
        console.print(panel)

        return {
            "status": "success",
            "content": [{"text": str(data)}]
        }
        
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "content": [{"text": "Request timeout. The API request took too long to complete."}]
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
        logger.error(f"Unexpected error in tavily_extract: {str(e)}")
        return {
            "status": "error",
            "content": [{"text": f"Unexpected error: {str(e)}"}]
        } 
