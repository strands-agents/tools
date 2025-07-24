
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
TAVILY_SEARCH_ENDPOINT = "/search"

# Initialize Rich console
console = Console()

TOOL_SPEC = {
    "name": "tavily_search",
    "description": (
        "Search the web for real-time information using Tavily's AI-optimized search engine.\n\n"
        "Tavily is a search engine specifically optimized for LLMs and AI agents. It handles all the "
        "complexity of searching, scraping, filtering, and extracting the most relevant information "
        "from online sources in a single API call.\n\n"
        "Key Features:\n"
        "- Real-time web search with AI-powered relevance ranking\n"
        "- Automatic content extraction and cleaning\n"
        "- Support for both general and news search topics\n"
        "- Advanced filtering and domain management\n"
        "- Image search capabilities with descriptions\n"
        "- Date range filtering for temporal queries\n\n"
        "Search Types:\n"
        "- general: Broader, general-purpose searches across various sources\n"
        "- news: Real-time updates from mainstream media sources\n\n"
        "Search Depth:\n"
        "- basic: Provides generic content snippets (1 API credit)\n"
        "- advanced: Tailored content snippets with better relevance (2 API credits)"
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The users's input. Should be a clear, specific search query.",
                },
                "search_depth": {   
                    "type": "string",
                    "description": "The depth of the search (basic or advanced)",
                    "enum": ["basic", "advanced"],
                },
                "topic": {
                    "type": "string", 
                    "description": "The category of the search (general or news)",
                    "enum": ["general", "news"],
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return (0-20)",
                    "minimum": 0,
                    "maximum": 20,
                },
                "auto_parameters": {
                    "type": "boolean",
                    "description": "When enabled, Tavily automatically configures search parameters based on query content",
                },
                "chunks_per_source": {
                    "type": "integer",
                    "description": "Number of content chunks per source (1-3). Only available with advanced search depth",
                    "minimum": 1,
                    "maximum": 3,
                },
                "time_range": {
                    "type": "string",
                    "description": "Filter results by time range",
                    "enum": ["day", "week", "month", "year", "d", "w", "m", "y"],
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days back from current date to include. Only available with news topic",
                },
                "start_date": {
                    "type": "string",
                    "description": "Include results after this date (YYYY-MM-DD format)",
                },
                "end_date": {
                    "type": "string",
                    "description": "Include results before this date (YYYY-MM-DD format)",
                },
                "include_answer": {
                    "oneOf": [
                        {"type": "boolean"},
                        {"type": "string", "enum": ["basic", "advanced"]},
                    ],
                    "description": "Include an LLM-generated answer (false, true/basic, or advanced)",
                },
                "include_raw_content": {
                    "oneOf": [
                        {"type": "boolean"},
                        {"type": "string", "enum": ["markdown", "text"]},
                    ],
                    "description": "Include cleaned HTML content (false, true/markdown, or text)",
                },
                "include_images": {
                    "type": "boolean",
                    "description": "Include query-related images in the response",
                },
                "include_image_descriptions": {
                    "type": "boolean",
                    "description": "When include_images is True, also add descriptive text for each image",
                },
                "include_favicon": {
                    "type": "boolean",
                    "description": "Include favicon URLs for each result",
                },
                "include_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of domains to specifically include in results",
                },
                "exclude_domains": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of domains to specifically exclude from results",
                },
            },
            "required": ["query"],
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


def tavily_search(tool: ToolUse, **kwargs: Any) -> ToolResult:
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

    Args:
        tool: ToolUse object containing the following input fields:
            - query: The search query to execute with Tavily. This should be a clear, specific question
                or search term. Examples: "What is machine learning?", "Latest news about climate change"
            - search_depth: The depth of the search. Options:
                - "basic": Provides generic content snippets (1 API credit)
                - "advanced": Tailored content snippets with better relevance (2 API credits)
            - topic: The category of the search. Options:
                - "general": Broader, general-purpose searches across various sources
                - "news": Real-time updates from mainstream media sources
            - max_results: Maximum number of search results to return (0-20). 
            - auto_parameters: When enabled, Tavily automatically configures search parameters based
                on query content and intent. May automatically use advanced search (2 credits).
            - chunks_per_source: Number of content chunks per source (1-3). Only available with
                advanced search depth. Chunks are 500-character snippets from each source.
            - time_range: Filter results by time range. Options: "day", "week", "month", "year" 
                (or shorthand "d", "w", "m", "y")
            - days: Number of days back from current date to include. Only available with news topic.
            - start_date: Include results after this date (YYYY-MM-DD format)
            - end_date: Include results before this date (YYYY-MM-DD format)
            - include_answer: Include an LLM-generated answer. Options:
                - False: No answer
                - True or "basic": Quick answer
                - "advanced": Detailed answer
            - include_raw_content: Include cleaned HTML content. Options:
                - False: No raw content
                - True or "markdown": Content in markdown format
                - "text": Plain text content (may increase latency)
            - include_images: Include query-related images in the response
            - include_image_descriptions: When include_images is True, also add descriptive text for each image
            - include_favicon: Include favicon URLs for each result
            - include_domains: List of domains to specifically include in results
            - exclude_domains: List of domains to specifically exclude from results  
            - country: Boost results from specific country (only with general topic). 
                Examples: "united states", "canada", "united kingdom"
        **kwargs: Additional keyword arguments

    Returns:
        ToolResult containing search results and metadata:
        {
            "toolUseId": "tool-use-id",
            "status": "success|error",
            "content": [
                {
                    "text": "JSON string containing search results"
                }
            ]
        }
    """
    
    try:
        # Extract input from tool use object
        tool_input = tool.get("input", {})
        tool_use_id = tool.get("toolUseId", "default-id")

        # Validate parameters
        query = tool_input.get("query")
        if not query or not query.strip():
            return ToolResult(
                toolUseId=tool_use_id,
                status="error",
                content=[ToolResultContent(text="Query parameter is required and cannot be empty")]
            )
    
        # Get API key
        api_key = _get_api_key()
        
        # Build request payload
        payload = {
            "query": query,
            "search_depth": tool_input.get("search_depth"),
            "topic": tool_input.get("topic"),
            "max_results": tool_input.get("max_results"),
            "auto_parameters": tool_input.get("auto_parameters"),
            "chunks_per_source": tool_input.get("chunks_per_source"),
            "time_range": tool_input.get("time_range"),
            "days": tool_input.get("days"),
            "start_date": tool_input.get("start_date"),
            "end_date": tool_input.get("end_date"),
            "include_answer": tool_input.get("include_answer"),
            "include_raw_content": tool_input.get("include_raw_content"),
            "include_images": tool_input.get("include_images"),
            "include_image_descriptions": tool_input.get("include_image_descriptions"),
            "include_favicon": tool_input.get("include_favicon"),
            "include_domains": tool_input.get("include_domains"),
            "exclude_domains": tool_input.get("exclude_domains"),
            "country": tool_input.get("country")
        }
        
        # Make API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        url = f"{TAVILY_API_BASE_URL}{TAVILY_SEARCH_ENDPOINT}"
        
        payload = {key: value for key, value in payload.items() if value is not None}

        logger.info(f"Making Tavily search request for query: {query}")
        response = requests.post(url, json=payload, headers=headers)
        
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
        panel = format_search_response(data)
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
            content=[ToolResultContent(text="Request timeout. The API request took too long to complete.")]
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
        logger.error(f"Unexpected error in tavily_search: {str(e)}")
        return ToolResult(
            toolUseId=tool_use_id,
            status="error",
            content=[ToolResultContent(text=f"Unexpected error: {str(e)}")]
        ) 
