
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
TAVILY_SEARCH_ENDPOINT = "/search"

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


@tool
def tavily_search(
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
            return {
                "status": "error",
                "content": [{"text": "Query parameter is required and cannot be empty"}]
            }

        # Validate max_results range
        if max_results is not None and not (0 <= max_results <= 20):
            return {
                "status": "error",
                "content": [{"text": "max_results must be between 0 and 20"}]
            }

        # Validate chunks_per_source range
        if chunks_per_source is not None and not (1 <= chunks_per_source <= 3):
            return {
                "status": "error",
                "content": [{"text": "chunks_per_source must be between 1 and 3"}]
            }
    
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
            "country": country
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
            return {
                "status": "error",
                "content": [{"text": f"Failed to parse API response: {str(e)}"}]
            }
    
        # Format and display response
        panel = format_search_response(data)
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
        logger.error(f"Unexpected error in tavily_search: {str(e)}")
        return {
            "status": "error",
            "content": [{"text": f"Unexpected error: {str(e)}"}]
        } 
