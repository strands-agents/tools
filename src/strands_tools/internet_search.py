"""
Internet search tool

Internet search tool for Strands Agent supporting multiple backends (DuckDuckGo, SerpAPI, Tavily, etc).

Example usage:

    # DuckDuckGo (default)
    result = agent.tool.internet_search(query="latest AI news", max_results=5)

    # SerpAPI (requires API key)
    result = agent.tool.internet_search(
        query="latest AI news",
        max_results=5,
        backend="serpapi",
        serpapi_api_key="YOUR_SERPAPI_KEY"
    )

    # Tavily (requires API key)
    result = agent.tool.internet_search(
        query="latest AI news",
        max_results=5,
        backend="tavily",
        tavily_api_key="YOUR_TAVILY_KEY"
    )

"""

from typing import Any, Dict, List

from duckduckgo_search import DDGS
from strands.types.tools import ToolResult, ToolUse

try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

TOOL_SPEC = {
    "name": "internet_search",
    "description": "Search the internet for up-to-date information using DuckDuckGo, SerpAPI, Tavily, etc.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to look up on the internet."},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return.",
                    "default": 5,
                },
                "backend": {
                    "type": "string",
                    "description": "Which backend to use: 'duckduckgo', 'serpapi', or 'tavily'.",
                    "default": "duckduckgo",
                },
                "serpapi_api_key": {
                    "type": "string",
                    "description": "API key for SerpAPI (required if backend is 'serpapi').",
                },
                "tavily_api_key": {
                    "type": "string",
                    "description": "API key for Tavily (required if backend is 'tavily').",
                },
            },
            "required": ["query"],
        }
    },
}


def _search_with_backend(
    backend: str,
    query: str,
    max_results: int,
    serpapi_api_key: str = None,
    tavily_api_key: str = None,
) -> List[Dict[str, Any]]:
    if backend == "duckduckgo":
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({"title": r.get("title"), "href": r.get("href"), "body": r.get("body")})
        return results

    elif backend == "serpapi":
        if GoogleSearch is None:
            raise ImportError("serpapi is not installed. Install with: pip install google-search-results")
        if not serpapi_api_key:
            raise ValueError("serpapi_api_key is required for SerpAPI backend.")
        params = {
            "q": query,
            "api_key": serpapi_api_key,
            "num": max_results,
            "engine": "google",
            "hl": "en",  # Language
        }
        search = GoogleSearch(params)
        serp_results = search.get_dict()

        # Check for error in response
        if "error" in serp_results:
            raise ValueError(f"SerpAPI error: {serp_results['error']}")

        if not serp_results or "organic_results" not in serp_results:
            raise ValueError("No organic results found in SerpAPI response.")

        results = []
        for item in serp_results.get("organic_results", [])[:max_results]:
            results.append(
                {
                    "title": item.get("title"),
                    "href": item.get("link"),
                    "body": item.get("snippet"),
                }
            )
        return results

    elif backend == "tavily":
        if TavilyClient is None:
            raise ImportError("tavily-python is not installed. Install with: pip install tavily-python")
        if not tavily_api_key:
            raise ValueError("tavily_api_key is required for Tavily backend.")
        client = TavilyClient(api_key=tavily_api_key)
        tavily_results = client.search(query=query, search_depth="advanced", max_results=max_results)
        results = []
        for item in tavily_results.get("results", [])[:max_results]:
            results.append(
                {
                    "title": item.get("title"),
                    "href": item.get("url"),
                    "body": item.get("content"),
                }
            )
        return results

    else:
        raise ValueError(f"Unknown backend: {backend}")


def internet_search(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_input = tool.get("input", tool)
    tool_use_id = tool.get("toolUseId", "default_id")
    query = tool_input.get("query")
    max_results = tool_input.get("max_results", 5)
    backend = tool_input.get("backend", "duckduckgo")
    serpapi_api_key = tool_input.get("serpapi_api_key") or kwargs.get("serpapi_api_key")
    tavily_api_key = tool_input.get("tavily_api_key") or kwargs.get("tavily_api_key")

    try:
        results = _search_with_backend(
            backend=backend,
            query=query,
            max_results=max_results,
            serpapi_api_key=serpapi_api_key,
            tavily_api_key=tavily_api_key,
        )
        return {"toolUseId": tool_use_id, "status": "success", "content": [{"json": {"results": results}}]}
    except Exception as e:
        return {"toolUseId": tool_use_id, "status": "error", "content": [{"text": f"Internet search failed: {str(e)}"}]}
