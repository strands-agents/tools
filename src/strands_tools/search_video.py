"""
TwelveLabs video search tool for Strands Agent.

This module provides semantic video search functionality using TwelveLabs' Marengo model,
enabling natural language queries against indexed video content. It searches across both
visual and audio modalities to find relevant video clips or segments.

Key Features:
1. Semantic Search:
   • Natural language queries against video content
   • Multi-modal search (visual and audio)
   • Relevance scoring (0.0-1.0)
   • Confidence-based filtering

2. Advanced Configuration:
   • Grouping by video or clip
   • Confidence thresholds (high, medium, low, none)
   • Custom result limits
   • Index selection

3. Response Format:
   • Sorted by relevance score
   • Includes timestamps
   • Video IDs for reference
   • Confidence levels

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import search_video

agent = Agent(tools=[search_video])

# Basic search
results = agent.tool.search_video(query="people discussing AI")

# Advanced search with custom parameters
results = agent.tool.search_video(
    query="product demo presentation",
    index_id="your-index-id",
    group_by="video",
    threshold="high",
    page_limit=5
)
```

See the search_video function docstring for more details on available parameters.
"""

import os
from typing import Any, List

from strands.types.tools import ToolResult, ToolUse
from twelvelabs import TwelveLabs
from twelvelabs.models.search import SearchData

TOOL_SPEC = {
    "name": "search_video",
    "description": """Searches video content using TwelveLabs' semantic search capabilities.

Key Features:
1. Semantic Search:
   - Natural language queries against video content
   - Multi-modal search (visual and audio)
   - Relevance scoring (0.0-1.0)
   - Confidence-based filtering
   
2. Advanced Configuration:
   - Group results by video or clip
   - Set confidence thresholds
   - Control result limits
   - Choose search modalities

3. Response Format:
   - Sorted by relevance score
   - Includes timestamps
   - Video IDs for reference
   - Confidence levels

4. Example Response:
   When grouped by clip:
   {
     "score": 0.85,
     "start": 120.5,
     "end": 145.3,
     "confidence": "high",
     "video_id": "video_123"
   }
   
   When grouped by video:
   {
     "video_id": "video_123",
     "clips": [
       {"score": 0.85, "start": 120.5, "end": 145.3, "confidence": "high"},
       {"score": 0.72, "start": 200.0, "end": 215.7, "confidence": "medium"}
     ]
   }

Usage Examples:
1. Basic search:
   search_video(query="people discussing technology")

2. Search specific index:
   search_video(query="product features", index_id="your-index-id")

3. High confidence results only:
   search_video(query="keynote presentation", threshold="high")

4. Group by video:
   search_video(query="tutorial steps", group_by="video")

5. Audio-only search:
   search_video(query="mentioned pricing", search_options=["audio"])""",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query for video content",
                },
                "index_id": {
                    "type": "string",
                    "description": (
                        "TwelveLabs index ID to search. Uses TWELVELABS_MARENGO_INDEX_ID env var if not provided"
                    ),
                },
                "search_options": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["visual", "audio"],
                    },
                    "description": "Search modalities to use. Default: ['visual', 'audio']",
                },
                "group_by": {
                    "type": "string",
                    "enum": ["video", "clip"],
                    "description": (
                        "How to group results. 'clip' returns individual segments, "
                        "'video' groups clips by video. Default: 'clip'"
                    ),
                },
                "threshold": {
                    "type": "string",
                    "enum": ["high", "medium", "low", "none"],
                    "description": "Minimum confidence threshold for results. Default: 'medium'",
                },
                "page_limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 10",
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            "required": ["query"],
        }
    },
}


def format_search_results(results: List[SearchData], group_by: str, total_count: int) -> str:
    """
    Format TwelveLabs search results for display.

    Args:
        results: List of search results from TwelveLabs
        group_by: How results are grouped ('video' or 'clip')
        total_count: Total number of results found

    Returns:
        Formatted string containing search results
    """
    if not results:
        return "No results found matching the search criteria."

    formatted = [f"Found {total_count} total results\n"]

    if group_by == "video":
        # Video-grouped results
        for i, video in enumerate(results, 1):
            formatted.append(f"\n{i}. Video ID: {video.id}")
            if hasattr(video, "clips") and video.clips:
                formatted.append(f"   Found {len(video.clips)} clips:")
                for j, clip in enumerate(video.clips[:3], 1):  # Show top 3 clips per video
                    formatted.append(
                        f"   {j}. Score: {clip.score:.3f} | "
                        f"{clip.start:.1f}s-{clip.end:.1f}s | "
                        f"Confidence: {clip.confidence}"
                    )
                if len(video.clips) > 3:
                    formatted.append(f"   ... and {len(video.clips) - 3} more clips")
    else:
        # Clip-level results
        for i, clip in enumerate(results, 1):
            formatted.append(f"\n{i}. Video: {clip.video_id}")
            formatted.append(f"   Score: {clip.score:.3f}")
            formatted.append(f"   Time: {clip.start:.1f}s - {clip.end:.1f}s")
            formatted.append(f"   Confidence: {clip.confidence}")

    return "\n".join(formatted)


def search_video(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Search video content using TwelveLabs semantic search.

    This tool enables semantic search across video content indexed in TwelveLabs,
    supporting both visual and audio modalities. It returns relevant video segments
    or entire videos based on natural language queries.

    How It Works:
    ------------
    1. Your query is sent to TwelveLabs' Marengo search model
    2. The model searches across visual and/or audio content in the indexed videos
    3. Results are scored by relevance (0.0-1.0) and filtered by confidence
    4. Results can be grouped by individual clips or by video
    5. Formatted results include timestamps and confidence levels

    Common Usage Scenarios:
    ---------------------
    - Finding specific moments in recorded meetings or presentations
    - Locating product demonstrations in marketing videos
    - Searching for mentions of topics in educational content
    - Identifying scenes or actions in surveillance footage
    - Finding spoken keywords in podcasts or interviews

    Args:
        tool: Tool use information containing input parameters:
            query: Natural language search query
            index_id: TwelveLabs index to search (default: from TWELVELABS_MARENGO_INDEX_ID env var)
            search_options: Modalities to search ['visual', 'audio'] (default: both)
            group_by: Group results by 'clip' or 'video' (default: 'clip')
            threshold: Confidence threshold 'high', 'medium', 'low', 'none' (default: 'medium')
            page_limit: Maximum results to return (default: 10)

    Returns:
        Dictionary containing status and search results:
        {
            "toolUseId": "unique_id",
            "status": "success|error",
            "content": [{"text": "Search results or error message"}]
        }

        Success: Returns formatted video search results with scores and timestamps
        Error: Returns information about what went wrong

    Notes:
        - Requires TWELVELABS_API_KEY environment variable
        - Index ID can be set via TWELVELABS_MARENGO_INDEX_ID environment variable
        - Visual search finds objects, actions, and scenes
        - Audio search finds spoken words and sounds
        - Results are sorted by relevance score
    """
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    try:
        # Get API key
        api_key = os.getenv("TWELVELABS_API_KEY")
        if not api_key:
            raise ValueError(
                "TWELVELABS_API_KEY environment variable not set. Please set it to your TwelveLabs API key."
            )

        # Extract parameters
        query = tool_input["query"]
        index_id = tool_input.get("index_id") or os.getenv("TWELVELABS_MARENGO_INDEX_ID")

        if not index_id:
            raise ValueError(
                "No index_id provided and TWELVELABS_MARENGO_INDEX_ID environment variable not set. "
                "Please provide an index_id or set the environment variable."
            )

        search_options = tool_input.get("search_options", ["visual", "audio"])
        group_by = tool_input.get("group_by", "clip")
        threshold = tool_input.get("threshold", "medium")
        page_limit = tool_input.get("page_limit", 10)

        # Initialize TwelveLabs client and perform search
        with TwelveLabs(api_key) as client:
            search_result = client.search.query(
                index_id=index_id,
                query_text=query,
                options=search_options,
                group_by=group_by,
                threshold=threshold,
                page_limit=page_limit,
            )

            # Get total count
            total_count = 0
            if hasattr(search_result, "pool") and hasattr(search_result.pool, "total_count"):
                total_count = search_result.pool.total_count

            # Format results
            results_list = list(search_result.data) if hasattr(search_result, "data") else []
            formatted_results = format_search_results(results_list, group_by, total_count)

            # Build response summary
            summary_parts = [
                f'Video Search Results for: "{query}"',
                f"Index: {index_id}",
                f"Search options: {', '.join(search_options)}",
                f"Confidence threshold: {threshold}",
                "",
                formatted_results,
            ]

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": "\n".join(summary_parts)}],
            }

    except Exception as e:
        error_message = f"Error searching videos: {e!s}"

        # Add helpful context for common errors
        if "api_key" in str(e).lower():
            error_message += "\n\nMake sure TWELVELABS_API_KEY environment variable is set correctly."
        elif "index" in str(e).lower():
            error_message += "\n\nMake sure the index_id is valid and you have access to it."
        elif "throttl" in str(e).lower() or "rate" in str(e).lower():
            error_message += "\n\nAPI rate limit exceeded. Please try again later."

        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": error_message}],
        }
