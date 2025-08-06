from strands import Agent
from strands_tools.tavily import (
    tavily_map, tavily_crawl, tavily_extract, tavily_search,
    tavily_map_async, tavily_crawl_async, tavily_extract_async, tavily_search_async
)
import asyncio

agent = Agent(tools=[tavily_map, tavily_search, tavily_extract, tavily_crawl])

from strands.types.tools import ToolResult, ToolResultContent, ToolUse

def test_sync_tools():
    # Basic search
    result = agent.tool.tavily_search(
        query="What is AI?",
        include_answer=True,
        include_images=True,
        include_favicon=True,
        include_raw_content=True,
        max_results=10
    )

    result = agent.tool.tavily_extract(
        urls=["https://www.tavily.com"],
        include_images=True,
        include_favicon=True,
        extract_depth="basic"
    )

    # breakpoint()
    result = agent("Extract the content of https://www.tavily.com")

    # Advanced search with features
    result = agent.tool.tavily_search(
        query="Latest AI developments", 
        search_depth="advanced",
        topic="news",
        include_answer=True,
        include_images=True,
        max_results=10
    )

    # Advanced extract with features
    result = agent.tool.tavily_extract(
        urls=["https://www.tavily.com"]
    )

    result = agent.tool.tavily_crawl(url="www.tavily.com")
    # print(result)
    # # result = agent.tool.tavily_map(url="www.tavily.com")

    response = agent("What is tavily.com")

# Async examples
async def test_async_tools():
    # 1. Basic async search
    result = await tavily_search_async(
        query="What is AI?",
        include_answer=True,
        max_results=5
    )
    
    # 2. Async extract
    result = await tavily_extract_async(
        urls=["https://www.tavily.com"],
        extract_depth="basic"
    )
    
    # 3. Async crawl
    result = await tavily_crawl_async(
        url="www.tavily.com",
        max_depth=2,
        limit=10
    )
    
    # 4. Async map
    result = await tavily_map_async(
        url="www.tavily.com",
        limit=10
    )

# Run async examples
# asyncio.run(test_async_tools())


agent("Do deep research on the company BMO ")