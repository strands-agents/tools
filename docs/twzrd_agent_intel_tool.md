# TWZRD Agent Intel Tool

TWZRD Agent Intel provides Solana-native trust scoring for AI agent wallets. Use it with the Strands `mcp_client` tool to score Solana agents before authorizing x402 micropayments.

## MCP Server

- **URL**: `https://intel.twzrd.xyz/mcp`
- **Transport**: Streamable HTTP (zero-install)
- **Auth**: None required for free tools

## Available Tools

| Tool | Description | Cost |
|------|-------------|------|
| `score_agent` | On-chain trust score (0–1) for a Solana wallet | Free |
| `preflight_check` | Full pre-payment due diligence | Free |
| `get_trust_receipt` | Signed trust receipt via HTTP 402 | Paid (USDC) |

## Usage with Strands Agents

### Using `mcp_client` (dynamic, runtime connection)

```python
from strands import Agent
from strands_tools import mcp_client

agent = Agent(tools=[mcp_client])

# Score a Solana agent wallet before an x402 payment
result = agent.tool.mcp_client_connect_and_call(
    server_url="https://intel.twzrd.xyz/mcp",
    transport="streamable_http",
    tool_name="score_agent",
    tool_args={"wallet": "D1QkbFJKiPsymJ65RKHhF6DFB8sPMfpBaFBzuHKfJGWi"},
)
print(result)  # {"score": 0.85, "payments": 48, ...}
```

### Using Strands SDK MCPClient (static configuration)

For persistent connections, use the Strands SDK's MCPClient directly:

```python
import asyncio
from strands import Agent
from strands.tools.mcp import MCPClient

async def main():
    # Connect to TWZRD Agent Intel MCP server
    async with MCPClient(
        lambda: (
            __import__("mcp.client.streamable_http", fromlist=["streamablehttp_client"])
            .streamablehttp_client("https://intel.twzrd.xyz/mcp")
        )
    ) as client:
        tools = await client.list_tools_sync()
        agent = Agent(tools=tools)

        result = await agent.ainvoke_tool(
            "score_agent",
            {"wallet": "D1QkbFJKiPsymJ65RKHhF6DFB8sPMfpBaFBzuHKfJGWi"}
        )
        print(result)

asyncio.run(main())
```

### Full Example: Trust-Gated x402 Payment

```python
"""
Trust-gated x402 payment using Strands + TWZRD Agent Intel.

Before paying an agent via x402, score it. Only pay if score >= 0.5.
"""

from strands import Agent
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client


def create_twzrd_client():
    return streamablehttp_client("https://intel.twzrd.xyz/mcp")


async def should_pay_agent(wallet: str, min_score: float = 0.5) -> bool:
    """Return True if the agent wallet meets the trust threshold."""
    async with MCPClient(create_twzrd_client) as client:
        tools = await client.list_tools_sync()
        agent = Agent(tools=tools)

        result = await agent.ainvoke_tool("score_agent", {"wallet": wallet})
        score = float(result.get("score", 0))
        return score >= min_score


# Usage:
# if await should_pay_agent("D1QkbFJKiPsymJ65RKHhF6DFB8sPMfpBaFBzuHKfJGWi"):
#     # proceed with x402 payment
```

## Configuration

No API key required. The MCP server is publicly accessible.

**MCP Config** (for use in Claude Desktop, Claude Code, or any MCP client):

```json
{
  "mcpServers": {
    "twzrd-agent-intel": {
      "url": "https://intel.twzrd.xyz/mcp"
    }
  }
}
```

## Links

- Website: https://intel.twzrd.xyz
- x402 Protocol: https://x402.org
