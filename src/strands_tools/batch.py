"""
Batch Tool for Parallel Tool Invocation

This tool enables invoking multiple other tools in parallel from a single LLM message response.
It is designed for use with agents that support tool registration and invocation by name.

Example usage:
    import os
    import sys

    from strands import Agent
    from strands_tools import batch, http_request, use_aws

    # Example usage of the batch with http_request and use_aws tools
    agent = Agent(tools=[batch, http_request, use_aws])
    result = agent.tool.batch(
        invocations=[
            {"name": "http_request", "arguments": {"method": "GET", "url": "https://api.ipify.org?format=json"}},
            {
                "name": "use_aws",
                "arguments": {
                    "service_name": "s3",
                    "operation_name": "list_buckets",
                    "parameters": {},
                    "region": "us-east-1",
                    "label": "List S3 Buckets"
                }
            },
        ]
    )
"""

import traceback
from typing import Any, Dict

from strands.types.tools import ToolUse

TOOL_SPEC = {
    "name": "batch",
    "description": "Invoke multiple other tool calls simultaneously",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "invocations": {
                    "type": "array",
                    "description": "The tool calls to invoke",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The name of the tool to invoke"},
                            "arguments": {"type": "object", "description": "The arguments to the tool"},
                        },
                        "required": ["name", "arguments"],
                    },
                }
            },
            "required": ["invocations"],
        }
    },
}


def batch(tool: ToolUse, **kwargs) -> Dict[str, Any]:
    """
    Batch tool for invoking multiple tools in parallel.

    Args:
        tool: The tool object passed by the framework.
        **kwargs: Additional arguments passed by the framework, including 'agent' and 'invocations'.

    Returns:
        Dict with status and a list of results for each invocation.

    Notes:
        - Each invocation should specify the tool name and its arguments.
        - The tool will attempt to call each specified tool function with the provided arguments.
        - If a tool function is not found or an error occurs, it will be captured in the results.
        - This tool is designed to work with agents that support dynamic tool invocation.

    Sammple output:
        {
            "status": "success",
            "results": [
                {"name": "http_request", "status": "success", "result": {...}},
                {"name": "use_aws", "status": "error", "error": "...", "traceback": "..."},
                ...
            ]
        }
    """
    # Retrieve 'agent' and 'invocations' from kwargs
    agent = kwargs.get("agent")
    invocations = kwargs.get("invocations", [])
    results = []
    try:
        for invocation in invocations:
            tool_name = invocation.get("name")
            arguments = invocation.get("arguments", {})
            tool_fn = getattr(agent.tool, tool_name, None)
            if callable(tool_fn):
                try:
                    # Only pass JSON-serializable arguments to the tool
                    result = tool_fn(**arguments)
                    results.append({"name": tool_name, "status": "success", "result": result})
                except Exception as e:
                    results.append(
                        {"name": tool_name, "status": "error", "error": str(e), "traceback": traceback.format_exc()}
                    )
            else:
                results.append(
                    {"name": tool_name, "status": "error", "error": f"Tool '{tool_name}' not found in agent."}
                )
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
