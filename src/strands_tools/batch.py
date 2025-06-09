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

from strands.types.tools import ToolResult, ToolUse

from strands_tools.utils import console_util

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


def batch(tool: ToolUse, **kwargs) -> ToolResult:
    """
    Batch tool for invoking multiple tools in parallel.

    Args:
        tool: Tool use object.
        **kwargs: Additional arguments passed by the framework, including 'agent' and 'invocations'.

    Returns:
        ToolResult with toolUseId, status and a list of results for each invocation.

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
    console = console_util.create()
    tool_use_id = tool["toolUseId"]

    # Retrieve 'agent' and 'invocations' from kwargs
    agent = kwargs.get("agent")
    invocations = kwargs.get("invocations", [])
    results = []
    try:
        if not hasattr(agent, "tool") or agent.tool is None:
            raise AttributeError("Agent does not have a valid 'tool' attribute.")
        for invocation in invocations:
            tool_name = invocation.get("name")
            arguments = invocation.get("arguments", {})
            tool_fn = getattr(agent.tool, tool_name, None)
            if callable(tool_fn):
                try:
                    # Only pass JSON-serializable arguments to the tool
                    result = tool_fn(**arguments)

                    if result["status"] == "success":
                        results.append({"json": {"name": tool_name, "status": "success", "result": result}})
                    else:
                        results.append(
                            {"toolUseId": tool_use_id, "status": "error", "content": [{"text": "Tool missing"}]}
                        )
                except Exception as e:
                    error_msg = f"Error in batch tool: {str(e)}\n{traceback.format_exc()}"
                    console.print(f"Error in batch tool: {str(e)}")
                    results.append({"toolUseId": tool_use_id, "status": "error", "content": [{"text": error_msg}]})
            else:
                results.append(
                    {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [{"text": f"Tool '{tool_name}' not found in agent or tool call failed."}],
                    }
                )
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": results,
        }
    except Exception as e:
        error_msg = f"Error in batch tool: {str(e)}\n{traceback.format_exc()}"
        console.print(f"Error in batch tool: {str(e)}")
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": error_msg}],
        }
