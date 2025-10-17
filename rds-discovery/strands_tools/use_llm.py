"""
Dynamic LLM instance creation for Strands Agent.

This module provides functionality to start new AI event loops with specified prompts,
allowing you to create isolated agent instances for specific tasks or use cases.
Each invocation creates a fresh agent with its own context and state.

Strands automatically handles the lifecycle of these nested agent instances,
making them powerful for delegation, specialized processing, or isolated computation.

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import use_llm

agent = Agent(tools=[use_llm])

# Basic usage with just a prompt and system prompt (inherits all parent tools)
result = agent.tool.use_llm(
    prompt="Tell me about the advantages of tool-building in AI agents",
    system_prompt="You are a helpful AI assistant specializing in AI development concepts."
)

# Usage with specific tools filtered from parent agent
result = agent.tool.use_llm(
    prompt="Calculate 2 + 2 and retrieve some information",
    system_prompt="You are a helpful assistant.",
    tools=["calculator", "retrieve"]
)

# Usage with mixed tool filtering from parent agent
result = agent.tool.use_llm(
    prompt="Analyze this data file",
    system_prompt="You are a data analyst.",
    tools=["file_read", "calculator", "python_repl"]
)

# The response is available in the returned object
print(result["content"][0]["text"])  # Prints the response text
```

See the use_llm function docstring for more details on configuration options and parameters.
"""

import logging
from typing import Any

from strands import Agent
from strands.telemetry.metrics import metrics_to_string
from strands.types.tools import ToolResult, ToolUse

logger = logging.getLogger(__name__)

TOOL_SPEC = {
    "name": "use_llm",
    "description": "Start a new AI event loop with a specified prompt",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "What should this AI event loop do?",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "System prompt for the new event loop",
                },
                "tools": {
                    "type": "array",
                    "description": "List of tool names to make available to the nested agent"
                    + "Tool names must exist in the parent agent's tool registry."
                    + "If not provided, inherits all tools from parent agent.",
                    "items": {"type": "string"},
                },
            },
            "required": ["prompt", "system_prompt"],
        }
    },
}


def use_llm(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Create a new LLM instance using the Agent interface.

    This function creates a new Strands Agent instance with the provided system prompt,
    runs it with the specified prompt, and returns the response with performance metrics.
    It allows for isolated processing in a fresh context separate from the main agent.

    How It Works:
    ------------
    1. The function initializes a new Agent instance with the provided system prompt
    2. The agent processes the given prompt in its own isolated context
    3. The response and metrics are captured and returned in a structured format
    4. The new agent instance exists only for the duration of this function call

    Agent Creation Process:
    ---------------------
    - A fresh Agent object is created with an empty message history
    - The provided system prompt configures the agent's behavior and capabilities
    - The agent processes the prompt in its own isolated context
    - Response and metrics are captured for return to the caller
    - The parent agent's callback_handler is used if one is not specified

    Common Use Cases:
    ---------------
    - Task delegation: Creating specialized agents for specific subtasks
    - Context isolation: Processing prompts in a clean context without history
    - Multi-agent systems: Creating multiple agents with different specializations
    - Learning and reasoning: Using nested agents for complex reasoning chains

    Args:
        tool (ToolUse): Tool use object containing the following:
            - prompt (str): The prompt to process with the new agent instance
            - system_prompt (str): Custom system prompt for the agent
            - tools (List[str], optional): List of tool names to make available to the nested agent.
                Tool names must exist in the parent agent's tool registry.
                Examples: ["calculator", "file_read", "retrieve"]
                If not provided, inherits all tools from the parent agent.
        **kwargs (Any): Additional keyword arguments

    Returns:
        ToolResult: Dictionary containing status and response content in the format:
        {
            "toolUseId": "unique-tool-use-id",
            "status": "success",
            "content": [
                {"text": "Response: The response text from the agent"},
                {"text": "Metrics: Performance metrics information"}
            ]
        }

    Notes:
        - The agent instance is temporary and will be garbage-collected after use
        - The agent(prompt) call is synchronous and will block until completion
        - Performance metrics include token usage and processing latency information
    """
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    logger.warning(
        "DEPRECATION WARNING: use_llm will be removed in the next major release. "
        "Migration path: replace use_llm calls with use_agent for equivalent functionality."
    )

    prompt = tool_input["prompt"]
    tool_system_prompt = tool_input.get("system_prompt")
    specified_tools = tool_input.get("tools")

    tools = []
    trace_attributes = {}

    extra_kwargs = {}
    parent_agent = kwargs.get("agent")
    if parent_agent:
        trace_attributes = parent_agent.trace_attributes
        extra_kwargs["callback_handler"] = parent_agent.callback_handler

        # If specific tools are provided, filter parent tools; otherwise inherit all tools from parent
        if specified_tools is not None:
            # Filter parent agent tools to only include specified tool names
            filtered_tools = []
            for tool_name in specified_tools:
                if tool_name in parent_agent.tool_registry.registry:
                    filtered_tools.append(parent_agent.tool_registry.registry[tool_name])
                else:
                    logger.warning(f"Tool '{tool_name}' not found in parent agent's tool registry")
            tools = filtered_tools
        else:
            tools = list(parent_agent.tool_registry.registry.values())

    if "callback_handler" in kwargs:
        extra_kwargs["callback_handler"] = kwargs["callback_handler"]

    # Display input prompt
    logger.debug(f"\n--- Input Prompt ---\n{prompt}\n")

    # Visual indicator for new LLM instance
    logger.debug("🔄 Creating new LLM instance...")

    # Initialize the new Agent with provided parameters
    agent = Agent(
        messages=[],
        tools=tools,
        system_prompt=tool_system_prompt,
        trace_attributes=trace_attributes,
        **extra_kwargs,
    )
    # Run the agent with the provided prompt
    result = agent(prompt)

    # Extract response
    assistant_response = str(result)

    # Display assistant response
    logger.debug(f"\n--- Assistant Response ---\n{assistant_response.strip()}\n")

    # Print metrics if available
    metrics_text = ""
    if result.metrics:
        metrics = result.metrics
        metrics_text = metrics_to_string(metrics)
        logger.debug(metrics_text)

    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [
            {"text": f"Response: {assistant_response}"},
            {"text": f"Metrics: {metrics_text}"},
        ],
    }
