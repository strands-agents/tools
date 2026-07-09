"""Dynamic Agent instance creation for Strands Agent with model switching support.

This module provides functionality to start new AI event loops with specified prompts
and optionally different model providers, allowing you to create isolated agent instances
for specific tasks or use cases with different AI models.

Each invocation creates a fresh agent with its own context and state, and can use
a different model provider than the parent agent.

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import use_agent

agent = Agent(tools=[use_agent])

# Basic usage with inherited model (original behavior)
result = agent.tool.use_agent(
    prompt="Tell me about the advantages of tool-building in AI agents",
    system_prompt="You are a helpful AI assistant specializing in AI development concepts."
)

# Usage with different model provider
result = agent.tool.use_agent(
    prompt="Calculate 2 + 2 and explain the result",
    system_prompt="You are a helpful math assistant.",
    model_provider="bedrock",  # Switch to Bedrock instead of parent's model
    model_settings={
      "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"
    },
    tools=["calculator"]
)

# Usage with custom model configuration
result = agent.tool.use_agent(
    prompt="Write a creative story",
    system_prompt="You are a creative writing assistant.",
    model_provider="github",
    model_settings={
        "model_id": "openai/o4-mini",
        "params": {"temperature": 1, "max_tokens": 4000}
    }
)

# Environment-based model switching
import os
os.environ["STRANDS_PROVIDER"] = "ollama"
os.environ["STRANDS_MODEL_ID"] = "qwen3:4b"
result = agent.tool.use_agent(
    prompt="Analyze this code",
    system_prompt="You are a code review assistant.",
    model_provider="env"  # Use environment variables
)
```

See the use_agent function docstring for more details on configuration options and parameters.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from strands import Agent, tool
from strands.telemetry.metrics import metrics_to_string

from strands_tools.utils.models.model import create_model

logger = logging.getLogger(__name__)


@tool
def use_agent(
    prompt: str,
    system_prompt: str,
    tools: Optional[List[str]] = None,
    model_provider: Optional[str] = None,
    model_settings: Optional[Dict[str, Any]] = None,
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """Start a new AI event loop with a specified prompt and optionally different model.

    This function creates a new Strands Agent instance with the provided system prompt,
    optionally using a different model provider than the parent agent, runs it with the
    specified prompt, and returns the response with performance metrics.

    How It Works:
    ------------
    1. Determines which model to use (parent's model, specified provider, or environment)
    2. Creates a new Agent instance with the model and system prompt
    3. The agent processes the given prompt in its own isolated context
    4. The response and metrics are captured and returned in a structured format
    5. The new agent instance exists only for the duration of this function call

    Model Selection Process:
    ----------------------
    1. If model_provider is None: Uses parent agent's model (original behavior)
    2. If model_provider is "env": Uses environment variables (STRANDS_PROVIDER, etc.)
    3. If model_provider is specified: Uses that provider with optional custom config
    4. Model utilities handle all provider-specific configuration automatically

    Common Use Cases:
    ---------------
    - Multi-model workflows: Use different models for different tasks
    - Model comparison: Compare responses from different providers
    - Cost optimization: Use cheaper models for simple tasks
    - Specialized models: Use domain-specific models (code, math, creative)
    - Fallback strategies: Switch to alternative models if primary fails

    Args:
        prompt: The prompt to process with the new agent instance.
        system_prompt: Custom system prompt for the agent.
        tools: List of tool names to make available to the nested agent.
            Tool names must exist in the parent agent's tool registry.
            Examples: ["calculator", "file_read", "retrieve"]
            If not provided, inherits all tools from the parent agent.
        model_provider: Model provider to use for the nested agent.
            Options: "bedrock", "anthropic", "litellm", "llamaapi", "ollama", "openai", "github"
            Special values:
            - None: Use parent agent's model (default, preserves original behavior)
            - "env": Use environment variables to determine provider
            Examples: "bedrock", anthropic", "litellm", "env"
        model_settings: Optional custom configuration for the model.
            If not provided, uses default configuration for the provider.
            Example: {"model_id": "claude-sonnet-4-20250514", "params": {"temperature": 1}}
        agent: The parent agent (automatically passed by Strands framework).

    Returns:
        Dict containing status and response content in the format:
        {
            "status": "success|error",
            "content": [
                {"text": "Response: The response text from the agent"},
                {"text": "Model: Information about the model used"},
                {"text": "Metrics: Performance metrics information"}
            ]
        }

        Success case: Returns the agent response with model info and performance metrics
        Error case: Returns information about what went wrong during processing

    Environment Variables for Model Switching:
    ----------------------------------------
    When model_provider="env", these variables are used:
    - STRANDS_PROVIDER: Model provider name
    - STRANDS_MODEL_ID: Specific model identifier, example;
        "us.anthropic.claude-sonnet-4-20250514-v1:0" for bedrock provider
    - STRANDS_MAX_TOKENS: Maximum tokens to generate
    - STRANDS_TEMPERATURE: Sampling temperature
    - Provider-specific keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)

    Examples:
    --------
    # Use Bedrock for creative tasks
    result = agent.tool.use_agent(
        prompt="Write a poem about AI",
        system_prompt="You are a creative poet.",
        model_provider="bedrock"
    )

    # Use Ollama for local processing
    result = agent.tool.use_agent(
        prompt="Summarize this text",
        system_prompt="You are a summarization assistant.",
        model_provider="ollama",
        model_settings={"host": "http://localhost:11434", "model_id": "qwen3:4b"}
    )

    # Use environment configuration
    os.environ["STRANDS_PROVIDER"] = "litellm"
    os.environ["STRANDS_MODEL_ID"] = "openai/gpt-4o"
    result = agent.tool.use_agent(
        prompt="Analyze this data",
        system_prompt="You are a data analyst.",
        model_provider="env"
    )

    Notes:
        - Model switching requires the appropriate dependencies (bedrock, anthropic, ollama, llamaapi, litellm, etc.)
        - When model_provider is None, behavior is identical to the original implementation
        - Custom model_settings overrides default environment-based configuration
        - Performance metrics include token usage for the specific model used
        - Model information is included in the response for transparency
    """
    try:
        # Get tools and trace attributes from parent agent
        filtered_tools = []
        trace_attributes = {}
        extra_kwargs = {}
        model_info = "Using parent agent's model"

        if agent:
            trace_attributes = agent.trace_attributes
            extra_kwargs["callback_handler"] = agent.callback_handler

            # If specific tools are provided, filter parent tools; otherwise inherit all tools from parent
            if tools is not None:
                # Filter parent agent tools to only include specified tool names
                for tool_name in tools:
                    if tool_name in agent.tool_registry.registry:
                        filtered_tools.append(agent.tool_registry.registry[tool_name])
                    else:
                        logger.warning(f"Tool '{tool_name}' not found in parent agent's tool registry")
            else:
                filtered_tools = list(agent.tool_registry.registry.values())

        # Determine which model to use
        selected_model = None

        if model_provider is None:
            # Use parent agent's model (original behavior)
            selected_model = agent.model if agent else None
            model_info = "Using parent agent's model"

        elif model_provider == "env":
            # Use environment variables to determine model
            try:
                env_provider = os.getenv("STRANDS_PROVIDER", "ollama")
                selected_model = create_model(provider=env_provider, config=model_settings)
                model_info = f"Using environment model: {env_provider}"
                logger.debug(f"🔄 Created model from environment: {env_provider}")

            except Exception as e:
                logger.warning(f"Failed to create model from environment: {e}")
                logger.debug("Falling back to parent agent's model")
                selected_model = agent.model if agent else None
                model_info = f"Failed to use environment model, using parent's model (Error: {str(e)})"

        else:
            # Use specified model provider
            try:
                selected_model = create_model(provider=model_provider, config=model_settings)
                model_info = f"Using {model_provider} model"
                logger.debug(f"🔄 Created {model_provider} model for nested agent")

            except Exception as e:
                logger.warning(f"Failed to create {model_provider} model: {e}")
                logger.debug("Falling back to parent agent's model")
                selected_model = agent.model if agent else None
                model_info = f"Failed to use {model_provider} model, using parent's model (Error: {str(e)})"

        # Display input prompt
        logger.debug(f"\n--- Input Prompt ---\n{prompt}\n")
        logger.debug(f"--- Model Info ---\n{model_info}\n")

        # Visual indicator for new LLM instance
        logger.debug("🔄 Creating new LLM instance...")

        # Initialize the new Agent with selected model
        new_agent = Agent(
            model=selected_model,
            messages=[],
            tools=filtered_tools,
            system_prompt=system_prompt,
            trace_attributes=trace_attributes,
            **extra_kwargs,
        )

        # Run the agent with the provided prompt
        result = new_agent(prompt)

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
            "status": "success",
            "content": [
                {"text": f"Response: {assistant_response}"},
                {"text": f"Model: {model_info}"},
                {"text": f"Metrics: {metrics_text}"},
            ],
        }

    except Exception as e:
        error_msg = f"Error in use_agent tool: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "content": [{"text": error_msg}],
        }
