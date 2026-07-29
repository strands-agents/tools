"""Recursive thinking tool for Strands Agent with model switching support.

This module provides functionality for deep analytical thinking through multiple recursive cycles,
enabling sophisticated thought processing, learning, and self-reflection capabilities with support
for different model providers for specialized thinking tasks.

How It Works:
1. The tool processes the initial thought through a specified number of thinking cycles
2. Each cycle uses the output from the previous cycle as a foundation for deeper analysis
3. A specialized system prompt guides the thinking process toward specific expertise domains
4. Each cycle's output is captured and included in the final comprehensive analysis
5. Recursion prevention: the think tool is automatically excluded from nested agents
6. Other tools are available and encouraged for analysis within thinking cycles
7. Optionally uses different model providers for specialized thinking capabilities

Model Selection Process:
1. If model_provider is None: uses the parent agent's model (original behavior)
2. If model_provider is "env": uses environment variables (STRANDS_PROVIDER, etc.)
3. If model_provider is specified: uses that provider with optional custom config
4. Model utilities handle all provider-specific configuration automatically

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import think

agent = Agent(tools=[think])

# Use Bedrock for creative thinking
result = agent.tool.think(
    thought="How can we make AI more creative?",
    cycle_count=3,
    system_prompt="You are a creative AI researcher.",
    model_provider="bedrock",
)

# Use Ollama for local processing
result = agent.tool.think(
    thought="Analyze this code architecture",
    cycle_count=5,
    system_prompt="You are a software architect.",
    model_provider="ollama",
    model_settings={"model_id": "qwen3:4b", "host": "http://localhost:11434"},
)

# Use environment configuration with a custom thinking methodology
os.environ["STRANDS_PROVIDER"] = "anthropic"
os.environ["STRANDS_MODEL_ID"] = "claude-sonnet-4-20250514"
result = agent.tool.think(
    thought="What are the ethical implications?",
    cycle_count=4,
    system_prompt="You are an AI ethics expert.",
    model_provider="env",
    thinking_system_prompt='''Use Socratic questioning method:
    1. Question fundamental assumptions
    2. Explore implications through dialogue
    3. Consider multiple perspectives
    4. Challenge each conclusion with 'but what if...'
    5. Build understanding through systematic inquiry''',
)

# Custom thinking methodology for creative problem solving
result = agent.tool.think(
    thought="How can we revolutionize online education?",
    cycle_count=3,
    system_prompt="You are an innovative education technology expert.",
    thinking_system_prompt='''Apply design thinking methodology:
    1. Empathize: Understand user pain points deeply
    2. Define: Clearly articulate the core problem
    3. Ideate: Generate diverse, unconventional solutions
    4. Prototype: Outline practical implementation steps
    5. Test: Consider potential challenges and iterations''',
)
```

Common Usage Scenarios:
- Creative thinking: use creative models for brainstorming and ideation
- Technical analysis: use analytical models for code review and system design
- Multi-model comparison: compare thinking approaches across different models
- Specialized domains: use domain-specific models (math, creative writing, etc.)
- Cost optimization: use cheaper models for exploratory thinking cycles

Configuration:
When model_provider="env", these environment variables are used:
- STRANDS_PROVIDER: Model provider name
- STRANDS_MODEL_ID: Specific model identifier
- STRANDS_MAX_TOKENS: Maximum tokens to generate
- STRANDS_TEMPERATURE: Sampling temperature
- Provider-specific keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)

Notes:
- Model switching requires the appropriate dependencies (bedrock, anthropic, ollama, etc.)
- When model_provider is None, behavior is identical to the original implementation
- Custom model_settings overrides default environment-based configuration
- Each cycle uses the same model - mixed model cycles are not currently supported
- Model information is logged for transparency and debugging
"""

import logging
import os
import traceback
import uuid
from typing import Any, Dict, List, Optional

from rich.console import Console
from strands import Agent, tool
from strands.telemetry.metrics import metrics_to_string
from typing_extensions import deprecated

from strands_tools.utils import console_util
from strands_tools.utils.models.model import create_model

logger = logging.getLogger(__name__)

_DEPRECATION_MESSAGE = (
    "think is deprecated. This warning becomes an error log in v0.9.0. Migration path: enable native "
    "extended thinking through your model provider's reasoning config instead of a tool. See "
    "https://strandsagents.com/docs/user-guide/concepts/model-providers/amazon-bedrock/"
)


class ThoughtProcessor:
    def __init__(self, tool_context: Dict[str, Any], console: Console):
        self.system_prompt = tool_context.get("system_prompt", "")
        self.messages = tool_context.get("messages", [])
        self.tool_use_id = str(uuid.uuid4())
        self.console = console

    def create_thinking_prompt(
        self,
        thought: str,
        cycle: int,
        total_cycles: int,
        thinking_system_prompt: Optional[str] = None,
    ) -> str:
        """Create a focused prompt for the thinking process with optional custom thinking instructions."""

        # Default thinking instructions
        default_instructions = """
Direct Tasks:
1. Process this thought deeply and analytically
2. Generate clear, structured insights
3. Consider implications and connections
4. Provide actionable conclusions
5. Use other available tools as needed for analysis
"""

        # Use custom thinking instructions if provided, otherwise use defaults
        if thinking_system_prompt:
            thinking_instructions = f"\n{thinking_system_prompt}\n"
        else:
            thinking_instructions = default_instructions

        prompt = f"""{thinking_instructions}
Current Cycle: {cycle}/{total_cycles}

Thought to process:
{thought}

Please provide your analysis directly:
"""
        return prompt.strip()

    def process_cycle(
        self,
        thought: str,
        cycle: int,
        total_cycles: int,
        custom_system_prompt: str,
        specified_tools=None,
        model_provider: Optional[str] = None,
        model_settings: Optional[Dict[str, Any]] = None,
        thinking_system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Process a single thinking cycle with optional model switching and custom thinking instructions."""

        logger.debug(f"🧠 Thinking Cycle {cycle}/{total_cycles}: Processing cycle...")
        self.console.print(f"\n🧠 Thinking Cycle {cycle}/{total_cycles}: Processing cycle...")

        # Create cycle-specific prompt with custom thinking instructions
        prompt = self.create_thinking_prompt(thought, cycle, total_cycles, thinking_system_prompt)

        # Display input prompt
        logger.debug(f"\n--- Input Prompt ---\n{prompt}\n")

        # Get tools and trace attributes from parent agent
        filtered_tools = []
        trace_attributes = {}
        extra_kwargs = {}
        model_info = "Using parent agent's model"

        parent_agent = kwargs.get("agent")
        if parent_agent:
            trace_attributes = parent_agent.trace_attributes
            extra_kwargs["callback_handler"] = parent_agent.callback_handler

            # If specific tools are provided, filter parent tools; otherwise inherit all tools from parent
            if specified_tools is not None:
                # Filter parent agent tools to only include specified tool names
                # ALWAYS exclude 'think' tool to prevent recursion
                for tool_name in specified_tools:
                    if tool_name == "think":
                        logger.warning("Excluding 'think' tool from nested agent to prevent recursion")
                        continue
                    if tool_name in parent_agent.tool_registry.registry:
                        filtered_tools.append(parent_agent.tool_registry.registry[tool_name])
                    else:
                        logger.warning(f"Tool '{tool_name}' not found in parent agent's tool registry")
            else:
                # Inherit all tools from parent EXCEPT the think tool to prevent recursion
                for tool_name, tool_obj in parent_agent.tool_registry.registry.items():
                    if tool_name == "think":
                        logger.debug("Automatically excluding 'think' tool from nested agent to prevent recursion")
                        continue
                    filtered_tools.append(tool_obj)

        # Determine which model to use
        selected_model = None

        if model_provider is None:
            # Use parent agent's model (original behavior)
            selected_model = parent_agent.model if parent_agent else None
            model_info = "Using parent agent's model"

        elif model_provider == "env":
            # Use environment variables to determine model
            try:
                env_provider = os.getenv("STRANDS_PROVIDER", "bedrock")
                selected_model = create_model(provider=env_provider, config=model_settings)
                model_info = f"Using environment model: {env_provider}"
                logger.debug(f"🔄 Created model from environment: {env_provider}")

            except Exception as e:
                logger.warning(f"Failed to create model from environment: {e}")
                logger.debug("Falling back to parent agent's model")
                selected_model = parent_agent.model if parent_agent else None
                model_info = f"Failed to use environment model, using parent's model (Error: {str(e)})"

        else:
            # Use specified model provider
            try:
                selected_model = create_model(provider=model_provider, config=model_settings)
                model_info = f"Using {model_provider} model"
                logger.debug(f"🔄 Created {model_provider} model for thinking cycle")

            except Exception as e:
                logger.warning(f"Failed to create {model_provider} model: {e}")
                logger.debug("Falling back to parent agent's model")
                selected_model = parent_agent.model if parent_agent else None
                model_info = f"Failed to use {model_provider} model, using parent's model (Error: {str(e)})"

        logger.debug(f"--- Model Info ---\n{model_info}\n")

        # Initialize the new Agent with selected model
        agent = Agent(
            model=selected_model,
            messages=[],
            tools=filtered_tools,
            system_prompt=custom_system_prompt,
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
        if result.metrics:
            metrics = result.metrics
            metrics_text = metrics_to_string(metrics)
            logger.debug(metrics_text)

        return assistant_response.strip()


# @deprecated surfaces in IDEs and type checkers; the logger.warning below is what
# users actually see, since DeprecationWarning raised from inside the SDK's tool
# invocation path is suppressed by Python's default warning filter. The message is
# spelled out here rather than passed as _DEPRECATION_MESSAGE because mypy only
# reports @deprecated when the argument is a string literal.
@tool
@deprecated(
    "think is deprecated. This warning becomes an error log in v0.9.0. Migration path: enable native "
    "extended thinking through your model provider's reasoning config instead of a tool. See "
    "https://strandsagents.com/docs/user-guide/concepts/model-providers/amazon-bedrock/"
)
def think(
    thought: str,
    cycle_count: int,
    system_prompt: str,
    tools: Optional[List[str]] = None,
    model_provider: Optional[str] = None,
    model_settings: Optional[Dict[str, Any]] = None,
    thinking_system_prompt: Optional[str] = None,
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """Process a thought through multiple recursive thinking cycles for deep analysis.

    Each cycle builds on the previous cycle's output, producing depth of analysis that is
    difficult to reach in a single pass. Use this for problems that benefit from sustained
    reasoning: architecture decisions, tradeoff analysis, ethical implications, or open-ended
    ideation. The nested agent can call other tools during its cycles, but never itself.

    Args:
        thought: The thought or idea to process. Can be a question, statement, problem
            description, or creative prompt.
        cycle_count: Number of thinking cycles to perform (1-10). More cycles give deeper
            analysis at higher latency and cost; 3-5 is a good default.
        system_prompt: System prompt for the thinking agent. Specifies WHO the agent is -
            its persona, role, and expertise domain. For example, "You are a creative AI
            researcher specializing in educational technology."
        tools: Tool names to make available to the nested agent. Must exist in the parent
            agent's tool registry, e.g. ["calculator", "file_read", "retrieve"]. Defaults to
            inheriting all of the parent agent's tools.
        model_provider: Provider for the thinking cycles. One of "bedrock", "anthropic",
            "litellm", "llamaapi", "ollama", "openai", "github", or "env" to select the
            provider from environment variables. Defaults to the parent agent's model.
        model_settings: Optional model configuration, e.g.
            {"model_id": "claude-sonnet-4-20250514", "params": {"temperature": 1}}.
            Defaults to the provider's standard configuration.
        thinking_system_prompt: Optional instructions controlling HOW the agent thinks, as
            opposed to system_prompt which controls who it is. For example, "Use first
            principles reasoning. Break down complex problems into fundamental components."
        agent: The parent agent (supplied automatically by the framework).

    Returns:
        Dict with "status" ("success" or "error") and "content", a list containing the
        concatenated output of all thinking cycles, or error details on failure.
    """
    logger.warning("DEPRECATION WARNING: %s", _DEPRECATION_MESSAGE)

    console = console_util.create()

    try:
        # Use provided system prompt or fall back to a default
        custom_system_prompt = system_prompt
        if not custom_system_prompt:
            custom_system_prompt = (
                "You are an expert analytical thinker. Process the thought deeply and provide clear insights."
            )

        kwargs = {"agent": agent}
        # Create thought processor instance with the available context
        processor = ThoughtProcessor(kwargs, console)

        # Initialize variables for cycle processing
        current_thought = thought
        all_responses = []

        # Process through each cycle
        for cycle in range(1, cycle_count + 1):
            # Process current cycle
            cycle_kwargs = kwargs.copy()
            if "thought" in cycle_kwargs:
                del cycle_kwargs["thought"]  # Prevent duplicate 'thought' parameter

            cycle_response = processor.process_cycle(
                current_thought,
                cycle,
                cycle_count,
                custom_system_prompt,
                specified_tools=tools,
                model_provider=model_provider,
                model_settings=model_settings,
                thinking_system_prompt=thinking_system_prompt,
                **cycle_kwargs,
            )

            # Store response
            all_responses.append({"cycle": cycle, "thought": current_thought, "response": cycle_response})

            # Update thought for next cycle based on current response
            current_thought = f"Previous cycle concluded: {cycle_response}\nContinue developing these ideas further."

        # Combine all responses into final output
        final_output = "\n\n".join([f"Cycle {r['cycle']}/{cycle_count}:\n{r['response']}" for r in all_responses])

        # Return combined result
        return {
            "status": "success",
            "content": [{"text": final_output}],
        }

    except Exception as e:
        error_msg = f"Error in think tool: {str(e)}\n{traceback.format_exc()}"
        console.print(f"Error in think tool: {str(e)}")
        return {
            "status": "error",
            "content": [{"text": error_msg}],
        }
