"""Recursive thinking tool for Strands Agent with model switching support.

This module provides functionality for deep analytical thinking through multiple recursive cycles,
enabling sophisticated thought processing, learning, and self-reflection capabilities with support
for different model providers for specialized thinking tasks.
"""

import logging
import os
import traceback
import uuid
from typing import Any, Dict, List, Optional

from rich.console import Console
from strands import Agent, tool
from strands.telemetry.metrics import metrics_to_string

from strands_tools.utils import console_util
from strands_tools.utils.models.model import create_model

logger = logging.getLogger(__name__)


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


@tool
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
    """Recursive thinking tool with model switching support for sophisticated thought generation.

    This tool implements a multi-cycle cognitive analysis approach that progressively refines thoughts
    through iterative processing, with the ability to use different model providers for specialized
    thinking tasks. Each cycle builds upon insights from the previous cycle, creating a depth of
    analysis that would be difficult to achieve in a single pass.

    How It Works:
    ------------
    1. The tool processes the initial thought through a specified number of thinking cycles
    2. Each cycle uses the output from the previous cycle as a foundation for deeper analysis
    3. A specialized system prompt guides the thinking process toward specific expertise domains
    4. Each cycle's output is captured and included in the final comprehensive analysis
    5. Recursion prevention: The think tool is automatically excluded from nested agents
    6. Other tools are available and encouraged for analysis within thinking cycles
    7. Optionally uses different model providers for specialized thinking capabilities

    Model Selection Process:
    ----------------------
    1. If model_provider is None: Uses parent agent's model (original behavior)
    2. If model_provider is "env": Uses environment variables (STRANDS_PROVIDER, etc.)
    3. If model_provider is specified: Uses that provider with optional custom config
    4. Model utilities handle all provider-specific configuration automatically

    System Prompt vs Thinking System Prompt:
    --------------------------------------
    - **system_prompt**: Controls the agent's persona, role, and expertise domain
      Example: "You are a creative AI researcher specializing in educational technology."

    - **thinking_system_prompt**: Controls the thinking methodology and approach
      Example: "Use design thinking: empathize, define, ideate, prototype, test."

    Together they provide: WHO the agent is (system_prompt) + HOW it thinks (thinking_system_prompt)

    Common Usage Scenarios:
    ---------------------
    - Creative thinking: Use creative models for brainstorming and ideation
    - Technical analysis: Use analytical models for code review and system design
    - Multi-model comparison: Compare thinking approaches across different models
    - Specialized domains: Use domain-specific models (math, creative writing, etc.)
    - Cost optimization: Use cheaper models for exploratory thinking cycles

    Args:
        thought: The detailed thought or idea to process through multiple thinking cycles.
            This can be a question, statement, problem description, or creative prompt.
        cycle_count: Number of thinking cycles to perform (1-10). More cycles allow for
            deeper analysis but require more time and resources. Typically 3-5 cycles
            provide a good balance of depth and efficiency.
        system_prompt: Custom system prompt to use for the LLM thinking process. This should
            specify the expertise domain and thinking approach for processing the thought.
        tools: List of tool names to make available to the nested agent. Tool names must
            exist in the parent agent's tool registry. Examples: ["calculator", "file_read", "retrieve"]
            If not provided, inherits all tools from the parent agent.
        model_provider: Model provider to use for the thinking cycles.
            Options: "bedrock", "anthropic", "litellm", "llamaapi", "ollama", "openai", "github"
            Special values:
            - None: Use parent agent's model (default, preserves original behavior)
            - "env": Use environment variables to determine provider
            Examples: "bedrock", "anthropic", "litellm", "env"
        model_settings: Optional custom configuration for the model.
            If not provided, uses default configuration for the provider.
            Example: {"model_id": "claude-sonnet-4-20250514", "params": {"temperature": 1}}
        thinking_system_prompt: Optional custom thinking instructions that override the default
            thinking methodology. This controls HOW the agent thinks about the problem, separate
            from the system_prompt which controls the agent's persona/role.
            Example: "Use first principles reasoning. Break down complex problems into fundamental
            components. Question assumptions at each step."
        agent: The parent agent (automatically passed by Strands framework)

    Returns:
        Dict containing status and response content in the format:
        {
            "status": "success|error",
            "content": [{"text": "Detailed thinking output across all cycles"}]
        }

        Success case: Returns concatenated results from all thinking cycles
        Error case: Returns information about what went wrong during processing

    Environment Variables for Model Switching:
    ----------------------------------------
    When model_provider="env", these variables are used:
    - STRANDS_PROVIDER: Model provider name
    - STRANDS_MODEL_ID: Specific model identifier
    - STRANDS_MAX_TOKENS: Maximum tokens to generate
    - STRANDS_TEMPERATURE: Sampling temperature
    - Provider-specific keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)

    Examples:
    --------
    # Use Bedrock for creative thinking
    result = agent.tool.think(
        thought="How can we make AI more creative?",
        cycle_count=3,
        system_prompt="You are a creative AI researcher.",
        model_provider="bedrock"
    )

    # Use Ollama for local processing
    result = agent.tool.think(
        thought="Analyze this code architecture",
        cycle_count=5,
        system_prompt="You are a software architect.",
        model_provider="ollama",
        model_settings={"model_id": "qwen3:4b", "host": "http://localhost:11434"}
    )

    # Use environment configuration with custom thinking methodology
    os.environ["STRANDS_PROVIDER"] = "anthropic"
    os.environ["STRANDS_MODEL_ID"] = "claude-sonnet-4-20250514"
    result = agent.tool.think(
        thought="What are the ethical implications?",
        cycle_count=4,
        system_prompt="You are an AI ethics expert.",
        model_provider="env",
        thinking_system_prompt=Use Socratic questioning method:
        1. Question fundamental assumptions
        2. Explore implications through dialogue
        3. Consider multiple perspectives
        4. Challenge each conclusion with 'but what if...'
        5. Build understanding through systematic inquiry
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
        5. Test: Consider potential challenges and iterations'''
    )

    Notes:
        - Model switching requires the appropriate dependencies (bedrock, anthropic, ollama, etc.)
        - When model_provider is None, behavior is identical to the original implementation
        - Custom model_settings overrides default environment-based configuration
        - Each cycle uses the same model - mixed model cycles not currently supported
        - Model information is logged for transparency and debugging
    """
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
