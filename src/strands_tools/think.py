"""
Advanced recursive thinking tool for Strands Agent.

This module provides functionality for deep analytical thinking through multiple recursive cycles,
enabling sophisticated thought processing, learning, and self-reflection capabilities.
The tool processes thoughts through sequential cycles, each building upon the previous,
to generate progressively refined insights.

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import think, stop

agent = Agent(tools=[think, stop])

# Basic usage with default system prompt
result = agent.tool.think(
    thought="How might we improve renewable energy storage solutions?",
    cycle_count=3,
    system_prompt="You are an expert energy systems analyst."
)

# Advanced usage with custom system prompt
result = agent.tool.think(
    thought="Analyze the implications of quantum computing on cryptography.",
    cycle_count=5,
    system_prompt="You are a specialist in quantum computing and cryptography. Analyze this topic deeply,
    considering both technical and practical aspects."
)
```

See the think function docstring for more details on configuration options and parameters.
"""

import logging
import traceback
import uuid
from typing import Any, Dict

from rich.console import Console
from strands import Agent, tool
from strands.telemetry.metrics import metrics_to_string

from strands_tools.utils import console_util

logger = logging.getLogger(__name__)


class ThoughtProcessor:
    def __init__(self, tool_context: Dict[str, Any], console: Console):
        self.system_prompt = tool_context.get("system_prompt", "")
        self.messages = tool_context.get("messages", [])
        self.tool_use_id = str(uuid.uuid4())
        self.console = console

    def create_thinking_prompt(self, thought: str, cycle: int, total_cycles: int) -> str:
        """Create a focused prompt for the thinking process."""
        prompt = f"""
Direct Tasks:
1. Process this thought deeply and analytically
2. Generate clear, structured insights
3. Consider implications and connections
4. Provide actionable conclusions
5. DO NOT call the think tool again
6. USE other tools.

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
        **kwargs: Any,
    ) -> str:
        """Process a single thinking cycle."""

        logger.debug(f"🧠 Thinking Cycle {cycle}/{total_cycles}: Processing cycle...")
        self.console.print(f"🧠 Thinking Cycle {cycle}/{total_cycles}: Processing cycle...")

        # Create cycle-specific prompt
        prompt = self.create_thinking_prompt(thought, cycle, total_cycles)

        # Display input prompt
        logger.debug(f"\n--- Input Prompt ---\n{prompt}\n")

        # Get tools from parent agent if available
        tools = []
        trace_attributes = {}
        parent_agent = kwargs.get("agent")
        if parent_agent:
            tools = list(parent_agent.tool_registry.registry.values())
            trace_attributes = parent_agent.trace_attributes

        # Initialize the new Agent with provided parameters
        agent = Agent(messages=[], tools=tools, system_prompt=custom_system_prompt, trace_attributes=trace_attributes)

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
def think(thought: str, cycle_count: int, system_prompt: str, agent: Any) -> Dict[str, Any]:
    """
    Recursive thinking tool for sophisticated thought generation, learning, and self-reflection.

    This tool implements a multi-cycle cognitive analysis approach that progressively refines thoughts
    through iterative processing. Each cycle builds upon insights from the previous cycle,
    creating a depth of analysis that would be difficult to achieve in a single pass.

    How It Works:
    ------------
    1. The tool processes the initial thought through a specified number of thinking cycles
    2. Each cycle uses the output from the previous cycle as a foundation for deeper analysis
    3. A specialized system prompt guides the thinking process toward specific expertise domains
    4. Each cycle's output is captured and included in the final comprehensive analysis
    5. The tool avoids recursive self-calls and encourages the use of other tools when appropriate

    Thinking Process:
    ---------------
    - First cycle processes the original thought directly with the provided system prompt
    - Each subsequent cycle builds upon the previous cycle's output
    - Cycles are tracked and labeled clearly in the output
    - The process creates a chain of progressive refinement and deeper insights
    - Final output includes the complete thought evolution across all cycles

    Common Usage Scenarios:
    ---------------------
    - Problem analysis: Breaking down complex problems into manageable components
    - Idea development: Progressively refining creative concepts
    - Learning exploration: Generating questions and insights about new domains
    - Strategic planning: Developing multi-step approaches to challenges
    - Self-reflection: Analyzing decision processes and potential biases

    Args:
        thought: The detailed thought or idea to process through multiple thinking cycles.
            This can be a question, statement, problem description, or creative prompt.
        cycle_count: Number of thinking cycles to perform (1-10). More cycles allow for
            deeper analysis but require more time and resources. Typically 3-5 cycles
            provide a good balance of depth and efficiency.
        system_prompt: Custom system prompt to use for the LLM thinking process. This should
            specify the expertise domain and thinking approach for processing the thought.
        **kwargs: Additional keyword arguments passed to the underlying LLM processing.

    Returns:
        Dict containing status and response content in the format:
        {
            "status": "success|error",
            "content": [{"text": "Detailed thinking output across all cycles"}]
        }

        Success case: Returns concatenated results from all thinking cycles
        Error case: Returns information about what went wrong during processing

    Notes:
        - Higher cycle counts provide deeper analysis but consume more resources
        - The system_prompt significantly influences the thinking style and domain expertise
        - For complex topics, more specific system prompts tend to yield better results
        - The tool is designed to avoid recursive self-calls that could cause infinite loops
        - Each cycle has visibility into previous cycle outputs to enable building upon insights
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
            # Process current cycle - need to remove thought from kwargs to prevent duplicate parameters
            cycle_kwargs = kwargs.copy()
            if "thought" in cycle_kwargs:
                del cycle_kwargs["thought"]  # Prevent duplicate 'thought' parameter

            cycle_response = processor.process_cycle(
                current_thought,
                cycle,
                cycle_count,
                custom_system_prompt,
                **cycle_kwargs,
            )

            # Store response
            all_responses.append({"cycle": cycle, "thought": current_thought, "response": cycle_response})

            # Update thought for next cycle based on current response
            current_thought = f"Previous cycle concluded: {cycle_response}\nContinue developing these ideas further."

        # Combine all responses into final output (removing duplicate code)
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
