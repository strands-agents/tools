"""Swarm intelligence tool for coordinating custom AI agent teams.

This module implements a flexible swarm intelligence system that enables users to define
custom teams of specialized AI agents that collaborate autonomously through shared context
and tool-based coordination. Built on the Strands SDK Swarm multi-agent pattern.

Key Features:
-------------
1. Custom Agent Teams:
   • User-defined agent specifications with individual system prompts
   • Per-agent tool configuration and model settings
   • Complete control over agent specializations and capabilities
   • Support for diverse model providers across agents

2. Autonomous Coordination:
   • Built on Strands SDK's native Swarm multi-agent pattern
   • Automatic injection of coordination tools (handoff_to_agent, complete_swarm_task)
   • Shared working memory and context across all agents
   • Self-organizing collaboration without central control

3. Advanced Configuration:
   • Individual model providers and settings per agent
   • Customizable tool access for each agent
   • Comprehensive timeout and safety mechanisms
   • Rich execution metrics and detailed status tracking

4. Emergent Collective Intelligence:
   • Agents autonomously decide when to collaborate or handoff
   • Shared context enables building upon each other's work
   • Dynamic task distribution based on agent capabilities
   • Self-completion when task objectives are achieved

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import swarm

agent = Agent(tools=[swarm])

# Define custom agent team
result = agent.tool.swarm(
    task="Develop a comprehensive product launch strategy",
    agents=[
        {
            "name": "market_researcher",
            "system_prompt": (
                "You are a market research specialist. Focus on market analysis, "
                "customer insights, and competitive landscape."
            ),
            "tools": ["retrieve", "calculator"],
            "model_provider": "bedrock",
            "model_settings": {"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"}
        },
        {
            "name": "product_strategist",
            "system_prompt": (
                "You are a product strategy specialist. Focus on positioning, "
                "value propositions, and go-to-market planning."
            ),
            "tools": ["file_write", "calculator"],
            "model_provider": "anthropic",
            "model_settings": {"model_id": "claude-sonnet-4-20250514"}
        },
        {
            "name": "creative_director",
            "system_prompt": (
                "You are a creative marketing specialist. Focus on campaigns, "
                "branding, messaging, and creative concepts."
            ),
            "tools": ["generate_image", "file_write"],
            "model_provider": "openai",
            "model_settings": {"model_id": "o4-mini"}
        }
    ]
)
```

The swarm tool provides maximum flexibility for creating specialized agent teams that work
together autonomously to solve complex, multi-faceted problems.
"""

import logging
import traceback
from typing import Any, Dict, List, Optional

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from strands import Agent, tool
from strands.multiagent import Swarm

from strands_tools.utils import console_util

logger = logging.getLogger(__name__)


def create_rich_status_panel(console: Console, result: Any) -> str:
    """
    Create a rich formatted status panel for swarm execution results.

    Args:
        console: Rich console for output capture
        result: SwarmResult object from swarm execution

    Returns:
        str: Formatted panel as a string for display
    """
    content = []
    content.append(f"[bold blue]Status:[/bold blue] {result.status}")
    content.append(f"[bold blue]Execution Time:[/bold blue] {result.execution_time}ms")
    content.append(f"[bold blue]Agents Involved:[/bold blue] {result.execution_count}")

    if hasattr(result, "node_history") and result.node_history:
        agent_chain = " → ".join([node.node_id for node in result.node_history])
        content.append(f"[bold blue]Agent Chain:[/bold blue] {agent_chain}")

    if hasattr(result, "accumulated_usage") and result.accumulated_usage:
        usage = result.accumulated_usage
        content.append("\n[bold magenta]Token Usage:[/bold magenta]")
        content.append(f"  [bold green]Input:[/bold green] {usage.get('inputTokens', 0):,}")
        content.append(f"  [bold green]Output:[/bold green] {usage.get('outputTokens', 0):,}")
        content.append(f"  [bold green]Total:[/bold green] {usage.get('totalTokens', 0):,}")

    panel = Panel("\n".join(content), title="🤖 Swarm Execution Results", box=ROUNDED)
    with console.capture() as capture:
        console.print(panel)
    return capture.get()


def _create_custom_agents(
    agent_specs: List[Dict[str, Any]],
    parent_agent: Optional[Any] = None,
) -> List[Agent]:
    """
    Create custom agents based on user specifications.

    Args:
        agent_specs: List of agent specification dictionaries
        parent_agent: Parent agent for inheriting default configuration

    Returns:
        List[Agent]: Custom agent instances

    Raises:
        ValueError: If agent specifications are invalid
    """
    if not agent_specs:
        raise ValueError("At least one agent specification is required")

    agents = []
    used_names = set()

    for i, spec in enumerate(agent_specs):
        # Validate required fields
        if not isinstance(spec, dict):
            raise ValueError(f"Agent specification {i} must be a dictionary")

        # Get agent name with fallback
        agent_name = spec.get("name", f"agent_{i + 1}")

        # Ensure unique names
        if agent_name in used_names:
            original_name = agent_name
            counter = 1
            while agent_name in used_names:
                agent_name = f"{original_name}_{counter}"
                counter += 1
        used_names.add(agent_name)

        # Get system prompt with fallback
        system_prompt = spec.get("system_prompt")
        if not system_prompt:
            if parent_agent and hasattr(parent_agent, "system_prompt") and parent_agent.system_prompt:
                system_prompt = (
                    "You are a helpful AI assistant specializing in collaborative problem solving.\n\n"
                    f"Base Instructions:\n{parent_agent.system_prompt}"
                )
            else:
                system_prompt = "You are a helpful AI assistant specializing in collaborative problem solving."
        else:
            # Optionally append parent system prompt
            if (
                parent_agent
                and hasattr(parent_agent, "system_prompt")
                and parent_agent.system_prompt
                and spec.get("inherit_parent_prompt", False)
            ):
                system_prompt = f"{system_prompt}\n\nBase Instructions:\n{parent_agent.system_prompt}"

        # Configure agent tools
        agent_tools = spec.get("tools")
        if agent_tools and parent_agent and hasattr(parent_agent, "tool_registry"):
            # Filter tools to ensure they exist in parent agent's registry
            available_tools = parent_agent.tool_registry.registry.keys()
            filtered_tool_names = [tool for tool in agent_tools if tool in available_tools]
            if len(filtered_tool_names) != len(spec.get("tools", [])):
                missing_tools = set(spec.get("tools", [])) - set(filtered_tool_names)
                logger.warning(f"Agent '{agent_name}' missing tools: {missing_tools}")

            # Get actual tool objects from parent agent's registry
            agent_tools = [parent_agent.tool_registry.registry[tool_name] for tool_name in filtered_tool_names]

        # Create agent
        swarm_agent = Agent(
            name=agent_name,
            system_prompt=system_prompt,
            tools=agent_tools,
            callback_handler=parent_agent.callback_handler if parent_agent else None,
            trace_attributes=parent_agent.trace_attributes if parent_agent else None,
        )

        # Configure model provider
        model_provider = spec.get("model_provider")
        if model_provider:
            swarm_agent.model_provider = model_provider
        elif parent_agent and hasattr(parent_agent, "model_provider"):
            swarm_agent.model_provider = parent_agent.model_provider

        # Configure model settings
        model_settings = spec.get("model_settings")
        if model_settings:
            swarm_agent.model_settings = model_settings
        elif parent_agent and hasattr(parent_agent, "model_settings"):
            swarm_agent.model_settings = parent_agent.model_settings

        agents.append(swarm_agent)
        logger.debug(f"Created agent '{agent_name}' with {len(agent_tools or [])} tools")

    return agents


@tool
def swarm(
    task: str,
    agents: List[Dict[str, Any]],
    max_handoffs: int = 20,
    max_iterations: int = 20,
    execution_timeout: float = 900.0,
    node_timeout: float = 300.0,
    repetitive_handoff_detection_window: int = 8,
    repetitive_handoff_min_unique_agents: int = 3,
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create and coordinate a custom team of AI agents for collaborative task solving.

    This function leverages the Strands SDK's Swarm multi-agent pattern to create custom teams
    of specialized AI agents with individual configurations. Each agent can have its own system
    prompt, tools, model provider, and settings, enabling precise control over team composition.

    How It Works:
    ------------
    1. Custom Agent Creation:
       • Each agent is created with individual specifications
       • Unique system prompts define each agent's role and expertise
       • Per-agent tool access controls what each agent can do
       • Individual model providers and settings for optimization

    2. Autonomous Coordination:
       • Agents automatically receive coordination tools (handoff_to_agent, complete_swarm_task)
       • Shared working memory maintains context across all handoffs
       • Agents decide when to collaborate based on task requirements
       • Self-organizing collaboration without central control

    3. Flexible Team Composition:
       • Mix different model providers for diverse capabilities
       • Assign specialized tools to relevant agents only
       • Custom temperature and model settings per agent
       • Support for any number of agents with unique roles

    4. Safety and Control:
       • Comprehensive timeout mechanisms prevent infinite loops
       • Handoff limits ensure efficient resource usage
       • Repetitive behavior detection prevents endless agent exchanges
       • Rich execution metrics for performance insights

    Args:
        task: The main task to be processed by the agent team.
        agents: List of agent specification dictionaries. Each dictionary can contain:
            - name (str): Agent name/identifier (optional, auto-generated if not provided)
            - system_prompt (str): Agent's system prompt defining its role and expertise
            - tools (List[str]): List of tool names available to this agent (optional)
            - model_provider (str): Model provider for this agent (optional, inherits from parent)
            - model_settings (Dict): Model configuration for this agent (optional)
            - inherit_parent_prompt (bool): Whether to append parent agent's system prompt (optional)
        max_handoffs: Maximum number of handoffs between agents (default: 20).
        max_iterations: Maximum total iterations across all agents (default: 20).
        execution_timeout: Maximum total execution time in seconds (default: 900).
        node_timeout: Maximum time per agent in seconds (default: 300).
        repetitive_handoff_detection_window: Number of recent handoffs to analyze for repetitive behavior (default: 8).
        repetitive_handoff_min_unique_agents: Minimum number of unique agents required in the
            detection window (default: 3).
        agent: The parent agent (automatically passed by Strands framework).

    Returns:
        Dict containing status and response content in the format:
        {
            "status": "success|error",
            "content": [{"text": "Comprehensive results from agent team collaboration"}]
        }

        Success case: Returns detailed results from swarm execution with agent contributions
        Error case: Returns information about what went wrong during processing

    Example Usage:
    -------------
    ```python
    # Research and development team
    result = agent.tool.swarm(
        task="Research and design a sustainable energy solution for rural communities",
        agents=[
            {
                "name": "researcher",
                "system_prompt": "You are a renewable energy specialist. Focus on feasibility and impact.",
                "tools": ["retrieve", "calculator"],
                "model_provider": "bedrock",
                "model_settings": {"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"}
            },
            {
                "name": "engineer",
                "system_prompt": "You are an engineering specialist. Focus on implementation and costs.",
                "tools": ["calculator", "file_write"],
                "model_provider": "anthropic",
                "model_settings": {"model_id": "claude-sonnet-4-20250514"}
            },
            {
                "name": "community_expert",
                "system_prompt": "You are a community specialist. Focus on social impact and adoption.",
                "tools": ["retrieve", "file_write"],
                "model_provider": "openai",
                "model_settings": {"model_id": "o4-mini"}
            }
        ]
    )

    # Creative content team
    result = agent.tool.swarm(
        task="Create a comprehensive brand identity and marketing campaign",
        agents=[
            {
                "name": "brand_strategist",
                "system_prompt": "You are a brand strategist. Focus on positioning and messaging.",
                "tools": ["retrieve", "file_write"]
            },
            {
                "name": "creative_director",
                "system_prompt": "You are a creative director. Focus on visual concepts and campaigns.",
                "tools": ["generate_image", "file_write"],
                "model_settings": {"params": {"temperature": 0.8}}
            },
            {
                "name": "copywriter",
                "system_prompt": "You are a copywriter. Focus on messaging and marketing copy.",
                "tools": ["file_write"],
                "model_settings": {"params": {"temperature": 0.7}}
            }
        ],
        execution_timeout=1200  # Extended timeout for creative work
    )

    # Minimal team with inheritance
    result = agent.tool.swarm(
        task="Analyze quarterly financial performance",
        agents=[
            {
                "system_prompt": "You are a financial analyst specializing in performance metrics and trend analysis.",
                "tools": ["calculator", "file_write"],
                "inherit_parent_prompt": True
            },
            {
                "system_prompt": "You are a business strategist focusing on insights and recommendations.",
                "tools": ["file_write"],
                "inherit_parent_prompt": True
            }
        ]
    )

    # Custom repetitive handoff detection
    result = agent.tool.swarm(
        task="Complex multi-step analysis requiring tight collaboration",
        agents=[...],
        repetitive_handoff_detection_window=12,  # Look at more recent handoffs
        repetitive_handoff_min_unique_agents=4,  # Require more variety in agent participation
    )
    ```

    Notes:
        - Built on Strands SDK's native Swarm multi-agent pattern
        - Each agent can use different models and tools for optimal performance
        - Agents coordinate autonomously through injected coordination tools
        - Shared context enables true collective intelligence
        - Safety mechanisms prevent infinite loops and resource exhaustion
        - Rich execution metrics provide insights into team collaboration
        - Supports complex multi-modal tasks and diverse expertise areas
        - Tool filtering ensures agents only get tools that exist in parent registry
    """
    console = console_util.create()

    try:
        # Validate input
        if not agents:
            raise ValueError("At least one agent specification is required")

        if len(agents) > 10:
            logger.warning(f"Large team size ({len(agents)} agents) may impact performance")

        logger.info(f"Creating custom swarm with {len(agents)} agents")

        # Create custom agents from specifications
        swarm_agents = _create_custom_agents(
            agent_specs=agents,
            parent_agent=agent,
        )

        # Create SDK Swarm with configuration
        sdk_swarm = Swarm(
            nodes=swarm_agents,
            max_handoffs=max_handoffs,
            max_iterations=max_iterations,
            execution_timeout=execution_timeout,
            node_timeout=node_timeout,
            repetitive_handoff_detection_window=repetitive_handoff_detection_window,
            repetitive_handoff_min_unique_agents=repetitive_handoff_min_unique_agents,
        )

        logger.info(f"Starting swarm execution with task: {task[:1000]}...")

        # Execute the swarm
        result = sdk_swarm(task)

        # Create rich status display
        create_rich_status_panel(console, result)

        # Extract and format results
        response_parts = []

        # Add execution summary
        response_parts.append("🎯 **Custom Agent Team Execution Complete**")
        response_parts.append(f"📊 **Status:** {result.status}")
        response_parts.append(f"⏱️ **Execution Time:** {result.execution_time}ms")
        response_parts.append(f"🤖 **Team Size:** {len(swarm_agents)} agents")
        response_parts.append(f"🔄 **Iterations:** {result.execution_count}")

        if hasattr(result, "node_history") and result.node_history:
            agent_chain = " → ".join([node.node_id for node in result.node_history])
            response_parts.append(f"🔗 **Collaboration Chain:** {agent_chain}")

        # Add individual agent results
        if hasattr(result, "results") and result.results:
            response_parts.append("\n**🤖 Individual Agent Contributions:**")
            for agent_name, node_result in result.results.items():
                if hasattr(node_result, "result") and hasattr(node_result.result, "content"):
                    agent_content = []
                    for content_block in node_result.result.content:
                        if hasattr(content_block, "text") and content_block.text:
                            agent_content.append(content_block.text)

                    if agent_content:
                        response_parts.append(f"\n**{agent_name.upper().replace('_', ' ')}:**")
                        response_parts.extend(agent_content)

        # Add final consolidated result
        if hasattr(result, "node_history") and result.node_history and hasattr(result, "results") and result.results:
            last_agent = result.node_history[-1].node_id
            if last_agent in result.results:
                last_result = result.results[last_agent]
                if hasattr(last_result, "result") and hasattr(last_result.result, "content"):
                    response_parts.append("\n**🎯 Final Team Result:**")
                    for content_block in last_result.result.content:
                        if hasattr(content_block, "text") and content_block.text:
                            response_parts.append(content_block.text)

        # Add resource usage metrics
        if hasattr(result, "accumulated_usage") and result.accumulated_usage:
            usage = result.accumulated_usage
            response_parts.append("\n**📈 Team Resource Usage:**")
            response_parts.append(f"• Input tokens: {usage.get('inputTokens', 0):,}")
            response_parts.append(f"• Output tokens: {usage.get('outputTokens', 0):,}")
            response_parts.append(f"• Total tokens: {usage.get('totalTokens', 0):,}")

        final_response = "\n".join(response_parts)

        return {
            "status": "success",
            "content": [{"text": final_response}],
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Custom swarm execution failed: {str(e)}\n{error_trace}")

        return {
            "status": "error",
            "content": [{"text": f"⚠️ Custom swarm execution failed: {str(e)}"}],
        }
