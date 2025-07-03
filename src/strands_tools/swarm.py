"""Swarm intelligence tool for coordinating multiple AI agents in parallel.

This module implements a swarm intelligence system that enables multiple Strands AI agents
to collaborate on complex tasks through parallel processing and emergent collective intelligence.
The swarm architecture facilitates distributed problem solving with shared memory and
various coordination patterns for optimal collaboration.

Key Features:
-------------
1. Multi-Agent Architecture:
   • Parallel execution of multiple AI instances
   • Shared memory system for collective intelligence
   • Phase-based progression for complex reasoning
   • Agent specialization and role assignment

2. Coordination Patterns:
   • Collaborative: Agents build upon each other's insights
   • Competitive: Agents independently seek optimal solutions
   • Hybrid: Balanced approach combining collaboration and competition

3. Intelligent Memory System:
   • Shared knowledge repository with phase tracking
   • Concurrent read/write operations with thread safety
   • Historical knowledge access across phases
   • Automatic phase progression

4. Real-Time Status Tracking:
   • Rich formatted status visualization
   • Individual agent performance metrics
   • Contribution tracking
   • Collective intelligence measurement

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import swarm

agent = Agent(tools=[swarm])

# Create a collaborative swarm to solve a complex problem
result = agent.tool.swarm(
    task="Analyze the environmental impact of renewable energy sources compared to fossil fuels",
    swarm_size=5,
    coordination_pattern="collaborative"
)

# Use a competitive pattern for exploring multiple solution paths
result = agent.tool.swarm(
    task="Generate five unique marketing campaign concepts for a new smartphone",
    swarm_size=5,
    coordination_pattern="competitive"
)

# Use a hybrid approach for balanced problem-solving
result = agent.tool.swarm(
    task="Develop a comprehensive business strategy for entering a new market",
    swarm_size=3,
    coordination_pattern="hybrid"
)
```

The swarm tool provides a powerful way to harness collective intelligence for complex tasks,
combining the strengths of multiple agents working in parallel with different perspectives.
"""

import json
import logging
import textwrap
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, List, Optional, cast

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from strands import tool

from strands_tools.use_llm import use_llm
from strands_tools.utils import console_util

logger = logging.getLogger(__name__)

# Constants for resource management
MAX_THREADS = 10
MESSAGE_PROCESSING_DELAY = 0.1  # seconds
MAX_QUEUE_SIZE = 1000


def create_rich_status_panel(console: Console, status: Dict[str, Any]) -> str:
    """
    Create a rich formatted status panel for swarm status.

    Args:
        status: Dictionary containing swarm status information

    Returns:
        str: Formatted panel as a string for display
    """
    content = []
    content.append(f"[bold blue]Task:[/bold blue] {status['task']}")
    content.append(f"[bold blue]Pattern:[/bold blue] {status['coordination_pattern']}")
    content.append(f"[bold blue]Shared Memory ID:[/bold blue] {status['memory_id']}")
    content.append("\n[bold magenta]Agents:[/bold magenta]")

    for agent in status["agents"]:
        agent_info = [
            f"  [bold green]ID:[/bold green] {agent['id']}",
            f"  [bold green]Status:[/bold green] {agent['status']}",
            f"  [bold green]Contributions:[/bold green] {agent['contributions']}\n",
        ]
        content.extend(agent_info)

    panel = Panel("\n".join(content), title="Swarm Status", box=ROUNDED)
    with console.capture() as capture:
        console.print(panel)
    return capture.get()


class SharedMemory:
    """
    Shared memory system for swarm agents to store and retrieve knowledge.

    This class implements a thread-safe shared memory system that enables
    agents to store and access collective knowledge across different phases
    of swarm operation. It supports concurrent access, phase tracking, and
    historical knowledge retrieval.

    Attributes:
        memory_id: Unique identifier for this memory instance
        lock: Thread lock for ensuring thread safety
        knowledge_store: List of knowledge entries from all agents
        current_phase: Current processing phase number
        last_update: Timestamp of the last update
    """

    def __init__(self) -> None:
        """Initialize a new shared memory instance with a unique ID."""
        self.memory_id = str(uuid.uuid4())
        self.lock = Lock()
        self.knowledge_store: List[Dict[str, Any]] = []
        self.current_phase = 0
        self.last_update = time.time()

    def store(self, agent_id: str, content: str) -> bool:
        """
        Store a new knowledge entry in shared memory.

        Args:
            agent_id: ID of the agent contributing the knowledge
            content: Knowledge content to store

        Returns:
            bool: True if storage was successful, False otherwise
        """
        try:
            with self.lock:
                entry = {
                    "id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "content": content,
                    "phase": self.current_phase,
                    "timestamp": time.time(),
                }
                self.knowledge_store.append(entry)
                self.last_update = time.time()
                return True
        except Exception as e:
            logger.error(f"Error storing in shared memory: {str(e)}")
            return False

    def get_current_knowledge(self) -> List[Dict[str, Any]]:
        """
        Get all knowledge entries from the current phase.

        Returns:
            List[Dict[str, Any]]: Knowledge entries from current phase
        """
        with self.lock:
            return [entry for entry in self.knowledge_store if entry["phase"] == self.current_phase]

    def get_all_knowledge(self) -> List[Dict[str, Any]]:
        """
        Get all stored knowledge across all phases.

        Returns:
            List[Dict[str, Any]]: All knowledge entries sorted by timestamp
        """
        with self.lock:
            return sorted(self.knowledge_store, key=lambda x: x["timestamp"])

    def advance_phase(self) -> None:
        """Move to the next collaboration phase."""
        with self.lock:
            self.current_phase += 1


class SwarmAgent:
    """
    Individual agent within the swarm intelligence system.

    This class represents a single AI agent in the swarm, with its own
    identity, role, and ability to process tasks and contribute to the
    collective intelligence through the shared memory system.

    Attributes:
        id: Unique identifier for this agent
        system_prompt: Agent-specific system prompt defining its role and behavior
        shared_memory: Reference to the shared memory instance
        status: Current status of the agent
        contributions: Number of contributions made by this agent
        results: List of results from this agent's processing
        lock: Thread lock for ensuring thread safety
    """

    def __init__(
        self,
        agent_id: str,
        system_prompt: str,
        shared_memory: SharedMemory,
    ) -> None:
        """
        Initialize a new swarm agent.

        Args:
            agent_id: Unique identifier for this agent
            system_prompt: System prompt defining the agent's role and behavior
            shared_memory: Reference to the shared memory instance
        """
        self.id = agent_id
        self.system_prompt = system_prompt
        self.shared_memory = shared_memory
        self.status = "initialized"
        self.contributions = 0
        self.results: List[Dict[str, Any]] = []
        self.lock = Lock()

    def process_task(
        self,
        task: str,
        parent_agent: Any,
        model_provider: Optional[str] = None,
        model_settings: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process the assigned task based on the agent's role and available knowledge.

        This method processes the task by:
        1. Retrieving relevant knowledge from shared memory
        2. Constructing a context-aware prompt based on the phase and available knowledge
        3. Generating a response using the use_llm tool
        4. Storing the contribution in shared memory

        Args:
            task: The task to be processed
            parent_agent: Parent agent to pass to use_llm
            model_provider: Model provider to use for this agent
            model_settings: Model configuration for this agent
            tools: List of tools available to this agent

        Returns:
            Dict[str, Any]: Result of the processing
        """
        try:
            self.status = "processing"

            # Get current phase knowledge
            current_knowledge = self.shared_memory.get_current_knowledge()

            # Get only relevant historical knowledge (skip current phase)
            historical_knowledge = [
                k for k in self.shared_memory.get_all_knowledge() if k["phase"] < self.shared_memory.current_phase
            ]

            # Construct focused collaborative prompt
            historical_knowledge_str = (
                json.dumps(historical_knowledge, indent=2)
                if historical_knowledge
                else "No previous knowledge available yet."
            )
            current_knowledge_str = (
                json.dumps([k for k in current_knowledge if k["agent_id"] != self.id], indent=2)
                if current_knowledge
                else "You are the first agent in this phase."
            )
            prompt = textwrap.dedent(f"""
                Task: {task}

                Current Phase: {self.shared_memory.current_phase}
                Your Role: Agent {self.id}

                Previous Phase Knowledge:
                {historical_knowledge_str}

                Current Phase Knowledge from Other Agents:
                {current_knowledge_str}

                Instructions:
                1. Review any existing knowledge from previous phases
                2. Consider current phase contributions from other agents
                3. Provide new insights or build upon others' findings
                4. Focus on unique value addition
                5. Be concise but thorough

                IMPORTANT: Do not use swarm tool in swarming state. You are in swarming state.

                Provide your contribution as a clear, focused response.
            """).strip()

            # Process task using new use_llm format with model and tools configuration
            result = use_llm(
                prompt=prompt,
                system_prompt=self.system_prompt,
                model_provider=model_provider,
                model_settings=model_settings,
                tools=tools,
                agent=parent_agent,
            )

            if result.get("status") == "success":
                # Extract and store contribution
                contribution = "\n".join(content.get("text", "") for content in result.get("content", []))

                if self.shared_memory.store(self.id, contribution):
                    self.contributions += 1
                    # Convert ToolResult to Dict[str, Any] before appending
                    self.results.append(cast(Dict[str, Any], result))
                    self.status = "completed"

            return cast(Dict[str, Any], result)

        except Exception as e:
            logger.error(f"Error in agent {self.id}: {str(e)}")
            self.status = "error"
            return {"status": "error", "content": [{"text": str(e)}]}

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of this agent.

        Returns:
            Dict[str, Any]: Status information for this agent
        """
        return {
            "id": self.id,
            "status": self.status,
            "contributions": self.contributions,
        }


class Swarm:
    """
    Swarm intelligence system coordinating multiple AI agents.

    This class implements the main swarm intelligence system that coordinates
    multiple AI agents working in parallel on a shared task. It manages agent
    creation, task assignment, phase progression, and result collection.

    Attributes:
        task: The main task being processed by the swarm
        coordination_pattern: The pattern used for agent coordination
        shared_memory: The shared memory instance for this swarm
        agents: Dictionary of agents in this swarm
        lock: Thread lock for ensuring thread safety
    """

    def __init__(self, task: str, coordination_pattern: str) -> None:
        """
        Initialize a new swarm intelligence system.

        Args:
            task: The main task to be processed
            coordination_pattern: Pattern for agent coordination ("collaborative", "competitive", or "hybrid")
        """
        self.task = task
        self.coordination_pattern = coordination_pattern
        self.shared_memory = SharedMemory()
        self.agents: Dict[str, SwarmAgent] = {}
        self.lock = Lock()

    def add_agent(self, agent_id: str, system_prompt: str) -> SwarmAgent:
        """
        Add a new agent to the swarm.

        Args:
            agent_id: Unique identifier for the new agent
            system_prompt: System prompt defining the agent's role and behavior

        Returns:
            SwarmAgent: The newly created agent
        """
        with self.lock:
            agent = SwarmAgent(agent_id, system_prompt, self.shared_memory)
            self.agents[agent_id] = agent
            return agent

    def process_phase(
        self,
        parent_agent: Any,
        model_provider: Optional[str] = None,
        model_settings: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process the current phase with all agents working in parallel.

        This method:
        1. Executes all agents' task processing in parallel using ThreadPoolExecutor
        2. Collects results as they become available
        3. Advances to the next phase after all agents complete

        Args:
            parent_agent: Parent agent to pass to agents
            model_provider: Model provider for all swarm agents
            model_settings: Model configuration for all swarm agents
            tools: List of tools available to all swarm agents

        Returns:
            List[Dict[str, Any]]: Results from all agents for this phase
        """
        results = []

        # Process agents in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            # Create futures for each agent
            future_to_agent = {
                executor.submit(
                    agent.process_task, self.task, parent_agent, model_provider, model_settings, tools
                ): agent
                for agent in self.agents.values()
            }

            # Collect results as they complete
            for future in future_to_agent:
                agent = future_to_agent[future]
                try:
                    result = future.result()
                    results.append({"agent_id": agent.id, "result": result})
                except Exception as e:
                    logger.error(f"Error processing agent {agent.id}: {str(e)}")
                    results.append(
                        {
                            "agent_id": agent.id,
                            "result": {
                                "status": "error",
                                "content": [{"text": f"Error: {str(e)}"}],
                            },
                        }
                    )

            # Wait a short time for shared memory updates to propagate
            time.sleep(0.5)

        # Advance to next phase only after all agents have completed
        self.shared_memory.advance_phase()

        return results

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the swarm.

        Returns:
            Dict[str, Any]: Status information for the swarm and its agents
        """
        return {
            "task": self.task,
            "coordination_pattern": self.coordination_pattern,
            "memory_id": self.shared_memory.memory_id,
            "agents": [agent.get_status() for agent in self.agents.values()],
        }


@tool
def swarm(
    task: str,
    swarm_size: int,
    coordination_pattern: str = "collaborative",
    model_provider: Optional[str] = None,
    model_settings: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None,
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create and coordinate a swarm of AI agents for parallel processing and collective intelligence.

    This function implements a swarm intelligence system that enables multiple Strands AI agents
    to collaborate on complex tasks through parallel processing and emergent collective intelligence.

    How It Works:
    ------------
    1. Task Analysis and Initialization:
       • Creates a swarm instance with the specified task and coordination pattern
       • Determines the optimal number of agents based on task complexity and swarm_size parameter
       • Establishes a shared memory system for knowledge exchange and storage

    2. Agent Creation and Specialization:
       • Creates multiple agent instances with specialized roles based on the coordination pattern
       • Configures each agent with a tailored system prompt defining its behavior and focus
       • Connects all agents to the shared memory system

    3. Multi-Phase Execution Process:
       • Executes tasks in multiple phases for progressive refinement
       • In each phase, all agents work in parallel, processing the task simultaneously
       • Results are collected, aggregated, and stored in shared memory
       • Phase transitions enable agents to build upon previous discoveries

    4. Coordination Patterns:
       • Collaborative: Agents focus on building upon others' insights
       • Competitive: Agents work independently to find unique solutions
       • Hybrid: Balanced approach combining collaboration and competition

    5. Result Aggregation and Synthesis:
       • Collects contributions from all agents across all phases
       • Combines insights into comprehensive collective knowledge
       • Presents results with agent-specific attributions and synthesis

    Common Error Scenarios:
    ---------------------
    • Resource constraints: Too many agents requested for available resources
    • Execution timeout: Individual agent processing exceeds time limits
    • Context limitations: Task complexity exceeds context window capacity
    • Agent conflicts: Competitive agents producing contradictory results
    • Memory overflow: Knowledge accumulation exceeds storage capacity

    Performance Considerations:
    -------------------------
    • Swarm size directly impacts execution time and resource usage
    • Multi-phase execution increases result quality but extends processing time
    • Recommended maximum swarm_size is 10 for optimal performance
    • Thread pool management prevents resource exhaustion

    Args:
        task: The main task to be processed by the swarm.
        swarm_size: Number of agents in the swarm (1-10).
        coordination_pattern: How agents should coordinate.
            Options: "collaborative", "competitive", "hybrid"
            Default: "collaborative"
        model_provider: Model provider for all swarm agents.
            Options: "bedrock", "anthropic", "litellm", "llamaapi", "ollama", "openai", "github"
            Special values:
            - None: Use parent agent's model (default)
            - "env": Use environment variables to determine provider
        model_settings: Model configuration for all swarm agents.
            Example: {"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0", "params": {"temperature": 0.7}}
        tools: List of tool names available to all swarm agents.
            Tool names must exist in the parent agent's tool registry.
            Examples: ["calculator", "file_read", "retrieve"]
            If not provided, inherits all tools from the parent agent.
        agent: The parent agent (automatically passed by Strands framework).

    Returns:
        Dict containing status and response content in the format:
        {
            "status": "success|error",
            "content": [{"text": "Operation result message"}]
        }

        Success case: Returns swarm results with collective intelligence
        Error case: Returns information about what went wrong during processing

    Example Usage:
    -------------
    ```python
    # Basic collaborative pattern for complex problem-solving
    result = agent.tool.swarm(
        task="Analyze the environmental impact of renewable energy sources",
        swarm_size=5,
        coordination_pattern="collaborative"
    )

    # Enhanced swarm with specific model and tools
    result = agent.tool.swarm(
        task="Generate unique marketing campaign concepts",
        swarm_size=3,
        coordination_pattern="competitive",
        model_provider="bedrock",
        model_settings={"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0", "params": {"temperature": 0.9}},
        tools=["retrieve", "generate_image", "file_write"]
    )

    # Environment-based model selection with tool restrictions
    result = agent.tool.swarm(
        task="Process sensitive financial data",
        swarm_size=2,
        coordination_pattern="collaborative",
        model_provider="env",  # Uses STRANDS_PROVIDER environment variable
        tools=["calculator", "file_read"]  # Only calculation and file reading tools
    )

    # Local model processing for privacy-sensitive tasks
    result = agent.tool.swarm(
        task="Analyze confidential research data",
        swarm_size=3,
        coordination_pattern="hybrid",
        model_provider="ollama",
        model_settings={"model_id": "llama3", "host": "http://localhost:11434"},
        tools=["calculator", "python_repl"]
    )
    ```

    Notes:
        - Swarm size is automatically clamped between 1 and 10 for resource management
        - Multi-phase execution provides progressive refinement of results
        - Shared memory enables collective intelligence across agents
        - Thread pool management prevents resource exhaustion
        - Rich formatted output provides visual status and results
        - Model configuration allows using different providers (bedrock, anthropic, ollama, etc.)
        - Tools parameter enables security isolation and role-based access control
        - Environment-based model selection supports dynamic configuration
        - All swarm agents use the same model and tools configuration for consistency
    """
    console = console_util.create()

    try:
        # Validate and clamp swarm size
        swarm_size = min(max(swarm_size, 1), 10)

        # Create swarm
        swarm_instance = Swarm(task, coordination_pattern)

        # Create agents with specialized roles
        for i in range(swarm_size):
            agent_id = f"agent_{i + 1}"

            if coordination_pattern == "collaborative":
                role = f"Collaborative Agent {i + 1} - Focus on building upon others' insights"
            elif coordination_pattern == "competitive":
                role = f"Competitive Agent {i + 1} - Focus on finding unique solutions"
            else:  # hybrid
                role = f"Hybrid Agent {i + 1} - Balance cooperation and innovation"

            # Get system prompt from parent agent if available
            base_system_prompt = agent.system_prompt if agent else "You are a helpful AI assistant."

            agent_prompt = f"""You are {role} in a swarm intelligence system.

Task: {task}
Coordination Pattern: {coordination_pattern}

Key Responsibilities:
1. Review existing knowledge
2. Add unique insights
3. Build upon others' contributions
4. Maintain task focus
5. Collaborate effectively

{base_system_prompt}"""

            swarm_instance.add_agent(agent_id, agent_prompt)

        # Process collaborative phases
        all_results = []
        for _phase in range(2):  # Two phases of collaboration
            phase_results = swarm_instance.process_phase(agent, model_provider, model_settings, tools)
            all_results.extend(phase_results)

        # Get final status
        status = swarm_instance.get_status()

        # Create rich output
        create_rich_status_panel(console, status)

        # Process results
        processed_results = []
        for result in all_results:
            if result["result"].get("status") == "success":
                for content in result["result"].get("content", []):
                    if content.get("text"):
                        processed_results.append(f"🤖 Agent {result['agent_id']}: {content['text']}")

        # Add collective knowledge
        collective_knowledge = swarm_instance.shared_memory.get_all_knowledge()

        # Format final response
        response_text = f"📊 Swarm Results ({len(all_results)} responses):\n\n"
        response_text += "\n".join(processed_results)
        response_text += f"\n\n🌟 Collective Knowledge:\n{json.dumps(collective_knowledge, indent=2)}"

        return {
            "status": "success",
            "content": [{"text": response_text}],
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{error_trace}"
        logger.error(f"\n[SWARM TOOL ERROR]\n{error_msg}")

        return {
            "status": "error",
            "content": [{"text": f"⚠️ Swarm Error: {str(e)}"}],
        }
