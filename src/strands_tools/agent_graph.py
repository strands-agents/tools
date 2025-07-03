"""Agent graph tool with @tool decorator, per-node model configuration, and tools support.

This module provides functionality to create and manage graphs of AI agents with different
topologies and communication patterns. Each agent can have its own model provider,
configuration, and tools for maximum flexibility, optimization, and security.

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import agent_graph

agent = Agent(tools=[agent_graph])

# Basic usage with default model (inherits from parent agent)
result = agent.tool.agent_graph(
    action="create",
    graph_id="analysis_graph",
    topology={
        "type": "star",
        "nodes": [
            {
                "id": "central",
                "role": "coordinator",
                "system_prompt": "You are the central coordinator."
            },
            {
                "id": "agent1",
                "role": "analyzer",
                "system_prompt": "You are a data analyzer."
            }
        ],
        "edges": [
            {"from": "central", "to": "agent1"}
        ]
    }
)

# Per-node model configuration - each agent can use different models!
result = agent.tool.agent_graph(
    action="create",
    graph_id="mixed_model_graph",
    topology={
        "type": "star",
        "nodes": [
            {
                "id": "coordinator",
                "role": "coordinator",
                "system_prompt": "You coordinate tasks efficiently.",
                "model_provider": "bedrock",
                "model_settings": {"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"}
            },
            {
                "id": "fast_analyst",
                "role": "analyst",
                "system_prompt": "You do quick analysis.",
                "model_provider": "bedrock",
                "model_settings": {"model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0"}
            },
            {
                "id": "local_processor",
                "role": "processor",
                "system_prompt": "You process data locally.",
                "model_provider": "ollama",
                "model_settings": {"model_id": "qwen3:1.7b", "host": "http://localhost:11434"}
            }
        ],
        "edges": [
            {"from": "coordinator", "to": "fast_analyst"},
            {"from": "coordinator", "to": "local_processor"}
        ]
    }
)

# Per-node tools configuration for security and specialization
result = agent.tool.agent_graph(
    action="create",
    graph_id="secure_team",
    tools=["retrieve", "memory"],  # Default tools for all agents
    topology={
        "type": "star",
        "nodes": [
            {
                "id": "coordinator",
                "role": "coordinator",
                "system_prompt": "You coordinate tasks efficiently.",
                "tools": ["agent_graph", "slack"]  # Only coordination tools
            },
            {
                "id": "file_processor",
                "role": "processor",
                "system_prompt": "You process files securely.",
                "tools": ["file_read", "file_write"]  # Only file operations
            },
            {
                "id": "calculator",
                "role": "analyst",
                "system_prompt": "You perform calculations.",
                "tools": ["calculator", "python_repl"]  # Only computation tools
            }
        ]
    }
)

# Graph-level model configuration with individual overrides
result = agent.tool.agent_graph(
    action="create",
    graph_id="hybrid_graph",
    model_provider="anthropic",  # Default for all nodes
    model_settings={"model_id": "claude-sonnet-4-20250514"},
    topology={
        "type": "mesh",
        "nodes": [
            {
                "id": "researcher",
                "role": "researcher",
                "system_prompt": "You research topics."
                # Uses graph-level anthropic model
            },
            {
                "id": "specialist",
                "role": "specialist",
                "system_prompt": "You provide specialized analysis.",
                "model_provider": "bedrock",  # Override: uses bedrock instead
                "model_settings": {"model_id": "claude-opus-4-20250514"}
            }
        ]
    }
)
```

See the agent_graph function docstring for more details on configuration options and parameters.
"""

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Lock
from typing import Any, Dict, List, Optional

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from strands import tool

from strands_tools.use_llm import use_llm
from strands_tools.utils import console_util

logger = logging.getLogger(__name__)

# Constants for resource management
MAX_THREADS = 10
MESSAGE_PROCESSING_DELAY = 0.1  # seconds
MAX_QUEUE_SIZE = 1000


def create_rich_table(console: Console, title: str, headers: List[str], rows: List[List[str]]) -> str:
    """Create a rich formatted table"""
    table = Table(title=title, box=ROUNDED, header_style="bold magenta")
    for header in headers:
        table.add_column(header)
    for row in rows:
        table.add_row(*row)
    with console.capture() as capture:
        console.print(table)
    return capture.get()


def create_rich_tree(console: Console, title: str, data: Dict) -> str:
    """Create a rich formatted tree view"""
    tree = Tree(title)

    def add_dict_to_tree(tree_node, data_dict):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                branch = tree_node.add(f"[bold blue]{key}")
                add_dict_to_tree(branch, value)
            elif isinstance(value, list):
                branch = tree_node.add(f"[bold blue]{key}")
                for item in value:
                    if isinstance(item, dict):
                        add_dict_to_tree(branch, item)
                    else:
                        branch.add(str(item))
            else:
                tree_node.add(f"[bold green]{key}:[/bold green] {value}")

    add_dict_to_tree(tree, data)
    with console.capture() as capture:
        console.print(tree)
    return capture.get()


def create_rich_status_panel(console: Console, status: Dict) -> str:
    """Create a rich formatted status panel"""
    content = []
    content.append(f"[bold blue]Graph ID:[/bold blue] {status['graph_id']}")
    content.append(f"[bold blue]Topology:[/bold blue] {status['topology']}")
    content.append(f"[bold blue]Default Model:[/bold blue] {status.get('default_model_provider', 'parent')}")
    content.append(f"[bold blue]Default Tools:[/bold blue] {status.get('default_tools_count', 'parent')}")
    content.append("\n[bold magenta]Nodes:[/bold magenta]")

    for node in status["nodes"]:
        node_info = [
            f"  [bold green]ID:[/bold green] {node['id']}",
            f"  [bold green]Role:[/bold green] {node['role']}",
            f"  [bold green]Model:[/bold green] {node.get('model_provider', 'default')}",
            f"  [bold green]Tools:[/bold green] {node.get('tools_count', 'default')}",
            f"  [bold green]Queue Size:[/bold green] {node['queue_size']}",
            f"  [bold green]Neighbors:[/bold green] {', '.join(node['neighbors'])}\n",
        ]
        content.extend(node_info)

    panel = Panel("\n".join(content), title="Graph Status", box=ROUNDED)
    with console.capture() as capture:
        console.print(panel)
    return capture.get()


class AgentNode:
    def __init__(
        self,
        node_id: str,
        role: str,
        system_prompt: str,
        model_provider: Optional[str] = None,
        model_settings: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
    ):
        self.id = node_id
        self.role = role
        self.system_prompt = system_prompt
        self.model_provider = model_provider
        self.model_settings = model_settings
        self.tools = tools
        self.neighbors = []
        self.input_queue = Queue(maxsize=MAX_QUEUE_SIZE)
        self.is_running = True
        self.thread = None
        self.last_process_time = 0
        self.lock = Lock()

    def add_neighbor(self, neighbor):
        with self.lock:
            if neighbor not in self.neighbors:
                self.neighbors.append(neighbor)

    def process_messages(self, parent_agent: Any):
        while self.is_running:
            try:
                # Rate limiting
                current_time = time.time()
                if current_time - self.last_process_time < MESSAGE_PROCESSING_DELAY:
                    time.sleep(MESSAGE_PROCESSING_DELAY)

                if not self.input_queue.empty():
                    message = self.input_queue.get_nowait()
                    self.last_process_time = current_time

                    try:
                        # Process message with LLM using per-node model and tools configuration
                        result = use_llm(
                            prompt=message["content"],
                            system_prompt=self.system_prompt,
                            model_provider=self.model_provider,
                            model_settings=self.model_settings,
                            tools=self.tools,
                            agent=parent_agent,
                        )

                        if result.get("status") == "success":
                            response_content = ""
                            for content in result.get("content", []):
                                if content.get("text"):
                                    response_content += content["text"] + "\n"

                            # Prepare message to send to neighbors
                            broadcast_message = {
                                "from": self.id,
                                "content": response_content.strip(),
                            }
                            for neighbor in self.neighbors:
                                if not neighbor.input_queue.full():
                                    neighbor.input_queue.put_nowait(broadcast_message)
                                else:
                                    logger.warning(f"Message queue full for neighbor {neighbor.id}")

                    except Exception as e:
                        logger.error(f"Error processing message in node {self.id}: {str(e)}")

                else:
                    # Sleep when queue is empty to prevent busy waiting
                    time.sleep(MESSAGE_PROCESSING_DELAY)

            except Exception as e:
                logger.error(f"Error in message processing loop for node {self.id}: {str(e)}")
                time.sleep(MESSAGE_PROCESSING_DELAY)


class AgentGraph:
    def __init__(
        self,
        graph_id: str,
        topology_type: str,
        parent_agent: Any,
        model_provider: Optional[str] = None,
        model_settings: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
    ):
        self.graph_id = graph_id
        self.topology_type = topology_type
        self.parent_agent = parent_agent
        self.default_model_provider = model_provider  # Graph-level default
        self.default_model_settings = model_settings  # Graph-level default
        self.default_tools = tools  # Graph-level default tools
        self.nodes = {}
        self.channel = f"agent_graph_{graph_id}"
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
        self.lock = Lock()

    def add_node(
        self,
        node_id: str,
        role: str,
        system_prompt: str,
        model_provider: Optional[str] = None,
        model_settings: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
    ):
        with self.lock:
            # Use per-node config if provided, otherwise fall back to graph defaults
            effective_provider = model_provider or self.default_model_provider
            effective_settings = model_settings or self.default_model_settings
            effective_tools = tools or self.default_tools

            node = AgentNode(node_id, role, system_prompt, effective_provider, effective_settings, effective_tools)
            self.nodes[node_id] = node
            return node

    def add_edge(self, from_id: str, to_id: str):
        with self.lock:
            if from_id in self.nodes and to_id in self.nodes:
                self.nodes[from_id].add_neighbor(self.nodes[to_id])
                if self.topology_type == "mesh":
                    self.nodes[to_id].add_neighbor(self.nodes[from_id])

    def start(self):
        try:
            # Start processing threads for all nodes using thread pool
            with self.lock:
                for node in self.nodes.values():
                    node.thread = self.thread_pool.submit(node.process_messages, self.parent_agent)
        except Exception as e:
            logger.error(f"Error starting graph {self.graph_id}: {str(e)}")
            raise

    def stop(self):
        try:
            # Stop all nodes
            with self.lock:
                for node in self.nodes.values():
                    node.is_running = False

            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error stopping graph {self.graph_id}: {str(e)}")
            raise

    def send_message(self, target_id: str, message: str):
        try:
            with self.lock:
                if target_id in self.nodes:
                    if not self.nodes[target_id].input_queue.full():
                        self.nodes[target_id].input_queue.put_nowait({"content": message})
                        return True
                    else:
                        logger.warning(f"Message queue full for node {target_id}")
                        return False
                return False
        except Exception as e:
            logger.error(f"Error sending message to node {target_id}: {str(e)}")
            return False

    def get_status(self):
        with self.lock:
            status = {
                "graph_id": self.graph_id,
                "topology": self.topology_type,
                "default_model_provider": self.default_model_provider or "parent",
                "default_tools_count": len(self.default_tools) if self.default_tools else "parent",
                "nodes": [
                    {
                        "id": node.id,
                        "role": node.role,
                        "model_provider": node.model_provider or "default",
                        "tools_count": len(node.tools) if node.tools else "default",
                        "neighbors": [n.id for n in node.neighbors],
                        "queue_size": node.input_queue.qsize(),
                    }
                    for node in self.nodes.values()
                ],
            }
            return status


class AgentGraphManager:
    def __init__(self):
        self.graphs = {}
        self.lock = Lock()

    def create_graph(
        self,
        graph_id: str,
        topology: Dict,
        parent_agent: Any,
        model_provider: Optional[str] = None,
        model_settings: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
    ) -> Dict:
        with self.lock:
            if graph_id in self.graphs:
                return {
                    "status": "error",
                    "message": f"Graph {graph_id} already exists",
                }

            try:
                # Create new graph with model and tools configuration
                graph = AgentGraph(graph_id, topology["type"], parent_agent, model_provider, model_settings, tools)

                # Add nodes with per-node model and tools configuration support
                for node_def in topology["nodes"]:
                    # Extract per-node configuration
                    node_model_provider = node_def.get("model_provider")
                    node_model_settings = node_def.get("model_settings")
                    node_tools = node_def.get("tools")

                    graph.add_node(
                        node_def["id"],
                        node_def["role"],
                        node_def["system_prompt"],
                        node_model_provider,
                        node_model_settings,
                        node_tools,
                    )

                # Add edges
                if "edges" in topology:
                    for edge in topology["edges"]:
                        graph.add_edge(edge["from"], edge["to"])

                # Store graph
                self.graphs[graph_id] = graph

                # Start graph
                graph.start()

                # Create detailed success message
                config_info = ""
                if model_provider:
                    config_info += f" with default {model_provider} model"
                if tools:
                    config_info += f" and {len(tools)} default tools"

                # Count nodes with custom configurations
                custom_model_count = sum(1 for node_def in topology["nodes"] if node_def.get("model_provider"))
                custom_tools_count = sum(1 for node_def in topology["nodes"] if node_def.get("tools"))

                custom_info = []
                if custom_model_count > 0:
                    custom_info.append(f"{custom_model_count} nodes with custom models")
                if custom_tools_count > 0:
                    custom_info.append(f"{custom_tools_count} nodes with custom tools")

                if custom_info:
                    config_info += f" ({', '.join(custom_info)})"

                return {
                    "status": "success",
                    "message": f"Graph {graph_id}{config_info} created and started",
                }

            except Exception as e:
                return {"status": "error", "message": f"Error creating graph: {str(e)}"}

    def stop_graph(self, graph_id: str) -> Dict:
        with self.lock:
            if graph_id not in self.graphs:
                return {"status": "error", "message": f"Graph {graph_id} not found"}

            try:
                self.graphs[graph_id].stop()
                del self.graphs[graph_id]
                return {
                    "status": "success",
                    "message": f"Graph {graph_id} stopped and removed",
                }

            except Exception as e:
                return {"status": "error", "message": f"Error stopping graph: {str(e)}"}

    def send_message(self, graph_id: str, message: Dict) -> Dict:
        with self.lock:
            if graph_id not in self.graphs:
                return {"status": "error", "message": f"Graph {graph_id} not found"}

            try:
                graph = self.graphs[graph_id]
                if graph.send_message(message["target"], message["content"]):
                    return {
                        "status": "success",
                        "message": f"Message sent to node {message['target']}",
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Target node {message['target']} not found or queue full",
                    }

            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error sending message: {str(e)}",
                }

    def get_graph_status(self, graph_id: str) -> Dict:
        with self.lock:
            if graph_id not in self.graphs:
                return {"status": "error", "message": f"Graph {graph_id} not found"}

            try:
                status = self.graphs[graph_id].get_status()
                return {"status": "success", "data": status}

            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error getting graph status: {str(e)}",
                }

    def list_graphs(self) -> Dict:
        with self.lock:
            try:
                graphs = [
                    {
                        "graph_id": graph_id,
                        "topology": graph.topology_type,
                        "default_model_provider": graph.default_model_provider or "parent",
                        "default_tools_count": len(graph.default_tools) if graph.default_tools else "parent",
                        "node_count": len(graph.nodes),
                        "custom_model_nodes": sum(
                            1 for node in graph.nodes.values() if node.model_provider != graph.default_model_provider
                        ),
                        "custom_tools_nodes": sum(
                            1 for node in graph.nodes.values() if node.tools != graph.default_tools
                        ),
                    }
                    for graph_id, graph in self.graphs.items()
                ]

                return {"status": "success", "data": graphs}

            except Exception as e:
                return {"status": "error", "message": f"Error listing graphs: {str(e)}"}


# Global manager instance with thread-safe initialization
_MANAGER_LOCK = Lock()
_MANAGER = None


def get_manager() -> AgentGraphManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = AgentGraphManager()
        return _MANAGER


@tool
def agent_graph(
    action: str,
    graph_id: Optional[str] = None,
    topology: Optional[Dict] = None,
    message: Optional[Dict] = None,
    model_provider: Optional[str] = None,
    model_settings: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None,
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create and manage graphs of agents with different topologies and per-node model configuration.

    This function provides functionality to create and manage multi-agent systems with
    support for custom model providers and configurations. Each individual agent in the
    graph can use its own model provider and settings for maximum flexibility and optimization.

    How It Works:
    ------------
    1. Creates interconnected graphs of AI agents with specified topologies
    2. Each agent node runs in its own thread with message queues for communication
    3. Supports star, mesh, and hierarchical topologies for different use cases
    4. Each agent can use different model providers (bedrock, anthropic, ollama, etc.)
    5. Graph-level defaults with per-node overrides for flexible configuration
    6. Real-time message passing and status monitoring across the agent network

    Model Selection Process:
    ----------------------
    1. Per-node model config: If specified in node definition, uses that exact configuration
    2. Graph-level defaults: Falls back to graph-level model_provider and model_settings
    3. Parent agent model: If no configuration specified, uses parent agent's model
    4. Environment variables: When model_provider="env" at any level

    Per-Node Model Configuration:
    ---------------------------
    Each node in the topology can specify:
    - model_provider: Individual model provider ("bedrock", "anthropic", "ollama", etc.)
    - model_settings: Custom settings for that specific node's model

    Node format:
    {
        "id": "node_id",
        "role": "role_name",
        "system_prompt": "Agent instructions",
        "model_provider": "bedrock",  # Optional: Override graph default
        "model_settings": {"model_id": "claude-sonnet-4", "params": {"temperature": 0.7}}  # Optional
    }

    Common Use Cases:
    ---------------
    - Specialized agents: Different models optimized for specific tasks
    - Cost optimization: Use cheaper models for simple tasks, expensive for complex
    - Load distribution: Spread agents across different providers to prevent throttling
    - Performance tuning: Fast models for coordination, powerful models for analysis
    - Hybrid processing: Mix of cloud and local models in the same graph

    Args:
        action: Action to perform with the agent graph.
            Options: "create", "list", "stop", "message", "status"
        graph_id: Unique identifier for the agent graph (required for most actions).
        topology: Graph topology definition with type, nodes, and edges (required for create).
            Format: {
                "type": "star" | "mesh" | "hierarchical",
                "nodes": [
                    {
                        "id": str,
                        "role": str,
                        "system_prompt": str,
                        "model_provider": str (optional),
                        "model_settings": dict (optional),
                        "tools": list[str] (optional)
                    }, ...
                ],
                "edges": [{"from": str, "to": str}, ...] (optional for some topologies)
            }
        message: Message to send to the graph (required for message action).
            Format: {"target": "node_id", "content": "message text"}
        model_provider: Default model provider for all agents in the graph.
            Individual nodes can override this with their own model_provider.
            Options: "bedrock", "anthropic", "litellm", "llamaapi", "ollama", "openai", "github"
            Special values:
            - None: Use parent agent's model (default)
            - "env": Use environment variables to determine provider
        model_settings: Default model configuration for all agents in the graph.
            Individual nodes can override this with their own model_settings.
            Example: {"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0", "params": {"temperature": 1}}
        tools: Default list of tool names for all agents in the graph.
            Individual nodes can override this with their own tools list.
            Tool names must exist in the parent agent's tool registry.
            Examples: ["calculator", "file_read", "retrieve"]
            If not provided at any level, inherits all tools from the parent agent.
        agent: The parent agent (automatically passed by Strands framework).

    Returns:
        Dict containing status and response content in the format:
        {
            "status": "success|error",
            "content": [{"text": "Operation result message"}]
        }

        Success case: Returns operation confirmation with model configuration details
        Error case: Returns information about what went wrong during processing

    Environment Variables for Model Switching:
    ----------------------------------------
    When model_provider="env" at graph or node level, these variables are used:
    - STRANDS_PROVIDER: Model provider name
    - STRANDS_MODEL_ID: Specific model identifier
    - STRANDS_MAX_TOKENS: Maximum tokens to generate
    - STRANDS_TEMPERATURE: Sampling temperature
    - Provider-specific keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)

    Examples:
    --------
    # Mixed model configuration - each agent uses optimal model for its role
    result = agent.tool.agent_graph(
        action="create",
        graph_id="optimized_team",
        topology={
            "type": "star",
            "nodes": [
                {
                    "id": "coordinator",
                    "role": "coordinator",
                    "system_prompt": "You coordinate and delegate tasks efficiently.",
                    "model_provider": "bedrock",
                    # Powerful for coordination
                    "model_settings": {"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"}
                },
                {
                    "id": "fast_analyst",
                    "role": "analyst",
                    "system_prompt": "You do quick data analysis.",
                    "model_provider": "bedrock",
                    "model_settings": {"model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0"}  # Fast and cheap
                },
                {
                    "id": "deep_thinker",
                    "role": "researcher",
                    "system_prompt": "You do deep research and complex reasoning.",
                    "model_provider": "bedrock",
                    "model_settings": {"model_id": "claude-opus-4-20250514"}  # Most powerful
                },
                {
                    "id": "local_processor",
                    "role": "processor",
                    "system_prompt": "You process data locally for privacy.",
                    "model_provider": "ollama",
                    "model_settings": {"model_id": "llama3", "host": "http://localhost:11434"}  # Local
                }
            ],
            "edges": [
                {"from": "coordinator", "to": "fast_analyst"},
                {"from": "coordinator", "to": "deep_thinker"},
                {"from": "coordinator", "to": "local_processor"}
            ]
        }
    )

    # Graph-level defaults with individual overrides
    result = agent.tool.agent_graph(
        action="create",
        graph_id="hybrid_graph",
        model_provider="anthropic",  # Default for most nodes
        model_settings={"model_id": "claude-opus-4-20250514"},
        topology={
            "type": "mesh",
            "nodes": [
                {
                    "id": "researcher",
                    "role": "researcher",
                    "system_prompt": "You research topics."
                    # Uses graph-level anthropic model
                },
                {
                    "id": "specialist",
                    "role": "specialist",
                    "system_prompt": "You provide specialized analysis.",
                    "model_provider": "bedrock",  # Override: uses bedrock instead
                    "model_settings": {"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"}
                }
            ]
        }
    )

    # Send message to specific agent
    result = agent.tool.agent_graph(
        action="message",
        graph_id="optimized_team",
        message={"target": "deep_thinker", "content": "Research quantum computing applications"}
    )

    # Get detailed graph status (shows model configuration per node)
    result = agent.tool.agent_graph(action="status", graph_id="optimized_team")

    # List all active graphs with model information
    result = agent.tool.agent_graph(action="list")

    # Stop and remove graph
    result = agent.tool.agent_graph(action="stop", graph_id="optimized_team")

    # Tools configuration examples - per-node tools for security and specialization
    result = agent.tool.agent_graph(
        action="create",
        graph_id="secure_workflow",
        tools=["retrieve", "memory"],  # Graph default: basic tools
        topology={
            "type": "star",
            "nodes": [
                {
                    "id": "coordinator",
                    "role": "coordinator",
                    "system_prompt": "You coordinate tasks efficiently.",
                    "tools": ["agent_graph", "slack"]  # Only coordination tools
                },
                {
                    "id": "file_processor",
                    "role": "processor",
                    "system_prompt": "You process files securely.",
                    "tools": ["file_read", "file_write"]  # Only file operations
                },
                {
                    "id": "calculator",
                    "role": "analyst",
                    "system_prompt": "You perform calculations and analysis.",
                    "tools": ["calculator", "python_repl"]  # Only computation tools
                }
            ],
            "edges": [
                {"from": "coordinator", "to": "file_processor"},
                {"from": "coordinator", "to": "calculator"}
            ]
        }
    )

    # Mixed model and tools configuration for optimal performance
    result = agent.tool.agent_graph(
        action="create",
        graph_id="optimized_workflow",
        model_provider="bedrock",
        tools=["retrieve", "memory"],  # Default tools for all agents
        topology={
            "type": "hierarchical",
            "nodes": [
                {
                    "id": "manager",
                    "role": "manager",
                    "system_prompt": "You manage the workflow efficiently.",
                    "model_settings": {"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"},  # Powerful model
                    "tools": ["agent_graph", "workflow", "slack"]  # Management tools
                },
                {
                    "id": "researcher",
                    "role": "researcher",
                    "system_prompt": "You research topics thoroughly.",
                    "model_settings": {"model_id": "claude-opus-4-20250514"},  # Most powerful
                    "tools": ["retrieve", "http_request", "file_read"]  # Research tools
                },
                {
                    "id": "writer",
                    "role": "writer",
                    "system_prompt": "You create content efficiently.",
                    "model_settings": {"model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0"},  # Fast model
                    "tools": ["file_write", "editor", "generate_image"]  # Content creation tools
                }
            ]
        }
    )

    Notes:
        - Per-node model and tools configuration allows fine-grained optimization
        - Graph-level defaults provide convenience for homogeneous graphs
        - Per-node tools enable security isolation and role-based access control
        - Tools filtering allows specialized agents with minimal attack surface
        - Model switching requires the appropriate dependencies per provider
        - Graphs run in separate threads with proper resource management
        - Message queues prevent blocking and handle backpressure
        - Performance scales with complexity and diversity of models used
        - Cost optimization through strategic model selection per agent role
    """
    console = console_util.create()

    try:
        # Get manager instance thread-safely
        manager = get_manager()

        if action == "create":
            if not graph_id or not topology:
                return {
                    "status": "error",
                    "content": [{"text": "graph_id and topology are required for create action"}],
                }

            result = manager.create_graph(graph_id, topology, agent, model_provider, model_settings, tools)
            if result["status"] == "success":
                # Count nodes with custom configurations for display
                custom_model_count = sum(1 for node_def in topology["nodes"] if node_def.get("model_provider"))
                custom_tools_count = sum(1 for node_def in topology["nodes"] if node_def.get("tools"))

                panel_content = (
                    f"✅ {result['message']}\n\n[bold blue]Graph ID:[/bold blue] {graph_id}\n"
                    f"[bold blue]Topology:[/bold blue] {topology['type']}\n"
                    f"[bold blue]Total Nodes:[/bold blue] {len(topology['nodes'])}\n"
                    f"[bold blue]Default Model:[/bold blue] {model_provider or 'parent'}\n"
                    f"[bold blue]Default Tools:[/bold blue] {len(tools) if tools else 'parent'}\n"
                    f"[bold blue]Custom Models:[/bold blue] {custom_model_count} nodes\n"
                    f"[bold blue]Custom Tools:[/bold blue] {custom_tools_count} nodes"
                )
                panel = Panel(panel_content, title="Multi-Agent Graph Created", box=ROUNDED)
                with console.capture() as capture:
                    console.print(panel)
                result["rich_output"] = capture.get()

        elif action == "stop":
            if not graph_id:
                return {
                    "status": "error",
                    "content": [{"text": "graph_id is required for stop action"}],
                }

            result = manager.stop_graph(graph_id)
            if result["status"] == "success":
                panel_content = f"🛑 {result['message']}"
                panel = Panel(panel_content, title="Graph Stopped", box=ROUNDED)
                with console.capture() as capture:
                    console.print(panel)
                result["rich_output"] = capture.get()

        elif action == "message":
            if not graph_id or not message:
                return {
                    "status": "error",
                    "content": [{"text": "graph_id and message are required for message action"}],
                }

            result = manager.send_message(graph_id, message)
            if result["status"] == "success":
                panel_content = (
                    f"📨 {result['message']}\n\n"
                    f"[bold blue]To:[/bold blue] {message['target']}\n"
                    f"[bold blue]Content:[/bold blue] {message['content'][:100]}..."
                )
                panel = Panel(panel_content, title="Message Sent", box=ROUNDED)
                with console.capture() as capture:
                    console.print(panel)
                result["rich_output"] = capture.get()

        elif action == "status":
            if not graph_id:
                return {
                    "status": "error",
                    "content": [{"text": "graph_id is required for status action"}],
                }

            result = manager.get_graph_status(graph_id)
            if result["status"] == "success":
                result["rich_output"] = create_rich_status_panel(console, result["data"])

        elif action == "list":
            result = manager.list_graphs()
            if result["status"] == "success":
                headers = [
                    "Graph ID",
                    "Topology",
                    "Default Model",
                    "Default Tools",
                    "Total Nodes",
                    "Custom Models",
                    "Custom Tools",
                ]
                rows = [
                    [
                        graph["graph_id"],
                        graph["topology"],
                        graph["default_model_provider"],
                        str(graph["default_tools_count"]),
                        str(graph["node_count"]),
                        str(graph["custom_model_nodes"]),
                        str(graph["custom_tools_nodes"]),
                    ]
                    for graph in result["data"]
                ]
                result["rich_output"] = create_rich_table(console, "Multi-Agent Graphs", headers, rows)

        else:
            return {
                "status": "error",
                "content": [{"text": f"Unknown action: {action}"}],
            }

        # Process result
        if result["status"] == "success":
            # Prepare clean message text without rich formatting
            if "data" in result:
                clean_message = f"Operation {action} completed successfully."
                if action == "create":
                    custom_model_count = sum(1 for node_def in topology["nodes"] if node_def.get("model_provider"))
                    custom_tools_count = sum(1 for node_def in topology["nodes"] if node_def.get("tools"))

                    config_info = ""
                    if model_provider:
                        config_info += f" with {model_provider} model"
                    if tools:
                        config_info += f" and {len(tools)} tools"

                    custom_info = []
                    if custom_model_count > 0:
                        custom_info.append(f"{custom_model_count} custom models")
                    if custom_tools_count > 0:
                        custom_info.append(f"{custom_tools_count} custom tools")

                    if custom_info:
                        config_info += f" ({', '.join(custom_info)})"

                    clean_message = f"Graph {graph_id}{config_info} created with {len(topology['nodes'])} nodes."
                elif action == "stop":
                    clean_message = f"Graph {graph_id} stopped and removed."
                elif action == "message":
                    clean_message = f"Message sent to {message['target']} in graph {graph_id}."
                elif action == "status":
                    clean_message = f"Graph {graph_id} status retrieved."
                elif action == "list":
                    graph_count = len(result["data"])
                    clean_message = f"Listed {graph_count} active multi-agent graphs."
            else:
                clean_message = result.get("message", "Operation completed successfully.")

            # Store only clean text in content for agent.messages
            content = [{"text": clean_message}]

            return {"status": "success", "content": content}
        else:
            error_message = f"❌ Error: {result['message']}"
            logger.error(error_message)
            return {
                "status": "error",
                "content": [{"text": error_message}],
            }

    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{error_trace}"
        logger.error(f"\n[AGENT GRAPH TOOL ERROR]\n{error_msg}")
        return {
            "status": "error",
            "content": [{"text": f"⚠️ Agent Graph Error: {str(e)}"}],
        }
