"""Graph tool using new Strands SDK Graph implementation.

This module provides functionality to create and manage multi-agent systems
using the new Strands SDK Graph implementation. Unlike the old message-passing approach,
this uses deterministic DAG execution with output propagation.

Usage with Strands Agent:

```python
from strands import Agent
from graph import graph

agent = Agent(tools=[graph])

# Create a agent graph
result = agent.tool.graph(
    action="create",
    graph_id="analysis_pipeline",
    topology={
        "nodes": [
            {
                "id": "researcher",
                "role": "researcher",
                "system_prompt": "You research topics thoroughly.",
                "model_provider": "bedrock",
                "model_settings": {"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"}
            },
            {
                "id": "analyst",
                "role": "analyst",
                "system_prompt": "You analyze research data.",
                "model_provider": "bedrock",
                "model_settings": {"model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0"}
            },
            {
                "id": "reporter",
                "role": "reporter",
                "system_prompt": "You create comprehensive reports.",
                "tools": ["file_write", "editor"]
            }
        ],
        "edges": [
            {"from": "researcher", "to": "analyst"},
            {"from": "analyst", "to": "reporter"}
        ],
        "entry_points": ["researcher"]
    }
)

# Execute a task through the graph
result = agent.tool.graph(
    action="execute",
    graph_id="analysis_pipeline",
    task="Research and analyze the impact of AI on healthcare"
)
```
"""

import datetime
import logging
import time
import traceback
from typing import Any, Dict, List, Optional

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from strands import Agent, tool
from strands.multiagent.graph import GraphBuilder

from strands_tools.utils import console_util
from strands_tools.utils.models.model import create_model

logger = logging.getLogger(__name__)


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


def create_rich_status_panel(console: Console, status: Dict) -> str:
    """Create a rich formatted status panel"""
    content = []
    content.append(f"[bold blue]Graph ID:[/bold blue] {status['graph_id']}")
    content.append(f"[bold blue]Total Nodes:[/bold blue] {status['total_nodes']}")
    content.append(
        f"[bold blue]Entry Points:[/bold blue] {', '.join([ep['node_id'] for ep in status['entry_points']])}"
    )
    content.append(f"[bold blue]Status:[/bold blue] {status['execution_status']}")

    if status.get("last_execution"):
        exec_info = status["last_execution"]
        content.append("\n[bold magenta]Last Execution:[/bold magenta]")
        content.append(f"  [bold green]Completed Nodes:[/bold green] {exec_info['completed_nodes']}")
        content.append(f"  [bold green]Failed Nodes:[/bold green] {exec_info['failed_nodes']}")
        content.append(f"  [bold green]Execution Time:[/bold green] {exec_info['execution_time']}ms")

    content.append("\n[bold magenta]Nodes:[/bold magenta]")
    for node_info in status["nodes"]:
        node_content = [
            f"  [bold green]ID:[/bold green] {node_info['id']}",
            f"  [bold green]Role:[/bold green] {node_info['role']}",
            f"  [bold green]Model:[/bold green] {node_info.get('model_provider', 'default')}",
            f"  [bold green]Tools:[/bold green] {node_info.get('tools_count', 'default')}",
            f"  [bold green]Dependencies:[/bold green] {len(node_info.get('dependencies', []))}",
            "",
        ]
        content.extend(node_content)

    panel = Panel("\n".join(content), title="Graph Status", box=ROUNDED)
    with console.capture() as capture:
        console.print(panel)
    return capture.get()


def create_agent_with_model(
    system_prompt: str,
    model_provider: Optional[str] = None,
    model_settings: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None,
    parent_agent: Optional[Agent] = None,
) -> Agent:
    """Create an Agent with custom model configuration.

    Args:
        system_prompt: System prompt for the new agent
        model_provider: Model provider to use
        model_settings: Custom model settings
        tools: List of tool names to include
        parent_agent: Parent agent to inherit from

    Returns:
        Configured Agent instance
    """
    # Create model
    model = create_model(provider=model_provider, config=model_settings)

    # Determine tools
    agent_tools = []
    if parent_agent:
        if tools:
            # Filter parent agent tools to only include specified tool names
            for tool_name in tools:
                if tool_name in parent_agent.tool_registry.registry:
                    agent_tools.append(parent_agent.tool_registry.registry[tool_name])
                else:
                    logger.warning(f"Tool '{tool_name}' not found in parent agent's tool registry")
        else:
            # Use all parent agent tools
            agent_tools = list(parent_agent.tool_registry.registry.values())

    # Create and return agent
    kwargs = {}
    if parent_agent:
        kwargs["trace_attributes"] = parent_agent.trace_attributes
        kwargs["callback_handler"] = parent_agent.callback_handler

    return Agent(system_prompt=system_prompt, model=model, tools=agent_tools, **kwargs)


class GraphManager:
    """Manager for SDK-based Graph instances"""

    def __init__(self):
        self.graphs: Dict[str, Dict] = {}  # graph_id -> {graph: Graph, metadata: dict}

    def create_graph(
        self,
        graph_id: str,
        topology: Dict,
        parent_agent: Agent,
        model_provider: Optional[str] = None,
        model_settings: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None,
    ) -> Dict:
        """Create a new Graph using SDK GraphBuilder"""

        if graph_id in self.graphs:
            return {"status": "error", "message": f"Graph {graph_id} already exists"}

        try:
            # Create GraphBuilder
            builder = GraphBuilder()

            # Create agents for each node
            node_agents = {}
            for node_def in topology["nodes"]:
                # Determine effective configuration for this node
                effective_model_provider = node_def.get("model_provider") or model_provider
                effective_model_settings = node_def.get("model_settings") or model_settings
                effective_tools = node_def.get("tools") or tools

                # Create specialized agent for this node
                if effective_model_provider or effective_model_settings:
                    # Create agent with custom model configuration
                    node_agent = create_agent_with_model(
                        system_prompt=node_def["system_prompt"],
                        model_provider=effective_model_provider,
                        model_settings=effective_model_settings,
                        tools=effective_tools,
                        parent_agent=parent_agent,
                    )
                else:
                    # Create basic agent with parent agent's model and tools
                    # Get all tools from parent agent if no specific tools configuration
                    parent_tools = (
                        list(parent_agent.tool_registry.registry.values()) if parent_agent.tool_registry else []
                    )
                    node_agent = Agent(
                        system_prompt=node_def["system_prompt"],
                        model=parent_agent.model,
                        tools=parent_tools,
                    )

                node_agents[node_def["id"]] = node_agent

                # Add node to builder
                builder.add_node(node_agent, node_def["id"])

            # Add edges
            for edge in topology.get("edges", []):
                builder.add_edge(edge["from"], edge["to"])

            # Set entry points
            for entry_point in topology.get("entry_points", []):
                builder.set_entry_point(entry_point)

            # Build the graph
            graph = builder.build()

            # Store graph with metadata
            self.graphs[graph_id] = {
                "graph": graph,
                "metadata": {
                    "graph_id": graph_id,
                    "created_at": time.time(),
                    "node_count": len(topology["nodes"]),
                    "edge_count": len(topology.get("edges", [])),
                    "entry_points": topology.get("entry_points", []),
                    "topology": topology,
                    "last_execution": None,
                },
            }

            return {
                "status": "success",
                "message": f"Graph {graph_id} created successfully with {len(topology['nodes'])} nodes",
            }

        except Exception as e:
            logger.error(f"Error creating graph {graph_id}: {str(e)}")
            return {"status": "error", "message": f"Error creating graph: {str(e)}"}

    def execute_graph(self, graph_id: str, task: str) -> Dict:
        """Execute a graph with the given task"""

        if graph_id not in self.graphs:
            return {"status": "error", "message": f"Graph {graph_id} not found"}

        try:
            graph_info = self.graphs[graph_id]
            graph = graph_info["graph"]

            # Execute the graph
            start_time = time.time()
            result = graph(task)
            execution_time = round((time.time() - start_time) * 1000)

            # Update metadata with execution info
            graph_info["metadata"]["last_execution"] = {
                "task": task,
                "status": result.status.value,
                "completed_nodes": result.completed_nodes,
                "failed_nodes": result.failed_nodes,
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

            # Extract results text
            results_text = []
            for node_id, node_result in result.results.items():
                agent_results = node_result.get_agent_results()
                for agent_result in agent_results:
                    results_text.append(f"Node {node_id}: {str(agent_result)}")

            return {
                "status": "success",
                "message": f"Graph {graph_id} executed successfully",
                "data": {
                    "execution_time": execution_time,
                    "completed_nodes": result.completed_nodes,
                    "failed_nodes": result.failed_nodes,
                    "results": results_text,
                },
            }

        except Exception as e:
            logger.error(f"Error executing graph {graph_id}: {str(e)}")
            return {"status": "error", "message": f"Error executing graph: {str(e)}"}

    def get_graph_status(self, graph_id: str) -> Dict:
        """Get status of a specific graph"""

        if graph_id not in self.graphs:
            return {"status": "error", "message": f"Graph {graph_id} not found"}

        try:
            graph_info = self.graphs[graph_id]
            metadata = graph_info["metadata"]
            topology = metadata["topology"]

            # Build status information
            status = {
                "graph_id": graph_id,
                "total_nodes": metadata["node_count"],
                "entry_points": [{"node_id": ep} for ep in metadata["entry_points"]],
                "execution_status": "ready",
                "last_execution": metadata.get("last_execution"),
                "nodes": [],
            }

            # Add node information
            for node_def in topology["nodes"]:
                node_info = {
                    "id": node_def["id"],
                    "role": node_def["role"],
                    "model_provider": node_def.get("model_provider", "default"),
                    "tools_count": (len(node_def.get("tools", [])) if node_def.get("tools") else "default"),
                    "dependencies": [],
                }

                # Find dependencies for this node
                for edge in topology.get("edges", []):
                    if edge["to"] == node_def["id"]:
                        node_info["dependencies"].append(edge["from"])

                status["nodes"].append(node_info)

            return {"status": "success", "data": status}

        except Exception as e:
            logger.error(f"Error getting graph status {graph_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting graph status: {str(e)}",
            }

    def list_graphs(self) -> Dict:
        """List all graphs"""

        try:
            graphs_list = []
            for graph_id, graph_info in self.graphs.items():
                metadata = graph_info["metadata"]
                graph_summary = {
                    "graph_id": graph_id,
                    "node_count": metadata["node_count"],
                    "edge_count": metadata["edge_count"],
                    "entry_points": len(metadata["entry_points"]),
                    "created_at": metadata["created_at"],
                    "last_executed": (
                        metadata.get("last_execution", {}).get("timestamp") if metadata.get("last_execution") else None
                    ),
                }
                graphs_list.append(graph_summary)

            return {"status": "success", "data": graphs_list}

        except Exception as e:
            logger.error(f"Error listing graphs: {str(e)}")
            return {"status": "error", "message": f"Error listing graphs: {str(e)}"}

    def delete_graph(self, graph_id: str) -> Dict:
        """Delete a graph"""

        if graph_id not in self.graphs:
            return {"status": "error", "message": f"Graph {graph_id} not found"}

        try:
            del self.graphs[graph_id]
            return {
                "status": "success",
                "message": f"Graph {graph_id} deleted successfully",
            }

        except Exception as e:
            logger.error(f"Error deleting graph {graph_id}: {str(e)}")
            return {"status": "error", "message": f"Error deleting graph: {str(e)}"}


# Global manager instance
_manager = GraphManager()


@tool
def graph(
    action: str,
    graph_id: Optional[str] = None,
    topology: Optional[Dict] = None,
    task: Optional[str] = None,
    model_provider: Optional[str] = None,
    model_settings: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None,
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create and manage multi-agent graphs using Strands SDK Graph implementation.

    This function provides functionality to create and manage multi-agent systems using
    the new Strands SDK Graph implementation. Unlike the old message-passing approach,
    this uses deterministic DAG (Directed Acyclic Graph) execution with output propagation.

    How It Works:
    ------------
    1. Creates graphs where agents are nodes with dependency relationships
    2. Execution follows topological order based on dependencies
    3. Output from one agent propagates as input to dependent agents
    4. Supports conditional routing and parallel execution where possible
    5. Each agent can use different model providers and configurations

    Key Differences from Old agent_graph:
    -----------------------------------
    - **Execution Model**: Task execution vs persistent message-passing
    - **Communication**: Output propagation vs real-time message queues
    - **Lifecycle**: Task-based execution vs long-running agent networks
    - **Architecture**: Uses SDK Graph classes vs custom implementation

    Args:
        action: Action to perform with the graph.
            Options: "create", "execute", "status", "list", "delete"
        graph_id: Unique identifier for the graph (required for most actions).
        topology: Graph topology definition (required for create).
            Format: {
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
                "edges": [{"from": str, "to": str}, ...],
                "entry_points": [str, ...] (optional, auto-detected if not provided)
            }
        task: Task to execute through the graph (required for execute action).
        model_provider: Default model provider for all agents in the graph.
            Individual nodes can override this with their own model_provider.
            Options: "bedrock", "anthropic", "litellm", "ollama", "openai", etc.
        model_settings: Default model configuration for all agents.
            Individual nodes can override this with their own model_settings.
            Example: {"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"}
        tools: Default list of tool names for all agents.
            Individual nodes can override this with their own tools list.
        agent: The parent agent (automatically passed by Strands framework).

    Returns:
        Dict containing status and response content in the format:
        {
            "status": "success|error",
            "content": [{"text": "Operation result message"}]
        }

    Examples:
    --------
    # Create a research pipeline
    result = agent.tool.graph(
        action="create",
        graph_id="research_pipeline",
        topology={
            "nodes": [
                {
                    "id": "researcher",
                    "role": "researcher",
                    "system_prompt": "You research topics thoroughly.",
                    "model_provider": "bedrock",
                    "model_settings": {"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"}
                },
                {
                    "id": "analyst",
                    "role": "analyst",
                    "system_prompt": "You analyze research data.",
                    "model_provider": "bedrock",
                    "model_settings": {"model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0"}
                },
                {
                    "id": "reporter",
                    "role": "reporter",
                    "system_prompt": "You create comprehensive reports.",
                    "tools": ["file_write", "editor"]
                }
            ],
            "edges": [
                {"from": "researcher", "to": "analyst"},
                {"from": "analyst", "to": "reporter"}
            ],
            "entry_points": ["researcher"]
        }
    )

    # Execute a task through the graph
    result = agent.tool.graph(
        action="execute",
        graph_id="research_pipeline",
        task="Research and analyze the impact of AI on healthcare"
    )

    # Get graph status
    result = agent.tool.graph(action="status", graph_id="research_pipeline")

    # List all graphs
    result = agent.tool.graph(action="list")

    # Delete a graph
    result = agent.tool.graph(action="delete", graph_id="research_pipeline")

    Notes:
        - Graphs execute tasks deterministically based on DAG structure
        - Entry points receive the original task; other nodes receive dependency outputs
        - Per-node model and tool configuration enables optimization and specialization
        - Execution is task-based rather than persistent like the old agent_graph
        - Uses the new Strands SDK Graph implementation for reliability and performance
    """
    console = console_util.create()

    try:
        if action == "create":
            if not graph_id or not topology:
                return {
                    "status": "error",
                    "content": [{"text": "graph_id and topology are required for create action"}],
                }

            result = _manager.create_graph(graph_id, topology, agent, model_provider, model_settings, tools)

            if result["status"] == "success":
                node_count = len(topology["nodes"])
                edge_count = len(topology.get("edges", []))
                entry_count = len(topology.get("entry_points", []))

                panel_content = (
                    f"✅ {result['message']}\n\n"
                    f"[bold blue]Graph ID:[/bold blue] {graph_id}\n"
                    f"[bold blue]Nodes:[/bold blue] {node_count}\n"
                    f"[bold blue]Edges:[/bold blue] {edge_count}\n"
                    f"[bold blue]Entry Points:[/bold blue] {entry_count}\n"
                    f"[bold blue]Default Model:[/bold blue] {model_provider or 'parent'}\n"
                    f"[bold blue]Default Tools:[/bold blue] {len(tools) if tools else 'parent'}"
                )

                panel = Panel(panel_content, title="Graph Created", box=ROUNDED)
                with console.capture() as capture:
                    console.print(panel)
                result["rich_output"] = capture.get()

        elif action == "execute":
            if not graph_id or not task:
                return {
                    "status": "error",
                    "content": [{"text": "graph_id and task are required for execute action"}],
                }

            result = _manager.execute_graph(graph_id, task)

            if result["status"] == "success":
                data = result["data"]
                panel_content = (
                    f"🚀 Graph execution completed successfully!\n\n"
                    f"[bold blue]Graph ID:[/bold blue] {graph_id}\n"
                    f"[bold blue]Task:[/bold blue] {task[:100]}{'...' if len(task) > 100 else ''}\n"
                    f"[bold blue]Execution Time:[/bold blue] {data['execution_time']}ms\n"
                    f"[bold blue]Completed Nodes:[/bold blue] {data['completed_nodes']}\n"
                    f"[bold blue]Failed Nodes:[/bold blue] {data['failed_nodes']}\n\n"
                    f"[bold magenta]Results:[/bold magenta]\n"
                )

                for result_text in data["results"][:3]:  # Show first 3 results
                    panel_content += f"{result_text[:200]}{'...' if len(result_text) > 200 else ''}\n"

                if len(data["results"]) > 3:
                    panel_content += f"... and {len(data['results']) - 3} more results"

                panel = Panel(panel_content, title="Graph Execution Complete", box=ROUNDED)
                with console.capture() as capture:
                    console.print(panel)
                result["rich_output"] = capture.get()

        elif action == "status":
            if not graph_id:
                return {
                    "status": "error",
                    "content": [{"text": "graph_id is required for status action"}],
                }

            result = _manager.get_graph_status(graph_id)
            if result["status"] == "success":
                result["rich_output"] = create_rich_status_panel(console, result["data"])

        elif action == "list":
            result = _manager.list_graphs()
            if result["status"] == "success":
                headers = [
                    "Graph ID",
                    "Nodes",
                    "Edges",
                    "Entry Points",
                    "Last Executed",
                ]
                rows = []
                for graph_data in result["data"]:
                    last_exec = "Never"
                    if graph_data["last_executed"]:
                        last_exec = datetime.datetime.fromtimestamp(graph_data["last_executed"]).strftime(
                            "%Y-%m-%d %H:%M"
                        )

                    rows.append(
                        [
                            graph_data["graph_id"],
                            str(graph_data["node_count"]),
                            str(graph_data["edge_count"]),
                            str(graph_data["entry_points"]),
                            last_exec,
                        ]
                    )
                result["rich_output"] = create_rich_table(console, "Graphs", headers, rows)

        elif action == "delete":
            if not graph_id:
                return {
                    "status": "error",
                    "content": [{"text": "graph_id is required for delete action"}],
                }

            result = _manager.delete_graph(graph_id)
            if result["status"] == "success":
                panel_content = f"🗑️ {result['message']}"
                panel = Panel(panel_content, title="Graph Deleted", box=ROUNDED)
                with console.capture() as capture:
                    console.print(panel)
                result["rich_output"] = capture.get()

        else:
            return {
                "status": "error",
                "content": [
                    {"text": f"Unknown action: {action}. Valid actions: create, execute, status, list, delete"}
                ],
            }

        # Process result for clean response
        if result["status"] == "success":
            if "data" in result:
                if action == "create":
                    clean_message = f"Graph {graph_id} created with {len(topology['nodes'])} nodes."
                elif action == "execute":
                    clean_message = f"Graph {graph_id} executed successfully in {result['data']['execution_time']}ms."
                elif action == "status":
                    clean_message = f"Graph {graph_id} status retrieved."
                elif action == "list":
                    clean_message = f"Listed {len(result['data'])} graphs."
                elif action == "delete":
                    clean_message = f"Graph {graph_id} deleted successfully."
                else:
                    clean_message = result.get("message", "Operation completed successfully.")
            else:
                clean_message = result.get("message", "Operation completed successfully.")

            return {"status": "success", "content": [{"text": clean_message}]}
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
        logger.error(f"\n[GRAPH TOOL ERROR]\n{error_msg}")
        return {
            "status": "error",
            "content": [{"text": f"⚠️ Graph Error: {str(e)}"}],
        }
