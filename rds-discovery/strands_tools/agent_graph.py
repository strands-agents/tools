import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Lock
from typing import Any, Dict, List

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from strands.types.tools import ToolResult, ToolUse

from strands_tools.use_llm import use_llm
from strands_tools.utils import console_util

logger = logging.getLogger(__name__)


# Constants for resource management
MAX_THREADS = 10
MESSAGE_PROCESSING_DELAY = 0.1  # seconds
MAX_QUEUE_SIZE = 1000

TOOL_SPEC = {
    "name": "agent_graph",
    "description": """Create and manage graphs of agents with different topologies and communication patterns.

Key Features:
1. Multiple topology support (star, mesh, hierarchical)
2. Dynamic message routing
3. Parallel agent execution
4. Real-time status monitoring
5. Flexible agent configuration

Example Usage:

1. Create a new agent graph:
{
    "action": "create",
    "graph_id": "analysis_graph",
    "topology": {
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
}

2. Send a message:
{
    "action": "message",
    "graph_id": "analysis_graph",
    "message": {
        "target": "agent1",
        "content": "Analyze this data pattern..."
    }
}

3. Check graph status:
{
    "action": "status",
    "graph_id": "analysis_graph"
}

4. List all graphs:
{
    "action": "list"
}

5. Stop a graph:
{
    "action": "stop",
    "graph_id": "analysis_graph"
}

Topology Types:
- star: Central node with radiating connections
- mesh: All nodes connected to each other
- hierarchical: Tree-like structure with parent-child relationships

Node Configuration:
- id: Unique identifier for the node
- role: Function/purpose of the agent
- system_prompt: Agent's system instructions""",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "stop", "message", "status"],
                    "description": "Action to perform with the agent graph",
                },
                "graph_id": {
                    "type": "string",
                    "description": "Unique identifier for the agent graph",
                },
                "topology": {
                    "type": "object",
                    "description": "Graph topology definition with type, nodes, and edges",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["star", "mesh", "hierarchical"],
                            "description": "Type of graph topology",
                        },
                        "nodes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "role": {"type": "string"},
                                    "system_prompt": {"type": "string"},
                                },
                            },
                            "description": "List of agent nodes",
                        },
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string"},
                                    "to": {"type": "string"},
                                },
                            },
                            "description": "List of connections between nodes",
                        },
                    },
                },
                "message": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Target node ID"},
                        "content": {"type": "string", "description": "Message content"},
                    },
                    "description": "Message to send to the graph",
                },
            },
            "required": ["action"],
        }
    },
}


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
    content.append("\n[bold magenta]Nodes:[/bold magenta]")

    for node in status["nodes"]:
        node_info = [
            f"  [bold green]ID:[/bold green] {node['id']}",
            f"  [bold green]Role:[/bold green] {node['role']}",
            f"  [bold green]Queue Size:[/bold green] {node['queue_size']}",
            f"  [bold green]Neighbors:[/bold green] {', '.join(node['neighbors'])}\n",
        ]
        content.extend(node_info)

    panel = Panel("\n".join(content), title="Graph Status", box=ROUNDED)
    with console.capture() as capture:
        console.print(panel)
    return capture.get()


class AgentNode:
    def __init__(self, node_id: str, role: str, system_prompt: str):
        self.id = node_id
        self.role = role
        self.system_prompt = system_prompt
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

    def process_messages(self, tool_context: Dict[str, Any], channel: str):
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
                        # Process message with LLM
                        result = use_llm(
                            {
                                "toolUseId": str(uuid.uuid4()),
                                "input": {
                                    "system_prompt": self.system_prompt,
                                    "prompt": message["content"],
                                },
                            },
                            **tool_context,
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
    def __init__(self, graph_id: str, topology_type: str, tool_context: Dict[str, Any]):
        self.graph_id = graph_id
        self.topology_type = topology_type
        self.nodes = {}
        self.tool_context = tool_context
        self.channel = f"agent_graph_{graph_id}"
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
        self.lock = Lock()

    def add_node(self, node_id: str, role: str, system_prompt: str):
        with self.lock:
            node = AgentNode(node_id, role, system_prompt)
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
                    node.thread = self.thread_pool.submit(node.process_messages, self.tool_context, self.channel)
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
                "nodes": [
                    {
                        "id": node.id,
                        "role": node.role,
                        "neighbors": [n.id for n in node.neighbors],
                        "queue_size": node.input_queue.qsize(),
                    }
                    for node in self.nodes.values()
                ],
            }
            return status


class AgentGraphManager:
    def __init__(self, tool_context: Dict[str, Any]):
        self.graphs = {}
        self.tool_context = tool_context
        self.lock = Lock()

    def create_graph(self, graph_id: str, topology: Dict) -> Dict:
        with self.lock:
            if graph_id in self.graphs:
                return {
                    "status": "error",
                    "message": f"Graph {graph_id} already exists",
                }

            try:
                # Create new graph
                graph = AgentGraph(graph_id, topology["type"], self.tool_context)

                # Add nodes
                for node_def in topology["nodes"]:
                    graph.add_node(
                        node_def["id"],
                        node_def["role"],
                        node_def["system_prompt"],
                    )

                # Add edges
                if "edges" in topology:
                    for edge in topology["edges"]:
                        graph.add_edge(edge["from"], edge["to"])

                # Store graph
                self.graphs[graph_id] = graph

                # Start graph
                graph.start()

                return {
                    "status": "success",
                    "message": f"Graph {graph_id} created and started",
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
                        "node_count": len(graph.nodes),
                    }
                    for graph_id, graph in self.graphs.items()
                ]

                return {"status": "success", "data": graphs}

            except Exception as e:
                return {"status": "error", "message": f"Error listing graphs: {str(e)}"}


# Global manager instance with thread-safe initialization
_MANAGER_LOCK = Lock()
_MANAGER = None


def get_manager(tool_context: Dict[str, Any]) -> AgentGraphManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = AgentGraphManager(tool_context)
        return _MANAGER


def agent_graph(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Create and manage graphs of AI agents.
    """
    console = console_util.create()

    tool_use_id = tool.get("toolUseId", str(uuid.uuid4()))
    tool_input = tool.get("input", {})
    bedrock_client = kwargs.get("bedrock_client")
    system_prompt = kwargs.get("system_prompt")
    inference_config = kwargs.get("inference_config")
    messages = kwargs.get("messages")
    tool_config = kwargs.get("tool_config")

    logger.warning(
        "DEPRECATION WARNING: agent_graph will be removed in the next major release. "
        "Migration path: replace agent_graph calls with graph for equivalent functionality."
    )

    try:
        # Create tool context
        tool_context = {
            "bedrock_client": bedrock_client,
            "system_prompt": system_prompt,
            "inference_config": inference_config,
            "messages": messages,
            "tool_config": tool_config,
        }

        # Get manager instance thread-safely
        manager = get_manager(tool_context)

        action = tool_input.get("action")

        if action == "create":
            if "graph_id" not in tool_input or "topology" not in tool_input:
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": "graph_id and topology are required for create action"}],
                }

            result = manager.create_graph(tool_input["graph_id"], tool_input["topology"])
            if result["status"] == "success":
                panel_content = (
                    f"✅ {result['message']}\n\n[bold blue]Graph ID:[/bold blue] {tool_input['graph_id']}\n"
                    f"[bold blue]Topology:[/bold blue] {tool_input['topology']['type']}\n"
                    f"[bold blue]Nodes:[/bold blue] {len(tool_input['topology']['nodes'])}"
                )
                panel = Panel(panel_content, title="Graph Created", box=ROUNDED)
                with console.capture() as capture:
                    console.print(panel)
                result["rich_output"] = capture.get()

        elif action == "stop":
            if "graph_id" not in tool_input:
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": "graph_id is required for stop action"}],
                }

            result = manager.stop_graph(tool_input["graph_id"])
            if result["status"] == "success":
                panel_content = f"🛑 {result['message']}"
                panel = Panel(panel_content, title="Graph Stopped", box=ROUNDED)
                with console.capture() as capture:
                    console.print(panel)
                result["rich_output"] = capture.get()

        elif action == "message":
            if "graph_id" not in tool_input or "message" not in tool_input:
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": "graph_id and message are required for message action"}],
                }

            result = manager.send_message(tool_input["graph_id"], tool_input["message"])
            if result["status"] == "success":
                panel_content = (
                    f"📨 {result['message']}\n\n"
                    f"[bold blue]To:[/bold blue] {tool_input['message']['target']}\n"
                    f"[bold blue]Content:[/bold blue] {tool_input['message']['content'][:100]}..."
                )
                panel = Panel(panel_content, title="Message Sent", box=ROUNDED)
                with console.capture() as capture:
                    console.print(panel)
                result["rich_output"] = capture.get()

        elif action == "status":
            if "graph_id" not in tool_input:
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": "graph_id is required for status action"}],
                }

            result = manager.get_graph_status(tool_input["graph_id"])
            if result["status"] == "success":
                result["rich_output"] = create_rich_status_panel(console, result["data"])

        elif action == "list":
            result = manager.list_graphs()
            if result["status"] == "success":
                headers = ["Graph ID", "Topology", "Nodes"]
                rows = [[graph["graph_id"], graph["topology"], str(graph["node_count"])] for graph in result["data"]]
                result["rich_output"] = create_rich_table(console, "Active Agent Graphs", headers, rows)

        else:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Unknown action: {action}"}],
            }

        # Process result
        if result["status"] == "success":
            # Prepare clean message text without rich formatting
            if "data" in result:
                clean_message = f"Operation {action} completed successfully."
                if action == "create":
                    clean_message = (
                        f"Graph {tool_input['graph_id']} created with {len(tool_input['topology']['nodes'])} nodes."
                    )
                elif action == "stop":
                    clean_message = f"Graph {tool_input['graph_id']} stopped and removed."
                elif action == "message":
                    clean_message = (
                        f"Message sent to {tool_input['message']['target']} in graph {tool_input['graph_id']}."
                    )
                elif action == "status":
                    clean_message = f"Graph {tool_input['graph_id']} status retrieved."
                elif action == "list":
                    graph_count = len(result["data"])
                    clean_message = f"Listed {graph_count} active agent graphs."
            else:
                clean_message = result.get("message", "Operation completed successfully.")

            # Store only clean text in content for agent.messages
            content = [{"text": clean_message}]

            return {"toolUseId": tool_use_id, "status": "success", "content": content}
        else:
            error_message = f"❌ Error: {result['message']}"
            logger.error(error_message)
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": error_message}],
            }

    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{error_trace}"
        logger.error(f"\n[AGENT GRAPH TOOL ERROR]\n{error_msg}")
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"⚠️ Agent Graph Error: {str(e)}"}],
        }
