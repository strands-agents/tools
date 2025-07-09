import matplotlib

matplotlib.use("Agg")  # Add this line before other imports
import importlib
import json
import logging
import os
import platform
import subprocess
from typing import Any, Dict, List, Union

import graphviz
import matplotlib.pyplot as plt
import networkx as nx
from diagrams import Diagram as CloudDiagram
from strands import tool

# AWS service categories
AWS_CATEGORIES = [
    "analytics",
    "compute",
    "database",
    "network",
    "storage",
    "security",
    "integration",
    "management",
    "ml",
    "general",
]


def load_aws_component_mapping() -> Dict:
    """Load AWS component mapping from JSON file"""
    mapping_path = os.path.join(os.path.dirname(__file__), "aws_component_mapping.json")
    try:
        with open(mapping_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"AWS component mapping file not found: {mapping_path}")
        return {"common_names": {}, "category_mapping": {}}
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in mapping file: {e}")
        return {"common_names": {}, "category_mapping": {}}


def get_aws_node(node_type: str) -> Any:
    """Dynamically import and return AWS component"""
    mapping = load_aws_component_mapping()
    common_names = mapping["common_names"]
    category_mapping = mapping["category_mapping"]

    normalized_type = common_names.get(node_type.lower(), node_type)

    if normalized_type in category_mapping:
        try:
            category = category_mapping[normalized_type]
            module = importlib.import_module(f"diagrams.aws.{category}")
            if hasattr(module, normalized_type):
                return getattr(module, normalized_type)
        except ImportError as e:
            logging.warning(f"Failed to import {category} module for {normalized_type}: {e}")

    for category in AWS_CATEGORIES:
        try:
            module = importlib.import_module(f"diagrams.aws.{category}")
            if hasattr(module, normalized_type):
                return getattr(module, normalized_type)
        except ImportError as e:
            logging.debug(f"Component {normalized_type} not found in {category}: {e}")
            continue

    raise ValueError(f"AWS component '{node_type}' not found")


def open_diagram(file_path: str) -> None:
    """Helper function to open diagram files across different operating systems"""
    if not os.path.exists(file_path):
        logging.error(f"Cannot open diagram: file does not exist: {file_path}")
        return
        
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", file_path], start_new_session=True)
        elif system == "Windows":
            os.startfile(file_path)
        else:
            subprocess.Popen(["xdg-open", file_path], start_new_session=True)
        logging.info(f"Opened diagram: {file_path}")
    except FileNotFoundError:
        logging.error(f"System command not found for opening files on {system}")
    except subprocess.SubprocessError as e:
        logging.error(f"Failed to open diagram {file_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error opening diagram {file_path}: {e}")


def create_cloud_diagram(nodes, edges, title, output_format, style):
    """Create AWS architecture diagram"""
    if not nodes:
        raise ValueError("At least one node is required for cloud diagram")
        
    nodes_dict = {}
    output_path = os.path.join(os.getcwd(), f"{title}")

    try:
        with CloudDiagram(name=title, filename=output_path, outformat=output_format):
            for node in nodes:
                if "id" not in node:
                    raise ValueError(f"Node missing required 'id' field: {node}")
                    
                node_type = node.get("type", "EC2")
                try:
                    node_class = get_aws_node(node_type)
                    node_label = node.get("label", node["id"])
                    nodes_dict[node["id"]] = node_class(node_label)
                except ValueError as e:
                    logging.error(f"Failed to create node {node['id']}: {e}")
                    raise

            for edge in edges:
                if "from" not in edge or "to" not in edge:
                    logging.warning(f"Edge missing 'from' or 'to' field, skipping: {edge}")
                    continue
                    
                from_node = nodes_dict.get(edge["from"])
                to_node = nodes_dict.get(edge["to"])
                
                if not from_node:
                    logging.warning(f"Source node '{edge['from']}' not found for edge")
                elif not to_node:
                    logging.warning(f"Target node '{edge['to']}' not found for edge")
                else:
                    from_node >> to_node

        output_file = f"{output_path}.{output_format}"
        open_diagram(output_file)
        return output_file
    except Exception as e:
        logging.error(f"Failed to create cloud diagram: {e}")
        raise


def create_graph_diagram(nodes, edges, title, output_format, style):
    """Create Graphviz diagram"""
    dot = graphviz.Digraph(comment=title)
    dot.attr(rankdir=style.get("rankdir", "LR"))

    for node in nodes:
        node_id = node["id"]
        node_label = node.get("label", node_id)
        dot.node(node_id, node_label)

    for edge in edges:
        dot.edge(edge["from"], edge["to"], edge.get("label", ""))

    output_path = os.path.join(os.getcwd(), title)
    rendered_path = dot.render(filename=output_path, format=output_format, cleanup=False)
    open_diagram(rendered_path)
    return rendered_path


def create_network_diagram(nodes, edges, title, output_format, style):
    """Create NetworkX diagram"""
    G = nx.Graph()

    for node in nodes:
        G.add_node(node["id"], label=node.get("label", node["id"]))

    edge_list = [(edge["from"], edge["to"]) for edge in edges]
    G.add_edges_from(edge_list)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1500)
    nx.draw_networkx_edges(G, pos)

    labels = {node["id"]: node.get("label", node["id"]) for node in nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")

    edge_labels = {(edge["from"], edge["to"]): edge.get("label", "") for edge in edges if "label" in edge}
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

    plt.title(title)
    output_path = os.path.join(os.getcwd(), f"{title}.{output_format}")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    open_diagram(output_path)
    return output_path


def create_sequence_diagram(nodes, edges, title, output_format, style):
    """Create sequence diagram showing service interactions"""
    if not nodes:
        raise ValueError("At least one node is required for sequence diagram")
        
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort edges by order
    sorted_edges = sorted(edges, key=lambda x: x.get("order", 0))
    
    # Position participants horizontally
    participant_positions = {node["id"]: i for i, node in enumerate(nodes)}
    participant_labels = {node["id"]: node.get("label", node["id"]) for node in nodes}
    
    # Draw participant lifelines
    for i, node in enumerate(nodes):
        if "id" not in node:
            raise ValueError(f"Node missing required 'id' field: {node}")
        ax.axvline(x=i, color='lightgray', linestyle='--', alpha=0.7)
        ax.text(i, len(sorted_edges) + 0.5, participant_labels[node["id"]], 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Draw interactions
    for i, edge in enumerate(sorted_edges):
        if "from" not in edge or "to" not in edge:
            logging.warning(f"Edge missing 'from' or 'to' field, skipping: {edge}")
            continue
            
        if edge["from"] not in participant_positions:
            logging.warning(f"Source participant '{edge['from']}' not found, skipping edge")
            continue
        if edge["to"] not in participant_positions:
            logging.warning(f"Target participant '{edge['to']}' not found, skipping edge")
            continue
            
        from_pos = participant_positions[edge["from"]]
        to_pos = participant_positions[edge["to"]]
        y_pos = len(sorted_edges) - i - 0.5
        
        # Draw arrow
        ax.annotate('', xy=(to_pos, y_pos), xytext=(from_pos, y_pos),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
        
        # Add message label
        mid_pos = (from_pos + to_pos) / 2
        ax.text(mid_pos, y_pos + 0.1, edge.get("label", ""), 
                ha='center', va='bottom', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlim(-0.5, len(nodes) - 0.5)
    ax.set_ylim(-0.5, len(sorted_edges) + 1)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    output_path = os.path.join(os.getcwd(), f"{title}.{output_format}")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    open_diagram(output_path)
    return output_path


@tool
def diagram(
    diagram_type: str,
    nodes: List[Dict[str, str]],
    edges: List[Dict[str, Union[str, int]]] = None,
    output_format: str = "png",
    title: str = "diagram",
    style: Dict[str, str] = None,
) -> str:
    """Create diagrams with a unified interface. Supports cloud architecture, network, graph, and sequence diagrams.

    Args:
        diagram_type: Type of diagram to create ("cloud", "graph", "network", "sequence")
        nodes: List of node objects with "id" (required), "label", and "type" (for cloud diagrams)
        edges: List of edge objects with "from", "to", optional "label", "order" (int), and "type"
        output_format: Output format ("png" or "svg")
        title: Title of the diagram
        style: Style parameters (e.g., {"rankdir": "LR"} for left-to-right layout)

    Returns:
        Path to the created diagram file
    """
    if edges is None:
        edges = []
    if style is None:
        style = {}

    try:
        if diagram_type == "cloud":
            output_path = create_cloud_diagram(nodes, edges, title, output_format, style)
        elif diagram_type == "graph":
            output_path = create_graph_diagram(nodes, edges, title, output_format, style)
        elif diagram_type == "network":
            output_path = create_network_diagram(nodes, edges, title, output_format, style)
        elif diagram_type == "sequence":
            output_path = create_sequence_diagram(nodes, edges, title, output_format, style)
        else:
            raise ValueError(f"Unsupported diagram type: {diagram_type}")

        return f"Created {diagram_type} diagram: {output_path}"

    except Exception as e:
        return f"Error creating diagram: {str(e)}"
