import matplotlib

matplotlib.use("Agg")  # Add this line before other imports
import importlib
import json
import os
import platform
import subprocess
from typing import Any, Dict, List

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
    except Exception:
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
        except ImportError:
            pass

    for category in AWS_CATEGORIES:
        try:
            module = importlib.import_module(f"diagrams.aws.{category}")
            if hasattr(module, normalized_type):
                return getattr(module, normalized_type)
        except ImportError:
            continue

    raise ValueError(f"AWS component '{node_type}' not found")


def create_cloud_diagram(nodes, edges, title, output_format, style):
    """Create AWS architecture diagram"""
    nodes_dict = {}
    output_path = os.path.join(os.getcwd(), f"{title}")

    with CloudDiagram(name=title, filename=output_path, outformat=output_format):
        for node in nodes:
            node_type = node.get("type", "EC2")
            node_class = get_aws_node(node_type)
            node_label = node.get("label", node["id"])
            nodes_dict[node["id"]] = node_class(node_label)

        for edge in edges:
            from_node = nodes_dict.get(edge["from"])
            to_node = nodes_dict.get(edge["to"])
            if from_node and to_node:
                from_node >> to_node

    output_file = f"{output_path}.{output_format}"

    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", output_file], start_new_session=True)
        elif system == "Windows":
            os.startfile(output_file)
        else:
            subprocess.Popen(["xdg-open", output_file], start_new_session=True)
    except Exception:
        pass

    return output_file


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

    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", rendered_path], start_new_session=True)
        elif system == "Windows":
            os.startfile(rendered_path)
        else:
            subprocess.Popen(["xdg-open", rendered_path], start_new_session=True)
    except Exception:
        pass

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

    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", output_path], start_new_session=True)
        elif system == "Windows":
            os.startfile(output_path)
        else:
            subprocess.Popen(["xdg-open", output_path], start_new_session=True)
    except Exception:
        pass

    return output_path


@tool
def diagram(
    diagram_type: str,
    nodes: List[Dict[str, str]],
    edges: List[Dict[str, str]] = None,
    output_format: str = "png",
    title: str = "diagram",
    style: Dict[str, str] = None,
) -> str:
    """Create diagrams with a unified interface. Supports cloud architecture, network, and graph diagrams.

    Args:
        diagram_type: Type of diagram to create ("cloud", "graph", "network")
        nodes: List of node objects with "id" (required), "label", and "type" (for cloud diagrams)
        edges: List of edge objects with "from", "to", and optional "label"
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
        else:
            raise ValueError(f"Unsupported diagram type: {diagram_type}")

        return f"Created {diagram_type} diagram: {output_path}"

    except Exception as e:
        return f"Error creating diagram: {str(e)}"
