import importlib
import inspect
import logging
import os
import pkgutil
import platform
import subprocess
from typing import Any, Dict, List, Union

import graphviz
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from diagrams import Diagram as CloudDiagram
from diagrams import aws
from strands import tool

matplotlib.use("Agg")  # Set the backend after importing matplotlib


class AWSComponentRegistry:
    """
    Class responsible for discovering and managing AWS components from the diagrams package.
    Encapsulates the component discovery, caching and lookup functionality.
    """

    def __init__(self):
        """Initialize the registry with discovered components and aliases"""
        self._component_cache = {}
        self.categories = self._discover_categories()
        self.components = self._discover_components()
        self.aliases = self._build_aliases()

    def _discover_categories(self) -> List[str]:
        """Dynamically discover all AWS categories from the diagrams package"""
        categories = []
        try:
            # Use pkgutil to discover all modules in diagrams.aws
            for _, name, is_pkg in pkgutil.iter_modules(aws.__path__):
                if not is_pkg and not name.startswith("_"):
                    categories.append(name)
        except Exception as e:
            logging.warning(f"Failed to discover AWS categories: {e}")
            return []
        return categories

    def _discover_components(self) -> Dict[str, List[str]]:
        """Dynamically discover all available AWS components by category"""
        components = {}
        for category in self.categories:
            try:
                module = importlib.import_module(f"diagrams.aws.{category}")
                # Get all public classes (components) from the module
                components[category] = [
                    name
                    for name, obj in inspect.getmembers(module)
                    if inspect.isclass(obj) and not name.startswith("_")
                ]
            except ImportError:
                continue
        return components

    def _build_aliases(self) -> Dict[str, str]:
        """Build aliases dictionary by analyzing available components"""
        aliases = {}

        # Add non-AWS components first
        aliases.update(
            {
                "users": "Users",
                "user": "Users",
                "client": "Users",
                "clients": "Users",
                "internet": "Internet",
                "web": "Internet",
                "mobile": "Mobile",
            }
        )

        # Analyze component names to create common aliases
        for _, component_list in self.components.items():
            for component in component_list:
                # Create lowercase alias
                aliases[component.lower()] = component

                # Create alias without service prefix/suffix
                clean_name = component.replace("Service", "").replace("Amazon", "").replace("AWS", "")
                if clean_name != component:
                    aliases[clean_name.lower()] = component

                # Add common abbreviations
                if component.isupper():  # Likely an acronym
                    aliases[component.lower()] = component

        return aliases

    def get_node(self, node_type: str) -> Any:
        """Get AWS component class using dynamic discovery with caching"""
        # Check cache first
        if node_type in self._component_cache:
            return self._component_cache[node_type]

        # Normalize input
        normalized = node_type.lower()

        # Try common aliases first
        canonical_name = self.aliases.get(normalized, node_type)

        # Search through all discovered components
        for category, component_list in self.components.items():
            try:
                module = importlib.import_module(f"diagrams.aws.{category}")
                # Try exact match first
                if canonical_name in component_list:
                    component = getattr(module, canonical_name)
                    self._component_cache[node_type] = component
                    return component
                # Try case-insensitive match
                for component_name in component_list:
                    if component_name.lower() == canonical_name.lower():
                        component = getattr(module, component_name)
                        self._component_cache[node_type] = component
                        return component
            except ImportError:
                continue

        raise ValueError(f"Component '{node_type}' not found in available AWS components")

    def list_available_components(self, category: str = None) -> Dict[str, List[str]]:
        """List all available AWS components and their aliases"""
        if category:
            return {category: self.components.get(category, [])}
        return self.components


# Initialize the AWS component registry as a singleton
aws_registry = AWSComponentRegistry()


# Expose necessary functions and variables at module level for backward compatibility
def get_aws_node(node_type: str) -> Any:
    """Get AWS component class using dynamic discovery"""
    return aws_registry.get_node(node_type)


def list_available_components(category: str = None) -> Dict[str, List[str]]:
    """List all available AWS components and their aliases"""
    return aws_registry.list_available_components(category)


# Export variables for backward compatibility
AWS_CATEGORIES = aws_registry.categories
AVAILABLE_AWS_COMPONENTS = aws_registry.components
COMMON_ALIASES = aws_registry.aliases


class DiagramBuilder:
    """Unified diagram builder that handles all diagram types and formats"""

    def __init__(self, nodes, edges=None, title="diagram", style=None, open_diagram_flag=True):
        self.nodes = nodes
        self.edges = edges or []
        self.title = title
        self.style = style or {}
        self.open_diagram_flag = open_diagram_flag

    def render(self, diagram_type: str, output_format: str) -> str:
        """Main render method that delegates to specific renderers"""

        method_map = {
            "cloud": self._render_cloud,
            "graph": self._render_graph,
            "network": self._render_network,
        }

        if diagram_type not in method_map:
            raise ValueError(f"Unsupported diagram type: {diagram_type}")

        return method_map[diagram_type](output_format)

    def _render_cloud(self, output_format: str) -> str:
        """Create AWS architecture diagram"""
        if not self.nodes:
            raise ValueError("At least one node is required for cloud diagram")

        # Pre-validate all node types before creating diagram
        invalid_nodes = []
        for node in self.nodes:
            if "id" not in node:
                raise ValueError(f"Node missing required 'id' field: {node}")

            node_type = node.get("type", "EC2")
            try:
                get_aws_node(node_type)
            except ValueError:
                invalid_nodes.append((node["id"], node_type))

        if invalid_nodes:
            suggestions = []
            for node_id, node_type in invalid_nodes:
                # Find close matches
                close_matches = [k for k in COMMON_ALIASES.keys() if node_type.lower() in k or k in node_type.lower()]
                # Find canonical names for suggestions
                canonical_suggestions = [COMMON_ALIASES[k] for k in close_matches[:3]] if close_matches else []

                if close_matches:
                    suggestions.append(
                        f"  - '{node_id}' (type: '{node_type}') -> try: \
                        {close_matches[:3]} (maps to: {canonical_suggestions})"
                    )
                else:
                    suggestions.append(f"  - '{node_id}' (type: '{node_type}') -> no close matches found")

            common_types = [
                "ec2",
                "s3",
                "lambda",
                "rds",
                "api_gateway",
                "cloudfront",
                "route53",
                "elb",
                "opensearch",
                "dynamodb",
            ]
            error_msg = (
                f"Invalid AWS component types found:\n{chr(10).join(suggestions)}\n\n"
                f"Common types: {common_types}\nNote: "
                f"All 532+ AWS components are supported - \
                    try using one of the aliases in COMMON_ALIASES or the exact AWS service name"
            )
            raise ValueError(error_msg)

        nodes_dict = {}
        output_path = save_diagram_to_directory(self.title, "")

        try:
            with CloudDiagram(name=self.title, filename=output_path, outformat=output_format):
                for node in self.nodes:
                    node_type = node.get("type", "EC2")
                    node_class = get_aws_node(node_type)
                    node_label = node.get("label", node["id"])
                    nodes_dict[node["id"]] = node_class(node_label)

                for edge in self.edges:
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
            if self.open_diagram_flag:
                open_diagram(output_file)
            return output_file
        except Exception as e:
            logging.error(f"Failed to create cloud diagram: {e}")
            raise

    def _render_graph(self, output_format: str) -> str:
        """Create Graphviz diagram with optional AWS icons"""
        dot = graphviz.Digraph(comment=self.title)
        dot.attr(rankdir=self.style.get("rankdir", "LR"))

        for node in self.nodes:
            node_id = node["id"]
            node_label = node.get("label", node_id)

            # Add AWS service type as tooltip if specified
            if "type" in node:
                try:
                    get_aws_node(node["type"])  # Validate AWS component exists
                    dot.node(node_id, node_label, tooltip=f"AWS {node['type']}")
                except ValueError:
                    dot.node(node_id, node_label)
            else:
                dot.node(node_id, node_label)

        for edge in self.edges:
            dot.edge(edge["from"], edge["to"], edge.get("label", ""))

        output_path = save_diagram_to_directory(self.title, "")
        rendered_path = dot.render(filename=output_path, format=output_format, cleanup=False)
        if self.open_diagram_flag:
            open_diagram(rendered_path)
        return rendered_path

    def _render_network(self, output_format: str) -> str:
        """Create NetworkX diagram with AWS-aware coloring"""
        G = nx.Graph()
        node_colors = []
        aws_color_map = {
            "compute": "orange",
            "database": "green",
            "network": "blue",
            "storage": "purple",
            "security": "red",
        }

        for node in self.nodes:
            G.add_node(node["id"], label=node.get("label", node["id"]))

            # Color nodes based on AWS service category
            if "type" in node:
                try:
                    get_aws_node(node["type"])  # Validate AWS component exists
                    # Simple category detection based on common patterns
                    node_type = node["type"].lower()
                    if any(x in node_type for x in ["ec2", "lambda", "fargate", "ecs", "eks"]):
                        node_colors.append(aws_color_map["compute"])
                    elif any(x in node_type for x in ["rds", "dynamo", "aurora", "redshift"]):
                        node_colors.append(aws_color_map["database"])
                    elif any(x in node_type for x in ["vpc", "elb", "api_gateway", "cloudfront"]):
                        node_colors.append(aws_color_map["network"])
                    elif any(x in node_type for x in ["s3", "efs", "fsx"]):
                        node_colors.append(aws_color_map["storage"])
                    elif any(x in node_type for x in ["iam", "kms", "cognito", "waf"]):
                        node_colors.append(aws_color_map["security"])
                    else:
                        node_colors.append("lightblue")
                except ValueError:
                    node_colors.append("lightblue")
            else:
                node_colors.append("lightblue")

        edge_list = [(edge["from"], edge["to"]) for edge in self.edges]
        G.add_edges_from(edge_list)

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500)
        nx.draw_networkx_edges(G, pos)

        labels = {node["id"]: node.get("label", node["id"]) for node in self.nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")

        edge_labels = {(edge["from"], edge["to"]): edge.get("label", "") for edge in self.edges if "label" in edge}
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels)

        plt.title(self.title)
        output_path = save_diagram_to_directory(self.title, output_format)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        if self.open_diagram_flag:
            open_diagram(output_path)
        return output_path


class UMLDiagramBuilder:
    """Builder for all 14 types of UML diagrams with proper notation"""

    def __init__(
        self,
        diagram_type: str,
        elements: List[Dict],
        relationships: List[Dict] = None,
        title: str = "UML_Diagram",
        style: Dict = None,
        open_diagram_flag: bool = True,
    ):
        self.diagram_type = diagram_type.lower().replace(" ", "_").replace("-", "_")
        self.elements = elements
        self.relationships = relationships or []
        self.title = title
        self.style = style or {}
        self.open_diagram_flag = open_diagram_flag

    def render(self, output_format: str = "png") -> str:
        """Render the UML diagram based on type"""

        method_map = {
            # Structural diagrams
            "class": self._render_class,
            "object": self._render_object,
            "component": self._render_component,
            "deployment": self._render_deployment,
            "package": self._render_package,
            "profile": self._render_profile,
            "composite_structure": self._render_composite_structure,
            # Behavioral diagrams
            "use_case": self._render_use_case,
            "activity": self._render_activity,
            "state_machine": self._render_state_machine,
            "sequence": self._render_sequence,
            "communication": self._render_communication,
            "interaction_overview": self._render_interaction_overview,
            "timing": self._render_timing,
        }

        if self.diagram_type not in method_map:
            raise ValueError(f"Unsupported UML diagram type: {self.diagram_type}")

        return method_map[self.diagram_type](output_format)

    def _create_dot_graph(self):
        dot = graphviz.Digraph()
        dot.attr(rankdir="TB")
        dot.attr("node", shape="record", fontname="Arial")
        dot.attr("edge", fontname="Arial", fontsize="10")
        dot.attr("graph", ranksep="0.5")
        return dot

    # STRUCTURAL DIAGRAMS

    def _render_class(self, output_format: str) -> str:
        dot = self._create_dot_graph()

        for element in self.elements:
            class_name = element["name"]
            # Handle both list and string formats for attributes and methods
            attributes = element.get("attributes", [])
            if isinstance(attributes, str):
                attributes = [attr.strip() for attr in attributes.split("\n") if attr.strip()]

            methods = element.get("methods", [])
            if isinstance(methods, str):
                methods = [method.strip() for method in methods.split("\n") if method.strip()]

            label_parts = [class_name]

            if attributes:
                attr_text = "\\n".join([self._format_visibility(attr) for attr in attributes])
                label_parts.append(attr_text)

            if methods:
                method_text = "\\n".join([self._format_visibility(method) for method in methods])
                label_parts.append(method_text)

            label = "{{{}}}".format("|".join(label_parts))

            dot.node(class_name, label, shape="record")

        for rel in self.relationships:
            self._add_class_relationship(dot, rel)

        return self._save_diagram(dot, output_format)

    def _render_component(self, output_format: str) -> str:
        """Component Diagram: Software components and interfaces with proper UML notation"""
        dot = self._create_dot_graph()
        dot.attr("node", shape="none")

        with dot.subgraph(name="cluster_0") as c:
            c.attr(label=self.title, style="rounded", bgcolor="white")

            # First pass: Create components and interfaces
            for element in self.elements:
                name = element["name"]
                elem_type = element.get("type", "component")

                if elem_type == "component":
                    # Improved component notation with stereotype and ports
                    ports = element.get("ports", [])
                    port_cells = ""

                    # Add port definitions if present
                    for port in ports:
                        port_id = port.get("id", "")
                        port_cells += f'<TR><TD PORT="{port_id}" BORDER="0">●</TD></TR>'

                    label = f"""<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                        <TR>
                            <TD>
                                <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">
                                    <TR>
                                        <TD>
                                            <TABLE BORDER="0" CELLSPACING="0">
                                                <TR>
                                                    <TD ALIGN="RIGHT" WIDTH="20">
                                                        <TABLE BORDER="1" CELLSPACING="0" CELLPADDING="0">
                                                            <TR><TD WIDTH="15" HEIGHT="7"></TD></TR>
                                                            <TR><TD WIDTH="15" HEIGHT="7"></TD></TR>
                                                        </TABLE>
                                                    </TD>
                                                    <TD ALIGN="LEFT" CELLPADDING="4">
                                                        <TABLE BORDER="0" CELLSPACING="0">
                                                            <TR><TD ALIGN="CENTER">«component»</TD></TR>
                                                            <TR><TD ALIGN="CENTER">{name}</TD></TR>
                                                        </TABLE>
                                                    </TD>
                                                </TR>
                                            </TABLE>
                                        </TD>
                                    </TR>
                                    {port_cells}
                                </TABLE>
                            </TD>
                        </TR>
                    </TABLE>>"""
                    dot.node(name, label, margin="0", style="rounded")

                elif elem_type == "interface":
                    interface_name = element.get("name", "")
                    stereotype = "«interface»"

                    if element.get("provided", False):
                        # Ball notation (provided interface)
                        dot.node(
                            f"{name}_provided",
                            f"""<<TABLE BORDER="0" CELLSPACING="0">
                                    <TR><TD>◯</TD></TR>
                                    <TR><TD>{stereotype}</TD></TR>
                                    <TR><TD>{interface_name}</TD></TR>
                                </TABLE>>""",
                            shape="none",
                        )

                    if element.get("required", False):
                        # Socket notation (required interface)
                        dot.node(
                            f"{name}_required",
                            f"""<<TABLE BORDER="0" CELLSPACING="0">
                                    <TR><TD>◐</TD></TR>
                                    <TR><TD>{stereotype}</TD></TR>
                                    <TR><TD>{interface_name}</TD></TR>
                                </TABLE>>""",
                            shape="none",
                        )

                elif elem_type == "port":
                    # Standard UML port notation
                    dot.node(name, "□", shape="none", fontsize="14")

        # Second pass: Create relationships with proper UML notation
        for rel in self.relationships:
            rel_type = rel.get("type", "connection")

            edge_attrs = {
                "dependency": {"style": "dashed", "arrowhead": "vee", "dir": "forward"},
                "realization": {"style": "dashed", "arrowhead": "empty", "dir": "forward"},
                "assembly": {"style": "solid", "arrowhead": "none", "dir": "both"},
                "delegation": {"style": "dashed", "arrowhead": "vee", "dir": "forward"},
                "connection": {"style": "solid", "arrowhead": "none"},
            }

            attrs = edge_attrs.get(rel_type, edge_attrs["connection"])

            # Add proper UML notation for multiplicity and constraints
            if "multiplicity" in rel:
                attrs["label"] = rel["multiplicity"]
                attrs["fontsize"] = "10"

            if "constraint" in rel:
                constraint = rel["constraint"]
                attrs["label"] = f"{{{constraint}}}"

            # Add standard label if present
            if "label" in rel:
                current_label = attrs.get("label", "")
                attrs["label"] = f"{current_label}\n{rel['label']}" if current_label else rel["label"]

            # Create the edge with proper attributes
            dot.edge(rel["from"], rel["to"], **attrs)

        # Set diagram-wide attributes for better UML compliance
        dot.attr(rankdir="LR")  # Standard left-to-right layout for component diagrams
        dot.attr(splines="ortho")  # Orthogonal lines for clearer relationships
        dot.attr(nodesep="1.0")  # Increased spacing between nodes
        dot.attr(ranksep="1.0")  # Increased spacing between ranks

        return self._save_diagram(dot, output_format)

    def _render_deployment(self, output_format: str) -> str:
        """Deployment Diagram: Hardware nodes and software artifacts"""
        dot = self._create_dot_graph()

        for element in self.elements:
            name = element["name"]
            elem_type = element.get("type", "node")

            if elem_type == "node":
                dot.node(name, f"<<device>>\\n{name}", shape="box3d", style="filled", fillcolor="lightyellow")
            elif elem_type == "artifact":
                dot.node(name, f"<<artifact>>\\n{name}", shape="note", style="filled", fillcolor="lightcyan")

        for rel in self.relationships:
            dot.edge(rel["from"], rel["to"], label=rel.get("label", ""))

        return self._save_diagram(dot, output_format)

    def _render_use_case(self, output_format: str) -> str:
        """Use Case Diagram: Actors, use cases, and system boundary"""
        dot = self._create_dot_graph()

        for element in self.elements:
            name = element["name"]
            elem_type = element.get("type", "use_case")

            if elem_type == "actor":
                dot.node(name, f"&lt;&lt;actor&gt;&gt;\\n{name}", shape="plaintext")
            elif elem_type == "use_case":
                dot.node(name, name, shape="ellipse", style="filled", fillcolor="lightblue")
            elif elem_type == "system":
                dot.node(name, name, shape="box", style="dashed")

        for rel in self.relationships:
            rel_type = rel.get("type", "association")
            if rel_type == "include":
                dot.edge(rel["from"], rel["to"], style="dashed", label="&lt;&lt;include&gt;&gt;")
            elif rel_type == "extend":
                dot.edge(rel["from"], rel["to"], style="dashed", label="&lt;&lt;extend&gt;&gt;")
            else:
                dot.edge(rel["from"], rel["to"])

        return self._save_diagram(dot, output_format)

    def _render_sequence(self, output_format: str) -> str:
        """Sequence Diagram: Objects and message exchanges over time"""
        if not self.elements:
            raise ValueError("At least one element is required for sequence diagram")

        fig, ax = plt.subplots(figsize=(12, 8))
        sorted_msgs = sorted(self.relationships, key=lambda x: x.get("sequence", 0))
        participant_positions = {element["name"]: i for i, element in enumerate(self.elements)}
        participant_labels = {element["name"]: element.get("label", element["name"]) for element in self.elements}

        # Draw participant boxes and lifelines
        for i, element in enumerate(self.elements):
            if "name" not in element:
                raise ValueError(f"Element missing required 'name' field: {element}")

            ax.add_patch(
                plt.Rectangle((i - 0.3, len(sorted_msgs) + 0.2), 0.6, 0.6, facecolor="lightblue", edgecolor="black")
            )
            ax.text(
                i,
                len(sorted_msgs) + 0.5,
                participant_labels[element["name"]],
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=10,
            )
            ax.axvline(
                x=i,
                ymin=0,
                ymax=(len(sorted_msgs) + 0.2) / (len(sorted_msgs) + 1),
                color="gray",
                linestyle="--",
                alpha=0.7,
            )

        # Draw message interactions
        for i, msg in enumerate(sorted_msgs):
            if "from" not in msg or "to" not in msg:
                logging.warning(f"Message missing 'from' or 'to' field, skipping: {msg}")
                continue

            if msg["from"] not in participant_positions or msg["to"] not in participant_positions:
                logging.warning("Participant not found, skipping message")
                continue

            from_pos = participant_positions[msg["from"]]
            to_pos = participant_positions[msg["to"]]
            y_pos = len(sorted_msgs) - i - 0.5

            ax.annotate(
                "", xy=(to_pos, y_pos), xytext=(from_pos, y_pos), arrowprops=dict(arrowstyle="->", color="blue", lw=1.5)
            )

            mid_pos = (from_pos + to_pos) / 2 if from_pos != to_pos else from_pos + 0.3
            sequence_num = msg.get("sequence", i + 1)
            # Use 'label' field first, then fall back to 'message' for backward compatibility
            message_text = msg.get("label", msg.get("message", ""))
            label = f"{sequence_num}. {message_text}" if message_text else str(sequence_num)

            ax.text(
                mid_pos,
                y_pos + 0.1,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        ax.set_xlim(-0.5, len(self.elements) - 0.5)
        ax.set_ylim(-0.5, len(sorted_msgs) + 1)
        ax.set_title(self.title, fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        output_path = save_diagram_to_directory(self.title, output_format)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        if self.open_diagram_flag:
            open_diagram(output_path)
        return output_path

    # Simplified implementations for other UML types
    def _render_object(self, output_format: str) -> str:
        """Object Diagram: Instance objects with their attribute values"""
        dot = self._create_dot_graph()

        for element in self.elements:
            name = element["name"]
            class_name = element.get("class", "Object")
            attributes = element.get("attributes", "")

            # Build object label with object name and attributes
            label_parts = [f"{name}:{class_name}"]

            if attributes:
                if isinstance(attributes, str):
                    attr_lines = [attr.strip() for attr in attributes.split("\n") if attr.strip()]
                else:
                    attr_lines = [f"{key} = {value}" for key, value in attributes.items()]
                label_parts.extend(attr_lines)

            label = "{{{}}}".format("|".join(label_parts))
            dot.node(name, label, shape="record", style="filled", fillcolor="lightblue")

        for rel in self.relationships:
            dot.edge(rel["from"], rel["to"], label=rel.get("label", ""))
        return self._save_diagram(dot, output_format)

    def _render_deployment(self, output_format: str) -> str:
        dot = self._create_dot_graph()
        for element in self.elements:
            name = element["name"]
            elem_type = element.get("type", "node")
            if elem_type == "node":
                dot.node(name, f"<<device>>\\n{name}", shape="box3d", style="filled", fillcolor="lightyellow")
            elif elem_type == "artifact":
                dot.node(name, f"<<artifact>>\\n{name}", shape="note", style="filled", fillcolor="lightcyan")
        for rel in self.relationships:
            dot.edge(rel["from"], rel["to"], label=rel.get("label", ""))
        return self._save_diagram(dot, output_format)

    def _render_package(self, output_format: str) -> str:
        dot = self._create_dot_graph()
        for element in self.elements:
            dot.node(element["name"], element["name"], shape="folder", style="filled", fillcolor="lightgray")
        for rel in self.relationships:
            style = "dashed" if rel.get("type") == "dependency" else "solid"
            dot.edge(rel["from"], rel["to"], style=style, arrowhead="open")
        return self._save_diagram(dot, output_format)

    def _render_profile(self, output_format: str) -> str:
        dot = self._create_dot_graph()
        for element in self.elements:
            name = element["name"]
            stereotype = element.get("stereotype", "")
            label = f"<<{stereotype}>>\\n{name}" if stereotype else name
            dot.node(name, label, shape="box", style="dashed")
        for rel in self.relationships:
            dot.edge(rel["from"], rel["to"], style="dashed", arrowhead="empty")
        return self._save_diagram(dot, output_format)

    def _render_composite_structure(self, output_format: str) -> str:
        """Composite Structure Diagram: Internal structure with parts and ports"""
        dot = self._create_dot_graph()
        dot.attr("node", shape="none")

        # Create main component boundary
        with dot.subgraph(name="cluster_main") as c:
            c.attr(label=self.title, style="rounded", bgcolor="lightgray")

            # Track ports by their owners for proper placement
            port_by_owner = {}

            # First pass: Create parts
            for element in self.elements:
                name = element["name"]
                elem_type = element.get("type", "part")

                if elem_type == "part":
                    # Add multiplicity if specified
                    multiplicity = element.get("multiplicity", "")
                    multiplicity_str = f"[{multiplicity}]" if multiplicity else ""

                    # Add proper UML part notation with type and name
                    label = f"""<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">
                            <TR><TD>{name}{multiplicity_str}</TD></TR>
                            </TABLE>>"""
                    c.node(name, label, margin="0")
                elif elem_type == "port":
                    owner = element.get("owner")
                    if owner not in port_by_owner:
                        port_by_owner[owner] = []
                    port_by_owner[owner].append(element)

            # Second pass: Add ports with proper placement
            for _, ports in port_by_owner.items():
                for port in ports:
                    port_name = port["name"]
                    port_type = port.get("interface_type", "")
                    is_provided = port.get("is_provided", True)

                    # Position port on boundary
                    if is_provided:
                        # Provided interface (lollipop notation)
                        port_label = f"""<<TABLE BORDER="0" CELLBORDER="0">
                                    <TR><TD>○</TD></TR>
                                    <TR><TD>{port_type}</TD></TR>
                                    </TABLE>>"""
                    else:
                        # Required interface (socket notation)
                        port_label = f"""<<TABLE BORDER="0" CELLBORDER="0">
                                    <TR><TD>◑</TD></TR>
                                    <TR><TD>{port_type}</TD></TR>
                                    </TABLE>>"""

                    dot.node(port_name, port_label)

            # Add relationships with proper UML notation
            for rel in self.relationships:
                rel_type = rel.get("type", "connector")
                label = rel.get("label", "")
                multiplicity_source = rel.get("multiplicity_source", "")
                multiplicity_target = rel.get("multiplicity_target", "")

                if multiplicity_source or multiplicity_target:
                    label = f"{multiplicity_source} {label} {multiplicity_target}"

                if rel_type == "assembly":
                    # Assembly connector (between parts)
                    dot.edge(rel["from"], rel["to"], arrowhead="none", style="solid", label=label)
                elif rel_type == "delegation":
                    # Delegation connector (typically to/from ports)
                    dot.edge(rel["from"], rel["to"], style="dashed", arrowhead="none", label=label)
                elif rel_type == "composition":
                    # Composition relationship
                    dot.edge(rel["from"], rel["to"], arrowhead="diamond", arrowsize="1.5", label=label)
                else:
                    # Default connector
                    dot.edge(rel["from"], rel["to"], arrowhead="none", label=label)

        return self._save_diagram(dot, output_format)

    def _render_activity(self, output_format: str) -> str:
        dot = self._create_dot_graph()
        for element in self.elements:
            name = element["name"]
            elem_type = element.get("type", "activity")
            if elem_type == "start":
                dot.node(name, "", shape="circle", style="filled", fillcolor="black", width="0.3")
            elif elem_type == "end":
                dot.node(name, "", shape="doublecircle", style="filled", fillcolor="black", width="0.3")
            elif elem_type == "activity":
                dot.node(name, name, shape="box", style="rounded,filled", fillcolor="lightblue")
            elif elem_type == "decision":
                dot.node(name, name, shape="diamond", style="filled", fillcolor="yellow")
        for rel in self.relationships:
            dot.edge(rel["from"], rel["to"], label=rel.get("label", ""))
        return self._save_diagram(dot, output_format)

    def _render_state_machine(self, output_format: str) -> str:
        dot = self._create_dot_graph()
        for element in self.elements:
            name = element["name"]
            elem_type = element.get("type", "state")
            if elem_type == "initial":
                dot.node(name, "", shape="circle", style="filled", fillcolor="black", width="0.3")
            elif elem_type == "final":
                dot.node(name, "", shape="doublecircle", style="filled", fillcolor="black", width="0.3")
            elif elem_type == "state":
                dot.node(name, name, shape="box", style="rounded,filled", fillcolor="lightgreen")
        for rel in self.relationships:
            label = rel.get("event", "")
            if rel.get("action"):
                label += f" / {rel['action']}"
            dot.edge(rel["from"], rel["to"], label=label)
        return self._save_diagram(dot, output_format)

    def _render_communication(self, output_format: str) -> str:
        dot = self._create_dot_graph()
        dot.attr(rankdir="LR")
        for element in self.elements:
            dot.node(element["name"], element["name"], shape="box", style="filled", fillcolor="lightblue")
        for rel in self.relationships:
            seq = rel.get("sequence", "")
            msg = rel.get("message", "")
            label = f"{seq}: {msg}" if seq else msg
            dot.edge(rel["from"], rel["to"], label=label, dir="both")
        return self._save_diagram(dot, output_format)

    def _render_interaction_overview(self, output_format: str) -> str:
        """Interaction Overview Diagram: High-level interaction flow"""
        dot = self._create_dot_graph()

        # Add UML frame around the diagram
        dot.attr("graph", compound="true")

        for element in self.elements:
            name = element["name"]
            elem_type = element.get("type", "interaction")

            if elem_type == "initial":
                # Initial node - solid black circle
                dot.node(name, "", shape="circle", style="filled", fillcolor="black", width="0.3")
            elif elem_type == "final":
                # Final node - circle with dot inside
                dot.node(
                    name, "", shape="doublecircle", style="filled,bold", fillcolor="white", color="black", width="0.3"
                )
            elif elem_type == "interaction":
                # Interaction frame with proper UML notation
                label = f"""<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                        <TR><TD BGCOLOR="lightgrey">sd {name}</TD></TR>
                        </TABLE>>"""
                dot.node(name, label, shape="none", margin="0")
            elif elem_type == "decision":
                # Decision node with proper diamond shape
                dot.node(
                    name,
                    name,
                    shape="diamond",
                    style="filled",
                    fillcolor="white",
                    color="black",
                    width="1.5",
                    height="1.5",
                )
            elif elem_type == "fork":
                # Fork/join node
                dot.node(name, "", shape="rect", style="filled", fillcolor="black", width="0.1", height="0.02")

        for rel in self.relationships:
            # Add proper UML guard conditions and labels
            label = rel.get("label", "")
            guard = rel.get("guard", "")
            if guard:
                label = f"[{guard}]" if not label else f"[{guard}] {label}"

            # Add proper arrow styling based on relationship type
            attrs = {"label": label, "fontsize": "10", "arrowhead": "vee", "arrowsize": "0.8"}

            # Special handling for different relationship types
            rel_type = rel.get("type", "sequence")
            if rel_type == "concurrent":
                attrs["style"] = "bold"
            elif rel_type == "alternative":
                attrs["style"] = "dashed"

            dot.edge(rel["from"], rel["to"], **attrs)

        # Set diagram-wide attributes for better UML compliance
        dot.attr(rankdir="TB")  # Top to bottom layout is standard for IODs
        dot.attr(splines="ortho")  # Orthogonal lines for clearer flow
        dot.attr(nodesep="0.5")
        dot.attr(ranksep="0.7")

        return self._save_diagram(dot, output_format)

    def _render_timing(self, output_format: str) -> str:
        if not self.elements:
            raise ValueError("At least one element is required for timing diagram")
        fig, ax = plt.subplots(figsize=(12, len(self.elements) * 1.5))
        y_ticks, y_labels = [], []
        for idx, element in enumerate(self.elements):
            name = element["name"]
            states = element.get("states", [])

            # Handle string format: "state1:0-10,state2:10-20"
            if isinstance(states, str):
                parsed_states = []
                for state_str in states.split(","):
                    if ":" in state_str and "-" in state_str:
                        state_name, time_range = state_str.strip().split(":")
                        start_str, end_str = time_range.split("-")
                        parsed_states.append(
                            {"state": state_name.strip(), "start": int(start_str.strip()), "end": int(end_str.strip())}
                        )
                states = parsed_states

            color_cycle = plt.cm.tab20.colors
            for state_idx, state in enumerate(states):
                start, end = state.get("start"), state.get("end")
                label = state.get("state", "")
                if start is None or end is None:
                    continue
                ax.broken_barh(
                    [(start, end - start)],
                    (idx - 0.4, 0.8),
                    facecolors=color_cycle[state_idx % len(color_cycle)],
                    edgecolor="black",
                )
                ax.text(start + (end - start) / 2, idx, label, ha="center", va="center", fontsize=9)
            y_ticks.append(idx)
            y_labels.append(name)
        ax.set_xlabel("Time")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_title(self.title, fontsize=14, fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        output_path = save_diagram_to_directory(self.title, output_format)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        if self.open_diagram_flag:
            open_diagram(output_path)
        return output_path

    # Helper methods for UML diagrams

    def _format_visibility(self, member: Union[str, Dict]) -> str:
        if isinstance(member, str):
            return member
        visibility = member.get("visibility", "public")
        name = member.get("name", "")
        member_type = member.get("type", "")
        marker = {"public": "+", "private": "-", "protected": "#", "package": "~"}.get(visibility, "+")
        if member_type:
            return f"{marker} {name}: {member_type}"
        return f"{marker} {name}"

    def _add_class_relationship(self, dot: graphviz.Digraph, rel: Dict):
        """Add class diagram relationships with proper notation"""
        rel_type = rel.get("type", "association")
        if rel_type == "inheritance":
            dot.edge(rel["from"], rel["to"], arrowhead="empty")
        elif rel_type == "composition":
            dot.edge(rel["from"], rel["to"], arrowhead="diamond", style="filled")
        elif rel_type == "aggregation":
            dot.edge(rel["from"], rel["to"], arrowhead="diamond")
        elif rel_type == "dependency":
            dot.edge(rel["from"], rel["to"], style="dashed", arrowhead="open")
        else:  # association
            multiplicity = rel.get("multiplicity", "")
            dot.edge(rel["from"], rel["to"], label=multiplicity)

    def _save_diagram(self, dot: graphviz.Digraph, output_format: str) -> str:
        """Save diagram and return file path"""
        output_path = save_diagram_to_directory(self.title, "")
        rendered_path = dot.render(filename=output_path, format=output_format, cleanup=False)
        if self.open_diagram_flag:
            open_diagram(rendered_path)
        return rendered_path


def save_diagram_to_directory(title: str, extension: str, content: str = None) -> str:
    """Helper function to save diagrams to the diagrams directory

    Args:
        title: Base filename for the diagram
        extension: File extension (with or without dot)
        content: Text content to write (for text-based formats)

    Returns:
        Full path to the saved file
    """
    diagrams_dir = os.path.join(os.getcwd(), "diagrams")
    os.makedirs(diagrams_dir, exist_ok=True)

    # Ensure extension starts with dot
    if not extension.startswith("."):
        extension = "." + extension

    output_path = os.path.join(diagrams_dir, f"{title}{extension}")

    # Write content if provided (for text-based formats)
    if content is not None:
        with open(output_path, "w") as f:
            f.write(content)

    return output_path


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


@tool
def diagram(
    diagram_type: str,
    nodes: List[Dict[str, str]] = None,
    edges: List[Dict[str, Union[str, int]]] = None,
    output_format: str = "png",
    title: str = "diagram",
    style: Dict[str, str] = None,
    elements: List[Dict[str, str]] = None,
    relationships: List[Dict[str, Union[str, int]]] = None,
    open_diagram_flag: bool = True,
) -> str:
    """Create diagrams including AWS cloud diagrams and all 14 UML diagram types.

    Args:
        diagram_type: Type of diagram - Basic: "cloud", "graph", "network" | UML: "class", "object", "component",
                     "deployment", "package", "profile", "composite_structure", "use_case", "activity",
                     "state_machine", "sequence", "communication", "interaction_overview", "timing"
        nodes: For basic diagrams - List of node objects with "id" (required), "label", and "type" (AWS service name)
        edges: For basic diagrams - List of edge objects with "from", "to", optional "label", "order" (int)
        elements: For UML diagrams - List of UML elements with "name" (required), "type", and type-specific properties
        relationships: For UML diagrams - List of UML relationships between elements
        output_format: Output format ("png", "svg", "pdf")
            - For mermaid diagrams, use the agent's LLM capabilities to generate mermaid code directly
            - Example: "Generate mermaid code for a class diagram with User and Order classes"
        title: Title of the diagram
        style: Style parameters (e.g., {"rankdir": "LR"} for left-to-right layout)
        open_diagram_flag: Whether to open the diagram after creation

    Note:
        For STATE MACHINE diagrams: Include an initial state (start point) and final state(s) (end points)
        in your elements to create proper UML state machine notation.

        For ACTIVITY diagrams: Include a start node (initial) and end node (final) in your elements
        to show the complete workflow process from beginning to completion.

        For COMPOSITE STRUCTURE diagrams: Add "multiplicity" field to elements and
        "multiplicity_source"/"multiplicity_target" to relationships for proper UML notation
        (e.g., "1", "*", "0..1").

        For OBJECT diagrams: Use "class" field for object type and "attributes" string for attribute values
        (e.g., {"name": "john", "class": "Customer", "attributes": "name = John Doe\nID = 12345"}).

        For TIMING diagrams: Use "states" string with format "state1:start-end,state2:start-end"
        (e.g., {"name": "microwave", "states": "Idle:0-10,Opening:10-15,Heating:15-30"}).

    Returns:
        Path to the created diagram file
    """
    try:
        # UML diagram types
        uml_types = [
            "class",
            "object",
            "component",
            "deployment",
            "package",
            "profile",
            "composite_structure",
            "use_case",
            "activity",
            "state_machine",
            "sequence",
            "communication",
            "interaction_overview",
            "timing",
        ]

        if diagram_type in uml_types:
            if not elements:
                return "Error: 'elements' parameter is required for UML diagrams"
            builder = UMLDiagramBuilder(diagram_type, elements, relationships, title, style, open_diagram_flag)
            output_path = builder.render(output_format)
            return f"Created {diagram_type} UML diagram: {output_path}"
        else:
            if not nodes:
                return "Error: 'nodes' parameter is required for basic diagrams"
            builder = DiagramBuilder(nodes, edges, title, style, open_diagram_flag)
            output_path = builder.render(diagram_type, output_format)
            return f"Created {diagram_type} diagram: {output_path}"
    except Exception as e:
        return f"Error creating diagram: {str(e)}"
