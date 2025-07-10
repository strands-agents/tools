"""
Tests for the diagram tool.
"""

import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Mock all external dependencies before importing
mock_modules = {
    "matplotlib": MagicMock(),
    "matplotlib.pyplot": MagicMock(),
    "diagrams": MagicMock(),
    "graphviz": MagicMock(),
    "networkx": MagicMock(),
    "strands": MagicMock(),
}

with patch.dict("sys.modules", mock_modules):
    from strands_tools.diagram import (
        DiagramBuilder,
        get_aws_node,
        load_aws_component_mapping,
        render_as_ascii,
        render_as_mermaid,
    )


@pytest.fixture
def sample_nodes():
    """Sample nodes for testing."""
    return [
        {"id": "web", "label": "Web Server", "type": "EC2"},
        {"id": "db", "label": "Database", "type": "RDS"},
        {"id": "cache", "label": "Cache", "type": "Elasticache"},
    ]


@pytest.fixture
def sample_edges():
    """Sample edges for testing."""
    return [
        {"from": "web", "to": "db", "label": "queries"},
        {"from": "web", "to": "cache", "label": "caches"},
    ]


@pytest.fixture
def sample_style():
    """Sample style configuration."""
    return {"rankdir": "LR"}


@pytest.fixture
def mock_aws_mapping():
    """Mock AWS component mapping."""
    return {
        "common_names": {
            "ec2": "EC2",
            "rds": "RDS",
            "elasticache": "Elasticache",
        },
        "category_mapping": {
            "EC2": "compute",
            "RDS": "database",
            "Elasticache": "database",
        },
    }


class TestLoadAwsComponentMapping:
    """Test the load_aws_component_mapping function."""

    @patch("os.path.join")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_mapping_success(self, mock_file, mock_join, mock_aws_mapping):
        """Test successful loading of AWS component mapping."""
        mock_join.return_value = "/path/to/mapping.json"
        mock_file.return_value.read.return_value = json.dumps(mock_aws_mapping)

        result = load_aws_component_mapping()

        assert result == mock_aws_mapping
        mock_file.assert_called_once_with("/path/to/mapping.json", "r")

    @patch("os.path.join")
    @patch("builtins.open", side_effect=FileNotFoundError)
    @patch("logging.warning")
    def test_load_mapping_file_not_found(self, mock_warning, mock_file, mock_join):
        """Test handling of missing mapping file."""
        mock_join.return_value = "/path/to/mapping.json"

        result = load_aws_component_mapping()

        assert result == {"common_names": {}, "category_mapping": {}}
        mock_warning.assert_called_once()

    @patch("os.path.join")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load", side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    @patch("logging.error")
    def test_load_mapping_invalid_json(self, mock_error, mock_json, mock_file, mock_join):
        """Test handling of invalid JSON in mapping file."""
        mock_join.return_value = "/path/to/mapping.json"

        result = load_aws_component_mapping()

        assert result == {"common_names": {}, "category_mapping": {}}
        mock_error.assert_called_once()


class TestGetAwsNode:
    """Test the get_aws_node function."""

    @patch("strands_tools.diagram.load_aws_component_mapping")
    @patch("importlib.import_module")
    def test_get_aws_node_with_common_name(self, mock_import, mock_load_mapping, mock_aws_mapping):
        """Test getting AWS node with common name mapping."""
        mock_load_mapping.return_value = mock_aws_mapping
        mock_module = MagicMock()
        mock_ec2_class = MagicMock()
        mock_module.EC2 = mock_ec2_class
        mock_import.return_value = mock_module

        result = get_aws_node("ec2")

        assert result == mock_ec2_class
        mock_import.assert_any_call("diagrams.aws.compute")

    @patch("strands_tools.diagram.load_aws_component_mapping")
    @patch("importlib.import_module")
    def test_get_aws_node_direct_mapping(self, mock_import, mock_load_mapping, mock_aws_mapping):
        """Test getting AWS node with direct category mapping."""
        mock_load_mapping.return_value = mock_aws_mapping
        mock_module = MagicMock()
        mock_rds_class = MagicMock()
        mock_module.RDS = mock_rds_class
        mock_import.return_value = mock_module

        result = get_aws_node("RDS")

        assert result == mock_rds_class
        mock_import.assert_any_call("diagrams.aws.database")

    @patch("strands_tools.diagram.load_aws_component_mapping")
    @patch("importlib.import_module")
    def test_get_aws_node_fallback_search(self, mock_import, mock_load_mapping):
        """Test getting AWS node with fallback category search."""
        mock_load_mapping.return_value = {"common_names": {}, "category_mapping": {}}

        mock_compute_module = MagicMock()
        mock_compute_module.UnknownService = MagicMock()

        def import_side_effect(module_name):
            if module_name == "diagrams.aws.compute":
                return mock_compute_module
            elif "diagrams.aws" in module_name:
                raise ImportError("Module not found")
            else:
                return MagicMock()

        mock_import.side_effect = import_side_effect

        result = get_aws_node("UnknownService")

        assert result == mock_compute_module.UnknownService

    @patch("strands_tools.diagram.load_aws_component_mapping")
    @patch("importlib.import_module")
    def test_get_aws_node_not_found(self, mock_import, mock_load_mapping):
        """Test error when AWS node is not found."""
        mock_load_mapping.return_value = {"common_names": {}, "category_mapping": {}}

        def import_side_effect(module_name):
            if "diagrams.aws" in module_name:
                raise ImportError("Module not found")
            else:
                return MagicMock()

        mock_import.side_effect = import_side_effect

        with pytest.raises(ValueError, match="AWS component 'NonExistentService' not found"):
            get_aws_node("NonExistentService")


class TestRenderAsMermaid:
    """Test the render_as_mermaid function."""

    def test_render_mermaid_basic(self, sample_nodes, sample_edges):
        """Test basic mermaid rendering."""
        result = render_as_mermaid(sample_nodes, sample_edges, "Test Diagram", {})

        expected_lines = [
            "```mermaid",
            "graph TD",
            "    web[Web Server]",
            "    db[Database]",
            "    cache[Cache]",
            "    web --> |queries|db",
            "    web --> |caches|cache",
            "```",
        ]

        assert result == "\n".join(expected_lines)

    def test_render_mermaid_left_to_right(self, sample_nodes, sample_edges):
        """Test mermaid rendering with left-to-right layout."""
        style = {"rankdir": "LR"}
        result = render_as_mermaid(sample_nodes, sample_edges, "Test Diagram", style)

        assert "graph LR" in result

    def test_render_mermaid_no_edges(self, sample_nodes):
        """Test mermaid rendering with no edges."""
        result = render_as_mermaid(sample_nodes, [], "Test Diagram", {})

        assert "web[Web Server]" in result
        assert "db[Database]" in result
        assert "-->" not in result

    def test_render_mermaid_edge_without_label(self, sample_nodes):
        """Test mermaid rendering with edge without label."""
        edges = [{"from": "web", "to": "db"}]
        result = render_as_mermaid(sample_nodes, edges, "Test Diagram", {})

        assert "web --> db" in result


class TestRenderAsAscii:
    """Test the render_as_ascii function."""

    def test_render_ascii_basic(self, sample_nodes, sample_edges):
        """Test basic ASCII rendering."""
        result = render_as_ascii(sample_nodes, sample_edges, "Test Diagram", {})

        assert "# Test Diagram" in result
        assert "[Web Server]" in result
        assert "[Database]" in result
        assert "[Cache]" in result
        assert "web ↓ db (queries)" in result
        assert "web ↓ cache (caches)" in result

    def test_render_ascii_left_to_right(self, sample_nodes, sample_edges):
        """Test ASCII rendering with left-to-right layout."""
        style = {"rankdir": "LR"}
        result = render_as_ascii(sample_nodes, sample_edges, "Test Diagram", style)

        assert "web --> db (queries)" in result
        assert "web --> cache (caches)" in result


class TestDiagramTool:
    """Test the main diagram tool function."""

    def test_diagram_builder_render_delegation(self):
        """Test that DiagramBuilder.render properly delegates to specific render methods."""
        nodes = [{"id": "test", "label": "Test"}]
        builder = DiagramBuilder(nodes)

        # Test that render method delegates correctly
        with patch.object(builder, "_render_cloud") as mock_cloud:
            mock_cloud.return_value = "/test/cloud.png"
            result = builder.render("cloud", "png")
            assert result == "/test/cloud.png"
            mock_cloud.assert_called_once_with("png")

        with patch.object(builder, "_render_graph") as mock_graph:
            mock_graph.return_value = "/test/graph.png"
            result = builder.render("graph", "png")
            assert result == "/test/graph.png"
            mock_graph.assert_called_once_with("png")

        with patch.object(builder, "_render_network") as mock_network:
            mock_network.return_value = "/test/network.png"
            result = builder.render("network", "png")
            assert result == "/test/network.png"
            mock_network.assert_called_once_with("png")

        with patch.object(builder, "_render_sequence") as mock_sequence:
            mock_sequence.return_value = "/test/sequence.png"
            result = builder.render("sequence", "png")
            assert result == "/test/sequence.png"
            mock_sequence.assert_called_once_with("png")

    def test_diagram_builder_text_format_delegation(self):
        """Test that text formats are handled by _render_text method."""
        nodes = [{"id": "test", "label": "Test"}]
        builder = DiagramBuilder(nodes)

        with patch.object(builder, "_render_text") as mock_text:
            mock_text.return_value = "/test/diagram.md"

            # Test mermaid format for different diagram types
            result = builder.render("graph", "mermaid")
            assert result == "/test/diagram.md"
            mock_text.assert_called_with("mermaid")

            result = builder.render("network", "ascii")
            assert result == "/test/diagram.md"
            mock_text.assert_called_with("ascii")


class TestSequenceDiagramRendering:
    """Test sequence diagram specific functionality."""

    def test_render_sequence_no_nodes(self):
        """Test sequence diagram rendering with no nodes."""
        builder = DiagramBuilder([])

        with pytest.raises(ValueError, match="At least one node is required for sequence diagram"):
            builder._render_sequence("png")


class TestEdgeCases:
    """Test various edge cases and error conditions."""

    def test_render_with_empty_nodes_list(self):
        """Test rendering with empty nodes list for non-cloud diagrams."""
        builder = DiagramBuilder([])

        # Should work for text formats
        with patch.object(builder, "_render_text") as mock_render_text:
            mock_render_text.return_value = "/test/empty.md"
            result = builder.render("graph", "mermaid")
            assert result == "/test/empty.md"
            mock_render_text.assert_called_once_with("mermaid")

    def test_render_with_none_edges(self):
        """Test rendering with None edges."""
        nodes = [{"id": "test", "label": "Test"}]
        builder = DiagramBuilder(nodes, None)  # None edges

        assert builder.edges == []

    def test_diagram_builder_style_defaults(self):
        """Test DiagramBuilder handles style defaults correctly."""
        nodes = [{"id": "test", "label": "Test"}]

        # Test with empty style dict
        builder = DiagramBuilder(nodes, [], "test", {})
        assert builder.style == {}

        # Test with None style
        builder = DiagramBuilder(nodes, [], "test", None)
        assert builder.style == {}

    def test_render_as_mermaid_edge_cases(self):
        """Test render_as_mermaid with various edge cases."""
        nodes = [{"id": "a"}, {"id": "b", "label": "Node B"}]
        edges = [{"from": "a", "to": "b"}]  # No label

        result = render_as_mermaid(nodes, edges, "Test", {})

        # Should handle missing labels gracefully
        assert "a[a]" in result  # Node without label uses ID
        assert "b[Node B]" in result  # Node with label
        assert "a --> b" in result  # Edge without label

    def test_render_as_ascii_edge_cases(self):
        """Test render_as_ascii with various edge cases."""
        nodes = [{"id": "a"}, {"id": "b", "label": "Node B"}]
        edges = [{"from": "a", "to": "b"}]  # No label

        result = render_as_ascii(nodes, edges, "Test", {})

        # Should handle missing labels gracefully
        assert "[a]" in result  # Node without label uses ID
        assert "[Node B]" in result  # Node with label
        assert "a ↓ b" in result  # Edge without label
