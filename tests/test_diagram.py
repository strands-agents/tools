"""Tests for the diagram module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from strands_tools.diagram import (
    DiagramBuilder,
    UMLDiagramBuilder,
    diagram,
    get_aws_node,
    open_diagram,
    save_diagram_to_directory,
)


class TestGetAwsNode:
    """Tests for the get_aws_node function."""

    def test_get_aws_node_case_insensitive_match(self):
        """Test retrieving a node with case-insensitive matching."""
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock(spec=["Lambda"])
            mock_module.Lambda = "LambdaClass"

            def side_effect(module_name):
                if module_name == "diagrams.aws.compute":
                    return mock_module
                return MagicMock(spec=[])

            mock_import.side_effect = side_effect

            # Should find Lambda even with different case
            result = get_aws_node("lambda")
            assert result == "LambdaClass"


class TestSaveDiagramToDirectory:
    """Tests for the save_diagram_to_directory function."""

    def test_save_diagram_to_directory_without_content(self, tmp_path):
        """Test saving a diagram without content."""
        with patch("os.getcwd", return_value=str(tmp_path)):
            result = save_diagram_to_directory("test_diagram", "png")
            expected_path = os.path.join(tmp_path, "diagrams", "test_diagram.png")
            assert result == expected_path
            assert os.path.exists(os.path.dirname(result))

    def test_save_diagram_to_directory_with_content(self, tmp_path):
        """Test saving a diagram with content."""
        with patch("os.getcwd", return_value=str(tmp_path)):
            result = save_diagram_to_directory("test_diagram", "txt", "Test content")
            expected_path = os.path.join(tmp_path, "diagrams", "test_diagram.txt")
            assert result == expected_path
            assert os.path.exists(result)
            with open(result, "r") as f:
                assert f.read() == "Test content"


class TestOpenDiagram:
    """Tests for the open_diagram function."""

    def test_open_diagram_file_not_exists(self, caplog):
        """Test opening a non-existent diagram."""
        open_diagram("non_existent_file.png")
        assert "Cannot open diagram: file does not exist" in caplog.text

    @patch("os.path.exists", return_value=True)
    def test_open_diagram_macos(self, mock_exists, caplog):
        """Test opening a diagram on macOS."""
        with patch("platform.system", return_value="Darwin"):
            with patch("subprocess.Popen") as mock_popen:
                with patch("logging.info") as mock_log:
                    open_diagram("test_diagram.png")
                    mock_popen.assert_called_once_with(["open", "test_diagram.png"], start_new_session=True)
                    mock_log.assert_called_once_with("Opened diagram: test_diagram.png")

    @patch("os.path.exists", return_value=True)
    def test_open_diagram_windows(self, mock_exists, caplog):
        """Test opening a diagram on Windows."""
        with patch("platform.system", return_value="Windows"):
            # Skip the actual startfile call since it's not available on macOS
            with patch("strands_tools.diagram.os") as mock_os:
                mock_startfile = MagicMock()
                mock_os.startfile = mock_startfile
                with patch("logging.info") as mock_log:
                    open_diagram("test_diagram.png")
                    mock_startfile.assert_called_once_with("test_diagram.png")
                    mock_log.assert_called_once_with("Opened diagram: test_diagram.png")

    @patch("os.path.exists", return_value=True)
    def test_open_diagram_linux(self, mock_exists, caplog):
        """Test opening a diagram on Linux."""
        with patch("platform.system", return_value="Linux"):
            with patch("subprocess.Popen") as mock_popen:
                with patch("logging.info") as mock_log:
                    open_diagram("test_diagram.png")
                    mock_popen.assert_called_once_with(["xdg-open", "test_diagram.png"], start_new_session=True)
                    mock_log.assert_called_once_with("Opened diagram: test_diagram.png")

    @patch("os.path.exists", return_value=True)
    def test_open_diagram_error(self, mock_exists, caplog):
        """Test error handling when opening a diagram."""
        with patch("platform.system", return_value="Darwin"):
            with patch("subprocess.Popen", side_effect=FileNotFoundError("Command not found")):
                open_diagram("test_diagram.png")
                assert "System command not found for opening files" in caplog.text


class TestDiagramBuilder:
    """Tests for the DiagramBuilder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.nodes = [{"id": "node1", "label": "Node 1"}, {"id": "node2", "label": "Node 2"}]
        self.edges = [{"from": "node1", "to": "node2", "label": "connects to"}]
        self.builder = DiagramBuilder(self.nodes, self.edges, "Test Diagram")

    def test_init(self):
        """Test initialization of DiagramBuilder."""
        assert self.builder.nodes == self.nodes
        assert self.builder.edges == self.edges
        assert self.builder.title == "Test Diagram"
        assert self.builder.style == {}

    def test_render_unsupported_type(self):
        """Test rendering with unsupported diagram type."""
        with pytest.raises(ValueError) as excinfo:
            self.builder.render("unsupported", "png")
        assert "Unsupported diagram type: unsupported" in str(excinfo.value)

    @patch("strands_tools.diagram.save_diagram_to_directory")
    @patch("strands_tools.diagram.open_diagram")
    @patch("graphviz.Digraph.render")
    def test_render_graph(self, mock_render, mock_open, mock_save):
        """Test rendering a graph diagram."""
        mock_save.return_value = "test_path"
        mock_render.return_value = "test_path.png"

        result = self.builder.render("graph", "png")

        assert result == "test_path.png"
        mock_save.assert_called_once()
        mock_render.assert_called_once()
        mock_open.assert_called_once_with("test_path.png")

    @patch("strands_tools.diagram.save_diagram_to_directory")
    @patch("strands_tools.diagram.open_diagram")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.figure")
    @patch("networkx.spring_layout")
    @patch("networkx.draw_networkx_nodes")
    @patch("networkx.draw_networkx_edges")
    @patch("networkx.draw_networkx_labels")
    @patch("networkx.draw_networkx_edge_labels")
    def test_render_network(
        self,
        mock_edge_labels,
        mock_labels,
        mock_edges,
        mock_nodes,
        mock_layout,
        mock_figure,
        mock_savefig,
        mock_open,
        mock_save,
    ):
        """Test rendering a network diagram."""
        mock_save.return_value = "test_path.png"
        mock_layout.return_value = {node["id"]: [0, 0] for node in self.nodes}

        # Create edges without labels to avoid the networkx issue
        builder = DiagramBuilder(self.nodes, [{"from": "node1", "to": "node2"}], "Test Diagram")

        result = builder.render("network", "png")

        assert result == "test_path.png"
        mock_save.assert_called_once()
        mock_savefig.assert_called_once()
        mock_open.assert_called_once_with("test_path.png")
        # Verify other mocks were used (even if we don't check specific calls)
        assert mock_figure.called
        assert mock_nodes.called
        assert mock_edges.called
        assert mock_labels.called

    @patch("strands_tools.diagram.save_diagram_to_directory")
    @patch("strands_tools.diagram.open_diagram")
    @patch("strands_tools.diagram.CloudDiagram")
    @patch("strands_tools.diagram.get_aws_node")
    def test_render_cloud(self, mock_get_aws_node, mock_cloud_diagram, mock_open, mock_save):
        """Test rendering a cloud diagram."""
        mock_save.return_value = "test_path"
        mock_get_aws_node.return_value = MagicMock()

        # Create a context manager mock for CloudDiagram
        mock_cm = MagicMock()
        mock_cloud_diagram.return_value = mock_cm

        nodes = [{"id": "ec2", "type": "EC2"}, {"id": "s3", "type": "S3"}]
        edges = [{"from": "ec2", "to": "s3"}]
        builder = DiagramBuilder(nodes, edges, "Cloud Diagram")

        result = builder.render("cloud", "png")

        assert result == "test_path.png"
        mock_save.assert_called_once()
        mock_cloud_diagram.assert_called_once()
        mock_open.assert_called_once_with("test_path.png")
        mock_get_aws_node.assert_called()

    def test_render_cloud_no_nodes(self):
        """Test rendering a cloud diagram with no nodes."""
        builder = DiagramBuilder([], [], "Empty Cloud")
        with pytest.raises(ValueError) as excinfo:
            builder.render("cloud", "png")
        assert "At least one node is required for cloud diagram" in str(excinfo.value)

    def test_render_cloud_invalid_node(self):
        """Test rendering a cloud diagram with invalid node type."""
        nodes = [{"id": "invalid", "type": "NonExistentService"}]
        builder = DiagramBuilder(nodes, [], "Invalid Cloud")

        with patch("strands_tools.diagram.get_aws_node", side_effect=ValueError("Component not found")):
            with pytest.raises(ValueError) as excinfo:
                builder.render("cloud", "png")
            assert "Invalid AWS component types found" in str(excinfo.value)

    def test_render_cloud_missing_node_id(self):
        """Test rendering a cloud diagram with a node missing an ID."""
        nodes = [{"type": "EC2"}]  # Missing 'id' field
        builder = DiagramBuilder(nodes, [], "Missing ID")

        with pytest.raises(ValueError) as excinfo:
            builder.render("cloud", "png")
        assert "Node missing required 'id' field" in str(excinfo.value)


class TestUMLDiagramBuilder:
    """Tests for the UMLDiagramBuilder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.elements = [
            {"name": "Class1", "attributes": ["attr1", "attr2"], "methods": ["method1", "method2"]},
            {"name": "Class2", "attributes": ["attr3"], "methods": ["method3"]},
        ]
        self.relationships = [{"from": "Class1", "to": "Class2", "type": "inheritance"}]
        self.builder = UMLDiagramBuilder("class", self.elements, self.relationships, "Test UML")

    def test_init(self):
        """Test initialization of UMLDiagramBuilder."""
        assert self.builder.diagram_type == "class"
        assert self.builder.elements == self.elements
        assert self.builder.relationships == self.relationships
        assert self.builder.title == "Test UML"
        assert self.builder.style == {}

    def test_render_unsupported_type(self):
        """Test rendering with unsupported UML diagram type."""
        builder = UMLDiagramBuilder("unsupported", self.elements)
        with pytest.raises(ValueError) as excinfo:
            builder.render("png")
        assert "Unsupported UML diagram type: unsupported" in str(excinfo.value)

    @patch("strands_tools.diagram.UMLDiagramBuilder._save_diagram")
    def test_render_class(self, mock_save):
        """Test rendering a class diagram."""
        mock_save.return_value = "test_path.png"

        result = self.builder.render("png")

        assert result == "test_path.png"
        mock_save.assert_called_once()

    @patch("strands_tools.diagram.UMLDiagramBuilder._save_diagram")
    def test_render_component(self, mock_save):
        """Test rendering a component diagram."""
        mock_save.return_value = "test_path.png"

        elements = [
            {"name": "Component1", "type": "component"},
            {"name": "Interface1", "type": "interface", "provided": True},
        ]
        relationships = [{"from": "Component1", "to": "Interface1", "type": "realization"}]
        builder = UMLDiagramBuilder("component", elements, relationships)

        result = builder.render("png")

        assert result == "test_path.png"
        mock_save.assert_called_once()

    @patch("strands_tools.diagram.UMLDiagramBuilder._save_diagram")
    def test_render_use_case(self, mock_save):
        """Test rendering a use case diagram."""
        mock_save.return_value = "test_path.png"

        elements = [{"name": "User", "type": "actor"}, {"name": "Login", "type": "use_case"}]
        relationships = [{"from": "User", "to": "Login"}]
        builder = UMLDiagramBuilder("use_case", elements, relationships)

        result = builder.render("png")

        assert result == "test_path.png"
        mock_save.assert_called_once()

    @patch("strands_tools.diagram.plt.savefig")
    @patch("strands_tools.diagram.open_diagram")
    @patch("strands_tools.diagram.save_diagram_to_directory")
    def test_render_sequence(self, mock_save, mock_open, mock_savefig):
        """Test rendering a sequence diagram."""
        mock_save.return_value = "test_path.png"

        elements = [{"name": "Object1"}, {"name": "Object2"}]
        relationships = [
            {"from": "Object1", "to": "Object2", "message": "request()", "sequence": 1},
            {"from": "Object2", "to": "Object1", "message": "response()", "sequence": 2},
        ]
        builder = UMLDiagramBuilder("sequence", elements, relationships)

        result = builder.render("png")

        assert result == "test_path.png"
        mock_save.assert_called_once()
        mock_savefig.assert_called_once()
        mock_open.assert_called_once_with("test_path.png")

    def test_render_sequence_no_elements(self):
        """Test rendering a sequence diagram with no elements."""
        builder = UMLDiagramBuilder("sequence", [], [])
        with pytest.raises(ValueError) as excinfo:
            builder.render("png")
        assert "At least one element is required for sequence diagram" in str(excinfo.value)

    @patch("strands_tools.diagram.plt.savefig")
    @patch("strands_tools.diagram.open_diagram")
    @patch("strands_tools.diagram.save_diagram_to_directory")
    def test_render_timing(self, mock_save, mock_open, mock_savefig):
        """Test rendering a timing diagram."""
        mock_save.return_value = "test_path.png"

        elements = [
            {
                "name": "Signal1",
                "states": [{"state": "High", "start": 0, "end": 5}, {"state": "Low", "start": 5, "end": 10}],
            },
            {"name": "Signal2", "states": "High:0-3,Low:3-10"},
        ]
        builder = UMLDiagramBuilder("timing", elements)

        result = builder.render("png")

        assert result == "test_path.png"
        mock_save.assert_called_once()
        mock_savefig.assert_called_once()
        mock_open.assert_called_once_with("test_path.png")

    def test_format_visibility(self):
        """Test the _format_visibility method."""
        # Test with string input
        assert self.builder._format_visibility("testMethod()") == "testMethod()"

        # Test with dict input - public visibility
        member = {"name": "testAttr", "visibility": "public", "type": "int"}
        assert self.builder._format_visibility(member) == "+ testAttr: int"

        # Test with dict input - private visibility
        member = {"name": "testAttr", "visibility": "private", "type": "int"}
        assert self.builder._format_visibility(member) == "- testAttr: int"

        # Test with dict input - no type
        member = {"name": "testAttr", "visibility": "protected"}
        assert self.builder._format_visibility(member) == "# testAttr"


if __name__ == "__main__":
    pytest.main(["-v", "test_diagram.py"])


class TestDiagramBuilderAdditional:
    """Additional tests for DiagramBuilder class."""

    def test_render_cloud_with_edge_errors(self):
        """Test rendering a cloud diagram with edge errors."""
        nodes = [{"id": "ec2", "type": "EC2"}, {"id": "s3", "type": "S3"}]
        edges = [
            {"from": "ec2", "to": "s3"},  # Valid edge
            {"from": "ec2"},  # Missing 'to'
            {"to": "s3"},  # Missing 'from'
            {"from": "nonexistent", "to": "s3"},  # Non-existent source
            {"from": "ec2", "to": "nonexistent"},  # Non-existent target
        ]

        builder = DiagramBuilder(nodes, edges, "Cloud Diagram")

        with (
            patch("strands_tools.diagram.save_diagram_to_directory") as mock_save,
            patch("strands_tools.diagram.open_diagram") as mock_open,
            patch("strands_tools.diagram.CloudDiagram") as mock_cloud_diagram,
            patch("strands_tools.diagram.get_aws_node") as mock_get_aws_node,
            patch("logging.warning") as mock_warning,
        ):
            mock_save.return_value = "test_path"
            mock_get_aws_node.return_value = MagicMock()
            mock_cm = MagicMock()
            mock_cloud_diagram.return_value = mock_cm

            result = builder.render("cloud", "png")

            assert result == "test_path.png"
            # Check that warnings were logged for the invalid edges
            assert mock_warning.call_count == 4
            # Verify other mocks were used
            assert mock_save.called
            assert mock_cloud_diagram.called
            assert mock_open.called
            assert mock_get_aws_node.called

    def test_render_graph_with_aws_types(self):
        """Test rendering a graph diagram with AWS service types."""
        nodes = [
            {"id": "ec2", "type": "EC2", "label": "EC2 Instance"},
            {"id": "s3", "label": "S3 Bucket"},  # No type specified
        ]
        edges = [{"from": "ec2", "to": "s3", "label": "stores data"}]

        builder = DiagramBuilder(nodes, edges, "Graph Diagram")

        with (
            patch("strands_tools.diagram.save_diagram_to_directory") as mock_save,
            patch("strands_tools.diagram.open_diagram") as mock_open,
            patch("graphviz.Digraph.render") as mock_render,
            patch("strands_tools.diagram.get_aws_node") as mock_get_aws_node,
        ):
            mock_save.return_value = "test_path"
            mock_render.return_value = "test_path.png"
            mock_get_aws_node.return_value = MagicMock()

            result = builder.render("graph", "png")

            assert result == "test_path.png"
            # Check that get_aws_node was called for the node with type
            mock_get_aws_node.assert_called_once_with("EC2")
            # Verify other mocks were used
            assert mock_save.called
            assert mock_render.called
            assert mock_open.called

    def test_render_graph_with_invalid_aws_type(self):
        """Test rendering a graph diagram with an invalid AWS service type."""
        nodes = [{"id": "invalid", "type": "NonExistentService"}]
        builder = DiagramBuilder(nodes, [], "Invalid Graph")

        with (
            patch("strands_tools.diagram.save_diagram_to_directory") as mock_save,
            patch("strands_tools.diagram.open_diagram") as mock_open,
            patch("graphviz.Digraph.render") as mock_render,
            patch("strands_tools.diagram.get_aws_node") as mock_get_aws_node,
        ):
            mock_save.return_value = "test_path"
            mock_render.return_value = "test_path.png"
            mock_get_aws_node.side_effect = ValueError("Component not found")

            # Should not raise an error, just skip AWS-specific handling
            result = builder.render("graph", "png")
            assert result == "test_path.png"
            # Verify mocks were used
            assert mock_save.called
            assert mock_render.called
            assert mock_open.called
            assert mock_get_aws_node.called

    def test_render_network_with_aws_service_categories(self):
        """Test rendering a network diagram with AWS service categories for coloring."""
        nodes = [
            {"id": "ec2", "type": "EC2"},  # compute
            {"id": "rds", "type": "RDS"},  # database
            {"id": "vpc", "type": "VPC"},  # network
            {"id": "s3", "type": "S3"},  # storage
            {"id": "iam", "type": "IAM"},  # security
            {"id": "unknown", "type": "UnknownService"},  # default color
            {"id": "noType"},  # no type specified
        ]
        edges = [{"from": "ec2", "to": "rds"}]

        builder = DiagramBuilder(nodes, edges, "Network Diagram")

        with (
            patch("strands_tools.diagram.save_diagram_to_directory") as mock_save,
            patch("strands_tools.diagram.open_diagram") as mock_open,
            patch("matplotlib.pyplot.savefig") as mock_savefig,
            patch("matplotlib.pyplot.figure") as mock_figure,
            patch("networkx.spring_layout") as mock_layout,
            patch("networkx.draw_networkx_nodes") as mock_nodes,
            patch("networkx.draw_networkx_edges") as mock_edges,
            patch("networkx.draw_networkx_labels") as mock_labels,
            patch("strands_tools.diagram.get_aws_node") as mock_get_aws_node,
        ):
            mock_save.return_value = "test_path.png"
            mock_layout.return_value = {node["id"]: [0, 0] for node in nodes}
            mock_get_aws_node.return_value = MagicMock()

            result = builder.render("network", "png")

            assert result == "test_path.png"
            # Should have called get_aws_node for each node with a type
            assert mock_get_aws_node.call_count == 6
            # Verify other mocks were used
            assert mock_savefig.called
            assert mock_figure.called
            assert mock_nodes.called
            assert mock_edges.called
            assert mock_labels.called
            assert mock_open.called


class TestUMLDiagramBuilderAdditional:
    """Additional tests for UMLDiagramBuilder class."""

    def test_render_deployment(self):
        """Test rendering a deployment diagram."""
        elements = [{"name": "Server", "type": "node"}, {"name": "WebApp", "type": "artifact"}]
        relationships = [{"from": "Server", "to": "WebApp", "label": "deploys"}]
        builder = UMLDiagramBuilder("deployment", elements, relationships)

        with patch("strands_tools.diagram.UMLDiagramBuilder._save_diagram") as mock_save:
            mock_save.return_value = "test_path.png"
            result = builder.render("png")
            assert result == "test_path.png"
            mock_save.assert_called_once()

    def test_render_package(self):
        """Test rendering a package diagram."""
        elements = [{"name": "UI", "type": "package"}, {"name": "Core", "type": "package"}]
        relationships = [{"from": "UI", "to": "Core", "type": "dependency"}]
        builder = UMLDiagramBuilder("package", elements, relationships)

        with patch("strands_tools.diagram.UMLDiagramBuilder._save_diagram") as mock_save:
            mock_save.return_value = "test_path.png"
            result = builder.render("png")
            assert result == "test_path.png"
            mock_save.assert_called_once()

    def test_render_profile(self):
        """Test rendering a profile diagram."""
        elements = [
            {"name": "CustomProfile", "stereotype": "profile"},
            {"name": "Extension", "stereotype": "extension"},
        ]
        relationships = [{"from": "Extension", "to": "CustomProfile"}]
        builder = UMLDiagramBuilder("profile", elements, relationships)

        with patch("strands_tools.diagram.UMLDiagramBuilder._save_diagram") as mock_save:
            mock_save.return_value = "test_path.png"
            result = builder.render("png")
            assert result == "test_path.png"
            mock_save.assert_called_once()

    def test_render_composite_structure(self):
        """Test rendering a composite structure diagram."""
        elements = [{"name": "Component", "type": "part"}, {"name": "Port", "type": "port", "owner": "Component"}]
        relationships = [{"from": "Component", "to": "Port", "type": "delegation"}]
        builder = UMLDiagramBuilder("composite_structure", elements, relationships)

        with patch("strands_tools.diagram.UMLDiagramBuilder._save_diagram") as mock_save:
            mock_save.return_value = "test_path.png"
            result = builder.render("png")
            assert result == "test_path.png"
            mock_save.assert_called_once()

    def test_render_activity(self):
        """Test rendering an activity diagram."""
        elements = [
            {"name": "Start", "type": "start"},
            {"name": "Process", "type": "activity"},
            {"name": "Decision", "type": "decision"},
            {"name": "End", "type": "end"},
        ]
        relationships = [
            {"from": "Start", "to": "Process"},
            {"from": "Process", "to": "Decision"},
            {"from": "Decision", "to": "End"},
        ]
        builder = UMLDiagramBuilder("activity", elements, relationships)

        with patch("strands_tools.diagram.UMLDiagramBuilder._save_diagram") as mock_save:
            mock_save.return_value = "test_path.png"
            result = builder.render("png")
            assert result == "test_path.png"
            mock_save.assert_called_once()

    def test_render_state_machine(self):
        """Test rendering a state machine diagram."""
        elements = [
            {"name": "Initial", "type": "initial"},
            {"name": "State1", "type": "state"},
            {"name": "State2", "type": "state"},
            {"name": "Final", "type": "final"},
        ]
        relationships = [
            {"from": "Initial", "to": "State1"},
            {"from": "State1", "to": "State2", "event": "trigger", "action": "doSomething"},
            {"from": "State2", "to": "Final"},
        ]
        builder = UMLDiagramBuilder("state_machine", elements, relationships)

        with patch("strands_tools.diagram.UMLDiagramBuilder._save_diagram") as mock_save:
            mock_save.return_value = "test_path.png"
            result = builder.render("png")
            assert result == "test_path.png"
            mock_save.assert_called_once()

    def test_render_communication(self):
        """Test rendering a communication diagram."""
        elements = [{"name": "Client"}, {"name": "Server"}]
        relationships = [{"from": "Client", "to": "Server", "sequence": "1", "message": "request()"}]
        builder = UMLDiagramBuilder("communication", elements, relationships)

        with patch("strands_tools.diagram.UMLDiagramBuilder._save_diagram") as mock_save:
            mock_save.return_value = "test_path.png"
            result = builder.render("png")
            assert result == "test_path.png"
            mock_save.assert_called_once()

    def test_render_interaction_overview(self):
        """Test rendering an interaction overview diagram."""
        elements = [
            {"name": "Start", "type": "initial"},
            {"name": "Interaction1", "type": "interaction"},
            {"name": "Decision", "type": "decision"},
            {"name": "End", "type": "final"},
        ]
        relationships = [
            {"from": "Start", "to": "Interaction1"},
            {"from": "Interaction1", "to": "Decision"},
            {"from": "Decision", "to": "End", "guard": "condition", "label": "path"},
        ]
        builder = UMLDiagramBuilder("interaction_overview", elements, relationships)

        with patch("strands_tools.diagram.UMLDiagramBuilder._save_diagram") as mock_save:
            mock_save.return_value = "test_path.png"
            result = builder.render("png")
            assert result == "test_path.png"
            mock_save.assert_called_once()

    def test_render_object(self):
        """Test rendering an object diagram."""
        elements = [
            {"name": "user1", "class": "User", "attributes": {"name": "John", "age": 30}},
            {"name": "order1", "class": "Order", "attributes": "id = 12345\ntotal = 99.99"},
        ]
        relationships = [{"from": "user1", "to": "order1", "label": "places"}]
        builder = UMLDiagramBuilder("object", elements, relationships)

        with patch("strands_tools.diagram.UMLDiagramBuilder._save_diagram") as mock_save:
            mock_save.return_value = "test_path.png"
            result = builder.render("png")
            assert result == "test_path.png"
            mock_save.assert_called_once()

    def test_add_class_relationship(self):
        """Test adding different types of class relationships."""
        elements = [{"name": "Class1"}, {"name": "Class2"}]
        builder = UMLDiagramBuilder("class", elements)

        dot = MagicMock()

        # Test inheritance relationship
        builder._add_class_relationship(dot, {"from": "Class1", "to": "Class2", "type": "inheritance"})
        dot.edge.assert_called_with("Class1", "Class2", arrowhead="empty")

        # Test composition relationship
        dot.reset_mock()
        builder._add_class_relationship(dot, {"from": "Class1", "to": "Class2", "type": "composition"})
        dot.edge.assert_called_with("Class1", "Class2", arrowhead="diamond", style="filled")

        # Test aggregation relationship
        dot.reset_mock()
        builder._add_class_relationship(dot, {"from": "Class1", "to": "Class2", "type": "aggregation"})
        dot.edge.assert_called_with("Class1", "Class2", arrowhead="diamond")

        # Test dependency relationship
        dot.reset_mock()
        builder._add_class_relationship(dot, {"from": "Class1", "to": "Class2", "type": "dependency"})
        dot.edge.assert_called_with("Class1", "Class2", style="dashed", arrowhead="open")

        # Test association relationship with multiplicity
        dot.reset_mock()
        builder._add_class_relationship(dot, {"from": "Class1", "to": "Class2", "multiplicity": "1..*"})
        dot.edge.assert_called_with("Class1", "Class2", label="1..*")


class TestDiagramTool:
    """Tests for the diagram tool function."""

    def test_diagram_basic_graph(self):
        """Test creating a basic graph diagram."""
        nodes = [{"id": "node1", "label": "Node 1"}, {"id": "node2", "label": "Node 2"}]
        edges = [{"from": "node1", "to": "node2"}]

        with patch("strands_tools.diagram.DiagramBuilder") as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.render.return_value = "test_path.png"

            result = diagram(diagram_type="graph", nodes=nodes, edges=edges)

            mock_builder.assert_called_once_with(nodes, edges, "diagram", None, True)
            mock_instance.render.assert_called_once_with("graph", "png")
            assert "Created graph diagram" in result

    def test_diagram_cloud(self):
        """Test creating a cloud diagram."""
        nodes = [{"id": "ec2", "type": "EC2"}, {"id": "s3", "type": "S3"}]
        edges = [{"from": "ec2", "to": "s3"}]

        with patch("strands_tools.diagram.DiagramBuilder") as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.render.return_value = "test_path.png"

            result = diagram(diagram_type="cloud", nodes=nodes, edges=edges, title="AWS Architecture")

            mock_builder.assert_called_once_with(nodes, edges, "AWS Architecture", None, True)
            mock_instance.render.assert_called_once_with("cloud", "png")
            assert "Created cloud diagram" in result

    def test_diagram_uml_class(self):
        """Test creating a UML class diagram."""
        elements = [
            {"name": "User", "attributes": ["name: string", "email: string"], "methods": ["login()", "logout()"]}
        ]
        relationships = []

        with patch("strands_tools.diagram.UMLDiagramBuilder") as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.render.return_value = "test_path.png"

            result = diagram(diagram_type="class", elements=elements, relationships=relationships, output_format="svg")

            mock_builder.assert_called_once_with("class", elements, relationships, "diagram", None, True)
            mock_instance.render.assert_called_once_with("svg")
            assert "Created class UML diagram" in result

    def test_diagram_missing_nodes(self):
        """Test diagram with missing nodes for basic diagram."""
        result = diagram(diagram_type="graph")
        assert "Error: 'nodes' parameter is required for basic diagrams" in result

    def test_diagram_missing_elements(self):
        """Test diagram with missing elements for UML diagram."""
        result = diagram(diagram_type="class")
        assert "Error: 'elements' parameter is required for UML diagrams" in result

    def test_diagram_error_handling(self):
        """Test error handling in diagram function."""
        nodes = [{"id": "node1"}]

        with patch("strands_tools.diagram.DiagramBuilder") as mock_builder:
            mock_instance = MagicMock()
            mock_builder.return_value = mock_instance
            mock_instance.render.side_effect = ValueError("Test error")

            result = diagram(diagram_type="graph", nodes=nodes)

            assert "Error creating diagram: Test error" in result


if __name__ == "__main__":
    pytest.main(["-v", "test_diagram.py"])
