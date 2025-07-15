"""
Tests for the agent_graph tool.
"""

# isort: skip_file
import time  # noqa: F401 - used in patches
import uuid  # noqa: F401 - used in patches
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent

from strands_tools.agent_graph import (
    AgentGraph,
    AgentGraphManager,
    AgentNode,
    MAX_QUEUE_SIZE,
    agent_graph,
    create_rich_status_panel,
    create_rich_table,
    create_rich_tree,
)


@pytest.fixture
def mock_use_llm():
    """Create a mock for the use_llm function."""
    with patch("strands_tools.agent_graph.use_llm") as mock:
        mock.return_value = {"status": "success", "content": [{"text": "Mocked response from LLM"}]}
        yield mock


@pytest.fixture
def mock_console():
    """Create a mock for the rich console."""
    with patch("strands_tools.agent_graph.console_util") as mock_console_util:
        mock_console = mock_console_util.create.return_value

        capture_mock = MagicMock()
        capture_mock.get.return_value = "Mocked formatted output"
        mock_console.capture.return_value.__enter__.return_value = capture_mock
        mock_console.print = MagicMock()
        yield mock_console


@pytest.fixture
def mock_thread_pool():
    """Create a mock for ThreadPoolExecutor."""
    with patch("strands_tools.agent_graph.ThreadPoolExecutor") as mock_pool_class:
        mock_pool = MagicMock()
        mock_pool.submit.return_value = MagicMock()
        mock_pool.shutdown = MagicMock()
        mock_pool_class.return_value = mock_pool
        yield mock_pool


@pytest.fixture
def tool_context():
    """Create a tool context for testing."""
    return {
        "bedrock_client": MagicMock(),
        "system_prompt": "Test system prompt",
        "inference_config": {},
        "messages": [],
        "tool_config": {},
    }


@pytest.fixture
def reset_manager():
    """Reset the global manager before and after tests."""
    import strands_tools.agent_graph

    strands_tools.agent_graph._MANAGER = None
    yield
    strands_tools.agent_graph._MANAGER = None


@pytest.fixture
def agent():
    """Create an agent with the agent_graph tool loaded."""
    from strands_tools import agent_graph as agent_graph_module

    return Agent(tools=[agent_graph_module])


def test_create_rich_table(mock_console):
    """Test the create_rich_table function."""
    result = create_rich_table(mock_console, "Test Table", ["Col1"], [["Value1"]])
    mock_console.print.assert_called_once()
    assert result == "Mocked formatted output"


def test_create_rich_tree(mock_console):
    """Test the create_rich_tree function."""
    result = create_rich_tree(mock_console, "Test Tree", {"key": "value"})
    mock_console.print.assert_called_once()
    assert result == "Mocked formatted output"


def test_create_rich_status_panel(mock_console):
    """Test the create_rich_status_panel function with all fields present."""
    status = {
        "graph_id": "test_graph",
        "topology": "star",
        "nodes": [{"id": "node1", "role": "test_role", "neighbors": ["node2"], "queue_size": 0}],
    }
    result = create_rich_status_panel(mock_console, status)
    mock_console.print.assert_called_once()
    assert result == "Mocked formatted output"


class TestAgentNode:
    """Tests for the AgentNode class."""

    def test_init(self):
        """Test AgentNode initialization."""
        node = AgentNode("test_node", "test_role", "Test system prompt")
        assert node.id == "test_node"
        assert node.role == "test_role"
        assert node.system_prompt == "Test system prompt"
        assert node.neighbors == []
        assert node.input_queue.maxsize == MAX_QUEUE_SIZE
        assert node.is_running is True

    def test_add_neighbor(self):
        """Test adding a neighbor to an AgentNode."""
        node1 = AgentNode("node1", "role1", "System prompt 1")
        node2 = AgentNode("node2", "role2", "System prompt 2")

        node1.add_neighbor(node2)
        assert node2 in node1.neighbors

        # Test adding the same neighbor again
        node1.add_neighbor(node2)
        assert len(node1.neighbors) == 1  # Should still be just one instance

    def test_process_messages(self, mock_use_llm, tool_context):
        """Test processing messages with mocked methods."""
        node1 = AgentNode("node1", "role1", "System prompt 1")
        node2 = AgentNode("node2", "role2", "System prompt 2")

        node1.add_neighbor(node2)
        node1.input_queue.put({"content": "Test message"})

        # Mock related functions to avoid actual processing
        with (
            patch("time.time", return_value=0),
            patch("time.sleep"),
            patch.object(node1, "process_messages", return_value=None) as mock_process,
        ):
            node1.is_running = False  # To ensure loop exits
            node1.process_messages(tool_context, "test_channel")
            mock_process.assert_called_once()


class TestAgentGraph:
    """Tests for the AgentGraph class."""

    def test_init(self, tool_context):
        """Test AgentGraph initialization."""
        graph = AgentGraph("test_graph", "star", tool_context)
        assert graph.graph_id == "test_graph"
        assert graph.topology_type == "star"
        assert graph.nodes == {}
        assert graph.tool_context == tool_context
        assert graph.channel == "agent_graph_test_graph"

    def test_add_node(self, tool_context):
        """Test adding a node to the graph."""
        graph = AgentGraph("test_graph", "star", tool_context)
        node = graph.add_node("test_node", "test_role", "Test system prompt")
        assert graph.nodes["test_node"] == node
        assert node.id == "test_node"
        assert node.role == "test_role"

    def test_add_edge(self, tool_context):
        """Test adding an edge to the graph."""
        graph = AgentGraph("test_graph", "star", tool_context)
        node1 = graph.add_node("node1", "role1", "prompt1")
        node2 = graph.add_node("node2", "role2", "prompt2")

        graph.add_edge("node1", "node2")
        assert node2 in node1.neighbors
        assert node1 not in node2.neighbors  # In star topology, connection is one-way

    def test_add_edge_mesh_topology(self, tool_context):
        """Test adding an edge in a mesh topology (bidirectional)."""
        graph = AgentGraph("test_graph", "mesh", tool_context)
        node1 = graph.add_node("node1", "role1", "prompt1")
        node2 = graph.add_node("node2", "role2", "prompt2")

        graph.add_edge("node1", "node2")
        assert node2 in node1.neighbors
        assert node1 in node2.neighbors  # In mesh topology, connection is bidirectional

    def test_start(self, tool_context, mock_thread_pool):
        """Test starting the graph."""
        graph = AgentGraph("test_graph", "star", tool_context)
        # No need to save the node reference as it's not used in assertions
        graph.add_node("node1", "role1", "prompt1")
        graph.start()
        assert mock_thread_pool.submit.call_count == 1

    def test_stop(self, tool_context, mock_thread_pool):
        """Test stopping the graph."""
        graph = AgentGraph("test_graph", "star", tool_context)
        node1 = graph.add_node("node1", "role1", "prompt1")
        node2 = graph.add_node("node2", "role2", "prompt2")

        graph.stop()
        assert node1.is_running is False
        assert node2.is_running is False
        mock_thread_pool.shutdown.assert_called_once_with(wait=True)

    def test_send_message(self, tool_context):
        """Test sending a message to a node."""
        graph = AgentGraph("test_graph", "star", tool_context)
        node1 = graph.add_node("node1", "role1", "prompt1")

        success = graph.send_message("node1", "Test message")
        assert success is True
        assert not node1.input_queue.empty()
        assert node1.input_queue.get()["content"] == "Test message"

    def test_get_status(self, tool_context):
        """Test getting the graph status."""
        graph = AgentGraph("test_graph", "star", tool_context)
        # Add nodes but we don't need to keep references since we access them through status
        graph.add_node("node1", "role1", "prompt1")
        graph.add_node("node2", "role2", "prompt2")
        graph.add_edge("node1", "node2")

        status = graph.get_status()
        assert status["graph_id"] == "test_graph"
        assert status["topology"] == "star"
        assert len(status["nodes"]) == 2

        node1_status = next(node for node in status["nodes"] if node["id"] == "node1")
        assert node1_status["neighbors"] == ["node2"]


class TestAgentGraphManager:
    """Tests for the AgentGraphManager class."""

    def test_init(self, tool_context):
        """Test AgentGraphManager initialization."""
        manager = AgentGraphManager(tool_context)
        assert manager.graphs == {}
        assert manager.tool_context == tool_context

    def test_create_graph(self, tool_context):
        """Test creating a new graph."""
        manager = AgentGraphManager(tool_context)

        # Define a simple graph topology
        graph_id = "test_graph"
        topology = {
            "type": "star",
            "nodes": [
                {
                    "id": "central",
                    "role": "coordinator",
                    "system_prompt": "You are a coordinator",
                },
                {
                    "id": "agent1",
                    "role": "agent",
                    "system_prompt": "You are an agent",
                },
            ],
            "edges": [{"from": "central", "to": "agent1"}],
        }

        # Mock the AgentGraph.start method
        with patch.object(AgentGraph, "start") as mock_start:
            result = manager.create_graph(graph_id, topology)
            assert result["status"] == "success"
            assert graph_id in manager.graphs
            mock_start.assert_called_once()

    def test_stop_graph(self, tool_context):
        """Test stopping a graph."""
        manager = AgentGraphManager(tool_context)

        # Add a mock graph
        mock_graph = MagicMock()
        manager.graphs["test_graph"] = mock_graph

        result = manager.stop_graph("test_graph")
        assert result["status"] == "success"
        assert "test_graph" not in manager.graphs
        mock_graph.stop.assert_called_once()

    def test_send_message(self, tool_context):
        """Test sending a message to a graph."""
        manager = AgentGraphManager(tool_context)

        # Add a mock graph
        mock_graph = MagicMock()
        mock_graph.send_message.return_value = True
        manager.graphs["test_graph"] = mock_graph

        message = {"target": "node1", "content": "Test message"}
        result = manager.send_message("test_graph", message)

        assert result["status"] == "success"
        mock_graph.send_message.assert_called_once_with("node1", "Test message")

    def test_get_graph_status(self, tool_context):
        """Test getting a graph's status."""
        manager = AgentGraphManager(tool_context)

        # Add a mock graph
        mock_graph = MagicMock()
        mock_graph.get_status.return_value = {"graph_id": "test_graph", "nodes": []}
        manager.graphs["test_graph"] = mock_graph

        result = manager.get_graph_status("test_graph")

        assert result["status"] == "success"
        assert "data" in result
        mock_graph.get_status.assert_called_once()

    def test_list_graphs(self, tool_context):
        """Test listing all graphs."""
        manager = AgentGraphManager(tool_context)

        # Add some mock graphs
        graph1 = MagicMock()
        graph1.topology_type = "star"
        graph1.nodes = {"node1": MagicMock(), "node2": MagicMock()}

        graph2 = MagicMock()
        graph2.topology_type = "mesh"
        graph2.nodes = {"node3": MagicMock()}

        manager.graphs = {"graph1": graph1, "graph2": graph2}

        result = manager.list_graphs()

        assert result["status"] == "success"
        assert len(result["data"]) == 2
        assert result["data"][0]["graph_id"] in ["graph1", "graph2"]


def test_agent_interface(agent):
    """Test using agent_graph through the agent interface."""
    try:
        result = agent.tool.agent_graph(action="list")
        assert isinstance(result, dict)
        assert "status" in result
    except Exception as e:
        pytest.fail(f"Agent interface call failed with {e}")


class TestAgentGraphTool:
    """Tests for the agent_graph function (tool itself)."""

    def test_create_action_success(self, mock_console):
        """Test the create action with valid inputs."""
        tool_use = {
            "toolUseId": "test-tool-use-id",
            "input": {
                "action": "create",
                "graph_id": "test_graph",
                "topology": {
                    "type": "star",
                    "nodes": [{"id": "central", "role": "coordinator", "system_prompt": "test"}],
                },
            },
        }

        # Mock the manager
        with patch("strands_tools.agent_graph.get_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.create_graph.return_value = {"status": "success", "message": "Graph created"}
            mock_get_manager.return_value = mock_manager

            result = agent_graph(tool=tool_use)
            assert result["status"] == "success"

    def test_stop_action_success(self, mock_console):
        """Test the stop action with valid inputs."""
        tool_use = {"toolUseId": "test-id", "input": {"action": "stop", "graph_id": "test_graph"}}

        with patch("strands_tools.agent_graph.get_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.stop_graph.return_value = {"status": "success", "message": "Graph stopped"}
            mock_get_manager.return_value = mock_manager

            result = agent_graph(tool=tool_use)
            assert result["status"] == "success"

    def test_message_action_success(self, mock_console):
        """Test the message action with valid inputs."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {
                "action": "message",
                "graph_id": "test_graph",
                "message": {"target": "node1", "content": "Test message"},
            },
        }

        with patch("strands_tools.agent_graph.get_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.send_message.return_value = {"status": "success", "message": "Message sent"}
            mock_get_manager.return_value = mock_manager

            result = agent_graph(tool=tool_use)
            assert result["status"] == "success"

    def test_status_action_success(self, mock_console):
        """Test the status action with valid inputs."""
        tool_use = {"toolUseId": "test-id", "input": {"action": "status", "graph_id": "test_graph"}}

        with patch("strands_tools.agent_graph.get_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.get_graph_status.return_value = {
                "status": "success",
                "data": {"graph_id": "test_graph", "topology": "star", "nodes": []},
            }
            mock_get_manager.return_value = mock_manager

            result = agent_graph(tool=tool_use)
            assert result["status"] == "success"

    def test_list_action_success(self, mock_console):
        """Test the list action."""
        tool_use = {"toolUseId": "test-id", "input": {"action": "list"}}

        with patch("strands_tools.agent_graph.get_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.list_graphs.return_value = {"status": "success", "data": []}
            mock_get_manager.return_value = mock_manager

            result = agent_graph(tool=tool_use)
            assert result["status"] == "success"

    def test_unknown_action(self):
        """Test handling an unknown action."""
        tool_use = {"toolUseId": "test-id", "input": {"action": "unknown_action"}}

        result = agent_graph(tool=tool_use)
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    def test_create_action_missing_topology(self):
        """Test the create action with missing topology parameter."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {
                "action": "create",
                "graph_id": "test_graph",  # Missing topology
            },
        }

        result = agent_graph(tool=tool_use)
        assert result["status"] == "error"
        assert "topology" in result["content"][0]["text"]

    def test_create_action_missing_graph_id(self):
        """Test the create action with missing graph_id parameter."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {
                "action": "create",
                "topology": {"type": "star", "nodes": []},  # Missing graph_id
            },
        }

        result = agent_graph(tool=tool_use)
        assert result["status"] == "error"
        assert "graph_id" in result["content"][0]["text"]

    def test_stop_action_missing_graph_id(self):
        """Test the stop action with missing graph_id parameter."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"action": "stop"},  # Missing graph_id
        }

        result = agent_graph(tool=tool_use)
        assert result["status"] == "error"
        assert "graph_id is required" in result["content"][0]["text"]

    def test_message_action_missing_params(self):
        """Test the message action with missing parameters."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"action": "message", "graph_id": "test_graph"},  # Missing message
        }

        result = agent_graph(tool=tool_use)
        assert result["status"] == "error"
        assert "message are required" in result["content"][0]["text"]

    def test_status_action_missing_graph_id(self):
        """Test the status action with missing graph_id parameter."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"action": "status"},  # Missing graph_id
        }

        result = agent_graph(tool=tool_use)
        assert result["status"] == "error"
        assert "graph_id is required" in result["content"][0]["text"]

    def test_exception_handling(self):
        """Test exception handling in the tool function."""
        tool_use = {"toolUseId": "test-id", "input": {"action": "list"}}

        # Mock get_manager to raise an exception
        with patch("strands_tools.agent_graph.get_manager", side_effect=Exception("Test exception")):
            with patch("strands_tools.agent_graph.logger.error") as mock_logger:
                result = agent_graph(tool=tool_use)
                assert result["status"] == "error"
                mock_logger.assert_called_once()

    def test_manager_error_result(self):
        """Test error result handling from the manager."""
        tool_use = {"toolUseId": "test-id", "input": {"action": "list"}}

        with patch("strands_tools.agent_graph.get_manager") as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.list_graphs.return_value = {"status": "error", "message": "Test error message"}
            mock_get_manager.return_value = mock_manager

            with patch("strands_tools.agent_graph.logger.error") as mock_logger:
                result = agent_graph(tool=tool_use)
                assert result["status"] == "error"
                assert "Test error message" in result["content"][0]["text"]
                mock_logger.assert_called_once()

    def test_default_tool_use_id(self):
        """Test generation of default tool_use_id."""
        tool_use = {
            # No toolUseId
            "input": {"action": "list"}
        }

        with (
            patch("strands_tools.agent_graph.get_manager") as mock_get_manager,
            patch("uuid.uuid4", return_value="generated-uuid"),
        ):
            mock_manager = MagicMock()
            mock_manager.list_graphs.return_value = {"status": "success", "data": []}
            mock_get_manager.return_value = mock_manager

            result = agent_graph(tool=tool_use)
            assert result["toolUseId"] == "generated-uuid"
