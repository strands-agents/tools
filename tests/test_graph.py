"""
Tests for the graph tool using the Strands SDK Graph implementation.
"""

from unittest.mock import MagicMock, patch

import pytest
from strands import Agent

from strands_tools import graph as graph_module


@pytest.fixture
def agent():
    """Create an agent with the graph tool loaded."""
    return Agent(tools=[graph_module])


@pytest.fixture
def mock_parent_agent():
    """Create a mock parent agent with tools and registry."""
    mock_agent = MagicMock()
    mock_tool_registry = MagicMock()
    mock_agent.tool_registry = mock_tool_registry

    # Mock some tools in the registry
    mock_tool_registry.registry = {
        "calculator": MagicMock(),
        "file_read": MagicMock(),
        "file_write": MagicMock(),
        "editor": MagicMock(),
        "http_request": MagicMock(),
        "researcher_tool": MagicMock(),
    }

    # Mock model and other attributes
    mock_agent.model = MagicMock()
    mock_agent.trace_attributes = {"test_attr": "test_value"}
    mock_agent.callback_handler = MagicMock()

    return mock_agent


@pytest.fixture
def mock_graph_builder():
    """Create a mock GraphBuilder."""
    mock_builder = MagicMock()
    mock_builder.add_node = MagicMock()
    mock_builder.add_edge = MagicMock()
    mock_builder.set_entry_point = MagicMock()

    # Mock the built graph
    mock_graph = MagicMock()
    mock_builder.build.return_value = mock_graph

    # Mock execution result
    mock_execution_result = MagicMock()
    mock_execution_result.status.value = "completed"
    mock_execution_result.completed_nodes = 3
    mock_execution_result.failed_nodes = 0
    mock_execution_result.results = {
        "researcher": MagicMock(),
        "analyst": MagicMock(),
        "reporter": MagicMock(),
    }

    # Mock agent results for each node
    for node_id, node_result in mock_execution_result.results.items():
        mock_agent_result = MagicMock()
        mock_agent_result.__str__ = MagicMock(return_value=f"Result from {node_id}")
        node_result.get_agent_results.return_value = [mock_agent_result]

    mock_graph.execute.return_value = mock_execution_result

    return mock_builder


@pytest.fixture
def sample_topology():
    """Create a sample graph topology for testing."""
    return {
        "nodes": [
            {
                "id": "researcher",
                "role": "researcher",
                "system_prompt": "You research topics thoroughly.",
                "model_provider": "bedrock",
                "model_settings": {"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
            },
            {
                "id": "analyst",
                "role": "analyst",
                "system_prompt": "You analyze research data.",
                "model_provider": "bedrock",
                "model_settings": {"model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0"},
            },
            {
                "id": "reporter",
                "role": "reporter",
                "system_prompt": "You create comprehensive reports.",
                "tools": ["file_write", "editor"],
            },
        ],
        "edges": [
            {"from": "researcher", "to": "analyst"},
            {"from": "analyst", "to": "reporter"},
        ],
        "entry_points": ["researcher"],
    }


def test_graph_create_basic(mock_parent_agent, mock_graph_builder, sample_topology):
    """Test basic graph creation."""
    with (
        patch("strands_tools.graph.GraphBuilder", return_value=mock_graph_builder),
        patch("strands_tools.graph.create_agent_with_model") as mock_create_agent,
        patch("strands_tools.graph.Agent") as mock_agent_class,
    ):
        # Mock agent creation
        mock_agents = {
            "researcher": MagicMock(),
            "analyst": MagicMock(),
            "reporter": MagicMock(),
        }

        def agent_side_effect(**kwargs):
            return mock_agents.get(kwargs.get("system_prompt", "").split()[0].lower(), MagicMock())

        mock_create_agent.side_effect = agent_side_effect
        mock_agent_class.side_effect = agent_side_effect

        result = graph_module.graph(
            action="create",
            graph_id="test_pipeline",
            topology=sample_topology,
            agent=mock_parent_agent,
        )

        # Verify success
        assert result["status"] == "success"
        assert "Graph test_pipeline created successfully with 3 nodes" in result["content"][0]["text"]

        # Verify GraphBuilder was called correctly
        assert mock_graph_builder.add_node.call_count == 3
        assert mock_graph_builder.add_edge.call_count == 2
        assert mock_graph_builder.set_entry_point.call_count == 1
        mock_graph_builder.build.assert_called_once()


def test_graph_create_with_model_settings(mock_parent_agent, mock_graph_builder, sample_topology):
    """Test graph creation with custom model settings."""
    with (
        patch("strands_tools.graph.GraphBuilder", return_value=mock_graph_builder),
        patch("strands_tools.graph.create_agent_with_model") as mock_create_agent,
    ):
        mock_create_agent.return_value = MagicMock()

        result = graph_module.graph(
            action="create",
            graph_id="custom_model_pipeline",
            topology=sample_topology,
            model_provider="anthropic",
            model_settings={"model_id": "claude-3-5-sonnet-20241022"},
            tools=["calculator", "file_read"],
            agent=mock_parent_agent,
        )

        # Verify success
        assert result["status"] == "success"

        # Verify create_agent_with_model was called with correct parameters
        assert mock_create_agent.call_count >= 1

        # Check that at least one call included the model provider
        calls = mock_create_agent.call_args_list
        model_provider_calls = [call for call in calls if call.kwargs.get("model_provider") == "anthropic"]
        assert len(model_provider_calls) > 0


def test_graph_create_duplicate_id(mock_parent_agent, sample_topology):
    """Test creating graph with duplicate ID."""
    # First create a graph
    with (
        patch("strands_tools.graph.GraphBuilder") as mock_gb_class,
        patch("strands_tools.graph.create_agent_with_model"),
    ):
        mock_graph_builder = MagicMock()
        mock_gb_class.return_value = mock_graph_builder
        mock_graph_builder.build.return_value = MagicMock()

        # First creation should succeed
        result1 = graph_module.graph(
            action="create",
            graph_id="duplicate_test",
            topology=sample_topology,
            agent=mock_parent_agent,
        )
        assert result1["status"] == "success"

        # Second creation should fail
        result2 = graph_module.graph(
            action="create",
            graph_id="duplicate_test",
            topology=sample_topology,
            agent=mock_parent_agent,
        )
        assert result2["status"] == "error"
        assert "already exists" in result2["content"][0]["text"]


def test_graph_create_missing_parameters():
    """Test graph creation with missing required parameters."""
    # Missing topology
    result1 = graph_module.graph(action="create", graph_id="missing_topology")
    assert result1["status"] == "error"
    assert "topology are required" in result1["content"][0]["text"]

    # Missing graph_id
    result2 = graph_module.graph(action="create", topology={"nodes": [], "edges": []})
    assert result2["status"] == "error"
    assert "graph_id and topology are required" in result2["content"][0]["text"]


def test_graph_execute_basic(mock_parent_agent, mock_graph_builder, sample_topology):
    """Test basic graph execution."""
    with (
        patch("strands_tools.graph.GraphBuilder", return_value=mock_graph_builder),
        patch("strands_tools.graph.create_agent_with_model"),
    ):
        # First create the graph
        graph_module.graph(
            action="create",
            graph_id="exec_test",
            topology=sample_topology,
            agent=mock_parent_agent,
        )

        # Then execute it
        result = graph_module.graph(
            action="execute",
            graph_id="exec_test",
            task="Research AI in healthcare",
            agent=mock_parent_agent,
        )

        # Verify success
        assert result["status"] == "success"
        assert "Graph exec_test executed successfully" in result["content"][0]["text"]


def test_graph_execute_nonexistent(mock_parent_agent):
    """Test executing non-existent graph."""
    result = graph_module.graph(
        action="execute",
        graph_id="nonexistent_graph",
        task="Some task",
        agent=mock_parent_agent,
    )

    assert result["status"] == "error"
    assert "not found" in result["content"][0]["text"]


def test_graph_execute_missing_parameters(mock_parent_agent):
    """Test graph execution with missing parameters."""
    # Missing task
    result1 = graph_module.graph(action="execute", graph_id="some_graph", agent=mock_parent_agent)
    assert result1["status"] == "error"
    assert "task are required" in result1["content"][0]["text"]

    # Missing graph_id
    result2 = graph_module.graph(action="execute", task="Some task", agent=mock_parent_agent)
    assert result2["status"] == "error"
    assert "graph_id and task are required" in result2["content"][0]["text"]


def test_graph_status(mock_parent_agent, mock_graph_builder, sample_topology):
    """Test getting graph status."""
    with (
        patch("strands_tools.graph.GraphBuilder", return_value=mock_graph_builder),
        patch("strands_tools.graph.create_agent_with_model"),
    ):
        # Create graph first
        graph_module.graph(
            action="create",
            graph_id="status_test",
            topology=sample_topology,
            agent=mock_parent_agent,
        )

        # Get status
        result = graph_module.graph(action="status", graph_id="status_test", agent=mock_parent_agent)

        assert result["status"] == "success"
        assert "Graph status_test status retrieved" in result["content"][0]["text"]


def test_graph_status_nonexistent(mock_parent_agent):
    """Test getting status of non-existent graph."""
    result = graph_module.graph(action="status", graph_id="nonexistent_status", agent=mock_parent_agent)

    assert result["status"] == "error"
    assert "not found" in result["content"][0]["text"]


def test_graph_delete(mock_parent_agent, mock_graph_builder, sample_topology):
    """Test deleting a graph."""
    with (
        patch("strands_tools.graph.GraphBuilder", return_value=mock_graph_builder),
        patch("strands_tools.graph.create_agent_with_model"),
    ):
        # Create graph
        graph_module.graph(
            action="create",
            graph_id="delete_test",
            topology=sample_topology,
            agent=mock_parent_agent,
        )

        # Delete graph
        result = graph_module.graph(action="delete", graph_id="delete_test", agent=mock_parent_agent)

        assert result["status"] == "success"
        assert "Graph delete_test deleted successfully" in result["content"][0]["text"]

        # Verify graph is gone
        status_result = graph_module.graph(action="status", graph_id="delete_test", agent=mock_parent_agent)
        assert status_result["status"] == "error"


def test_graph_delete_nonexistent(mock_parent_agent):
    """Test deleting non-existent graph."""
    result = graph_module.graph(action="delete", graph_id="nonexistent_delete", agent=mock_parent_agent)

    assert result["status"] == "error"
    assert "not found" in result["content"][0]["text"]


def test_graph_invalid_action(mock_parent_agent):
    """Test graph with invalid action."""
    result = graph_module.graph(action="invalid_action", agent=mock_parent_agent)

    assert result["status"] == "error"
    assert "Unknown action" in result["content"][0]["text"]


def test_graph_create_with_tool_filtering(mock_parent_agent, mock_graph_builder):
    """Test graph creation with specific tool filtering."""
    topology_with_tools = {
        "nodes": [
            {
                "id": "writer",
                "role": "writer",
                "system_prompt": "You write content.",
                "tools": ["file_write", "editor"],
            }
        ],
        "edges": [],
        "entry_points": ["writer"],
    }

    with (
        patch("strands_tools.graph.GraphBuilder", return_value=mock_graph_builder),
        patch("strands_tools.graph.Agent") as mock_agent_class,
    ):
        mock_agent_class.return_value = MagicMock()

        result = graph_module.graph(
            action="create",
            graph_id="tool_filter_test",
            topology=topology_with_tools,
            agent=mock_parent_agent,
        )

        assert result["status"] == "success"

        # Verify Agent was called (at least once for nodes without custom models)
        assert mock_agent_class.call_count >= 1


def test_graph_execution_with_failure(mock_parent_agent, mock_graph_builder, sample_topology):
    """Test graph execution when some nodes fail."""
    # Mock execution result with failures
    mock_execution_result = MagicMock()
    mock_execution_result.status.value = "partial_failure"
    mock_execution_result.completed_nodes = 2
    mock_execution_result.failed_nodes = 1
    mock_execution_result.results = {"researcher": MagicMock(), "analyst": MagicMock()}

    for node_id, node_result in mock_execution_result.results.items():
        mock_agent_result = MagicMock()
        mock_agent_result.__str__ = MagicMock(return_value=f"Result from {node_id}")
        node_result.get_agent_results.return_value = [mock_agent_result]

    mock_graph_builder.build.return_value.execute.return_value = mock_execution_result

    with (
        patch("strands_tools.graph.GraphBuilder", return_value=mock_graph_builder),
        patch("strands_tools.graph.create_agent_with_model"),
    ):
        # Create and execute graph
        graph_module.graph(
            action="create",
            graph_id="failure_test",
            topology=sample_topology,
            agent=mock_parent_agent,
        )

        result = graph_module.graph(
            action="execute",
            graph_id="failure_test",
            task="Test task with failures",
            agent=mock_parent_agent,
        )

        assert result["status"] == "success"  # Tool call succeeds even if some nodes fail


def test_graph_create_exception_handling(mock_parent_agent, sample_topology):
    """Test graph creation exception handling."""
    with patch("strands_tools.graph.GraphBuilder") as mock_gb_class:
        # Make GraphBuilder construction fail
        mock_gb_class.side_effect = Exception("GraphBuilder creation failed")

        result = graph_module.graph(
            action="create",
            graph_id="exception_test",
            topology=sample_topology,
            agent=mock_parent_agent,
        )

        assert result["status"] == "error"
        assert "Error creating graph" in result["content"][0]["text"]


def test_graph_execute_exception_handling(mock_parent_agent, mock_graph_builder, sample_topology):
    """Test graph execution exception handling."""
    # Make execution fail by mocking the graph call
    mock_graph = mock_graph_builder.build.return_value
    mock_graph.side_effect = Exception("Execution failed")

    with (
        patch("strands_tools.graph.GraphBuilder", return_value=mock_graph_builder),
        patch("strands_tools.graph.create_agent_with_model"),
    ):
        # Create graph
        graph_module.graph(
            action="create",
            graph_id="exec_exception_test",
            topology=sample_topology,
            agent=mock_parent_agent,
        )

        # Execute with failure
        result = graph_module.graph(
            action="execute",
            graph_id="exec_exception_test",
            task="Test task",
            agent=mock_parent_agent,
        )

        assert result["status"] == "error"
        assert "Error executing graph" in result["content"][0]["text"]


def test_create_agent_with_model_function(mock_parent_agent):
    """Test the create_agent_with_model helper function."""
    with (
        patch("strands_tools.graph.create_model") as mock_create_model,
        patch("strands_tools.graph.Agent") as mock_agent_class,
    ):
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent

        result_agent = graph_module.create_agent_with_model(
            system_prompt="Test prompt",
            model_provider="bedrock",
            model_settings={"model_id": "test-model"},
            tools=["calculator", "file_read"],
            parent_agent=mock_parent_agent,
        )

        # Verify model was created
        mock_create_model.assert_called_once_with(provider="bedrock", config={"model_id": "test-model"})

        # Verify Agent was created with correct parameters
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args.kwargs
        assert call_kwargs["system_prompt"] == "Test prompt"
        assert call_kwargs["model"] == mock_model
        assert call_kwargs["trace_attributes"] == mock_parent_agent.trace_attributes
        assert call_kwargs["callback_handler"] == mock_parent_agent.callback_handler

        assert result_agent == mock_agent


def test_create_agent_with_model_no_parent():
    """Test create_agent_with_model without parent agent."""
    with (
        patch("strands_tools.graph.create_model") as mock_create_model,
        patch("strands_tools.graph.Agent") as mock_agent_class,
    ):
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent

        graph_module.create_agent_with_model(system_prompt="Test prompt", model_provider="anthropic", parent_agent=None)

        # Verify Agent was created without parent-specific attributes
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args.kwargs
        assert call_kwargs["system_prompt"] == "Test prompt"
        assert call_kwargs["model"] == mock_model
        assert len(call_kwargs) == 3  # system_prompt, model, tools


def test_graph_manager_singleton():
    """Test that GraphManager uses singleton pattern."""
    # Access the global manager multiple times
    manager1 = graph_module._manager
    manager2 = graph_module._manager

    # Should be the same instance
    assert manager1 is manager2


def test_graph_via_agent_interface(agent):
    """Test graph via the agent interface (integration test)."""
    sample_topology = {
        "nodes": [
            {
                "id": "simple_node",
                "role": "assistant",
                "system_prompt": "You are a helpful assistant.",
            }
        ],
        "edges": [],
        "entry_points": ["simple_node"],
    }

    with (
        patch("strands_tools.graph.GraphBuilder") as mock_gb_class,
        patch("strands_tools.graph.Agent"),
    ):
        mock_graph_builder = MagicMock()
        mock_gb_class.return_value = mock_graph_builder
        mock_graph_builder.build.return_value = MagicMock()

        try:
            # Test calling through agent interface
            result = agent.tool.graph(action="create", graph_id="integration_test", topology=sample_topology)
            # If we get here without an exception, consider the test passed
            assert result is not None
        except Exception as e:
            pytest.fail(f"Agent graph call raised an exception: {e}")


def test_graph_with_auto_entry_points(mock_parent_agent, mock_graph_builder):
    """Test graph creation with auto-detected entry points."""
    topology_no_entry = {
        "nodes": [
            {
                "id": "start",
                "role": "starter",
                "system_prompt": "You start the process.",
            },
            {"id": "middle", "role": "processor", "system_prompt": "You process data."},
        ],
        "edges": [{"from": "start", "to": "middle"}],
        # No explicit entry_points
    }

    with (
        patch("strands_tools.graph.GraphBuilder", return_value=mock_graph_builder),
        patch("strands_tools.graph.Agent"),
    ):
        result = graph_module.graph(
            action="create",
            graph_id="auto_entry_test",
            topology=topology_no_entry,
            agent=mock_parent_agent,
        )

        assert result["status"] == "success"


def test_graph_empty_topology(mock_parent_agent, mock_graph_builder):
    """Test graph creation with minimal topology."""
    minimal_topology = {
        "nodes": [{"id": "single", "role": "solo", "system_prompt": "You work alone."}],
        "edges": [],
        "entry_points": ["single"],
    }

    with (
        patch("strands_tools.graph.GraphBuilder", return_value=mock_graph_builder),
        patch("strands_tools.graph.Agent"),
    ):
        result = graph_module.graph(
            action="create",
            graph_id="minimal_test",
            topology=minimal_topology,
            agent=mock_parent_agent,
        )

        assert result["status"] == "success"
        assert "Graph minimal_test created successfully with 1 nodes" in result["content"][0]["text"]


def test_graph_complex_topology(mock_parent_agent, mock_graph_builder):
    """Test graph creation with complex topology (many nodes and edges)."""
    complex_topology = {
        "nodes": [
            {
                "id": f"node_{i}",
                "role": f"role_{i}",
                "system_prompt": f"You are node {i}.",
            }
            for i in range(5)
        ],
        "edges": [{"from": f"node_{i}", "to": f"node_{i + 1}"} for i in range(4)]
        + [
            {"from": "node_0", "to": "node_2"},  # Additional connection
            {"from": "node_1", "to": "node_4"},  # Skip connection
        ],
        "entry_points": ["node_0"],
    }

    with (
        patch("strands_tools.graph.GraphBuilder", return_value=mock_graph_builder),
        patch("strands_tools.graph.Agent"),
    ):
        result = graph_module.graph(
            action="create",
            graph_id="complex_test",
            topology=complex_topology,
            agent=mock_parent_agent,
        )

        assert result["status"] == "success"
        assert "Graph complex_test created successfully with 5 nodes" in result["content"][0]["text"]

        # Verify all nodes were added
        assert mock_graph_builder.add_node.call_count == 5
        # Verify all edges were added
        assert mock_graph_builder.add_edge.call_count == 6


def test_graph_with_mixed_model_configurations(mock_parent_agent, mock_graph_builder):
    """Test graph with nodes having different model configurations."""
    mixed_topology = {
        "nodes": [
            {
                "id": "bedrock_node",
                "role": "bedrock_worker",
                "system_prompt": "You use Bedrock.",
                "model_provider": "bedrock",
                "model_settings": {"model_id": "claude-v1"},
            },
            {
                "id": "anthropic_node",
                "role": "anthropic_worker",
                "system_prompt": "You use Anthropic.",
                "model_provider": "anthropic",
                "model_settings": {"model_id": "claude-3-5-sonnet"},
            },
            {
                "id": "default_node",
                "role": "default_worker",
                "system_prompt": "You use default settings.",
                # No model configuration - should inherit from parent
            },
        ],
        "edges": [
            {"from": "bedrock_node", "to": "anthropic_node"},
            {"from": "anthropic_node", "to": "default_node"},
        ],
        "entry_points": ["bedrock_node"],
    }

    with (
        patch("strands_tools.graph.GraphBuilder", return_value=mock_graph_builder),
        patch("strands_tools.graph.create_agent_with_model") as mock_create_agent,
        patch("strands_tools.graph.Agent") as mock_agent_class,
    ):
        mock_create_agent.return_value = MagicMock()
        mock_agent_class.return_value = MagicMock()

        result = graph_module.graph(
            action="create",
            graph_id="mixed_models_test",
            topology=mixed_topology,
            agent=mock_parent_agent,
        )

        assert result["status"] == "success"

        # Verify specialized agents were created for custom model nodes
        assert mock_create_agent.call_count >= 2  # bedrock_node and anthropic_node

        # Verify default agent was created for node without custom model
        assert mock_agent_class.call_count >= 1  # default_node
