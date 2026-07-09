"""
Tests for the swarm tool using the Agent interface.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent

from strands_tools import swarm as swarm_module


@pytest.fixture
def agent():
    """Create an agent with the swarm tool loaded."""
    return Agent(tools=[swarm_module])


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
        "editor": MagicMock(),
        "retrieve": MagicMock(),
        "generate_image": MagicMock(),
        "file_write": MagicMock(),
    }

    # Mock model and other attributes
    mock_agent.model = MagicMock()
    mock_agent.trace_attributes = {"test_attr": "test_value"}
    mock_agent.callback_handler = MagicMock()
    mock_agent.system_prompt = "You are a helpful assistant."
    mock_agent.model_provider = "bedrock"
    mock_agent.model_settings = {"model_id": "default-model"}

    return mock_agent


@pytest.fixture
def mock_swarm_result():
    """Create a mock result from Swarm execution."""
    mock_result = MagicMock()
    mock_result.status = "completed"
    mock_result.execution_time = 2500
    mock_result.execution_count = 3

    # Mock node history
    mock_node1 = MagicMock()
    mock_node1.node_id = "researcher"
    mock_node2 = MagicMock()
    mock_node2.node_id = "analyst"
    mock_node3 = MagicMock()
    mock_node3.node_id = "writer"
    mock_result.node_history = [mock_node1, mock_node2, mock_node3]

    # Mock results from individual agents
    mock_agent_result1 = MagicMock()
    mock_agent_result1.result.content = [MagicMock(text="Research findings: The market shows strong growth potential.")]
    mock_agent_result2 = MagicMock()
    mock_agent_result2.result.content = [MagicMock(text="Analysis complete: Data indicates 25% market opportunity.")]
    mock_agent_result3 = MagicMock()
    mock_agent_result3.result.content = [MagicMock(text="Final report: Comprehensive strategy document created.")]

    mock_result.results = {
        "researcher": mock_agent_result1,
        "analyst": mock_agent_result2,
        "writer": mock_agent_result3,
    }

    # Mock usage metrics properly to avoid formatting issues
    mock_result.accumulated_usage = {
        "inputTokens": 150,
        "outputTokens": 300,
        "totalTokens": 450,
    }

    return mock_result


@pytest.fixture
def sample_agents():
    """Sample agent specifications for testing."""
    return [
        {
            "name": "researcher",
            "system_prompt": "You are a research specialist. Focus on gathering and analyzing information.",
            "tools": ["retrieve", "file_read"],
            "model_provider": "bedrock",
            "model_settings": {"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
        },
        {
            "name": "analyst",
            "system_prompt": "You are a data analyst. Focus on interpreting research and providing insights.",
            "tools": ["calculator", "file_write"],
            "model_provider": "anthropic",
            "model_settings": {"model_id": "claude-sonnet-4-20250514"},
        },
        {
            "name": "writer",
            "system_prompt": "You are a content writer. Focus on creating clear, compelling documents.",
            "tools": ["file_write"],
            "model_provider": "openai",
            "model_settings": {"model_id": "o4-mini"},
        },
    ]


class TestSwarmTool:
    """Tests for the swarm tool function."""

    def test_swarm_create_success(self, mock_parent_agent, mock_swarm_result, sample_agents):
        """Test successful swarm creation and execution."""
        with (
            patch("strands_tools.swarm.Swarm") as MockSwarm,
            patch("strands_tools.swarm._create_custom_agents") as mock_create_agents,
        ):
            # Mock agent creation
            mock_agents = [MagicMock(name=f"agent_{i}") for i in range(3)]
            mock_create_agents.return_value = mock_agents

            # Mock Swarm creation and execution
            mock_swarm_instance = MagicMock()
            mock_swarm_instance.return_value = mock_swarm_result
            MockSwarm.return_value = mock_swarm_instance

            result = swarm_module.swarm(
                task="Develop a comprehensive product launch strategy",
                agents=sample_agents,
                agent=mock_parent_agent,
            )

            # Verify result structure
            assert result["status"] == "success"
            assert len(result["content"]) == 1
            content = result["content"][0]["text"]

            # Verify key information is included
            assert "Custom Agent Team Execution Complete" in content
            assert "**Status:** completed" in content
            assert "**Execution Time:** 2500ms" in content
            assert "**Team Size:** 3 agents" in content
            assert "**Iterations:** 3" in content
            assert "researcher → analyst → writer" in content

            # Verify agent creation was called
            mock_create_agents.assert_called_once_with(agent_specs=sample_agents, parent_agent=mock_parent_agent)

            # Verify Swarm was configured correctly
            MockSwarm.assert_called_once()
            swarm_call_kwargs = MockSwarm.call_args.kwargs
            assert swarm_call_kwargs["nodes"] == mock_agents
            assert swarm_call_kwargs["max_handoffs"] == 20
            assert swarm_call_kwargs["max_iterations"] == 20
            assert swarm_call_kwargs["execution_timeout"] == 900.0
            assert swarm_call_kwargs["node_timeout"] == 300.0

    def test_swarm_with_custom_configuration(self, mock_parent_agent, mock_swarm_result, sample_agents):
        """Test swarm with custom timeout and iteration settings."""
        with (
            patch("strands_tools.swarm.Swarm") as MockSwarm,
            patch("strands_tools.swarm._create_custom_agents") as mock_create_agents,
        ):
            mock_create_agents.return_value = [MagicMock()]
            mock_swarm_instance = MagicMock()
            mock_swarm_instance.return_value = mock_swarm_result
            MockSwarm.return_value = mock_swarm_instance

            result = swarm_module.swarm(
                task="Complex analysis task",
                agents=sample_agents,
                max_handoffs=10,
                max_iterations=15,
                execution_timeout=600.0,
                node_timeout=120.0,
                agent=mock_parent_agent,
            )

            assert result["status"] == "success"

            # Verify custom configuration was passed to Swarm
            swarm_call_kwargs = MockSwarm.call_args.kwargs
            assert swarm_call_kwargs["max_handoffs"] == 10
            assert swarm_call_kwargs["max_iterations"] == 15
            assert swarm_call_kwargs["execution_timeout"] == 600.0
            assert swarm_call_kwargs["node_timeout"] == 120.0

    def test_swarm_empty_agents_error(self, mock_parent_agent):
        """Test swarm creation with empty agents list raises error."""
        result = swarm_module.swarm(task="Test task", agents=[], agent=mock_parent_agent)

        assert result["status"] == "error"
        assert "At least one agent specification is required" in result["content"][0]["text"]

    def test_swarm_large_team_warning(self, mock_parent_agent, mock_swarm_result):
        """Test swarm creation with large team logs warning."""
        large_agents = [{"system_prompt": f"You are agent {i}."} for i in range(12)]

        with (
            patch("strands_tools.swarm.Swarm") as MockSwarm,
            patch("strands_tools.swarm._create_custom_agents") as mock_create_agents,
            patch("strands_tools.swarm.logger") as mock_logger,
        ):
            mock_create_agents.return_value = [MagicMock() for _ in range(12)]
            mock_swarm_instance = MagicMock()
            mock_swarm_instance.return_value = mock_swarm_result
            MockSwarm.return_value = mock_swarm_instance

            result = swarm_module.swarm(task="Large team task", agents=large_agents, agent=mock_parent_agent)

            assert result["status"] == "success"
            mock_logger.warning.assert_called_with("Large team size (12 agents) may impact performance")

    def test_swarm_execution_failure(self, mock_parent_agent, sample_agents):
        """Test swarm execution failure handling."""
        with (
            patch("strands_tools.swarm.Swarm") as MockSwarm,
            patch("strands_tools.swarm._create_custom_agents") as mock_create_agents,
        ):
            mock_create_agents.return_value = [MagicMock()]

            # Make Swarm execution raise an exception
            mock_swarm_instance = MagicMock()
            mock_swarm_instance.side_effect = Exception("Swarm execution failed")
            MockSwarm.return_value = mock_swarm_instance

            result = swarm_module.swarm(task="Failing task", agents=sample_agents, agent=mock_parent_agent)

            assert result["status"] == "error"
            assert "Custom swarm execution failed: Swarm execution failed" in result["content"][0]["text"]

    def test_swarm_agent_creation_failure(self, mock_parent_agent, sample_agents):
        """Test swarm when agent creation fails."""
        with patch("strands_tools.swarm._create_custom_agents") as mock_create_agents:
            # Make agent creation fail
            mock_create_agents.side_effect = ValueError("Invalid agent specification")

            result = swarm_module.swarm(task="Test task", agents=sample_agents, agent=mock_parent_agent)

            assert result["status"] == "error"
            assert "Invalid agent specification" in result["content"][0]["text"]

    def test_swarm_without_parent_agent(self, mock_swarm_result, sample_agents):
        """Test swarm creation without parent agent."""
        with (
            patch("strands_tools.swarm.Swarm") as MockSwarm,
            patch("strands_tools.swarm._create_custom_agents") as mock_create_agents,
        ):
            mock_create_agents.return_value = [MagicMock()]
            mock_swarm_instance = MagicMock()
            mock_swarm_instance.return_value = mock_swarm_result
            MockSwarm.return_value = mock_swarm_instance

            result = swarm_module.swarm(task="Test task", agents=sample_agents, agent=None)

            assert result["status"] == "success"

            # Verify agent creation was called with None parent
            mock_create_agents.assert_called_once_with(agent_specs=sample_agents, parent_agent=None)

    def test_swarm_rich_console_creation(self, mock_parent_agent, mock_swarm_result, sample_agents):
        """Test that rich console is properly created for formatting."""
        with (
            patch("strands_tools.swarm.Swarm") as MockSwarm,
            patch("strands_tools.swarm._create_custom_agents") as mock_create_agents,
            patch("strands_tools.swarm.console_util.create") as mock_console_create,
        ):
            mock_create_agents.return_value = [MagicMock()]
            mock_swarm_instance = MagicMock()
            mock_swarm_instance.return_value = mock_swarm_result
            MockSwarm.return_value = mock_swarm_instance

            mock_console = MagicMock()
            mock_console_create.return_value = mock_console

            result = swarm_module.swarm(task="Test task", agents=sample_agents, agent=mock_parent_agent)

            assert result["status"] == "success"
            mock_console_create.assert_called_once()


class TestCreateCustomAgents:
    """Tests for the _create_custom_agents helper function."""

    def test_create_agents_basic(self, mock_parent_agent):
        """Test basic agent creation from specifications."""
        agent_specs = [
            {
                "name": "test_agent",
                "system_prompt": "You are a helpful assistant.",
                "tools": ["calculator"],
                "model_provider": "bedrock",
                "model_settings": {"model_id": "claude-3-sonnet"},
            }
        ]

        with patch("strands_tools.swarm.Agent") as MockAgent:
            mock_agent = MagicMock()
            MockAgent.return_value = mock_agent

            agents = swarm_module._create_custom_agents(agent_specs=agent_specs, parent_agent=mock_parent_agent)

            assert len(agents) == 1
            assert agents[0] == mock_agent

            # Verify Agent was called with correct parameters
            MockAgent.assert_called_once()
            call_kwargs = MockAgent.call_args.kwargs
            assert call_kwargs["name"] == "test_agent"
            assert call_kwargs["system_prompt"] == "You are a helpful assistant."
            assert call_kwargs["callback_handler"] == mock_parent_agent.callback_handler
            assert call_kwargs["trace_attributes"] == mock_parent_agent.trace_attributes

    def test_create_agents_auto_naming(self, mock_parent_agent):
        """Test automatic agent naming when names are not provided."""
        agent_specs = [
            {"system_prompt": "You are agent 1."},
            {"system_prompt": "You are agent 2."},
        ]

        with patch("strands_tools.swarm.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()

            agents = swarm_module._create_custom_agents(agent_specs=agent_specs, parent_agent=mock_parent_agent)

            assert len(agents) == 2

            # Verify auto-generated names
            call_names = [call.kwargs["name"] for call in MockAgent.call_args_list]
            assert "agent_1" in call_names
            assert "agent_2" in call_names

    def test_create_agents_duplicate_names(self, mock_parent_agent):
        """Test handling of duplicate agent names."""
        agent_specs = [
            {"name": "duplicate", "system_prompt": "First agent."},
            {"name": "duplicate", "system_prompt": "Second agent."},
            {"name": "duplicate", "system_prompt": "Third agent."},
        ]

        with patch("strands_tools.swarm.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()

            agents = swarm_module._create_custom_agents(agent_specs=agent_specs, parent_agent=mock_parent_agent)

            assert len(agents) == 3

            # Verify unique names were generated
            call_names = [call.kwargs["name"] for call in MockAgent.call_args_list]
            assert "duplicate" in call_names
            assert "duplicate_1" in call_names
            assert "duplicate_2" in call_names

    def test_create_agents_tool_filtering(self, mock_parent_agent):
        """Test that tools are filtered based on parent agent registry."""
        agent_specs = [
            {
                "name": "filtered_agent",
                "system_prompt": "Test agent.",
                "tools": ["calculator", "nonexistent_tool", "file_read"],
            }
        ]

        with (
            patch("strands_tools.swarm.Agent") as MockAgent,
            patch("strands_tools.swarm.logger") as mock_logger,
        ):
            MockAgent.return_value = MagicMock()

            agents = swarm_module._create_custom_agents(agent_specs=agent_specs, parent_agent=mock_parent_agent)

            assert len(agents) == 1

            # Check that warning was logged for missing tool
            mock_logger.warning.assert_called_with("Agent 'filtered_agent' missing tools: {'nonexistent_tool'}")

            # Verify only valid tools were passed
            call_kwargs = MockAgent.call_args.kwargs
            tools = call_kwargs["tools"]
            assert len(tools) == 2  # Only calculator and file_read

    def test_create_agents_inherit_parent_prompt(self, mock_parent_agent):
        """Test inheriting parent system prompt when requested."""
        agent_specs = [
            {
                "name": "inheriting_agent",
                "system_prompt": "You are a specialist.",
                "inherit_parent_prompt": True,
            }
        ]

        with patch("strands_tools.swarm.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()

            swarm_module._create_custom_agents(agent_specs=agent_specs, parent_agent=mock_parent_agent)

            # Verify combined system prompt
            call_kwargs = MockAgent.call_args.kwargs
            system_prompt = call_kwargs["system_prompt"]
            assert "You are a specialist." in system_prompt
            assert "You are a helpful assistant." in system_prompt

    def test_create_agents_model_configuration(self, mock_parent_agent):
        """Test agent model configuration inheritance and override."""
        agent_specs = [
            {
                "name": "custom_model_agent",
                "system_prompt": "Custom model agent.",
                "model_provider": "anthropic",
                "model_settings": {"model_id": "claude-3-haiku"},
            },
            {
                "name": "inherit_model_agent",
                "system_prompt": "Inherit model agent.",
                # No model configuration - should inherit from parent
            },
        ]

        with patch("strands_tools.swarm.Agent") as MockAgent:
            mock_agents = [MagicMock(), MagicMock()]
            MockAgent.side_effect = mock_agents

            agents = swarm_module._create_custom_agents(agent_specs=agent_specs, parent_agent=mock_parent_agent)

            assert len(agents) == 2

            # Check model configuration assignment
            assert mock_agents[0].model_provider == "anthropic"
            assert mock_agents[0].model_settings == {"model_id": "claude-3-haiku"}

            assert mock_agents[1].model_provider == mock_parent_agent.model_provider
            assert mock_agents[1].model_settings == mock_parent_agent.model_settings

    def test_create_agents_invalid_spec_type(self, mock_parent_agent):
        """Test error handling for invalid agent specification type."""
        agent_specs = [
            "invalid_spec_string",  # Should be dict
            {"valid_spec": True},
        ]

        with pytest.raises(ValueError, match="Agent specification 0 must be a dictionary"):
            swarm_module._create_custom_agents(agent_specs=agent_specs, parent_agent=mock_parent_agent)

    def test_create_agents_empty_specs(self, mock_parent_agent):
        """Test error handling for empty agent specifications."""
        with pytest.raises(ValueError, match="At least one agent specification is required"):
            swarm_module._create_custom_agents(agent_specs=[], parent_agent=mock_parent_agent)

    def test_create_agents_without_parent(self):
        """Test agent creation without parent agent."""
        agent_specs = [{"name": "standalone_agent", "system_prompt": "You are a standalone agent."}]

        with patch("strands_tools.swarm.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()

            agents = swarm_module._create_custom_agents(agent_specs=agent_specs, parent_agent=None)

            assert len(agents) == 1

            # Verify agent was created with defaults
            call_kwargs = MockAgent.call_args.kwargs
            assert call_kwargs["name"] == "standalone_agent"
            assert call_kwargs["callback_handler"] is None
            assert call_kwargs["trace_attributes"] is None

    def test_create_agents_fallback_system_prompt(self, mock_parent_agent):
        """Test fallback system prompt when none is provided."""
        agent_specs = [{"name": "no_prompt_agent"}]  # No system_prompt provided

        with patch("strands_tools.swarm.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()

            swarm_module._create_custom_agents(agent_specs=agent_specs, parent_agent=mock_parent_agent)

            # Verify fallback system prompt was used
            call_kwargs = MockAgent.call_args.kwargs
            system_prompt = call_kwargs["system_prompt"]
            assert "helpful AI assistant" in system_prompt
            assert "collaborative problem solving" in system_prompt


class TestCreateRichStatusPanel:
    """Tests for the create_rich_status_panel helper function."""

    def test_create_status_panel_complete(self, mock_swarm_result):
        """Test creating status panel with complete swarm result."""
        with patch("strands_tools.swarm.console_util.create") as mock_create_console:
            mock_console = MagicMock()
            mock_create_console.return_value = mock_console

            # Mock capture context manager
            capture_mock = MagicMock()
            capture_mock.get.return_value = "Formatted panel output"
            mock_console.capture.return_value.__enter__.return_value = capture_mock

            result = swarm_module.create_rich_status_panel(mock_console, mock_swarm_result)

            assert result == "Formatted panel output"
            mock_console.capture.assert_called_once()
            mock_console.print.assert_called_once()

    def test_create_status_panel_minimal_result(self):
        """Test creating status panel with minimal result data."""
        minimal_result = MagicMock()
        minimal_result.status = "completed"
        minimal_result.execution_time = 1000
        minimal_result.execution_count = 1
        # Mock accumulated_usage as None to avoid the formatting issue
        minimal_result.accumulated_usage = None

        with patch("strands_tools.swarm.console_util.create") as mock_create_console:
            mock_console = MagicMock()
            mock_create_console.return_value = mock_console

            capture_mock = MagicMock()
            capture_mock.get.return_value = "Minimal panel output"
            mock_console.capture.return_value.__enter__.return_value = capture_mock

            result = swarm_module.create_rich_status_panel(mock_console, minimal_result)

            assert result == "Minimal panel output"


class TestIntegration:
    """Integration tests for the swarm tool."""

    def test_swarm_via_agent_interface(self, agent):
        """Test swarm via the agent interface (integration test)."""
        simple_agents = [
            {
                "system_prompt": "You are a helpful assistant.",
            }
        ]

        with (
            patch("strands_tools.swarm.Swarm") as MockSwarm,
            patch("strands_tools.swarm._create_custom_agents") as mock_create_agents,
        ):
            mock_create_agents.return_value = [MagicMock()]
            mock_swarm_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.status = "completed"
            mock_result.execution_time = 1000
            mock_result.execution_count = 1
            mock_result.node_history = []
            mock_result.results = {}
            mock_swarm_instance.return_value = mock_result
            MockSwarm.return_value = mock_swarm_instance

            # Test calling through agent interface
            try:
                result = agent.tool.swarm(task="Simple integration test", agents=simple_agents)
                # If we get here without an exception, consider the test passed
                assert result is not None
                assert isinstance(result, dict)
            except Exception as e:
                pytest.fail(f"Agent swarm call raised an exception: {e}")

    def test_swarm_logging_behavior(self, mock_parent_agent, sample_agents):
        """Test that swarm properly logs execution details."""
        with (
            patch("strands_tools.swarm.Swarm") as MockSwarm,
            patch("strands_tools.swarm._create_custom_agents") as mock_create_agents,
            patch("strands_tools.swarm.logger") as mock_logger,
        ):
            mock_create_agents.return_value = [MagicMock(), MagicMock(), MagicMock()]
            mock_swarm_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.status = "completed"
            mock_result.execution_time = 1500
            mock_result.execution_count = 2
            mock_result.node_history = []
            mock_result.results = {}
            # Properly mock accumulated_usage to avoid formatting issues
            mock_result.accumulated_usage = None
            mock_swarm_instance.return_value = mock_result
            MockSwarm.return_value = mock_swarm_instance

            # Set logger to appropriate level
            mock_logger.level = logging.INFO

            result = swarm_module.swarm(task="Logging test task", agents=sample_agents, agent=mock_parent_agent)

            assert result["status"] == "success"

            # Verify appropriate logging calls were made
            mock_logger.info.assert_any_call("Creating custom swarm with 3 agents")
            mock_logger.info.assert_any_call("Starting swarm execution with task: Logging test task...")

    def test_swarm_error_traceback_logging(self, mock_parent_agent, sample_agents):
        """Test that errors are properly logged with tracebacks."""
        with (
            patch("strands_tools.swarm.Swarm") as MockSwarm,
            patch("strands_tools.swarm._create_custom_agents") as mock_create_agents,
            patch("strands_tools.swarm.logger") as mock_logger,
            patch("strands_tools.swarm.traceback") as mock_traceback,
        ):
            mock_create_agents.return_value = [MagicMock()]
            MockSwarm.side_effect = RuntimeError("Test error")
            mock_traceback.format_exc.return_value = "Test traceback"

            result = swarm_module.swarm(task="Error test task", agents=sample_agents, agent=mock_parent_agent)

            assert result["status"] == "error"

            # Verify error logging with traceback
            mock_logger.error.assert_called_with("Custom swarm execution failed: Test error\nTest traceback")

    def test_create_agents_tool_objects_retrieval(self, mock_parent_agent):
        """Test that actual tool objects are retrieved from parent registry, not just names."""
        # Create mock tool objects
        mock_calculator_tool = MagicMock()
        mock_file_read_tool = MagicMock()

        # Update the parent agent's registry to return actual tool objects
        mock_parent_agent.tool_registry.registry = {
            "calculator": mock_calculator_tool,
            "file_read": mock_file_read_tool,
            "editor": MagicMock(),
        }

        agent_specs = [
            {
                "name": "test_agent",
                "system_prompt": "Test agent.",
                "tools": ["calculator", "file_read"],
            }
        ]

        with patch("strands_tools.swarm.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance

            agents = swarm_module._create_custom_agents(agent_specs=agent_specs, parent_agent=mock_parent_agent)

            assert len(agents) == 1

            # Verify Agent was called with actual tool objects, not tool names
            call_kwargs = MockAgent.call_args.kwargs
            tools_passed = call_kwargs["tools"]

            # Should be actual tool objects, not strings
            assert len(tools_passed) == 2
            assert mock_calculator_tool in tools_passed
            assert mock_file_read_tool in tools_passed

            # Verify these are the actual mock objects, not strings
            for tool in tools_passed:
                assert not isinstance(tool, str), "Tool should be an object, not a string"

            # Verify the specific objects we expect are present
            assert tools_passed[0] == mock_calculator_tool
            assert tools_passed[1] == mock_file_read_tool
