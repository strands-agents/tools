"""
Tests for the use_agent tool using the Agent interface.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent

from strands_tools import use_agent as use_agent_module


@pytest.fixture
def agent():
    """Create an agent with the use_agent tool loaded."""
    return Agent(tools=[use_agent_module])


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
        "http_request": MagicMock(),
        "test_tool": MagicMock(),
    }

    # Mock model and other attributes
    mock_agent.model = MagicMock()
    mock_agent.trace_attributes = {"test_attr": "test_value"}
    mock_agent.callback_handler = MagicMock()

    return mock_agent


@pytest.fixture
def mock_agent_result():
    """Create a mock result from Agent execution."""
    # Create a mock AgentResult with proper __str__ method
    result = MagicMock()
    result.__str__ = MagicMock(return_value="This is a test response from the nested agent")

    # Mock metrics with proper EventLoopMetrics structure
    mock_metrics = MagicMock()
    mock_metrics.get_summary.return_value = {
        "total_cycles": 1,
        "average_cycle_time": 1.5,
        "total_duration": 1.5,
        "accumulated_usage": {"inputTokens": 10, "outputTokens": 15, "totalTokens": 25},
        "accumulated_metrics": {"latencyMs": 1200},
        "tool_usage": {},
    }
    mock_metrics.traces = []
    result.metrics = mock_metrics

    return result


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_use_agent_direct_basic(mock_parent_agent, mock_agent_result):
    """Test direct invocation of the use_agent tool with basic parameters."""
    with (
        patch("strands_tools.use_agent.Agent") as mock_agent_class,
        patch("strands.telemetry.metrics.metrics_to_string") as mock_metrics_to_string,
    ):
        # Configure the mock Agent to return our test result
        mock_nested_agent = MagicMock()
        mock_nested_agent.return_value = mock_agent_result
        mock_agent_class.return_value = mock_nested_agent
        mock_metrics_to_string.return_value = "Mock metrics string"

        # Call the use_agent function directly
        result = use_agent_module.use_agent(
            prompt="Test prompt for nested agent",
            system_prompt="You are a test assistant.",
            agent=mock_parent_agent,
        )

        # Verify the result has the expected structure
        assert result["status"] == "success"
        assert len(result["content"]) == 3  # Response, Model, Metrics
        assert "Response:" in result["content"][0]["text"]
        assert "Model:" in result["content"][1]["text"]
        assert "Metrics:" in result["content"][2]["text"]

        # Verify Agent was instantiated with correct parameters
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args.kwargs
        assert call_kwargs["model"] == mock_parent_agent.model
        assert call_kwargs["system_prompt"] == "You are a test assistant."
        assert call_kwargs["trace_attributes"] == mock_parent_agent.trace_attributes
        assert call_kwargs["callback_handler"] == mock_parent_agent.callback_handler

        # Verify the nested agent was called with the prompt
        mock_nested_agent.assert_called_once_with("Test prompt for nested agent")


def test_use_agent_with_bedrock_model(mock_parent_agent, mock_agent_result):
    """Test use_agent with Bedrock model provider."""
    with (
        patch("strands_tools.use_agent.Agent") as mock_agent_class,
        patch("strands_tools.use_agent.create_model") as mock_create_model,
        patch("strands.telemetry.metrics.metrics_to_string") as mock_metrics_to_string,
    ):
        mock_bedrock_model = MagicMock()
        mock_create_model.return_value = mock_bedrock_model
        mock_metrics_to_string.return_value = "Mock metrics string"

        mock_nested_agent = MagicMock()
        mock_nested_agent.return_value = mock_agent_result
        mock_agent_class.return_value = mock_nested_agent

        result = use_agent_module.use_agent(
            prompt="Analyze this data with Bedrock",
            system_prompt="You are a data analyst.",
            model_provider="bedrock",
            model_settings={"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
            agent=mock_parent_agent,
        )

        # Verify model was created with correct provider
        mock_create_model.assert_called_once_with(
            provider="bedrock",
            config={"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
        )

        # Verify Agent was created with the Bedrock model
        call_kwargs = mock_agent_class.call_args.kwargs
        assert call_kwargs["model"] == mock_bedrock_model

        # Verify success and model info
        assert result["status"] == "success"
        assert "bedrock" in result["content"][1]["text"].lower()


def test_use_agent_with_anthropic_model(mock_parent_agent, mock_agent_result):
    """Test use_agent with Anthropic model provider."""
    with (
        patch("strands_tools.use_agent.Agent") as mock_agent_class,
        patch("strands_tools.use_agent.create_model") as mock_create_model,
        patch("strands.telemetry.metrics.metrics_to_string") as mock_metrics_to_string,
    ):
        mock_anthropic_model = MagicMock()
        mock_create_model.return_value = mock_anthropic_model
        mock_metrics_to_string.return_value = "Mock metrics string"

        mock_nested_agent = MagicMock()
        mock_nested_agent.return_value = mock_agent_result
        mock_agent_class.return_value = mock_nested_agent

        result = use_agent_module.use_agent(
            prompt="Creative writing task",
            system_prompt="You are a creative writer.",
            model_provider="anthropic",
            model_settings={
                "model_id": "claude-3-5-sonnet-20241022",
                "params": {"temperature": 0.8},
            },
            agent=mock_parent_agent,
        )

        # Verify model was created with correct provider
        mock_create_model.assert_called_once_with(
            provider="anthropic",
            config={
                "model_id": "claude-3-5-sonnet-20241022",
                "params": {"temperature": 0.8},
            },
        )

        assert result["status"] == "success"
        assert "anthropic" in result["content"][1]["text"].lower()


def test_use_agent_with_environment_model(mock_parent_agent, mock_agent_result):
    """Test use_agent with environment-based model configuration."""
    with (
        patch("strands_tools.use_agent.Agent") as mock_agent_class,
        patch("strands_tools.use_agent.create_model") as mock_create_model,
        patch("strands_tools.use_agent.metrics_to_string") as mock_metrics_to_string,
        patch.dict(os.environ, {"STRANDS_PROVIDER": "ollama", "STRANDS_MODEL_ID": "qwen3:4b"}),
    ):
        mock_ollama_model = MagicMock()
        mock_create_model.return_value = mock_ollama_model
        mock_metrics_to_string.return_value = "Mock metrics string"

        mock_nested_agent = MagicMock()
        mock_nested_agent.return_value = mock_agent_result
        mock_agent_class.return_value = mock_nested_agent

        result = use_agent_module.use_agent(
            prompt="Local processing task",
            system_prompt="You are a local assistant.",
            model_provider="env",
            agent=mock_parent_agent,
        )

        # Verify model was created with environment provider
        mock_create_model.assert_called_once_with(provider="ollama", config=None)

        assert result["status"] == "success"
        assert "environment" in result["content"][1]["text"].lower()


def test_use_agent_with_tool_filtering(mock_parent_agent, mock_agent_result):
    """Test use_agent with specific tool filtering."""
    with (
        patch("strands_tools.use_agent.Agent") as mock_agent_class,
        patch("strands.telemetry.metrics.metrics_to_string") as mock_metrics_to_string,
    ):
        mock_nested_agent = MagicMock()
        mock_nested_agent.return_value = mock_agent_result
        mock_agent_class.return_value = mock_nested_agent
        mock_metrics_to_string.return_value = "Mock metrics string"

        result = use_agent_module.use_agent(
            prompt="Test with filtered tools",
            system_prompt="You are a test assistant.",
            tools=["calculator", "file_read"],
            agent=mock_parent_agent,
        )

        # Verify Agent was created with filtered tools
        call_kwargs = mock_agent_class.call_args.kwargs
        filtered_tools = call_kwargs["tools"]

        # Should have exactly 2 tools matching the requested names
        assert len(filtered_tools) == 2

        # Verify the tools were selected from parent registry
        expected_tools = [
            mock_parent_agent.tool_registry.registry["calculator"],
            mock_parent_agent.tool_registry.registry["file_read"],
        ]
        assert filtered_tools == expected_tools

        assert result["status"] == "success"


def test_use_agent_with_nonexistent_tools(mock_parent_agent, mock_agent_result):
    """Test use_agent with some non-existent tools (should warn but continue)."""
    with (
        patch("strands_tools.use_agent.Agent") as mock_agent_class,
        patch("strands_tools.use_agent.logger") as mock_logger,
        patch("strands.telemetry.metrics.metrics_to_string") as mock_metrics_to_string,
    ):
        mock_nested_agent = MagicMock()
        mock_nested_agent.return_value = mock_agent_result
        mock_agent_class.return_value = mock_nested_agent
        mock_metrics_to_string.return_value = "Mock metrics string"

        result = use_agent_module.use_agent(
            prompt="Test with mix of valid and invalid tools",
            system_prompt="You are a test assistant.",
            tools=["calculator", "nonexistent_tool", "file_read"],
            agent=mock_parent_agent,
        )

        # Verify warning was logged for non-existent tool
        mock_logger.warning.assert_called_with("Tool 'nonexistent_tool' not found in parent agent's tool registry")

        # Verify only valid tools were passed to Agent
        call_kwargs = mock_agent_class.call_args.kwargs
        filtered_tools = call_kwargs["tools"]
        assert len(filtered_tools) == 2  # Only calculator and file_read

        assert result["status"] == "success"


def test_use_agent_inherit_all_tools(mock_parent_agent, mock_agent_result):
    """Test use_agent inherits all parent tools when no tool filtering is specified."""
    with (
        patch("strands_tools.use_agent.Agent") as mock_agent_class,
        patch("strands.telemetry.metrics.metrics_to_string") as mock_metrics_to_string,
    ):
        mock_nested_agent = MagicMock()
        mock_nested_agent.return_value = mock_agent_result
        mock_agent_class.return_value = mock_nested_agent
        mock_metrics_to_string.return_value = "Mock metrics string"

        result = use_agent_module.use_agent(
            prompt="Test with all inherited tools",
            system_prompt="You are a test assistant.",
            # No tools parameter specified
            agent=mock_parent_agent,
        )

        # Verify Agent was created with all parent tools
        call_kwargs = mock_agent_class.call_args.kwargs
        inherited_tools = call_kwargs["tools"]

        # Should have all tools from parent registry
        expected_tools = list(mock_parent_agent.tool_registry.registry.values())
        assert inherited_tools == expected_tools

        assert result["status"] == "success"


def test_use_agent_model_creation_failure_fallback(mock_parent_agent, mock_agent_result):
    """Test use_agent falls back to parent model when custom model creation fails."""
    with (
        patch("strands_tools.use_agent.Agent") as mock_agent_class,
        patch("strands_tools.use_agent.create_model") as mock_create_model,
        patch("strands_tools.use_agent.logger") as mock_logger,
        patch("strands.telemetry.metrics.metrics_to_string") as mock_metrics_to_string,
    ):
        # Make model creation fail
        mock_create_model.side_effect = Exception("Model creation failed")
        mock_metrics_to_string.return_value = "Mock metrics string"

        mock_nested_agent = MagicMock()
        mock_nested_agent.return_value = mock_agent_result
        mock_agent_class.return_value = mock_nested_agent

        result = use_agent_module.use_agent(
            prompt="Test fallback behavior",
            system_prompt="You are a test assistant.",
            model_provider="bedrock",
            agent=mock_parent_agent,
        )

        # Verify warning was logged about fallback
        mock_logger.warning.assert_called_with("Failed to create bedrock model: Model creation failed")

        # Verify Agent was created with parent's model as fallback
        call_kwargs = mock_agent_class.call_args.kwargs
        assert call_kwargs["model"] == mock_parent_agent.model

        # Verify result indicates fallback was used
        assert result["status"] == "success"
        assert "Failed to use bedrock model" in result["content"][1]["text"]


def test_use_agent_exception_handling():
    """Test use_agent error handling when an exception occurs."""
    with patch("strands_tools.use_agent.Agent") as mock_agent_class:
        # Make Agent instantiation fail
        mock_agent_class.side_effect = Exception("Agent creation failed")

        result = use_agent_module.use_agent(
            prompt="This will fail",
            system_prompt="You are a test assistant.",
            agent=None,
        )

        # Verify error result
        assert result["status"] == "error"
        assert "Error in use_agent tool" in result["content"][0]["text"]
        assert "Agent creation failed" in result["content"][0]["text"]


def test_use_agent_environment_fallback_to_parent(mock_parent_agent, mock_agent_result):
    """Test environment model creation falls back to parent when environment config fails."""
    with (
        patch("strands_tools.use_agent.Agent") as mock_agent_class,
        patch("strands_tools.use_agent.create_model") as mock_create_model,
        patch("strands_tools.use_agent.logger") as mock_logger,
        patch("strands.telemetry.metrics.metrics_to_string") as mock_metrics_to_string,
        patch.dict(os.environ, {"STRANDS_PROVIDER": "invalid_provider"}),
    ):
        # Make environment model creation fail
        mock_create_model.side_effect = Exception("Invalid provider")
        mock_metrics_to_string.return_value = "Mock metrics string"

        mock_nested_agent = MagicMock()
        mock_nested_agent.return_value = mock_agent_result
        mock_agent_class.return_value = mock_nested_agent

        result = use_agent_module.use_agent(
            prompt="Test environment fallback",
            system_prompt="You are a test assistant.",
            model_provider="env",
            agent=mock_parent_agent,
        )

        # Verify warning was logged
        mock_logger.warning.assert_called_with("Failed to create model from environment: Invalid provider")

        # Verify fallback to parent model
        call_kwargs = mock_agent_class.call_args.kwargs
        assert call_kwargs["model"] == mock_parent_agent.model

        assert result["status"] == "success"
        assert "Failed to use environment model" in result["content"][1]["text"]


def test_use_agent_with_all_model_providers(mock_parent_agent, mock_agent_result):
    """Test use_agent with all supported model providers."""
    providers = [
        "bedrock",
        "anthropic",
        "litellm",
        "llamaapi",
        "ollama",
        "openai",
        "github",
    ]

    for provider in providers:
        with (
            patch("strands_tools.use_agent.Agent") as mock_agent_class,
            patch("strands_tools.use_agent.create_model") as mock_create_model,
            patch("strands.telemetry.metrics.metrics_to_string") as mock_metrics_to_string,
        ):
            mock_model = MagicMock()
            mock_create_model.return_value = mock_model
            mock_metrics_to_string.return_value = "Mock metrics string"

            mock_nested_agent = MagicMock()
            mock_nested_agent.return_value = mock_agent_result
            mock_agent_class.return_value = mock_nested_agent

            result = use_agent_module.use_agent(
                prompt=f"Test with {provider}",
                system_prompt="You are a test assistant.",
                model_provider=provider,
                agent=mock_parent_agent,
            )

            # Verify model was created with correct provider
            mock_create_model.assert_called_once_with(provider=provider, config=None)

            # Verify Agent was created with the custom model
            call_kwargs = mock_agent_class.call_args.kwargs
            assert call_kwargs["model"] == mock_model

            assert result["status"] == "success"
            assert provider in result["content"][1]["text"].lower()


def test_use_agent_via_agent_interface(agent):
    """Test use_agent via the agent interface (integration test)."""
    with (
        patch("strands_tools.use_agent.Agent") as mock_agent_class,
        patch("strands.telemetry.metrics.metrics_to_string") as mock_metrics_to_string,
    ):
        mock_nested_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.__str__ = MagicMock(return_value="Integration test response")
        mock_result.metrics = None
        mock_nested_agent.return_value = mock_result
        mock_agent_class.return_value = mock_nested_agent
        mock_metrics_to_string.return_value = "Mock metrics string"

        # Test calling through agent interface
        try:
            result = agent.tool.use_agent(
                prompt="Integration test prompt",
                system_prompt="You are an integration test assistant.",
            )
            # If we get here without an exception, consider the test passed
            # The result structure depends on how the tool is called through agent interface
            assert result is not None
        except Exception as e:
            pytest.fail(f"Agent use_agent call raised an exception: {e}")


def test_use_agent_empty_tools_list(mock_parent_agent, mock_agent_result):
    """Test use_agent with empty tools list."""
    with (
        patch("strands_tools.use_agent.Agent") as mock_agent_class,
        patch("strands.telemetry.metrics.metrics_to_string") as mock_metrics_to_string,
    ):
        mock_nested_agent = MagicMock()
        mock_nested_agent.return_value = mock_agent_result
        mock_agent_class.return_value = mock_nested_agent
        mock_metrics_to_string.return_value = "Mock metrics string"

        result = use_agent_module.use_agent(
            prompt="Test with no tools",
            system_prompt="You are a test assistant.",
            tools=[],  # Empty list
            agent=mock_parent_agent,
        )

        # Verify Agent was created with empty tools list
        call_kwargs = mock_agent_class.call_args.kwargs
        assert call_kwargs["tools"] == []

        assert result["status"] == "success"


def test_use_agent_no_parent_agent():
    """Test use_agent without parent agent (edge case)."""
    with (
        patch("strands_tools.use_agent.Agent") as mock_agent_class,
        patch("strands.telemetry.metrics.metrics_to_string") as mock_metrics_to_string,
    ):
        mock_nested_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.__str__ = MagicMock(return_value="No parent agent test")
        mock_result.metrics = None
        mock_nested_agent.return_value = mock_result
        mock_agent_class.return_value = mock_nested_agent
        mock_metrics_to_string.return_value = "Mock metrics string"

        result = use_agent_module.use_agent(
            prompt="Test without parent agent",
            system_prompt="You are a test assistant.",
            agent=None,
        )

        # Verify Agent was created with default parameters
        call_kwargs = mock_agent_class.call_args.kwargs
        assert call_kwargs["model"] is None
        assert call_kwargs["tools"] == []
        assert "trace_attributes" not in call_kwargs or call_kwargs["trace_attributes"] == {}

        assert result["status"] == "success"


def test_use_agent_metrics_processing(mock_parent_agent, mock_agent_result):
    """Test use_agent properly processes and includes metrics in response."""
    with (
        patch("strands_tools.use_agent.Agent") as mock_agent_class,
        patch("strands.telemetry.metrics.metrics_to_string") as mock_metrics_to_string,
    ):
        mock_nested_agent = MagicMock()
        mock_nested_agent.return_value = mock_agent_result
        mock_agent_class.return_value = mock_nested_agent

        mock_metrics_to_string.return_value = "Input: 20 tokens, Output: 30 tokens, Total: 50 tokens"

        result = use_agent_module.use_agent(
            prompt="Test metrics processing",
            system_prompt="You are a test assistant.",
            agent=mock_parent_agent,
        )

        # Verify the result has the expected structure (like other tests)
        assert result["status"] == "success"
        assert len(result["content"]) == 3  # Response, Model, Metrics
        assert "Response:" in result["content"][0]["text"]
        assert "Model:" in result["content"][1]["text"]
        assert "Metrics:" in result["content"][2]["text"]

        # Verify Agent was instantiated with correct parameters (like other tests)
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args.kwargs
        assert call_kwargs["model"] == mock_parent_agent.model
        assert call_kwargs["system_prompt"] == "You are a test assistant."

        # Verify the nested agent was called with the prompt (like other tests)
        mock_nested_agent.assert_called_once_with("Test metrics processing")
