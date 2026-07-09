"""
Tests for the use_llm tool using the Agent interface.
"""

from unittest.mock import MagicMock, patch

import pytest
from strands.agent import AgentResult

from strands_tools import use_llm


@pytest.fixture
def mock_agent_response():
    """Create a mock response from an Agent."""
    return AgentResult(
        stop_reason="end_turn",
        message={"content": [{"text": "This is a test response from the LLM"}]},
        metrics=None,
        state=MagicMock(),
    )


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_use_llm_tool_direct(mock_agent_response):
    """Test direct invocation of the use_llm tool."""
    # Create a tool use dictionary similar to how the agent would call it
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "prompt": "Test prompt",
            "system_prompt": "You are a helpful test assistant",
        },
    }

    # Mock the Agent class to avoid actual LLM calls
    with patch("strands_tools.use_llm.Agent") as MockAgent:
        # Configure the mock agent to return our pre-defined response
        mock_instance = MockAgent.return_value
        mock_instance.return_value = mock_agent_response
        mock_agent_response.message = {
            "role": "assistant",
            "content": [{"text": "This is a test response from the LLM"}],
        }

        # Suppress print output
        with patch("builtins.print"):
            # Call the use_llm function directly
            result = use_llm.use_llm(tool=tool_use)

        # Verify the result has the expected structure
        assert result["toolUseId"] == "test-tool-use-id"
        assert result["status"] == "success"
        assert "This is a test response from the LLM" in str(result)

        # Verify the Agent was created with the correct parameters
        MockAgent.assert_called_once_with(
            messages=[], tools=[], system_prompt="You are a helpful test assistant", trace_attributes={}
        )


def test_use_llm_with_custom_system_prompt(mock_agent_response):
    """Test use_llm with a custom system prompt."""
    tool_use = {
        "toolUseId": "test-custom-prompt",
        "input": {
            "prompt": "Custom prompt test",
            "system_prompt": "You are a specialized test assistant",
        },
    }

    with patch("strands_tools.use_llm.Agent") as MockAgent:
        mock_instance = MockAgent.return_value
        mock_instance.return_value = mock_agent_response
        mock_agent_response.message = {"content": [{"text": "Custom response"}]}

        # Suppress print output
        with patch("builtins.print"):
            result = use_llm.use_llm(tool=tool_use)

        # Verify agent was created with correct system prompt
        MockAgent.assert_called_once_with(
            messages=[], tools=[], system_prompt="You are a specialized test assistant", trace_attributes={}
        )

        assert result["status"] == "success"
        assert "Custom response" in result["content"][0]["text"]


def test_use_llm_error_handling():
    """Test error handling in the use_llm tool."""
    tool_use = {
        "toolUseId": "test-error-handling",
        "input": {"prompt": "Error test", "system_prompt": "Test system prompt"},
    }

    # Simulate an error in the Agent
    with patch("strands_tools.use_llm.Agent") as MockAgent:
        # First we need to create a mock instance with the right return structure
        mock_instance = MockAgent.return_value
        # Then make the call to the mock instance raise an exception
        mock_instance.side_effect = Exception("Test error")

        # Add a try/except block to match the function behavior
        with patch("builtins.print"):  # Suppress print statements
            try:
                use_llm.use_llm(tool=tool_use)
                raise AssertionError("Should have raised an exception")
            except Exception as e:
                # This matches the current behavior - the error isn't caught in the function
                assert str(e) == "Test error"


def test_use_llm_metrics_handling(mock_agent_response):
    """Test that metrics from the agent response are properly processed."""
    tool_use = {
        "toolUseId": "test-metrics-handling",
        "input": {"prompt": "Test with metrics", "system_prompt": "Test system prompt"},
    }

    with patch("strands_tools.use_llm.Agent") as MockAgent:
        mock_instance = MockAgent.return_value
        mock_instance.return_value = mock_agent_response
        mock_agent_response.metrics = MagicMock()
        # Add tool_config attribute to mock
        mock_instance.tool_config = {"tools": [{"toolSpec": {"name": "test_tool"}}]}

        with patch("strands_tools.use_llm.metrics_to_string") as mock_metrics:
            mock_metrics.return_value = "Tokens: 30, Latency: 0.5s"

            # Suppress print output
            with patch("builtins.print"):
                result = use_llm.use_llm(tool=tool_use)

            # Verify metrics_to_string was called with the correct parameters
            mock_metrics.assert_called_once()
            assert mock_metrics.call_args[0][0] == mock_agent_response.metrics

            assert result["status"] == "success"


def test_use_llm_complex_response_handling():
    """Test that complex responses from the nested agent are properly handled."""
    tool_use = {
        "toolUseId": "test-complex-response",
        "input": {
            "prompt": "Complex response test",
            "system_prompt": "Test system prompt",
        },
    }

    # Create a complex response with multiple content items
    complex_response = AgentResult(
        stop_reason="end_turn",
        metrics=None,
        message={
            "role": "assistant",
            "content": [
                {"text": "First part of response"},
                {"text": "Second part of response"},
            ],
        },
        state=MagicMock(),
    )

    with patch("strands_tools.use_llm.Agent") as MockAgent:
        mock_instance = MockAgent.return_value
        mock_instance.return_value = complex_response

        # Suppress print output
        with patch("builtins.print"):
            result = use_llm.use_llm(tool=tool_use)

        assert result["status"] == "success"
        assert "First part of response\nSecond part of response" in result["content"][0]["text"]


def test_use_llm_with_parent_agent_callback():
    """Test that the parent agent's callback handler is properly passed to the new agent."""
    tool_use = {
        "toolUseId": "test-parent-callback",
        "input": {
            "prompt": "Test with parent callback",
            "system_prompt": "Test system prompt",
        },
    }

    # Create a mock parent agent with a callback handler
    mock_parent_agent = MagicMock()
    mock_parent_agent.tool_registry.registry.values.return_value = []
    mock_parent_agent.trace_attributes = {"test_attribute": "test_value"}
    mock_parent_agent.callback_handler = MagicMock(name="parent_callback_handler")

    # Create a mock response
    mock_response = MagicMock()
    mock_response.metrics = None
    mock_response.__str__.return_value = "Test response with parent callback"

    with patch("strands_tools.use_llm.Agent") as MockAgent:
        # Configure the mock agent
        mock_instance = MockAgent.return_value
        mock_instance.return_value = mock_response

        # Call use_llm with the parent agent
        result = use_llm.use_llm(tool=tool_use, agent=mock_parent_agent)

        # Verify the Agent was created with the parent's callback handler
        MockAgent.assert_called_once()
        call_kwargs = MockAgent.call_args.kwargs
        assert call_kwargs["callback_handler"] == mock_parent_agent.callback_handler
        assert call_kwargs["trace_attributes"] == {"test_attribute": "test_value"}

        # Verify the result
        assert result["status"] == "success"
        assert "Test response with parent callback" in result["content"][0]["text"]


def test_use_llm_with_explicit_callback():
    """Test that an explicitly provided callback handler takes precedence over the parent agent's callback."""
    tool_use = {
        "toolUseId": "test-explicit-callback",
        "input": {
            "prompt": "Test with explicit callback",
            "system_prompt": "Test system prompt",
        },
    }

    # Create a mock parent agent with a callback handler
    mock_parent_agent = MagicMock()
    mock_parent_agent.tool_registry.registry.values.return_value = []
    mock_parent_agent.trace_attributes = {"test_attribute": "test_value"}
    mock_parent_agent.callback_handler = MagicMock(name="parent_callback_handler")

    # Create an explicit callback handler that should take precedence
    explicit_callback_handler = MagicMock(name="explicit_callback_handler")

    # Create a mock response
    mock_response = MagicMock()
    mock_response.metrics = None
    mock_response.__str__.return_value = "Test response with explicit callback"

    with patch("strands_tools.use_llm.Agent") as MockAgent:
        # Configure the mock agent
        mock_instance = MockAgent.return_value
        mock_instance.return_value = mock_response

        # Call use_llm with both parent agent and explicit callback handler
        result = use_llm.use_llm(tool=tool_use, agent=mock_parent_agent, callback_handler=explicit_callback_handler)

        # Verify the Agent was created with the explicit callback handler, not the parent's
        MockAgent.assert_called_once()
        call_kwargs = MockAgent.call_args.kwargs
        assert call_kwargs["callback_handler"] == explicit_callback_handler
        assert call_kwargs["callback_handler"] != mock_parent_agent.callback_handler

        # Verify other parameters were still passed correctly
        assert call_kwargs["trace_attributes"] == {"test_attribute": "test_value"}

        # Verify the result
        assert result["status"] == "success"
        assert "Test response with explicit callback" in result["content"][0]["text"]


def test_use_llm_with_tool_filtering():
    """Test use_llm with specific tools filtering from parent agent."""
    tool_use = {
        "toolUseId": "test-tool-filtering",
        "input": {
            "prompt": "Test with tool filtering",
            "system_prompt": "Test system prompt",
            "tools": ["calculator", "file_read"],
        },
    }

    # Create mock tools for the parent agent
    mock_calculator_tool = MagicMock(name="calculator_tool")
    mock_file_read_tool = MagicMock(name="file_read_tool")
    mock_other_tool = MagicMock(name="other_tool")

    # Create a mock parent agent with multiple tools
    mock_parent_agent = MagicMock()
    mock_parent_agent.tool_registry.registry = {
        "calculator": mock_calculator_tool,
        "file_read": mock_file_read_tool,
        "other_tool": mock_other_tool,
    }
    mock_parent_agent.trace_attributes = {}
    mock_parent_agent.callback_handler = MagicMock()

    # Create a mock response
    mock_response = MagicMock()
    mock_response.metrics = None
    mock_response.__str__.return_value = "Test response with filtered tools"

    with patch("strands_tools.use_llm.Agent") as MockAgent:
        # Configure the mock agent
        mock_instance = MockAgent.return_value
        mock_instance.return_value = mock_response

        # Call use_llm with tool filtering
        result = use_llm.use_llm(tool=tool_use, agent=mock_parent_agent)

        # Verify the Agent was created with only the specified tools
        MockAgent.assert_called_once()
        call_kwargs = MockAgent.call_args.kwargs

        # Should only include calculator and file_read tools, not other_tool
        passed_tools = call_kwargs["tools"]
        assert len(passed_tools) == 2
        assert mock_calculator_tool in passed_tools
        assert mock_file_read_tool in passed_tools
        assert mock_other_tool not in passed_tools

        # Verify the result
        assert result["status"] == "success"
        assert "Test response with filtered tools" in result["content"][0]["text"]


def test_use_llm_with_nonexistent_tool_filtering():
    """Test use_llm with tool filtering that includes non-existent tools."""
    tool_use = {
        "toolUseId": "test-nonexistent-tool",
        "input": {
            "prompt": "Test with non-existent tool",
            "system_prompt": "Test system prompt",
            "tools": ["calculator", "nonexistent_tool", "file_read"],
        },
    }

    # Create mock tools for the parent agent (missing nonexistent_tool)
    mock_calculator_tool = MagicMock(name="calculator_tool")
    mock_file_read_tool = MagicMock(name="file_read_tool")

    # Create a mock parent agent with limited tools
    mock_parent_agent = MagicMock()
    mock_parent_agent.tool_registry.registry = {"calculator": mock_calculator_tool, "file_read": mock_file_read_tool}
    mock_parent_agent.trace_attributes = {}
    mock_parent_agent.callback_handler = MagicMock()

    # Create a mock response
    mock_response = MagicMock()
    mock_response.metrics = None
    mock_response.__str__.return_value = "Test response with missing tool"

    with patch("strands_tools.use_llm.Agent") as MockAgent, patch("strands_tools.use_llm.logger") as mock_logger:
        # Configure the mock agent
        mock_instance = MockAgent.return_value
        mock_instance.return_value = mock_response

        # Call use_llm with tool filtering including non-existent tool
        result = use_llm.use_llm(tool=tool_use, agent=mock_parent_agent)

        # Verify warning was logged for non-existent tool
        mock_logger.warning.assert_called_with("Tool 'nonexistent_tool' not found in parent agent's tool registry")

        # Verify the Agent was created with only the existing tools
        MockAgent.assert_called_once()
        call_kwargs = MockAgent.call_args.kwargs

        # Should only include existing tools (calculator and file_read)
        passed_tools = call_kwargs["tools"]
        assert len(passed_tools) == 2
        assert mock_calculator_tool in passed_tools
        assert mock_file_read_tool in passed_tools

        # Verify the result
        assert result["status"] == "success"


def test_use_llm_with_empty_tool_filtering():
    """Test use_llm with empty tools list (should result in no tools)."""
    tool_use = {
        "toolUseId": "test-empty-tools",
        "input": {"prompt": "Test with empty tools list", "system_prompt": "Test system prompt", "tools": []},
    }

    # Create a mock parent agent with tools
    mock_parent_agent = MagicMock()
    mock_parent_agent.tool_registry.registry = {"calculator": MagicMock(), "file_read": MagicMock()}
    mock_parent_agent.trace_attributes = {}
    mock_parent_agent.callback_handler = MagicMock()

    # Create a mock response
    mock_response = MagicMock()
    mock_response.metrics = None
    mock_response.__str__.return_value = "Test response with no tools"

    with patch("strands_tools.use_llm.Agent") as MockAgent:
        # Configure the mock agent
        mock_instance = MockAgent.return_value
        mock_instance.return_value = mock_response

        # Call use_llm with empty tools list
        result = use_llm.use_llm(tool=tool_use, agent=mock_parent_agent)

        # Verify the Agent was created with no tools
        MockAgent.assert_called_once()
        call_kwargs = MockAgent.call_args.kwargs

        # Should be empty tools list
        passed_tools = call_kwargs["tools"]
        assert len(passed_tools) == 0

        # Verify the result
        assert result["status"] == "success"


def test_use_llm_without_tool_filtering_inherits_all():
    """Test use_llm without tools parameter inherits all parent tools."""
    tool_use = {
        "toolUseId": "test-inherit-all-tools",
        "input": {
            "prompt": "Test inheriting all tools",
            "system_prompt": "Test system prompt",
            # No tools parameter - should inherit all
        },
    }

    # Create mock tools for the parent agent
    mock_calculator_tool = MagicMock(name="calculator_tool")
    mock_file_read_tool = MagicMock(name="file_read_tool")
    mock_other_tool = MagicMock(name="other_tool")

    # Create a mock parent agent with multiple tools
    mock_parent_agent = MagicMock()
    mock_parent_agent.tool_registry.registry.values.return_value = [
        mock_calculator_tool,
        mock_file_read_tool,
        mock_other_tool,
    ]
    mock_parent_agent.trace_attributes = {}
    mock_parent_agent.callback_handler = MagicMock()

    # Create a mock response
    mock_response = MagicMock()
    mock_response.metrics = None
    mock_response.__str__.return_value = "Test response with all tools"

    with patch("strands_tools.use_llm.Agent") as MockAgent:
        # Configure the mock agent
        mock_instance = MockAgent.return_value
        mock_instance.return_value = mock_response

        # Call use_llm without tools parameter
        result = use_llm.use_llm(tool=tool_use, agent=mock_parent_agent)

        # Verify the Agent was created with all parent tools
        MockAgent.assert_called_once()
        call_kwargs = MockAgent.call_args.kwargs

        # Should include all tools from parent
        passed_tools = call_kwargs["tools"]
        assert len(passed_tools) == 3
        assert mock_calculator_tool in passed_tools
        assert mock_file_read_tool in passed_tools
        assert mock_other_tool in passed_tools

        # Verify the result
        assert result["status"] == "success"
