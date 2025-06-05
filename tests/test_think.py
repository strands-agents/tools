"""
Tests for the think tool using the Agent interface.
"""

from unittest.mock import MagicMock, patch

from strands.agent import AgentResult
from strands_tools import think
from strands_tools.think import ThoughtProcessor


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


def test_think_tool_direct():
    """Test direct invocation of the think tool."""
    # Create a tool use dictionary similar to how the agent would call it
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "name": "think",
        "input": {
            "thought": "What are the implications of quantum computing on cryptography?",
            "cycle_count": 2,
            "system_prompt": "You are an expert analytical thinker.",
        },
    }

    # Mock Agent class since we don't want to actually call the LLM
    with patch("strands_tools.think.Agent") as mock_agent_class:
        # Setup mock agent and response
        mock_agent = mock_agent_class.return_value
        mock_result = AgentResult(
            message={"content": [{"text": "This is a mock analysis of quantum computing."}]},
            stop_reason="end_turn",
            metrics=None,
            state=MagicMock(),
        )
        mock_agent.return_value = mock_result

        # Call the think function directly
        tool_input = tool_use.get("input", {})
        result = think.think(
            thought=tool_input.get("thought"),
            cycle_count=tool_input.get("cycle_count"),
            system_prompt=tool_input.get("system_prompt"),
            agent=None,
        )

        # Verify the result has the expected structure
        assert result["status"] == "success"
        assert "Cycle 1/2" in result["content"][0]["text"]
        assert "Cycle 2/2" in result["content"][0]["text"]

        # Verify Agent was called twice (once for each cycle)
        assert mock_agent.call_count == 2


def test_think_one_cycle():
    """Test think tool with a single cycle."""
    tool_use = {
        "toolUseId": "test-one-cycle",
        "name": "think",
        "input": {
            "thought": "Simple thought for one cycle",
            "cycle_count": 1,
            "system_prompt": "You are an expert analytical thinker.",
        },
    }

    with patch("strands_tools.think.Agent") as mock_agent_class:
        mock_agent = mock_agent_class.return_value
        mock_result = AgentResult(
            message={"content": [{"text": "Analysis for single cycle."}]},
            stop_reason="end_turn",
            metrics=None,
            state=MagicMock(),
        )
        mock_agent.return_value = mock_result

        tool_input = tool_use.get("input", {})
        result = think.think(
            thought=tool_input.get("thought"),
            cycle_count=tool_input.get("cycle_count"),
            system_prompt=tool_input.get("system_prompt"),
            agent=None,
        )

        assert result["status"] == "success"
        assert "Cycle 1/1" in result["content"][0]["text"]
        assert mock_agent.call_count == 1


def test_think_error_handling():
    """Test error handling in the think tool."""
    tool_use = {
        "toolUseId": "test-error-case",
        "name": "think",
        "input": {
            "thought": "Thought that will cause an error",
            "cycle_count": 2,
            "system_prompt": "You are an expert analytical thinker.",
        },
    }

    with patch("strands_tools.think.Agent") as mock_agent_class:
        # Make Agent raise an exception
        mock_agent_class.side_effect = Exception("Test error")

        tool_input = tool_use.get("input", {})
        result = think.think(
            thought=tool_input.get("thought"),
            cycle_count=tool_input.get("cycle_count"),
            system_prompt=tool_input.get("system_prompt"),
            agent=None,
        )

        assert result["status"] == "error"
        assert "Error in think tool" in result["content"][0]["text"]


def test_thought_processor():
    """Test the ThoughtProcessor class."""
    mock_console = MagicMock()
    processor = ThoughtProcessor({"system_prompt": "System prompt", "messages": []}, mock_console)

    # Test creating thinking prompt
    prompt = processor.create_thinking_prompt("Test thought", 1, 3)
    assert "Test thought" in prompt
    assert "Current Cycle: 1/3" in prompt
    assert "DO NOT call the think tool again" in prompt


def test_think_with_tool_filtering():
    """Test think tool with specific tools filtering from parent agent."""
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

    with patch("strands_tools.think.Agent") as mock_agent_class:
        mock_agent = mock_agent_class.return_value
        mock_result = AgentResult(
            message={"content": [{"text": "Analysis with filtered tools."}]},
            stop_reason="end_turn",
            metrics=None,
            state=MagicMock(),
        )
        mock_agent.return_value = mock_result

        # Call think with tool filtering
        result = think.think(
            thought="Test thought with tool filtering",
            cycle_count=1,
            system_prompt="You are an expert analytical thinker.",
            tools=["calculator", "file_read"],
            agent=mock_parent_agent,
        )

        # Verify the Agent was created with only the specified tools
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args.kwargs

        # Should only include calculator and file_read tools, not other_tool
        passed_tools = call_kwargs["tools"]
        assert len(passed_tools) == 2
        assert mock_calculator_tool in passed_tools
        assert mock_file_read_tool in passed_tools
        assert mock_other_tool not in passed_tools

        # Verify the result
        assert result["status"] == "success"
        assert "Cycle 1/1" in result["content"][0]["text"]


def test_think_with_nonexistent_tool_filtering():
    """Test think tool with tool filtering that includes non-existent tools."""
    # Create mock tools for the parent agent (missing nonexistent_tool)
    mock_calculator_tool = MagicMock(name="calculator_tool")
    mock_file_read_tool = MagicMock(name="file_read_tool")

    # Create a mock parent agent with limited tools
    mock_parent_agent = MagicMock()
    mock_parent_agent.tool_registry.registry = {"calculator": mock_calculator_tool, "file_read": mock_file_read_tool}
    mock_parent_agent.trace_attributes = {}

    with patch("strands_tools.think.Agent") as mock_agent_class, patch("strands_tools.think.logger") as mock_logger:
        mock_agent = mock_agent_class.return_value
        mock_result = AgentResult(
            message={"content": [{"text": "Analysis with missing tool."}]},
            stop_reason="end_turn",
            metrics=None,
            state=MagicMock(),
        )
        mock_agent.return_value = mock_result

        # Call think with tool filtering including non-existent tool
        result = think.think(
            thought="Test thought with non-existent tool",
            cycle_count=1,
            system_prompt="You are an expert analytical thinker.",
            tools=["calculator", "nonexistent_tool", "file_read"],
            agent=mock_parent_agent,
        )

        # Verify warning was logged for non-existent tool
        mock_logger.warning.assert_called_once_with("Tool 'nonexistent_tool' not found in parent agent's tool registry")

        # Verify the Agent was created with only the existing tools
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args.kwargs

        # Should only include existing tools (calculator and file_read)
        passed_tools = call_kwargs["tools"]
        assert len(passed_tools) == 2
        assert mock_calculator_tool in passed_tools
        assert mock_file_read_tool in passed_tools

        # Verify the result
        assert result["status"] == "success"


def test_think_with_empty_tool_filtering():
    """Test think tool with empty tools list (should result in no tools)."""
    # Create a mock parent agent with tools
    mock_parent_agent = MagicMock()
    mock_parent_agent.tool_registry.registry = {"calculator": MagicMock(), "file_read": MagicMock()}
    mock_parent_agent.trace_attributes = {}

    with patch("strands_tools.think.Agent") as mock_agent_class:
        mock_agent = mock_agent_class.return_value
        mock_result = AgentResult(
            message={"content": [{"text": "Analysis with no tools."}]},
            stop_reason="end_turn",
            metrics=None,
            state=MagicMock(),
        )
        mock_agent.return_value = mock_result

        # Call think with empty tools list
        result = think.think(
            thought="Test thought with no tools",
            cycle_count=1,
            system_prompt="You are an expert analytical thinker.",
            tools=[],
            agent=mock_parent_agent,
        )

        # Verify the Agent was created with no tools
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args.kwargs

        # Should be empty tools list
        passed_tools = call_kwargs["tools"]
        assert len(passed_tools) == 0

        # Verify the result
        assert result["status"] == "success"


def test_think_without_tool_filtering_inherits_all():
    """Test think tool without tools parameter inherits all parent tools."""
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

    with patch("strands_tools.think.Agent") as mock_agent_class:
        mock_agent = mock_agent_class.return_value
        mock_result = AgentResult(
            message={"content": [{"text": "Analysis with all tools."}]},
            stop_reason="end_turn",
            metrics=None,
            state=MagicMock(),
        )
        mock_agent.return_value = mock_result

        # Call think without tools parameter (should inherit all)
        result = think.think(
            thought="Test thought inheriting all tools",
            cycle_count=1,
            system_prompt="You are an expert analytical thinker.",
            agent=mock_parent_agent,
        )

        # Verify the Agent was created with all parent tools
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args.kwargs

        # Should include all tools from parent
        passed_tools = call_kwargs["tools"]
        assert len(passed_tools) == 3
        assert mock_calculator_tool in passed_tools
        assert mock_file_read_tool in passed_tools
        assert mock_other_tool in passed_tools

        # Verify the result
        assert result["status"] == "success"


def test_think_tool_filtering_with_multiple_cycles():
    """Test think tool with tool filtering across multiple cycles."""
    # Create mock tools for the parent agent
    mock_calculator_tool = MagicMock(name="calculator_tool")
    mock_file_read_tool = MagicMock(name="file_read_tool")

    # Create a mock parent agent with tools
    mock_parent_agent = MagicMock()
    mock_parent_agent.tool_registry.registry = {"calculator": mock_calculator_tool, "file_read": mock_file_read_tool}
    mock_parent_agent.trace_attributes = {}

    with patch("strands_tools.think.Agent") as mock_agent_class:
        mock_agent = mock_agent_class.return_value
        mock_result = AgentResult(
            message={"content": [{"text": "Multi-cycle analysis with filtered tools."}]},
            stop_reason="end_turn",
            metrics=None,
            state=MagicMock(),
        )
        mock_agent.return_value = mock_result

        # Call think with tool filtering and multiple cycles
        result = think.think(
            thought="Multi-cycle thought with tool filtering",
            cycle_count=3,
            system_prompt="You are an expert analytical thinker.",
            tools=["calculator"],
            agent=mock_parent_agent,
        )

        # Verify the Agent was created with filtered tools for each cycle
        assert mock_agent_class.call_count == 3  # One for each cycle

        # Check that all calls used the filtered tools
        for call in mock_agent_class.call_args_list:
            call_kwargs = call.kwargs
            passed_tools = call_kwargs["tools"]
            assert len(passed_tools) == 1
            assert mock_calculator_tool in passed_tools
            assert mock_file_read_tool not in passed_tools

        # Verify the result contains all cycles
        assert result["status"] == "success"
        assert "Cycle 1/3" in result["content"][0]["text"]
        assert "Cycle 2/3" in result["content"][0]["text"]
        assert "Cycle 3/3" in result["content"][0]["text"]


def test_think_no_parent_agent_with_tools_parameter():
    """Test think tool with tools parameter but no parent agent (should use empty tools)."""
    with patch("strands_tools.think.Agent") as mock_agent_class:
        mock_agent = mock_agent_class.return_value
        mock_result = AgentResult(
            message={"content": [{"text": "Analysis without parent agent."}]},
            stop_reason="end_turn",
            metrics=None,
            state=MagicMock(),
        )
        mock_agent.return_value = mock_result

        # Call think with tools parameter but no parent agent
        result = think.think(
            thought="Test thought without parent agent",
            cycle_count=1,
            system_prompt="You are an expert analytical thinker.",
            tools=["calculator", "file_read"],
            agent=None,
        )

        # Verify the Agent was created with empty tools (since no parent to filter from)
        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args.kwargs

        # Should be empty tools list since no parent agent
        passed_tools = call_kwargs["tools"]
        assert len(passed_tools) == 0

        # Verify the result
        assert result["status"] == "success"
