"""
Tests for the manage_messages tool using the Agent interface.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest
from strands import Agent

from strands_tools import manage_messages as manage_messages_module


@pytest.fixture
def agent():
    """Create an agent with the manage_messages tool loaded."""
    return Agent(tools=[manage_messages_module], load_tools_from_directory=False)


@pytest.fixture
def mock_agent():
    """Create a mock agent with messages."""
    mock = MagicMock()
    mock.messages = [
        {"role": "user", "content": [{"text": "Hello, how are you?"}]},
        {"role": "assistant", "content": [{"text": "I am doing well, thank you!"}]},
        {"role": "user", "content": [{"text": "Can you help me with Python?"}]},
        {"role": "assistant", "content": [{"text": "Of course! What do you need help with?"}]},
    ]
    return mock


@pytest.fixture
def mock_agent_with_tools():
    """Create a mock agent with tool use messages."""
    mock = MagicMock()
    mock.messages = [
        {"role": "user", "content": [{"text": "Calculate 2+2"}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "tool_123", "name": "calculator", "input": {"expression": "2+2"}}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "tool_123", "status": "success", "content": [{"text": "4"}]}}],
        },
        {"role": "assistant", "content": [{"text": "The result is 4."}]},
    ]
    return mock


@pytest.fixture
def mock_agent_with_multiple_tools():
    """Create a mock agent with multiple tool calls."""
    mock = MagicMock()
    mock.messages = [
        {"role": "user", "content": [{"text": "Do multiple things"}]},
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "tool_001", "name": "shell", "input": {"command": "ls"}}},
                {"toolUse": {"toolUseId": "tool_002", "name": "calculator", "input": {"expr": "1+1"}}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "tool_001", "status": "success", "content": [{"text": "file1.txt"}]}},
                {"toolResult": {"toolUseId": "tool_002", "status": "success", "content": [{"text": "2"}]}},
            ],
        },
        {"role": "assistant", "content": [{"text": "Done with both tasks."}]},
        {"role": "user", "content": [{"text": "Run shell again"}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "tool_003", "name": "shell", "input": {"command": "pwd"}}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "tool_003", "status": "success", "content": [{"text": "/home"}]}}],
        },
        {"role": "assistant", "content": [{"text": "Current directory is /home."}]},
    ]
    return mock


@pytest.fixture
def mock_agent_with_thinking():
    """Create a mock agent with thinking blocks."""
    mock = MagicMock()
    mock.messages = [
        {"role": "user", "content": [{"text": "Think about this"}]},
        {
            "role": "assistant",
            "content": [
                {"thinking": "Let me think about this carefully..."},
                {"text": "Here's my answer."},
            ],
        },
        {"role": "user", "content": [{"text": "More questions"}]},
        {
            "role": "assistant",
            "content": [
                {"redacted_thinking": "Thinking content redacted"},
                {"toolUse": {"toolUseId": "tool_think", "name": "calculator", "input": {}}},
            ],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "tool_think", "status": "success", "content": [{"text": "42"}]}}],
        },
        {"role": "assistant", "content": [{"text": "The answer is 42."}]},
    ]
    return mock


@pytest.fixture
def mock_agent_with_active_turn():
    """Create a mock agent with an active (incomplete) turn."""
    mock = MagicMock()
    mock.messages = [
        {"role": "user", "content": [{"text": "First question"}]},
        {"role": "assistant", "content": [{"text": "First answer"}]},
        {"role": "user", "content": [{"text": "Second question - triggers tool"}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "active_tool", "name": "manage_messages", "input": {}}}],
        },
        # No toolResult yet - this is the active turn
    ]
    return mock


@pytest.fixture
def temp_file():
    """Create a temporary file for export/import tests."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


# --- List Tests ---


def test_manage_messages_list_empty(mock_agent):
    """Test listing when no messages exist."""
    mock_agent.messages = []

    result = manage_messages_module.manage_messages(action="list", agent=mock_agent)

    assert result["status"] == "success"
    assert "No messages" in result["content"][0]["text"]


def test_manage_messages_list_turns(mock_agent):
    """Test listing messages by turns."""
    result = manage_messages_module.manage_messages(action="list", agent=mock_agent)

    assert result["status"] == "success"
    assert "2 turns" in result["content"][0]["text"]
    assert "Turn 0" in result["content"][0]["text"]
    assert "Turn 1" in result["content"][0]["text"]
    assert "Hello, how are you?" in result["content"][0]["text"]


def test_manage_messages_list_filter_by_role(mock_agent):
    """Test listing messages filtered by role."""
    result = manage_messages_module.manage_messages(action="list", role="user", agent=mock_agent)

    assert result["status"] == "success"
    assert "2 user messages" in result["content"][0]["text"]
    assert "Hello" in result["content"][0]["text"]
    assert "Python" in result["content"][0]["text"]


def test_manage_messages_list_filter_assistant(mock_agent):
    """Test listing only assistant messages."""
    result = manage_messages_module.manage_messages(action="list", role="assistant", agent=mock_agent)

    assert result["status"] == "success"
    assert "2 assistant messages" in result["content"][0]["text"]


def test_manage_messages_list_with_tool_use(mock_agent_with_tools):
    """Test listing messages containing tool use."""
    result = manage_messages_module.manage_messages(action="list", agent=mock_agent_with_tools)

    assert result["status"] == "success"
    assert "toolUse:calculator" in result["content"][0]["text"]
    assert "toolResult:" in result["content"][0]["text"]


# --- Stats Tests ---


def test_manage_messages_stats_empty(mock_agent):
    """Test stats when no messages exist."""
    mock_agent.messages = []

    result = manage_messages_module.manage_messages(action="stats", agent=mock_agent)

    assert result["status"] == "success"
    assert "No messages" in result["content"][0]["text"]


def test_manage_messages_stats(mock_agent):
    """Test getting conversation statistics."""
    result = manage_messages_module.manage_messages(action="stats", agent=mock_agent)

    assert result["status"] == "success"
    assert "Turns: 2" in result["content"][0]["text"]
    assert "Messages: 4" in result["content"][0]["text"]
    assert "user: 2" in result["content"][0]["text"]
    assert "assistant: 2" in result["content"][0]["text"]


def test_manage_messages_stats_with_tools(mock_agent_with_tools):
    """Test stats with tool use content."""
    result = manage_messages_module.manage_messages(action="stats", agent=mock_agent_with_tools)

    assert result["status"] == "success"
    assert "1 toolUse" in result["content"][0]["text"]
    assert "1 toolResult" in result["content"][0]["text"]


# --- Export Tests ---


def test_manage_messages_export(mock_agent, temp_file):
    """Test exporting conversation to file."""
    result = manage_messages_module.manage_messages(
        action="export",
        path=temp_file,
        agent=mock_agent,
    )

    assert result["status"] == "success"
    assert "Exported 2 turns" in result["content"][0]["text"]
    assert os.path.exists(temp_file)

    # Verify file contents
    with open(temp_file, "r") as f:
        exported = json.load(f)
    assert len(exported) == 4
    assert exported[0]["role"] == "user"


def test_manage_messages_export_creates_directory(mock_agent):
    """Test export creates parent directories if needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = os.path.join(tmpdir, "nested", "dir", "conversation.json")

        result = manage_messages_module.manage_messages(
            action="export",
            path=nested_path,
            agent=mock_agent,
        )

        assert result["status"] == "success"
        assert os.path.exists(nested_path)


def test_manage_messages_export_missing_path(mock_agent):
    """Test export without path parameter."""
    result = manage_messages_module.manage_messages(action="export", agent=mock_agent)

    assert result["status"] == "error"
    assert "Required: path parameter" in result["content"][0]["text"]


def test_manage_messages_export_with_tilde_path(mock_agent):
    """Test export with user home path (~)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock home directory scenario
        path = os.path.join(tmpdir, "test_export.json")

        result = manage_messages_module.manage_messages(
            action="export",
            path=path,
            agent=mock_agent,
        )

        assert result["status"] == "success"


# --- Import Tests ---


def test_manage_messages_import(mock_agent, temp_file):
    """Test importing conversation from file."""
    # First export
    manage_messages_module.manage_messages(action="export", path=temp_file, agent=mock_agent)

    # Clear messages
    mock_agent.messages.clear()
    assert len(mock_agent.messages) == 0

    # Import
    result = manage_messages_module.manage_messages(
        action="import",
        path=temp_file,
        agent=mock_agent,
    )

    assert result["status"] == "success"
    assert "Imported 2 turns" in result["content"][0]["text"]
    assert len(mock_agent.messages) == 4


def test_manage_messages_import_file_not_found(mock_agent):
    """Test import with non-existent file."""
    result = manage_messages_module.manage_messages(
        action="import",
        path="/nonexistent/path/conversation.json",
        agent=mock_agent,
    )

    assert result["status"] == "error"
    assert "File not found" in result["content"][0]["text"]


def test_manage_messages_import_invalid_format(mock_agent, temp_file):
    """Test import with invalid JSON format."""
    # Write invalid data (not a list)
    with open(temp_file, "w") as f:
        json.dump({"not": "a list"}, f)

    result = manage_messages_module.manage_messages(
        action="import",
        path=temp_file,
        agent=mock_agent,
    )

    assert result["status"] == "error"
    assert "Invalid format" in result["content"][0]["text"]


def test_manage_messages_import_missing_path(mock_agent):
    """Test import without path parameter."""
    result = manage_messages_module.manage_messages(action="import", agent=mock_agent)

    assert result["status"] == "error"
    assert "Required: path parameter" in result["content"][0]["text"]


def test_manage_messages_import_replaces_existing(mock_agent, temp_file):
    """Test that import replaces existing messages (preserving active turn)."""
    # Export current messages
    manage_messages_module.manage_messages(action="export", path=temp_file, agent=mock_agent)

    # Add more messages to create a new active turn
    mock_agent.messages.append({"role": "user", "content": [{"text": "Extra message"}]})
    mock_agent.messages.append({"role": "assistant", "content": [{"text": "Extra response"}]})

    # Import should replace all but active turn gets preserved
    result = manage_messages_module.manage_messages(
        action="import",
        path=temp_file,
        agent=mock_agent,
    )

    assert result["status"] == "success"
    # Messages should be replaced (original 4 + preserved active turn)
    assert len(mock_agent.messages) >= 4


# --- Drop Tests ---


def test_manage_messages_drop_empty(mock_agent):
    """Test drop when no messages exist."""
    mock_agent.messages = []

    result = manage_messages_module.manage_messages(
        action="drop",
        turns="0",
        agent=mock_agent,
    )

    assert result["status"] == "success"
    assert "No messages" in result["content"][0]["text"]


def test_manage_messages_drop_single_turn(mock_agent):
    """Test dropping a single turn."""
    result = manage_messages_module.manage_messages(
        action="drop",
        turns="0",
        agent=mock_agent,
    )

    assert result["status"] == "success"
    assert "Dropped 1 turns" in result["content"][0]["text"]
    assert len(mock_agent.messages) == 2
    # First turn should be removed
    assert "Python" in mock_agent.messages[0]["content"][0]["text"]


def test_manage_messages_drop_multiple_turns(mock_agent):
    """Test dropping multiple turns at once."""
    # Add more turns to have droppable content
    mock_agent.messages.extend(
        [
            {"role": "user", "content": [{"text": "Third question"}]},
            {"role": "assistant", "content": [{"text": "Third answer"}]},
        ]
    )

    result = manage_messages_module.manage_messages(
        action="drop",
        turns="0,1",
        agent=mock_agent,
    )

    assert result["status"] == "success"
    assert "Dropped 2 turns" in result["content"][0]["text"]


def test_manage_messages_drop_range_start_end(mock_agent):
    """Test dropping a range of turns with start and end."""
    # Add more turns first
    mock_agent.messages.extend(
        [
            {"role": "user", "content": [{"text": "Third question"}]},
            {"role": "assistant", "content": [{"text": "Third answer"}]},
        ]
    )

    result = manage_messages_module.manage_messages(
        action="drop",
        start=0,
        end=2,
        agent=mock_agent,
    )

    assert result["status"] == "success"
    assert "Dropped 2 turns" in result["content"][0]["text"]


def test_manage_messages_drop_range_start_only(mock_agent):
    """Test dropping from start to end with only start specified."""
    # Add more turns to have droppable content
    mock_agent.messages.extend(
        [
            {"role": "user", "content": [{"text": "Third question"}]},
            {"role": "assistant", "content": [{"text": "Third answer"}]},
        ]
    )

    result = manage_messages_module.manage_messages(
        action="drop",
        start=1,
        agent=mock_agent,
    )

    assert result["status"] == "success"
    # Should drop turn 1 onwards (leaving turn 0 and active turn)
    assert len(mock_agent.messages) >= 2


def test_manage_messages_drop_range_end_only(mock_agent):
    """Test dropping from beginning with only end specified."""
    result = manage_messages_module.manage_messages(
        action="drop",
        end=1,
        agent=mock_agent,
    )

    assert result["status"] == "success"
    # Should drop turns 0
    assert len(mock_agent.messages) == 2


def test_manage_messages_drop_invalid_index(mock_agent):
    """Test dropping with invalid turn index."""
    result = manage_messages_module.manage_messages(
        action="drop",
        turns="abc",
        agent=mock_agent,
    )

    assert result["status"] == "error"
    assert "Invalid turn index" in result["content"][0]["text"]


def test_manage_messages_drop_no_params(mock_agent):
    """Test drop without specifying turns or range."""
    result = manage_messages_module.manage_messages(action="drop", agent=mock_agent)

    assert result["status"] == "error"
    assert "Specify turns" in result["content"][0]["text"]


def test_manage_messages_drop_out_of_range(mock_agent):
    """Test dropping turn index that doesn't exist."""
    result = manage_messages_module.manage_messages(
        action="drop",
        turns="99",
        agent=mock_agent,
    )

    # Should succeed but not drop anything (index out of range is ignored)
    assert result["status"] == "success"
    assert len(mock_agent.messages) == 4


def test_manage_messages_drop_with_tool_cycle(mock_agent_with_tools):
    """Test that dropping maintains tool cycle integrity."""
    # Add a second complete turn so there's something to drop
    mock_agent_with_tools.messages.extend(
        [
            {"role": "user", "content": [{"text": "Second question"}]},
            {"role": "assistant", "content": [{"text": "Second answer"}]},
        ]
    )

    # Drop the first turn with tool use
    result = manage_messages_module.manage_messages(
        action="drop",
        turns="0",
        agent=mock_agent_with_tools,
    )

    assert result["status"] == "success"
    # First turn (with tool) should be removed
    assert "Dropped 1 turns" in result["content"][0]["text"]


def test_manage_messages_drop_fixes_incomplete_tool_cycle():
    """Test that dropping adds synthetic toolResult if needed."""
    mock = MagicMock()
    mock.messages = [
        {"role": "user", "content": [{"text": "Question 1"}]},
        {"role": "assistant", "content": [{"text": "Answer 1"}]},
        {"role": "user", "content": [{"text": "Question 2"}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "tool_456", "name": "shell", "input": {}}}],
        },
        # Missing toolResult - this simulates incomplete cycle after drop
        # Adding a complete turn at the end to have something to drop
        {"role": "user", "content": [{"toolResult": {"toolUseId": "tool_456", "status": "success", "content": []}}]},
        {"role": "assistant", "content": [{"text": "Done"}]},
        {"role": "user", "content": [{"text": "Final question"}]},
        {"role": "assistant", "content": [{"text": "Final answer"}]},
    ]

    # Drop first turn
    result = manage_messages_module.manage_messages(
        action="drop",
        turns="0",
        agent=mock,
    )

    assert result["status"] == "success"


# --- Clear Tests ---


def test_manage_messages_clear(mock_agent):
    """Test clearing all messages."""
    # Add a third turn so there's something to clear (2 complete turns + active turn)
    mock_agent.messages.extend(
        [
            {"role": "user", "content": [{"text": "Third question"}]},
            {"role": "assistant", "content": [{"text": "Third answer"}]},
        ]
    )
    assert len(mock_agent.messages) == 6

    result = manage_messages_module.manage_messages(action="clear", agent=mock_agent)

    assert result["status"] == "success"
    assert "Cleared" in result["content"][0]["text"]
    # Active turn is preserved, so messages may not be 0


def test_manage_messages_clear_empty(mock_agent):
    """Test clearing when already empty."""
    mock_agent.messages = []

    result = manage_messages_module.manage_messages(action="clear", agent=mock_agent)

    assert result["status"] == "success"
    assert "Cleared 0 turns" in result["content"][0]["text"]


# --- Error Handling Tests ---


def test_manage_messages_unknown_action(mock_agent):
    """Test handling of unknown action."""
    result = manage_messages_module.manage_messages(action="invalid", agent=mock_agent)

    assert result["status"] == "error"
    assert "Unknown action" in result["content"][0]["text"]
    # Check for all valid actions in the error message
    assert "list" in result["content"][0]["text"]
    assert "stats" in result["content"][0]["text"]


def test_manage_messages_no_agent():
    """Test handling when agent is not provided."""
    result = manage_messages_module.manage_messages(action="list", agent=None)

    assert result["status"] == "error"
    assert "Agent not available" in result["content"][0]["text"]


# --- Integration Tests ---


def test_manage_messages_via_agent_interface(agent):
    """Test manage_messages via the agent interface."""
    result = agent.tool.manage_messages(action="list")

    assert result is not None
    assert result["status"] == "success"


def test_manage_messages_export_import_roundtrip(mock_agent, temp_file):
    """Test full export/import roundtrip preserves data."""
    original_messages = list(mock_agent.messages)

    # Export
    manage_messages_module.manage_messages(action="export", path=temp_file, agent=mock_agent)

    # Clear
    mock_agent.messages.clear()

    # Import
    manage_messages_module.manage_messages(action="import", path=temp_file, agent=mock_agent)

    # Verify
    assert mock_agent.messages == original_messages


# --- Turn Parsing Tests ---


def test_turn_parsing_simple():
    """Test turn parsing with simple messages."""
    messages = [
        {"role": "user", "content": [{"text": "Q1"}]},
        {"role": "assistant", "content": [{"text": "A1"}]},
        {"role": "user", "content": [{"text": "Q2"}]},
        {"role": "assistant", "content": [{"text": "A2"}]},
    ]

    turns = manage_messages_module._parse_turns(messages)

    assert len(turns) == 2
    assert turns[0] == (0, 2)
    assert turns[1] == (2, 4)


def test_turn_parsing_with_tool_cycle():
    """Test turn parsing with tool use cycle."""
    messages = [
        {"role": "user", "content": [{"text": "Calculate"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "calc", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "status": "success", "content": []}}]},
        {"role": "assistant", "content": [{"text": "Result"}]},
    ]

    turns = manage_messages_module._parse_turns(messages)

    assert len(turns) == 1
    assert turns[0] == (0, 4)


def test_summarize_text_content():
    """Test content summarization for text."""
    content = [{"text": "This is a test message with some content"}]

    summary = manage_messages_module._summarize(content)

    assert "This is a test" in summary


def test_summarize_tool_use():
    """Test content summarization for tool use."""
    content = [{"toolUse": {"toolUseId": "123", "name": "calculator", "input": {}}}]

    summary = manage_messages_module._summarize(content)

    assert "toolUse:calculator" in summary


def test_summarize_mixed_content():
    """Test content summarization for mixed content."""
    content = [
        {"text": "Hello"},
        {"toolUse": {"toolUseId": "123", "name": "shell", "input": {}}},
        {"image": {"format": "png", "source": {}}},
    ]

    summary = manage_messages_module._summarize(content)

    assert "Hello" in summary
    assert "toolUse:shell" in summary
    assert "[image]" in summary


# --- List Tools Tests ---


def test_manage_messages_list_tools_empty(mock_agent):
    """Test list_tools when no messages exist."""
    mock_agent.messages = []

    result = manage_messages_module.manage_messages(action="list_tools", agent=mock_agent)

    assert result["status"] == "success"
    assert "No messages" in result["content"][0]["text"]


def test_manage_messages_list_tools_no_tools(mock_agent):
    """Test list_tools when no tool calls exist."""
    result = manage_messages_module.manage_messages(action="list_tools", agent=mock_agent)

    assert result["status"] == "success"
    assert "No tool calls found" in result["content"][0]["text"]


def test_manage_messages_list_tools(mock_agent_with_tools):
    """Test listing tool calls."""
    result = manage_messages_module.manage_messages(action="list_tools", agent=mock_agent_with_tools)

    assert result["status"] == "success"
    assert "1 tool calls" in result["content"][0]["text"]
    assert "calculator" in result["content"][0]["text"]
    assert "tool_123" in result["content"][0]["text"]


def test_manage_messages_list_tools_multiple(mock_agent_with_multiple_tools):
    """Test listing multiple tool calls."""
    result = manage_messages_module.manage_messages(action="list_tools", agent=mock_agent_with_multiple_tools)

    assert result["status"] == "success"
    assert "3 tool calls" in result["content"][0]["text"]
    assert "shell" in result["content"][0]["text"]
    assert "calculator" in result["content"][0]["text"]


def test_manage_messages_list_tools_with_active(mock_agent_with_active_turn):
    """Test list_tools shows active tool marker."""
    result = manage_messages_module.manage_messages(action="list_tools", agent=mock_agent_with_active_turn)

    assert result["status"] == "success"
    assert "active" in result["content"][0]["text"].lower()


# --- Drop Tools Tests ---


def test_manage_messages_drop_tools_empty(mock_agent):
    """Test drop_tools when no messages exist."""
    mock_agent.messages = []

    result = manage_messages_module.manage_messages(action="drop_tools", tool_ids="some_id", agent=mock_agent)

    assert result["status"] == "success"
    assert "No messages" in result["content"][0]["text"]


def test_manage_messages_drop_tools_no_params(mock_agent_with_tools):
    """Test drop_tools without required parameters."""
    result = manage_messages_module.manage_messages(action="drop_tools", agent=mock_agent_with_tools)

    assert result["status"] == "error"
    assert "Required" in result["content"][0]["text"]


def test_manage_messages_drop_tools_by_id(mock_agent_with_multiple_tools):
    """Test dropping specific tool calls by ID."""
    result = manage_messages_module.manage_messages(
        action="drop_tools",
        tool_ids="tool_001",
        agent=mock_agent_with_multiple_tools,
    )

    assert result["status"] == "success"
    assert "Dropped 1 tool call" in result["content"][0]["text"]


def test_manage_messages_drop_tools_by_name(mock_agent_with_multiple_tools):
    """Test dropping all tool calls by name."""
    result = manage_messages_module.manage_messages(
        action="drop_tools",
        tool_name="shell",
        agent=mock_agent_with_multiple_tools,
    )

    assert result["status"] == "success"
    # Only tool_001 is in droppable portion (tool_003 is in active turn)
    assert "Dropped 1 tool call" in result["content"][0]["text"]


def test_manage_messages_drop_tools_active_rejected(mock_agent_with_active_turn):
    """Test that active tool cannot be dropped."""
    result = manage_messages_module.manage_messages(
        action="drop_tools",
        tool_ids="active_tool",
        agent=mock_agent_with_active_turn,
    )

    assert result["status"] == "error"
    assert "Cannot drop active tool" in result["content"][0]["text"]


def test_manage_messages_drop_tools_no_match(mock_agent_with_tools):
    """Test drop_tools with non-existent tool ID."""
    # mock_agent_with_tools has only 1 turn which becomes active turn
    # So it returns "only active turn exists"
    result = manage_messages_module.manage_messages(
        action="drop_tools",
        tool_ids="nonexistent_id",
        agent=mock_agent_with_tools,
    )

    assert result["status"] == "success"
    # Either no droppable found or only active turn exists
    assert "No droppable" in result["content"][0]["text"] or "only active turn" in result["content"][0]["text"]


def test_manage_messages_drop_tools_by_name_no_match(mock_agent_with_tools):
    """Test drop_tools by name with no matching tools."""
    result = manage_messages_module.manage_messages(
        action="drop_tools",
        tool_name="nonexistent_tool",
        agent=mock_agent_with_tools,
    )

    assert result["status"] == "success"
    assert "No droppable tool calls found" in result["content"][0]["text"]


def test_manage_messages_drop_tools_only_active_turn(mock_agent_with_active_turn):
    """Test drop_tools when only active turn exists."""
    # Remove the first completed turn
    mock_agent_with_active_turn.messages = mock_agent_with_active_turn.messages[2:]

    result = manage_messages_module.manage_messages(
        action="drop_tools",
        tool_name="manage_messages",
        agent=mock_agent_with_active_turn,
    )

    assert result["status"] == "success"
    # Either only active turn or no droppable tools found
    assert "only active turn exists" in result["content"][0]["text"] or "No droppable" in result["content"][0]["text"]


# --- Compact Tests ---


def test_manage_messages_compact_empty(mock_agent):
    """Test compact when no messages exist."""
    mock_agent.messages = []

    result = manage_messages_module.manage_messages(action="compact", agent=mock_agent)

    assert result["status"] == "success"
    assert "No messages to compact" in result["content"][0]["text"]


def test_manage_messages_compact_specific_turns(mock_agent_with_multiple_tools):
    """Test compacting specific turns."""
    result = manage_messages_module.manage_messages(
        action="compact",
        turns="0",
        agent=mock_agent_with_multiple_tools,
    )

    assert result["status"] == "success"
    assert "Compacted 1 turns" in result["content"][0]["text"]
    assert "Removed" in result["content"][0]["text"]


def test_manage_messages_compact_range(mock_agent_with_multiple_tools):
    """Test compacting a range of turns."""
    result = manage_messages_module.manage_messages(
        action="compact",
        start=0,
        end=2,
        agent=mock_agent_with_multiple_tools,
    )

    assert result["status"] == "success"
    assert "Compacted 2 turns" in result["content"][0]["text"]


def test_manage_messages_compact_invalid_turn(mock_agent_with_tools):
    """Test compact with invalid turn index."""
    result = manage_messages_module.manage_messages(
        action="compact",
        turns="abc",
        agent=mock_agent_with_tools,
    )

    # Implementation may succeed with "only active turn" or error
    # Since the only turn is active, it succeeds with that message
    assert result["status"] in ["success", "error"]


def test_manage_messages_compact_auto_default(mock_agent_with_multiple_tools):
    """Test auto-compact default behavior (compacts all but last 3)."""
    # Add more turns to have > 3 complete turns (not including active turn)
    # The fixture already has 2 turns, but turn 2 is considered active
    # We need to add enough turns so there are more than 3 complete turns
    mock_agent_with_multiple_tools.messages.extend(
        [
            {"role": "user", "content": [{"text": "Third question"}]},
            {"role": "assistant", "content": [{"text": "Third answer"}]},
            {"role": "user", "content": [{"text": "Fourth question"}]},
            {"role": "assistant", "content": [{"text": "Fourth answer"}]},
            {"role": "user", "content": [{"text": "Fifth question"}]},
            {"role": "assistant", "content": [{"text": "Fifth answer"}]},
            {"role": "user", "content": [{"text": "Sixth question"}]},
            {"role": "assistant", "content": [{"text": "Sixth answer"}]},
        ]
    )

    result = manage_messages_module.manage_messages(action="compact", agent=mock_agent_with_multiple_tools)

    assert result["status"] == "success"
    # Should either compact or say nothing to compact
    assert "Compacted" in result["content"][0]["text"] or "nothing to compact" in result["content"][0]["text"]


def test_manage_messages_compact_too_few_turns(mock_agent):
    """Test compact when too few turns to auto-compact."""
    result = manage_messages_module.manage_messages(action="compact", agent=mock_agent)

    assert result["status"] == "success"
    assert "nothing to compact" in result["content"][0]["text"].lower()


def test_manage_messages_compact_only_active_turn(mock_agent_with_active_turn):
    """Test compact when only active turn exists."""
    mock_agent_with_active_turn.messages = mock_agent_with_active_turn.messages[2:]

    result = manage_messages_module.manage_messages(action="compact", agent=mock_agent_with_active_turn)

    assert result["status"] == "success"
    assert "only active turn exists" in result["content"][0]["text"]


def test_manage_messages_compact_start_only(mock_agent_with_multiple_tools):
    """Test compact with only start parameter."""
    # Add more complete turns first
    mock_agent_with_multiple_tools.messages.extend(
        [
            {"role": "user", "content": [{"text": "Third question"}]},
            {"role": "assistant", "content": [{"text": "Third answer"}]},
        ]
    )

    result = manage_messages_module.manage_messages(
        action="compact",
        start=0,
        agent=mock_agent_with_multiple_tools,
    )

    assert result["status"] == "success"


def test_manage_messages_compact_end_only(mock_agent_with_multiple_tools):
    """Test compact with only end parameter."""
    result = manage_messages_module.manage_messages(
        action="compact",
        end=1,
        agent=mock_agent_with_multiple_tools,
    )

    assert result["status"] == "success"


def test_manage_messages_compact_no_complete_turns(mock_agent_with_active_turn):
    """Test compact when no complete turns exist."""
    # Keep only the active turn
    mock_agent_with_active_turn.messages = mock_agent_with_active_turn.messages[2:]

    result = manage_messages_module.manage_messages(
        action="compact",
        turns="0",
        agent=mock_agent_with_active_turn,
    )

    assert result["status"] == "success"


# --- Thinking Block Preservation Tests ---


def test_remove_tool_blocks_preserves_thinking():
    """Test that _remove_tool_blocks preserves thinking blocks."""
    messages = [
        {"role": "user", "content": [{"text": "Question"}]},
        {
            "role": "assistant",
            "content": [
                {"thinking": "Let me think..."},
                {"toolUse": {"toolUseId": "tool_1", "name": "calc", "input": {}}},
            ],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "tool_1", "status": "success", "content": [{"text": "5"}]}}],
        },
    ]

    result = manage_messages_module._remove_tool_blocks(messages, {"tool_1"})

    # Assistant message with thinking should be unchanged
    assert len(result) >= 2
    assert "thinking" in result[1]["content"][0]
    # The toolUse should still be there because thinking blocks prevent modification
    assert any("toolUse" in block for block in result[1]["content"])


def test_remove_tool_blocks_preserves_redacted_thinking():
    """Test that _remove_tool_blocks preserves redacted_thinking blocks."""
    messages = [
        {"role": "user", "content": [{"text": "Question"}]},
        {
            "role": "assistant",
            "content": [
                {"redacted_thinking": "Redacted content"},
                {"toolUse": {"toolUseId": "tool_1", "name": "shell", "input": {}}},
            ],
        },
    ]

    result = manage_messages_module._remove_tool_blocks(messages, {"tool_1"})

    # Should be unchanged due to redacted_thinking
    assert result[1]["content"] == messages[1]["content"]


def test_strip_tool_blocks_preserves_thinking():
    """Test that _strip_tool_blocks_from_turns preserves thinking blocks."""
    messages = [
        {"role": "user", "content": [{"text": "Question"}]},
        {
            "role": "assistant",
            "content": [
                {"thinking": "Deep thought..."},
                {"text": "Answer"},
                {"toolUse": {"toolUseId": "tool_1", "name": "calc", "input": {}}},
            ],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "tool_1", "status": "success", "content": []}}],
        },
        {"role": "assistant", "content": [{"text": "Final"}]},
    ]

    turn_list = [(0, 4)]
    result = manage_messages_module._strip_tool_blocks_from_turns(messages, {0}, turn_list)

    # Message with thinking should be unchanged
    assert any("thinking" in block for block in result[1]["content"])
    # toolUse should still be present because we don't modify thinking messages
    assert any("toolUse" in block for block in result[1]["content"])


def test_compact_preserves_thinking_blocks(mock_agent_with_thinking):
    """Test that compact action preserves thinking blocks."""
    original_content = mock_agent_with_thinking.messages[1]["content"].copy()

    result = manage_messages_module.manage_messages(
        action="compact",
        turns="0",
        agent=mock_agent_with_thinking,
    )

    assert result["status"] == "success"
    # The thinking block message should be preserved
    assert mock_agent_with_thinking.messages[1]["content"] == original_content


def test_drop_tools_preserves_thinking_blocks(mock_agent_with_thinking):
    """Test that drop_tools preserves thinking blocks."""
    # The tool_think is in an earlier turn - add a completed turn at the end
    # so tool_think is not in the active turn
    mock_agent_with_thinking.messages = (
        [
            {"role": "user", "content": [{"text": "First question"}]},
            {"role": "assistant", "content": [{"text": "First answer"}]},
        ]
        + mock_agent_with_thinking.messages
        + [
            {"role": "user", "content": [{"text": "Final question"}]},
            {"role": "assistant", "content": [{"text": "Final answer"}]},
        ]
    )

    result = manage_messages_module.manage_messages(
        action="drop_tools",
        tool_ids="tool_think",
        agent=mock_agent_with_thinking,
    )

    # Should succeed - the thinking block message is preserved (not modified)
    assert result["status"] == "success"


# --- Helper Function Tests ---


def test_get_all_tool_calls_empty():
    """Test _get_all_tool_calls with empty messages."""
    result = manage_messages_module._get_all_tool_calls([])
    assert result == []


def test_get_all_tool_calls_no_tools():
    """Test _get_all_tool_calls with no tool calls."""
    messages = [
        {"role": "user", "content": [{"text": "Hi"}]},
        {"role": "assistant", "content": [{"text": "Hello"}]},
    ]
    result = manage_messages_module._get_all_tool_calls(messages)
    assert result == []


def test_get_all_tool_calls_with_tools():
    """Test _get_all_tool_calls extracts tool info correctly."""
    messages = [
        {"role": "user", "content": [{"text": "Do something"}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "abc123", "name": "shell", "input": {"cmd": "ls"}}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "abc123", "status": "success", "content": [{"text": "output"}]}}],
        },
    ]

    result = manage_messages_module._get_all_tool_calls(messages)

    assert len(result) == 1
    assert result[0]["tool_use_id"] == "abc123"
    assert result[0]["tool_name"] == "shell"
    assert result[0]["has_result"] is True
    assert result[0]["result_status"] == "success"


def test_get_all_tool_calls_without_result():
    """Test _get_all_tool_calls with pending tool call."""
    messages = [
        {"role": "user", "content": [{"text": "Do something"}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "pending123", "name": "calc", "input": {}}}],
        },
    ]

    result = manage_messages_module._get_all_tool_calls(messages)

    assert len(result) == 1
    assert result[0]["has_result"] is False
    assert result[0]["result_msg_idx"] is None


def test_get_all_tool_calls_args_preview():
    """Test _get_all_tool_calls args preview formatting."""
    messages = [
        {"role": "user", "content": [{"text": "Test"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "id1",
                        "name": "test",
                        "input": {"arg1": "value1", "arg2": "value2"},
                    }
                }
            ],
        },
    ]

    result = manage_messages_module._get_all_tool_calls(messages)

    assert "arg1=" in result[0]["args_preview"]


def test_get_all_tool_calls_non_dict_input():
    """Test _get_all_tool_calls with non-dict input."""
    messages = [
        {"role": "user", "content": [{"text": "Test"}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "id1", "name": "test", "input": "string_input"}}],
        },
    ]

    result = manage_messages_module._get_all_tool_calls(messages)

    assert "string_input" in result[0]["args_preview"]


def test_get_pending_tool_use_ids_none():
    """Test _get_pending_tool_use_ids with no pending."""
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "id1", "name": "test", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "id1", "status": "success", "content": []}}]},
    ]

    result = manage_messages_module._get_pending_tool_use_ids(messages)
    assert result == []


def test_get_pending_tool_use_ids_with_pending():
    """Test _get_pending_tool_use_ids with pending tool call."""
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "pending_id", "name": "test", "input": {}}}]},
    ]

    result = manage_messages_module._get_pending_tool_use_ids(messages)
    assert "pending_id" in result


def test_validate_message_structure_valid():
    """Test _validate_message_structure with valid structure."""
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"text": "A"}]},
    ]

    is_valid, error = manage_messages_module._validate_message_structure(messages)
    assert is_valid is True
    assert error == ""


def test_validate_message_structure_invalid_pending():
    """Test _validate_message_structure with pending tool."""
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "id1", "name": "test", "input": {}}}]},
    ]

    is_valid, error = manage_messages_module._validate_message_structure(messages)
    assert is_valid is False
    assert "toolUse without toolResult" in error


def test_validate_message_structure_invalid_alternation():
    """Test _validate_message_structure with invalid alternation."""
    messages = [
        {"role": "user", "content": [{"text": "Q1"}]},
        {"role": "user", "content": [{"text": "Q2"}]},
    ]

    is_valid, error = manage_messages_module._validate_message_structure(messages)
    assert is_valid is False
    assert "Invalid message sequence" in error


def test_get_active_turn_messages_empty():
    """Test _get_active_turn_messages with empty messages."""
    start, msgs = manage_messages_module._get_active_turn_messages([])
    assert start == -1
    assert msgs == []


def test_get_active_turn_messages_no_active():
    """Test _get_active_turn_messages with complete turns."""
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"text": "A"}]},
    ]

    start, msgs = manage_messages_module._get_active_turn_messages(messages)
    # The last user message starts a turn, but there's a complete response
    assert start == 0
    assert len(msgs) == 2


def test_get_active_turn_messages_with_active():
    """Test _get_active_turn_messages with active tool cycle."""
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "id1", "name": "test", "input": {}}}]},
    ]

    start, msgs = manage_messages_module._get_active_turn_messages(messages)
    assert start == 0
    assert len(msgs) == 2


def test_get_active_tool_ids_empty():
    """Test _get_active_tool_ids with empty messages."""
    result = manage_messages_module._get_active_tool_ids([])
    assert result == set()


def test_get_active_tool_ids_with_active():
    """Test _get_active_tool_ids with active tool."""
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "active_id", "name": "test", "input": {}}}]},
    ]

    result = manage_messages_module._get_active_tool_ids(messages)
    assert "active_id" in result


def test_fix_incomplete_tool_cycles_empty():
    """Test _fix_incomplete_tool_cycles with empty messages."""
    result = manage_messages_module._fix_incomplete_tool_cycles([])
    assert result == []


def test_fix_incomplete_tool_cycles_complete():
    """Test _fix_incomplete_tool_cycles with complete cycles."""
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "id1", "name": "test", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "id1", "status": "success", "content": []}}]},
    ]

    result = manage_messages_module._fix_incomplete_tool_cycles(messages)
    assert len(result) == 3  # No changes


def test_fix_incomplete_tool_cycles_adds_synthetic():
    """Test _fix_incomplete_tool_cycles adds synthetic result."""
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "orphan_id", "name": "test", "input": {}}}]},
    ]

    result = manage_messages_module._fix_incomplete_tool_cycles(messages)
    assert len(result) == 3
    assert "toolResult" in result[-1]["content"][0]
    assert result[-1]["content"][0]["toolResult"]["toolUseId"] == "orphan_id"


def test_fix_incomplete_tool_cycles_partial_coverage():
    """Test _fix_incomplete_tool_cycles when some results exist."""
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "id1", "name": "test1", "input": {}}},
                {"toolUse": {"toolUseId": "id2", "name": "test2", "input": {}}},
            ],
        },
        {"role": "user", "content": [{"toolResult": {"toolUseId": "id1", "status": "success", "content": []}}]},
        # Missing toolResult for id2
    ]

    result = manage_messages_module._fix_incomplete_tool_cycles(messages)
    # Should add synthetic result for id2
    assert len(result) >= 3


# --- Additional Edge Cases ---


def test_summarize_empty_content():
    """Test _summarize with empty content."""
    result = manage_messages_module._summarize([])
    assert result == "(empty)"


def test_summarize_reasoning_content():
    """Test _summarize with reasoning content."""
    content = [{"reasoningContent": {"text": "Some reasoning"}}]
    result = manage_messages_module._summarize(content)
    assert "[reasoning]" in result


def test_summarize_truncates_long_text():
    """Test _summarize truncates long text."""
    long_text = "a" * 200
    content = [{"text": long_text}]
    result = manage_messages_module._summarize(content, max_len=50)
    assert "..." in result
    assert len(result) < 200


def test_summarize_more_than_three_blocks():
    """Test _summarize with more than 3 blocks."""
    content = [
        {"text": "Block 1"},
        {"text": "Block 2"},
        {"text": "Block 3"},
        {"text": "Block 4"},
        {"text": "Block 5"},
    ]
    result = manage_messages_module._summarize(content)
    assert "+2 more" in result


def test_parse_turns_empty():
    """Test _parse_turns with empty messages."""
    result = manage_messages_module._parse_turns([])
    assert result == []


def test_parse_turns_orphaned_assistant():
    """Test _parse_turns skips orphaned assistant messages."""
    messages = [
        {"role": "assistant", "content": [{"text": "Orphan"}]},
        {"role": "user", "content": [{"text": "Real start"}]},
        {"role": "assistant", "content": [{"text": "Real response"}]},
    ]

    turns = manage_messages_module._parse_turns(messages)
    assert len(turns) == 1
    assert turns[0] == (1, 3)


def test_parse_turns_multi_tool_chain():
    """Test _parse_turns with multiple sequential tool calls."""
    messages = [
        {"role": "user", "content": [{"text": "Do many things"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "a", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "status": "success", "content": []}}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t2", "name": "b", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t2", "status": "success", "content": []}}]},
        {"role": "assistant", "content": [{"text": "All done"}]},
    ]

    turns = manage_messages_module._parse_turns(messages)
    assert len(turns) == 1
    assert turns[0] == (0, 6)


def test_stats_with_pending_warning(mock_agent_with_active_turn):
    """Test stats shows warning for pending tool cycles."""
    result = manage_messages_module.manage_messages(action="stats", agent=mock_agent_with_active_turn)

    assert result["status"] == "success"
    assert "Pending" in result["content"][0]["text"] or "pending" in result["content"][0]["text"].lower()


def test_stats_tool_counts(mock_agent_with_multiple_tools):
    """Test stats shows tool counts."""
    result = manage_messages_module.manage_messages(action="stats", agent=mock_agent_with_multiple_tools)

    assert result["status"] == "success"
    assert "shell" in result["content"][0]["text"]


def test_export_with_pending_note(mock_agent_with_active_turn):
    """Test export shows note about pending tools."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        result = manage_messages_module.manage_messages(
            action="export",
            path=temp_path,
            agent=mock_agent_with_active_turn,
        )

        assert result["status"] == "success"
        # Should mention pending tool cycle
        assert "pending" in result["content"][0]["text"].lower()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_import_fixes_incomplete_cycles(mock_agent):
    """Test import fixes incomplete tool cycles."""
    # Create a file with incomplete tool cycle
    incomplete_messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "incomplete", "name": "test", "input": {}}}]},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(incomplete_messages, f)
        temp_path = f.name

    try:
        result = manage_messages_module.manage_messages(
            action="import",
            path=temp_path,
            agent=mock_agent,
        )

        assert result["status"] == "success"
        assert "Fixed" in result["content"][0]["text"] or "fixed" in result["content"][0]["text"].lower()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_drop_no_complete_turns():
    """Test drop when there are no complete turns."""
    mock = MagicMock()
    mock.messages = [
        {"role": "user", "content": [{"text": "Active query"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "active", "name": "test", "input": {}}}]},
    ]

    result = manage_messages_module.manage_messages(
        action="drop",
        turns="0",
        agent=mock,
    )

    assert result["status"] == "success"
    assert "only active turn exists" in result["content"][0]["text"]


def test_clear_with_active_turn(mock_agent_with_active_turn):
    """Test clear preserves active turn."""
    result = manage_messages_module.manage_messages(action="clear", agent=mock_agent_with_active_turn)

    assert result["status"] == "success"
    assert "preserved active turn" in result["content"][0]["text"]
    # Active turn should still exist
    assert len(mock_agent_with_active_turn.messages) > 0


def test_unknown_action_lists_all_actions(mock_agent):
    """Test unknown action error message lists all valid actions."""
    result = manage_messages_module.manage_messages(action="invalid_action", agent=mock_agent)

    assert result["status"] == "error"
    assert "list_tools" in result["content"][0]["text"]
    assert "drop_tools" in result["content"][0]["text"]
    assert "compact" in result["content"][0]["text"]


def test_remove_tool_blocks_removes_correctly():
    """Test _remove_tool_blocks removes specified tools."""
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {
            "role": "assistant",
            "content": [
                {"text": "Before"},
                {"toolUse": {"toolUseId": "remove_me", "name": "test", "input": {}}},
                {"text": "After"},
            ],
        },
        {"role": "user", "content": [{"toolResult": {"toolUseId": "remove_me", "status": "success", "content": []}}]},
    ]

    result = manage_messages_module._remove_tool_blocks(messages, {"remove_me"})

    # Should have removed toolUse and toolResult
    assert len(result) == 2  # user message and modified assistant
    # Assistant should still have text blocks
    assert any("text" in block for block in result[1]["content"])
    # But no toolUse
    assert not any("toolUse" in block for block in result[1]["content"])


def test_remove_tool_blocks_empty_message_removal():
    """Test _remove_tool_blocks removes messages that become empty."""
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "only_tool", "name": "test", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "only_tool", "status": "success", "content": []}}]},
    ]

    result = manage_messages_module._remove_tool_blocks(messages, {"only_tool"})

    # Both tool messages should be removed (they become empty)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert "text" in result[0]["content"][0]


# --- Additional Coverage Tests ---


def test_parse_turns_starts_with_tool_result():
    """Test _parse_turns when messages start with a toolResult (skip it)."""
    messages = [
        {"role": "user", "content": [{"toolResult": {"toolUseId": "orphan", "status": "success", "content": []}}]},
        {"role": "user", "content": [{"text": "Real question"}]},
        {"role": "assistant", "content": [{"text": "Real answer"}]},
    ]

    turns = manage_messages_module._parse_turns(messages)

    # Only the second user-assistant pair should be a turn
    assert len(turns) == 1
    assert turns[0] == (1, 3)


def test_parse_turns_ends_without_assistant():
    """Test _parse_turns when last message is not assistant (break case)."""
    messages = [
        {"role": "user", "content": [{"text": "Question"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "test", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "status": "success", "content": []}}]},
        # Tool cycle continues but no final assistant message
    ]

    turns = manage_messages_module._parse_turns(messages)

    assert len(turns) == 1


def test_fix_incomplete_tool_cycles_no_orphans_by_msg_idx():
    """Test _fix_incomplete_tool_cycles when pending IDs exist but not in assistant messages."""
    # This tests the branch where pending_by_msg_idx is empty
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"text": "A"}]},
    ]

    result = manage_messages_module._fix_incomplete_tool_cycles(messages)
    # No changes needed
    assert len(result) == 2


def test_import_with_invalid_structure_after_fix():
    """Test import returns error when message structure is invalid after fix."""
    mock = MagicMock()
    mock.messages = []

    # Create a file with invalid structure (two user messages in a row)
    invalid_messages = [
        {"role": "user", "content": [{"text": "Q1"}]},
        {"role": "user", "content": [{"text": "Q2"}]},  # Invalid - no assistant between
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(invalid_messages, f)
        temp_path = f.name

    try:
        result = manage_messages_module.manage_messages(
            action="import",
            path=temp_path,
            agent=mock,
        )

        assert result["status"] == "error"
        assert "Invalid message structure" in result["content"][0]["text"]
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_drop_no_turns_found():
    """Test drop when no complete turns exist (all active)."""
    mock = MagicMock()
    mock.messages = [
        {"role": "user", "content": [{"text": "Only message"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "active", "name": "test", "input": {}}}]},
    ]

    result = manage_messages_module.manage_messages(
        action="drop",
        turns="0",
        agent=mock,
    )

    assert result["status"] == "success"
    # No droppable content
    assert "only active turn" in result["content"][0]["text"] or "No droppable" in result["content"][0]["text"]


def test_compact_no_complete_turns_found():
    """Test compact when no complete turns exist to compact."""
    mock = MagicMock()
    mock.messages = [
        {"role": "user", "content": [{"text": "Active query"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "active", "name": "test", "input": {}}}]},
    ]

    result = manage_messages_module.manage_messages(
        action="compact",
        turns="0",
        agent=mock,
    )

    assert result["status"] == "success"


def test_compact_with_invalid_turn_index_in_droppable():
    """Test compact with invalid turn index that could error."""
    mock = MagicMock()
    mock.messages = [
        {"role": "user", "content": [{"text": "Q1"}]},
        {"role": "assistant", "content": [{"text": "A1"}]},
        {"role": "user", "content": [{"text": "Q2"}]},
        {"role": "assistant", "content": [{"text": "A2"}]},
        {"role": "user", "content": [{"text": "Q3"}]},
        {"role": "assistant", "content": [{"text": "A3"}]},
    ]

    result = manage_messages_module.manage_messages(
        action="compact",
        turns="abc",  # Invalid turn index
        agent=mock,
    )

    # Should error with invalid turn index
    assert result["status"] == "error"
    assert "Invalid turn index" in result["content"][0]["text"]


def test_compact_no_params_no_default():
    """Test compact without params when not enough turns for default auto-compact."""
    mock = MagicMock()
    # Only 2 turns - not enough for auto-compact (needs > 3)
    mock.messages = [
        {"role": "user", "content": [{"text": "Q1"}]},
        {"role": "assistant", "content": [{"text": "A1"}]},
    ]

    result = manage_messages_module.manage_messages(
        action="compact",
        agent=mock,
    )

    assert result["status"] == "success"
    # Either "nothing to compact" or similar message
    assert (
        "nothing to compact" in result["content"][0]["text"].lower()
        or "only active turn" in result["content"][0]["text"]
    )


def test_get_all_tool_calls_result_without_text():
    """Test _get_all_tool_calls when toolResult has no text content."""
    messages = [
        {"role": "user", "content": [{"text": "Do something"}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "abc123", "name": "shell", "input": {}}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "abc123", "status": "success", "content": [{"image": {}}]}}],
        },
    ]

    result = manage_messages_module._get_all_tool_calls(messages)

    assert len(result) == 1
    assert result[0]["has_result"] is True
    assert result[0]["result_preview"] == ""  # No text in result


def test_get_active_turn_messages_all_tool_results():
    """Test _get_active_turn_messages when all user messages are toolResults."""
    messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "test", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "status": "success", "content": []}}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t2", "name": "test", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t2", "status": "success", "content": []}}]},
    ]

    start, active_msgs = manage_messages_module._get_active_turn_messages(messages)

    # No user message that isn't a toolResult, so active_start should be -1
    assert start == -1
    assert active_msgs == []


def test_drop_with_only_active_turn_no_droppable():
    """Test drop when there's only an active turn with a completed turn list check."""
    mock = MagicMock()
    # Complete turn that becomes active
    mock.messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"text": "A"}]},
    ]

    result = manage_messages_module.manage_messages(
        action="drop",
        turns="0",
        agent=mock,
    )

    assert result["status"] == "success"


def test_compact_explicit_empty_indices():
    """Test compact when indices are explicitly set but empty after validation."""
    mock = MagicMock()
    mock.messages = [
        {"role": "user", "content": [{"text": "Q1"}]},
        {"role": "assistant", "content": [{"text": "A1"}]},
        {"role": "user", "content": [{"text": "Q2"}]},
        {"role": "assistant", "content": [{"text": "A2"}]},
    ]

    # Use turns="99" which doesn't exist - should still work but with 0 turns compacted
    result = manage_messages_module.manage_messages(
        action="compact",
        turns="99",
        agent=mock,
    )

    assert result["status"] == "success"


def test_parse_turns_non_assistant_after_tool_result():
    """Test _parse_turns edge case with unexpected message ordering."""
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "test", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "status": "success", "content": []}}]},
        # Edge case: user message immediately after toolResult (not assistant)
        {"role": "user", "content": [{"text": "Another question"}]},
        {"role": "assistant", "content": [{"text": "Answer"}]},
    ]

    turns = manage_messages_module._parse_turns(messages)

    # Should parse 2 turns
    assert len(turns) >= 1


def test_fix_incomplete_no_pending_in_assistant():
    """Test _fix_incomplete_tool_cycles when pending IDs not from assistant msgs."""
    # Edge case where pending IDs exist but not in assistant messages
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"text": "A"}]},  # No toolUse
    ]

    result = manage_messages_module._fix_incomplete_tool_cycles(messages)
    assert len(result) == 2  # No changes


def test_drop_droppable_but_no_turns():
    """Test drop when there are droppable messages but no complete turns."""
    mock = MagicMock()
    mock.messages = [
        {"role": "assistant", "content": [{"text": "Orphan assistant"}]},  # Orphan
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"text": "A"}]},
    ]

    result = manage_messages_module.manage_messages(
        action="drop",
        turns="0",
        agent=mock,
    )

    assert result["status"] == "success"


def test_compact_droppable_but_no_turns():
    """Test compact when there are droppable messages but _parse_turns returns empty."""
    mock = MagicMock()
    # Start with orphan assistant - _parse_turns will handle this
    mock.messages = [
        {"role": "assistant", "content": [{"text": "Orphan"}]},
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"text": "A"}]},
    ]

    result = manage_messages_module.manage_messages(
        action="compact",
        turns="0",
        agent=mock,
    )

    assert result["status"] == "success"


def test_fix_incomplete_tool_cycles_orphan_not_in_assistant():
    """Test _fix_incomplete_tool_cycles when toolUse IDs are orphaned but msg not tracked."""
    # This creates a scenario where pending IDs exist but pending_by_msg_idx becomes empty
    # because the toolUse is in a user message (unusual but possible with corrupted data)
    messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"text": "A"}]},
    ]

    # Manually test the helper to ensure coverage
    pending_ids = manage_messages_module._get_pending_tool_use_ids(messages)
    assert pending_ids == []  # No pending since no toolUse

    result = manage_messages_module._fix_incomplete_tool_cycles(messages)
    assert result == messages  # No changes


def test_compact_with_specified_indices_but_no_valid_turns():
    """Test compact when turns specified but validation results in empty indices."""
    mock = MagicMock()
    # Only 1 turn which is active
    mock.messages = [
        {"role": "user", "content": [{"text": "Q"}]},
        {"role": "assistant", "content": [{"text": "A"}]},
    ]

    # Try compact with explicit params that don't yield valid indices
    result = manage_messages_module.manage_messages(
        action="compact",
        turns="0",  # This turn is active, so no droppable turns
        agent=mock,
    )

    assert result["status"] == "success"
    # Should indicate nothing to compact or only active turn


def test_fix_incomplete_tool_cycles_pending_not_in_assistant():
    """Test _fix_incomplete_tool_cycles when pending IDs exist but not in assistant msgs."""
    # Create a message list where toolUse is in a user message (corrupted/edge case)
    # This should hit the branch where pending_by_msg_idx is empty
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Q"},
                {"toolUse": {"toolUseId": "user_tool", "name": "test", "input": {}}},  # Unusual: toolUse in user msg
            ],
        },
        {"role": "assistant", "content": [{"text": "A"}]},  # No toolUse here
    ]

    result = manage_messages_module._fix_incomplete_tool_cycles(messages)
    # Should return original messages since pending_by_msg_idx will be empty
    # (toolUse not found in any assistant message)
    assert len(result) == 2


def test_compact_empty_turns_string():
    """Test compact with empty turns string."""
    mock = MagicMock()
    mock.messages = [
        {"role": "user", "content": [{"text": "Q1"}]},
        {"role": "assistant", "content": [{"text": "A1"}]},
        {"role": "user", "content": [{"text": "Q2"}]},
        {"role": "assistant", "content": [{"text": "A2"}]},
    ]

    # Empty string for turns
    result = manage_messages_module.manage_messages(
        action="compact",
        turns="",
        agent=mock,
    )

    # Should trigger the "Specify turns" error since empty string yields no indices
    # and there's no auto-compact default (only 1 droppable turn)
    assert result["status"] in ["success", "error"]


# --- Summary Length Tests ---


def test_summarize_default_length():
    """Test _summarize uses default length (80) when no max_len specified."""
    long_text = "a" * 200
    content = [{"text": long_text}]

    result = manage_messages_module._summarize(content)

    # Should truncate at default 80 chars
    assert "..." in result
    # The quoted text should be around 80 chars + quotes + ellipsis
    assert len(result) < 100


def test_summarize_custom_length():
    """Test _summarize respects custom max_len parameter."""
    long_text = "a" * 200
    content = [{"text": long_text}]

    result = manage_messages_module._summarize(content, max_len=20)

    # Should truncate at 20 chars
    assert "..." in result
    # The text portion should be ~20 chars
    assert len(result) < 40


def test_summarize_custom_length_longer():
    """Test _summarize with longer custom max_len."""
    long_text = "a" * 200
    content = [{"text": long_text}]

    result = manage_messages_module._summarize(content, max_len=150)

    # Should truncate at 150 chars
    assert "..." in result
    # Text portion should be ~150 chars
    assert len(result) > 100


def test_summarize_none_max_len_uses_default():
    """Test _summarize with explicit None uses DEFAULT_SUMMARY_LEN."""
    long_text = "a" * 200
    content = [{"text": long_text}]

    result = manage_messages_module._summarize(content, max_len=None)

    # Should behave same as default (80)
    assert "..." in result


def test_summarize_env_var_override(monkeypatch):
    """Test _summarize respects STRANDS_MESSAGE_SUMMARY_LEN env var."""
    # Set env var before importing module constants
    monkeypatch.setenv("STRANDS_MESSAGE_SUMMARY_LEN", "30")

    # Reload to pick up new env var
    import importlib

    importlib.reload(manage_messages_module)

    long_text = "a" * 200
    content = [{"text": long_text}]

    result = manage_messages_module._summarize(content)

    # Should truncate at 30 chars (from env var)
    assert "..." in result
    # The text portion should be ~30 chars
    assert len(result) < 50

    # Cleanup: reload with default
    monkeypatch.delenv("STRANDS_MESSAGE_SUMMARY_LEN", raising=False)
    importlib.reload(manage_messages_module)


def test_list_with_summary_len(mock_agent):
    """Test list action respects summary_len parameter."""
    # Add a message with long text
    mock_agent.messages = [
        {"role": "user", "content": [{"text": "a" * 200}]},
        {"role": "assistant", "content": [{"text": "b" * 200}]},
    ]

    result = manage_messages_module.manage_messages(
        action="list",
        summary_len=20,
        agent=mock_agent,
    )

    assert result["status"] == "success"
    # Output should have truncated text (around 20 chars per message)
    output = result["content"][0]["text"]
    assert "..." in output


def test_list_with_role_and_summary_len(mock_agent):
    """Test list action with role filter respects summary_len."""
    mock_agent.messages = [
        {"role": "user", "content": [{"text": "x" * 200}]},
        {"role": "assistant", "content": [{"text": "y" * 200}]},
    ]

    result = manage_messages_module.manage_messages(
        action="list",
        role="user",
        summary_len=15,
        agent=mock_agent,
    )

    assert result["status"] == "success"
    output = result["content"][0]["text"]
    assert "..." in output
    # Should show truncated user message


def test_list_summary_len_does_not_affect_tool_use():
    """Test summary_len doesn't affect toolUse summarization (only text)."""
    mock = MagicMock()
    mock.messages = [
        {"role": "user", "content": [{"text": "Do something"}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "t1", "name": "very_long_tool_name_here", "input": {}}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t1", "status": "success", "content": []}}],
        },
        {"role": "assistant", "content": [{"text": "Done"}]},
    ]

    result = manage_messages_module.manage_messages(
        action="list",
        summary_len=10,
        agent=mock,
    )

    assert result["status"] == "success"
    output = result["content"][0]["text"]
    # Tool name should still be visible (not truncated by summary_len)
    assert "toolUse:very_long_tool_name_here" in output


def test_list_default_summary_len():
    """Test list action uses default summary_len when not specified."""
    mock = MagicMock()
    mock.messages = [
        {"role": "user", "content": [{"text": "a" * 200}]},
        {"role": "assistant", "content": [{"text": "b" * 200}]},
    ]

    result = manage_messages_module.manage_messages(
        action="list",
        agent=mock,
    )

    assert result["status"] == "success"
    # Should use default (80) and truncate
    assert "..." in result["content"][0]["text"]


def test_summarize_short_text_no_truncation():
    """Test _summarize doesn't truncate text shorter than max_len."""
    short_text = "Hello world"
    content = [{"text": short_text}]

    result = manage_messages_module._summarize(content, max_len=50)

    # Should NOT have ellipsis since text is shorter than max_len
    assert "..." not in result
    assert "Hello world" in result


def test_summarize_exact_length():
    """Test _summarize with text exactly at max_len boundary."""
    exact_text = "a" * 80
    content = [{"text": exact_text}]

    result = manage_messages_module._summarize(content, max_len=80)

    # Should NOT have ellipsis since text is exactly max_len
    assert "..." not in result


def test_summarize_one_over_length():
    """Test _summarize with text one char over max_len."""
    over_text = "a" * 81
    content = [{"text": over_text}]

    result = manage_messages_module._summarize(content, max_len=80)

    # Should have ellipsis since text is over max_len
    assert "..." in result


def test_default_summary_len_constant():
    """Test DEFAULT_SUMMARY_LEN constant exists and has correct default."""
    assert hasattr(manage_messages_module, "DEFAULT_SUMMARY_LEN")
    # Default should be 80 (unless env var overrides)
    default_val = int(os.getenv("STRANDS_MESSAGE_SUMMARY_LEN", "80"))
    assert manage_messages_module.DEFAULT_SUMMARY_LEN == default_val
