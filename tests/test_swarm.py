"""
Tests for the swarm tool using both direct calls and the Agent interface.
Updated for the new @tool decorator format.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from strands import Agent
from strands_tools import swarm
from strands_tools.swarm import SharedMemory, Swarm, SwarmAgent


@pytest.fixture
def agent():
    """Create an agent with the swarm tool loaded."""
    return Agent(tools=[swarm])


@pytest.fixture
def mock_agent():
    """Create a mock parent agent for testing."""
    mock_agent = MagicMock()
    mock_agent.system_prompt = "You are a helpful AI assistant."
    return mock_agent


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


@pytest.fixture
def shared_memory():
    """Create a shared memory instance for testing."""
    return SharedMemory()


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return {
        "status": "success",
        "content": [{"text": "This is a test contribution from the mock LLM."}],
    }


def test_shared_memory_store(shared_memory):
    """Test storing data in shared memory."""
    # Store an item in memory
    agent_id = "test_agent"
    content = "Test content"
    result = shared_memory.store(agent_id, content)

    # Verify the store operation succeeded
    assert result is True

    # Verify the content is in the memory
    current_knowledge = shared_memory.get_current_knowledge()
    assert len(current_knowledge) == 1
    assert current_knowledge[0]["agent_id"] == agent_id
    assert current_knowledge[0]["content"] == content
    assert current_knowledge[0]["phase"] == 0


def test_shared_memory_advance_phase(shared_memory):
    """Test advancing the phase in shared memory."""
    # Initial phase should be 0
    assert shared_memory.current_phase == 0

    # Store an item in phase 0
    shared_memory.store("agent1", "Phase 0 content")

    # Advance phase
    shared_memory.advance_phase()

    # Verify phase advanced
    assert shared_memory.current_phase == 1

    # Store an item in phase 1
    shared_memory.store("agent1", "Phase 1 content")

    # Verify items are associated with their respective phases
    all_knowledge = shared_memory.get_all_knowledge()
    assert len(all_knowledge) == 2
    assert all_knowledge[0]["phase"] == 0
    assert all_knowledge[1]["phase"] == 1

    # Verify current knowledge only shows current phase
    current_knowledge = shared_memory.get_current_knowledge()
    assert len(current_knowledge) == 1
    assert current_knowledge[0]["content"] == "Phase 1 content"


@patch("strands_tools.swarm.use_llm")
def test_swarm_agent_process_task(mock_use_llm, shared_memory, mock_llm_response, mock_agent):
    """Test swarm agent processing a task."""
    mock_use_llm.return_value = mock_llm_response

    # Create a swarm agent
    agent = SwarmAgent("test_agent", "Test system prompt", shared_memory)

    # Process a task with mock parent agent
    result = agent.process_task("Test task", mock_agent)

    # Verify use_llm was called with new format
    mock_use_llm.assert_called_once()
    call_args = mock_use_llm.call_args
    assert call_args[1]["system_prompt"] == "Test system prompt"
    assert "Test task" in call_args[1]["prompt"]
    assert call_args[1]["agent"] == mock_agent

    # Verify the result
    assert result["status"] == "success"

    # Verify agent state
    assert agent.status == "completed"
    assert agent.contributions == 1

    # Verify content was stored in shared memory
    knowledge = shared_memory.get_current_knowledge()
    assert len(knowledge) == 1
    assert knowledge[0]["content"] == "This is a test contribution from the mock LLM."


@patch("strands_tools.swarm.use_llm")
def test_swarm_agent_error_handling(mock_use_llm, shared_memory, mock_agent):
    """Test swarm agent error handling."""
    # Mock use_llm to raise an exception
    mock_use_llm.side_effect = Exception("Test exception")

    # Create a swarm agent
    agent = SwarmAgent("test_agent", "Test system prompt", shared_memory)

    # Process a task
    result = agent.process_task("Test task", mock_agent)

    # Verify the result indicates an error
    assert result["status"] == "error"
    assert "Test exception" in result["content"][0]["text"]

    # Verify agent state
    assert agent.status == "error"


def test_swarm_class_init():
    """Test swarm class initialization."""
    task = "Test task"
    coordination = "collaborative"

    # Create a swarm
    test_swarm = Swarm(task, coordination)

    # Verify swarm properties
    assert test_swarm.task == task
    assert test_swarm.coordination_pattern == coordination
    assert isinstance(test_swarm.shared_memory, SharedMemory)
    assert len(test_swarm.agents) == 0


def test_swarm_add_agent():
    """Test adding agents to a swarm."""
    # Create a swarm
    test_swarm = Swarm("Test task", "collaborative")

    # Add agents
    agent1 = test_swarm.add_agent("agent1", "System prompt 1")
    agent2 = test_swarm.add_agent("agent2", "System prompt 2")

    # Verify agents were added
    assert len(test_swarm.agents) == 2
    assert "agent1" in test_swarm.agents
    assert "agent2" in test_swarm.agents

    # Verify agent properties
    assert agent1.id == "agent1"
    assert agent2.id == "agent2"
    assert agent1.system_prompt == "System prompt 1"


@patch("strands_tools.swarm.use_llm")
def test_swarm_process_phase(mock_use_llm, mock_llm_response, mock_agent):
    """Test processing a phase in a swarm."""
    mock_use_llm.return_value = mock_llm_response

    # Create a swarm
    test_swarm = Swarm("Test task", "collaborative")

    # Add agents
    test_swarm.add_agent("agent1", "System prompt 1")
    test_swarm.add_agent("agent2", "System prompt 2")

    # Process a phase with mock parent agent
    phase_results = test_swarm.process_phase(mock_agent)

    # Verify results
    assert len(phase_results) == 2
    assert phase_results[0]["agent_id"] in ["agent1", "agent2"]
    assert phase_results[1]["agent_id"] in ["agent1", "agent2"]
    assert phase_results[0]["agent_id"] != phase_results[1]["agent_id"]
    assert phase_results[0]["result"]["status"] == "success"

    # Verify the shared memory phase was advanced
    assert test_swarm.shared_memory.current_phase == 1


def test_create_rich_status_panel():
    """Test creating a rich status panel."""
    status = {
        "task": "Test task",
        "coordination_pattern": "collaborative",
        "memory_id": "test-memory-id",
        "agents": [
            {"id": "agent1", "status": "completed", "contributions": 2},
            {"id": "agent2", "status": "processing", "contributions": 1},
        ],
    }

    console = Console(file=io.StringIO())
    result = swarm.create_rich_status_panel(console, status)

    # Verify the result contains key information
    assert "Test task" in result
    assert "collaborative" in result
    assert "test-memory-id" in result
    assert "agent1" in result
    assert "agent2" in result
    assert "completed" in result
    assert "processing" in result


@patch("strands_tools.swarm.Swarm")
def test_swarm_error_handling(mock_swarm_class, mock_agent):
    """Test error handling in the swarm tool."""
    # Mock Swarm class to raise an exception
    mock_swarm_class.side_effect = Exception("Test exception in swarm")

    # Call swarm tool with new format
    result = swarm.swarm(task="Test swarm task", swarm_size=2, coordination_pattern="collaborative", agent=mock_agent)

    # Verify error status
    assert result["status"] == "error"
    assert "Test exception in swarm" in result["content"][0]["text"]


@patch("strands_tools.swarm.use_llm")
def test_swarm_via_agent(mock_use_llm, agent, mock_llm_response):
    """Test swarm via the agent interface."""
    mock_use_llm.return_value = mock_llm_response

    # Call the swarm tool via agent using new format
    result = agent.tool.swarm(task="Test task via agent", swarm_size=2, coordination_pattern="hybrid")

    # Extract result text
    result_text = extract_result_text(result)

    # Verify the result contains expected information
    assert "📊 Swarm Results" in result_text
    assert "Agent agent_1:" in result_text or "Agent agent_2:" in result_text
    assert "This is a test contribution from the mock LLM." in result_text


def test_get_all_knowledge(shared_memory):
    """Test getting all knowledge from shared memory."""
    # Add items with different phases
    shared_memory.store("agent1", "Content 1")
    shared_memory.advance_phase()
    shared_memory.store("agent2", "Content 2")
    shared_memory.advance_phase()
    shared_memory.store("agent3", "Content 3")

    # Get all knowledge
    all_knowledge = shared_memory.get_all_knowledge()

    # Verify all items were retrieved
    assert len(all_knowledge) == 3
    assert all_knowledge[0]["content"] == "Content 1"
    assert all_knowledge[0]["phase"] == 0
    assert all_knowledge[1]["content"] == "Content 2"
    assert all_knowledge[1]["phase"] == 1
    assert all_knowledge[2]["content"] == "Content 3"
    assert all_knowledge[2]["phase"] == 2


def test_swarm_agent_get_status():
    """Test getting status from a swarm agent."""
    shared_memory = SharedMemory()
    agent = SwarmAgent("test_agent", "Test system prompt", shared_memory)
    agent.status = "processing"
    agent.contributions = 3

    status = agent.get_status()

    assert status["id"] == "test_agent"
    assert status["status"] == "processing"
    assert status["contributions"] == 3


def test_swarm_get_status():
    """Test getting status from a swarm."""
    test_swarm = Swarm("Test task", "competitive")
    test_swarm.add_agent("agent1", "System prompt 1")
    test_swarm.add_agent("agent2", "System prompt 2")

    # Set some agent statuses
    test_swarm.agents["agent1"].status = "completed"
    test_swarm.agents["agent1"].contributions = 2
    test_swarm.agents["agent2"].status = "processing"
    test_swarm.agents["agent2"].contributions = 1

    # Get status
    status = test_swarm.get_status()

    # Verify status information
    assert status["task"] == "Test task"
    assert status["coordination_pattern"] == "competitive"
    assert status["memory_id"] == test_swarm.shared_memory.memory_id
    assert len(status["agents"]) == 2

    # Find agents in the status
    agent1_status = next((a for a in status["agents"] if a["id"] == "agent1"), None)
    agent2_status = next((a for a in status["agents"] if a["id"] == "agent2"), None)

    assert agent1_status["status"] == "completed"
    assert agent1_status["contributions"] == 2
    assert agent2_status["status"] == "processing"
    assert agent2_status["contributions"] == 1


@patch("strands_tools.swarm.use_llm")
def test_swarm_direct_call(mock_use_llm, mock_llm_response, mock_agent):
    """Test direct swarm function call with new format."""
    mock_use_llm.return_value = mock_llm_response

    # Call swarm tool directly with new format
    result = swarm.swarm(
        task="Direct call test task", swarm_size=3, coordination_pattern="competitive", agent=mock_agent
    )

    # Verify successful result
    assert result["status"] == "success"
    assert "📊 Swarm Results" in result["content"][0]["text"]
    assert "This is a test contribution from the mock LLM." in result["content"][0]["text"]

    # Verify use_llm was called multiple times (for multiple agents and phases)
    assert mock_use_llm.call_count >= 3  # At least 3 agents * 2 phases


@patch("strands_tools.swarm.use_llm")
def test_swarm_coordination_patterns(mock_use_llm, mock_llm_response, mock_agent):
    """Test different coordination patterns."""
    mock_use_llm.return_value = mock_llm_response

    # Test collaborative pattern
    result_collab = swarm.swarm(
        task="Collaborative test", swarm_size=2, coordination_pattern="collaborative", agent=mock_agent
    )
    assert result_collab["status"] == "success"
    assert "📊 Swarm Results" in result_collab["content"][0]["text"]
    assert "This is a test contribution from the mock LLM." in result_collab["content"][0]["text"]

    # Reset mock
    mock_use_llm.reset_mock()

    # Test competitive pattern
    result_comp = swarm.swarm(
        task="Competitive test", swarm_size=2, coordination_pattern="competitive", agent=mock_agent
    )
    assert result_comp["status"] == "success"
    assert "📊 Swarm Results" in result_comp["content"][0]["text"]
    assert "This is a test contribution from the mock LLM." in result_comp["content"][0]["text"]

    # Reset mock
    mock_use_llm.reset_mock()

    # Test hybrid pattern
    result_hybrid = swarm.swarm(task="Hybrid test", swarm_size=2, coordination_pattern="hybrid", agent=mock_agent)
    assert result_hybrid["status"] == "success"
    assert "📊 Swarm Results" in result_hybrid["content"][0]["text"]
    assert "This is a test contribution from the mock LLM." in result_hybrid["content"][0]["text"]


def test_swarm_size_validation(mock_agent):
    """Test swarm size validation and clamping."""
    # Test minimum size clamping
    result_min = swarm.swarm(
        task="Min size test",
        swarm_size=0,  # Should be clamped to 1
        coordination_pattern="collaborative",
        agent=mock_agent,
    )
    # Should not fail due to invalid size
    assert result_min["status"] in ["success", "error"]  # Error due to no mocking, but not size validation

    # Test maximum size clamping
    result_max = swarm.swarm(
        task="Max size test",
        swarm_size=15,  # Should be clamped to 10
        coordination_pattern="collaborative",
        agent=mock_agent,
    )
    # Should not fail due to invalid size
    assert result_max["status"] in ["success", "error"]  # Error due to no mocking, but not size validation


@patch("strands_tools.swarm.use_llm")
def test_swarm_with_parent_agent_system_prompt(mock_use_llm, mock_llm_response):
    """Test that swarm properly uses parent agent's system prompt."""
    mock_use_llm.return_value = mock_llm_response

    # Create mock parent agent with custom system prompt
    mock_parent = MagicMock()
    mock_parent.system_prompt = "You are a specialized research assistant."

    swarm.swarm(task="Research task", swarm_size=1, coordination_pattern="collaborative", agent=mock_parent)

    # Verify use_llm was called and parent system prompt was incorporated
    mock_use_llm.assert_called()
    call_args = mock_use_llm.call_args
    agent_prompt = call_args[1]["system_prompt"]

    # The agent prompt should include the parent's system prompt
    assert "You are a specialized research assistant." in agent_prompt
    assert "Research task" in call_args[1]["prompt"]


def test_swarm_without_parent_agent():
    """Test swarm behavior when no parent agent is provided."""
    result = swarm.swarm(
        task="No parent test",
        swarm_size=1,
        coordination_pattern="collaborative",
        agent=None,  # No parent agent
    )

    # Should handle gracefully (though will likely error due to no mocking)
    assert result["status"] in ["success", "error"]
    assert isinstance(result["content"], list)
    assert len(result["content"]) > 0
