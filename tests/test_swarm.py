"""
Tests for the swarm tool using both direct calls and the Agent interface.
"""

import io
from unittest.mock import patch

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
def mock_request_state():
    """Create a mock request state dictionary."""
    return {}


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
def test_swarm_agent_process_task(mock_use_llm, shared_memory, mock_llm_response):
    """Test swarm agent processing a task."""
    mock_use_llm.return_value = mock_llm_response

    # Create a swarm agent
    agent = SwarmAgent("test_agent", "Test system prompt", shared_memory)

    # Process a task
    result = agent.process_task("Test task", {})

    # Verify use_llm was called
    mock_use_llm.assert_called_once()

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
def test_swarm_agent_error_handling(mock_use_llm, shared_memory):
    """Test swarm agent error handling."""
    # Mock use_llm to raise an exception
    mock_use_llm.side_effect = Exception("Test exception")

    # Create a swarm agent
    agent = SwarmAgent("test_agent", "Test system prompt", shared_memory)

    # Process a task
    result = agent.process_task("Test task", {})

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
def test_swarm_process_phase(mock_use_llm, mock_llm_response):
    """Test processing a phase in a swarm."""
    mock_use_llm.return_value = mock_llm_response

    # Create a swarm
    test_swarm = Swarm("Test task", "collaborative")

    # Add agents
    test_swarm.add_agent("agent1", "System prompt 1")
    test_swarm.add_agent("agent2", "System prompt 2")

    # Process a phase
    phase_results = test_swarm.process_phase({})

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
        "phases": 2,  # Add phases field
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
def test_swarm_error_handling(mock_swarm_class, mock_request_state):
    """Test error handling in the swarm tool."""
    # Mock Swarm class to raise an exception
    mock_swarm_class.side_effect = Exception("Test exception in swarm")

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {"task": "Test swarm task", "swarm_size": 2, "coordination_pattern": "collaborative"},
    }

    result = swarm.swarm(
        tool=tool_use,
        system_prompt="Test system prompt",
        inference_config={},
        messages=[],
        tool_config={},
        request_state=mock_request_state,
    )

    # Verify error status
    assert result["status"] == "error"
    assert "Test exception in swarm" in result["content"][0]["text"]


@patch("strands_tools.swarm.use_llm")
def test_swarm_via_agent(mock_use_llm, agent, mock_llm_response):
    """Test swarm via the agent interface."""
    mock_use_llm.return_value = mock_llm_response

    # Call the swarm tool via agent
    result = agent.tool.swarm(task="Test task via agent", swarm_size=2, coordination_pattern="hybrid")

    # Extract result text
    result_text = extract_result_text(result)

    # Verify the result contains expected information
    assert "Swarm Status" in result_text
    assert "Test task via agent" in result_text
    assert "hybrid" in result_text


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
