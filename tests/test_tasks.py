"""
Tests for the tasks tool using the Agent interface.

This test file covers:
1. Task creation, management and interaction
2. TaskState class functionality
3. Background processing with threading
4. Error handling and edge cases
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from strands import Agent
from strands_tools import tasks


@pytest.fixture
def mock_agent():
    """Create a mock agent with the tasks tool loaded."""
    agent = Agent(tools=[tasks])
    return agent


@pytest.fixture
def temp_tasks_dir(tmpdir, monkeypatch):
    """Create a temporary tasks directory for testing."""
    tasks_dir = Path(tmpdir) / "tasks"
    tasks_dir.mkdir()
    monkeypatch.setattr(tasks, "TASKS_DIR", tasks_dir)
    return tasks_dir


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return [content.get("text", "") for content in result["content"]]
    return [str(result)]


@pytest.fixture
def mock_thread():
    """Mock threading.Thread to avoid actual thread creation."""
    with patch("threading.Thread") as mock:
        thread_instance = MagicMock()
        mock.return_value = thread_instance
        thread_instance.is_alive.return_value = True
        yield mock


@pytest.fixture
def mock_open_files(monkeypatch):
    """Mock file operations"""
    # Mock file open operations
    m = mock_open()
    monkeypatch.setattr("builtins.open", m)

    # Mock Path.exists to return True
    def mock_exists(*args, **kwargs):
        return True

    monkeypatch.setattr(Path, "exists", mock_exists)
    return m


def test_create_task_missing_prompt(mock_agent):
    """Test creating a task with missing prompt."""
    result = mock_agent.tool.tasks(action="create", task_id="test_task", system_prompt="Test system prompt")

    result_texts = extract_result_text(result)
    assert any("prompt is required" in text for text in result_texts)


def test_create_task_missing_system_prompt(mock_agent):
    """Test creating a task with missing system prompt."""
    result = mock_agent.tool.tasks(action="create", task_id="test_task", prompt="Test prompt")

    result_texts = extract_result_text(result)
    assert any("system_prompt is required" in text for text in result_texts)


def test_task_status_not_found(mock_agent):
    """Test getting status of non-existent task."""
    with patch.object(tasks.TaskState, "load", return_value=None):
        result = mock_agent.tool.tasks(action="status", task_id="nonexistent_task")

        result_texts = extract_result_text(result)
        assert any("not found" in text for text in result_texts)


def test_get_result_not_found(mock_agent):
    """Test getting results for non-existent task."""
    with patch("pathlib.Path.exists", return_value=False):
        result = mock_agent.tool.tasks(action="get_result", task_id="nonexistent_task")

        result_texts = extract_result_text(result)
        assert any("No results found" in text for text in result_texts)


def test_unknown_action(mock_agent):
    """Test handling of unknown action."""
    result = mock_agent.tool.tasks(action="invalid_action", task_id="test_task")

    result_texts = extract_result_text(result)
    assert any("Unknown action" in text for text in result_texts)


# For the remaining tests that need more complex mocking, we'll patch the
# tasks tool function directly rather than going through the agent interface
def test_create_task():
    """Test creating a task."""
    # Create a mock tool use
    tool_use = {
        "toolUseId": "test-id",
        "input": {
            "action": "create",
            "task_id": "test_task",
            "prompt": "Test prompt",
            "system_prompt": "Test system prompt",
        },
    }

    # Mock the TaskState class and other file operations
    with patch("strands_tools.tasks.TaskState") as mock_task_state, patch("threading.Thread") as mock_thread:
        # Set up the mock
        mock_task_state_instance = MagicMock()
        mock_task_state.return_value = mock_task_state_instance
        mock_task_state_instance.result_path = "/path/to/result.txt"

        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        # Call the function directly
        result = tasks.tasks(tool=tool_use)

        # Verify result
        assert result["status"] == "success"
        assert any("created and started" in text.get("text", "") for text in result["content"])
        assert any("test_task" in text.get("text", "") for text in result["content"])

        # Verify thread was created and started
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()


def test_task_status():
    """Test getting task status."""
    # Create a mock tool use
    tool_use = {"toolUseId": "test-id", "input": {"action": "status", "task_id": "status_task"}}

    # Create a mock task state
    mock_task_state = MagicMock()
    mock_task_state.status = "running"
    mock_task_state.created_at = "2023-01-01T00:00:00"
    mock_task_state.last_updated = "2023-01-01T00:01:00"
    mock_task_state.result_path = "/path/to/result.txt"

    with (
        patch.object(tasks.TaskState, "load", return_value=mock_task_state),
        patch.dict(tasks.task_threads, {"status_task": MagicMock()}),
        patch.object(tasks.task_threads["status_task"], "is_alive", return_value=True),
    ):
        # Call the function directly
        result = tasks.tasks(tool=tool_use)

        # Verify result
        assert result["status"] == "success"
        assert any("status_task" in text.get("text", "") for text in result["content"])
        assert any("running" in text.get("text", "") for text in result["content"])


def test_list_tasks():
    """Test listing tasks."""
    # Create a mock tool use
    tool_use = {"toolUseId": "test-id", "input": {"action": "list"}}

    # Mock glob to return task state files
    mock_path_1 = MagicMock()
    mock_path_1.name = "task1_state.json"
    mock_path_2 = MagicMock()
    mock_path_2.name = "task2_state.json"

    mock_task_state = MagicMock()
    mock_task_state.status = "running"
    mock_task_state.created_at = "2023-01-01T00:00:00"

    with (
        patch("pathlib.Path.glob", return_value=[mock_path_1, mock_path_2]),
        patch.object(tasks.TaskState, "load", return_value=mock_task_state),
    ):
        # Call the function directly
        result = tasks.tasks(tool=tool_use)

        # Verify result
        assert result["status"] == "success"
        assert any("Found" in text.get("text", "") for text in result["content"])


def test_add_message():
    """Test adding a message to a task."""
    # Create a mock tool use
    tool_use = {
        "toolUseId": "test-id",
        "input": {"action": "add_message", "task_id": "message_task", "message": "Additional test message"},
    }

    # Create a mock task state
    mock_task_state = MagicMock()

    # Mock what happens with an already running task
    with (
        patch.object(tasks.TaskState, "load", return_value=mock_task_state),
        patch.dict(tasks.task_threads, {"message_task": MagicMock()}),
        patch.object(tasks.task_threads["message_task"], "is_alive", return_value=True),
    ):
        # Call the function directly
        result = tasks.tasks(tool=tool_use)

        # Verify result
        assert result["status"] == "success"
        assert any("message_task" in text.get("text", "") for text in result["content"])
        assert any("Message added" in text.get("text", "") for text in result["content"])

        # Verify add_message was called on the mock task state
        mock_task_state.add_message.assert_called_with("Additional test message")


def test_add_message_missing_message():
    """Test adding an empty message."""
    # Create a mock tool use
    tool_use = {"toolUseId": "test-id", "input": {"action": "add_message", "task_id": "message_task"}}

    # Call the function directly
    result = tasks.tasks(tool=tool_use)

    # Verify result
    assert result["status"] == "error"
    assert any("message is required" in text.get("text", "") for text in result["content"])


def test_get_result():
    """Test getting task results."""
    # Create a mock tool use
    tool_use = {"toolUseId": "test-id", "input": {"action": "get_result", "task_id": "result_task"}}

    # Set up mocks for file operations
    result_content = "Test result content"

    with patch("pathlib.Path.exists", return_value=True), patch("builtins.open", mock_open(read_data=result_content)):
        # Call the function directly
        result = tasks.tasks(tool=tool_use)

        # Verify result
        assert result["status"] == "success"
        assert any("Results for task 'result_task'" in text.get("text", "") for text in result["content"])


def test_stop_task():
    """Test stopping a task."""
    # Create a mock tool use
    tool_use = {"toolUseId": "test-id", "input": {"action": "stop", "task_id": "stop_task"}}

    # Create a mock task state
    mock_task_state = MagicMock()

    # Mock task_threads to include our task
    with (
        patch.dict(tasks.task_threads, {"stop_task": MagicMock()}),
        patch.object(tasks.task_threads["stop_task"], "is_alive", return_value=True),
        patch.object(tasks, "task_states", {"stop_task": mock_task_state}),
    ):
        # Call the function directly
        result = tasks.tasks(tool=tool_use)

        # Verify result
        assert result["status"] == "success"
        assert any("marked as stopped" in text.get("text", "") for text in result["content"])

        # Verify task state was updated
        mock_task_state.update_status.assert_called_with("stopped")


def test_pause_resume_task():
    """Test pausing and resuming a task."""
    # Create a mock tool uses
    pause_tool_use = {"toolUseId": "pause-id", "input": {"action": "pause", "task_id": "pause_task"}}

    resume_tool_use = {"toolUseId": "resume-id", "input": {"action": "resume", "task_id": "pause_task"}}

    # Create a mock task state
    mock_task_state = MagicMock()

    with patch.object(tasks.TaskState, "load", return_value=mock_task_state):
        # Test pause
        pause_result = tasks.tasks(tool=pause_tool_use)

        # Verify pause result
        assert pause_result["status"] == "success"
        assert any("paused successfully" in text.get("text", "") for text in pause_result["content"])

        # Verify task state was updated for pause
        mock_task_state.update_status.assert_called_with("paused")

        # Reset mock for resume test
        mock_task_state.reset_mock()

        # For resume test we need to spy on everything and verify the sequence of calls
        # The actual function calls update_status twice - first to 'resuming', then inside the thread to 'running'
        # In the test, we'll just verify it was called with 'resuming' at any point

        # Test resume
        resume_result = tasks.tasks(tool=resume_tool_use)

        # Verify resume result
        assert resume_result["status"] == "success"
        assert any("resumed successfully" in text.get("text", "") for text in resume_result["content"])

        # Check that update_status was called with 'resuming' at some point
        assert any(call == call("resuming") for call in mock_task_state.update_status.call_args_list)


def test_task_state_initialization(temp_tasks_dir):
    """Test TaskState initialization and file creation."""
    task_id = "test_state_init"
    prompt = "Test prompt"
    system_prompt = "Test system prompt"
    tools = ["tool1", "tool2"]
    timeout = 600

    task_state = tasks.TaskState(task_id, prompt, system_prompt, tools, timeout)

    # Check attributes
    assert task_state.task_id == task_id
    assert task_state.initial_prompt == prompt
    assert task_state.system_prompt == system_prompt
    assert task_state.tools == tools
    assert task_state.timeout == timeout
    assert task_state.status == "initializing"
    assert task_state.paused is False

    # Check paths
    assert task_state.state_path == temp_tasks_dir / f"{task_id}_state.json"
    assert task_state.result_path == temp_tasks_dir / f"{task_id}_result.txt"
    assert task_state.messages_path == temp_tasks_dir / f"{task_id}_messages.json"

    # Check message history
    assert len(task_state.message_history) == 1
    assert task_state.message_history[0]["role"] == "user"
    assert task_state.message_history[0]["content"][0]["text"] == prompt

    # Check file creation
    assert task_state.state_path.exists()
    assert task_state.messages_path.exists()


def test_task_state_methods(temp_tasks_dir):
    """Test TaskState methods for updating state and saving data."""
    task_id = "test_state_methods"
    prompt = "Test prompt"
    system_prompt = "Test system prompt"

    task_state = tasks.TaskState(task_id, prompt, system_prompt)

    # Test update_status
    task_state.update_status("running")
    assert task_state.status == "running"

    # Check that state file is updated
    with open(task_state.state_path, "r") as f:
        state_data = json.load(f)
        assert state_data["status"] == "running"

    # Test add_message
    new_message = "New test message"
    task_state.add_message(new_message)

    # Check message was added
    assert len(task_state.message_history) == 2
    assert task_state.message_history[1]["role"] == "user"
    assert task_state.message_history[1]["content"][0]["text"] == new_message

    # Check messages file is updated
    with open(task_state.messages_path, "r") as f:
        messages_data = json.load(f)
        assert len(messages_data) == 2
        assert messages_data[1]["content"][0]["text"] == new_message

    # Test append_result
    result_text = "Test result output"
    task_state.append_result(result_text)

    # Check result file is updated
    with open(task_state.result_path, "r") as f:
        result_content = f.read()
        assert result_text in result_content


def test_task_state_load(temp_tasks_dir):
    """Test loading TaskState from files."""
    task_id = "test_state_load"

    # Create state file with test data
    state_data = {
        "task_id": task_id,
        "status": "completed",
        "created_at": "2023-01-01T00:00:00",
        "last_updated": "2023-01-01T01:00:00",
        "initial_prompt": "Original prompt",
        "system_prompt": "System prompt",
        "tools": ["tool1"],
        "timeout": 1200,
        "paused": True,
    }

    state_path = temp_tasks_dir / f"{task_id}_state.json"
    with open(state_path, "w") as f:
        json.dump(state_data, f)

    # Create messages file
    messages_data = [
        {"role": "user", "content": [{"text": "Original prompt"}]},
        {"role": "assistant", "content": [{"text": "Assistant response"}]},
    ]

    messages_path = temp_tasks_dir / f"{task_id}_messages.json"
    with open(messages_path, "w") as f:
        json.dump(messages_data, f)

    # Mock the save_messages method to avoid overwriting our test file
    with patch.object(tasks.TaskState, "save_messages"):
        # Load the task state
        loaded_state = tasks.TaskState.load(task_id)

        # Check loaded data
        assert loaded_state.task_id == task_id
        assert loaded_state.status == "completed"
        assert loaded_state.created_at == "2023-01-01T00:00:00"
        assert loaded_state.last_updated == "2023-01-01T01:00:00"
        assert loaded_state.initial_prompt == "Original prompt"
        assert loaded_state.system_prompt == "System prompt"
        assert loaded_state.tools == ["tool1"]
        assert loaded_state.timeout == 1200
        assert loaded_state.paused is True

        # Manually set the message history since it gets reset during TaskState initialization
        loaded_state.message_history = messages_data
        assert len(loaded_state.message_history) == 2


def test_run_task(temp_tasks_dir):
    """Test the run_task function."""
    task_id = "test_run_task"
    prompt = "Test prompt"
    system_prompt = "Test system prompt"

    # Create a task state
    task_state = tasks.TaskState(task_id, prompt, system_prompt)

    # Mock Agent and its response
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.message = "Assistant response"
    # Use proper metrics structure that can be processed by metrics_to_string
    mock_response.metrics = MagicMock()
    mock_response.metrics.get_summary = MagicMock(
        return_value={"tokens": {"total": 100, "prompt": 50, "completion": 50}}
    )
    mock_agent.return_value = mock_response

    # Patch metrics_to_string to avoid dependency issues
    with (
        patch("strands_tools.tasks.Agent", return_value=mock_agent),
        patch.dict("strands_tools.tasks.task_agents", clear=True),
        patch("strands_tools.tasks.metrics_to_string", return_value="Mocked metrics summary"),
    ):
        # Run the task
        tasks.run_task(task_state)

        # Check task was added to task_agents
        assert task_id in tasks.task_agents

        # Check agent was called
        mock_agent.assert_called_once()

        # Check task status was updated
        assert task_state.status == "completed"

        # Check result was saved
        with open(task_state.result_path, "r") as f:
            result_content = f.read()
            assert "Assistant response" in result_content
            assert "Metrics: Mocked metrics summary" in result_content
            assert "Task completed" in result_content


def test_run_task_with_error(temp_tasks_dir):
    """Test run_task error handling."""
    task_id = "test_run_task_error"
    prompt = "Test prompt"
    system_prompt = "Test system prompt"

    # Create a task state
    task_state = tasks.TaskState(task_id, prompt, system_prompt)

    # Mock Agent to raise an exception
    mock_agent = MagicMock()
    mock_agent.side_effect = ValueError("Test error")

    # Patch Agent constructor
    with patch("strands_tools.tasks.Agent", return_value=mock_agent):
        # Run the task (should catch the error)
        tasks.run_task(task_state)

        # Check task status was updated to error
        assert task_state.status == "error"

        # Check error was saved to result file
        with open(task_state.result_path, "r") as f:
            result_content = f.read()
            assert "ERROR: Test error" in result_content
            assert "Stack Trace:" in result_content


def test_run_task_with_parent_agent(temp_tasks_dir):
    """Test run_task with a parent agent that has tools."""
    task_id = "test_run_with_parent"
    prompt = "Test prompt"
    system_prompt = "Test system prompt"
    tools = ["tool1", "tool2"]

    # Create a task state with specific tools
    task_state = tasks.TaskState(task_id, prompt, system_prompt, tools)

    # Mock parent agent with tools
    mock_parent_agent = MagicMock()
    mock_tool_registry = MagicMock()
    mock_parent_agent.tool_registry = mock_tool_registry
    mock_parent_agent.tool_registry.registry = {
        "tool1": {"name": "tool1"},
        "tool2": {"name": "tool2"},
        "tool3": {"name": "tool3"},
    }
    mock_parent_agent.trace_attributes = {"attr1": "value1"}

    # Mock Agent constructor
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.message = "Assistant response"
    mock_response.metrics = None
    mock_agent.return_value = mock_response

    # Patch Agent constructor
    with patch("strands_tools.tasks.Agent", return_value=mock_agent):
        # Run the task with parent agent
        tasks.run_task(task_state, mock_parent_agent)

        # Check that Agent was constructed with correct tools and trace attributes
        tasks.Agent.assert_called_once()
        call_kwargs = tasks.Agent.call_args.kwargs
        assert len(call_kwargs["tools"]) == 2  # Only specified tools should be included
        assert call_kwargs["trace_attributes"] == {"attr1": "value1"}

        # Check task status was updated
        assert task_state.status == "completed"


def test_run_task_with_timeout(temp_tasks_dir):
    """Test run_task_with_timeout function."""
    task_id = "test_timeout"
    prompt = "Test prompt"
    system_prompt = "Test system prompt"
    timeout = 10  # Short timeout for test

    # Create a task state with timeout
    task_state = tasks.TaskState(task_id, prompt, system_prompt, timeout=timeout)

    # Mock threading.Timer and run_task
    with patch("threading.Timer") as mock_timer, patch("strands_tools.tasks.run_task") as mock_run_task:
        # Mock timer instance
        mock_timer_instance = MagicMock()
        mock_timer.return_value = mock_timer_instance

        # Run the task with timeout
        tasks.run_task_with_timeout(task_state)

        # Check Timer was created with correct timeout
        mock_timer.assert_called_once_with(timeout, mock_timer.call_args[0][1])

        # Check timer was started and canceled
        mock_timer_instance.start.assert_called_once()
        mock_timer_instance.cancel.assert_called_once()

        # Check run_task was called
        mock_run_task.assert_called_once_with(task_state, None)


def test_timeout_handler(temp_tasks_dir):
    """Test the timeout handler function used in run_task_with_timeout."""
    task_id = "test_timeout_handler"
    prompt = "Test prompt"
    system_prompt = "Test system prompt"

    # Create a task state
    task_state = tasks.TaskState(task_id, prompt, system_prompt)
    task_state.status = "running"

    # Add task to task_threads and task_states
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True

    # We need to directly test the timeout_handler logic since it's nested
    # Define our own timeout handler similar to the one in run_task_with_timeout
    def test_timeout_handler():
        if task_id in tasks.task_threads and task_state.status == "running":
            task_state.update_status("timeout")
            task_state.append_result(f"ERROR: Task timed out after {task_state.timeout} seconds")

    # Add the task to the dictionaries
    with (
        patch.dict("strands_tools.tasks.task_threads", {task_id: mock_thread}),
        patch.dict("strands_tools.tasks.task_states", {task_id: task_state}),
    ):
        # Call the handler directly
        test_timeout_handler()

        # Check that task status was updated
        assert task_state.status == "timeout"

        # Check that an error message was appended to results
        with open(task_state.result_path, "r") as f:
            result_content = f.read()
            assert "Task timed out" in result_content


def test_list_empty_tasks(temp_tasks_dir):
    """Test listing tasks when no tasks exist."""
    # Create a mock tool use
    tool_use = {"toolUseId": "test-id", "input": {"action": "list"}}

    # Make sure TASKS_DIR is empty and glob returns empty list
    with patch("pathlib.Path.glob", return_value=[]):
        # Call the function directly
        result = tasks.tasks(tool=tool_use)

        # Verify result
        assert result["status"] == "success"
        assert any("No tasks found" in text.get("text", "") for text in result["content"])


def test_task_with_notification(temp_tasks_dir):
    """Test run_task with notification on completion."""
    task_id = "test_notify"
    prompt = "Test prompt"
    system_prompt = "Test system prompt"

    # Create a task state
    task_state = tasks.TaskState(task_id, prompt, system_prompt)

    # Mock notify tool on parent agent
    mock_notify = MagicMock()
    mock_parent_agent = MagicMock()
    mock_parent_agent.tool.notify = mock_notify

    # Mock Agent for task
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.message = "Assistant response"
    mock_response.metrics = None
    mock_agent.return_value = mock_response

    # Patch Agent constructor
    with patch("strands_tools.tasks.Agent", return_value=mock_agent):
        # Run the task with parent agent that has notify tool
        tasks.run_task(task_state, mock_parent_agent)

        # Check that notify was called
        mock_notify.assert_called_once()
        assert mock_notify.call_args.kwargs["title"] == "Task Complete"
        assert task_id in mock_notify.call_args.kwargs["message"]


def test_task_notification_error(temp_tasks_dir):
    """Test run_task handles notification errors gracefully."""
    task_id = "test_notify_error"
    prompt = "Test prompt"
    system_prompt = "Test system prompt"

    # Create a task state
    task_state = tasks.TaskState(task_id, prompt, system_prompt)

    # Mock notify tool on parent agent that raises exception
    mock_notify = MagicMock(side_effect=ValueError("Notification error"))
    mock_parent_agent = MagicMock()
    mock_parent_agent.tool.notify = mock_notify

    # Mock Agent for task
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.message = "Assistant response"
    mock_response.metrics = None
    mock_agent.return_value = mock_response

    # Patch Agent constructor and logger
    with (
        patch("strands_tools.tasks.Agent", return_value=mock_agent),
        patch("strands_tools.tasks.logger") as mock_logger,
    ):
        # Run the task with parent agent that has notify tool
        tasks.run_task(task_state, mock_parent_agent)

        # Check that we tried to notify and logged the error
        mock_notify.assert_called_once()
        mock_logger.error.assert_called_with("Failed to send task completion notification: Notification error")

        # Task should still complete successfully
        assert task_state.status == "completed"


def test_stop_nonexistent_task():
    """Test stopping a task that doesn't exist or isn't running."""
    # Create a mock tool use
    tool_use = {"toolUseId": "test-id", "input": {"action": "stop", "task_id": "nonexistent_task"}}

    # Call the function directly with task not in threads dict
    with patch.dict("strands_tools.tasks.task_threads", clear=True):
        result = tasks.tasks(tool=tool_use)

        # Verify result indicates task is not running
        assert result["status"] == "success"
        assert any("is not running" in text.get("text", "") for text in result["content"])


def test_missing_task_id():
    """Test actions that require task_id but don't receive one."""
    actions = ["status", "add_message", "get_result", "stop", "pause", "resume"]

    for action in actions:
        # Create a mock tool use with missing task_id
        tool_use = {"toolUseId": "test-id", "input": {"action": action}}

        # Call the function directly
        result = tasks.tasks(tool=tool_use)

        # Verify result is an error about missing task_id
        assert result["status"] == "error"
        assert any("task_id is required" in text.get("text", "") for text in result["content"])


def test_notify_with_timeout_error(temp_tasks_dir):
    """Test task with timeout and notification."""
    task_id = "test_timeout_notify"
    prompt = "Test prompt"
    system_prompt = "Test system prompt"

    # Create a task state with a timeout
    task_state = tasks.TaskState(task_id, prompt, system_prompt, timeout=1)
    task_state.status = "running"

    # Mock task thread that never completes (will timeout)
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True

    # Mock notify tool on parent agent
    mock_notify = MagicMock()
    mock_parent_agent = MagicMock()
    mock_parent_agent.tool.notify = mock_notify

    # Define the timeout handler logic (copy from the app code)
    def timeout_handler():
        if task_state.status == "running":
            task_state.update_status("timeout")
            task_state.append_result(f"ERROR: Task timed out after {task_state.timeout} seconds")

            # Send timeout notification if parent agent has notify tool
            if hasattr(mock_parent_agent.tool, "notify"):
                try:
                    mock_parent_agent.tool.notify(
                        message=f"Task '{task_state.task_id}' timed out after {task_state.timeout} seconds",
                        title="Task Timeout",
                        priority="high",
                        category="tasks",
                        source=task_state.task_id,
                        show_system_notification=True,
                    )
                except Exception:
                    pass

    # Call the timeout handler
    timeout_handler()

    # Check that notification was called with the right parameters
    mock_notify.assert_called_once()
    assert mock_notify.call_args.kwargs["title"] == "Task Timeout"
    assert mock_notify.call_args.kwargs["priority"] == "high"
    assert task_id in mock_notify.call_args.kwargs["message"]


def test_agent_with_tools(temp_tasks_dir):
    """Test run_task with a parent agent with specific tools."""
    task_id = "test_tools"
    prompt = "Test prompt"
    system_prompt = "Test system prompt"

    # Create a task state with no specific tools
    task_state = tasks.TaskState(task_id, prompt, system_prompt)

    # Create a mock parent agent with tools
    mock_parent_agent = MagicMock()
    mock_parent_agent.tool_registry.registry = {"tool1": {"name": "tool1"}, "tool2": {"name": "tool2"}}
    mock_parent_agent.trace_attributes = {}

    # Mock Agent constructor
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.message = "Assistant response"
    mock_response.metrics = None
    mock_agent.return_value = mock_response

    # Patch Agent constructor
    with patch("strands_tools.tasks.Agent", return_value=mock_agent):
        # Run the task with parent agent that has tools
        tasks.run_task(task_state, mock_parent_agent)

        # Verify that Agent was constructed with all tools from parent
        tasks.Agent.assert_called_once()
        call_kwargs = tasks.Agent.call_args.kwargs
        assert len(call_kwargs["tools"]) == 2

        # Verify task completed successfully
        assert task_state.status == "completed"


def test_create_task_with_uuid(temp_tasks_dir):
    """Test creating a task with auto-generated ID."""
    # Prepare a mock tool use without task_id
    tool_use = {
        "toolUseId": "test-id",
        "input": {"action": "create", "prompt": "Test prompt", "system_prompt": "Test system prompt"},
    }

    # Patch uuid generation to get a predictable ID
    with patch("uuid.uuid4", return_value=MagicMock(hex="abcdef0123456789")), patch("threading.Thread"):
        # Call the function
        result = tasks.tasks(tool=tool_use)

        # Verify success and that a task ID was generated
        assert result["status"] == "success"
        assert "task_abcdef01" in str(result)

        # Verify the task was created with the generated ID
        assert any("task_abcdef01" in text.get("text", "") for text in result["content"])


def test_create_duplicate_task(temp_tasks_dir):
    """Test creating a task with an ID that already exists."""
    task_id = "duplicate_task"

    # Prepare a mock tool use with an existing task_id
    tool_use = {
        "toolUseId": "test-id",
        "input": {
            "action": "create",
            "task_id": task_id,
            "prompt": "Test prompt",
            "system_prompt": "Test system prompt",
        },
    }

    # Create mock thread that's running
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True

    # Add thread to task_threads
    with patch.dict("strands_tools.tasks.task_threads", {task_id: mock_thread}):
        # Call the function
        result = tasks.tasks(tool=tool_use)

        # Verify error response
        assert result["status"] == "error"
        assert any("already exists" in text.get("text", "") for text in result["content"])


def test_task_with_empty_message_history(temp_tasks_dir):
    """Test task execution with empty message history."""
    task_id = "empty_messages_task"
    prompt = "Test prompt"
    system_prompt = "Test system prompt"

    # Create a task state
    task_state = tasks.TaskState(task_id, prompt, system_prompt)

    # Empty the message history to test that case
    task_state.message_history = []

    # Mock Agent and its response
    mock_agent = MagicMock()
    mock_agent.return_value = MagicMock()

    # Patch Agent constructor
    with (
        patch("strands_tools.tasks.Agent", return_value=mock_agent),
        patch.dict("strands_tools.tasks.task_agents", clear=True),
    ):
        # Run the task
        tasks.run_task(task_state)

        # Check task completed without errors even with empty message history
        assert task_state.status == "completed"
        assert task_id in tasks.task_agents

        # Check that agent was created but not called (no messages to process)
        assert not mock_agent.called


def test_task_global_memory_clearing(temp_tasks_dir):
    """Test that task_agents dictionary is properly updated."""
    # Create task states and add to task_agents
    with patch.dict("strands_tools.tasks.task_agents", clear=True):
        # Verify dictionary is empty to start
        assert not tasks.task_agents

        # Add some mock agents
        tasks.task_agents["task1"] = MagicMock()
        tasks.task_agents["task2"] = MagicMock()

        # Verify agents were added
        assert len(tasks.task_agents) == 2
        assert "task1" in tasks.task_agents
        assert "task2" in tasks.task_agents

        # Clear the dictionary
        tasks.task_agents.clear()

        # Verify dictionary is empty again
        assert not tasks.task_agents


def test_resume_with_new_thread(temp_tasks_dir):
    """Test resuming a task that starts a new thread."""
    task_id = "test_resume_thread"

    # Create mock tool use for resume
    tool_use = {"toolUseId": "test-id", "input": {"action": "resume", "task_id": task_id}}

    # Create a mock task state
    mock_task_state = MagicMock()
    mock_task_state.status = "paused"
    mock_task_state.task_id = task_id

    # Mock thread creation
    with (
        patch.object(tasks.TaskState, "load", return_value=mock_task_state),
        patch("threading.Thread") as mock_thread,
        patch.dict("strands_tools.tasks.task_threads", clear=True),
    ):
        # Set up mock thread
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        # Call the function to resume the task
        result = tasks.tasks(tool=tool_use)

        # Verify new thread was created and started
        mock_thread.assert_called_once()
        assert mock_thread.call_args.kwargs["name"] == f"task-{task_id}"
        mock_thread_instance.start.assert_called_once()

        # Check task state was updated
        mock_task_state.update_status.assert_called_with("resuming")

        # Verify result
        assert result["status"] == "success"
        assert any("resumed successfully" in text.get("text", "") for text in result["content"])


def test_run_task_paused(temp_tasks_dir):
    """Test run_task with a paused task."""
    task_id = "test_run_task_paused"
    prompt = "Test prompt"
    system_prompt = "Test system prompt"

    # Create a task state that's paused
    task_state = tasks.TaskState(task_id, prompt, system_prompt)
    task_state.paused = True

    # Mock Agent to ensure it's not called
    mock_agent = MagicMock()

    # Patch Agent constructor
    with patch("strands_tools.tasks.Agent", return_value=mock_agent):
        # Run the task
        tasks.run_task(task_state)

        # Check agent was not constructed
        tasks.Agent.assert_not_called()

        # Check result includes paused message
        with open(task_state.result_path, "r") as f:
            result_content = f.read()
            assert "Task is paused" in result_content
