"""
Tests for the tasks tool using proper mocking to avoid real execution.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from strands_tools import tasks


@pytest.fixture
def temp_tasks_dir():
    """Create a temporary directory for tasks."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with patch.object(tasks, "TASKS_DIR", temp_path):
            yield temp_path


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    mock_agent = MagicMock()
    mock_agent.model = "anthropic.claude-3-sonnet-20240229-v1:0"
    mock_agent.tool_registry.registry = {"tool1": {"name": "tool1"}, "tool2": {"name": "tool2"}}
    mock_agent.trace_attributes = {}
    return mock_agent


@pytest.fixture
def cleanup_global_state():
    """Clean up global state before and after each test."""
    # Clear global dictionaries
    tasks.task_threads.clear()
    tasks.task_agents.clear()
    tasks.task_states.clear()
    tasks.task_message_queues.clear()

    yield

    # Clean up after test
    tasks.task_threads.clear()
    tasks.task_agents.clear()
    tasks.task_states.clear()
    tasks.task_message_queues.clear()


class TestTaskState:
    """Test the TaskState class."""

    def test_task_state_creation(self, temp_tasks_dir, cleanup_global_state):
        """Test TaskState creation and file persistence."""
        task_state = tasks.TaskState(
            task_id="test_task",
            prompt="Test prompt",
            system_prompt="Test system prompt",
            tools=["tool1", "tool2"],
            timeout=60,
        )

        assert task_state.task_id == "test_task"
        assert task_state.initial_prompt == "Test prompt"
        assert task_state.system_prompt == "Test system prompt"
        assert task_state.tools == ["tool1", "tool2"]
        assert task_state.timeout == 60
        assert task_state.status == "initializing"
        assert task_state.paused is False

        # Check files were created
        assert task_state.state_path.exists()
        assert task_state.messages_path.exists()

        # Check message queue was created
        assert task_state.task_id in tasks.task_message_queues

    def test_task_state_save_and_load(self, temp_tasks_dir, cleanup_global_state):
        """Test saving and loading TaskState."""
        # Create and save a task state
        original_state = tasks.TaskState(
            task_id="test_load",
            prompt="Load test prompt",
            system_prompt="Load test system prompt",
            tools=["tool1"],
            timeout=120,
        )
        original_state.update_status("running")

        # Load the task state
        loaded_state = tasks.TaskState.load("test_load")

        assert loaded_state is not None
        assert loaded_state.task_id == "test_load"
        assert loaded_state.initial_prompt == "Load test prompt"
        assert loaded_state.system_prompt == "Load test system prompt"
        assert loaded_state.tools == ["tool1"]
        assert loaded_state.timeout == 120
        assert loaded_state.status == "running"

    def test_task_state_load_nonexistent(self, temp_tasks_dir, cleanup_global_state):
        """Test loading a non-existent task state."""
        loaded_state = tasks.TaskState.load("nonexistent")
        assert loaded_state is None

    def test_append_result(self, temp_tasks_dir, cleanup_global_state):
        """Test appending results to task."""
        task_state = tasks.TaskState(task_id="test_result", prompt="Test prompt", system_prompt="Test system prompt")

        task_state.append_result("First result")
        task_state.append_result("Second result")

        # Check result file contents
        with open(task_state.result_path, "r") as f:
            content = f.read()

        assert "First result" in content
        assert "Second result" in content

    def test_update_message_history(self, temp_tasks_dir, cleanup_global_state):
        """Test updating message history."""
        task_state = tasks.TaskState(task_id="test_messages", prompt="Test prompt", system_prompt="Test system prompt")

        new_messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi there!"}]},
        ]

        task_state.update_message_history(new_messages)

        # Check messages file
        with open(task_state.messages_path, "r") as f:
            saved_messages = json.load(f)

        assert saved_messages == new_messages

    def test_add_message_to_queue(self, temp_tasks_dir, cleanup_global_state):
        """Test adding message to task queue."""
        task_state = tasks.TaskState(task_id="queue_test", prompt="Test prompt", system_prompt="Test system prompt")

        task_state.add_message_to_queue("Test queued message")

        # Verify message was queued
        assert task_state.task_id in tasks.task_message_queues
        queue_obj = tasks.task_message_queues[task_state.task_id]
        assert not queue_obj.empty()

        # Get the message
        message = queue_obj.get()
        assert message == "Test queued message"

    def test_update_status(self, temp_tasks_dir, cleanup_global_state):
        """Test status update and save."""
        task_state = tasks.TaskState(task_id="status_test", prompt="Test prompt", system_prompt="Test system prompt")

        task_state.update_status("running")

        assert task_state.status == "running"

        # Check that state was saved to file
        with open(task_state.state_path, "r") as f:
            state_data = json.load(f)

        assert state_data["status"] == "running"


class TestTasksFunction:
    """Test the main tasks function with proper mocking."""

    @patch("threading.Thread")
    def test_create_task_success(self, mock_thread, temp_tasks_dir, mock_agent, cleanup_global_state):
        """Test successful task creation."""
        tool_use = {
            "toolUseId": "test-create",
            "input": {
                "action": "create",
                "task_id": "test_task",
                "prompt": "Test prompt",
                "system_prompt": "Test system prompt",
                "tools": ["tool1"],
                "timeout": 60,
            },
        }

        result = tasks.tasks(tool=tool_use, agent=mock_agent)

        assert result["toolUseId"] == "test-create"
        assert result["status"] == "success"
        assert "test_task" in result["content"][0]["text"]
        assert "created and started" in result["content"][0]["text"]

        # Verify thread was started
        mock_thread.assert_called_once()

        # Verify task state was created
        assert "test_task" in tasks.task_states

    def test_create_task_missing_prompt(self, temp_tasks_dir, cleanup_global_state):
        """Test task creation with missing prompt."""
        tool_use = {"toolUseId": "test-error", "input": {"action": "create", "system_prompt": "Test system prompt"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "error"
        assert "prompt is required" in result["content"][0]["text"]

    def test_create_task_missing_system_prompt(self, temp_tasks_dir, cleanup_global_state):
        """Test task creation with missing system prompt."""
        tool_use = {"toolUseId": "test-error", "input": {"action": "create", "prompt": "Test prompt"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "error"
        assert "system_prompt is required" in result["content"][0]["text"]

    @patch("threading.Thread")
    def test_create_task_auto_generate_id(self, mock_thread, temp_tasks_dir, mock_agent, cleanup_global_state):
        """Test task creation with auto-generated ID."""
        tool_use = {
            "toolUseId": "test-auto-id",
            "input": {"action": "create", "prompt": "Test prompt", "system_prompt": "Test system prompt"},
        }

        result = tasks.tasks(tool=tool_use, agent=mock_agent)

        assert result["status"] == "success"
        assert "task_" in result["content"][0]["text"]
        mock_thread.assert_called_once()

    def test_create_task_duplicate_running(self, temp_tasks_dir, mock_agent, cleanup_global_state):
        """Test creating task with ID that's already running."""
        # Mock a running thread
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        tasks.task_threads["duplicate_test"] = mock_thread

        tool_use = {
            "toolUseId": "test-duplicate",
            "input": {
                "action": "create",
                "task_id": "duplicate_test",
                "prompt": "Test prompt",
                "system_prompt": "Test system prompt",
            },
        }

        result = tasks.tasks(tool=tool_use, agent=mock_agent)

        assert result["status"] == "error"
        assert "already exists and is running" in result["content"][0]["text"]

    def test_status_task_exists_in_memory(self, temp_tasks_dir, cleanup_global_state):
        """Test getting status of an existing task in memory."""
        # Create a task state in memory
        task_state = tasks.TaskState(task_id="status_test", prompt="Test prompt", system_prompt="Test system prompt")
        tasks.task_states["status_test"] = task_state

        tool_use = {"toolUseId": "test-status", "input": {"action": "status", "task_id": "status_test"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "success"
        assert "status_test" in result["content"][0]["text"]
        assert "initializing" in result["content"][0]["text"]

    def test_status_task_from_disk(self, temp_tasks_dir, cleanup_global_state):
        """Test getting status of task loaded from disk."""
        # Create task state on disk but not in memory
        task_state = tasks.TaskState(task_id="disk_task", prompt="Disk prompt", system_prompt="Disk system prompt")
        task_state.update_status("completed")
        # Clear from memory to force loading from disk
        tasks.task_states.clear()
        tasks.task_message_queues.clear()

        tool_use = {"toolUseId": "test-status-disk", "input": {"action": "status", "task_id": "disk_task"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "success"
        assert "disk_task" in result["content"][0]["text"]
        assert "completed" in result["content"][0]["text"]

    def test_status_task_not_found(self, temp_tasks_dir, cleanup_global_state):
        """Test getting status of non-existent task."""
        tool_use = {"toolUseId": "test-status-error", "input": {"action": "status", "task_id": "nonexistent"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_list_tasks_empty(self, temp_tasks_dir, cleanup_global_state):
        """Test listing tasks when none exist."""
        tool_use = {"toolUseId": "test-list", "input": {"action": "list"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "success"
        assert "No tasks found" in result["content"][0]["text"]

    def test_list_tasks_with_tasks(self, temp_tasks_dir, cleanup_global_state):
        """Test listing tasks when some exist."""
        # Create a few task states
        task1 = tasks.TaskState("task1", "Prompt 1", "System 1")
        task2 = tasks.TaskState("task2", "Prompt 2", "System 2")
        tasks.task_states["task1"] = task1
        tasks.task_states["task2"] = task2

        tool_use = {"toolUseId": "test-list", "input": {"action": "list"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "success"
        assert "Found 2 tasks" in result["content"][0]["text"]
        assert "task1" in result["content"][1]["text"]
        assert "task2" in result["content"][1]["text"]

    def test_add_message_to_running_task(self, temp_tasks_dir, cleanup_global_state):
        """Test adding message to a running task."""
        # Create task state
        task_state = tasks.TaskState(
            task_id="running_task", prompt="Original prompt", system_prompt="Test system prompt"
        )
        tasks.task_states["running_task"] = task_state

        # Mock as running thread
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        tasks.task_threads["running_task"] = mock_thread

        tool_use = {
            "toolUseId": "test-add-msg",
            "input": {"action": "add_message", "task_id": "running_task", "message": "New message"},
        }

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "success"
        assert "Message queued for processing" in result["content"][0]["text"]

    @patch("threading.Thread")
    def test_add_message_to_stopped_task(self, mock_thread, temp_tasks_dir, mock_agent, cleanup_global_state):
        """Test adding message to a stopped task."""
        # Create a task state
        task_state = tasks.TaskState(
            task_id="stopped_task", prompt="Original prompt", system_prompt="Test system prompt"
        )
        task_state.update_status("completed")
        tasks.task_states["stopped_task"] = task_state

        tool_use = {
            "toolUseId": "test-add-msg",
            "input": {"action": "add_message", "task_id": "stopped_task", "message": "New message"},
        }

        result = tasks.tasks(tool=tool_use, agent=mock_agent)

        assert result["status"] == "success"
        assert "Message added to task" in result["content"][0]["text"]
        mock_thread.assert_called_once()

    def test_add_message_to_nonexistent_task(self, temp_tasks_dir, cleanup_global_state):
        """Test adding message to non-existent task."""
        tool_use = {
            "toolUseId": "test-add-msg-error",
            "input": {"action": "add_message", "task_id": "nonexistent", "message": "Test message"},
        }

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_get_result_exists(self, temp_tasks_dir, cleanup_global_state):
        """Test getting result from existing task."""
        # Create task and add result
        task_state = tasks.TaskState("result_task", "Test prompt", "Test system prompt")
        task_state.append_result("Test result content")

        tool_use = {"toolUseId": "test-get-result", "input": {"action": "get_result", "task_id": "result_task"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "success"
        assert "Test result content" in result["content"][1]["text"]

    def test_get_result_not_found(self, temp_tasks_dir, cleanup_global_state):
        """Test getting result from non-existent task."""
        tool_use = {"toolUseId": "test-get-result-error", "input": {"action": "get_result", "task_id": "nonexistent"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "error"
        assert "No results found" in result["content"][0]["text"]

    def test_get_messages(self, temp_tasks_dir, cleanup_global_state):
        """Test getting message history."""
        # Create task with message history
        task_state = tasks.TaskState("msg_task", "Test prompt", "Test system prompt")
        messages = [
            {"role": "user", "content": [{"text": "Hello"}]},
            {"role": "assistant", "content": [{"text": "Hi there!"}]},
            {"role": "user", "content": [{"toolUse": {"name": "test_tool", "toolUseId": "123"}}]},
            {"role": "user", "content": [{"toolResult": {"status": "success", "toolUseId": "123"}}]},
        ]
        task_state.update_message_history(messages)
        tasks.task_states["msg_task"] = task_state

        tool_use = {"toolUseId": "test-get-messages", "input": {"action": "get_messages", "task_id": "msg_task"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "success"
        result_text = result["content"][1]["text"]
        assert "Hello" in result_text
        assert "Hi there!" in result_text
        assert "toolUse: test_tool" in result_text
        assert "toolResult: success" in result_text

    def test_stop_task_running(self, temp_tasks_dir, cleanup_global_state):
        """Test stopping a running task."""
        # Create task state
        task_state = tasks.TaskState("stop_task", "Test prompt", "Test system prompt")
        tasks.task_states["stop_task"] = task_state

        # Mock running thread
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        tasks.task_threads["stop_task"] = mock_thread

        tool_use = {"toolUseId": "test-stop", "input": {"action": "stop", "task_id": "stop_task"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "success"
        assert "marked as stopped" in result["content"][0]["text"]
        assert task_state.status == "stopped"

    def test_stop_task_not_running(self, temp_tasks_dir, cleanup_global_state):
        """Test stopping a task that's not running."""
        tool_use = {"toolUseId": "test-stop-not-running", "input": {"action": "stop", "task_id": "not_running_test"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "success"
        assert "not running" in result["content"][0]["text"]

    def test_pause_task(self, temp_tasks_dir, cleanup_global_state):
        """Test pausing a task."""
        # Create task state
        task_state = tasks.TaskState("pause_task", "Test prompt", "Test system prompt")
        tasks.task_states["pause_task"] = task_state

        tool_use = {"toolUseId": "test-pause", "input": {"action": "pause", "task_id": "pause_task"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "success"
        assert "paused successfully" in result["content"][0]["text"]
        assert task_state.paused is True
        assert task_state.status == "paused"

    @patch("threading.Thread")
    def test_resume_task(self, mock_thread, temp_tasks_dir, mock_agent, cleanup_global_state):
        """Test resuming a paused task."""
        # Create paused task state
        task_state = tasks.TaskState("resume_task", "Test prompt", "Test system prompt")
        task_state.paused = True
        task_state.update_status("paused")
        tasks.task_states["resume_task"] = task_state

        tool_use = {"toolUseId": "test-resume", "input": {"action": "resume", "task_id": "resume_task"}}

        result = tasks.tasks(tool=tool_use, agent=mock_agent)

        assert result["status"] == "success"
        assert "resumed successfully" in result["content"][0]["text"]
        assert task_state.paused is False
        mock_thread.assert_called_once()

    def test_unknown_action(self, temp_tasks_dir, cleanup_global_state):
        """Test unknown action."""
        tool_use = {"toolUseId": "test-unknown", "input": {"action": "unknown_action"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]


class TestTaskExecution:
    """Test task execution functions with proper mocking to prevent real execution."""

    @patch("threading.Thread")
    @patch("strands_tools.tasks.run_task_with_timeout")
    def test_create_task_calls_run_task_properly(
        self, mock_run_task_with_timeout, mock_thread, temp_tasks_dir, cleanup_global_state
    ):
        """Test that creating a task properly calls run_task_with_timeout without actually executing it."""
        # Setup mock thread
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        # Create mock parent agent
        mock_parent = MagicMock()
        mock_parent.model = "anthropic.claude-3-sonnet-20240229-v1:0"
        mock_parent.tool_registry.registry = {}
        mock_parent.trace_attributes = {}

        # Test create task
        tool_use = {
            "toolUseId": "test-create-task",
            "input": {
                "action": "create",
                "task_id": "test_task_execution",
                "prompt": "Test prompt",
                "system_prompt": "Test system prompt",
                "tools": ["test_tool"],
            },
        }

        result = tasks.tasks(tool=tool_use, agent=mock_parent)

        # Verify task creation succeeded
        assert result["status"] == "success"
        assert "test_task_execution" in result["content"][0]["text"]

        # Verify thread was created and started
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()

        # Verify run_task_with_timeout was passed as target (but not called directly)
        thread_call = mock_thread.call_args
        assert thread_call[1]["target"] == mock_run_task_with_timeout

        # Verify correct parameters were passed to the thread target
        thread_args = thread_call[1]["args"]
        task_state = thread_args[0]
        parent_agent = thread_args[1]

        # Verify correct parameters were passed
        assert task_state.task_id == "test_task_execution"
        assert task_state.initial_prompt == "Test prompt"
        assert task_state.system_prompt == "Test system prompt"
        assert task_state.tools == ["test_tool"]
        assert parent_agent == mock_parent

    @patch("strands_tools.tasks.Agent")
    def test_run_task_agent_creation_and_setup(self, mock_agent_class, temp_tasks_dir, cleanup_global_state):
        """Test run_task agent creation without actually running the agent."""
        # Setup mock agent
        mock_agent_instance = MagicMock()
        mock_agent_instance.messages = [
            {"role": "user", "content": [{"text": "Test prompt"}]},
            {"role": "assistant", "content": [{"text": "Test response"}]},
        ]
        mock_agent_class.return_value = mock_agent_instance

        # Create task state
        task_state = tasks.TaskState(
            task_id="agent_creation_test",
            prompt="Test prompt",
            system_prompt="Test system prompt",
            tools=["allowed_tool"],
        )

        # Create mock parent agent
        mock_parent = MagicMock()
        mock_parent.model = "anthropic.claude-3-sonnet-20240229-v1:0"
        mock_parent.tool_registry.registry = {
            "allowed_tool": {"name": "allowed_tool"},
            "blocked_tool": {"name": "blocked_tool"},
        }
        mock_parent.trace_attributes = {"trace": "test"}

        # Mock the agent call to avoid actual execution but check the setup
        def mock_agent_call(prompt):
            # Just return a mock result without doing actual work
            result = MagicMock()
            result.metrics = None
            return result

        mock_agent_instance.side_effect = mock_agent_call

        # Run the task (it will call our mocked agent)
        tasks.run_task(task_state, mock_parent)

        # Verify agent was created with correct parameters
        mock_agent_class.assert_called_once()
        agent_kwargs = mock_agent_class.call_args[1]

        assert agent_kwargs["model"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert agent_kwargs["system_prompt"] == "Test system prompt"
        assert agent_kwargs["trace_attributes"] == {"trace": "test"}

        # Verify only allowed tools were passed
        assert len(agent_kwargs["tools"]) == 1
        assert agent_kwargs["tools"][0]["name"] == "allowed_tool"

    def test_run_task_paused_early_exit(self, temp_tasks_dir, cleanup_global_state):
        """Test run_task early exit when task is paused."""
        # Create paused task state
        task_state = tasks.TaskState(task_id="paused_test", prompt="Test prompt", system_prompt="Test system prompt")
        task_state.paused = True

        # Run the task - should exit early without creating agent
        tasks.run_task(task_state, None)

        # Verify paused message was written
        with open(task_state.result_path, "r") as f:
            content = f.read()
        assert "Task is paused" in content

    @patch("strands_tools.tasks.Agent")
    def test_run_task_exception_handling(self, mock_agent_class, temp_tasks_dir, cleanup_global_state):
        """Test run_task exception handling."""
        # Make agent creation raise an exception
        mock_agent_class.side_effect = Exception("Test exception")

        # Create task state
        task_state = tasks.TaskState(task_id="exception_test", prompt="Test prompt", system_prompt="Test system prompt")

        # Create mock parent agent
        mock_parent = MagicMock()
        mock_parent.model = "test-model"
        mock_parent.tool_registry.registry = {}
        mock_parent.trace_attributes = {}

        # Run the task
        tasks.run_task(task_state, mock_parent)

        # Verify error status was set
        assert task_state.status == "error"

        # Check that error was written to result file
        with open(task_state.result_path, "r") as f:
            content = f.read()
        assert "ERROR: Test exception" in content

    @patch("strands_tools.tasks.Agent")
    def test_run_task_timeout_error_handling(self, mock_agent_class, temp_tasks_dir, cleanup_global_state):
        """Test run_task handles TimeoutError."""
        # Make agent creation raise TimeoutError
        mock_agent_class.side_effect = TimeoutError("Task timeout")

        # Create task state
        task_state = tasks.TaskState(task_id="timeout_test", prompt="Test prompt", system_prompt="Test system prompt")

        # Create mock parent agent
        mock_parent = MagicMock()
        mock_parent.model = "test-model"
        mock_parent.tool_registry.registry = {}
        mock_parent.trace_attributes = {}

        # Run the task
        tasks.run_task(task_state, mock_parent)

        # Verify timeout status was set
        assert task_state.status == "timeout"

        # Check that timeout error was written
        with open(task_state.result_path, "r") as f:
            content = f.read()
        assert "ERROR: Task timed out" in content

    @patch("strands_tools.tasks.Agent")
    @patch("strands_tools.tasks.metrics_to_string")
    def test_run_task_with_successful_execution(
        self, mock_metrics_to_string, mock_agent_class, temp_tasks_dir, cleanup_global_state
    ):
        """Test successful run_task execution with controlled mocking."""
        # Setup mock agent that doesn't actually call AWS
        mock_agent_instance = MagicMock()
        mock_agent_instance.messages = [
            {"role": "user", "content": [{"text": "Test prompt"}]},
            {"role": "assistant", "content": [{"text": "Mocked response"}]},
        ]
        mock_agent_class.return_value = mock_agent_instance

        # Mock the agent call to return immediately
        mock_result = MagicMock()
        mock_result.metrics = None
        mock_agent_instance.return_value = mock_result

        # Mock metrics function
        mock_metrics_to_string.return_value = "Test metrics"

        # Create task state with message history to trigger processing
        task_state = tasks.TaskState(
            task_id="successful_execution_test", prompt="Test prompt", system_prompt="Test system prompt"
        )
        task_state.message_history = [{"role": "user", "content": [{"text": "Test prompt"}]}]

        # Create mock parent
        mock_parent = MagicMock()
        mock_parent.model = "test-model"
        mock_parent.tool_registry.registry = {}
        mock_parent.trace_attributes = {}

        # Run the task with controlled execution
        tasks.run_task(task_state, mock_parent)

        # Verify agent was created and called correctly
        mock_agent_class.assert_called_once()
        mock_agent_instance.assert_called_once_with("Test prompt")

        # Verify results were written
        with open(task_state.result_path, "r") as f:
            content = f.read()
        assert "Processed initial message" in content


class TestToolSpec:
    """Test tool specification structure."""

    def test_tool_spec_structure(self):
        """Test that TOOL_SPEC has the correct structure."""
        assert tasks.TOOL_SPEC["name"] == "tasks"
        assert "description" in tasks.TOOL_SPEC
        assert "inputSchema" in tasks.TOOL_SPEC

        schema = tasks.TOOL_SPEC["inputSchema"]["json"]
        assert "action" in schema["properties"]
        assert schema["required"] == ["action"]

        # Check action enum values
        action_enum = schema["properties"]["action"]["enum"]
        expected_actions = [
            "create",
            "status",
            "list",
            "stop",
            "resume",
            "pause",
            "add_message",
            "get_result",
            "get_messages",
        ]
        for action in expected_actions:
            assert action in action_enum

    def test_module_level_variables(self):
        """Test module-level variables are properly initialized."""
        assert hasattr(tasks, "TASKS_DIR")
        assert hasattr(tasks, "task_threads")
        assert hasattr(tasks, "task_agents")
        assert hasattr(tasks, "task_states")
        assert hasattr(tasks, "task_message_queues")

        assert isinstance(tasks.task_threads, dict)
        assert isinstance(tasks.task_agents, dict)
        assert isinstance(tasks.task_states, dict)
        assert isinstance(tasks.task_message_queues, dict)


class TestAdvancedTaskExecution:
    """Test advanced task execution scenarios including error handling, notifications, threading, and complex logic."""

    def test_task_state_filesystem_error_handling(self, temp_tasks_dir, cleanup_global_state):
        """Test TaskState handling when filesystem operations fail - Lines 198-202, 210-214."""
        task_state = tasks.TaskState(task_id="fs_error_test", prompt="Test prompt", system_prompt="Test system prompt")

        # Make the parent directory read-only to simulate filesystem error
        import os

        # Remove all files from temp directory first
        for file in temp_tasks_dir.glob("*"):
            if file.is_file():
                file.unlink()

        # Now make the directory read-only
        os.chmod(temp_tasks_dir, 0o444)  # Read-only permissions

        try:
            # This should handle the OSError/PermissionError gracefully (lines 198-202)
            task_state.append_result("This should handle filesystem error")

            # Test save_state with filesystem error
            # This should handle the OSError/PermissionError gracefully (lines 210-214)
            task_state.update_status("error_test")
        finally:
            # Restore permissions for cleanup
            os.chmod(temp_tasks_dir, 0o755)

    @patch("strands_tools.tasks.Agent")
    def test_tool_filtering_with_specific_tools(
        self, mock_agent_class, temp_tasks_dir, mock_agent, cleanup_global_state
    ):
        """Test tool registry filtering logic - Lines 302-320, 316-317."""
        # Setup mock agent
        mock_agent_instance = MagicMock()
        mock_agent_instance.messages = [{"role": "user", "content": [{"text": "Test"}]}]
        mock_agent_class.return_value = mock_agent_instance

        # Mock agent call to return immediately
        mock_result = MagicMock()
        mock_result.metrics = None
        mock_agent_instance.return_value = mock_result

        # Create task state with specific tools that exist in registry
        task_state = tasks.TaskState(
            task_id="tool_filter_test",
            prompt="Test prompt",
            system_prompt="Test system prompt",
            tools=["tool1", "nonexistent_tool"],  # Mix of existing and non-existing tools
        )

        # Create parent agent with multiple tools
        mock_parent = MagicMock()
        mock_parent.model = "test-model"
        mock_parent.tool_registry.registry = {
            "tool1": {"name": "tool1"},
            "tool2": {"name": "tool2"},
            "tool3": {"name": "tool3"},
        }
        mock_parent.trace_attributes = {}

        # Run the task - this should trigger tool filtering logic
        tasks.run_task(task_state, mock_parent)

        # Verify agent was created with only the existing tools
        mock_agent_class.assert_called_once()
        agent_kwargs = mock_agent_class.call_args[1]

        # Should have filtered to only include "tool1" since "nonexistent_tool" doesn't exist
        assert len(agent_kwargs["tools"]) == 1
        assert agent_kwargs["tools"][0]["name"] == "tool1"

    @patch("strands_tools.tasks.Agent")
    def test_tool_filtering_with_no_specific_tools(
        self, mock_agent_class, temp_tasks_dir, mock_agent, cleanup_global_state
    ):
        """Test tool registry filtering when no specific tools are requested - Lines 302-320."""
        # Setup mock agent
        mock_agent_instance = MagicMock()
        mock_agent_instance.messages = [{"role": "user", "content": [{"text": "Test"}]}]
        mock_agent_class.return_value = mock_agent_instance

        # Mock agent call
        mock_result = MagicMock()
        mock_result.metrics = None
        mock_agent_instance.return_value = mock_result

        # Create task state with no specific tools (should get all tools)
        task_state = tasks.TaskState(
            task_id="all_tools_test",
            prompt="Test prompt",
            system_prompt="Test system prompt",
            tools=[],  # Empty tools list should result in all tools being loaded
        )

        # Create parent agent with multiple tools
        mock_parent = MagicMock()
        mock_parent.model = "test-model"
        mock_parent.tool_registry.registry = {
            "tool1": {"name": "tool1"},
            "tool2": {"name": "tool2"},
            "tool3": {"name": "tool3"},
        }
        mock_parent.trace_attributes = {}

        # Run the task
        tasks.run_task(task_state, mock_parent)

        # Verify agent was created with all tools
        mock_agent_class.assert_called_once()
        agent_kwargs = mock_agent_class.call_args[1]

        # Should have all 3 tools since no specific tools were requested
        assert len(agent_kwargs["tools"]) == 3

    @patch("strands_tools.tasks.Agent")
    def test_message_queue_processing_with_multiple_messages(
        self, mock_agent_class, temp_tasks_dir, cleanup_global_state
    ):
        """Test message queue processing loop with multiple queued messages - Lines 325-373."""
        # Setup mock agent that processes multiple messages
        mock_agent_instance = MagicMock()
        mock_agent_instance.messages = [{"role": "user", "content": [{"text": "Initial"}]}]
        mock_agent_class.return_value = mock_agent_instance

        # Mock multiple agent calls
        mock_results = []
        for _ in range(3):
            mock_result = MagicMock()
            mock_result.metrics = None
            mock_results.append(mock_result)

        mock_agent_instance.side_effect = mock_results

        # Create task state
        task_state = tasks.TaskState(
            task_id="queue_processing_test", prompt="Initial prompt", system_prompt="Test system prompt"
        )

        # Pre-populate the message queue with multiple messages
        task_state.message_queue.put("Message 1")
        task_state.message_queue.put("Message 2")
        task_state.message_queue.put("Message 3")

        # Create mock parent
        mock_parent = MagicMock()
        mock_parent.model = "test-model"
        mock_parent.tool_registry.registry = {}
        mock_parent.trace_attributes = {}

        # Run the task - should process initial message + 3 queued messages
        tasks.run_task(task_state, mock_parent)

        # Verify agent was called 4 times (initial + 3 queued)
        assert mock_agent_instance.call_count == 4

        # Verify all messages were processed
        assert task_state.message_queue.empty()

    @patch("strands_tools.tasks.Agent")
    def test_notification_calls_with_notify_tool(self, mock_agent_class, temp_tasks_dir, cleanup_global_state):
        """Test notification system when parent agent has notify tool - Lines 398, 408-409, 419."""
        # Setup mock agent
        mock_agent_instance = MagicMock()
        mock_agent_instance.messages = [{"role": "user", "content": [{"text": "Test"}]}]
        mock_agent_class.return_value = mock_agent_instance

        # Mock agent call
        mock_result = MagicMock()
        mock_result.metrics = None
        mock_agent_instance.return_value = mock_result

        # Create task state with long timeout to trigger system notification
        task_state = tasks.TaskState(
            task_id="notification_test",
            prompt="Test prompt",
            system_prompt="Test system prompt",
            timeout=120,  # > 60 seconds should trigger system notification
        )

        # Create parent agent with notify tool
        mock_parent = MagicMock()
        mock_parent.model = "test-model"
        mock_parent.tool_registry.registry = {}
        mock_parent.trace_attributes = {}

        # Mock the notify tool
        mock_notify_tool = MagicMock()
        mock_parent.tool.notify = mock_notify_tool

        # Run the task
        tasks.run_task(task_state, mock_parent)

        # Verify notification was called
        mock_notify_tool.assert_called_once()
        call_args = mock_notify_tool.call_args[1]
        assert "completed" in call_args["message"]
        assert call_args["title"] == "Task Complete"
        assert call_args["show_system_notification"] is True  # Should be True for timeout > 60

    @patch("strands_tools.tasks.Agent")
    def test_notification_failure_handling(self, mock_agent_class, temp_tasks_dir, cleanup_global_state):
        """Test notification failure handling - Lines 429-430."""
        # Setup mock agent
        mock_agent_instance = MagicMock()
        mock_agent_instance.messages = [{"role": "user", "content": [{"text": "Test"}]}]
        mock_agent_class.return_value = mock_agent_instance

        # Mock agent call
        mock_result = MagicMock()
        mock_result.metrics = None
        mock_agent_instance.return_value = mock_result

        # Create task state
        task_state = tasks.TaskState(
            task_id="notification_error_test", prompt="Test prompt", system_prompt="Test system prompt"
        )

        # Create parent agent with notify tool that raises exception
        mock_parent = MagicMock()
        mock_parent.model = "test-model"
        mock_parent.tool_registry.registry = {}
        mock_parent.trace_attributes = {}

        # Mock the notify tool to raise an exception
        mock_notify_tool = MagicMock()
        mock_notify_tool.side_effect = Exception("Notification failed")
        mock_parent.tool.notify = mock_notify_tool

        # Run the task - should handle notification failure gracefully
        tasks.run_task(task_state, mock_parent)

        # Task should complete successfully despite notification failure
        assert task_state.status == "completed"

    def test_timeout_handler_function(self, temp_tasks_dir, cleanup_global_state):
        """Test run_task_with_timeout function and timeout handler - Lines 436-454."""
        # Create task state with very short timeout
        task_state = tasks.TaskState(
            task_id="timeout_handler_test",
            prompt="Test prompt",
            system_prompt="Test system prompt",
            timeout=1,  # 1 second timeout
        )

        # Set initial status to running to test timeout handler
        task_state.update_status("running")
        tasks.task_states[task_state.task_id] = task_state

        # Mock a running thread
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        tasks.task_threads[task_state.task_id] = mock_thread

        # Mock run_task to take longer than timeout and NOT change status
        def slow_run_task(task_state, parent_agent):
            time.sleep(2)  # Sleep longer than timeout
            # Don't change status - let timeout handler do it

        # Patch run_task to use our slow version
        with patch("strands_tools.tasks.run_task", side_effect=slow_run_task):
            # Run with timeout - should trigger timeout handler
            tasks.run_task_with_timeout(task_state, None)

        # Give timeout handler a moment to execute
        time.sleep(0.1)

        # Verify timeout status was set by timeout handler
        assert task_state.status == "timeout"

    @patch("threading.Thread")
    def test_resume_task_thread_creation(self, mock_thread, temp_tasks_dir, mock_agent, cleanup_global_state):
        """Test resume task thread creation logic - Lines 831-841."""
        # Create paused task state
        task_state = tasks.TaskState(
            task_id="resume_thread_test", prompt="Test prompt", system_prompt="Test system prompt"
        )
        task_state.paused = True
        task_state.update_status("paused")
        tasks.task_states["resume_thread_test"] = task_state

        # Ensure no thread is currently running for this task
        if "resume_thread_test" in tasks.task_threads:
            del tasks.task_threads["resume_thread_test"]

        # Mock thread instance
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        # Resume the task
        tool_use = {"toolUseId": "test-resume-thread", "input": {"action": "resume", "task_id": "resume_thread_test"}}

        result = tasks.tasks(tool=tool_use, agent=mock_agent)

        # Verify successful resume
        assert result["status"] == "success"
        assert "resumed successfully" in result["content"][0]["text"]

        # Verify new thread was created and started
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()

        # Verify thread target is run_task_with_timeout
        thread_call = mock_thread.call_args
        assert thread_call[1]["target"] == tasks.run_task_with_timeout

    def test_task_state_loading_edge_cases(self, temp_tasks_dir, cleanup_global_state):
        """Test TaskState loading edge cases - Lines 782-784."""
        # Test loading task with missing messages file (should not crash)
        task_state = tasks.TaskState(
            task_id="edge_case_load_test", prompt="Test prompt", system_prompt="Test system prompt"
        )

        # Delete the messages file to test edge case
        task_state.messages_path.unlink()

        # Loading should work even without messages file
        loaded_state = tasks.TaskState.load("edge_case_load_test")

        assert loaded_state is not None
        assert loaded_state.task_id == "edge_case_load_test"
        # Should have default message history since messages file is missing
        assert len(loaded_state.message_history) >= 1

    @patch("strands_tools.tasks.Agent")
    def test_run_task_with_metrics(self, mock_agent_class, temp_tasks_dir, cleanup_global_state):
        """Test run_task with metrics handling - Lines covering metrics processing."""
        # Setup mock agent
        mock_agent_instance = MagicMock()
        mock_agent_instance.messages = [{"role": "user", "content": [{"text": "Test"}]}]
        mock_agent_class.return_value = mock_agent_instance

        # Mock agent call with metrics
        mock_result = MagicMock()
        mock_metrics = MagicMock()
        mock_result.metrics = mock_metrics
        mock_agent_instance.return_value = mock_result

        # Mock metrics_to_string function
        with patch("strands_tools.tasks.metrics_to_string") as mock_metrics_to_string:
            mock_metrics_to_string.return_value = "Input tokens: 10, Output tokens: 20"

            # Create task state
            task_state = tasks.TaskState(
                task_id="metrics_test", prompt="Test prompt", system_prompt="Test system prompt"
            )

            # Create mock parent
            mock_parent = MagicMock()
            mock_parent.model = "test-model"
            mock_parent.tool_registry.registry = {}
            mock_parent.trace_attributes = {}

            # Run the task
            tasks.run_task(task_state, mock_parent)

            # Verify metrics were processed
            mock_metrics_to_string.assert_called_with(mock_metrics)

            # Check that metrics were written to result file
            with open(task_state.result_path, "r") as f:
                content = f.read()
            assert "Input tokens: 10, Output tokens: 20" in content


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch("threading.Thread")
    def test_create_task_with_custom_timeout(self, mock_thread, temp_tasks_dir, cleanup_global_state):
        """Test creating task with custom timeout."""
        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.model = "anthropic.claude-3-sonnet-20240229-v1:0"
        mock_agent.tool_registry.registry = {}
        mock_agent.trace_attributes = {}

        tool_use = {
            "toolUseId": "test-timeout",
            "input": {
                "action": "create",
                "task_id": "timeout_test",
                "prompt": "Test prompt",
                "system_prompt": "Test system prompt",
                "timeout": 300,
            },
        }

        result = tasks.tasks(tool=tool_use, agent=mock_agent)

        assert result["status"] == "success"

        # Verify timeout was set correctly
        task_state = tasks.task_states["timeout_test"]
        assert task_state.timeout == 300

    def test_get_messages_task_not_found(self, temp_tasks_dir, cleanup_global_state):
        """Test get_messages action with non-existent task."""
        tool_use = {
            "toolUseId": "test-get-messages-not-found",
            "input": {"action": "get_messages", "task_id": "nonexistent"},
        }

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_pause_task_not_found(self, temp_tasks_dir, cleanup_global_state):
        """Test pausing a non-existent task."""
        tool_use = {"toolUseId": "test-pause-not-found", "input": {"action": "pause", "task_id": "nonexistent"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_resume_task_not_found(self, temp_tasks_dir, cleanup_global_state):
        """Test resuming a non-existent task."""
        tool_use = {"toolUseId": "test-resume-not-found", "input": {"action": "resume", "task_id": "nonexistent"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_pause_resume_missing_task_id(self, temp_tasks_dir, cleanup_global_state):
        """Test pause/resume actions with missing task_id."""
        for action in ["pause", "resume"]:
            tool_use = {"toolUseId": f"test-{action}-error", "input": {"action": action}}

            result = tasks.tasks(tool=tool_use)

            assert result["status"] == "error"
            assert "task_id is required" in result["content"][0]["text"]

    def test_add_message_missing_message(self, temp_tasks_dir, cleanup_global_state):
        """Test adding message without message content."""
        tool_use = {"toolUseId": "test-add-msg-error", "input": {"action": "add_message", "task_id": "test_task"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "error"
        assert "message is required" in result["content"][0]["text"]

    def test_get_result_missing_task_id(self, temp_tasks_dir, cleanup_global_state):
        """Test get_result action with missing task_id."""
        tool_use = {"toolUseId": "test-get-result-error", "input": {"action": "get_result"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "error"
        assert "task_id is required" in result["content"][0]["text"]

    def test_status_missing_task_id(self, temp_tasks_dir, cleanup_global_state):
        """Test status action with missing task_id."""
        tool_use = {"toolUseId": "test-status-error", "input": {"action": "status"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "error"
        assert "task_id is required" in result["content"][0]["text"]

    def test_stop_missing_task_id(self, temp_tasks_dir, cleanup_global_state):
        """Test stop action with missing task_id."""
        tool_use = {"toolUseId": "test-stop-error", "input": {"action": "stop"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "error"
        assert "task_id is required" in result["content"][0]["text"]

    def test_get_messages_missing_task_id(self, temp_tasks_dir, cleanup_global_state):
        """Test get_messages action with missing task_id."""
        tool_use = {"toolUseId": "test-get-messages-error", "input": {"action": "get_messages"}}

        result = tasks.tasks(tool=tool_use)

        assert result["status"] == "error"
        assert "task_id is required" in result["content"][0]["text"]
