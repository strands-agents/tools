"""
Background task management system for Strands Agent.

This module provides functionality to create, manage, and interact with background tasks
that run in parallel threads. Each task can be created, monitored, paused, resumed,
and continued by adding messages. Results are saved to the filesystem for persistence
between sessions.

Features:
- Create background tasks with specific prompts and system prompts
- Run multiple tasks in parallel threads
- Save task state and results to filesystem
- Retrieve task results and state
- Continue existing tasks by adding messages
- List all running and completed tasks
- Manage task lifecycle (pause, resume, stop)

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import tasks

agent = Agent(tools=[tasks])

# Create a new background task
result = agent.tool.tasks(
    action="create",
    task_id="research_task",
    prompt="Research quantum computing applications in healthcare",
    system_prompt="You are an expert in quantum computing and medical research."
)

# Check task status
status = agent.tool.tasks(
    action="status",
    task_id="research_task"
)

# Add a message to continue the task
continue_result = agent.tool.tasks(
    action="add_message",
    task_id="research_task",
    message="Focus on cancer treatment applications specifically."
)

# List all tasks
all_tasks = agent.tool.tasks(
    action="list"
)
```
"""

import json
import logging
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from strands import Agent
from strands.telemetry.metrics import metrics_to_string
from strands.types.tools import ToolResult, ToolUse

logger = logging.getLogger(__name__)

TOOL_SPEC = {
    "name": "tasks",
    "description": "Create and manage background tasks that run in parallel threads and persist to the filesystem",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create", "status", "list", "stop", 
                        "resume", "pause", "add_message", "get_result"
                    ],
                    "description": "Action to perform on tasks (create, status, list, stop, "
                    "resume, pause, add_message, get_result)",
                },
                "task_id": {
                    "type": "string",
                    "description": "Unique identifier for the task. If not provided for 'create' action, "
                    "one will be generated.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Initial prompt to start the task with (required for 'create' action)",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "System prompt for the task agent (required for 'create' action)",
                },
                "message": {
                    "type": "string",
                    "description": "Message to add to an existing task (required for 'add_message' action)",
                },
                "tools": {
                    "type": "array",
                    "description": "List of tool names to make available to the task agent",
                    "items": {"type": "string"},
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds for the task execution (default: 900)",
                },
            },
            "required": ["action"],
        }
    },
}

# Directory to store task states and results
TASKS_DIR = Path.cwd() / "tasks"
TASKS_DIR.mkdir(parents=True, exist_ok=True)

# Global dictionary to track running task threads
task_threads = {}
task_agents = {}
task_states = {}


class TaskState:
    """Class to track and manage task state."""

    def __init__(self, task_id: str, prompt: str, system_prompt: str, tools: List[str] = None, timeout: int = 900):
        self.task_id = task_id
        self.state_path = TASKS_DIR / f"{task_id}_state.json"
        self.result_path = TASKS_DIR / f"{task_id}_result.txt"
        self.messages_path = TASKS_DIR / f"{task_id}_messages.json"
        self.status = "initializing"
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
        self.initial_prompt = prompt
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.timeout = timeout  # Default timeout in seconds (15 minutes)
        self.paused = False
        self.message_history = [{"role": "user", "content": [{"text": prompt}]}]
        self.save_state()
        self.save_messages()

    def save_state(self):
        """Save task state to filesystem."""
        state = {
            "task_id": self.task_id,
            "status": self.status,
            "created_at": self.created_at,
            "last_updated": datetime.now().isoformat(),
            "initial_prompt": self.initial_prompt,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "timeout": self.timeout,
            "paused": self.paused,
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def save_messages(self):
        """Save message history to filesystem."""
        with open(self.messages_path, "w") as f:
            json.dump(self.message_history, f, indent=2)

    def append_result(self, result: str):
        """Append result to the result file."""
        with open(self.result_path, "a") as f:
            f.write(f"--- {datetime.now().isoformat()} ---\n")
            f.write(result)
            f.write("\n\n")

    def update_status(self, status: str):
        """Update task status and save state."""
        self.status = status
        self.last_updated = datetime.now().isoformat()
        self.save_state()

    def add_message(self, message: str, role: str = "user"):
        """Add a message to the history."""
        # Format the message properly as a content block
        content = [{"text": message}]
        self.message_history.append({"role": role, "content": content})
        self.save_messages()

    @classmethod
    def load(cls, task_id: str):
        """Load task state from filesystem."""
        state_path = TASKS_DIR / f"{task_id}_state.json"
        messages_path = TASKS_DIR / f"{task_id}_messages.json"

        if not state_path.exists():
            return None

        with open(state_path, "r") as f:
            state_data = json.load(f)

        task_state = cls(
            task_id=state_data["task_id"],
            prompt=state_data["initial_prompt"],
            system_prompt=state_data["system_prompt"],
            tools=state_data.get("tools", []),
            timeout=state_data.get("timeout", 900),
        )

        task_state.status = state_data["status"]
        task_state.created_at = state_data["created_at"]
        task_state.last_updated = state_data["last_updated"]
        task_state.paused = state_data.get("paused", False)

        if messages_path.exists():
            with open(messages_path, "r") as f:
                task_state.message_history = json.load(f)

        return task_state


def run_task(task_state: TaskState, parent_agent: Optional[Agent] = None):
    """Run a task in the background and save results."""
    start_time = time.time()

    try:
        # Check if task is paused
        if task_state.paused:
            task_state.append_result("Task is paused. Resume it to continue processing.")
            return

        # Update task status
        task_state.update_status("running")

        # Initialize tools from parent agent if available
        tools = []
        trace_attributes = {}
        if parent_agent:
            if task_state.tools:
                # Only load specified tools if provided
                for tool_name in task_state.tools:
                    if tool_name in parent_agent.tool_registry.registry:
                        tools.append(parent_agent.tool_registry.registry[tool_name])
            else:
                # Otherwise load all tools
                tools = list(parent_agent.tool_registry.registry.values())
            trace_attributes = parent_agent.trace_attributes

        # Initialize the agent
        agent = Agent(
            messages=task_state.message_history,
            tools=tools,
            system_prompt=task_state.system_prompt,
            trace_attributes=trace_attributes,
            callback_handler=None,
        )

        # Store agent in global dict for later interaction
        task_agents[task_state.task_id] = agent

        # Process the last message if it's from the user
        if task_state.message_history and task_state.message_history[-1]["role"] == "user":
            # Get the last user message content
            last_message_content = task_state.message_history[-1]["content"]
            if isinstance(last_message_content, list) and last_message_content:
                if "text" in last_message_content[0]:
                    user_message = last_message_content[0]["text"]

                    # Use the __call__ method to process the message
                    result = agent(user_message)

                    # Extract response
                    assistant_response = str(result.message)

                    # Save response to message history
                    task_state.add_message(assistant_response, "assistant")

                    # Append result to result file
                    task_state.append_result(assistant_response)

                    # Update metrics
                    if result.metrics:
                        metrics = result.metrics
                        metrics_text = metrics_to_string(metrics)
                        task_state.append_result(f"Metrics: {metrics_text}")

        # Update task status
        elapsed_time = time.time() - start_time
        task_state.update_status("completed")
        task_state.append_result(f"Task completed in {elapsed_time:.2f} seconds")

        # Send notification if parent agent has notify tool
        if parent_agent and hasattr(parent_agent.tool, "notify"):
            try:
                parent_agent.tool.notify(
                    message=f"Task '{task_state.task_id}' completed in {elapsed_time:.2f} seconds",
                    title="Task Complete",
                    category="tasks",
                    source=task_state.task_id,
                    show_system_notification=True
                    if task_state.timeout > 60
                    else False,  # Only show system notification for longer-running tasks
                )
            except Exception as e:
                logger.error(f"Failed to send task completion notification: {e}")

    except TimeoutError:
        logger.error(f"Task {task_state.task_id} timed out after {task_state.timeout} seconds")
        task_state.update_status("timeout")
        task_state.append_result(f"ERROR: Task timed out after {task_state.timeout} seconds")

        # Send timeout notification if parent agent has notify tool
        if parent_agent and hasattr(parent_agent.tool, "notify"):
            try:
                parent_agent.tool.notify(
                    message=f"Task '{task_state.task_id}' timed out after {task_state.timeout} seconds",
                    title="Task Timeout",
                    priority="high",
                    category="tasks",
                    source=task_state.task_id,
                    show_system_notification=True,
                )
            except Exception as e:
                logger.error(f"Failed to send task timeout notification: {e}")

    except Exception as e:
        # Get the full stack trace
        stack_trace = traceback.format_exc()
        logger.error(f"Error in task {task_state.task_id}: {str(e)}\n{stack_trace}")
        task_state.update_status("error")
        task_state.append_result(f"ERROR: {str(e)}\n\nStack Trace:\n{stack_trace}")

        # Send error notification if parent agent has notify tool
        if parent_agent and hasattr(parent_agent.tool, "notify"):
            try:
                parent_agent.tool.notify(
                    message=f"Error in task '{task_state.task_id}': {str(e)}",
                    title="Task Error",
                    priority="high",
                    category="tasks",
                    source=task_state.task_id,
                    show_system_notification=True,
                )
            except Exception as notify_err:
                logger.error(f"Failed to send task error notification: {notify_err}")


def run_task_with_timeout(task_state: TaskState, parent_agent: Optional[Agent] = None):
    """Run a task with timeout handling."""

    def timeout_handler():
        # This will be called if the task exceeds its timeout
        if task_id in task_threads and task_state.status == "running":
            task_state.update_status("timeout")
            task_state.append_result(f"ERROR: Task timed out after {task_state.timeout} seconds")

    task_id = task_state.task_id

    # Start a timer for the timeout
    timer = threading.Timer(task_state.timeout, timeout_handler)
    timer.daemon = True
    timer.start()

    try:
        # Run the actual task
        run_task(task_state, parent_agent)
    finally:
        # Cancel the timer if task completes before timeout
        timer.cancel()


def tasks(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Create and manage background tasks with persistence.

    This function creates and manages background tasks that run in parallel threads
    and persist their state and results to the filesystem. Tasks can be created,
    monitored, paused, resumed, and continued by adding messages.

    Args:
        tool (ToolUse): Tool use object containing the following:
            - action: Action to perform (create, status, list, stop, resume, pause, add_message, get_result)
            - task_id: Unique identifier for the task
            - prompt: Initial prompt to start the task with (for create action)
            - system_prompt: System prompt for the task agent (for create action)
            - message: Message to add to an existing task (for add_message action)
            - tools: List of tool names to make available to the task agent
        **kwargs (Any): Additional keyword arguments

    Returns:
        ToolResult: Dictionary containing status and response content
    """
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    action = tool_input["action"]
    task_id = tool_input.get("task_id")

    parent_agent = kwargs.get("agent")

    if action == "create":
        # Create a new task
        prompt = tool_input.get("prompt")
        system_prompt = tool_input.get("system_prompt")
        tools = tool_input.get("tools", [])

        if not prompt:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "Error: prompt is required for create action"}],
            }

        if not system_prompt:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "Error: system_prompt is required for create action"}],
            }

        # Generate a task_id if not provided
        if not task_id:
            task_id = f"task_{uuid.uuid4().hex[:8]}"

        # Check if task already exists
        if task_id in task_threads and task_threads[task_id].is_alive():
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error: Task with ID '{task_id}' already exists and is running"}],
            }

        # Create task state
        task_state = TaskState(task_id, prompt, system_prompt, tools)
        task_states[task_id] = task_state

        # Add timeout if provided
        if "timeout" in tool_input:
            task_state.timeout = tool_input.get("timeout")

        # Start task in a new thread
        thread = threading.Thread(target=run_task_with_timeout, args=(task_state, parent_agent), name=f"task-{task_id}")
        thread.daemon = True
        thread.start()

        # Store thread reference
        task_threads[task_id] = thread

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {"text": f"Task '{task_id}' created and started in background"},
                {"text": f"Initial prompt: {prompt[:100]}..."},
                {"text": f"Results will be saved to: {task_state.result_path}"},
            ],
        }

    elif action == "status":
        if not task_id:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "Error: task_id is required for status action"}],
            }

        # Try to get task state from memory or load from filesystem
        task_state = task_states.get(task_id)
        if not task_state:
            task_state = TaskState.load(task_id)

        if not task_state:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error: Task with ID '{task_id}' not found"}],
            }

        # Check if thread is still running
        is_running = task_id in task_threads and task_threads[task_id].is_alive()

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {"text": f"Task '{task_id}' status: {task_state.status}"},
                {"text": f"Created: {task_state.created_at}"},
                {"text": f"Last updated: {task_state.last_updated}"},
                {"text": f"Thread running: {'Yes' if is_running else 'No'}"},
                {"text": f"Result file: {task_state.result_path}"},
            ],
        }

    elif action == "list":
        # Collect all task IDs from memory and filesystem
        all_task_ids = set(task_states.keys())

        # Add tasks from filesystem
        for path in TASKS_DIR.glob("*_state.json"):
            task_id = path.name.replace("_state.json", "")
            all_task_ids.add(task_id)

        if not all_task_ids:
            return {"toolUseId": tool_use_id, "status": "success", "content": [{"text": "No tasks found"}]}

        # Build task list with status
        tasks_info = []
        for tid in sorted(all_task_ids):
            # Get task state
            state = task_states.get(tid)
            if not state:
                state = TaskState.load(tid)
                if not state:
                    continue

            # Check if thread is running
            is_running = tid in task_threads and task_threads[tid].is_alive()
            running_status = "Running" if is_running else "Not running"

            tasks_info.append(f"Task '{tid}': Status={state.status}, {running_status}, Created={state.created_at}")

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": f"Found {len(tasks_info)} tasks:"}, {"text": "\n".join(tasks_info)}],
        }

    elif action == "add_message":
        if not task_id:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "Error: task_id is required for add_message action"}],
            }

        message = tool_input.get("message")
        if not message:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "Error: message is required for add_message action"}],
            }

        # Try to get task state from memory or load from filesystem
        task_state = task_states.get(task_id)
        if not task_state:
            task_state = TaskState.load(task_id)

        if not task_state:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error: Task with ID '{task_id}' not found"}],
            }

        # Add message to history
        task_state.add_message(message)

        # If task is completed or not running, start a new thread
        is_running = task_id in task_threads and task_threads[task_id].is_alive()
        if not is_running:
            thread = threading.Thread(target=run_task, args=(task_state, parent_agent), name=f"task-{task_id}")
            thread.daemon = True
            thread.start()

            # Store thread reference
            task_threads[task_id] = thread

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [
                    {"text": f"Message added to task '{task_id}' and processing started"},
                    {"text": f"Results will be saved to: {task_state.result_path}"},
                ],
            }
        else:
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [
                    {"text": f"Message added to task '{task_id}'"},
                    {
                        "text": "Note: Task is already running, message will be processed "
                        "after current execution completes"
                    },
                ],
            }

    elif action == "get_result":
        if not task_id:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "Error: task_id is required for get_result action"}],
            }

        result_path = TASKS_DIR / f"{task_id}_result.txt"
        if not result_path.exists():
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error: No results found for task '{task_id}'"}],
            }

        # Read result file
        with open(result_path, "r") as f:
            result_content = f.read()

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": f"Results for task '{task_id}':"}, {"text": result_content}],
        }

    elif action == "stop":
        if not task_id:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "Error: task_id is required for stop action"}],
            }

        # Check if task is running
        if task_id not in task_threads or not task_threads[task_id].is_alive():
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": f"Task '{task_id}' is not running"}],
            }

        # Update task status
        if task_id in task_states:
            task_states[task_id].update_status("stopped")
        else:
            task_state = TaskState.load(task_id)
            if task_state:
                task_state.update_status("stopped")

        # Cannot directly stop a thread in Python, but we can mark it as stopped
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": f"Task '{task_id}' marked as stopped"}],
        }

    elif action == "pause" or action == "resume":
        if not task_id:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "Error: task_id is required for pause/resume action"}],
            }

        # Try to get task state from memory or load from filesystem
        task_state = task_states.get(task_id)
        if not task_state:
            task_state = TaskState.load(task_id)

        if not task_state:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Error: Task with ID '{task_id}' not found"}],
            }

        # Update pause state
        if action == "pause":
            task_state.paused = True
            task_state.update_status("paused")
            task_state.append_result("Task paused by user request")
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": f"Task '{task_id}' paused successfully"}],
            }
        else:  # resume
            task_state.paused = False
            old_status = task_state.status
            task_state.update_status("resuming")
            task_state.append_result("Task resumed by user request")

            # Check if we need to start a new thread
            is_running = task_id in task_threads and task_threads[task_id].is_alive()
            if not is_running:
                thread = threading.Thread(
                    target=run_task_with_timeout, args=(task_state, parent_agent), name=f"task-{task_id}"
                )
                thread.daemon = True
                thread.start()

                # Store thread reference
                task_threads[task_id] = thread

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [
                    {"text": f"Task '{task_id}' resumed successfully"},
                    {"text": f"Previous status: {old_status}"},
                ],
            }

    else:
        return {"toolUseId": tool_use_id, "status": "error", "content": [{"text": f"Unknown action: {action}"}]}
