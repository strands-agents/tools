"""
Workflow orchestration tool for managing parallel AI tasks with background execution.
Supports task dependencies, concurrent execution, and real-time status tracking.
"""

import json
import logging
import os
import random
import textwrap
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from threading import Lock, RLock
from typing import Any, Dict, List, Optional

from rich.table import Table
from strands import Agent
from strands.types.tools import ToolResult, ToolUse
from tenacity import retry, stop_after_attempt, wait_exponential
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

# Constants
WORKFLOW_DIR = Path.cwd() / "workflows"
os.makedirs(WORKFLOW_DIR, exist_ok=True)

# Default thread pool settings
MIN_THREADS = 2
MAX_THREADS = 8
CPU_THRESHOLD = 80  # CPU usage threshold for scaling down

TOOL_SPEC = {
    "name": "workflow",
    "description": textwrap.dedent("""
        Advanced workflow orchestration system for parallel AI task execution featuring:
        1. Task Management:
            - Parallel execution with dynamic thread pooling
            - Priority-based scheduling (1-5 levels)
            - Complex dependency resolution
            - Timeout and resource controls

        2. Resource Optimization:
            - Automatic scaling (2-8 threads)
            - Rate limiting with backoff
            - Resource-aware task distribution

        3. Reliability Features:
            - Persistent state storage
            - Automatic error recovery
            - Real-time file monitoring
            - Task state preservation

        4. Monitoring & Control:
            - Detailed status tracking
            - Progress reporting
            - Resource utilization metrics
            - Task timing statistics

        Required fields: task_id, description
        Optional: system_prompt, dependencies, timeout, priority

        Supports file-based persistence (~/.strands/workflows), real-time monitoring, and resource-optimized parallel
        execution.
    """).strip(),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create",
                        "list",
                        "start",
                        "pause",
                        "resume",
                        "status",
                        "delete",
                    ],
                    "description": "Action to perform on workflows",
                },
                "workflow_id": {
                    "type": "string",
                    "description": "Unique identifier for the workflow",
                },
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "Unique identifier for the task",
                            },
                            "description": {
                                "type": "string",
                                "description": "Task description for AI execution",
                            },
                            "system_prompt": {
                                "type": "string",
                                "description": "Custom system prompt for the task",
                            },
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of task IDs this task depends on",
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Optional timeout in seconds",
                            },
                            "priority": {
                                "type": "integer",
                                "description": "Task priority (1-5, higher is more important)",
                                "minimum": 1,
                                "maximum": 5,
                            },
                        },
                        "required": ["task_id", "description"],
                    },
                    "description": "List of tasks in the workflow",
                },
            },
            "required": ["action"],
        }
    },
}


class WorkflowFileHandler(FileSystemEventHandler):
    def __init__(self, manager):
        self.manager = manager
        super().__init__()

    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".json"):
            workflow_id = Path(event.src_path).stem
            self.manager.load_workflow(workflow_id)


class TaskExecutor:
    def __init__(self, min_workers=MIN_THREADS, max_workers=MAX_THREADS):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = Queue()
        self.active_tasks = set()
        self.lock = Lock()
        self.results = {}
        self.start_times = {}  # Track task start times
        self.active_workers = 0  # Track number of active workers

    def submit_task(self, task_id: str, task_func, *args, **kwargs):
        with self.lock:
            if task_id in self.active_tasks:
                return None
            future = self._executor.submit(task_func, *args, **kwargs)
            self.active_tasks.add(task_id)
            self.start_times[task_id] = time.time()
            self.active_workers += 1

            # Monitor task completion
            def task_done_callback(fut):
                with self.lock:
                    self.active_workers -= 1

            future.add_done_callback(task_done_callback)
            return future

    def submit_tasks(self, tasks):
        """Submit multiple tasks at once and return their futures."""
        futures = {}
        for task_id, task_func, args, kwargs in tasks:
            future = self.submit_task(task_id, task_func, *args, **kwargs)
            if future:
                futures[task_id] = future
        return futures

    def get_result(self, task_id: str):
        return self.results.get(task_id)

    def task_completed(self, task_id: str, result):
        with self.lock:
            self.results[task_id] = result
            self.active_tasks.remove(task_id)

    def shutdown(self):
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)


class WorkflowManager:
    _workflows: Dict[str, Dict] = {}
    _observer = None
    _watch_paths = set()
    _instance = None

    def __new__(cls, tool_context: Dict[str, Any]):
        if cls._instance is None:
            cls._instance = super(WorkflowManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, tool_context: Dict[str, Any]):
        if not hasattr(self, "initialized"):
            # Initialize core attributes
            extra_kwargs = {}
            parent_agent = tool_context.get("agent", None)
            if parent_agent:
                extra_kwargs["tools"] = list(parent_agent.tool_registry.registry.values())
                extra_kwargs["trace_attributes"] = parent_agent.trace_attributes
                extra_kwargs["callback_handler"] = parent_agent.callback_handler
                extra_kwargs["model"] = parent_agent.model
            self.system_prompt = tool_context["system_prompt"]
            self.inference_config = tool_context["inference_config"]
            self.messages = tool_context["messages"]
            self.tool_config = tool_context["tool_config"]

            # Initialize task executor
            self.task_executor = TaskExecutor()

            # Initialize base agent for task execution
            self.base_agent = Agent(system_prompt=self.system_prompt, **extra_kwargs)

            # Start file watching if not already started
            if not self._observer:
                self._start_file_watching()

            # Load existing workflows
            self._load_all_workflows()
            self.initialized = True

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """Cleanup observers and executors"""
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join()
                self._observer = None
                self._watch_paths.clear()
            except BaseException:
                pass

        if hasattr(self, "task_executor"):
            self.task_executor.shutdown()

    def _start_file_watching(self):
        """Initialize and start the file system observer"""
        try:
            if self._observer is None:
                self._observer = Observer()
                if WORKFLOW_DIR not in self._watch_paths:
                    self._observer.schedule(WorkflowFileHandler(self), WORKFLOW_DIR, recursive=False)
                    self._watch_paths.add(WORKFLOW_DIR)
                    self._observer.start()
        except Exception as e:
            logger.error(f"\nError starting file watcher: {str(e)}")
            self.cleanup()

    def _load_all_workflows(self):
        """Load all workflow files from disk"""
        for file_path in Path(WORKFLOW_DIR).glob("*.json"):
            workflow_id = file_path.stem
            self.load_workflow(workflow_id)

    def load_workflow(self, workflow_id: str) -> Optional[Dict]:
        """Load a workflow from its JSON file"""
        try:
            file_path = os.path.join(WORKFLOW_DIR, f"{workflow_id}.json")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    self._workflows[workflow_id] = json.load(f)
                return self._workflows[workflow_id]
        except Exception as e:
            logger.error(f"Error loading workflow {workflow_id}: {str(e)}")
        return None

    def store_workflow(self, workflow_id: str, workflow_data: Dict, tool_use_id: str) -> Dict:
        """Store workflow data in memory and to file."""
        try:
            # Store in memory
            self._workflows[workflow_id] = workflow_data

            # Store to file
            file_path = os.path.join(WORKFLOW_DIR, f"{workflow_id}.json")
            with open(file_path, "w") as f:
                json.dump(workflow_data, f, indent=2)

            return {"status": "success"}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error storing workflow: {error_msg}")
            return {"status": "error", "error": error_msg}

    def get_workflow(self, workflow_id: str) -> Optional[Dict]:
        """Retrieve workflow data from memory or file."""
        workflow = self._workflows.get(workflow_id)
        if workflow is None:
            return self.load_workflow(workflow_id)
        return workflow

    def create_workflow(self, workflow_id: str, tasks: List[Dict], tool_use_id: str) -> Dict:
        """Create a new workflow with the given tasks."""
        try:
            if not workflow_id:
                workflow_id = str(uuid.uuid4())

            # Add default priorities if not specified
            for task in tasks:
                if "priority" not in task:
                    task["priority"] = 3  # Default priority

            workflow = {
                "workflow_id": workflow_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "created",
                "tasks": tasks,
                "current_task_index": 0,
                "task_results": {
                    task["task_id"]: {
                        "status": "pending",
                        "result": None,
                        "priority": task.get("priority", 3),
                    }
                    for task in tasks
                },
                "parallel_execution": True,  # Enable parallel execution by default
            }

            store_result = self.store_workflow(workflow_id, workflow, tool_use_id)
            if store_result["status"] == "error":
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to create workflow: {store_result['error']}"}],
                }
            return {
                "status": "success",
                "content": [{"text": f"Created workflow {workflow_id} with {len(tasks)} tasks"}],
            }

        except Exception as e:
            error_msg = f"Error creating workflow: {str(e)}"
            logger.error(f"\nError: {error_msg}")
            return {"status": "error", "content": [{"text": error_msg}]}

    # Rate limiting lock and configuration
    _rate_limit_lock = RLock()
    _last_request_time = 0
    _MIN_REQUEST_INTERVAL = 0.1  # Minimum time between requests (100ms)
    _MAX_BACKOFF = 30  # Maximum backoff time in seconds

    def _wait_for_rate_limit(self):
        """Implements rate limiting for API calls."""
        with self._rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < self._MIN_REQUEST_INTERVAL:
                sleep_time = self._MIN_REQUEST_INTERVAL - time_since_last
                time.sleep(sleep_time)
            self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        reraise=True,
    )
    def execute_task(self, task: Dict, workflow: Dict, tool_use_id: str) -> Dict:
        """Execute a single task using use_llm with rate limiting and retries."""
        try:
            # Build context from dependent tasks
            context = []
            if task.get("dependencies"):
                for dep_id in task["dependencies"]:
                    dep_result = workflow["task_results"].get(dep_id, {})
                    if dep_result.get("status") == "completed" and dep_result.get("result"):
                        # Format the dependency results
                        dep_content = [msg.get("text", "") for msg in dep_result["result"]]
                        context.append(f"Results from {dep_id}:\n" + "\n".join(dep_content))

            # Build comprehensive task prompt with context
            task_prompt = task["description"]
            if context:
                task_prompt = "Previous task results:\n" + "\n\n".join(context) + "\n\nTask:\n" + task_prompt

            # Add jitter to prevent thundering herd
            jitter = random.uniform(0, 1)
            time.sleep(jitter)

            # Apply rate limiting before making API call
            self._wait_for_rate_limit()

            # Get task-specific overrides if provided
            task_kwargs = {}
            if task.get("system_prompt"):
                task_kwargs["system_prompt"] = task["system_prompt"]

            # Execute task using the base agent with overrides
            result = self.base_agent(task_prompt, **task_kwargs)

            # Extract response content - handle both dict and custom object return types
            try:
                # If result is a dict or has .get() method
                content = result.get("content", [])
            except AttributeError:
                # If result is an object with .content attribute
                content = getattr(result, "content", [])

            # Extract stop_reason - handle both dict and custom object return types
            try:
                # If result is a dict or has .get() method
                stop_reason = result.get("stop_reason", "")
            except AttributeError:
                # If result is an object with .stop_reason attribute
                stop_reason = getattr(result, "stop_reason", "")

            # Update task status
            status = "success" if stop_reason != "error" else "error"
            return {
                "toolUseId": tool_use_id,
                "status": status,
                "content": content,
            }

        except Exception as e:
            error_msg = f"Error executing task {task['task_id']}: {str(e)}"
            logger.error(f"\nError: {error_msg}")
            if "ThrottlingException" in str(e):
                # Log retry attempt before re-raising
                logger.error(f"\n[red]Task {task['task_id']} hit throttling, will retry with exponential backoff[/red]")
                raise
            return {"status": "error", "content": [{"text": error_msg}]}

    def get_ready_tasks(self, workflow: Dict) -> List[Dict]:
        """Get list of tasks that are ready to execute (dependencies satisfied)."""
        ready_tasks = []
        for task in workflow["tasks"]:
            task_id = task["task_id"]
            # Skip completed or running tasks
            if workflow["task_results"][task_id]["status"] != "pending":
                continue

            # Check dependencies
            dependencies_met = True
            if task.get("dependencies"):
                for dep_id in task["dependencies"]:
                    if workflow["task_results"][dep_id]["status"] != "completed":
                        dependencies_met = False
                        break

            if dependencies_met:
                ready_tasks.append(task)

        # Sort by priority (higher priority first)
        ready_tasks.sort(key=lambda x: x.get("priority", 3), reverse=True)
        return ready_tasks

    def start_workflow(self, workflow_id: str, tool_use_id: str) -> Dict:
        """Start or resume workflow execution with true parallel processing."""
        try:
            # Get workflow data
            workflow = self.get_workflow(workflow_id)
            if not workflow:
                return {
                    "status": "error",
                    "content": [{"text": f"Workflow {workflow_id} not found"}],
                }

            # Update status
            workflow["status"] = "running"
            self.store_workflow(workflow_id, workflow, tool_use_id)

            # Track completed tasks and active futures
            completed_tasks = set()
            active_futures = {}

            while len(completed_tasks) < len(workflow["tasks"]):
                # Get all ready tasks
                ready_tasks = self.get_ready_tasks(workflow)

                # Prepare tasks for parallel submission with batching
                tasks_to_submit = []
                max_concurrent = self.task_executor.max_workers
                current_batch_size = min(len(ready_tasks), max_concurrent - len(active_futures))

                for task in ready_tasks[:current_batch_size]:
                    task_id = task["task_id"]
                    if task_id not in active_futures and task_id not in completed_tasks:
                        tasks_to_submit.append(
                            (
                                task_id,
                                self.execute_task,
                                (task, workflow, tool_use_id),
                                {},
                            )
                        )

                # Submit batch of tasks in parallel
                if tasks_to_submit:
                    new_futures = self.task_executor.submit_tasks(tasks_to_submit)
                    active_futures.update(new_futures)

                # Wait for any task to complete
                if active_futures:
                    done, _ = wait(active_futures.values(), return_when=FIRST_COMPLETED)

                    # Process completed tasks
                    completed_task_ids = []
                    for task_id, future in active_futures.items():
                        if future in done:
                            completed_task_ids.append(task_id)
                            try:
                                result = future.result()
                                # Ensure content uses valid type
                                content = []
                                for item in result.get("content", []):
                                    if isinstance(item, dict):
                                        # Convert any 'file' type to 'text'
                                        if "file" in item:
                                            content.append({"text": str(item["file"])})
                                        elif any(
                                            key in item
                                            for key in [
                                                "text",
                                                "json",
                                                "image",
                                                "document",
                                                "video",
                                            ]
                                        ):
                                            content.append(item)
                                        else:
                                            content.append({"text": str(item)})
                                    else:
                                        content.append({"text": str(item)})

                                workflow["task_results"][task_id] = {
                                    "status": ("completed" if result["status"] == "success" else "error"),
                                    "result": content,
                                    "completed_at": datetime.now(timezone.utc).isoformat(),
                                }
                                completed_tasks.add(task_id)
                            except Exception as e:
                                workflow["task_results"][task_id] = {
                                    "status": "error",
                                    "result": [{"text": f"Task execution error: {str(e)}"}],
                                    "completed_at": datetime.now(timezone.utc).isoformat(),
                                }
                                completed_tasks.add(task_id)
                                logger.error(f"\nTask {task_id} failed with error: {str(e)}")

                    # Remove completed tasks from active futures
                    for task_id in completed_task_ids:
                        del active_futures[task_id]

                # Store updated workflow state
                self.store_workflow(workflow_id, workflow, tool_use_id)

                # Brief pause to prevent tight loop
                time.sleep(0.1)

            # Workflow completed
            workflow["status"] = "completed"
            self.store_workflow(workflow_id, workflow, tool_use_id)
            return {
                "status": "success",
                "content": [{"text": f"Workflow {workflow_id} completed successfully"}],
            }

        except Exception as e:
            error_trace = traceback.format_exc()
            error_msg = f"Error in workflow execution: {str(e)}\n{error_trace}"
            logger.error(f"\nError: {error_msg}")
            return {"status": "error", "content": [{"text": error_msg}]}

    def list_workflows(self, tool_use_id: str) -> Dict:
        """List all workflows and their current status."""
        try:
            # Refresh from files first
            self._load_all_workflows()

            if self._workflows:
                table = Table(show_header=True)
                table.add_column("ID")
                table.add_column("Status")
                table.add_column("Tasks")
                table.add_column("Created")
                table.add_column("Parallel")

                for workflow_id, workflow_data in self._workflows.items():
                    table.add_row(
                        workflow_id,
                        workflow_data["status"],
                        str(len(workflow_data["tasks"])),
                        workflow_data["created_at"].split("T")[0],
                        ("Yes" if workflow_data.get("parallel_execution", True) else "No"),
                    )
                return {
                    "status": "success",
                    "content": [{"text": "Workflows listed successfully"}],
                }
            else:
                return {
                    "status": "success",
                    "content": [{"text": "No workflows found"}],
                }

        except Exception as e:
            error_msg = f"Error listing workflows: {str(e)}"
            logger.error(f"\nError: {error_msg}")
            return {"status": "error", "content": [{"text": error_msg}]}

    def get_workflow_status(self, workflow_id: str, tool_use_id: str) -> Dict:
        """Get detailed status of a workflow including task results and timing information."""
        try:
            workflow = self.get_workflow(workflow_id)
            if not workflow:
                return {
                    "status": "error",
                    "content": [{"text": f"Workflow {workflow_id} not found"}],
                }

            # Create status table
            table = Table(show_header=True)
            table.add_column("Task ID")
            table.add_column("Status")
            table.add_column("Priority")
            table.add_column("Dependencies")
            table.add_column("Duration")
            table.add_column("Completed At")

            tasks_details = []
            active_count = 0
            completed_count = 0
            pending_count = 0
            error_count = 0

            for task in workflow["tasks"]:
                task_id = task["task_id"]
                task_result = workflow["task_results"].get(task_id, {})

                dependencies = task.get("dependencies", [])
                completed_at = task_result.get("completed_at", "N/A")
                status = task_result.get("status", "pending")
                priority = task.get("priority", 3)

                # Calculate duration if available
                duration = "N/A"
                if status == "completed" and task_id in self.task_executor.start_times:
                    start_time = self.task_executor.start_times[task_id]
                    if completed_at != "N/A":
                        end_time = datetime.fromisoformat(completed_at).timestamp()
                        duration = f"{(end_time - start_time):.2f}s"

                # Update counters
                if status == "pending":
                    pending_count += 1
                elif status == "completed":
                    completed_count += 1
                elif status == "error":
                    error_count += 1
                if task_id in self.task_executor.active_tasks:
                    active_count += 1

                table.add_row(
                    task_id,
                    status,
                    str(priority),
                    ", ".join(dependencies) if dependencies else "None",
                    duration,
                    completed_at.split("T")[0] if completed_at != "N/A" else "N/A",
                )

                # Add detailed task results
                task_detail = [
                    f"\nTask: {task_id}",
                    f"Description: {task.get('description', 'N/A')}",
                    f"Status: {status}",
                    f"Priority: {priority}",
                    f"Duration: {duration}",
                ]

                if task_result.get("result"):
                    task_detail.append("Results:")
                    for msg in task_result["result"]:
                        if isinstance(msg, dict) and "text" in msg:
                            task_detail.append(f"  {msg['text']}")

                tasks_details.append("\n".join(task_detail))

            # Calculate overall progress
            total_tasks = len(workflow["tasks"])
            progress_pct = (completed_count / total_tasks) * 100 if total_tasks > 0 else 0

            status_text = [
                f"Workflow ID: {workflow_id}",
                f"Overall Status: {workflow['status']}",
                f"Progress: {progress_pct:.1f}% ({completed_count}/{total_tasks})",
                f"Active Tasks: {active_count}",
                f"Completed Tasks: {completed_count}",
                f"Pending Tasks: {pending_count}",
                f"Failed Tasks: {error_count}",
                f"Active Workers: {self.task_executor.active_workers}/{self.task_executor.max_workers}",
                "\nTask Details:",
                "\n".join(tasks_details),
            ]

            return {"status": "success", "content": [{"text": "\n".join(status_text)}]}

        except Exception as e:
            error_msg = f"Error getting workflow status: {str(e)}"
            logger.error(f"\nError: {error_msg}")
            return {"status": "error", "content": [{"text": error_msg}]}

    def delete_workflow(self, workflow_id: str, tool_use_id: str) -> Dict:
        """Delete a workflow and its results."""
        try:
            # Remove from memory
            if workflow_id in self._workflows:
                del self._workflows[workflow_id]

            # Remove file if exists
            file_path = os.path.join(WORKFLOW_DIR, f"{workflow_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                return {
                    "status": "success",
                    "content": [{"text": f"Workflow {workflow_id} deleted successfully"}],
                }
            else:
                return {
                    "status": "error",
                    "content": [{"text": f"Workflow {workflow_id} not found"}],
                }

        except Exception as e:
            error_msg = f"Error deleting workflow: {str(e)}"
            logger.error(f"\nError: {error_msg}")
            return {"status": "error", "content": [{"text": error_msg}]}


def workflow(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Main workflow tool implementation."""
    system_prompt = kwargs.get("system_prompt")
    inference_config = kwargs.get("inference_config")
    messages = kwargs.get("messages")
    tool_config = kwargs.get("tool_config")
    agent = kwargs.get("agent")

    try:
        tool_use_id = tool.get("toolUseId", str(uuid.uuid4()))
        tool_input = tool.get("input", {})
        action = tool_input.get("action")

        # Initialize workflow manager
        manager = WorkflowManager(
            {
                "system_prompt": system_prompt,
                "inference_config": inference_config,
                "messages": messages,
                "tool_config": tool_config,
                "agent": agent,
            }
        )

        if action == "create":
            workflow_id = tool_input.get("workflow_id", str(uuid.uuid4()))
            if not tool_input.get("tasks"):
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": "Tasks are required for create action"}],
                }

            result = manager.create_workflow(workflow_id, tool_input["tasks"], tool_use_id)

        elif action == "start":
            if not tool_input.get("workflow_id"):
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": "workflow_id is required for start action"}],
                }

            result = manager.start_workflow(tool_input["workflow_id"], tool_use_id)

        elif action == "list":
            result = manager.list_workflows(tool_use_id)

        elif action == "status":
            if not tool_input.get("workflow_id"):
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": "workflow_id is required for status action"}],
                }

            result = manager.get_workflow_status(tool_input["workflow_id"], tool_use_id)

        elif action == "delete":
            if not tool_input.get("workflow_id"):
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": "workflow_id is required for delete action"}],
                }

            result = manager.delete_workflow(tool_input["workflow_id"], tool_use_id)

        else:
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Unknown action: {action}"}],
            }

        return {
            "toolUseId": tool_use_id,
            "status": result["status"],
            "content": result["content"],
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{error_trace}"
        logger.error(f"\nError in workflow tool: {error_msg}")
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": error_msg}],
        }
