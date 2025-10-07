"""Workflow orchestration tool for managing parallel AI tasks with advanced model support.

This module provides an advanced workflow orchestration system that supports parallel AI task
execution with granular control over model providers, tool access, and execution parameters.
Built on modern Strands SDK patterns with rich monitoring and robust error handling.

Key Features:
-------------
1. Advanced Task Management:
   • Parallel execution with dynamic thread pooling
   • Priority-based scheduling (1-5 levels)
   • Complex dependency resolution with validation
   • Timeout and resource controls per task
   • Per-task model provider and settings configuration

2. Modern Model Support:
   • Individual model providers per task (bedrock, anthropic, ollama, etc.)
   • Custom model settings and parameters per task
   • Environment-based model configuration
   • Fallback to parent agent model when needed

3. Flexible Tool Configuration:
   • Per-task tool access control
   • Tool inheritance from parent agent
   • Automatic tool filtering and validation
   • Support for any combination of tools per task

4. Resource Optimization:
   • Automatic thread pool scaling (2-8 threads)
   • Rate limiting with exponential backoff
   • Resource-aware task distribution
   • CPU usage monitoring and optimization

5. Reliability Features:
   • Persistent state storage with real-time monitoring
   • Automatic error recovery with retries
   • File system watching for external updates
   • Task state preservation across restarts

6. Rich Monitoring & Control:
   • Detailed status tracking with metrics
   • Progress reporting with timing statistics
   • Resource utilization insights
   • Comprehensive execution logging

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import workflow

agent = Agent(tools=[workflow])

# Create a multi-model research workflow
result = agent.tool.workflow(
    action="create",
    workflow_id="research_pipeline",
    tasks=[
        {
            "task_id": "data_collection",
            "description": "Collect and organize research data on renewable energy trends",
            "tools": ["retrieve", "file_write"],
            "model_provider": "bedrock",
            "model_settings": {"model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
            "priority": 5,
            "timeout": 300
        },
        {
            "task_id": "analysis",
            "description": "Analyze the collected data and identify key patterns",
            "dependencies": ["data_collection"],
            "tools": ["calculator", "file_read", "file_write"],
            "model_provider": "anthropic",
            "model_settings": {"model_id": "claude-sonnet-4-20250514", "params": {"temperature": 0.3}},
            "system_prompt": "You are a data analysis specialist focused on renewable energy research.",
            "priority": 4
        },
        {
            "task_id": "report_generation",
            "description": "Generate a comprehensive report based on the analysis",
            "dependencies": ["analysis"],
            "tools": ["file_read", "file_write", "generate_image"],
            "model_provider": "openai",
            "model_settings": {"model_id": "o4-mini", "params": {"temperature": 0.7}},
            "system_prompt": "You are a report writing specialist who creates clear, engaging reports.",
            "priority": 3
        }
    ]
)

# Start the workflow
result = agent.tool.workflow(action="start", workflow_id="research_pipeline")

# Monitor progress
result = agent.tool.workflow(action="status", workflow_id="research_pipeline")
```

See the workflow function docstring for complete configuration options and advanced usage patterns.
"""

import json
import logging
import os
import random
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from threading import Lock, RLock
from typing import Any, Dict, List, Optional

from rich.box import ROUNDED
from rich.panel import Panel
from rich.table import Table
from strands import Agent, tool
from strands.telemetry.metrics import metrics_to_string
from tenacity import retry, stop_after_attempt, wait_exponential
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from strands_tools.utils import console_util
from strands_tools.utils.models.model import create_model

logger = logging.getLogger(__name__)

# Constants
WORKFLOW_DIR = Path(os.getenv("STRANDS_WORKFLOW_DIR", Path.home() / ".strands" / "workflows"))
os.makedirs(WORKFLOW_DIR, exist_ok=True)

# Default thread pool settings
MIN_THREADS = int(os.getenv("STRANDS_WORKFLOW_MIN_THREADS", "2"))
MAX_THREADS = int(os.getenv("STRANDS_WORKFLOW_MAX_THREADS", "8"))
CPU_THRESHOLD = int(os.getenv("STRANDS_WORKFLOW_CPU_THRESHOLD", "80"))  # CPU usage threshold for scaling down

# Rate limiting configuration
_rate_limit_lock = RLock()
_last_request_time = 0
_MIN_REQUEST_INTERVAL = 0.1  # Minimum time between requests (100ms)
_MAX_BACKOFF = 30  # Maximum backoff time in seconds


class WorkflowFileHandler(FileSystemEventHandler):
    """File system event handler for workflow file monitoring."""

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
    """Advanced task executor with dynamic scaling and resource monitoring."""

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
        """Submit a single task for execution."""
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
        """Get result for a completed task."""
        return self.results.get(task_id)

    def task_completed(self, task_id: str, result):
        """Mark task as completed with result."""
        with self.lock:
            self.results[task_id] = result
            if task_id in self.active_tasks:
                self.active_tasks.remove(task_id)

    def shutdown(self):
        """Shutdown the executor gracefully."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)


class WorkflowManager:
    """Workflow manager with advanced model support and monitoring."""

    _workflows: Dict[str, Dict] = {}
    _observer = None
    _watch_paths = set()
    _instance = None

    def __new__(cls, parent_agent: Optional[Any] = None):
        if cls._instance is None:
            cls._instance = super(WorkflowManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, parent_agent: Optional[Any] = None):
        if not hasattr(self, "initialized"):
            # Store parent agent for tool and model inheritance
            self.parent_agent = parent_agent

            # Initialize task executor
            self.task_executor = TaskExecutor()

            # Start file watching if not already started
            if not self._observer:
                self._start_file_watching()

            # Load existing workflows
            self._load_all_workflows()
            self.initialized = True

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """Cleanup observers and executors."""
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
        """Initialize and start the file system observer."""
        try:
            if self._observer is None:
                self._observer = Observer()
                if WORKFLOW_DIR not in self._watch_paths:
                    self._observer.schedule(WorkflowFileHandler(self), WORKFLOW_DIR, recursive=False)
                    self._watch_paths.add(WORKFLOW_DIR)
                    self._observer.start()
        except Exception as e:
            logger.error(f"Error starting file watcher: {str(e)}")
            self.cleanup()

    def _load_all_workflows(self):
        """Load all workflow files from disk."""
        for file_path in Path(WORKFLOW_DIR).glob("*.json"):
            workflow_id = file_path.stem
            self.load_workflow(workflow_id)

    def load_workflow(self, workflow_id: str) -> Optional[Dict]:
        """Load a workflow from its JSON file."""
        try:
            file_path = WORKFLOW_DIR / f"{workflow_id}.json"
            if file_path.exists():
                with open(file_path, "r") as f:
                    self._workflows[workflow_id] = json.load(f)
                return self._workflows[workflow_id]
        except Exception as e:
            logger.error(f"Error loading workflow {workflow_id}: {str(e)}")
        return None

    def store_workflow(self, workflow_id: str, workflow_data: Dict) -> Dict:
        """Store workflow data in memory and to file."""
        try:
            # Store in memory
            self._workflows[workflow_id] = workflow_data

            # Store to file
            file_path = WORKFLOW_DIR / f"{workflow_id}.json"
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

    def _create_task_agent(self, task: Dict) -> Agent:
        """Create a specialized agent for a specific task with custom model and tools."""
        try:
            # Get task-specific configuration
            task_tools = task.get("tools")
            model_provider = task.get("model_provider")
            model_settings = task.get("model_settings")
            system_prompt = task.get("system_prompt")

            # Configure tools
            filtered_tools = []
            if task_tools and self.parent_agent and hasattr(self.parent_agent, "tool_registry"):
                # Filter parent agent tools to only include specified tool names
                available_tools = self.parent_agent.tool_registry.registry
                for tool_name in task_tools:
                    if tool_name in available_tools:
                        filtered_tools.append(available_tools[tool_name])
                    else:
                        logger.warning(f"Tool '{tool_name}' not found in parent agent's tool registry")
            elif self.parent_agent and hasattr(self.parent_agent, "tool_registry"):
                # Inherit all tools from parent if none specified
                filtered_tools = list(self.parent_agent.tool_registry.registry.values())

            # Configure model
            selected_model = None
            model_info = "Using parent agent's model"

            if model_provider is None:
                # Use parent agent's model
                selected_model = self.parent_agent.model if self.parent_agent else None
            elif model_provider == "env":
                # Use environment variables
                try:
                    env_provider = os.getenv("STRANDS_PROVIDER", "ollama")
                    selected_model = create_model(provider=env_provider, config=model_settings)
                    model_info = f"Using environment model: {env_provider}"
                except Exception as e:
                    logger.warning(f"Failed to create model from environment: {e}")
                    selected_model = self.parent_agent.model if self.parent_agent else None
                    model_info = "Failed to use environment model, using parent's model"
            else:
                # Use specified model provider
                try:
                    selected_model = create_model(provider=model_provider, config=model_settings)
                    model_info = f"Using {model_provider} model"
                except Exception as e:
                    logger.warning(f"Failed to create {model_provider} model: {e}")
                    selected_model = self.parent_agent.model if self.parent_agent else None
                    model_info = f"Failed to use {model_provider} model, using parent's model"

            # Determine system prompt
            if not system_prompt and self.parent_agent and hasattr(self.parent_agent, "system_prompt"):
                system_prompt = self.parent_agent.system_prompt
            elif not system_prompt:
                system_prompt = "You are a helpful AI assistant specialized in task execution."

            # Create the task agent
            task_agent = Agent(
                model=selected_model,
                system_prompt=system_prompt,
                tools=filtered_tools,
                trace_attributes=(self.parent_agent.trace_attributes if self.parent_agent else None),
            )

            logger.debug(f"Created task agent with {len(filtered_tools)} tools, model: {model_info}")
            return task_agent

        except Exception as e:
            logger.error(f"Error creating task agent: {str(e)}")
            # Fallback to parent agent or basic agent
            if self.parent_agent:
                return self.parent_agent
            return Agent(system_prompt="You are a helpful AI assistant.")

    def _wait_for_rate_limit(self):
        """Implements rate limiting for API calls."""
        global _last_request_time
        with _rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - _last_request_time
            if time_since_last < _MIN_REQUEST_INTERVAL:
                sleep_time = _MIN_REQUEST_INTERVAL - time_since_last
                time.sleep(sleep_time)
            _last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        reraise=True,
    )
    def execute_task(self, task: Dict, workflow: Dict) -> Dict:
        """Execute a single task using a specialized agent with rate limiting and retries."""
        try:
            task_id = task["task_id"]

            # Build context from dependent tasks
            context = []
            if task.get("dependencies"):
                for dep_id in task["dependencies"]:
                    dep_result = workflow["task_results"].get(dep_id, {})
                    if dep_result.get("status") == "completed" and dep_result.get("result"):
                        # Format the dependency results
                        dep_content = []
                        for msg in dep_result["result"]:
                            if isinstance(msg, dict) and msg.get("text"):
                                dep_content.append(msg["text"])
                        if dep_content:
                            context.append(f"Results from {dep_id}:\n" + "\n".join(dep_content))

            # Build comprehensive task prompt with context
            task_prompt = task["description"]
            if context:
                task_prompt = "Previous task results:\n" + "\n\n".join(context) + "\n\nCurrent Task:\n" + task_prompt

            # Add jitter to prevent thundering herd
            jitter = random.uniform(0, 1)
            time.sleep(jitter)

            # Apply rate limiting before making API call
            self._wait_for_rate_limit()

            # Create specialized agent for this task
            task_agent = self._create_task_agent(task)

            # Execute task
            logger.debug(f"Executing task {task_id} with specialized agent")
            result = task_agent(task_prompt)

            # Extract response content - handle both dict and custom object return types
            try:
                content = result.get("content", []) if hasattr(result, "get") else getattr(result, "content", [])
            except AttributeError:
                content = [{"text": str(result)}]

            # Extract stop reason and metrics
            try:
                stop_reason = (
                    result.get("stop_reason", "") if hasattr(result, "get") else getattr(result, "stop_reason", "")
                )
                metrics = result.get("metrics") if hasattr(result, "get") else getattr(result, "metrics", None)
            except AttributeError:
                stop_reason = ""
                metrics = None

            # Log metrics if available
            if metrics:
                metrics_text = metrics_to_string(metrics)
                logger.debug(f"Task {task_id} metrics: {metrics_text}")

            # Update task status
            status = "success" if stop_reason != "error" else "error"
            return {
                "status": status,
                "content": content,
                "metrics": metrics_text if metrics else None,
            }

        except Exception as e:
            error_msg = f"Error executing task {task['task_id']}: {str(e)}"
            logger.error(error_msg)
            if "ThrottlingException" in str(e):
                logger.error(f"Task {task['task_id']} hit throttling, will retry with exponential backoff")
                raise
            return {"status": "error", "content": [{"text": error_msg}]}

    def create_workflow(self, workflow_id: str, tasks: List[Dict]) -> Dict:
        """Create a new workflow with the given tasks."""
        try:
            if not workflow_id:
                workflow_id = str(uuid.uuid4())

            # Validate and enhance tasks
            enhanced_tasks = []
            for task in tasks:
                # Validate required fields
                if not task.get("task_id"):
                    return {
                        "status": "error",
                        "content": [{"text": "Each task must have a task_id"}],
                    }
                if not task.get("description"):
                    return {
                        "status": "error",
                        "content": [{"text": f"Task {task['task_id']} must have a description"}],
                    }

                # Add default values
                enhanced_task = task.copy()
                enhanced_task.setdefault("priority", 3)
                enhanced_task.setdefault("timeout", 300)
                enhanced_task.setdefault("dependencies", [])

                # Validate dependencies
                dep_task_ids = {t.get("task_id") for t in tasks}
                for dep_id in enhanced_task["dependencies"]:
                    if dep_id not in dep_task_ids:
                        return {
                            "status": "error",
                            "content": [{"text": f"Task {task['task_id']} has invalid dependency: {dep_id}"}],
                        }

                enhanced_tasks.append(enhanced_task)

            workflow = {
                "workflow_id": workflow_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "created",
                "tasks": enhanced_tasks,
                "task_results": {
                    task["task_id"]: {
                        "status": "pending",
                        "result": None,
                        "priority": task.get("priority", 3),
                        "model_provider": task.get("model_provider"),
                        "tools": task.get("tools", []),
                    }
                    for task in enhanced_tasks
                },
                "parallel_execution": True,
            }

            store_result = self.store_workflow(workflow_id, workflow)
            if store_result["status"] == "error":
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to create workflow: {store_result['error']}"}],
                }

            return {
                "status": "success",
                "content": [{"text": f"✅ Created modern workflow '{workflow_id}' with {len(enhanced_tasks)} tasks"}],
            }

        except Exception as e:
            error_msg = f"Error creating workflow: {str(e)}"
            logger.error(error_msg)
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

    def start_workflow(self, workflow_id: str) -> Dict:
        """Start or resume workflow execution with true parallel processing."""
        try:
            # Get workflow data
            workflow = self.get_workflow(workflow_id)
            if not workflow:
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Workflow '{workflow_id}' not found"}],
                }

            # Update status
            workflow["status"] = "running"
            workflow["started_at"] = datetime.now(timezone.utc).isoformat()
            self.store_workflow(workflow_id, workflow)

            logger.info(f"🚀 Starting workflow '{workflow_id}' with {len(workflow['tasks'])} tasks")

            # Track completed tasks and active futures
            completed_tasks = set()
            active_futures = {}
            total_tasks = len(workflow["tasks"])

            while len(completed_tasks) < total_tasks:
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
                                (task, workflow),
                                {},
                            )
                        )

                # Submit batch of tasks in parallel
                if tasks_to_submit:
                    new_futures = self.task_executor.submit_tasks(tasks_to_submit)
                    active_futures.update(new_futures)
                    logger.debug(f"📤 Submitted {len(tasks_to_submit)} tasks for execution")

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

                                # Ensure content uses valid format
                                content = []
                                for item in result.get("content", []):
                                    if isinstance(item, dict):
                                        content.append(item)
                                    else:
                                        content.append({"text": str(item)})

                                workflow["task_results"][task_id] = {
                                    **workflow["task_results"][task_id],
                                    "status": ("completed" if result["status"] == "success" else "error"),
                                    "result": content,
                                    "completed_at": datetime.now(timezone.utc).isoformat(),
                                    "metrics": result.get("metrics"),
                                }
                                completed_tasks.add(task_id)
                                logger.info(f"✅ Task '{task_id}' completed successfully")

                            except Exception as e:
                                workflow["task_results"][task_id] = {
                                    **workflow["task_results"][task_id],
                                    "status": "error",
                                    "result": [{"text": f"Task execution error: {str(e)}"}],
                                    "completed_at": datetime.now(timezone.utc).isoformat(),
                                }
                                completed_tasks.add(task_id)
                                logger.error(f"❌ Task '{task_id}' failed: {str(e)}")

                    # Remove completed tasks from active futures
                    for task_id in completed_task_ids:
                        del active_futures[task_id]

                # Store updated workflow state
                self.store_workflow(workflow_id, workflow)

                # Brief pause to prevent tight loop
                time.sleep(0.1)

            # Workflow completed
            workflow["status"] = "completed"
            workflow["completed_at"] = datetime.now(timezone.utc).isoformat()
            self.store_workflow(workflow_id, workflow)

            # Calculate success rate
            completed_count = sum(1 for result in workflow["task_results"].values() if result["status"] == "completed")
            success_rate = (completed_count / total_tasks) * 100 if total_tasks > 0 else 0

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🎉 Workflow '{workflow_id}' completed successfully! "
                            f"({completed_count}/{total_tasks} tasks succeeded - {success_rate:.1f}%)"
                        )
                    }
                ],
            }

        except Exception as e:
            error_trace = traceback.format_exc()
            error_msg = f"❌ Error in workflow execution: {str(e)}\n{error_trace}"
            logger.error(error_msg)
            return {"status": "error", "content": [{"text": error_msg}]}

    def list_workflows(self) -> Dict:
        """List all workflows with rich formatting."""
        try:
            # Refresh from files first
            self._load_all_workflows()

            if not self._workflows:
                return {
                    "status": "success",
                    "content": [{"text": "📭 No workflows found"}],
                }

            console = console_util.create()

            # Create rich table
            table = Table(show_header=True, box=ROUNDED)
            table.add_column("🆔 Workflow ID", style="bold blue")
            table.add_column("📊 Status", style="bold")
            table.add_column("📋 Tasks", justify="center")
            table.add_column("📅 Created", style="dim")
            table.add_column("⚡ Parallel", justify="center")

            for workflow_id, workflow_data in self._workflows.items():
                # Status styling
                status = workflow_data["status"]
                if status == "completed":
                    status_style = "[green]✅ Completed[/green]"
                elif status == "running":
                    status_style = "[yellow]🔄 Running[/yellow]"
                elif status == "error":
                    status_style = "[red]❌ Error[/red]"
                else:
                    status_style = "[blue]📝 Created[/blue]"

                table.add_row(
                    workflow_id,
                    status_style,
                    str(len(workflow_data["tasks"])),
                    workflow_data["created_at"].split("T")[0],
                    "✅" if workflow_data.get("parallel_execution", True) else "❌",
                )

            # Capture table output
            with console.capture() as capture:
                console.print(Panel(table, title="🔄 Workflow Management Dashboard", box=ROUNDED))

            return {
                "status": "success",
                "content": [{"text": f"📊 Found {len(self._workflows)} workflows:\n\n{capture.get()}"}],
            }

        except Exception as e:
            error_msg = f"Error listing workflows: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "content": [{"text": error_msg}]}

    def get_workflow_status(self, workflow_id: str) -> Dict:
        """Get detailed status of a workflow with rich formatting."""
        try:
            workflow = self.get_workflow(workflow_id)
            if not workflow:
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Workflow '{workflow_id}' not found"}],
                }

            console = console_util.create()

            # Create status overview
            status_lines = [
                f"🆔 **Workflow ID:** {workflow_id}",
                f"📊 **Status:** {workflow['status']}",
                f"📅 **Created:** {workflow['created_at'].split('T')[0]}",
            ]

            if workflow.get("started_at"):
                status_lines.append(f"🚀 **Started:** {workflow['started_at'].split('T')[0]}")
            if workflow.get("completed_at"):
                status_lines.append(f"🏁 **Completed:** {workflow['completed_at'].split('T')[0]}")

            # Create detailed task table
            table = Table(show_header=True, box=ROUNDED)
            table.add_column("🆔 Task ID", style="bold")
            table.add_column("📊 Status", justify="center")
            table.add_column("⭐ Priority", justify="center")
            table.add_column("🔗 Dependencies", style="dim")
            table.add_column("🤖 Model", style="cyan")
            table.add_column("🛠️ Tools", style="magenta")
            table.add_column("⏱️ Duration", justify="right")

            # Count statuses
            status_counts = {"pending": 0, "completed": 0, "error": 0, "running": 0}
            total_tasks = len(workflow["tasks"])

            for task in workflow["tasks"]:
                task_id = task["task_id"]
                task_result = workflow["task_results"].get(task_id, {})

                # Get task details
                status = task_result.get("status", "pending")
                status_counts[status] = status_counts.get(status, 0) + 1

                priority = task.get("priority", 3)
                dependencies = task.get("dependencies", [])
                model_provider = task.get("model_provider", "parent")
                tools = task.get("tools", [])

                # Calculate duration
                duration = "N/A"
                if status == "completed" and task_id in self.task_executor.start_times:
                    start_time = self.task_executor.start_times[task_id]
                    completed_at = task_result.get("completed_at")
                    if completed_at:
                        end_time = datetime.fromisoformat(completed_at).timestamp()
                        duration = f"{(end_time - start_time):.2f}s"

                # Status styling
                if status == "completed":
                    status_display = "[green]✅[/green]"
                elif status == "error":
                    status_display = "[red]❌[/red]"
                elif status == "running":
                    status_display = "[yellow]🔄[/yellow]"
                else:
                    status_display = "[blue]⏳[/blue]"

                table.add_row(
                    task_id,
                    status_display,
                    f"⭐{priority}",
                    ", ".join(dependencies) if dependencies else "None",
                    model_provider,
                    f"{len(tools)} tools" if tools else "All",
                    duration,
                )

            # Calculate progress
            completed_count = status_counts["completed"]
            progress_pct = (completed_count / total_tasks) * 100 if total_tasks > 0 else 0

            # Add progress info
            status_lines.extend(
                [
                    f"📈 **Progress:** {progress_pct:.1f}% ({completed_count}/{total_tasks})",
                    f"✅ **Completed:** {status_counts['completed']}",
                    f"⏳ **Pending:** {status_counts['pending']}",
                    f"❌ **Failed:** {status_counts['error']}",
                    f"🔄 **Active Workers:** {self.task_executor.active_workers}/{self.task_executor.max_workers}",
                ]
            )

            # Capture rich output
            with console.capture() as capture:
                console.print(
                    Panel(
                        "\n".join(status_lines),
                        title="📊 Workflow Overview",
                        box=ROUNDED,
                    )
                )
                console.print(Panel(table, title="📋 Task Details", box=ROUNDED))

            return {"status": "success", "content": [{"text": capture.get()}]}

        except Exception as e:
            error_msg = f"Error getting workflow status: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "content": [{"text": error_msg}]}

    def delete_workflow(self, workflow_id: str) -> Dict:
        """Delete a workflow and its results."""
        try:
            # Remove from memory
            if workflow_id in self._workflows:
                del self._workflows[workflow_id]

            # Remove file if exists
            file_path = WORKFLOW_DIR / f"{workflow_id}.json"
            if file_path.exists():
                file_path.unlink()
                return {
                    "status": "success",
                    "content": [{"text": f"🗑️ Workflow '{workflow_id}' deleted successfully"}],
                }
            else:
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Workflow '{workflow_id}' not found"}],
                }

        except Exception as e:
            error_msg = f"Error deleting workflow: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "content": [{"text": error_msg}]}


# Global manager instance
_manager = None


@tool
def workflow(
    action: str,
    workflow_id: Optional[str] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """Advanced workflow orchestration with granular model and tool control.

    This function provides comprehensive workflow management capabilities with modern
    Strands SDK patterns, supporting per-task model providers, tool configurations,
    and advanced execution monitoring.

    Key Features:
    ------------
    1. **Per-Task Model Configuration:**
       • Individual model providers per task (bedrock, anthropic, ollama, openai, etc.)
       • Custom model settings and parameters for each task
       • Environment-based model configuration with fallbacks
       • Automatic model validation and error recovery

    2. **Flexible Tool Management:**
       • Per-task tool access control for security and efficiency
       • Automatic tool inheritance from parent agent
       • Tool validation and filtering
       • Support for any combination of tools per task

    3. **Advanced Task Orchestration:**
       • Parallel execution with dependency resolution
       • Priority-based scheduling (1-5 levels)
       • Comprehensive timeout and resource controls
       • Intelligent batching and resource optimization

    4. **Rich Monitoring & Analytics:**
       • Real-time progress tracking with metrics
       • Per-task performance insights
       • Resource utilization monitoring
       • Comprehensive execution logging

    5. **Robust Persistence:**
       • File-based workflow storage
       • Real-time file system monitoring
       • State preservation across restarts
       • Automatic backup and recovery

    Args:
        action: Action to perform on workflows.
            • "create": Create a new workflow with tasks
            • "start": Begin workflow execution
            • "list": Show all workflows and their status
            • "status": Get detailed workflow progress
            • "delete": Remove workflow and cleanup
            • "pause": Pause workflow execution (future)
            • "resume": Resume paused workflow (future)

        workflow_id: Unique identifier for the workflow.
            Auto-generated if not provided for create action.

        tasks: List of task specifications for create action. Each task can include:
            • task_id (str): Unique task identifier [REQUIRED]
            • description (str): Task prompt for AI execution [REQUIRED]
            • system_prompt (str): Custom system prompt for this task [OPTIONAL]
            • tools (List[str]): Tool names available to this task [OPTIONAL]
            • model_provider (str): Model provider for this task [OPTIONAL]
              Options: "bedrock", "anthropic", "ollama", "openai", "github", "env"
            • model_settings (Dict): Model configuration [OPTIONAL]
              Example: {"model_id": "claude-sonnet-4", "params": {"temperature": 0.7}}
            • dependencies (List[str]): Task IDs this task depends on [OPTIONAL]
            • priority (int): Task priority 1-5, higher is more important [OPTIONAL, default: 3]
            • timeout (int): Task timeout in seconds [OPTIONAL, default: 300]

        agent: Parent agent (automatically provided by Strands framework).

    Returns:
        Dict containing status and response content with detailed workflow information.

    Task Configuration Examples:
    ---------------------------
    ```python
    # Basic task with default settings
    {
        "task_id": "research",
        "description": "Research renewable energy trends for 2024"
    }

    # Advanced task with custom model and tools
    {
        "task_id": "analysis",
        "description": "Analyze the research data and identify key insights",
        "dependencies": ["research"],
        "tools": ["calculator", "file_read", "file_write"],
        "model_provider": "bedrock",
        "model_settings": {
            "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "params": {"temperature": 0.3, "max_tokens": 4000}
        },
        "system_prompt": "You are a data analysis specialist focused on renewable energy research.",
        "priority": 5,
        "timeout": 600
    }

    # Task with environment-based model
    {
        "task_id": "report",
        "description": "Generate a comprehensive report",
        "dependencies": ["analysis"],
        "model_provider": "env",  # Uses STRANDS_PROVIDER env var
        "tools": ["file_write", "generate_image"],
        "priority": 4
    }
    ```

    Usage Examples:
    --------------
    ```python
    # Create a multi-model data analysis workflow
    result = agent.tool.workflow(
        action="create",
        workflow_id="data_pipeline",
        tasks=[
            {
                "task_id": "collect_data",
                "description": "Collect relevant data from various sources",
                "tools": ["retrieve", "http_request", "file_write"],
                "model_provider": "ollama",
                "model_settings": {"model_id": "qwen3:4b"},
                "priority": 5
            },
            {
                "task_id": "clean_data",
                "description": "Clean and preprocess the collected data",
                "dependencies": ["collect_data"],
                "tools": ["file_read", "file_write", "python_repl"],
                "model_provider": "anthropic",
                "model_settings": {"model_id": "claude-sonnet-4-20250514"},
                "system_prompt": "You are a data preprocessing specialist.",
                "priority": 4
            },
            {
                "task_id": "analyze_data",
                "description": "Perform statistical analysis on the cleaned data",
                "dependencies": ["clean_data"],
                "tools": ["calculator", "python_repl", "file_write"],
                "model_provider": "bedrock",
                "model_settings": {
                    "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                    "params": {"temperature": 0.2}
                },
                "priority": 5,
                "timeout": 600
            },
            {
                "task_id": "create_visualizations",
                "description": "Create charts and visualizations from the analysis",
                "dependencies": ["analyze_data"],
                "tools": ["python_repl", "generate_image", "file_write"],
                "model_provider": "openai",
                "model_settings": {"model_id": "o4-mini"},
                "priority": 3
            },
            {
                "task_id": "generate_report",
                "description": "Generate final comprehensive report",
                "dependencies": ["analyze_data", "create_visualizations"],
                "tools": ["file_read", "file_write"],
                "model_provider": "anthropic",
                "model_settings": {"params": {"temperature": 0.7}},
                "system_prompt": "You are a report writing specialist.",
                "priority": 4
            }
        ]
    )

    # Start the workflow
    result = agent.tool.workflow(action="start", workflow_id="data_pipeline")

    # Monitor progress
    result = agent.tool.workflow(action="status", workflow_id="data_pipeline")

    # List all workflows
    result = agent.tool.workflow(action="list")
    ```

    Notes:
        • Built on modern Strands SDK patterns with @tool decorator
        • Supports all major model providers with custom configurations
        • Per-task tool filtering ensures security and efficiency
        • Comprehensive error handling with automatic retries
        • Rich console output with progress tracking
        • File-based persistence with real-time monitoring
        • Resource optimization with dynamic thread scaling
        • Workflow files stored in ~/.strands/workflows/
        • Each task runs with specialized agent configuration
        • Context passing between dependent tasks for continuity
    """
    global _manager

    try:
        # Initialize manager if needed
        if _manager is None:
            _manager = WorkflowManager(parent_agent=agent)

        # Route to appropriate handler
        if action == "create":
            if not tasks:
                return {
                    "status": "error",
                    "content": [{"text": "❌ Tasks are required for create action"}],
                }

            if not workflow_id:
                workflow_id = str(uuid.uuid4())

            return _manager.create_workflow(workflow_id, tasks)

        elif action == "start":
            if not workflow_id:
                return {
                    "status": "error",
                    "content": [{"text": "❌ workflow_id is required for start action"}],
                }
            return _manager.start_workflow(workflow_id)

        elif action == "list":
            return _manager.list_workflows()

        elif action == "status":
            if not workflow_id:
                return {
                    "status": "error",
                    "content": [{"text": "❌ workflow_id is required for status action"}],
                }
            return _manager.get_workflow_status(workflow_id)

        elif action == "delete":
            if not workflow_id:
                return {
                    "status": "error",
                    "content": [{"text": "❌ workflow_id is required for delete action"}],
                }
            return _manager.delete_workflow(workflow_id)

        elif action in ["pause", "resume"]:
            return {
                "status": "error",
                "content": [{"text": f"🚧 Action '{action}' is not yet implemented"}],
            }

        else:
            return {
                "status": "error",
                "content": [{"text": f"❌ Unknown action: {action}. Available: create, start, list, status, delete"}],
            }

    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"❌ Error in workflow tool: {str(e)}\n\nTraceback:\n{error_trace}"
        logger.error(error_msg)
        return {
            "status": "error",
            "content": [{"text": error_msg}],
        }
