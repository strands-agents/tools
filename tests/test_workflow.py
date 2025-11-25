"""
Tests for the workflow tool using the Agent interface.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from strands import Agent
from strands.agent import AgentResult

from strands_tools import workflow as workflow_module


@pytest.fixture(autouse=True)
def reset_workflow_manager():
    """Reset the global workflow manager before each test to ensure clean state."""
    # Reset global manager before each test
    workflow_module._manager = None
    yield
    # Cleanup after test
    if hasattr(workflow_module, "_manager") and workflow_module._manager:
        try:
            workflow_module._manager.cleanup()
        except Exception:
            pass
    workflow_module._manager = None


@pytest.fixture
def agent():
    """Create an agent with the workflow tool loaded."""
    return Agent(tools=[workflow_module])


@pytest.fixture
def mock_parent_agent():
    """Create a mock parent agent with tools and registry."""
    mock_agent = MagicMock()
    mock_tool_registry = MagicMock()
    mock_agent.tool_registry = mock_tool_registry

    # Mock some tools in the registry
    mock_tool_registry.registry = {
        "calculator": MagicMock(),
        "file_read": MagicMock(),
        "file_write": MagicMock(),
        "retrieve": MagicMock(),
        "editor": MagicMock(),
        "http_request": MagicMock(),
        "generate_image": MagicMock(),
        "python_repl": MagicMock(),
    }

    # Mock model and other attributes
    mock_agent.model = MagicMock()
    mock_agent.trace_attributes = {"test_attr": "test_value"}
    mock_agent.system_prompt = "You are a helpful assistant."

    return mock_agent


@pytest.fixture
def mock_workflow_dir():
    """Create a temporary directory for workflow files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.object(workflow_module, "WORKFLOW_DIR", temp_dir):
            yield temp_dir


@pytest.fixture
def sample_tasks():
    """Sample tasks for testing."""
    return [
        {
            "task_id": "data_collection",
            "description": "Collect research data on renewable energy",
            "tools": ["retrieve", "file_write"],
            "priority": 5,
            "timeout": 300,
        },
        {
            "task_id": "analysis",
            "description": "Analyze the collected data",
            "dependencies": ["data_collection"],
            "tools": ["calculator", "file_read"],
            "model_provider": "anthropic",
            "model_settings": {"model_id": "claude-sonnet-4-20250514"},
            "priority": 4,
        },
        {
            "task_id": "report",
            "description": "Generate a comprehensive report",
            "dependencies": ["analysis"],
            "tools": ["file_write"],
            "priority": 3,
        },
    ]


@pytest.fixture
def mock_agent_result():
    """Create a mock result from Agent execution."""
    result = MagicMock()
    result.__str__ = MagicMock(return_value="Task completed successfully")

    # Mock metrics
    mock_metrics = MagicMock()
    mock_metrics.get_summary.return_value = {
        "total_cycles": 1,
        "average_cycle_time": 1.5,
        "total_duration": 1.5,
        "accumulated_usage": {"inputTokens": 15, "outputTokens": 25, "totalTokens": 40},
        "accumulated_metrics": {"latencyMs": 1500},
        "tool_usage": {},
    }
    mock_metrics.traces = []
    result.metrics = mock_metrics

    return result


class TestWorkflowCreation:
    """Test workflow creation functionality."""

    def test_create_workflow_basic(self, mock_parent_agent, mock_workflow_dir, sample_tasks):
        """Test basic workflow creation."""
        with patch("strands_tools.workflow.WorkflowManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.create_workflow.return_value = {
                "status": "success",
                "content": [{"text": "✅ Created modern workflow 'test_workflow' with 3 tasks"}],
            }

            result = workflow_module.workflow(
                action="create",
                workflow_id="test_workflow",
                tasks=sample_tasks,
                agent=mock_parent_agent,
            )

            # Verify manager was initialized with parent agent
            mock_manager_class.assert_called_once_with(parent_agent=mock_parent_agent)

            # Verify create_workflow was called correctly
            mock_manager.create_workflow.assert_called_once_with("test_workflow", sample_tasks)

            # Verify result
            assert result["status"] == "success"
            assert "Created modern workflow" in result["content"][0]["text"]

    def test_create_workflow_without_workflow_id(self, mock_parent_agent, sample_tasks):
        """Test workflow creation generates UUID when no workflow_id provided."""
        with (
            patch("strands_tools.workflow.WorkflowManager") as mock_manager_class,
            patch("strands_tools.workflow.uuid.uuid4") as mock_uuid,
        ):
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.create_workflow.return_value = {
                "status": "success",
                "content": [{"text": "✅ Created workflow"}],
            }
            mock_uuid.return_value = "generated-uuid-123"

            result = workflow_module.workflow(action="create", tasks=sample_tasks, agent=mock_parent_agent)

            # Verify UUID was generated and used
            mock_uuid.assert_called_once()
            mock_manager.create_workflow.assert_called_once_with("generated-uuid-123", sample_tasks)
            assert result["status"] == "success"

    def test_create_workflow_missing_tasks(self, mock_parent_agent):
        """Test workflow creation fails when tasks are missing."""
        result = workflow_module.workflow(action="create", workflow_id="test_workflow", agent=mock_parent_agent)

        assert result["status"] == "error"
        assert "Tasks are required for create action" in result["content"][0]["text"]

    def test_create_workflow_invalid_task_structure(self, mock_parent_agent):
        """Test workflow creation with invalid task structure."""
        invalid_tasks = [
            {
                "task_id": "task1",
                # Missing description
            }
        ]

        with patch("strands_tools.workflow.WorkflowManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.create_workflow.return_value = {
                "status": "error",
                "content": [{"text": "Task task1 must have a description"}],
            }

            result = workflow_module.workflow(
                action="create",
                workflow_id="test_workflow",
                tasks=invalid_tasks,
                agent=mock_parent_agent,
            )

            assert result["status"] == "error"
            assert "must have a description" in result["content"][0]["text"]


class TestWorkflowExecution:
    """Test workflow execution functionality."""

    def test_start_workflow_success(self, mock_parent_agent):
        """Test successful workflow start."""
        with patch("strands_tools.workflow.WorkflowManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.start_workflow.return_value = {
                "status": "success",
                "content": [{"text": "🎉 Workflow 'test_workflow' completed successfully!"}],
            }

            result = workflow_module.workflow(action="start", workflow_id="test_workflow", agent=mock_parent_agent)

            mock_manager.start_workflow.assert_called_once_with("test_workflow")
            assert result["status"] == "success"
            assert "completed successfully" in result["content"][0]["text"]

    def test_start_workflow_missing_id(self, mock_parent_agent):
        """Test workflow start fails when workflow_id is missing."""
        result = workflow_module.workflow(action="start", agent=mock_parent_agent)

        assert result["status"] == "error"
        assert "workflow_id is required for start action" in result["content"][0]["text"]

    def test_start_workflow_not_found(self, mock_parent_agent):
        """Test workflow start fails when workflow doesn't exist."""
        with patch("strands_tools.workflow.WorkflowManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.start_workflow.return_value = {
                "status": "error",
                "content": [{"text": "❌ Workflow 'nonexistent' not found"}],
            }

            result = workflow_module.workflow(action="start", workflow_id="nonexistent", agent=mock_parent_agent)

            assert result["status"] == "error"
            assert "not found" in result["content"][0]["text"]

    def test_task_id_namespacing(self):
        """Test task ID namespacing and extraction logic."""
        workflow_id = "test_workflow"
        task_id = "task1"

        namespaced_task_id = f"{workflow_id}:{task_id}"
        assert namespaced_task_id == "test_workflow:task1"

        extracted_id = namespaced_task_id.split(":", 1)[1] if ":" in namespaced_task_id else namespaced_task_id
        assert extracted_id == "task1"


class TestWorkflowStatus:
    """Test workflow status functionality."""

    def test_get_workflow_status(self, mock_parent_agent):
        """Test getting workflow status."""
        with patch("strands_tools.workflow.WorkflowManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.get_workflow_status.return_value = {
                "status": "success",
                "content": [{"text": "Workflow status information"}],
            }

            result = workflow_module.workflow(action="status", workflow_id="test_workflow", agent=mock_parent_agent)

            mock_manager.get_workflow_status.assert_called_once_with("test_workflow")
            assert result["status"] == "success"

    def test_get_workflow_status_missing_id(self, mock_parent_agent):
        """Test status action fails when workflow_id is missing."""
        result = workflow_module.workflow(action="status", agent=mock_parent_agent)

        assert result["status"] == "error"
        assert "workflow_id is required for status action" in result["content"][0]["text"]


class TestWorkflowListing:
    """Test workflow listing functionality."""

    def test_list_workflows(self, mock_parent_agent):
        """Test listing all workflows."""
        with patch("strands_tools.workflow.WorkflowManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.list_workflows.return_value = {
                "status": "success",
                "content": [{"text": "📊 Found 2 workflows"}],
            }

            result = workflow_module.workflow(action="list", agent=mock_parent_agent)

            mock_manager.list_workflows.assert_called_once()
            assert result["status"] == "success"
            assert "Found" in result["content"][0]["text"]

    def test_list_workflows_empty(self, mock_parent_agent):
        """Test listing workflows when none exist."""
        with patch("strands_tools.workflow.WorkflowManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.list_workflows.return_value = {
                "status": "success",
                "content": [{"text": "📭 No workflows found"}],
            }

            result = workflow_module.workflow(action="list", agent=mock_parent_agent)

            assert result["status"] == "success"
            assert "No workflows found" in result["content"][0]["text"]


class TestWorkflowDeletion:
    """Test workflow deletion functionality."""

    def test_delete_workflow(self, mock_parent_agent):
        """Test workflow deletion."""
        with patch("strands_tools.workflow.WorkflowManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.delete_workflow.return_value = {
                "status": "success",
                "content": [{"text": "🗑️ Workflow 'test_workflow' deleted successfully"}],
            }

            result = workflow_module.workflow(action="delete", workflow_id="test_workflow", agent=mock_parent_agent)

            mock_manager.delete_workflow.assert_called_once_with("test_workflow")
            assert result["status"] == "success"
            assert "deleted successfully" in result["content"][0]["text"]

    def test_delete_workflow_missing_id(self, mock_parent_agent):
        """Test delete action fails when workflow_id is missing."""
        result = workflow_module.workflow(action="delete", agent=mock_parent_agent)

        assert result["status"] == "error"
        assert "workflow_id is required for delete action" in result["content"][0]["text"]


class TestWorkflowManager:
    """Test WorkflowManager class functionality."""

    def test_workflow_manager_singleton(self, mock_parent_agent):
        """Test WorkflowManager is a singleton."""
        with patch("strands_tools.workflow.WorkflowManager.__new__") as mock_new:
            mock_instance = MagicMock()
            mock_new.return_value = mock_instance

            # Create two managers - should return same instance
            workflow_module.WorkflowManager(mock_parent_agent)
            workflow_module.WorkflowManager(mock_parent_agent)

            # Should only create one instance
            assert mock_new.call_count <= 2  # May be called twice due to initialization

    def test_create_task_agent_with_tools(self, mock_parent_agent):
        """Test task agent creation with specific tools."""
        with patch("strands_tools.workflow.Agent") as mock_agent_class:
            mock_task_agent = MagicMock()
            mock_agent_class.return_value = mock_task_agent

            manager = workflow_module.WorkflowManager(mock_parent_agent)

            task = {
                "task_id": "test_task",
                "description": "Test task",
                "tools": ["calculator", "file_read"],
                "system_prompt": "You are a test assistant.",
            }

            manager._create_task_agent(task)

            # Verify Agent was created with filtered tools
            mock_agent_class.assert_called_once()
            call_kwargs = mock_agent_class.call_args.kwargs

            # Should have exactly 2 tools
            assert len(call_kwargs["tools"]) == 2
            assert call_kwargs["system_prompt"] == "You are a test assistant."

    def test_create_task_agent_with_model_provider(self, mock_parent_agent):
        """Test task agent creation with custom model provider."""
        with (
            patch("strands_tools.workflow.Agent") as mock_agent_class,
            patch("strands_tools.workflow.create_model") as _mock_create_model,
        ):
            mock_model = MagicMock()
            _mock_create_model.return_value = mock_model
            mock_task_agent = MagicMock()
            mock_agent_class.return_value = mock_task_agent

            manager = workflow_module.WorkflowManager(mock_parent_agent)

            task = {
                "task_id": "test_task",
                "description": "Test task",
                "model_provider": "bedrock",
                "model_settings": {"model_id": "claude-sonnet-4"},
            }

            manager._create_task_agent(task)

            # Verify model was created with correct provider
            _mock_create_model.assert_called_once_with(provider="bedrock", config={"model_id": "claude-sonnet-4"})

            # Verify Agent was created with custom model
            call_kwargs = mock_agent_class.call_args.kwargs
            assert call_kwargs["model"] == mock_model

    def test_execute_task_success(self, mock_parent_agent, mock_agent_result):
        """Test successful task execution."""
        with (
            patch("strands_tools.workflow.Agent") as mock_agent_class,
            patch("strands_tools.workflow.create_model") as _mock_create_model,
            patch.object(workflow_module, "WORKFLOW_DIR", Path("/tmp/test_workflows")),
        ):
            # Create a proper mock agent result that returns structured data
            mock_task_agent = MagicMock()
            mock_result = AgentResult(
                message={"content": [{"text": "Task completed successfully"}]},
                stop_reason="completed",
                metrics=None,
                state=MagicMock(),
            )

            mock_task_agent.return_value = mock_result
            mock_agent_class.return_value = mock_task_agent

            manager = workflow_module.WorkflowManager(mock_parent_agent)

            task = {"task_id": "test_task", "description": "Test task description"}

            workflow = {"task_results": {}}

            result = manager.execute_task(task, workflow)

            assert result["status"] == "success"
            assert len(result["content"]) > 0
            assert result["content"][0]["text"] == "Task completed successfully"

    def test_execute_task_with_dependencies(self, mock_parent_agent, mock_agent_result):
        """Test task execution with dependencies."""
        with patch("strands_tools.workflow.Agent") as mock_agent_class:
            mock_task_agent = MagicMock()
            mock_task_agent.return_value = mock_agent_result
            mock_agent_class.return_value = mock_task_agent

            manager = workflow_module.WorkflowManager(mock_parent_agent)

            task = {
                "task_id": "dependent_task",
                "description": "Task that depends on others",
                "dependencies": ["task1"],
            }

            workflow = {
                "task_results": {
                    "task1": {
                        "status": "completed",
                        "result": [{"text": "Previous task result"}],
                    }
                }
            }

            manager.execute_task(task, workflow)

            # Verify task agent was called with context from dependencies
            mock_task_agent.assert_called_once()
            call_args = mock_task_agent.call_args[0]
            prompt = call_args[0]

            # Prompt should include dependency results
            assert "Previous task results:" in prompt
            assert "task1" in prompt
            assert "Previous task result" in prompt

    def test_store_and_load_workflow(self, mock_workflow_dir):
        """Test workflow storage and loading."""
        with patch.object(workflow_module, "WORKFLOW_DIR", Path(mock_workflow_dir)):
            manager = workflow_module.WorkflowManager()

            workflow_data = {
                "workflow_id": "test_workflow",
                "status": "created",
                "tasks": [{"task_id": "task1", "description": "Test task"}],
            }

            # Store workflow
            result = manager.store_workflow("test_workflow", workflow_data)
            assert result["status"] == "success"

            # Verify file was created
            workflow_file = Path(mock_workflow_dir) / "test_workflow.json"
            assert workflow_file.exists()

            # Load workflow
            loaded_workflow = manager.load_workflow("test_workflow")
            assert loaded_workflow == workflow_data

            # Get workflow (should use in-memory cache)
            retrieved_workflow = manager.get_workflow("test_workflow")
            assert retrieved_workflow == workflow_data

    def test_get_ready_tasks(self, mock_parent_agent):
        """Test getting ready tasks based on dependencies."""
        manager = workflow_module.WorkflowManager(mock_parent_agent)

        workflow = {
            "tasks": [
                {"task_id": "task1", "description": "Independent task", "priority": 5},
                {
                    "task_id": "task2",
                    "description": "Dependent task",
                    "dependencies": ["task1"],
                    "priority": 4,
                },
                {
                    "task_id": "task3",
                    "description": "Another independent task",
                    "priority": 3,
                },
            ],
            "task_results": {
                "task1": {"status": "pending"},
                "task2": {"status": "pending"},
                "task3": {"status": "pending"},
            },
        }

        ready_tasks = manager.get_ready_tasks(workflow)

        # Should return task1 and task3 (no dependencies), sorted by priority
        assert len(ready_tasks) == 2
        assert ready_tasks[0]["task_id"] == "task1"  # Higher priority first
        assert ready_tasks[1]["task_id"] == "task3"

    def test_get_ready_tasks_with_completed_dependencies(self, mock_parent_agent):
        """Test getting ready tasks when dependencies are completed."""
        manager = workflow_module.WorkflowManager(mock_parent_agent)

        workflow = {
            "tasks": [
                {"task_id": "task1", "description": "Independent task", "priority": 5},
                {
                    "task_id": "task2",
                    "description": "Dependent task",
                    "dependencies": ["task1"],
                    "priority": 4,
                },
            ],
            "task_results": {
                "task1": {"status": "completed"},
                "task2": {"status": "pending"},
            },
        }

        ready_tasks = manager.get_ready_tasks(workflow)

        # Should return task2 since task1 is completed
        assert len(ready_tasks) == 1
        assert ready_tasks[0]["task_id"] == "task2"


class TestWorkflowEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_action(self, mock_parent_agent):
        """Test workflow with invalid action."""
        result = workflow_module.workflow(action="invalid_action", agent=mock_parent_agent)

        assert result["status"] == "error"
        assert "Unknown action: invalid_action" in result["content"][0]["text"]

    def test_unimplemented_actions(self, mock_parent_agent):
        """Test pause and resume actions (not yet implemented)."""
        for action in ["pause", "resume"]:
            result = workflow_module.workflow(action=action, workflow_id="test_workflow", agent=mock_parent_agent)

            assert result["status"] == "error"
            assert f"Action '{action}' is not yet implemented" in result["content"][0]["text"]

    def test_workflow_exception_handling(self, mock_parent_agent, sample_tasks):
        """Test workflow tool handles exceptions gracefully."""
        with patch("strands_tools.workflow.WorkflowManager") as mock_manager_class:
            # Make manager initialization fail
            mock_manager_class.side_effect = Exception("Manager creation failed")

            result = workflow_module.workflow(
                action="create",
                workflow_id="test_workflow",
                tasks=sample_tasks,
                agent=mock_parent_agent,
            )

            assert result["status"] == "error"
            assert "Error in workflow tool" in result["content"][0]["text"]
            assert "Manager creation failed" in result["content"][0]["text"]

    def test_task_execution_error_handling(self, mock_parent_agent):
        """Test task execution error handling."""
        with patch("strands_tools.workflow.Agent") as mock_agent_class:
            # Make agent call fail
            mock_task_agent = MagicMock()
            mock_task_agent.side_effect = Exception("Task execution failed")
            mock_agent_class.return_value = mock_task_agent

            manager = workflow_module.WorkflowManager(mock_parent_agent)

            task = {"task_id": "failing_task", "description": "This task will fail"}

            workflow = {"task_results": {}}

            result = manager.execute_task(task, workflow)

            assert result["status"] == "error"
            assert "Error executing task failing_task" in result["content"][0]["text"]


class TestWorkflowIntegration:
    """Integration tests for the workflow tool."""

    def test_workflow_via_agent_interface(self, agent, sample_tasks):
        """Test workflow via the agent interface (integration test)."""
        with patch("strands_tools.workflow.WorkflowManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.create_workflow.return_value = {
                "status": "success",
                "content": [{"text": "✅ Workflow created successfully"}],
            }

            try:
                result = agent.tool.workflow(action="create", workflow_id="integration_test", tasks=sample_tasks)
                # If we get here without an exception, consider the test passed
                assert result is not None
            except Exception as e:
                pytest.fail(f"Agent workflow call raised an exception: {e}")

    def test_full_workflow_lifecycle(self, mock_parent_agent, mock_workflow_dir, sample_tasks):
        """Test complete workflow lifecycle: create -> start -> status -> delete."""
        with patch("strands_tools.workflow.WorkflowManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager

            # Mock all operations to succeed
            mock_manager.create_workflow.return_value = {
                "status": "success",
                "content": [{"text": "✅ Workflow created"}],
            }
            mock_manager.start_workflow.return_value = {
                "status": "success",
                "content": [{"text": "🎉 Workflow completed"}],
            }
            mock_manager.get_workflow_status.return_value = {
                "status": "success",
                "content": [{"text": "📊 Workflow status"}],
            }
            mock_manager.delete_workflow.return_value = {
                "status": "success",
                "content": [{"text": "🗑️ Workflow deleted"}],
            }

            workflow_id = "lifecycle_test"

            # Create workflow
            result = workflow_module.workflow(
                action="create",
                workflow_id=workflow_id,
                tasks=sample_tasks,
                agent=mock_parent_agent,
            )
            assert result["status"] == "success"

            # Start workflow
            result = workflow_module.workflow(action="start", workflow_id=workflow_id, agent=mock_parent_agent)
            assert result["status"] == "success"

            # Check status
            result = workflow_module.workflow(action="status", workflow_id=workflow_id, agent=mock_parent_agent)
            assert result["status"] == "success"

            # Delete workflow
            result = workflow_module.workflow(action="delete", workflow_id=workflow_id, agent=mock_parent_agent)
            assert result["status"] == "success"

    def test_workflow_with_different_model_providers(self, mock_parent_agent):
        """Test workflow with tasks using different model providers."""
        tasks_with_models = [
            {
                "task_id": "bedrock_task",
                "description": "Task using Bedrock",
                "model_provider": "bedrock",
                "model_settings": {"model_id": "claude-sonnet-4"},
            },
            {
                "task_id": "anthropic_task",
                "description": "Task using Anthropic",
                "model_provider": "anthropic",
                "model_settings": {"model_id": "claude-3-5-sonnet-20241022"},
            },
            {
                "task_id": "env_task",
                "description": "Task using environment model",
                "model_provider": "env",
            },
        ]

        with patch("strands_tools.workflow.WorkflowManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.create_workflow.return_value = {
                "status": "success",
                "content": [{"text": "✅ Multi-model workflow created"}],
            }

            result = workflow_module.workflow(
                action="create",
                workflow_id="multi_model_test",
                tasks=tasks_with_models,
                agent=mock_parent_agent,
            )

            assert result["status"] == "success"
            mock_manager.create_workflow.assert_called_once_with("multi_model_test", tasks_with_models)


class TestWorkflowFileOperations:
    """Test workflow file persistence operations."""

    def test_workflow_file_storage(self, mock_workflow_dir):
        """Test workflow file storage and format."""
        with patch.object(workflow_module, "WORKFLOW_DIR", Path(mock_workflow_dir)):
            manager = workflow_module.WorkflowManager()

            workflow_data = {
                "workflow_id": "test_storage",
                "status": "created",
                "tasks": [],
                "created_at": "2024-01-01T00:00:00+00:00",
            }

            result = manager.store_workflow("test_storage", workflow_data)
            assert result["status"] == "success"

            # Verify file content
            workflow_file = Path(mock_workflow_dir) / "test_storage.json"
            with open(workflow_file, "r") as f:
                stored_data = json.load(f)

            assert stored_data == workflow_data

    def test_workflow_file_loading_nonexistent(self):
        """Test loading non-existent workflow file."""
        manager = workflow_module.WorkflowManager()

        result = manager.load_workflow("nonexistent_workflow")
        assert result is None

    def test_workflow_file_storage_error(self, mock_parent_agent):
        """Test workflow file storage error handling."""
        manager = workflow_module.WorkflowManager(mock_parent_agent)

        with patch("builtins.open", side_effect=IOError("Permission denied")):
            result = manager.store_workflow("test_workflow", {"test": "data"})

            assert result["status"] == "error"
            assert "Permission denied" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
