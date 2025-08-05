"""
Simplified workflow tests that avoid hanging issues.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.strands_tools.workflow import workflow
from tests.workflow_test_isolation import isolated_workflow_environment, mock_workflow_threading_components


# Workflow state reset is now handled by the global fixture in conftest.py


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.model = MagicMock()
    agent.system_prompt = "Test system prompt"
    agent.trace_attributes = {"test": "value"}
    
    tool_registry = MagicMock()
    tool_registry.registry = {"calculator": MagicMock(), "file_read": MagicMock()}
    agent.tool_registry = tool_registry
    
    return agent


@pytest.fixture
def sample_tasks():
    """Create sample tasks for testing."""
    return [
        {"task_id": "task1", "description": "First task", "priority": 5},
        {"task_id": "task2", "description": "Second task", "dependencies": ["task1"], "priority": 3}
    ]


class TestWorkflowBasic:
    """Test basic workflow functionality."""
    
    def test_workflow_create_missing_tasks(self, mock_agent):
        """Test workflow creation without tasks."""
        result = workflow(action="create", workflow_id="test", agent=mock_agent)
        assert result["status"] == "error"
        assert "Tasks are required" in result["content"][0]["text"]
        
    def test_workflow_start_missing_id(self, mock_agent):
        """Test workflow start without ID."""
        result = workflow(action="start", agent=mock_agent)
        assert result["status"] == "error"
        assert "workflow_id is required" in result["content"][0]["text"]
        
    def test_workflow_unknown_action(self, mock_agent):
        """Test workflow with unknown action."""
        result = workflow(action="unknown", agent=mock_agent)
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]


class TestWorkflowMocked:
    """Test workflow with mocked manager."""
    
    def test_workflow_create_success(self, mock_agent, sample_tasks):
        """Test successful workflow creation."""
        with patch('src.strands_tools.workflow.WorkflowManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.create_workflow.return_value = {
                "status": "success",
                "content": [{"text": "Workflow created"}]
            }
            
            result = workflow(action="create", workflow_id="test", tasks=sample_tasks, agent=mock_agent)
            assert result["status"] == "success"
            
    def test_workflow_list(self, mock_agent):
        """Test workflow list action."""
        with patch('src.strands_tools.workflow.WorkflowManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.list_workflows.return_value = {
                "status": "success",
                "content": [{"text": "Workflows listed"}]
            }
            
            result = workflow(action="list", agent=mock_agent)
            assert result["status"] == "success"
            
    def test_workflow_general_exception(self, mock_agent):
        """Test workflow with general exception."""
        with patch('src.strands_tools.workflow.WorkflowManager', side_effect=Exception("General error")):
            result = workflow(action="list", agent=mock_agent)
            assert result["status"] == "error"
            assert "Error in workflow tool" in result["content"][0]["text"]


def test_workflow_imports():
    """Test that workflow module imports correctly."""
    from src.strands_tools.workflow import workflow, TaskExecutor, WorkflowManager
    assert workflow is not None
    assert TaskExecutor is not None
    assert WorkflowManager is not None