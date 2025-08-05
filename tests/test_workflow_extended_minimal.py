"""
Minimal workflow tests to avoid hanging issues.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.strands_tools.workflow import workflow
from tests.workflow_test_isolation import isolated_workflow_environment, mock_workflow_threading_components


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.model = MagicMock()
    agent.system_prompt = "Test system prompt"
    agent.trace_attributes = {"test": "value"}
    
    # Mock tool registry
    tool_registry = MagicMock()
    tool_registry.registry = {
        "calculator": MagicMock(),
        "file_read": MagicMock(),
    }
    agent.tool_registry = tool_registry
    
    return agent


@pytest.fixture
def sample_tasks():
    """Create sample tasks for testing."""
    return [
        {
            "task_id": "task1",
            "description": "First task description",
            "priority": 5,
        }
    ]


class TestWorkflowToolMinimal:
    """Minimal workflow tool tests."""
    
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
        
    def test_workflow_status_missing_id(self, mock_agent):
        """Test workflow status without ID."""
        result = workflow(action="status", agent=mock_agent)
        
        assert result["status"] == "error"
        assert "workflow_id is required" in result["content"][0]["text"]
        
    def test_workflow_delete_missing_id(self, mock_agent):
        """Test workflow delete without ID."""
        result = workflow(action="delete", agent=mock_agent)
        
        assert result["status"] == "error"
        assert "workflow_id is required" in result["content"][0]["text"]
        
    def test_workflow_unknown_action(self, mock_agent):
        """Test workflow with unknown action."""
        result = workflow(action="unknown", agent=mock_agent)
        
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]
        
    def test_workflow_pause_not_implemented(self, mock_agent):
        """Test pause action (not implemented)."""
        result = workflow(action="pause", workflow_id="test", agent=mock_agent)
        assert result["status"] == "error"
        assert "not yet implemented" in result["content"][0]["text"]
        
    def test_workflow_resume_not_implemented(self, mock_agent):
        """Test resume action (not implemented)."""
        result = workflow(action="resume", workflow_id="test", agent=mock_agent)
        assert result["status"] == "error"
        assert "not yet implemented" in result["content"][0]["text"]


class TestWorkflowToolMocked:
    """Test workflow tool with mocked manager."""
    
    def test_workflow_create_success(self, mock_agent, sample_tasks):
        """Test successful workflow creation via tool."""
        # Reset global state
        import src.strands_tools.workflow
        src.strands_tools.workflow._manager = None
        
        with patch('src.strands_tools.workflow.WorkflowManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.create_workflow.return_value = {
                "status": "success",
                "content": [{"text": "Workflow created"}]
            }
            
            result = workflow(
                action="create",
                workflow_id="test_workflow",
                tasks=sample_tasks,
                agent=mock_agent
            )
            
            assert result["status"] == "success"
            mock_manager.create_workflow.assert_called_once_with("test_workflow", sample_tasks)
            
    def test_workflow_list(self, mock_agent):
        """Test workflow list action."""
        # Reset global state
        import src.strands_tools.workflow
        src.strands_tools.workflow._manager = None
        
        with patch('src.strands_tools.workflow.WorkflowManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.list_workflows.return_value = {
                "status": "success",
                "content": [{"text": "Workflows listed"}]
            }
            
            result = workflow(action="list", agent=mock_agent)
            
            assert result["status"] == "success"
            mock_manager.list_workflows.assert_called_once()
            
    def test_workflow_status_success(self, mock_agent):
        """Test workflow status action."""
        # Reset global state
        import src.strands_tools.workflow
        src.strands_tools.workflow._manager = None
        
        with patch('src.strands_tools.workflow.WorkflowManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.get_workflow_status.return_value = {
                "status": "success",
                "content": [{"text": "Status retrieved"}]
            }
            
            result = workflow(action="status", workflow_id="test_workflow", agent=mock_agent)
            
            assert result["status"] == "success"
            mock_manager.get_workflow_status.assert_called_once_with("test_workflow")
            
    def test_workflow_delete_success(self, mock_agent):
        """Test workflow delete action."""
        # Reset global state
        import src.strands_tools.workflow
        src.strands_tools.workflow._manager = None
        
        with patch('src.strands_tools.workflow.WorkflowManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.delete_workflow.return_value = {
                "status": "success",
                "content": [{"text": "Workflow deleted"}]
            }
            
            result = workflow(action="delete", workflow_id="test_workflow", agent=mock_agent)
            
            assert result["status"] == "success"
            mock_manager.delete_workflow.assert_called_once_with("test_workflow")