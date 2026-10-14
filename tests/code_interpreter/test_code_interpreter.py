"""
Tests for the base CodeInterpreter class.
"""

import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

mock_bedrock_agentcore = MagicMock()
sys.modules["bedrock_agentcore"] = mock_bedrock_agentcore
sys.modules["bedrock_agentcore.tools"] = MagicMock()
sys.modules["bedrock_agentcore.tools.code_interpreter_client"] = MagicMock()

from strands_tools.code_interpreter.code_interpreter import CodeInterpreter  # noqa: E402
from strands_tools.code_interpreter.models import (  # noqa: E402
    CodeInterpreterInput,
    ExecuteCodeAction,
    ExecuteCommandAction,
    FileContent,
    InitSessionAction,
    LanguageType,
    ListFilesAction,
    ListLocalSessionsAction,
    ReadFilesAction,
    RemoveFilesAction,
    WriteFilesAction,
)


class MockCodeInterpreter(CodeInterpreter):
    """Mock implementation of CodeInterpreter for testing."""

    def __init__(self):
        super().__init__()
        self.platform_started = False
        self.platform_cleaned = False

    def start_platform(self) -> None:
        self.platform_started = True

    def cleanup_platform(self) -> None:
        self.platform_cleaned = True

    def init_session(self, action: InitSessionAction) -> Dict[str, Any]:
        return {
            "status": "success",
            "content": [
                {"json": {"sessionName": action.session_name or "test-session", "description": action.description}}
            ],
        }

    def list_local_sessions(self) -> Dict[str, Any]:
        return {"status": "success", "content": [{"json": {"sessions": [], "totalSessions": 0}}]}

    def execute_code(self, action: ExecuteCodeAction) -> Dict[str, Any]:
        return {"status": "success", "content": [{"json": {"code": action.code, "language": action.language.value}}]}

    def execute_command(self, action: ExecuteCommandAction) -> Dict[str, Any]:
        return {"status": "success", "content": [{"json": {"command": action.command}}]}

    def read_files(self, action: ReadFilesAction) -> Dict[str, Any]:
        return {"status": "success", "content": [{"json": {"paths": action.paths}}]}

    def list_files(self, action: ListFilesAction) -> Dict[str, Any]:
        return {"status": "success", "content": [{"json": {"path": action.path}}]}

    def remove_files(self, action: RemoveFilesAction) -> Dict[str, Any]:
        return {"status": "success", "content": [{"json": {"paths": action.paths}}]}

    def write_files(self, action: WriteFilesAction) -> Dict[str, Any]:
        return {"status": "success", "content": [{"json": {"filesWritten": len(action.content)}}]}

    @staticmethod
    def get_supported_languages() -> List[LanguageType]:
        return [LanguageType.PYTHON, LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]


@pytest.fixture
def mock_interpreter():
    """Create a mock code interpreter for testing."""
    return MockCodeInterpreter()


def test_code_interpreter_initialization(mock_interpreter):
    """Test CodeInterpreter initialization."""
    assert not mock_interpreter._started
    assert not mock_interpreter.platform_started
    assert not mock_interpreter.platform_cleaned


def test_auto_start_on_first_use(mock_interpreter):
    """Test that platform starts automatically on first tool call."""
    action_input = CodeInterpreterInput(
        action=InitSessionAction(type="initSession", description="Test session", session_name="test-session")
    )

    result = mock_interpreter.code_interpreter(action_input)

    assert mock_interpreter._started
    assert mock_interpreter.platform_started
    assert result["status"] == "success"


def test_init_session_action(mock_interpreter):
    """Test init session action."""
    action_input = CodeInterpreterInput(
        action=InitSessionAction(type="initSession", description="Test session", session_name="my-session")
    )

    result = mock_interpreter.code_interpreter(action_input)

    assert result["status"] == "success"
    assert result["content"][0]["json"]["sessionName"] == "my-session"
    assert result["content"][0]["json"]["description"] == "Test session"


def test_init_session_dict_input(mock_interpreter):
    """Test init session with dictionary input."""
    action_dict = {"action": {"type": "initSession", "description": "Test session", "session_name": "dict-session"}}

    result = mock_interpreter.code_interpreter(action_dict)

    assert result["status"] == "success"
    assert result["content"][0]["json"]["sessionName"] == "dict-session"


def test_list_local_sessions_action(mock_interpreter):
    """Test list local sessions action."""
    action_input = CodeInterpreterInput(action=ListLocalSessionsAction(type="listLocalSessions"))

    result = mock_interpreter.code_interpreter(action_input)

    assert result["status"] == "success"
    assert "sessions" in result["content"][0]["json"]
    assert "totalSessions" in result["content"][0]["json"]


def test_execute_code_action(mock_interpreter):
    """Test execute code action."""
    action_input = CodeInterpreterInput(
        action=ExecuteCodeAction(
            type="executeCode", session_name="test-session", code="print('Hello, World!')", language=LanguageType.PYTHON
        )
    )

    result = mock_interpreter.code_interpreter(action_input)

    assert result["status"] == "success"
    assert result["content"][0]["json"]["code"] == "print('Hello, World!')"
    assert result["content"][0]["json"]["language"] == "python"


def test_execute_command_action(mock_interpreter):
    """Test execute command action."""
    action_input = CodeInterpreterInput(
        action=ExecuteCommandAction(type="executeCommand", session_name="test-session", command="ls -la")
    )

    result = mock_interpreter.code_interpreter(action_input)

    assert result["status"] == "success"
    assert result["content"][0]["json"]["command"] == "ls -la"


def test_read_files_action(mock_interpreter):
    """Test read files action."""
    action_input = CodeInterpreterInput(
        action=ReadFilesAction(type="readFiles", session_name="test-session", paths=["file1.txt", "file2.py"])
    )

    result = mock_interpreter.code_interpreter(action_input)

    assert result["status"] == "success"
    assert result["content"][0]["json"]["paths"] == ["file1.txt", "file2.py"]


def test_list_files_action(mock_interpreter):
    """Test list files action."""
    action_input = CodeInterpreterInput(
        action=ListFilesAction(type="listFiles", session_name="test-session", path="/tmp")
    )

    result = mock_interpreter.code_interpreter(action_input)

    assert result["status"] == "success"
    assert result["content"][0]["json"]["path"] == "/tmp"


def test_remove_files_action(mock_interpreter):
    """Test remove files action."""
    action_input = CodeInterpreterInput(
        action=RemoveFilesAction(type="removeFiles", session_name="test-session", paths=["file1.txt", "file2.py"])
    )

    result = mock_interpreter.code_interpreter(action_input)

    assert result["status"] == "success"
    assert result["content"][0]["json"]["paths"] == ["file1.txt", "file2.py"]


def test_write_files_action(mock_interpreter):
    """Test write files action."""
    action_input = CodeInterpreterInput(
        action=WriteFilesAction(
            type="writeFiles",
            session_name="test-session",
            content=[
                FileContent(path="hello.py", text="print('Hello, World!')"),
                FileContent(path="data.txt", text="Some data"),
            ],
        )
    )

    result = mock_interpreter.code_interpreter(action_input)

    assert result["status"] == "success"
    assert result["content"][0]["json"]["filesWritten"] == 2


def test_unknown_action_type(mock_interpreter):
    """Test handling of unknown action type."""
    with patch("strands_tools.code_interpreter.models.CodeInterpreterInput.model_validate") as mock_validate:
        mock_action = MagicMock()
        mock_action.action = MagicMock()
        mock_validate.return_value = mock_action

        result = mock_interpreter.code_interpreter({"action": {"type": "unknownAction"}})

        assert result["status"] == "error"
        assert "Unknown action:" in result["content"][0]["text"]


def test_cleanup_method(mock_interpreter):
    """Test explicit cleanup method."""
    mock_interpreter._start()
    assert mock_interpreter._started
    assert mock_interpreter.platform_started

    mock_interpreter._cleanup()
    assert not mock_interpreter._started
    assert mock_interpreter.platform_cleaned


def test_cleanup_when_not_started(mock_interpreter):
    """Test cleanup when platform was never started."""
    assert not mock_interpreter._started
    mock_interpreter._cleanup()
    assert not mock_interpreter._started


def test_destructor_cleanup(mock_interpreter):
    """Test that destructor calls cleanup."""
    mock_interpreter._start()
    assert mock_interpreter._started

    mock_interpreter.__del__()
    assert not mock_interpreter._started


def test_destructor_cleanup_exception_handling():
    """Test destructor handles exceptions gracefully."""
    mock_interpreter = MockCodeInterpreter()
    mock_interpreter._start()

    with patch.object(mock_interpreter, "_cleanup", side_effect=Exception("Cleanup error")):
        mock_interpreter.__del__()


def test_multiple_tool_calls_only_start_once(mock_interpreter):
    """Test that platform only starts once even with multiple tool calls."""
    action_input = CodeInterpreterInput(
        action=InitSessionAction(type="initSession", description="Test session", session_name="test-session")
    )

    mock_interpreter.code_interpreter(action_input)
    assert mock_interpreter._started

    mock_interpreter.platform_started = False

    mock_interpreter.code_interpreter(action_input)
    assert mock_interpreter._started
    assert not mock_interpreter.platform_started


def test_dynamic_tool_spec(mock_interpreter):
    assert (
        "The tool supports the following programming languages: PYTHON, JAVASCRIPT, TYPESCRIPT"
        in mock_interpreter.code_interpreter.tool_spec["description"]
    )
