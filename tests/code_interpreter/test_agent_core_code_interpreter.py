"""
Tests for the AgentCoreCodeInterpreter class.
"""

from unittest.mock import MagicMock, patch

import pytest
from strands_tools.code_interpreter.agent_core_code_interpreter import (
    AgentCoreCodeInterpreter,
    SessionInfo,
)
from strands_tools.code_interpreter.models import (
    ExecuteCodeAction,
    ExecuteCommandAction,
    FileContent,
    InitSessionAction,
    LanguageType,
    ListFilesAction,
    ReadFilesAction,
    RemoveFilesAction,
    WriteFilesAction,
)


@pytest.fixture
def mock_client():
    """Create a mock Bedrock AgentCore client."""
    client = MagicMock()
    client.session_id = "test-session-id-123"
    client.start.return_value = None
    client.stop.return_value = None
    client.invoke.return_value = {"stream": [{"result": {"content": "Mock response"}}], "isError": False}
    return client


@pytest.fixture
def interpreter():
    """Create a AgentCoreCodeInterpreter instance."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        return AgentCoreCodeInterpreter(region="us-west-2")


def test_initialization(interpreter):
    """Test AgentCoreCodeInterpreter initialization."""
    assert interpreter.region == "us-west-2"
    assert interpreter._sessions == {}
    assert not interpreter._started


def test_initialization_with_default_region():
    """Test initialization with default region resolution."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-east-1"
        interpreter = AgentCoreCodeInterpreter()
        assert interpreter.region == "us-east-1"
        mock_resolve.assert_called_once_with(None)


def test_cleanup_platform_when_not_initialized(interpreter):
    """Test cleanup when platform is not initialized."""
    interpreter.cleanup_platform()
    # Should not raise any errors


def test_cleanup_platform_with_sessions(interpreter, mock_client):
    """Test cleanup with active sessions."""
    interpreter._started = True
    session_info = SessionInfo(session_id="test-session-id-123", description="Test session", client=mock_client)
    interpreter._sessions["test-session"] = session_info

    interpreter.cleanup_platform()

    mock_client.stop.assert_called_once()
    assert interpreter._sessions == {}


def test_cleanup_platform_with_client_without_stop_method(interpreter):
    """Test cleanup with client that doesn't have stop method."""
    interpreter._started = True
    mock_client_no_stop = MagicMock()
    del mock_client_no_stop.stop  # Remove stop method
    session_info = SessionInfo(session_id="test-session-id-123", description="Test session", client=mock_client_no_stop)
    interpreter._sessions["test-session"] = session_info

    # Should not raise exception
    interpreter.cleanup_platform()
    assert interpreter._sessions == {}


def test_cleanup_platform_with_exception_in_stop(interpreter, mock_client):
    """Test cleanup handles exceptions in client.stop()."""
    interpreter._started = True
    mock_client.stop.side_effect = Exception("Stop failed")
    session_info = SessionInfo(session_id="test-session-id-123", description="Test session", client=mock_client)
    interpreter._sessions["test-session"] = session_info

    # Should not raise exception
    interpreter.cleanup_platform()
    assert interpreter._sessions == {}


@patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient")
def test_init_session_success(mock_client_class, interpreter, mock_client):
    """Test successful session initialization."""
    mock_client_class.return_value = mock_client

    action = InitSessionAction(type="initSession", description="Test session", session_name="my-session")

    result = interpreter.init_session(action)

    assert result["status"] == "success"
    assert result["content"][0]["json"]["sessionName"] == "my-session"
    assert result["content"][0]["json"]["description"] == "Test session"
    assert result["content"][0]["json"]["sessionId"] == "test-session-id-123"

    # Verify client was created and started
    mock_client_class.assert_called_once_with(region="us-west-2")
    mock_client.start.assert_called_once()

    # Verify session was stored with SessionInfo structure
    assert "my-session" in interpreter._sessions
    session_info = interpreter._sessions["my-session"]
    assert isinstance(session_info, SessionInfo)
    assert session_info.session_id == "test-session-id-123"
    assert session_info.description == "Test session"
    assert session_info.client == mock_client


@patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient")
def test_init_session_already_exists(mock_client_class, interpreter, mock_client):
    """Test session initialization when session already exists."""
    # Pre-populate a session
    session_info = SessionInfo(session_id="existing-id", description="Existing session", client=mock_client)
    interpreter._sessions["existing-session"] = session_info

    action = InitSessionAction(type="initSession", description="Test session", session_name="existing-session")

    result = interpreter.init_session(action)

    assert result["status"] == "error"
    assert "already exists" in result["content"][0]["text"]

    # Client should not be created
    mock_client_class.assert_not_called()


@patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient")
def test_init_session_client_start_exception(mock_client_class, interpreter, mock_client):
    """Test session initialization when client.start() raises exception."""
    mock_client.start.side_effect = Exception("Start failed")
    mock_client_class.return_value = mock_client

    action = InitSessionAction(type="initSession", description="Test session", session_name="fail-session")

    with pytest.raises(Exception, match="Start failed"):
        interpreter.init_session(action)


def test_list_local_sessions_empty(interpreter):
    """Test listing sessions when no sessions exist."""
    result = interpreter.list_local_sessions()

    assert result["status"] == "success"
    assert result["content"][0]["json"]["sessions"] == []
    assert result["content"][0]["json"]["totalSessions"] == 0


def test_list_local_sessions_with_sessions(interpreter, mock_client):
    """Test listing sessions with active sessions."""
    # Add some sessions using SessionInfo structure
    session_info1 = SessionInfo(session_id="id1", description="First session", client=mock_client)
    session_info2 = SessionInfo(session_id="id2", description="Second session", client=mock_client)
    interpreter._sessions["session1"] = session_info1
    interpreter._sessions["session2"] = session_info2

    result = interpreter.list_local_sessions()

    assert result["status"] == "success"
    assert result["content"][0]["json"]["totalSessions"] == 2
    sessions = result["content"][0]["json"]["sessions"]
    assert len(sessions) == 2

    session_names = [s["sessionName"] for s in sessions]
    assert "session1" in session_names
    assert "session2" in session_names

    # Verify session details
    for session in sessions:
        if session["sessionName"] == "session1":
            assert session["description"] == "First session"
            assert session["sessionId"] == "id1"
        elif session["sessionName"] == "session2":
            assert session["description"] == "Second session"
            assert session["sessionId"] == "id2"


def test_execute_code_success(interpreter, mock_client):
    """Test successful code execution."""
    session_info = SessionInfo(session_id="test-session-id-123", description="Test session", client=mock_client)
    interpreter._sessions["test-session"] = session_info

    action = ExecuteCodeAction(
        type="executeCode", session_name="test-session", code="print('Hello, World!')", language=LanguageType.PYTHON
    )

    result = interpreter.execute_code(action)

    assert result["status"] == "success"
    mock_client.invoke.assert_called_once_with(
        "executeCode", {"code": "print('Hello, World!')", "language": "python", "clearContext": False}
    )


def test_execute_code_session_not_found(interpreter):
    """Test code execution with non-existent session."""
    action = ExecuteCodeAction(
        type="executeCode", session_name="non-existent", code="print('Hello')", language=LanguageType.PYTHON
    )

    result = interpreter.execute_code(action)

    assert result["status"] == "error"
    assert "not found" in result["content"][0]["text"]


def test_execute_command_success(interpreter, mock_client):
    """Test successful command execution."""
    session_info = SessionInfo(session_id="test-session-id-123", description="Test session", client=mock_client)
    interpreter._sessions["test-session"] = session_info

    action = ExecuteCommandAction(type="executeCommand", session_name="test-session", command="ls -la")

    result = interpreter.execute_command(action)

    assert result["status"] == "success"
    mock_client.invoke.assert_called_once_with("executeCommand", {"command": "ls -la"})


def test_execute_command_session_not_found(interpreter):
    """Test command execution with non-existent session."""
    action = ExecuteCommandAction(type="executeCommand", session_name="non-existent", command="ls -la")

    result = interpreter.execute_command(action)

    assert result["status"] == "error"
    assert "not found" in result["content"][0]["text"]


def test_read_files_success(interpreter, mock_client):
    """Test successful file reading."""
    session_info = SessionInfo(session_id="test-session-id-123", description="Test session", client=mock_client)
    interpreter._sessions["test-session"] = session_info

    action = ReadFilesAction(type="readFiles", session_name="test-session", paths=["file1.txt", "file2.py"])

    result = interpreter.read_files(action)

    assert result["status"] == "success"
    mock_client.invoke.assert_called_once_with("readFiles", {"paths": ["file1.txt", "file2.py"]})


def test_list_files_success(interpreter, mock_client):
    """Test successful file listing."""
    session_info = SessionInfo(session_id="test-session-id-123", description="Test session", client=mock_client)
    interpreter._sessions["test-session"] = session_info

    action = ListFilesAction(type="listFiles", session_name="test-session", path="/tmp")

    result = interpreter.list_files(action)

    assert result["status"] == "success"
    mock_client.invoke.assert_called_once_with("listFiles", {"path": "/tmp"})


def test_remove_files_success(interpreter, mock_client):
    """Test successful file removal."""
    session_info = SessionInfo(session_id="test-session-id-123", description="Test session", client=mock_client)
    interpreter._sessions["test-session"] = session_info

    action = RemoveFilesAction(type="removeFiles", session_name="test-session", paths=["file1.txt", "file2.py"])

    result = interpreter.remove_files(action)

    assert result["status"] == "success"
    mock_client.invoke.assert_called_once_with("removeFiles", {"paths": ["file1.txt", "file2.py"]})


def test_write_files_success(interpreter, mock_client):
    """Test successful file writing."""
    session_info = SessionInfo(session_id="test-session-id-123", description="Test session", client=mock_client)
    interpreter._sessions["test-session"] = session_info

    action = WriteFilesAction(
        type="writeFiles",
        session_name="test-session",
        content=[
            FileContent(path="hello.py", text="print('Hello, World!')"),
            FileContent(path="data.txt", text="Some data"),
        ],
    )

    result = interpreter.write_files(action)

    assert result["status"] == "success"
    mock_client.invoke.assert_called_once_with(
        "writeFiles",
        {
            "content": [
                {"path": "hello.py", "text": "print('Hello, World!')"},
                {"path": "data.txt", "text": "Some data"},
            ]
        },
    )


def test_create_tool_result_with_stream(interpreter):
    """Test _create_tool_result with stream response."""
    response = {"stream": [{"result": {"content": "Test output"}}], "isError": False}

    result = interpreter._create_tool_result(response)

    assert result["status"] == "success"
    assert result["content"][0]["text"] == "Test output"


def test_create_tool_result_with_error_stream(interpreter):
    """Test _create_tool_result with error stream response."""
    response = {"stream": [{"result": {"content": "Error occurred"}}], "isError": True}

    result = interpreter._create_tool_result(response)

    assert result["status"] == "error"
    assert result["content"][0]["text"] == "Error occurred"


def test_create_tool_result_with_empty_stream(interpreter):
    """Test _create_tool_result with empty stream."""
    response = {"stream": [], "isError": False}

    result = interpreter._create_tool_result(response)

    assert result["status"] == "error"
    assert "Failed to create tool result" in result["content"][0]["text"]


def test_create_tool_result_without_stream(interpreter):
    """Test _create_tool_result with direct response (no stream)."""
    response = {"status": "success", "content": [{"text": "Direct response"}]}

    result = interpreter._create_tool_result(response)

    assert result == response


def test_create_tool_result_with_stream_no_result(interpreter):
    """Test _create_tool_result with stream but no result key."""
    response = {"stream": [{"other": "data"}], "isError": False}

    result = interpreter._create_tool_result(response)

    assert result["status"] == "error"
    assert "Failed to create tool result" in result["content"][0]["text"]


def test_session_info_dataclass():
    """Test SessionInfo dataclass functionality."""
    mock_client = MagicMock()
    session_info = SessionInfo(session_id="test-id", description="Test description", client=mock_client)

    assert session_info.session_id == "test-id"
    assert session_info.description == "Test description"
    assert session_info.client == mock_client


def test_get_supported_languages():
    """Test get_supported_languages static method."""
    languages = AgentCoreCodeInterpreter.get_supported_languages()

    assert LanguageType.PYTHON in languages
    assert LanguageType.JAVASCRIPT in languages
    assert LanguageType.TYPESCRIPT in languages
    assert len(languages) == 3
