"""
Tests for the AgentCoreCodeInterpreter class.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

mock_bedrock_agentcore = MagicMock()
sys.modules["bedrock_agentcore"] = mock_bedrock_agentcore
sys.modules["bedrock_agentcore.tools"] = MagicMock()
sys.modules["bedrock_agentcore.tools.code_interpreter_client"] = MagicMock()

from strands_tools.code_interpreter.agent_core_code_interpreter import (  # noqa: E402
    AgentCoreCodeInterpreter,
    SessionInfo,
    _session_mapping,
)
from strands_tools.code_interpreter.models import (  # noqa: E402
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
    client.get_session.return_value = {"status": "READY"}
    client.invoke.return_value = {"stream": [{"result": {"content": "Mock response"}}], "isError": False}
    return client


@pytest.fixture
def interpreter():
    """Create a AgentCoreCodeInterpreter instance."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        return AgentCoreCodeInterpreter(region="us-west-2")


@pytest.fixture(autouse=True)
def clear_session_mapping():
    """Clear module-level session mapping before each test."""
    _session_mapping.clear()
    yield
    _session_mapping.clear()


def test_initialization(interpreter):
    """Test AgentCoreCodeInterpreter initialization."""
    assert interpreter.region == "us-west-2"
    assert interpreter.identifier == "aws.codeinterpreter.v1"
    assert interpreter._sessions == {}
    assert not interpreter._started
    assert interpreter.default_session.startswith("session-")
    assert interpreter.auto_create is True
    assert interpreter.persist_sessions is True


def test_initialization_with_new_parameters():
    """Test initialization with new session persistence parameters."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", persist_sessions=False)
        assert interpreter.persist_sessions is False


def test_session_name_no_cleaning():
    """Test that session names are used as-is without cleaning."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"

        # Test with UUID containing dashes - should be preserved
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", session_name="16e4dcba-5792-4643-9e54-42b43dc58637")
        # Session name should be used as-is
        assert interpreter.default_session == "16e4dcba-5792-4643-9e54-42b43dc58637"

        # Test with underscores - should be preserved
        interpreter2 = AgentCoreCodeInterpreter(region="us-west-2", session_name="my_test_session")
        assert interpreter2.default_session == "my_test_session"

        # Test with mixed characters - should be preserved
        interpreter3 = AgentCoreCodeInterpreter(region="us-west-2", session_name="session-name_123")
        assert interpreter3.default_session == "session-name_123"


def test_ensure_session_existing_session(interpreter, mock_client):
    """Test _ensure_session with existing session in local cache."""
    session_info = SessionInfo(session_id="test-id", description="Test", client=mock_client)
    interpreter._sessions["test-session"] = session_info

    session_name, error = interpreter._ensure_session("test-session")

    assert session_name == "test-session"
    assert error is None


def test_ensure_session_reconnection_via_module_cache(mock_client):
    """Test _ensure_session successfully reconnects using module-level cache."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        with patch(
            "strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient"
        ) as mock_client_class:
            mock_resolve.return_value = "us-west-2"

            # Setup module cache with existing session
            _session_mapping["target-session"] = "found-session-id"

            reconnect_client = MagicMock()
            reconnect_client.session_id = "found-session-id"
            reconnect_client.get_session.return_value = {"status": "READY"}
            mock_client_class.return_value = reconnect_client

            interpreter = AgentCoreCodeInterpreter(region="us-west-2", persist_sessions=True)

            session_name, error = interpreter._ensure_session("target-session")

            assert session_name == "target-session"
            assert error is None
            assert "target-session" in interpreter._sessions
            assert interpreter._sessions["target-session"].session_id == "found-session-id"

            # Verify get_session was called, not list_sessions
            reconnect_client.get_session.assert_called_once_with(
                interpreter_id="aws.codeinterpreter.v1", session_id="found-session-id"
            )


def test_ensure_session_module_cache_session_not_ready():
    """Test _ensure_session removes from cache when session not READY."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        with patch(
            "strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient"
        ) as mock_client_class:
            mock_resolve.return_value = "us-west-2"

            # Setup module cache
            _session_mapping["stale-session"] = "stale-session-id"

            reconnect_client = MagicMock()
            reconnect_client.session_id = "new-session-id"
            reconnect_client.get_session.return_value = {"status": "STOPPED"}
            mock_client_class.return_value = reconnect_client

            interpreter = AgentCoreCodeInterpreter(region="us-west-2", persist_sessions=True, auto_create=True)

            with patch.object(interpreter, "init_session") as mock_init:
                mock_init.return_value = {"status": "success", "content": [{"text": "Created"}]}

                session_name, error = interpreter._ensure_session("stale-session")

                assert session_name == "stale-session"
                assert error is None
                # Session should be removed from cache
                assert "stale-session" not in _session_mapping
                # New session should be created
                mock_init.assert_called_once()


def test_ensure_session_module_cache_get_session_fails():
    """Test _ensure_session falls back to creation when get_session fails."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        with patch(
            "strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient"
        ) as mock_client_class:
            mock_resolve.return_value = "us-west-2"

            # Setup module cache
            _session_mapping["missing-session"] = "missing-session-id"

            reconnect_client = MagicMock()
            reconnect_client.get_session.side_effect = Exception("Session not found")
            reconnect_client.session_id = "new-session-id"
            mock_client_class.return_value = reconnect_client

            interpreter = AgentCoreCodeInterpreter(region="us-west-2", persist_sessions=True, auto_create=True)

            with patch.object(interpreter, "init_session") as mock_init:
                mock_init.return_value = {"status": "success", "content": [{"text": "Created"}]}

                session_name, error = interpreter._ensure_session("missing-session")

                assert session_name == "missing-session"
                assert error is None
                # Session should be removed from cache after error
                assert "missing-session" not in _session_mapping
                mock_init.assert_called_once()


def test_ensure_session_new_session_with_auto_create(interpreter):
    """Test _ensure_session with auto session creation."""
    with patch.object(interpreter, "init_session") as mock_init:
        mock_init.return_value = {"status": "success", "content": [{"text": "Created"}]}

        session_name, error = interpreter._ensure_session("new-session")

        assert session_name == "new-session"
        assert error is None
        mock_init.assert_called_once()
        call_args = mock_init.call_args[0][0]
        assert call_args.session_name == "new-session"
        assert "Auto-initialized" in call_args.description


def test_ensure_session_no_auto_create():
    """Test _ensure_session with auto create disabled raises ValueError."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", auto_create=False)

        with pytest.raises(ValueError, match="Session 'non-existent' not found"):
            interpreter._ensure_session("non-existent")


def test_ensure_session_default_session_name(interpreter):
    """Test _ensure_session uses default session name when none provided."""
    with patch.object(interpreter, "init_session") as mock_init:
        mock_init.return_value = {"status": "success", "content": [{"text": "Created"}]}

        session_name, error = interpreter._ensure_session(None)

        assert session_name == interpreter.default_session
        assert error is None


def test_initialization_with_default_region():
    """Test initialization with default region resolution."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-east-1"
        interpreter = AgentCoreCodeInterpreter()
        assert interpreter.region == "us-east-1"
        mock_resolve.assert_called_once_with(None)


def test_constructor_custom_identifier_initialization():
    """Test initialization with custom identifier."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        custom_id = "custom-interpreter-def456"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier=custom_id)

        assert interpreter.region == "us-west-2"
        assert interpreter.identifier == custom_id
        assert interpreter._sessions == {}
        assert not interpreter._started


def test_constructor_default_identifier_fallback():
    """Test default identifier when none provided."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2")

        assert interpreter.region == "us-west-2"
        assert interpreter.identifier == "aws.codeinterpreter.v1"
        assert interpreter._sessions == {}
        assert not interpreter._started


def test_constructor_backward_compatibility_region_only():
    """Test backward compatibility with existing constructor calls (region only)."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-east-1"
        interpreter = AgentCoreCodeInterpreter("us-east-1")

        assert interpreter.region == "us-east-1"
        assert interpreter.identifier == "aws.codeinterpreter.v1"
        assert interpreter._sessions == {}
        assert not interpreter._started


def test_constructor_backward_compatibility_no_params():
    """Test backward compatibility with existing constructor calls (no parameters)."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-east-1"
        interpreter = AgentCoreCodeInterpreter()

        assert interpreter.region == "us-east-1"
        assert interpreter.identifier == "aws.codeinterpreter.v1"
        assert interpreter._sessions == {}
        assert not interpreter._started


def test_constructor_instance_variable_storage_scenarios():
    """Test that instance variable is set correctly in all scenarios."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"

        custom_id = "test.codeinterpreter.v1"
        interpreter1 = AgentCoreCodeInterpreter(region="us-west-2", identifier=custom_id)
        assert hasattr(interpreter1, "identifier")
        assert interpreter1.identifier == custom_id

        interpreter2 = AgentCoreCodeInterpreter(region="us-west-2", identifier=None)
        assert hasattr(interpreter2, "identifier")
        assert interpreter2.identifier == "aws.codeinterpreter.v1"

        interpreter3 = AgentCoreCodeInterpreter(region="us-west-2", identifier="")
        assert hasattr(interpreter3, "identifier")
        assert interpreter3.identifier == "aws.codeinterpreter.v1"

        interpreter4 = AgentCoreCodeInterpreter(region="us-west-2")
        assert hasattr(interpreter4, "identifier")
        assert interpreter4.identifier == "aws.codeinterpreter.v1"


def test_constructor_custom_identifier_with_complex_format():
    """Test initialization with complex custom identifier."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        complex_id = "my-custom-interpreter-abc123-prod"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier=complex_id)

        assert interpreter.region == "us-west-2"
        assert interpreter.identifier == complex_id
        assert interpreter._sessions == {}
        assert not interpreter._started


def test_cleanup_platform_when_not_initialized(interpreter):
    """Test cleanup when platform is not initialized."""
    interpreter.cleanup_platform()


def test_cleanup_platform_with_persist_sessions_true(interpreter, mock_client):
    """Test cleanup does NOT stop sessions when persist_sessions=True (default)."""
    interpreter._started = True
    interpreter.persist_sessions = True

    session_info = SessionInfo(session_id="test-session-id-123", description="Test session", client=mock_client)
    interpreter._sessions["test-session"] = session_info

    interpreter.cleanup_platform()

    mock_client.stop.assert_not_called()
    assert len(interpreter._sessions) > 0


def test_cleanup_platform_with_persist_sessions_false(mock_client):
    """Test cleanup DOES stop sessions when persist_sessions=False."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", persist_sessions=False)

        interpreter._started = True

        session_info = SessionInfo(session_id="test-session-id-123", description="Test session", client=mock_client)
        interpreter._sessions["test-session"] = session_info

        interpreter.cleanup_platform()

        mock_client.stop.assert_called_once()
        assert interpreter._sessions == {}


def test_cleanup_platform_with_exception_in_stop(mock_client):
    """Test cleanup handles exceptions in client.stop()."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", persist_sessions=False)

        interpreter._started = True
        mock_client.stop.side_effect = Exception("Stop failed")

        session_info = SessionInfo(session_id="test-session-id-123", description="Test session", client=mock_client)
        interpreter._sessions["test-session"] = session_info

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

    mock_client_class.assert_called_once_with(region="us-west-2")
    mock_client.start.assert_called_once_with(identifier="aws.codeinterpreter.v1", name="my-session")

    assert "my-session" in interpreter._sessions
    session_info = interpreter._sessions["my-session"]
    assert isinstance(session_info, SessionInfo)
    assert session_info.session_id == "test-session-id-123"
    assert session_info.description == "Test session"
    assert session_info.client == mock_client

    # Check module-level cache
    assert _session_mapping.get("my-session") == "test-session-id-123"


@patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient")
def test_init_session_with_custom_identifier(mock_client_class, mock_client):
    """Test session initialization with custom identifier passes identifier to client.start()."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        mock_client_class.return_value = mock_client

        custom_id = "my-custom-interpreter-abc123"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier=custom_id)

        action = InitSessionAction(type="initSession", description="Test session", session_name="custom-session")

        result = interpreter.init_session(action)

        assert result["status"] == "success"
        assert result["content"][0]["json"]["sessionName"] == "custom-session"
        assert result["content"][0]["json"]["description"] == "Test session"
        assert result["content"][0]["json"]["sessionId"] == "test-session-id-123"

        mock_client_class.assert_called_once_with(region="us-west-2")
        mock_client.start.assert_called_once_with(identifier=custom_id, name="custom-session")

        assert "custom-session" in interpreter._sessions
        session_info = interpreter._sessions["custom-session"]
        assert isinstance(session_info, SessionInfo)
        assert session_info.session_id == "test-session-id-123"
        assert session_info.description == "Test session"
        assert session_info.client == mock_client


@patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient")
def test_init_session_with_default_identifier(mock_client_class, mock_client):
    """Test session initialization uses default identifier when none provided."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        mock_client_class.return_value = mock_client

        interpreter = AgentCoreCodeInterpreter(region="us-west-2")

        action = InitSessionAction(type="initSession", description="Test session", session_name="default-session")

        result = interpreter.init_session(action)

        assert result["status"] == "success"
        assert result["content"][0]["json"]["sessionName"] == "default-session"
        assert result["content"][0]["json"]["description"] == "Test session"
        assert result["content"][0]["json"]["sessionId"] == "test-session-id-123"

        mock_client_class.assert_called_once_with(region="us-west-2")
        mock_client.start.assert_called_once_with(identifier="aws.codeinterpreter.v1", name="default-session")

        assert "default-session" in interpreter._sessions
        session_info = interpreter._sessions["default-session"]
        assert isinstance(session_info, SessionInfo)
        assert session_info.session_id == "test-session-id-123"
        assert session_info.description == "Test session"
        assert session_info.client == mock_client


@patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient")
def test_init_session_multiple_identifiers_verification(mock_client_class, mock_client):
    """Test that different interpreter instances with different identifiers work correctly."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        mock_client_class.return_value = mock_client

        custom_id1 = "first.codeinterpreter.v1"
        interpreter1 = AgentCoreCodeInterpreter(region="us-west-2", identifier=custom_id1)

        custom_id2 = "second.codeinterpreter.v1"
        interpreter2 = AgentCoreCodeInterpreter(region="us-west-2", identifier=custom_id2)

        interpreter3 = AgentCoreCodeInterpreter(region="us-west-2")

        action1 = InitSessionAction(type="initSession", description="First session", session_name="session1")
        result1 = interpreter1.init_session(action1)
        assert result1["status"] == "success"

        action2 = InitSessionAction(type="initSession", description="Second session", session_name="session2")
        result2 = interpreter2.init_session(action2)
        assert result2["status"] == "success"

        action3 = InitSessionAction(type="initSession", description="Third session", session_name="session3")
        result3 = interpreter3.init_session(action3)
        assert result3["status"] == "success"

        assert mock_client.start.call_count == 3
        call_args_list = mock_client.start.call_args_list

        assert call_args_list[0] == ((), {"identifier": custom_id1, "name": "session1"})
        assert call_args_list[1] == ((), {"identifier": custom_id2, "name": "session2"})
        assert call_args_list[2] == ((), {"identifier": "aws.codeinterpreter.v1", "name": "session3"})


@patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient")
def test_init_session_already_exists(mock_client_class, interpreter, mock_client):
    """Test session initialization when session already exists."""
    session_info = SessionInfo(session_id="existing-id", description="Existing session", client=mock_client)
    interpreter._sessions["existing-session"] = session_info

    action = InitSessionAction(type="initSession", description="Test session", session_name="existing-session")

    result = interpreter.init_session(action)

    assert result["status"] == "error"
    assert "already exists" in result["content"][0]["text"]

    mock_client_class.assert_not_called()


@patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient")
def test_init_session_client_start_exception(mock_client_class, interpreter, mock_client):
    """Test session initialization when client.start() raises exception."""
    mock_client.start.side_effect = Exception("Start failed")
    mock_client_class.return_value = mock_client

    action = InitSessionAction(type="initSession", description="Test session", session_name="fail-session")

    result = interpreter.init_session(action)

    assert result["status"] == "error"
    assert "Failed to initialize session 'fail-session': Start failed" in result["content"][0]["text"]


def test_list_local_sessions_empty(interpreter):
    """Test listing sessions when no sessions exist."""
    result = interpreter.list_local_sessions()

    assert result["status"] == "success"
    assert result["content"][0]["json"]["sessions"] == []
    assert result["content"][0]["json"]["totalSessions"] == 0


def test_list_local_sessions_with_sessions(interpreter, mock_client):
    """Test listing sessions with active sessions."""
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


def test_execute_code_session_not_found():
    """Test code execution with non-existent session when auto_create is disabled."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", auto_create=False)

        action = ExecuteCodeAction(
            type="executeCode", session_name="non-existent", code="print('Hello')", language=LanguageType.PYTHON
        )

        with pytest.raises(ValueError, match="Session 'non-existent' not found"):
            interpreter.execute_code(action)


def test_execute_command_success(interpreter, mock_client):
    """Test successful command execution."""
    session_info = SessionInfo(session_id="test-session-id-123", description="Test session", client=mock_client)
    interpreter._sessions["test-session"] = session_info

    action = ExecuteCommandAction(type="executeCommand", session_name="test-session", command="ls -la")

    result = interpreter.execute_command(action)

    assert result["status"] == "success"
    mock_client.invoke.assert_called_once_with("executeCommand", {"command": "ls -la"})


def test_execute_command_session_not_found():
    """Test command execution with non-existent session when auto_create is disabled."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", auto_create=False)

        action = ExecuteCommandAction(type="executeCommand", session_name="non-existent", command="ls -la")

        with pytest.raises(ValueError, match="Session 'non-existent' not found"):
            interpreter.execute_command(action)


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


@patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient")
def test_execute_code_with_auto_session_creation(mock_client_class, interpreter):
    """Test code execution with automatic session creation."""
    mock_client = mock_client_class.return_value
    mock_client.session_id = "auto-session-id"
    mock_client.invoke.return_value = {"stream": [{"result": {"content": "Success"}}], "isError": False}

    action = ExecuteCodeAction(
        type="executeCode",
        session_name=None,
        code="print('Auto session')",
        language=LanguageType.PYTHON,
    )

    result = interpreter.execute_code(action)

    assert result["status"] == "success"

    assert len(interpreter._sessions) == 1
    auto_created_session = list(interpreter._sessions.keys())[0]
    assert auto_created_session.startswith("session-")

    mock_client.invoke.assert_called_with(
        "executeCode", {"code": "print('Auto session')", "language": "python", "clearContext": False}
    )


def test_module_level_session_mapping():
    """Test module-level session mapping persists across instances."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        with patch(
            "strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient"
        ) as mock_client_class:
            mock_resolve.return_value = "us-west-2"

            mock_client1 = MagicMock()
            mock_client1.session_id = "aws-session-123"

            # First instance creates session
            interpreter1 = AgentCoreCodeInterpreter(region="us-west-2")
            mock_client_class.return_value = mock_client1

            action = InitSessionAction(type="initSession", description="Test", session_name="shared-session")
            result = interpreter1.init_session(action)

            assert result["status"] == "success"
            assert _session_mapping["shared-session"] == "aws-session-123"

            # Second instance should find session in module cache
            mock_client2 = MagicMock()
            mock_client2.get_session.return_value = {"status": "READY"}
            mock_client_class.return_value = mock_client2

            interpreter2 = AgentCoreCodeInterpreter(region="us-west-2")
            session_name, error = interpreter2._ensure_session("shared-session")

            assert session_name == "shared-session"
            assert error is None
            # Should have called get_session with cached ID
            mock_client2.get_session.assert_called_once_with(
                interpreter_id="aws.codeinterpreter.v1", session_id="aws-session-123"
            )
