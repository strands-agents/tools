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
    assert interpreter.identifier == "aws.codeinterpreter.v1"  # Should use default identifier
    assert interpreter._sessions == {}
    assert not interpreter._started
    assert interpreter.default_session.startswith("session-")


def test_ensure_session_existing_session(interpreter, mock_client):
    """Test _ensure_session with existing session."""
    # Create a test session
    session_info = SessionInfo(session_id="test-id", description="Test", client=mock_client)
    interpreter._sessions["test-session"] = session_info

    # Test with existing session
    session_name, error = interpreter._ensure_session("test-session")

    assert session_name == "test-session"
    assert error is None


def test_ensure_session_new_session_with_auto_session(interpreter):
    """Test _ensure_session with auto session creation."""
    with patch.object(interpreter, "init_session") as mock_init:
        mock_init.return_value = {"status": "success", "content": [{"text": "Created"}]}

        # Test with non-existent session but auto_session enabled
        session_name, error = interpreter._ensure_session("new-session")

        assert session_name == "new-session"
        assert error is None
        mock_init.assert_called_once()
        # Verify init_session was called with correct parameters
        call_args = mock_init.call_args[0][0]
        assert call_args.session_name == "new-session"
        assert "Auto-initialized" in call_args.description


def test_ensure_session_no_auto_session():
    """Test _ensure_session with auto create disabled."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", auto_create=False)  # Disable auto_create

        # Test with non-existent session - should raise ValueError
        with pytest.raises(ValueError, match="Session 'non-existent' not found. Create it first using initSession."):
            interpreter._ensure_session("non-existent")


def test_ensure_session_default_session_name(interpreter):
    """Test _ensure_session uses default session name when none provided."""
    with patch.object(interpreter, "init_session") as mock_init:
        mock_init.return_value = {"status": "success", "content": [{"text": "Created"}]}

        # Test with None session name
        session_name, error = interpreter._ensure_session(None)

        assert session_name == interpreter.default_session  # Use actual default session name
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
        # This is how existing code would call the constructor
        interpreter = AgentCoreCodeInterpreter("us-east-1")

        assert interpreter.region == "us-east-1"
        assert interpreter.identifier == "aws.codeinterpreter.v1"
        assert interpreter._sessions == {}
        assert not interpreter._started


def test_constructor_backward_compatibility_no_params():
    """Test backward compatibility with existing constructor calls (no parameters)."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-east-1"
        # This is how existing code would call the constructor
        interpreter = AgentCoreCodeInterpreter()

        assert interpreter.region == "us-east-1"
        assert interpreter.identifier == "aws.codeinterpreter.v1"
        assert interpreter._sessions == {}
        assert not interpreter._started


def test_constructor_instance_variable_storage_scenarios():
    """Test that instance variable is set correctly in all scenarios."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"

        # Scenario 1: Custom identifier provided
        custom_id = "test.codeinterpreter.v1"
        interpreter1 = AgentCoreCodeInterpreter(region="us-west-2", identifier=custom_id)
        assert hasattr(interpreter1, "identifier")
        assert interpreter1.identifier == custom_id

        # Scenario 2: None identifier provided (explicit None)
        interpreter2 = AgentCoreCodeInterpreter(region="us-west-2", identifier=None)
        assert hasattr(interpreter2, "identifier")
        assert interpreter2.identifier == "aws.codeinterpreter.v1"

        # Scenario 3: Empty string identifier provided
        interpreter3 = AgentCoreCodeInterpreter(region="us-west-2", identifier="")
        assert hasattr(interpreter3, "identifier")
        assert interpreter3.identifier == "aws.codeinterpreter.v1"

        # Scenario 4: No identifier parameter provided
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
def test_init_session_with_custom_identifier(mock_client_class, mock_client):
    """Test session initialization with custom identifier passes identifier to client.start()."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        mock_client_class.return_value = mock_client

        # Create interpreter with custom identifier
        custom_id = "my-custom-interpreter-abc123"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier=custom_id)

        action = InitSessionAction(type="initSession", description="Test session", session_name="custom-session")

        result = interpreter.init_session(action)

        assert result["status"] == "success"
        assert result["content"][0]["json"]["sessionName"] == "custom-session"
        assert result["content"][0]["json"]["description"] == "Test session"
        assert result["content"][0]["json"]["sessionId"] == "test-session-id-123"

        # Verify client was created and started with custom identifier
        mock_client_class.assert_called_once_with(region="us-west-2")
        mock_client.start.assert_called_once_with(identifier=custom_id)

        # Verify session was stored
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

        # Create interpreter without custom identifier (should use default)
        interpreter = AgentCoreCodeInterpreter(region="us-west-2")

        action = InitSessionAction(type="initSession", description="Test session", session_name="default-session")

        result = interpreter.init_session(action)

        assert result["status"] == "success"
        assert result["content"][0]["json"]["sessionName"] == "default-session"
        assert result["content"][0]["json"]["description"] == "Test session"
        assert result["content"][0]["json"]["sessionId"] == "test-session-id-123"

        # Verify client was created and started with default identifier
        mock_client_class.assert_called_once_with(region="us-west-2")
        mock_client.start.assert_called_once_with(identifier="aws.codeinterpreter.v1")

        # Verify session was stored
        assert "default-session" in interpreter._sessions
        session_info = interpreter._sessions["default-session"]
        assert isinstance(session_info, SessionInfo)
        assert session_info.session_id == "test-session-id-123"
        assert session_info.description == "Test session"
        assert session_info.client == mock_client


@patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient")
@patch("strands_tools.code_interpreter.agent_core_code_interpreter.logger")
def test_init_session_logging_includes_identifier(mock_logger, mock_client_class, mock_client):
    """Test that session initialization logging includes identifier information."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        mock_client_class.return_value = mock_client

        # Test with custom identifier
        custom_id = "test.codeinterpreter.v1"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier=custom_id)

        action = InitSessionAction(type="initSession", description="Test session", session_name="log-test-session")

        result = interpreter.init_session(action)

        assert result["status"] == "success"

        # Verify logging calls include identifier information
        mock_logger.info.assert_any_call(
            f"Initializing Bedrock AgentCoresandbox session: Test session with identifier: {custom_id}"
        )
        mock_logger.info.assert_any_call(
            f"Initialized session: log-test-session (ID: test-session-id-123) with identifier: {custom_id}"
        )


@patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient")
@patch("strands_tools.code_interpreter.agent_core_code_interpreter.logger")
def test_init_session_logging_includes_default_identifier(mock_logger, mock_client_class, mock_client):
    """Test that session initialization logging includes default identifier when none provided."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        mock_client_class.return_value = mock_client

        # Test with default identifier (none provided)
        interpreter = AgentCoreCodeInterpreter(region="us-west-2")

        action = InitSessionAction(type="initSession", description="Test session", session_name="log-default-session")

        result = interpreter.init_session(action)

        assert result["status"] == "success"

        # Verify logging calls include default identifier information
        default_id = "aws.codeinterpreter.v1"
        mock_logger.info.assert_any_call(
            f"Initializing Bedrock AgentCoresandbox session: Test session with identifier: {default_id}"
        )
        mock_logger.info.assert_any_call(
            f"Initialized session: log-default-session (ID: test-session-id-123) with identifier: {default_id}"
        )


@patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient")
@patch("strands_tools.code_interpreter.agent_core_code_interpreter.logger")
def test_init_session_error_logging_includes_identifier(mock_logger, mock_client_class, mock_client):
    """Test that session initialization error logging includes identifier information."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        mock_client.start.side_effect = Exception("Start failed")
        mock_client_class.return_value = mock_client

        # Test with custom identifier
        custom_id = "error.codeinterpreter.v1"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", identifier=custom_id)

        action = InitSessionAction(type="initSession", description="Test session", session_name="error-session")

        result = interpreter.init_session(action)

        assert result["status"] == "error"
        assert "Failed to initialize session 'error-session': Start failed" in result["content"][0]["text"]

        # Verify error logging includes identifier information
        mock_logger.error.assert_called_once_with(
            f"Failed to initialize session 'error-session' with identifier: {custom_id}. Error: Start failed"
        )


@patch("strands_tools.code_interpreter.agent_core_code_interpreter.BedrockAgentCoreCodeInterpreterClient")
def test_init_session_multiple_identifiers_verification(mock_client_class, mock_client):
    """Test that different interpreter instances with different identifiers work correctly."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        mock_client_class.return_value = mock_client

        # Create first interpreter with custom identifier
        custom_id1 = "first.codeinterpreter.v1"
        interpreter1 = AgentCoreCodeInterpreter(region="us-west-2", identifier=custom_id1)

        # Create second interpreter with different custom identifier
        custom_id2 = "second.codeinterpreter.v1"
        interpreter2 = AgentCoreCodeInterpreter(region="us-west-2", identifier=custom_id2)

        # Create third interpreter with default identifier
        interpreter3 = AgentCoreCodeInterpreter(region="us-west-2")

        # Test first interpreter
        action1 = InitSessionAction(type="initSession", description="First session", session_name="session1")
        result1 = interpreter1.init_session(action1)
        assert result1["status"] == "success"

        # Test second interpreter
        action2 = InitSessionAction(type="initSession", description="Second session", session_name="session2")
        result2 = interpreter2.init_session(action2)
        assert result2["status"] == "success"

        # Test third interpreter
        action3 = InitSessionAction(type="initSession", description="Third session", session_name="session3")
        result3 = interpreter3.init_session(action3)
        assert result3["status"] == "success"

        # Verify each interpreter used its correct identifier
        assert mock_client.start.call_count == 3
        call_args_list = mock_client.start.call_args_list

        # First call should use custom_id1
        assert call_args_list[0] == ((), {"identifier": custom_id1})

        # Second call should use custom_id2
        assert call_args_list[1] == ((), {"identifier": custom_id2})

        # Third call should use default identifier
        assert call_args_list[2] == ((), {"identifier": "aws.codeinterpreter.v1"})


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


def test_execute_code_session_not_found():
    """Test code execution with non-existent session when auto_create is disabled."""
    with patch("strands_tools.code_interpreter.agent_core_code_interpreter.resolve_region") as mock_resolve:
        mock_resolve.return_value = "us-west-2"
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", auto_create=False)  # Disable auto_create

        action = ExecuteCodeAction(
            type="executeCode", session_name="non-existent", code="print('Hello')", language=LanguageType.PYTHON
        )

        # Expect ValueError to be raised
        with pytest.raises(ValueError, match="Session 'non-existent' not found. Create it first using initSession."):
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
        interpreter = AgentCoreCodeInterpreter(region="us-west-2", auto_create=False)  # Disable auto_create

        action = ExecuteCommandAction(type="executeCommand", session_name="non-existent", command="ls -la")

        # Expect ValueError to be raised
        with pytest.raises(ValueError, match="Session 'non-existent' not found. Create it first using initSession."):
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
    # Configure the mock client
    mock_client = mock_client_class.return_value
    mock_client.session_id = "auto-session-id"
    mock_client.invoke.return_value = {"stream": [{"result": {"content": "Success"}}], "isError": False}

    # Execute code without creating session first
    action = ExecuteCodeAction(
        type="executeCode",
        session_name=None,  # Use default session
        code="print('Auto session')",
        language=LanguageType.PYTHON,
    )

    result = interpreter.execute_code(action)

    assert result["status"] == "success"

    # Verify session was created (will be random UUID, not "default")
    assert len(interpreter._sessions) == 1
    auto_created_session = list(interpreter._sessions.keys())[0]
    assert auto_created_session.startswith("session-")  # Check pattern instead of exact name

    # Verify code was executed
    mock_client.invoke.assert_called_with(
        "executeCode", {"code": "print('Auto session')", "language": "python", "clearContext": False}
    )
