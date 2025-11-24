import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter as BedrockAgentCoreCodeInterpreterClient

from ..utils.aws_util import resolve_region
from .code_interpreter import CodeInterpreter
from .models import (
    ExecuteCodeAction,
    ExecuteCommandAction,
    InitSessionAction,
    LanguageType,
    ListFilesAction,
    ReadFilesAction,
    RemoveFilesAction,
    WriteFilesAction,
)

logger = logging.getLogger(__name__)

# Module-level session cache - persists across object instances
_session_mapping: Dict[str, str] = {}  # user_session_name -> aws_session_id


@dataclass
class SessionInfo:
    """
    Information about a code interpreter session.

    This dataclass stores the essential information for managing active code
    interpreter sessions, including the session identifier, description, and
    the underlying Bedrock client instance.

    Attributes:
        session_id (str): Unique identifier for the session assigned by AWS Bedrock.
        description (str): Human-readable description of the session purpose.
        client (BedrockAgentCoreCodeInterpreterClient): The underlying Bedrock client
            instance used for code execution and file operations in this session.
    """

    session_id: str  # AWS CI session ID
    description: str
    client: BedrockAgentCoreCodeInterpreterClient


class AgentCoreCodeInterpreter(CodeInterpreter):
    def __init__(
        self,
        region: Optional[str] = None,
        identifier: Optional[str] = None,
        session_name: Optional[str] = None,
        auto_create: bool = True,
        persist_sessions: bool = True,
    ) -> None:
        """
        Initialize the Bedrock AgentCore code interpreter with session persistence support.

        This integration enables code execution in AWS Bedrock AgentCore sandboxed environments
        with automatic session management and cross-invocation persistence. Sessions are tracked
        via module-level cache, allowing new object instances to reconnect to existing sessions
        without recreation overhead.

        Architecture:
            - Module-level cache stores user_session_name → aws_session_id mappings
            - New instances check cache and reconnect via get_session(aws_session_id)
            - persist_sessions=True (default) prevents cleanup on object destruction
            - Sessions survive across invocations in long-running AgentCore runtimes

        Args:
            region (Optional[str]): AWS region for the code interpreter service. If None,
                resolves from environment or defaults to configured region. Example: "us-west-2"

            identifier (Optional[str]): Custom code interpreter identifier for the AWS service.
                Defaults to "aws.codeinterpreter.v1". This must match the interpreter type
                configured in your AWS account.

            session_name (Optional[str]): Session identifier for tracking and reconnection.
                - None (default): Generates random session ID per instance (e.g., "session-a1b2c3d4e5f6")
                - String value: Uses provided name

                Recommended: Pass context.session_id from AgentCore for automatic persistence:
                    session_id = getattr(context, 'session_id', 'default')
                    interpreter = AgentCoreCodeInterpreter(session_name=session_id)

            auto_create (bool): Automatically create sessions if they don't exist. Default: True
                - True: Calls init_session() automatically when session not found
                - False: Raises ValueError if session doesn't exist (strict mode)

                Use False when you want explicit control over session lifecycle or when
                pre-initializing sessions with specific configurations.

            persist_sessions (bool): Prevent session cleanup on object destruction. Default: True
                - True: Sessions survive object destruction (recommended for AgentCore)
                - False: Sessions cleaned up in __del__() (use for short-lived scripts)

                In AgentCore's long-running runtime, new object instances are created per
                invocation but the Python process persists. Setting this to True allows
                sessions to survive across invocations and be reconnected by subsequent
                instances via module-level cache.

        Session Lifecycle:
            Invocation 1 (Instance #1):
                1. Create new instance with session_name="user-abc-123"
                2. Session not found → auto_create=True → init_session()
                3. AWS returns session_id="01K9QWSZFRC2..." (random ULID)
                4. Store in module cache: {"userabc123": "01K9QWSZFRC2..."}
                5. Execute code successfully
                6. Object destroyed → persist_sessions=True → skip cleanup
                7. AWS session remains READY

            Invocation 2 (Instance #2, same session_name):
                1. Create new instance (new object, empty self._sessions)
                2. Check module cache → found "userabc123": "01K9QWSZFRC2..."
                3. Call get_session("01K9QWSZFRC2...") → status: READY
                4. Reconnect to existing session (no recreation)
                5. Execute code in same session (variables/state preserved)

        Performance:
            - Session creation: ~800ms (first invocation only)
            - Session reconnection: ~50-100ms (subsequent invocations)
            - Performance improvement: 30-70% on invocations 2+

        Thread Safety:
            Module-level cache access is not thread-safe. If using in multi-threaded
            environments, ensure session names are unique per thread or add external
            synchronization.

        Examples:
            # Production usage with AgentCore context (recommended):
            @app.entrypoint
            def invoke(payload, context):
                session_id = getattr(context, 'session_id', 'default')
                interpreter = AgentCoreCodeInterpreter(
                    region="us-west-2",
                    session_name=session_id,
                    auto_create=True,
                    persist_sessions=True  # Default, but explicit is good
                )
                # Sessions automatically persist and reconnect

            # Simple script usage (auto-generated session):
            interpreter = AgentCoreCodeInterpreter(region="us-west-2")
            # Uses random session name, auto-creates, persists by default

            # Strict mode (must pre-initialize):
            interpreter = AgentCoreCodeInterpreter(
                session_name="my-session",
                auto_create=False
            )
            # Raises ValueError if "my-session" doesn't exist

            # Short-lived script (cleanup sessions):
            interpreter = AgentCoreCodeInterpreter(
                session_name="temp-session",
                persist_sessions=False
            )
            # Session cleaned up when object destroyed

        Notes:
            - Module-level cache persists in long-running Python processes (AgentCore)
            - Cache does NOT persist across container restarts (cold starts)
            - Session names must be unique per user/conversation for isolation
            - AWS session IDs are globally unique (ULID format)
            - Sessions can be manually stopped via AWS console/API if needed

        Raises:
            ValueError: If auto_create=False and session doesn't exist, or if
                       session_name is already in use by another instance
            Exception: If AWS service errors occur during session operations

        See Also:
            - init_session(): Explicitly create a new session
            - execute_code(): Run Python/JavaScript/TypeScript code
            - list_local_sessions(): View sessions created by this instance
        """
        super().__init__()
        self.region = resolve_region(region)
        self.identifier = identifier or "aws.codeinterpreter.v1"
        self.auto_create = auto_create
        self.persist_sessions = persist_sessions

        if session_name is None:
            self.default_session = f"session-{uuid.uuid4().hex[:12]}"
        else:
            self.default_session = session_name

        self._sessions: Dict[str, SessionInfo] = {}

        logger.info(
            f"Initialized CodeInterpreter with session='{self.default_session}', "
            f"identifier='{self.identifier}', auto_create={auto_create}, "
            f"persist_sessions={persist_sessions}"
        )

    def start_platform(self) -> None:
        """Initialize the Bedrock AgentCoreplatform connection."""
        pass

    def cleanup_platform(self) -> None:
        """Clean up Bedrock AgentCoreplatform resources."""
        if not self._started:
            return

        # Only cleanup if persist_sessions=False
        if not self.persist_sessions:
            logger.info("Cleaning up Bedrock Agent Core platform resources")

            for session_name, session in list(self._sessions.items()):
                try:
                    session.client.stop()
                    logger.debug(f"Stopped session: {session_name}")
                except Exception as e:
                    logger.debug(f"Session {session_name} cleanup skipped: {e}")

            self._sessions.clear()
            logger.info("Bedrock AgentCore platform cleanup completed")
        else:
            logger.debug("Skipping cleanup - sessions persist (persist_sessions=True)")

    def init_session(self, action: InitSessionAction) -> Dict[str, Any]:
        """
        Initialize a new Bedrock AgentCore sandbox session.

        Creates a new code interpreter session using the configured identifier.
        The session will use the identifier specified during class initialization,
        or the default "aws.codeinterpreter.v1" if none was provided.

        Args:
            action (InitSessionAction): Action containing session initialization parameters
                including session_name and description.

        Returns:
            Dict[str, Any]: Response dictionary containing session information on success
                or error details on failure. Success response includes sessionName,
                description, and sessionId.

        Raises:
            Exception: If session initialization fails due to AWS service issues,
                invalid identifier, or other configuration problems.
        """
        logger.info(f"Initializing Bedrock AgentCore sandbox session: {action.description}")

        session_name = action.session_name

        # Check if session already exists in instance cache
        if session_name in self._sessions:
            return {"status": "error", "content": [{"text": f"Session '{session_name}' already exists"}]}

        # Check if session name already in use (module-level cache)
        if session_name in _session_mapping:
            error_msg = (
                f"Session '{session_name}' is already in use by another instance. "
                f"Use a unique session name or reconnect to the existing session "
                f"via _ensure_session() instead of calling init_session() directly."
            )
            logger.error(error_msg)
            return {"status": "error", "content": [{"text": error_msg}]}

        try:
            # Create new sandbox client
            client = BedrockAgentCoreCodeInterpreterClient(region=self.region)

            # Start session with identifier and name
            client.start(identifier=self.identifier, name=session_name)

            aws_session_id = client.session_id

            # Store mapping in module-level cache
            _session_mapping[session_name] = aws_session_id

            # Store session info locally
            self._sessions[session_name] = SessionInfo(
                session_id=aws_session_id, description=action.description, client=client
            )

            logger.info(f"Initialized session: {session_name} (AWS ID: {aws_session_id})")

            response = {
                "status": "success",
                "content": [
                    {
                        "json": {
                            "sessionName": session_name,
                            "description": action.description,
                            "sessionId": aws_session_id,
                        }
                    }
                ],
            }

            return self._create_tool_result(response)

        except Exception as e:
            logger.error(f"Failed to initialize session '{session_name}': {str(e)}")
            return {
                "status": "error",
                "content": [{"text": f"Failed to initialize session '{session_name}': {str(e)}"}],
            }

    def list_local_sessions(self) -> Dict[str, Any]:
        """List all sessions created by this Bedrock AgentCoreplatform instance."""
        sessions_info = []
        for name, info in self._sessions.items():
            sessions_info.append(
                {
                    "sessionName": name,
                    "description": info.description,
                    "sessionId": info.session_id,
                }
            )

        return {
            "status": "success",
            "content": [{"json": {"sessions": sessions_info, "totalSessions": len(sessions_info)}}],
        }

    def _ensure_session(self, session_name: Optional[str]) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Ensure a session exists based on configuration.

        Behavior matrix:
        | session_name | auto_create | Behavior |
        |--------------|-------------|----------|
        | None         | True        | Use default_session, create if needed |
        | None         | False       | Use default_session, error if missing |
        | "my-session" | True        | Use "my-session", create if needed |
        | "my-session" | False       | Use "my-session", error if missing |

        Args:
            session_name: Explicit session name from action, or None to use default

        Returns:
            Tuple of (session_name, error_dict or None)
        """
        target_session = session_name if session_name else self.default_session

        logger.debug(f"Ensuring session: {target_session}")

        # Check local cache first
        if target_session in self._sessions:
            logger.debug(f"Using cached session: {target_session}")
            return target_session, None

        # Check module-level cache for AWS session ID
        aws_session_id = _session_mapping.get(target_session)

        if aws_session_id:
            # Found in module cache - try to reconnect
            logger.debug(f"Found session in module cache: {target_session} -> {aws_session_id}")

            try:
                client = BedrockAgentCoreCodeInterpreterClient(region=self.region)

                # Verify session still exists and is ready
                session_info = client.get_session(interpreter_id=self.identifier, session_id=aws_session_id)

                if session_info.get("status") == "READY":
                    # Session is ready - reconnect to it
                    client.identifier = self.identifier
                    client.session_id = aws_session_id

                    self._sessions[target_session] = SessionInfo(
                        session_id=aws_session_id, description="Reconnected via module cache", client=client
                    )

                    logger.info(f"Reconnected to existing session: {target_session}")
                    return target_session, None
                else:
                    # Session exists but not ready - remove from cache
                    logger.warning(f"Session {target_session} not READY, removing from cache")
                    del _session_mapping[target_session]

            except Exception as e:
                # Session doesn't exist or error - remove from cache
                logger.debug(f"Session reconnection failed: {e}")
                if target_session in _session_mapping:
                    del _session_mapping[target_session]

        # Session not found - create new if auto_create enabled
        if self.auto_create:
            logger.info(f"Auto-creating session: {target_session}")

            init_action = InitSessionAction(
                type="initSession", session_name=target_session, description="Auto-initialized session"
            )
            result = self.init_session(init_action)

            if result.get("status") != "success":
                return target_session, result

            logger.info(f"Successfully auto-created session: {target_session}")
            return target_session, None

        # auto_create=False and session doesn't exist
        logger.debug(f"Session '{target_session}' not found (auto_create disabled)")
        raise ValueError(f"Session '{target_session}' not found. Create it first using initSession.")

    def execute_code(self, action: ExecuteCodeAction) -> Dict[str, Any]:
        """Execute code in a Bedrock AgentCore session with automatic session management."""
        session_name, error = self._ensure_session(action.session_name)
        if error:
            return error

        logger.debug(f"Executing {action.language} code in session '{session_name}'")

        params = {"code": action.code, "language": action.language.value, "clearContext": action.clear_context}
        response = self._sessions[session_name].client.invoke("executeCode", params)

        return self._create_tool_result(response)

    def execute_command(self, action: ExecuteCommandAction) -> Dict[str, Any]:
        """Execute a command in a Bedrock AgentCore session with automatic session management."""
        session_name, error = self._ensure_session(action.session_name)
        if error:
            return error

        logger.debug(f"Executing command in session '{session_name}'")

        params = {"command": action.command}
        response = self._sessions[session_name].client.invoke("executeCommand", params)

        return self._create_tool_result(response)

    def read_files(self, action: ReadFilesAction) -> Dict[str, Any]:
        """Read files from a Bedrock AgentCore session with automatic session management."""
        session_name, error = self._ensure_session(action.session_name)
        if error:
            return error

        logger.debug(f"Reading files from session '{session_name}'")

        params = {"paths": action.paths}
        response = self._sessions[session_name].client.invoke("readFiles", params)

        return self._create_tool_result(response)

    def list_files(self, action: ListFilesAction) -> Dict[str, Any]:
        """List files in a Bedrock AgentCore session directory with automatic session management."""
        session_name, error = self._ensure_session(action.session_name)
        if error:
            return error

        logger.debug(f"Listing files in session '{session_name}'")

        params = {"path": action.path}
        response = self._sessions[session_name].client.invoke("listFiles", params)

        return self._create_tool_result(response)

    def remove_files(self, action: RemoveFilesAction) -> Dict[str, Any]:
        """Remove files from a Bedrock AgentCore session with automatic session management."""
        session_name, error = self._ensure_session(action.session_name)
        if error:
            return error

        logger.debug(f"Removing files from session '{session_name}'")

        params = {"paths": action.paths}
        response = self._sessions[session_name].client.invoke("removeFiles", params)

        return self._create_tool_result(response)

    def write_files(self, action: WriteFilesAction) -> Dict[str, Any]:
        """Write files to a Bedrock AgentCore session with automatic session management."""
        session_name, error = self._ensure_session(action.session_name)
        if error:
            return error

        logger.debug(f"Writing {len(action.content)} files to session '{session_name}'")

        content_dicts = [{"path": fc.path, "text": fc.text} for fc in action.content]
        params = {"content": content_dicts}
        response = self._sessions[session_name].client.invoke("writeFiles", params)

        return self._create_tool_result(response)

    def _create_tool_result(self, response) -> Dict[str, Any]:
        """Create tool result from response."""
        if "stream" in response:
            event_stream = response["stream"]
            for event in event_stream:
                if "result" in event:
                    result = event["result"]
                    is_error = response.get("isError", False)
                    return {
                        "status": "success" if not is_error else "error",
                        "content": [{"text": str(result.get("content"))}],
                    }

            return {"status": "error", "content": [{"text": f"Failed to create tool result: {str(response)}"}]}

        return response

    @staticmethod
    def get_supported_languages() -> List[LanguageType]:
        return [LanguageType.PYTHON, LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]
