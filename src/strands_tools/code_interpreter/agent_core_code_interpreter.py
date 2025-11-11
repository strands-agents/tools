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

    session_id: str
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
        verify_session_before_use: bool = True,
    ) -> None:
        """
        Initialize the Bedrock AgentCore code interpreter.

        Args:
            region: AWS region for the sandbox service.
            identifier: Custom code interpreter identifier.
            session_name: Session name or strategy:
                - None (default): Generate random session ID per instance
                - "runtime": Use AgentCore runtime session_id (set at invocation time)
                - Specific string: Use this exact session name
            auto_create: Automatically create sessions if they don't exist.
                - True (default): Create sessions on-demand
                - False: Fail if session doesn't exist (strict mode)
            persist_sessions: If True, don't cleanup sessions on object destruction.
            verify_session_before_use: If True, verify session still exists before reusing.

        Examples:
            # Case 1: Random session per instance (default)
            interpreter = AgentCoreCodeInterpreter()

            # Case 2: Bind to runtime session (recommended for production)
            session_id = getattr(context, 'session_id', None)
            interpreter = AgentCoreCodeInterpreter(session_name=session_id)

            # Case 3: Named session with auto-create
            interpreter = AgentCoreCodeInterpreter(session_name="my-analysis")

            # Case 4: Strict mode - must pre-initialize
            interpreter = AgentCoreCodeInterpreter(
                session_name="must-exist",
                auto_create=False
            )
        """
        super().__init__()
        self.region = resolve_region(region)
        self.identifier = identifier or "aws.codeinterpreter.v1"
        self.auto_create = auto_create
        self.persist_sessions = persist_sessions
        self.verify_session_before_use = verify_session_before_use

        # Generate session name strategy
        if session_name is None:
            self.default_session = f"session-{uuid.uuid4().hex[:12]}"
        else:
            # Clean session name to meet validation: [0-9a-zA-Z]{1,40}
            self.default_session = session_name.replace("-", "").replace("_", "")[:40]

        self._sessions: Dict[str, SessionInfo] = {}
        self._session_name_to_ci_id: Dict[str, str] = {}

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

        # Only cleanup if configured to do so
        if not self.persist_sessions:
            logger.info("Cleaning up Bedrock Agent Core platform resources")

            for session_name, session in list(self._sessions.items()):
                try:
                    # Verify session status before stopping
                    if self.verify_session_before_use:
                        try:
                            session_info = session.client.get_session()
                            if session_info["status"] != "READY":
                                logger.debug(f"Session {session_name} already {session_info['status']}")
                                continue
                        except Exception as e:
                            logger.debug(f"Session {session_name} status check failed: {e}")
                            continue

                    session.client.stop()
                    logger.debug(f"Stopped session: {session_name}")
                except Exception as e:
                    logger.debug(f"Session {session_name} cleanup skipped: {e}")

            self._sessions.clear()
            self._session_name_to_ci_id.clear()
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

        logger.info(
            f"Initializing Bedrock AgentCoresandbox session: {action.description} with identifier: {self.identifier}"
        )

        session_name = action.session_name

        # Check if session already exists
        if session_name in self._sessions:
            return {"status": "error", "content": [{"text": f"Session '{session_name}' already exists"}]}

        try:
            # Create new sandbox client
            client = BedrockAgentCoreCodeInterpreterClient(
                region=self.region,
            )

            # Start session with identifier and name
            client.start(identifier=self.identifier, name=session_name)

            # Store session mapping for reconnection
            self._session_name_to_ci_id[session_name] = client.session_id

            # Store session info
            self._sessions[session_name] = SessionInfo(
                session_id=client.session_id, description=action.description, client=client
            )

            logger.info(
                f"Initialized session: {session_name} (ID: {client.session_id}) with identifier: {self.identifier}"
            )

            response = {
                "status": "success",
                "content": [
                    {
                        "json": {
                            "sessionName": session_name,
                            "description": action.description,
                            "sessionId": client.session_id,
                        }
                    }
                ],
            }

            return self._create_tool_result(response)

        except Exception as e:
            logger.error(
                f"Failed to initialize session '{session_name}' with identifier: {self.identifier}. Error: {str(e)}"
            )
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
        # Determine which session to use
        target_session = session_name if session_name else self.default_session

        # Check local cache and verify if enabled
        if target_session in self._sessions:
            if self.verify_session_before_use:
                try:
                    session_info = self._sessions[target_session].client.get_session()
                    if session_info["status"] == "READY":
                        logger.debug(f"Using cached session (verified): {target_session}")
                        return target_session, None
                    else:
                        logger.warning(
                            f"Cached session {target_session} is {session_info['status']}, " "will create new session"
                        )
                        del self._sessions[target_session]
                except Exception as e:
                    logger.warning(f"Session verification failed for {target_session}: {e}")
                    del self._sessions[target_session]
            else:
                logger.debug(f"Using cached session (unverified): {target_session}")
                return target_session, None

        # Attempt to reconnect to existing session in AgentCore
        if self.persist_sessions and self.verify_session_before_use:
            try:
                logger.debug(f"Attempting to reconnect to session: {target_session}")

                # Create a new client
                client = BedrockAgentCoreCodeInterpreterClient(region=self.region)

                # List sessions and find by name
                try:
                    sessions_response = client.list_sessions(
                        interpreter_id=self.identifier, status="READY", max_results=100
                    )

                    logger.debug(f"Found {len(sessions_response.get('items', []))} READY sessions")

                    # Look for session with matching name
                    for session_item in sessions_response.get("items", []):
                        if session_item.get("name") == target_session:
                            # Found matching session
                            ci_session_id = session_item["sessionId"]

                            logger.info(f"Found existing session: {target_session} " f"(CI ID: {ci_session_id})")

                            # Attach to existing session
                            client.identifier = self.identifier
                            client.session_id = ci_session_id

                            self._sessions[target_session] = SessionInfo(
                                session_id=ci_session_id, description="Reconnected to persisted session", client=client
                            )
                            self._session_name_to_ci_id[target_session] = ci_session_id

                            logger.info(f"Reconnected to existing session: {target_session}")
                            return target_session, None

                    logger.debug(f"No existing session found with name: {target_session}")

                except Exception as e:
                    logger.debug(f"Session listing failed: {e}")

            except Exception as e:
                logger.debug(f"Reconnection attempt failed: {e}")

        # Session doesn't exist - create if auto_create enabled
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
        return target_session, {
            "status": "error",
            "content": [{"text": f"Session '{target_session}' not found. Create it first using initSession."}],
        }

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

        logger.debug(f"Executing command in session '{session_name}': {action.command}")

        params = {"command": action.command}
        response = self._sessions[session_name].client.invoke("executeCommand", params)

        return self._create_tool_result(response)

    def read_files(self, action: ReadFilesAction) -> Dict[str, Any]:
        """Read files from a Bedrock AgentCore session with automatic session management."""
        session_name, error = self._ensure_session(action.session_name)
        if error:
            return error

        logger.debug(f"Reading files from session '{session_name}': {action.paths}")

        params = {"paths": action.paths}
        response = self._sessions[session_name].client.invoke("readFiles", params)

        return self._create_tool_result(response)

    def list_files(self, action: ListFilesAction) -> Dict[str, Any]:
        """List files in a Bedrock AgentCore session directory with automatic session management."""
        session_name, error = self._ensure_session(action.session_name)
        if error:
            return error

        logger.debug(f"Listing files in session '{session_name}' at path: {action.path}")

        params = {"path": action.path}
        response = self._sessions[session_name].client.invoke("listFiles", params)

        return self._create_tool_result(response)

    def remove_files(self, action: RemoveFilesAction) -> Dict[str, Any]:
        """Remove files from a Bedrock AgentCore session with automatic session management."""
        session_name, error = self._ensure_session(action.session_name)
        if error:
            return error

        logger.debug(f"Removing files from session '{session_name}': {action.paths}")

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
        """Create tool result from code interpreter response."""
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
