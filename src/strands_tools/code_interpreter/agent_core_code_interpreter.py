import logging
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
    """
    Bedrock AgentCore implementation of the CodeInterpreter.

    This class provides a code interpreter interface using AWS Bedrock AgentCore services.
    It supports executing Python, JavaScript, and TypeScript code in isolated sandbox
    environments with automatic session management.

    The class maintains session state and provides methods for code execution, file
    operations, and session management. It supports both default AWS code interpreter
    environments and custom environments specified by identifier.

    Args:
        region (Optional[str]): AWS region for the sandbox service. If not provided,
            the region will be resolved from AWS configuration.
        identifier (Optional[str]): Custom code interpreter identifier to use
            for code execution sessions. If not provided, defaults to the AWS-managed
            identifier "aws.codeinterpreter.v1".
        auto_session (bool): Enable automatic session creation when a session doesn't exist.
            Defaults to True.
        default_session (str): Default session name to use when session_name is not specified
            in operations. Defaults to "default".

    Attributes:
        region (str): The AWS region where the code interpreter service is hosted.
        identifier (str): The code interpreter identifier being used for sessions.
        auto_session (bool): Whether automatic session creation is enabled.
        default_session (str): The default session name.
    """

    def __init__(
        self,
        region: Optional[str] = None,
        identifier: Optional[str] = None,
        auto_session: bool = True,
        default_session: str = "default",
    ) -> None:
        """
        Initialize the Bedrock AgentCore code interpreter.

        Args:
            region (Optional[str]): AWS region for the sandbox service.
            identifier (Optional[str]): Custom code interpreter identifier.
            auto_session (bool): Enable automatic session creation. Defaults to True.
            default_session (str): Default session name. Defaults to "default".
        """
        super().__init__()
        self.region = resolve_region(region)
        self.identifier = identifier or "aws.codeinterpreter.v1"
        self.auto_session = auto_session
        self.default_session = default_session
        self._sessions: Dict[str, SessionInfo] = {}

    def start_platform(self) -> None:
        """Initialize the Bedrock AgentCoreplatform connection."""
        pass

    def cleanup_platform(self) -> None:
        """Clean up Bedrock AgentCoreplatform resources."""
        if not self._started:
            return

        logger.info("Cleaning up Bedrock Agent Core platform resources")

        # Stop all active sessions with better error handling
        for session_name, session in list(self._sessions.items()):
            try:
                session.client.stop()
                logger.debug(f"Stopped session: {session_name}")
            except Exception as e:
                # Handle weak reference errors and other cleanup issues gracefully
                logger.debug(
                    "session=<%s>, exception=<%s> | cleanup skipped (already cleaned up)", session_name, str(e)
                )

        self._sessions.clear()
        logger.info("Bedrock AgentCoreplatform cleanup completed")

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

            # Start the session with custom identifier
            client.start(identifier=self.identifier)

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
        Ensure a session exists, creating it automatically if needed.

        This method checks if the specified session exists. If auto_session is enabled
        and the session doesn't exist, it will be created automatically.

        Args:
            session_name (Optional[str]): The session name to ensure exists. If None,
                uses the default_session.

        Returns:
            tuple[str, Optional[Dict[str, Any]]]: A tuple containing:
                - The session name (either provided or default)
                - An error dictionary if session creation failed, None otherwise
        """
        if not session_name:
            session_name = self.default_session

        if session_name in self._sessions:
            return session_name, None

        if self.auto_session:
            logger.info(f"Auto-creating session: {session_name}")
            init_action = InitSessionAction(
                type="initSession",
                session_name=session_name,
                description=f"Auto-initialized session for {session_name}",
            )
            result = self.init_session(init_action)

            if result.get("status") != "success":
                return session_name, result

            logger.info(f"Successfully auto-created session: {session_name}")
            return session_name, None

        return session_name, {"status": "error", "content": [{"text": f"Session '{session_name}' not found"}]}

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
        """ """
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
