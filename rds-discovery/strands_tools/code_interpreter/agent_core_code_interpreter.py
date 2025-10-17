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
    environments with custom code interpreter identifiers.

    The class maintains session state and provides methods for code execution, file
    operations, and session management. It supports both default AWS code interpreter
    environments and custom environments specified by identifier.

    Examples:
        Basic usage with default identifier:

        >>> interpreter = AgentCoreCodeInterpreter(region="us-west-2")
        >>> # Uses default identifier: "aws.codeinterpreter.v1"

        Using a custom code interpreter identifier:

        >>> custom_id = "my-custom-interpreter-abc123"
        >>> interpreter = AgentCoreCodeInterpreter(
        ...     region="us-west-2",
        ...     identifier=custom_id
        ... )

        Environment-specific usage:

        >>> # For testing environments
        >>> test_interpreter = AgentCoreCodeInterpreter(
        ...     region="us-east-1",
        ...     identifier="test-interpreter-xyz789"
        ... )

        >>> # For production environments
        >>> prod_interpreter = AgentCoreCodeInterpreter(
        ...     region="us-west-2",
        ...     identifier="prod-interpreter-def456"
        ... )

    Attributes:
        region (str): The AWS region where the code interpreter service is hosted.
        identifier (str): The code interpreter identifier being used for sessions.
    """

    def __init__(self, region: Optional[str] = None, identifier: Optional[str] = None) -> None:
        """
        Initialize the Bedrock AgentCore code interpreter.

        Args:
            region (Optional[str]): AWS region for the sandbox service. If not provided,
                the region will be resolved from AWS configuration (environment variables,
                AWS config files, or instance metadata). Defaults to None.
            identifier (Optional[str]): Custom code interpreter identifier to use
                for code execution sessions. This allows you to specify custom code
                interpreter environments instead of the default AWS-provided one.

                Valid formats include:
                - Default identifier: "aws.codeinterpreter.v1" (used when None)
                - Custom identifier: "my-custom-interpreter-abc123"
                - Environment-specific: "test-interpreter-xyz789"

                Note: Use the code interpreter ID, not the full ARN. The AWS service
                expects the identifier portion only (e.g., "my-interpreter-123" rather
                than "arn:aws:bedrock-agentcore:region:account:code-interpreter-custom/my-interpreter-123").

                If not provided, defaults to "aws.codeinterpreter.v1" for backward
                compatibility. Defaults to None.

        Note:
            This constructor maintains full backward compatibility. Existing code
            that doesn't specify the identifier parameter will continue to work
            unchanged with the default AWS code interpreter environment.

        Raises:
            Exception: If there are issues with AWS region resolution or client
                initialization during session creation.
        """
        super().__init__()
        self.region = resolve_region(region)
        self.identifier = identifier or "aws.codeinterpreter.v1"
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

    def execute_code(self, action: ExecuteCodeAction) -> Dict[str, Any]:
        """Execute code in a Bedrock AgentCoresession."""
        if action.session_name not in self._sessions:
            return {"status": "error", "content": [{"text": f"Session '{action.session_name}' not found"}]}

        logger.debug(f"Executing {action.language} code in session '{action.session_name}'")

        # Use the invoke method with proper parameters as shown in the example
        params = {"code": action.code, "language": action.language.value, "clearContext": action.clear_context}
        response = self._sessions[action.session_name].client.invoke("executeCode", params)

        return self._create_tool_result(response)

    def execute_command(self, action: ExecuteCommandAction) -> Dict[str, Any]:
        """Execute a command in a Bedrock AgentCoresession."""
        if action.session_name not in self._sessions:
            return {"status": "error", "content": [{"text": f"Session '{action.session_name}' not found"}]}

        logger.debug(f"Executing command in session '{action.session_name}': {action.command}")

        # Use the invoke method with proper parameters as shown in the example
        params = {"command": action.command}
        response = self._sessions[action.session_name].client.invoke("executeCommand", params)

        return self._create_tool_result(response)

    def read_files(self, action: ReadFilesAction) -> Dict[str, Any]:
        """Read files from a Bedrock AgentCoresession."""
        if action.session_name not in self._sessions:
            return {"status": "error", "content": [{"text": f"Session '{action.session_name}' not found"}]}

        logger.debug(f"Reading files from session '{action.session_name}': {action.paths}")

        # Use the invoke method with proper parameters as shown in the example
        params = {"paths": action.paths}
        response = self._sessions[action.session_name].client.invoke("readFiles", params)

        return self._create_tool_result(response)

    def list_files(self, action: ListFilesAction) -> Dict[str, Any]:
        """List files in a Bedrock AgentCoresession directory."""
        if action.session_name not in self._sessions:
            return {"status": "error", "content": [{"text": f"Session '{action.session_name}' not found"}]}

        logger.debug(f"Listing files in session '{action.session_name}' at path: {action.path}")

        # Use the invoke method with proper parameters as shown in the example
        params = {"path": action.path}
        response = self._sessions[action.session_name].client.invoke("listFiles", params)

        return self._create_tool_result(response)

    def remove_files(self, action: RemoveFilesAction) -> Dict[str, Any]:
        """Remove files from a Bedrock AgentCoresession."""
        if action.session_name not in self._sessions:
            return {"status": "error", "content": [{"text": f"Session '{action.session_name}' not found"}]}

        logger.debug(f"Removing files from session '{action.session_name}': {action.paths}")

        # Use the invoke method with proper parameters as shown in the example
        params = {"paths": action.paths}
        response = self._sessions[action.session_name].client.invoke("removeFiles", params)

        return self._create_tool_result(response)

    def write_files(self, action: WriteFilesAction) -> Dict[str, Any]:
        """Write files to a Bedrock AgentCoresession."""
        if action.session_name not in self._sessions:
            return {"status": "error", "content": [{"text": f"Session '{action.session_name}' not found"}]}

        logger.debug(f"Writing {len(action.content)} files to session '{action.session_name}'")

        # Convert FileContent objects to dictionaries for the API
        content_dicts = [{"path": fc.path, "text": fc.text} for fc in action.content]

        # Use the invoke method with proper parameters as shown in the example
        params = {"content": content_dicts}
        response = self._sessions[action.session_name].client.invoke("writeFiles", params)

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
