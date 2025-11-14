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
    """Information about a code interpreter session."""
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
        Initialize the Bedrock AgentCore code interpreter.

        Args:
            region: AWS region for the sandbox service.
            identifier: Custom code interpreter identifier.
            session_name: Session name (will be cleaned to meet AWS validation).
            auto_create: Automatically create sessions if they don't exist.
            persist_sessions: If True, don't cleanup sessions on object destruction.
        """
        super().__init__()
        self.region = resolve_region(region)
        self.identifier = identifier or "aws.codeinterpreter.v1"
        self.auto_create = auto_create
        self.persist_sessions = persist_sessions

        # Clean session name to meet AWS validation: [0-9a-zA-Z]{1,40}
        if session_name is None:
            self.default_session = f"session-{uuid.uuid4().hex[:12]}"
        else:
            self.default_session = session_name.replace('-', '').replace('_', '')[:40]

        self._sessions: Dict[str, SessionInfo] = {}

        logger.info(
            f"Initialized CodeInterpreter with session='{self.default_session}', "
            f"identifier='{self.identifier}', auto_create={auto_create}, "
            f"persist_sessions={persist_sessions}"
        )

    def start_platform(self) -> None:
        """Initialize platform connection."""
        pass

    def cleanup_platform(self) -> None:
        """Clean up platform resources."""
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
        """Initialize a new Bedrock AgentCore sandbox session."""
        logger.info(
            f"Initializing Bedrock AgentCore sandbox session: {action.description}"
        )

        session_name = action.session_name

        # Check if session already exists
        if session_name in self._sessions:
            return {
                "status": "error",
                "content": [{"text": f"Session '{session_name}' already exists"}]
            }

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
                session_id=aws_session_id,
                description=action.description,
                client=client
            )

            logger.info(
                f"Initialized session: {session_name} (AWS ID: {aws_session_id})"
            )

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
        """List all sessions created by this instance."""
        sessions_info = []
        for name, info in self._sessions.items():
            sessions_info.append({
                "sessionName": name,
                "description": info.description,
                "sessionId": info.session_id,
            })

        return {
            "status": "success",
            "content": [{"json": {"sessions": sessions_info, "totalSessions": len(sessions_info)}}],
        }

    def _ensure_session(self, session_name: Optional[str]) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Ensure session exists with module-level cache reconnection.
        
        Uses module-level cache to store session mappings across object instances.
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
                session_info = client.get_session(
                    interpreter_id=self.identifier,
                    session_id=aws_session_id
                )
                
                if session_info.get('status') == 'READY':
                    # Session is ready - reconnect to it
                    client.identifier = self.identifier
                    client.session_id = aws_session_id
                    
                    self._sessions[target_session] = SessionInfo(
                        session_id=aws_session_id,
                        description="Reconnected via module cache",
                        client=client
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
                type="initSession",
                session_name=target_session,
                description="Auto-initialized session"
            )
            result = self.init_session(init_action)

            if result.get("status") != "success":
                return target_session, result

            logger.info(f"Successfully auto-created session: {target_session}")
            return target_session, None

        # auto_create=False and session doesn't exist
        logger.debug(f"Session '{target_session}' not found (auto_create disabled)")
        raise ValueError(
            f"Session '{target_session}' not found. Create it first using initSession."
        )

    def execute_code(self, action: ExecuteCodeAction) -> Dict[str, Any]:
        """Execute code in a Bedrock AgentCore session."""
        session_name, error = self._ensure_session(action.session_name)
        if error:
            return error

        logger.debug(f"Executing {action.language} code in session '{session_name}'")

        params = {
            "code": action.code,
            "language": action.language.value,
            "clearContext": action.clear_context
        }
        response = self._sessions[session_name].client.invoke("executeCode", params)

        return self._create_tool_result(response)

    def execute_command(self, action: ExecuteCommandAction) -> Dict[str, Any]:
        """Execute a command in a Bedrock AgentCore session."""
        session_name, error = self._ensure_session(action.session_name)
        if error:
            return error

        logger.debug(f"Executing command in session '{session_name}'")

        params = {"command": action.command}
        response = self._sessions[session_name].client.invoke("executeCommand", params)

        return self._create_tool_result(response)

    def read_files(self, action: ReadFilesAction) -> Dict[str, Any]:
        """Read files from a Bedrock AgentCore session."""
        session_name, error = self._ensure_session(action.session_name)
        if error:
            return error

        logger.debug(f"Reading files from session '{session_name}'")

        params = {"paths": action.paths}
        response = self._sessions[session_name].client.invoke("readFiles", params)

        return self._create_tool_result(response)

    def list_files(self, action: ListFilesAction) -> Dict[str, Any]:
        """List files in a Bedrock AgentCore session directory."""
        session_name, error = self._ensure_session(action.session_name)
        if error:
            return error

        logger.debug(f"Listing files in session '{session_name}'")

        params = {"path": action.path}
        response = self._sessions[session_name].client.invoke("listFiles", params)

        return self._create_tool_result(response)

    def remove_files(self, action: RemoveFilesAction) -> Dict[str, Any]:
        """Remove files from a Bedrock AgentCore session."""
        session_name, error = self._ensure_session(action.session_name)
        if error:
            return error

        logger.debug(f"Removing files from session '{session_name}'")

        params = {"paths": action.paths}
        response = self._sessions[session_name].client.invoke("removeFiles", params)

        return self._create_tool_result(response)

    def write_files(self, action: WriteFilesAction) -> Dict[str, Any]:
        """Write files to a Bedrock AgentCore session."""
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

            return {
                "status": "error",
                "content": [{"text": f"Failed to create tool result: {str(response)}"}]
            }

        return response

    @staticmethod
    def get_supported_languages() -> List[LanguageType]:
        return [LanguageType.PYTHON, LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]