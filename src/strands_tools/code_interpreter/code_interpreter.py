"""
Code Interpreter Tool implementation using Strands @tool decorator.

This module contains the base tool class that provides lifecycle management
and can be extended by specific platform implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from strands import tool

from .models import (
    CodeInterpreterInput,
    ExecuteCodeAction,
    ExecuteCommandAction,
    InitSessionAction,
    LanguageType,
    ListFilesAction,
    ListLocalSessionsAction,
    ReadFilesAction,
    RemoveFilesAction,
    WriteFilesAction,
)

logger = logging.getLogger(__name__)


class CodeInterpreter(ABC):
    def __init__(self):
        self._started = False

        # Existing description (keep all of it)
        description_template = """
        Code Interpreter tool for executing code in isolated sandbox environments.

        This tool provides a comprehensive code execution platform that supports multiple programming
        languages with persistent session management, file operations, and shell command execution. 
        Built on the Bedrock AgentCore Code Sandbox platform, it offers secure, isolated environments 
        for code execution with full lifecycle management.

        Key Features:
        1. Multi-Language Support:
           The tool supports the following programming languages: {supported_languages_list}
           • Full standard library access for each supported language
           • Runtime environment appropriate for each language
           • Shell command execution for system operations

        2. Session Management:
           • Create named, persistent sessions for stateful code execution
           • List and manage multiple concurrent sessions
           • Automatic session cleanup and resource management
           • Session isolation for security and resource separation

        3. File System Operations:
           • Read files from the sandbox environment
           • Write multiple files with custom content
           • List directory contents and navigate file structures
           • Remove files and manage sandbox storage

        4. Advanced Execution Features:
           • Context preservation across code executions within sessions
           • Optional context clearing for fresh execution environments
           • Real-time output capture and error handling
           • Support for long-running processes and interactive code

        How It Works:
        ------------
        1. The tool accepts structured action inputs defining the operation type
        2. Sessions are created on-demand with isolated sandbox environments
        3. Code is executed within the Bedrock AgentCore platform with full runtime support
        4. Results, outputs, and errors are captured and returned in structured format
        5. File operations interact directly with the sandbox file system
        6. Platform lifecycle is managed automatically with cleanup on completion

        Operation Types:
        --------------
        - initSession: Create a new isolated code execution session
        - listLocalSessions: View all active sessions and their status
        - executeCode: Run code in a specified programming language
        - executeCommand: Execute shell commands in the sandbox
        - readFiles: Read file contents from the sandbox file system
        - writeFiles: Create or update files in the sandbox
        - listFiles: Browse directory contents and file structures
        - removeFiles: Delete files from the sandbox environment

        Common Usage Scenarios:
        ---------------------
        - Data analysis: Execute Python scripts for data processing and visualization
        - Web development: Run JavaScript/TypeScript for frontend/backend development
        - System administration: Execute shell commands for environment setup
        - File processing: Read, transform, and write files programmatically
        - Educational coding: Provide safe environments for learning and experimentation
        - CI/CD workflows: Execute build scripts and deployment commands
        - API testing: Run code to test external services and APIs

        Usage with Strands Agent:
```python
        from strands import Agent
        from strands_tools.code_interpreter import AgentCoreCodeInterpreter

        # Create the code interpreter tool
        bedrock_agent_core_code_interpreter = AgentCoreCodeInterpreter(region="us-west-2")
        agent = Agent(tools=[bedrock_agent_core_code_interpreter.code_interpreter])

        # Create a session
        agent.tool.code_interpreter(
            code_interpreter_input={{
                "action": {{
                    "type": "initSession",
                    "description": "Data analysis session",
                    "session_name": "analysis-session"
                }}
            }}
        )

        # Execute Python code
        agent.tool.code_interpreter(
            code_interpreter_input={{
                "action": {{
                    "type": "executeCode",
                    "session_name": "analysis-session",
                    "code": "import pandas as pd\\ndf = pd.read_csv('data.csv')\\nprint(df.head())",
                    "language": "python"
                }}
            }}
        )

        # Write files to the sandbox
        agent.tool.code_interpreter(
            code_interpreter_input={{
                "action": {{
                    "type": "writeFiles",
                    "session_name": "analysis-session",
                    "content": [
                        {{"path": "config.json", "text": '{{"debug": true}}'}},
                        {{"path": "script.py", "text": "print('Hello, World!')"}}
                    ]
                }}
            }}
        )

        # Execute shell commands
        agent.tool.code_interpreter(
            code_interpreter_input={{
                "action": {{
                    "type": "executeCommand",
                    "session_name": "analysis-session",
                    "command": "ls -la && python script.py"
                }}
            }}
        )
```

        Args:
            code_interpreter_input: Structured input containing the action to perform.
                Must be a CodeInterpreterInput object with an 'action' field specifying
                the operation type and required parameters.

                Action Types and Required Fields:
                - InitSessionAction: type="initSession", description (required), session_name (optional)
                - ExecuteCodeAction: type="executeCode", session_name, code, language, clear_context (optional)
                  * language must be one of: {{supported_languages_enum}}
                - ExecuteCommandAction: type="executeCommand", session_name, command
                - ReadFilesAction: type="readFiles", session_name, paths (list)
                - WriteFilesAction: type="writeFiles", session_name, content (list of FileContent objects)
                - ListFilesAction: type="listFiles", session_name, path
                - RemoveFilesAction: type="removeFiles", session_name, paths (list)
                - ListLocalSessionsAction: type="listLocalSessions"

        Returns:
            Dict containing execution results in the format:
            {{
                "status": "success|error",
                "content": [{{"text": "...", "json": {{...}}}}]
            }}

            Success responses include:
            - Session information for session operations
            - Code execution output and results
            - File contents for read operations
            - Operation confirmations for write/delete operations

            Error responses include:
            - Session not found errors
            - Code compilation/execution errors
            - File system operation errors
            - Platform connectivity issues
        """

        auto_session_note = """**session_name is now optional in most operations**

        Sessions are automatically created when needed. You can now:
        • Omit session_name to use an auto-generated session (recommended for simple use cases)
        • Provide session_name for named sessions (useful for multi-step workflows)
        
        Quick example without session_name:
```python
        # No need to call initSession first
        agent.tool.code_interpreter(
            code_interpreter_input={{
                "action": {{
                    "type": "executeCode",
                    "code": "print('Hello, World!')",
                    "language": "python"
                }}
            }}
        )
```
        
        ---
        
        """

        # Set description: note + existing content
        self.code_interpreter.tool_spec["description"] = auto_session_note + description_template.format(
            supported_languages_list=", ".join([f"{lang.name}" for lang in self.get_supported_languages()])
        )

    @tool
    def code_interpreter(self, code_interpreter_input: CodeInterpreterInput) -> Dict[str, Any]:
        """Execute code in isolated sandbox environments."""

        if not self._started:
            self._start()

        if isinstance(code_interpreter_input, dict):
            logger.debug("Mapping dict to CodeInterpreterInput")
            action = CodeInterpreterInput.model_validate(code_interpreter_input).action
        else:
            action = code_interpreter_input.action

        logger.debug(f"Processing action: {type(action).__name__}")

        # Delegate to implementations
        if isinstance(action, InitSessionAction):
            return self.init_session(action)
        elif isinstance(action, ListLocalSessionsAction):
            return self.list_local_sessions()
        elif isinstance(action, ExecuteCodeAction):
            return self.execute_code(action)
        elif isinstance(action, ExecuteCommandAction):
            return self.execute_command(action)
        elif isinstance(action, ReadFilesAction):
            return self.read_files(action)
        elif isinstance(action, ListFilesAction):
            return self.list_files(action)
        elif isinstance(action, RemoveFilesAction):
            return self.remove_files(action)
        elif isinstance(action, WriteFilesAction):
            return self.write_files(action)
        else:
            return {"status": "error", "content": [{"text": f"Unknown action: {type(action)}"}]}

    def _start(self) -> None:
        """Start the platform."""
        if not self._started:
            self.start_platform()
            self._started = True
            logger.debug("Code Interpreter started")

    def _cleanup(self) -> None:
        """Clean up platform resources."""
        if self._started:
            self.cleanup_platform()
            self._started = False
            logger.debug("Code Interpreter cleaned up")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            if self._started:
                logger.debug("Platform cleanup completed successfully")
                self._cleanup()
        except Exception as e:
            logger.debug("exception=<%s> | platform cleanup during destruction skipped", str(e))

    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def start_platform(self) -> None:
        """Initialize the platform connection and resources."""
        ...

    @abstractmethod
    def cleanup_platform(self) -> None:
        """Clean up platform resources and connections."""
        ...

    @abstractmethod
    def init_session(self, action: InitSessionAction) -> Dict[str, Any]:
        """Initialize a new sandbox session."""
        ...

    @abstractmethod
    def execute_code(self, action: ExecuteCodeAction) -> Dict[str, Any]:
        """Execute code in a sandbox session."""
        ...

    @abstractmethod
    def execute_command(self, action: ExecuteCommandAction) -> Dict[str, Any]:
        """Execute a shell command in a sandbox session."""
        ...

    @abstractmethod
    def read_files(self, action: ReadFilesAction) -> Dict[str, Any]:
        """Read files from a sandbox session."""
        ...

    @abstractmethod
    def list_files(self, action: ListFilesAction) -> Dict[str, Any]:
        """List files in a session directory."""
        ...

    @abstractmethod
    def remove_files(self, action: RemoveFilesAction) -> Dict[str, Any]:
        """Remove files from a sandbox session."""
        ...

    @abstractmethod
    def write_files(self, action: WriteFilesAction) -> Dict[str, Any]:
        """Write files to a sandbox session."""
        ...

    @abstractmethod
    def list_local_sessions(self) -> Dict[str, Any]:
        """List all sessions created by this platform instance."""
        ...

    @abstractmethod
    def get_supported_languages(self) -> List[LanguageType]:
        """list supported languages"""
        ...
