"""Code Interpreter base class - IMPROVED VERSION"""

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

        self.code_interpreter.tool_spec["description"] = """
Execute code in isolated sandbox environments with automatic session management.

COMMON USE CASE - Execute Code Directly:

To execute code, provide the code to run. Sessions are created and managed automatically:

{{
    "action": {{
        "type": "executeCode",
        "code": "result = 25 * 48\\nprint(f'Result: {{result}}')",
        "language": "python"
    }}
}}

The session_name parameter is optional. If not provided, a default session will be used
and created automatically if it doesn't exist.

Supported Languages: {supported_languages_list}

KEY FEATURES:

1. Automatic Session Management
   - Sessions are created on-demand when executing code
   - No manual initialization required for simple use cases
   - State persists within a session between executions

2. File System Operations
   - Read, write, list, and remove files in the sandbox
   - Files persist within a session

3. Shell Command Execution
   - Execute system commands in the sandbox environment

4. Multiple Programming Languages
   - Python, JavaScript, and TypeScript supported

OPERATION EXAMPLES:

1. Execute Python Code:
   {{
       "action": {{
           "type": "executeCode",
           "code": "import pandas as pd\\ndf = pd.DataFrame({{'a': [1,2,3]}})\\nprint(df)",
           "language": "python"
       }}
   }}

2. Execute Shell Command:
   {{
       "action": {{
           "type": "executeCommand",
           "command": "ls -la"
       }}
   }}

3. Write Files:
   {{
       "action": {{
           "type": "writeFiles",
           "content": [
               {{"path": "data.txt", "text": "Hello World"}},
               {{"path": "config.json", "text": '{{"debug": true}}'}}
           ]
       }}
   }}

4. Read Files:
   {{
       "action": {{
           "type": "readFiles",
           "paths": ["data.txt", "config.json"]
       }}
   }}

5. List Files:
   {{
       "action": {{
           "type": "listFiles",
           "path": "."
       }}
   }}

ADVANCED: Manual Session Management

For organizing multiple execution contexts, you can explicitly create named sessions:

{{
    "action": {{
        "type": "initSession",
        "session_name": "data-analysis",
        "description": "Session for data analysis tasks"
    }}
}}

Then reference the session in subsequent operations:

{{
    "action": {{
        "type": "executeCode",
        "session_name": "data-analysis",
        "code": "print('Using named session')"
    }}
}}

RESPONSE FORMAT:

Success:
{{
    "status": "success",
    "content": [{{"text": "execution output"}}]
}}

Error:
{{
    "status": "error",
    "content": [{{"text": "error description"}}]
}}

Args:
    code_interpreter_input: Structured input containing the action to perform.

Returns:
    Dictionary containing execution results with status and content.
        """.format(
            supported_languages_list=", ".join([f"{lang.name}" for lang in self.get_supported_languages()]),
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
                logger.debug("Cleaning up in destructor")
                self._cleanup()
        except Exception as e:
            logger.debug(f"Cleanup during destruction skipped: {e}")

    # Abstract methods
    @abstractmethod
    def start_platform(self) -> None: ...

    @abstractmethod
    def cleanup_platform(self) -> None: ...

    @abstractmethod
    def init_session(self, action: InitSessionAction) -> Dict[str, Any]: ...

    @abstractmethod
    def execute_code(self, action: ExecuteCodeAction) -> Dict[str, Any]: ...

    @abstractmethod
    def execute_command(self, action: ExecuteCommandAction) -> Dict[str, Any]: ...

    @abstractmethod
    def read_files(self, action: ReadFilesAction) -> Dict[str, Any]: ...

    @abstractmethod
    def list_files(self, action: ListFilesAction) -> Dict[str, Any]: ...

    @abstractmethod
    def remove_files(self, action: RemoveFilesAction) -> Dict[str, Any]: ...

    @abstractmethod
    def write_files(self, action: WriteFilesAction) -> Dict[str, Any]: ...

    @abstractmethod
    def list_local_sessions(self) -> Dict[str, Any]: ...

    @abstractmethod
    def get_supported_languages(self) -> List[LanguageType]: ...
