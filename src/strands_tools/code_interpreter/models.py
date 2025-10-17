"""
Pydantic models for BedrockAgentCore Code Sandbox Strands tool.

This module contains all the Pydantic models used for type-safe action definitions
with discriminated unions, ensuring required fields are present for each action type.
"""

from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class LanguageType(str, Enum):
    """Supported programming languages for code execution."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"


class FileContent(BaseModel):
    """Represents a file with its path and text content for writing to the sandbox file system. Used when creating or
    updating files during code execution sessions."""

    path: str = Field(description="The file path where content should be written")
    text: str = Field(description="Text content for the file")


# Action-specific Pydantic models using discriminated unions
class InitSessionAction(BaseModel):
    """Create a new isolated code execution environment. Use this when starting a new coding task, data analysis
    project, or when you need a fresh sandbox environment. Each session maintains its own state, variables,
    and file system."""

    type: Literal["initSession"] = Field(description="Initialize a new code interpreter session")
    description: str = Field(description="Required description of what this session will be used for")
    session_name: Optional[str] = Field(
        default=None, description="Session name. If not provided, a default session will be used."
    )


class ListLocalSessionsAction(BaseModel):
    """View all active code interpreter sessions managed by this tool instance. Use this to see what sessions are
    available, check their status, or find the session name you need for other operations."""

    type: Literal["listLocalSessions"] = Field(description="List all local sessions managed by this tool instance")


class ExecuteCodeAction(BaseModel):
    """Execute code in a specific programming language within an existing session. Use this for running Python
    scripts, JavaScript/TypeScript code, data analysis, calculations, or any programming task. The session maintains
    state between executions."""

    type: Literal["executeCode"] = Field(description="Execute code in the code interpreter")

    session_name: Optional[str] = Field(
        default=None,
        description="Session name. If not provided, uses the default session which will be auto-created if needed.",
    )

    code: str = Field(description="Required code to execute")
    language: LanguageType = Field(default=LanguageType.PYTHON, description="Programming language for code execution")
    clear_context: bool = Field(default=False, description="Whether to clear the execution context before running code")


class ExecuteCommandAction(BaseModel):
    """Execute shell/terminal commands within the sandbox environment. Use this for system operations like installing
    packages, running scripts, file management, or any command-line tasks that need to be performed in the session."""

    type: Literal["executeCommand"] = Field(description="Execute a shell command in the code interpreter")

    session_name: Optional[str] = Field(
        default=None, description="Session name. If not provided, uses the default session."
    )

    command: str = Field(description="Required shell command to execute")


class ReadFilesAction(BaseModel):
    """Read the contents of one or more files from the sandbox file system. Use this to examine data files,
    configuration files, code files, or any other files that have been created or uploaded to the session."""

    type: Literal["readFiles"] = Field(description="Read files from the code interpreter")

    session_name: Optional[str] = Field(
        default=None, description="Session name. If not provided, uses the default session."
    )

    paths: List[str] = Field(description="List of file paths to read")


class ListFilesAction(BaseModel):
    """Browse and list files and directories within the sandbox file system. Use this to explore the directory
    structure, find files, or understand what's available in the session before reading or manipulating files."""

    type: Literal["listFiles"] = Field(description="List files in a directory")

    session_name: Optional[str] = Field(
        default=None, description="Session name. If not provided, uses the default session."
    )

    path: str = Field(default=".", description="Directory path to list (defaults to current directory)")


class RemoveFilesAction(BaseModel):
    """Delete one or more files from the sandbox file system. Use this to clean up temporary files, remove outdated
    data, or manage storage space within the session. Be careful as this permanently removes files."""

    type: Literal["removeFiles"] = Field(description="Remove files from the code interpreter")

    session_name: Optional[str] = Field(
        default=None, description="Session name. If not provided, uses the default session."
    )

    paths: List[str] = Field(description="Required list of file paths to remove")


class WriteFilesAction(BaseModel):
    """Create or update multiple files in the sandbox file system with specified content. Use this to save data,
    create configuration files, write code files, or store any text-based content that your code execution will need."""

    type: Literal["writeFiles"] = Field(description="Write files to the code interpreter")

    session_name: Optional[str] = Field(
        default=None, description="Session name. If not provided, uses the default session."
    )

    content: List[FileContent] = Field(description="Required list of file content to write")


class CodeInterpreterInput(BaseModel):
    action: Union[
        InitSessionAction,
        ListLocalSessionsAction,
        ExecuteCodeAction,
        ExecuteCommandAction,
        ReadFilesAction,
        ListFilesAction,
        RemoveFilesAction,
        WriteFilesAction,
    ] = Field(discriminator="type")
