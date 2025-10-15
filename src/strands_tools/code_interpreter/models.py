"""
Pydantic models for Code Interpreter
"""

from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class LanguageType(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"


class FileContent(BaseModel):
    """File content for writing to sandbox."""

    path: str = Field(description="File path")
    text: str = Field(description="File content")


class InitSessionAction(BaseModel):
    """Create a new session."""

    type: Literal["initSession"] = Field(description="Initialize session")
    description: str = Field(description="Session purpose")
    session_name: str = Field(description="Session name")


class ListLocalSessionsAction(BaseModel):
    """List all sessions."""

    type: Literal["listLocalSessions"] = Field(description="List sessions")


class ExecuteCodeAction(BaseModel):
    """Execute code in a specific programming language within an existing session."""

    type: Literal["executeCode"] = Field(description="Execute code")

    session_name: Optional[str] = Field(
        default=None,
        description="Session name. If not provided, uses the " "default session which will be auto-created.",
    )

    code: str = Field(description="Code to execute")
    language: LanguageType = Field(default=LanguageType.PYTHON, description="Programming language for code execution")
    clear_context: bool = Field(default=False, description="Whether to clear the execution context before running code")


class ExecuteCommandAction(BaseModel):
    """Execute shell command within the sandbox environment."""

    type: Literal["executeCommand"] = Field(description="Execute a shell command")

    session_name: Optional[str] = Field(
        default=None, description="Session name. If not provided, uses the default session."
    )

    command: str = Field(description="Shell command to execute")


class ReadFilesAction(BaseModel):
    """Read the contents of one or more files from the sandbox file system."""

    type: Literal["readFiles"] = Field(description="Read files")

    session_name: Optional[str] = Field(
        default=None, description="Session name. If not provided, uses the default session."
    )

    paths: List[str] = Field(description="List of file paths to read")


class ListFilesAction(BaseModel):
    """Browse and list files and directories within the sandbox file system."""

    type: Literal["listFiles"] = Field(description="List files in a directory")

    session_name: Optional[str] = Field(
        default=None, description="Session name. If not provided, uses the default session."
    )

    path: str = Field(default=".", description="Directory path to list (defaults to current directory)")


class RemoveFilesAction(BaseModel):
    """Delete one or more files from the sandbox file system."""

    type: Literal["removeFiles"] = Field(description="Remove files")

    session_name: Optional[str] = Field(
        default=None, description="Session name. If not provided, uses the default session."
    )

    paths: List[str] = Field(description="List of file paths to remove")


class WriteFilesAction(BaseModel):
    """Create or update multiple files in the sandbox file system with specified content."""

    type: Literal["writeFiles"] = Field(description="Write files")

    session_name: Optional[str] = Field(
        default=None, description="Session name. If not provided, uses the default session."
    )

    content: List[FileContent] = Field(description="List of file content to write")


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
