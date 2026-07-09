"""
File writing tool for Strands Agent with interactive confirmation.

This module provides a secure file writing capability with rich output formatting,
directory creation, and user confirmation. It's designed to safely write content to
files while providing clear feedback and requiring explicit confirmation for writes
in non-development environments.

Key Features:

1. Interactive Confirmation:
   • User approval required before write operations
   • Syntax-highlighted preview of content to be written
   • Cancellation with custom reason tracking

2. Rich Output Display:
   • Syntax highlighting based on file type
   • Formatted panels for operation information
   • Color-coded status messages
   • Clear success and error indicators

3. Safety Features:
   • Directory creation if parent directories don't exist
   • Development mode toggle (BYPASS_TOOL_CONSENT environment variable)
   • Write operation confirmation dialog
   • Detailed error reporting

4. File Management:
   • Automatic file type detection
   • Proper encoding handling
   • Parent directory creation
   • Character count reporting

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import file_write

agent = Agent(tools=[file_write])

# Write to a file with user confirmation
agent.tool.file_write(
    path="/path/to/file.txt",
    content="Hello World!"
)

# Write to a file with code syntax highlighting
agent.tool.file_write(
    path="/path/to/script.py",
    content="def hello():\n    print('Hello world!')"
)
```

See the file_write function docstring for more details on usage options and parameters.
"""

import os
from os.path import expanduser
from typing import Any, Optional, Union

from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from strands.types.tools import ToolResult, ToolUse

from strands_tools.utils import console_util
from strands_tools.utils.user_input import get_user_input

TOOL_SPEC = {
    "name": "file_write",
    "description": "Write content to a file with proper formatting and validation based on file type",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["path", "content"],
        }
    },
}


def detect_language(file_path: str) -> str:
    """
    Detect syntax language based on file extension.

    Examines the file extension to determine the appropriate syntax highlighting
    language for rich text display.

    Args:
        file_path: Path to the file

    Returns:
        str: Detected language identifier or 'text' if unknown extension
    """
    file_extension = file_path.split(".")[-1] if "." in file_path else ""
    return file_extension if file_extension else "text"


def create_rich_panel(content: str, title: Optional[str] = None, syntax_language: Optional[str] = None) -> Panel:
    """
    Create a Rich panel with optional syntax highlighting.

    Generates a visually appealing panel containing the provided content,
    with optional syntax highlighting based on the specified language.

    Args:
        content: Content to display in panel
        title: Optional panel title
        syntax_language: Optional language for syntax highlighting

    Returns:
        Panel: Rich panel object for console display
    """
    if syntax_language:
        syntax = Syntax(content, syntax_language, theme="monokai", line_numbers=True)
        content_for_panel: Union[Syntax, Text] = syntax
    else:
        content_for_panel = Text(content)

    return Panel(
        content_for_panel,
        title=title,
        border_style="blue",
        box=box.DOUBLE,
        expand=False,
        padding=(1, 1),
    )


def file_write(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Write content to a file with interactive confirmation and rich feedback.

    This tool safely writes the provided content to a specified file path with
    proper formatting, validation, and user confirmation. It displays a preview
    of the content to be written with syntax highlighting based on the file type,
    and requires explicit user confirmation in non-development environments.

    How It Works:
    ------------
    1. Expands the user path to handle tilde (~) in paths
    2. Displays file information and content to be written in formatted panels
    3. In non-development environments, requests user confirmation before writing
    4. Creates any necessary parent directories if they don't exist
    5. Writes the content to the file with proper encoding
    6. Provides rich visual feedback on operation success or failure

    Common Usage Scenarios:
    ---------------------
    - Creating configuration files from templates
    - Saving generated code to files
    - Writing logs or output data to specific locations
    - Creating or updating documentation files
    - Saving user-specific settings or preferences

    Args:
        tool: ToolUse object containing the following input fields:
            - path: The path to the file to write. User paths with tilde (~)
                    are automatically expanded.
            - content: The content to write to the file.
        **kwargs: Additional keyword arguments (not used currently)

    Returns:
        ToolResult containing status and response content in the format:
        {
            "toolUseId": "<tool_use_id>",
            "status": "success|error",
            "content": [{"text": "Response message"}]
        }

    Notes:
        - The BYPASS_TOOL_CONSENT environment variable can be set to "true" to bypass the confirmation step
        - Parent directories are automatically created if they don't exist
        - File content is previewed with syntax highlighting based on file extension
        - User can cancel the write operation and provide a reason for cancellation
        - All operations use rich formatting for clear visual feedback
    """
    console = console_util.create()

    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    path = expanduser(tool_input["path"])
    content = tool_input["content"]

    strands_dev = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

    # Create a panel with file information
    info_panel = Panel(
        Text.assemble(
            ("Path: ", "cyan"),
            (path, "yellow"),
            ("\nSize: ", "cyan"),
            (f"{len(content)} characters", "yellow"),
        ),
        title="[bold blue]File Write Operation",
        border_style="blue",
        box=box.DOUBLE,
        expand=False,
        padding=(1, 1),
    )
    console.print(info_panel)

    if not strands_dev:
        # Detect language and display content with syntax highlighting
        language = detect_language(path)
        content_panel = create_rich_panel(
            content,
            title=f"File Content ({language})",
            syntax_language=language,
        )
        console.print(content_panel)

        # Confirm write operation
        user_input = get_user_input("<red><bold>Do you want to proceed with the file write?</bold> [y/*]</red>")
        if user_input.lower().strip() != "y":
            cancellation_reason = (
                user_input if user_input.strip() != "n" else get_user_input("Please provide a reason for cancellation:")
            )
            error_message = f"File write cancelled by the user. Reason: {cancellation_reason}"
            error_panel = Panel(
                Text(error_message, style="bold blue"),
                title="[bold blue]Operation Cancelled",
                border_style="blue",
                box=box.HEAVY,
                expand=False,
            )
            console.print(error_panel)
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": error_message}],
            }

    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            console.print(
                Panel(
                    Text(f"Created directory: {directory}", style="bold blue"),
                    title="[bold blue]Directory Created",
                    border_style="blue",
                    box=box.DOUBLE,
                    expand=False,
                )
            )

        # Write the file
        with open(path, "w") as file:
            file.write(content)

        success_message = f"File written successfully to {path}"
        success_panel = Panel(
            Text(success_message, style="bold green"),
            title="[bold green]Write Successful",
            border_style="green",
            box=box.DOUBLE,
            expand=False,
        )
        console.print(success_panel)
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": f"File write success: {success_message}"}],
        }
    except Exception as e:
        error_message = f"Error writing file: {str(e)}"
        error_panel = Panel(
            Text(error_message, style="bold red"),
            title="[bold red]Write Failed",
            border_style="red",
            box=box.HEAVY,
            expand=False,
        )
        console.print(error_panel)
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": error_message}],
        }
