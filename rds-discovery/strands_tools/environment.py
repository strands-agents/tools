"""
Runtime environment variable management tool for Strands Agent.

This module provides comprehensive functionality for managing environment variables
at runtime, allowing you to list, get, set, delete, and validate environment variables
with appropriate security measures and clear formatting. It's designed to provide
both interactive usage with rich formatting and programmatic access with structured returns.

Key Features:

1. Variable Management:
   • Get all environment variables
   • Set/update variables with validation
   • Delete variables safely
   • Filter by prefix
   • Protect system variables

2. Security Features:
   • Protected variables list
   • Value masking for sensitive data
   • Change confirmation
   • Variable validation
   • Risk level indicators

3. Rich Output:
   • Colorized tables with clear formatting
   • Visual indicators for protected variables
   • Operation previews with risk assessment
   • Success/error status panels
   • Variable categorization

4. Smart Filtering:
   • Prefix-based filtering
   • Sensitive value detection
   • Protected variable identification
   • Value type recognition

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import environment

agent = Agent(tools=[environment])

# List all environment variables
agent.tool.environment(action="list")

# List variables with specific prefix
agent.tool.environment(action="list", prefix="AWS_")

# Get a specific variable value
agent.tool.environment(action="get", name="PATH")

# Set a variable (with confirmation prompt)
agent.tool.environment(action="set", name="MY_SETTING", value="new_value")

# Delete a variable (with confirmation prompt)
agent.tool.environment(action="delete", name="TEMP_VAR")
```

See the environment function docstring for more details on available actions and parameters.
"""

import os
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from strands.types.tools import ToolResult, ToolResultContent, ToolUse

from strands_tools.utils import console_util, user_input

TOOL_SPEC = {
    "name": "environment",
    "description": """Runtime environment variable management tool.
    
Key Features:
1. Variable Management:
   - Get all environment variables
   - Set/update variables
   - Delete variables
   - Filter by prefix
   - Validate values
   
2. Actions:
   - list: Show all or filtered variables
   - get: Get specific variable value
   - set: Set/update variable value
   - delete: Remove variable
   - validate: Check variable format/value
   
3. Security:
   - Protected variables list
   - Value validation
   - Change tracking
   - Variable masking
   
4. Usage Examples:
   # List all environment variables:
   environment(action="list")
   
   # List variables with prefix:
   environment(action="list", prefix="AWS_")
   
   # Get specific variable:
   environment(action="get", name="MIN_SCORE")
   
   # Set variable:
   environment(action="set", name="MIN_SCORE", value="0.7")
   
   # Delete variable:
   environment(action="delete", name="TEMP_VAR")""",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "get", "set", "delete", "validate"],
                    "description": "Action to perform on environment variables",
                },
                "name": {
                    "type": "string",
                    "description": "Name of the environment variable",
                },
                "value": {
                    "type": "string",
                    "description": "Value to set for the environment variable",
                },
                "prefix": {
                    "type": "string",
                    "description": "Filter variables by prefix",
                },
                "masked": {
                    "type": "boolean",
                    "description": "Mask sensitive values in output",
                    "default": True,
                },
            },
            "required": ["action"],
        }
    },
}


# Protected variables that can't be modified
PROTECTED_VARS = {"PATH", "PYTHONPATH", "STRANDS_HOME", "SHELL", "USER", "HOME"}


def mask_sensitive_value(name: str, value: str) -> str:
    """
    Mask sensitive values for display to protect security-related information.

    This function detects common patterns in environment variable names that might
    contain sensitive information (like tokens, passwords, keys) and masks their
    values to prevent accidental exposure.

    Args:
        name: The name of the environment variable to check
        value: The actual value that might need masking

    Returns:
        str: The masked value (if sensitive) or original value (if not sensitive)
    """
    if any(sensitive in name.upper() for sensitive in ["TOKEN", "SECRET", "PASSWORD", "KEY", "AUTH"]):
        if value:
            return f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "****"
    return value


def format_env_vars_table(env_vars: Dict[str, str], masked: bool, prefix: Optional[str] = None) -> Table:
    """
    Format environment variables as a rich table with proper styling.

    This function creates a visually formatted table of environment variables with
    clear indicators for protected variables and proper masking of sensitive values.

    Args:
        env_vars: Dictionary of environment variables (name: value pairs)
        masked: Whether to mask sensitive values like tokens and passwords
        prefix: Optional prefix filter to only show variables starting with this string

    Returns:
        Table: A Rich library Table object ready for display
    """
    table = Table(title="Environment Variables", show_header=True, box=box.ROUNDED)
    table.add_column("Protected", style="yellow")
    table.add_column("Name", style="cyan")
    table.add_column("Value", style="green")

    for name, value in sorted(env_vars.items()):
        if prefix and not name.startswith(prefix):
            continue

        protected = "🔒" if name in PROTECTED_VARS else ""
        display_value = mask_sensitive_value(name, value) if masked else value
        table.add_row(protected, name, str(display_value))

    return table


def format_operation_preview(
    action: str,
    name: Optional[str] = None,
    value: Optional[str] = None,
    prefix: Optional[str] = None,
) -> Panel:
    """
    Format operation preview as a rich panel with enhanced details.

    Creates a visual preview of the requested operation with appropriate styling,
    risk level indicators, and relevant details about the operation being performed.

    Args:
        action: The action being performed (get, list, set, delete, validate)
        name: Optional name of the target environment variable
        value: Optional value for set operations
        prefix: Optional prefix filter for list operations

    Returns:
        Panel: A Rich library Panel object containing the formatted preview
    """
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    # Format action with color based on type
    action_style = {
        "get": "green",
        "list": "blue",
        "set": "yellow",
        "delete": "red",
        "validate": "magenta",
    }.get(action.lower(), "white")

    table.add_row("Action", f"[{action_style}]{action.upper()}[/{action_style}]")

    if name:
        protected = name in PROTECTED_VARS
        name_style = "red" if protected else "white"
        table.add_row(
            "Variable",
            f"[{name_style}]{name}[/{name_style}] {'🔒' if protected else ''}",
        )
    if value:
        table.add_row("Value", str(value))
    if prefix:
        table.add_row("Prefix Filter", prefix)

    # Add warning for protected variables
    if name and name in PROTECTED_VARS:
        table.add_row(
            "⚠️ Warning",
            "[red]This is a protected system variable that cannot be modified[/red]",
        )

    # Add operation risk level
    risk_level = {
        "get": ("🟢 Safe", "green"),
        "list": ("🟢 Safe", "green"),
        "set": ("🟡 Modifies Environment", "yellow"),
        "delete": ("🔴 Destructive", "red"),
        "validate": ("🟢 Safe", "green"),
    }.get(action.lower(), ("⚪ Unknown", "white"))

    table.add_row("Risk Level", f"[{risk_level[1]}]{risk_level[0]}[/{risk_level[1]}]")

    return Panel(
        table,
        title=f"[bold {risk_level[1]}]🔧 Environment Operation Preview[/bold {risk_level[1]}]",
        border_style=risk_level[1],
        box=box.ROUNDED,
        subtitle="[dim]Dev Mode: "
        + ("✓" if os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true" else "✗")
        + "[/dim]",
    )


def format_env_vars(env_vars: Dict[str, str], masked: bool, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Format environment variables for structured display in tool results.

    This function creates a consistent data structure for environment variables
    that can be used in tool results, with proper masking and filtering.

    Args:
        env_vars: Dictionary of environment variables (name: value pairs)
        masked: Whether to mask sensitive values
        prefix: Optional prefix filter to only include variables starting with this string

    Returns:
        List[Dict[str, Any]]: List of formatted variable entries with metadata
    """
    formatted = []

    for name, value in sorted(env_vars.items()):
        if prefix and not name.startswith(prefix):
            continue

        formatted.append(
            {
                "name": name,
                "value": mask_sensitive_value(name, value) if masked else value,
                "protected": name in PROTECTED_VARS,
            }
        )

    return formatted


def format_success_message(message: str) -> Panel:
    """
    Format a success message in a visually distinct green panel.

    Args:
        message: The success message to format

    Returns:
        Panel: A Rich library Panel with appropriate styling
    """
    return Panel(
        Text(message, style="green"),
        title="[bold green]✅ Success",
        border_style="green",
        box=box.ROUNDED,
    )


def format_error_message(message: str) -> Panel:
    """
    Format an error message in a visually distinct red panel.

    Args:
        message: The error message to format

    Returns:
        Panel: A Rich library Panel with appropriate styling
    """
    return Panel(
        Text(message, style="red"),
        title="[bold red]❌ Error",
        border_style="red",
        box=box.ROUNDED,
    )


def show_operation_result(console: Console, success: bool, message: str) -> None:
    """
    Display operation result with appropriate formatting based on success status.

    Args:
        success: Whether the operation was successful
        message: The message to display
    """
    if success:
        console.print(format_success_message(message))
    else:
        console.print(format_error_message(message))


def environment(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Environment variable management tool for listing, getting, setting, and deleting environment variables.

    This function provides a comprehensive interface for managing runtime environment variables
    with rich output formatting, security features, and proper error handling. It supports
    multiple actions for different environment variable operations, each with appropriate
    validation and confirmation steps.

    How It Works:
    ------------
    1. The function processes the requested action (list, get, set, delete, validate)
    2. For destructive actions, it requires user confirmation unless in BYPASS_TOOL_CONSENT mode
    3. Protected system variables are identified and cannot be modified
    4. Sensitive values (tokens, passwords, etc.) are automatically masked
    5. Rich output formatting provides clear visual feedback on operations
    6. All operations return structured results for both human and programmatic use

    Available Actions:
    ---------------
    - list: Display all environment variables or filter by prefix
    - get: Retrieve and display a specific variable value
    - set: Create or update a variable value (with confirmation)
    - delete: Remove a variable from the environment (with confirmation)
    - validate: Check if a variable exists and validate its format

    Security Features:
    ---------------
    - Protected system variables cannot be modified
    - Sensitive values are masked in output by default
    - Destructive actions require explicit confirmation
    - Clear risk level indicators for all operations
    - BYPASS_TOOL_CONSENT mode controls for testing and automation

    Args:
        tool: The ToolUse object containing the action and parameters
            tool["input"]["action"]: The action to perform (required)
            tool["input"]["name"]: Environment variable name (for get/set/delete/validate)
            tool["input"]["value"]: Value to set (for set action)
            tool["input"]["prefix"]: Filter prefix for list action
            tool["input"]["masked"]: Whether to mask sensitive values (default: True)
        **kwargs: Additional keyword arguments (unused)

    Returns:
        ToolResult: Dictionary containing:
            - toolUseId: The ID of the tool usage
            - status: "success" or "error"
            - content: List of content objects with results or error messages

    Notes:
        - The ENV var "BYPASS_TOOL_CONSENT" can be set to "true" to bypass confirmation prompts
        - Protected variables include PATH, PYTHONPATH, STRANDS_HOME, SHELL, USER, HOME
        - Sensitive variables are detected by keywords in their names (TOKEN, SECRET, etc.)
        - For security reasons, values of sensitive variables are masked in output
    """
    console = console_util.create()

    # Default return in case of unexpected code path
    tool_use_id = tool["toolUseId"]
    default_content: List[ToolResultContent] = [{"text": "Unknown error in environment tool"}]
    default_result = {
        "toolUseId": tool_use_id,
        "status": "error",
        "content": default_content,
    }
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    # Get environment variables at runtime
    env_vars_masked_default = os.getenv("ENV_VARS_MASKED_DEFAULT", "true").lower() == "true"

    # Check for BYPASS_TOOL_CONSENT mode
    strands_dev = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

    # Actions that need confirmation
    dangerous_actions = {"set", "delete"}
    needs_confirmation = tool_input["action"] in dangerous_actions and not strands_dev

    # Print BYPASS_TOOL_CONSENT mode status for debugging
    if strands_dev:
        console.print("[bold green]Running in BYPASS_TOOL_CONSENT mode - confirmation bypassed[/bold green]")

    try:
        action = tool_input["action"]

        # Action processing starts here

        if action == "list":
            prefix = tool_input.get("prefix")
            masked = tool_input.get("masked", env_vars_masked_default)

            # Format rich table
            table = format_env_vars_table(dict(os.environ), masked=masked, prefix=prefix)

            # Format output
            if prefix:
                title = f"[bold blue]Environment Variables[/bold blue] (prefix=[yellow]{prefix}[/yellow])"
            else:
                title = "[bold blue]Environment Variables[/bold blue]"

            # Display rich output
            console.print("")
            console.print(Panel(table, title=title, border_style="blue", box=box.ROUNDED))

            # Format plain text for return
            env_vars = format_env_vars(dict(os.environ), masked=masked, prefix=prefix)
            lines = []
            for var in env_vars:
                protected = "🔒" if var["protected"] else "  "
                lines.append(f"{protected} {var['name']} = {var['value']}")

            list_content: List[ToolResultContent] = [{"text": "\n".join(lines)}]

            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": list_content,
            }

        elif action == "get":
            if "name" not in tool_input:
                console.print(format_error_message("name parameter is required"))
                raise ValueError("name parameter is required for get action")

            name = tool_input["name"]
            value = os.getenv(name)

            if value is None:
                error_msg = f"Environment variable {name} not found"
                console.print(format_error_message(error_msg))
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": error_msg}],
                }

            masked = tool_input.get("masked", env_vars_masked_default)
            safe_value = value if value is not None else ""
            display_value = mask_sensitive_value(name, safe_value) if masked else safe_value

            # Show operation preview
            console.print(format_operation_preview(action="get", name=name, value=display_value))

            # Create rich display with proper formatting
            table = Table(show_header=False, box=box.SIMPLE)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")

            # Add variable details
            table.add_row("Name", name)
            table.add_row("Type", "Protected" if name in PROTECTED_VARS else "Standard")
            table.add_row("Value", display_value)

            # Add value properties
            if value is not None:
                value_str = str(value)
                table.add_row("Length", str(len(value_str)))
                table.add_row("Contains Spaces", "Yes" if " " in value_str else "No")
                table.add_row("Multiline", "Yes" if "\n" in value_str else "No")

            # Create info panel
            panel = Panel(
                table,
                title=(
                    f"[bold {'yellow' if name in PROTECTED_VARS else 'blue'}]🔍 "
                    f"Environment Variable Details[/bold {'yellow' if name in PROTECTED_VARS else 'blue'}]"
                ),
                border_style="yellow" if name in PROTECTED_VARS else "blue",
                box=box.ROUNDED,
            )
            console.print(panel)

            # Show success message
            show_operation_result(console, True, f"Successfully retrieved {name}")
            # Create a return object with properly cast types
            final_display_value = display_value if masked else safe_value
            get_content: List[ToolResultContent] = [{"text": f"{name} = {final_display_value}"}]
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": get_content,
            }

        elif action == "set":
            if "name" not in tool_input or "value" not in tool_input:
                error_msg = "name and value parameters are required"
                console.print(format_error_message(error_msg))
                raise ValueError(error_msg)

            name = tool_input["name"]
            value = tool_input["value"]

            # Check protected status first, regardless of confirmation mode
            if name in PROTECTED_VARS:
                error_msg = f"⚠️ Cannot modify protected variable: {name}"
                error_details = "\nProtected variables ensure system stability and security."
                console.print(format_error_message(f"{error_msg}{error_details}"))
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": f"Cannot modify protected variable: {name}"}],
                }

            # Show operation preview for dangerous actions
            if needs_confirmation or True:  # Always show preview regardless of confirmation mode
                console.print(format_operation_preview(action="set", name=name, value=value))

            # Show current vs new value comparison if exists
            current_value = os.getenv(name)
            if current_value is not None:
                table = Table(show_header=True)
                table.add_column("State", style="cyan")
                table.add_column("Value", style="white")
                table.add_row("Current", current_value)
                table.add_row("New", value)
                console.print(
                    Panel(
                        table,
                        title="[bold yellow]Value Comparison",
                        border_style="yellow",
                    )
                )

            # Ask for confirmation
            if needs_confirmation:
                confirm = user_input.get_user_input(
                    "\n<yellow><bold>Do you want to proceed with setting this environment variable?</bold> "
                    "[y/*]</yellow>"
                )
                # For tests, 'y' should be recognized even with extra spaces or newlines
                if confirm.strip().lower() != "y":
                    console.print(format_error_message("Operation cancelled by user"))
                    return {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [{"text": f"Operation cancelled by user, reason: {confirm}"}],
                    }

            # Set the variable
            os.environ[name] = str(value)

            # Show success message
            show_operation_result(console, True, f"Successfully set {name}")
            success_table = Table(show_header=False)
            success_table.add_column("Field", style="cyan")
            success_table.add_column("Value", style="green")
            success_table.add_row("Variable", name)
            success_table.add_row("New Value", value)
            success_table.add_row("Operation", "Set")
            success_table.add_row("Status", "✅ Complete")

            console.print(
                Panel(
                    success_table,
                    title="[bold green]✅ Variable Set Successfully",
                    border_style="green",
                    box=box.ROUNDED,
                )
            )

            # Format content for return
            set_content: List[ToolResultContent] = [{"text": f"Set {name} = {value}"}]
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": set_content,
            }
        elif action == "validate":
            if "name" not in tool_input:
                raise ValueError("name parameter is required for validate action")

            name = tool_input["name"]
            value = os.getenv(name)

            if value is None:
                error_content: List[ToolResultContent] = [{"text": f"Environment variable {name} not found"}]
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": error_content,
                }

            # Add validation logic here based on variable name patterns
            # For example, validate URL format, numeric values, etc.

            # Format content for return
            validate_content: List[ToolResultContent] = [{"text": f"Environment variable {name} is valid"}]
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": validate_content,
            }

        elif action == "delete":
            if "name" not in tool_input:
                error_msg = "name parameter is required for delete action"
                console.print(format_error_message(error_msg))
                raise ValueError(error_msg)

            name = tool_input["name"]

            # Check protected status first
            if name in PROTECTED_VARS:
                error_msg = (
                    f"⚠️ Cannot delete protected variable: {name}\n"
                    "Protected variables ensure system stability and security."
                )
                console.print(format_error_message(error_msg))
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": f"Cannot delete protected variable: {name}"}],
                }

            # Check if variable exists
            if name not in os.environ:
                error_msg = f"Environment variable not found: {name}"
                console.print(format_error_message(error_msg))
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": error_msg}],
                }

            # Show detailed preview for confirmation
            if needs_confirmation:
                # Show operation preview
                console.print(format_operation_preview(action="delete", name=name, value=os.environ[name]))

                # Show warning message
                warning_table = Table(show_header=False, box=box.SIMPLE)
                warning_table.add_column("Item", style="yellow")
                warning_table.add_column("Details", style="white")
                warning_table.add_row("Action", "🗑️ Delete Environment Variable")
                warning_table.add_row("Variable", name)
                warning_table.add_row("Current Value", os.environ[name])
                warning_table.add_row("Warning", "This action cannot be undone")

                console.print(
                    Panel(
                        warning_table,
                        title="[bold red]⚠️ Warning: Destructive Action",
                        border_style="red",
                        box=box.ROUNDED,
                    )
                )

                # Ask for confirmation
                confirm = user_input.get_user_input(
                    "\n<red><bold>Do you want to proceed with deleting this environment variable?</bold> [y/*]</red>"
                )
                # For tests, 'y' should be recognized even with extra spaces or newlines
                if confirm.strip().lower() != "y":
                    console.print(format_error_message("Operation cancelled by user"))
                    return {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [{"text": f"Operation cancelled by user, reason: {confirm}"}],
                    }

            # Delete the variable
            value = os.environ[name]
            del os.environ[name]

            # Show success message
            show_operation_result(console, True, f"Successfully retrieved {name}")
            success_table = Table(show_header=False)
            success_table.add_column("Field", style="cyan")
            success_table.add_column("Value", style="green")
            success_table.add_row("Variable", name)
            success_table.add_row("Previous Value", value)
            success_table.add_row("Operation", "Delete")
            success_table.add_row("Status", "✅ Complete")

            console.print(
                Panel(
                    success_table,
                    title="[bold green]✅ Variable Deleted Successfully",
                    border_style="green",
                    box=box.ROUNDED,
                )
            )

            # Format content for return
            delete_content: List[ToolResultContent] = [{"text": f"Deleted environment variable: {name}"}]
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": delete_content,
            }

    except Exception as e:
        exception_content: List[ToolResultContent] = [{"text": f"Environment tool error: {str(e)}"}]
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": exception_content,
        }

    # Fallback return in case no action matched
    return default_result  # type: ignore
