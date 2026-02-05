"""
Programmatic Tool Calling for Strands Agents.

This module provides a tool that enables programmatic/code-based tool invocation,
similar to Anthropic's Programmatic Tool Calling feature. It allows an agent to
write Python code that calls other tools as functions, reducing API round-trips
and enabling complex orchestration logic.

The key feature is that within the executed code, agent tools are exposed as
callable async functions (e.g., `await calculator(expression="2+2")`) that
route back to the agent's actual tool execution system.

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import programmatic_tool_caller, calculator, file_read

# Create an agent with programmatic tool calling and other tools
agent = Agent(tools=[programmatic_tool_caller, calculator, file_read])

# The agent can now write code that calls tools programmatically
result = agent.tool.programmatic_tool_caller(
    code='''
import asyncio

async def main():
    # Tools are available as async functions
    result = await calculator(expression="2 + 2")
    print(f"Calculator result: {result}")

    # Complex orchestration with loops and conditionals
    total = 0
    for i in range(5):
        r = await calculator(expression=f"{i} * 10")
        total += int(r)
    print(f"Total: {total}")

asyncio.run(main())
'''
)
```

Note: Only the print() output from the code is returned to the agent.
Tool results are processed within the code and do not enter the agent's context
unless explicitly printed.

See the programmatic_tool_caller function docstring for more details.
"""

import asyncio
import logging
import os
import re
import sys
import traceback
from io import StringIO
from typing import Any, Dict, List, Optional

from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from strands import tool
from strands.types.tools import ToolContext

from strands_tools.utils import console_util
from strands_tools.utils.user_input import get_user_input

logger = logging.getLogger(__name__)


class OutputCapture:
    """Captures stdout and stderr output during code execution."""

    def __init__(self) -> None:
        """Initialize the output capture."""
        self.stdout = StringIO()
        self.stderr = StringIO()
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def __enter__(self) -> "OutputCapture":
        """Start capturing output."""
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop capturing output."""
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def get_output(self) -> str:
        """
        Get captured output from both stdout and stderr.

        Returns:
            Combined output string, with stderr marked if present.
        """
        output = self.stdout.getvalue()
        errors = self.stderr.getvalue()
        if errors:
            output += f"\n[stderr]\n{errors}"
        return output


def _execute_tool(
    agent: Any,
    tool_name: str,
    tool_input: Dict[str, Any],
) -> Any:
    """
    Execute a tool through the agent's tool system.

    This function routes tool calls through the agent's tool registry.
    Results are returned directly to the calling code, NOT to the agent's context.

    Args:
        agent: The Strands agent instance.
        tool_name: Name of the tool to execute.
        tool_input: Dictionary of tool input parameters.

    Returns:
        The tool result content (string or extracted from ToolResult dict).

    Raises:
        RuntimeError: If tool execution fails.
    """
    if agent is None:
        raise RuntimeError("No agent available for tool execution")

    # Get the tool from registry
    tool_impl = agent.tool_registry.registry.get(tool_name)
    if tool_impl is None:
        raise RuntimeError(f"Tool '{tool_name}' not found in registry")

    # Execute the tool
    try:
        if callable(tool_impl):
            result = tool_impl(**tool_input)
        else:
            raise RuntimeError(f"Tool '{tool_name}' is not callable")

        # Handle the result - could be various types
        if isinstance(result, dict):
            # Check for error status in dict result
            if result.get("status") == "error":
                error_content = result.get("content", [{"text": "Unknown error"}])
                error_text = error_content[0].get("text", "Unknown error") if error_content else "Unknown error"
                raise RuntimeError(f"Tool error: {error_text}")

            # Extract all text content if available
            content = result.get("content", [])
            if content and isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                if text_parts:
                    return "\n".join(text_parts)
            return str(result)

        # Return non-dict results directly
        return result

    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}")
        raise RuntimeError(f"Failed to execute tool '{tool_name}': {e}") from e


def _create_tool_function(agent: Any, tool_name: str) -> Any:
    """
    Create an async function wrapper for a tool.

    Tools are exposed as async functions to support parallel execution patterns
    and match Anthropic's programmatic tool calling interface.

    Args:
        agent: The Strands agent instance.
        tool_name: Name of the tool to wrap.

    Returns:
        An async function that calls the tool.
    """

    async def tool_function(**kwargs: Any) -> Any:
        """Async wrapper for tool execution."""
        # Run the synchronous tool execution in a thread pool to not block
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: _execute_tool(agent, tool_name, kwargs))

    # Also create a sync version for simpler use cases
    def tool_function_sync(**kwargs: Any) -> Any:
        """Sync wrapper for tool execution."""
        return _execute_tool(agent, tool_name, kwargs)

    # Return both - we'll inject both into the namespace
    return tool_function, tool_function_sync


def _validate_code(code: str) -> List[str]:
    """
    Validate Python code for potential security issues.

    Args:
        code: The Python code to validate.

    Returns:
        List of warning messages for potentially dangerous patterns.
    """
    warnings: List[str] = []

    # Check for potentially dangerous imports
    dangerous_imports = [
        r"\bimport\s+subprocess\b",
        r"\bfrom\s+subprocess\b",
        r"\bimport\s+shutil\b",
        r"\bfrom\s+shutil\b",
        r"\b__import__\s*\(",
        r"\bexec\s*\(",
        r"\beval\s*\(",
        r"\bcompile\s*\(",
        r"\bgetattr\s*\(",
        r"\bsetattr\s*\(",
        r"\bdelattr\s*\(",
        r"\bglobals\s*\(",
        r"\blocals\s*\(",
    ]

    for pattern in dangerous_imports:
        if re.search(pattern, code):
            warnings.append(f"Potentially dangerous pattern detected: {pattern}")

    # Check for file system operations
    fs_patterns = [
        r"open\s*\(",
        r"os\.remove",
        r"os\.unlink",
        r"os\.rmdir",
        r"pathlib.*unlink",
        r"pathlib.*rmdir",
    ]

    for pattern in fs_patterns:
        if re.search(pattern, code):
            warnings.append(f"File system operation detected: {pattern}")

    return warnings


@tool(context=True)
def programmatic_tool_caller(
    code: str,
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """
    Execute Python code with access to agent tools as callable async functions.

    This tool enables programmatic tool calling, where an agent can write Python
    code that invokes other tools as functions. Tools are exposed as async functions
    at the top level (e.g., `await calculator(expression="2+2")`), matching
    Anthropic's programmatic tool calling interface.

    How It Works:
    ------------
    1. The code is executed in a namespace where each tool is an async function
    2. Tool calls are executed and results returned directly to the code
    3. Only print() output is captured and returned to the agent
    4. Tool results do NOT enter the agent's context unless explicitly printed

    Key Features:
    ------------
    - Tools exposed as async functions: `await tool_name(**kwargs)`
    - Also available as sync functions: `tool_name_sync(**kwargs)`
    - Supports complex orchestration with loops, conditionals, data processing
    - Only print() output is returned (reduces context window usage)
    - Tool results are processed in code, not sent back to agent

    Example Code Patterns:
    --------------------
    ```python
    import asyncio

    async def main():
        # Simple tool call
        result = await calculator(expression="10 * 5")
        print(f"Result: {result}")

        # Loop with tool calls
        for item in ["apple", "banana", "cherry"]:
            info = await http_request(url=f"https://api.example.com/info/{item}")
            print(f"{item}: {info}")

        # Conditional logic
        stats = await file_read(path="config.json", mode="stats")
        if stats:
            config = await file_read(path="config.json", mode="view")
            print(f"Config loaded")

    asyncio.run(main())
    ```

    Or using sync functions for simpler cases:
    ```python
    result = calculator_sync(expression="2 + 2")
    print(f"Result: {result}")
    ```

    Args:
        code: Python code to execute. Tools are available as async functions
            (e.g., `await calculator(...)`) or sync functions (`calculator_sync(...)`).
        tool_context: The Strands tool context (automatically injected).

    Returns:
        Dict with status and content:
        - status: "success" or "error"
        - content: List with single text block containing print() output only

    Design Notes:
    ------------
    - Tools are async to support parallel execution patterns
    - Only print() output is returned to minimize context usage
    - Tool results are processed in code, enabling filtering/aggregation
    - Matches Anthropic's programmatic tool calling interface
    """
    console = console_util.create()

    # Check for development mode (bypass consent)
    bypass_consent = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

    try:
        # Get agent from tool_context
        if tool_context is None or tool_context.agent is None:
            return {
                "status": "error",
                "content": [{"text": "No agent context available. This tool requires an agent."}],
            }

        agent = tool_context.agent

        # Validate code
        code_warnings = _validate_code(code)

        # Show code preview
        console.print(
            Panel(
                Syntax(code, "python", theme="monokai", line_numbers=True),
                title="[bold blue]Programmatic Tool Calling - Code Preview[/]",
                border_style="blue",
            )
        )

        # Get available tools (excluding self)
        available_tools = set(agent.tool_registry.registry.keys()) - {"programmatic_tool_caller"}

        tools_table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE)
        tools_table.add_column("Available Tools", style="green")
        tools_table.add_column("Usage", style="dim")

        for tool_name in sorted(available_tools):
            tools_table.add_row(tool_name, f"await {tool_name}(...) or {tool_name}_sync(...)")

        console.print(tools_table)

        # Show warnings if any
        if code_warnings:
            warnings_panel = Panel(
                "\n".join(f"⚠️ {w}" for w in code_warnings),
                title="[bold yellow]Security Warnings[/]",
                border_style="yellow",
            )
            console.print(warnings_panel)

        # Request user confirmation if not in bypass mode
        if not bypass_consent:
            details_table = Table(show_header=False, box=box.SIMPLE)
            details_table.add_column("Property", style="cyan", justify="right")
            details_table.add_column("Value", style="green")
            details_table.add_row("Code Length", f"{len(code)} characters")
            details_table.add_row("Line Count", f"{len(code.splitlines())} lines")
            details_table.add_row("Available Tools", str(len(available_tools)))

            console.print(
                Panel(
                    details_table,
                    title="[bold blue]🔧 Programmatic Tool Calling Preview",
                    border_style="blue",
                    box=box.ROUNDED,
                )
            )

            user_input = get_user_input("<yellow><bold>Execute this code with tool access?</bold> [y/*]</yellow>")

            if user_input.lower().strip() != "y":
                cancel_reason = user_input if user_input.strip() != "n" else get_user_input("Reason for cancellation:")
                return {
                    "status": "error",
                    "content": [{"text": f"Execution cancelled by user. Reason: {cancel_reason}"}],
                }

        # Create the execution namespace with tools as functions
        exec_namespace: Dict[str, Any] = {
            "__builtins__": __builtins__,
            # Standard library imports
            "asyncio": __import__("asyncio"),
            "json": __import__("json"),
            "re": __import__("re"),
            "datetime": __import__("datetime"),
            "math": __import__("math"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
        }

        # Inject each tool as both async and sync functions
        for tool_name in available_tools:
            async_func, sync_func = _create_tool_function(agent, tool_name)
            exec_namespace[tool_name] = async_func  # await tool_name(...)
            exec_namespace[f"{tool_name}_sync"] = sync_func  # tool_name_sync(...)

        # Execute with output capture
        output_capture = OutputCapture()

        console.print("[green]Executing code...[/]")

        with output_capture:
            exec(code, exec_namespace)

        captured_output = output_capture.get_output()

        # Show success
        console.print("[bold green]✓ Code executed successfully[/]")

        if captured_output.strip():
            console.print(
                Panel(
                    captured_output,
                    title="[bold green]Output[/]",
                    border_style="green",
                )
            )

        # Return ONLY the print() output - this is what goes to the agent's context
        return {
            "status": "success",
            "content": [{"text": captured_output.strip() if captured_output.strip() else "(no output)"}],
        }

    except SyntaxError:
        error_msg = f"Syntax error in code:\n{traceback.format_exc()}"
        console.print(Panel(error_msg, title="[bold red]Syntax Error[/]", border_style="red"))
        return {
            "status": "error",
            "content": [{"text": error_msg}],
        }

    except Exception:
        error_msg = f"Execution error:\n{traceback.format_exc()}"
        console.print(Panel(error_msg, title="[bold red]Execution Error[/]", border_style="red"))
        return {
            "status": "error",
            "content": [{"text": error_msg}],
        }
