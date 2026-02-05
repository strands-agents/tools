"""
Programmatic Tool Calling for Strands Agents.

This module provides a tool that enables programmatic/code-based tool invocation,
similar to Anthropic's Programmatic Tool Calling feature. It allows an agent to
write Python code that calls other tools as functions, reducing API round-trips
and enabling complex orchestration logic.

The key feature is that within the executed code, agent tools are exposed as
callable methods (e.g., `tools.calculator(expression="2+2")`) with callbacks
that route back to the agent's actual tool execution system.

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import programmatic_tool_caller, calculator, file_read

# Create an agent with programmatic tool calling and other tools
agent = Agent(tools=[programmatic_tool_caller, calculator, file_read])

# The agent can now write code that calls tools programmatically
result = agent.tool.programmatic_tool_caller(
    code='''
# Tools are available as callable functions
result = tools.calculator(expression="2 + 2")
print(f"Calculator result: {result}")

# Multiple tool calls in one execution
content = tools.file_read(path="example.txt", mode="view")
print(f"File content: {content}")

# Complex orchestration with loops and conditionals
total = 0
for i in range(5):
    r = tools.calculator(expression=f"{i} * 10")
    total += int(r)
print(f"Total: {total}")
'''
)
```

See the programmatic_tool_caller function docstring for more details.
"""

import logging
import os
import re
import sys
import traceback
from datetime import datetime
from io import StringIO
from typing import Any, Callable, Dict, List, Optional

from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from strands import tool

from strands_tools.utils import console_util
from strands_tools.utils.user_input import get_user_input

logger = logging.getLogger(__name__)


class ToolProxy:
    """
    Proxy object that exposes agent tools as callable methods.

    This class creates a namespace where each registered tool in the agent's
    tool registry becomes a callable method. When a tool is called, the proxy
    routes the execution through the provided callback function to the actual
    tool implementation.

    Attributes:
        _tool_registry: The agent's tool registry containing available tools.
        _callback: Function to execute tool calls and return results.
        _tool_calls: List tracking all tool calls made during execution.
        _available_tools: Set of tool names available for programmatic calling.
    """

    def __init__(
        self,
        tool_registry: Any,
        callback: Callable[[str, Dict[str, Any]], Any],
        allowed_tools: Optional[List[str]] = None,
    ):
        """
        Initialize the ToolProxy.

        Args:
            tool_registry: The agent's tool registry containing available tools.
            callback: Function to execute tool calls. Should accept tool name and
                     input dict, returning the tool result.
            allowed_tools: Optional list of tool names to expose. If None, all
                          registered tools are available.
        """
        self._tool_registry = tool_registry
        self._callback = callback
        self._tool_calls: List[Dict[str, Any]] = []

        # Determine which tools are available
        if allowed_tools is not None:
            self._available_tools = set(allowed_tools) & set(tool_registry.registry.keys())
        else:
            # Exclude the programmatic_tool_caller itself to avoid recursion
            self._available_tools = {
                name for name in tool_registry.registry.keys() if name != "programmatic_tool_caller"
            }

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """
        Get a callable for the specified tool name.

        Args:
            name: The name of the tool to get.

        Returns:
            A callable that executes the tool with the provided arguments.

        Raises:
            AttributeError: If the tool is not available.
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if name not in self._available_tools:
            available = ", ".join(sorted(self._available_tools))
            raise AttributeError(f"Tool '{name}' is not available. Available tools: {available}")

        def tool_caller(**kwargs: Any) -> Any:
            """Execute the tool with the given arguments."""
            # Record the tool call
            call_record = {
                "tool_name": name,
                "input": kwargs,
                "timestamp": datetime.now().isoformat(),
            }
            self._tool_calls.append(call_record)

            # Execute the tool through the callback
            try:
                result = self._callback(name, kwargs)
                call_record["status"] = "success"
                call_record["result"] = result
                return result
            except Exception as e:
                call_record["status"] = "error"
                call_record["error"] = str(e)
                raise

        return tool_caller

    def list_tools(self) -> List[str]:
        """
        List all available tools.

        Returns:
            Sorted list of available tool names.
        """
        return sorted(self._available_tools)

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific tool.

        Args:
            name: The name of the tool.

        Returns:
            Dictionary with tool specification, or None if not found.
        """
        if name not in self._available_tools:
            return None

        tool_impl = self._tool_registry.registry.get(name)
        if tool_impl and hasattr(tool_impl, "tool_spec"):
            return tool_impl.tool_spec
        return None

    def get_call_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of tool calls made during execution.

        Returns:
            List of dictionaries containing call information.
        """
        return self._tool_calls.copy()


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
    For Strands DecoratedFunctionTool, tools can be called directly with
    keyword arguments.

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
        # Call the tool - Strands DecoratedFunctionTool can be called directly
        # with keyword arguments
        if callable(tool_impl):
            result = tool_impl(**tool_input)
        else:
            raise RuntimeError(f"Tool '{tool_name}' is not callable")

        # Handle the result
        if isinstance(result, dict):
            # Check for error status in dict result
            if result.get("status") == "error":
                error_content = result.get("content", [{"text": "Unknown error"}])
                error_text = error_content[0].get("text", "Unknown error") if error_content else "Unknown error"
                raise RuntimeError(f"Tool error: {error_text}")

            # Extract text content if available
            content = result.get("content", [])
            if content and isinstance(content, list) and len(content) > 0:
                first_content = content[0]
                if isinstance(first_content, dict) and "text" in first_content:
                    return first_content["text"]
            return str(result)

        # Return non-dict results directly (could be string, int, etc.)
        return result

    except RuntimeError:
        # Re-raise RuntimeError without wrapping
        raise
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}")
        raise RuntimeError(f"Failed to execute tool '{tool_name}': {e}") from e


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
        r"\beval\s*\(",  # Allow tools.* but warn on raw eval
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


@tool
def programmatic_tool_caller(
    code: str,
    allowed_tools: Optional[List[str]] = None,
    timeout: int = 30,
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute Python code with access to agent tools as callable functions.

    This tool enables programmatic tool calling, where an agent can write Python
    code that invokes other tools as functions. Tools are exposed through a `tools`
    namespace, allowing patterns like `tools.calculator(expression="2+2")`.

    How It Works:
    ------------
    1. The code is executed in a restricted namespace with a `tools` object
    2. The `tools` object proxies calls to the agent's registered tools
    3. Tool calls are executed synchronously and results returned to the code
    4. All output (print statements, tool results) is captured and returned

    Key Features:
    ------------
    - Tools exposed as `tools.<tool_name>(**kwargs)`
    - Supports complex orchestration with loops, conditionals, data processing
    - Captures stdout/stderr from the executed code
    - Records all tool calls for transparency
    - Validates code for potentially dangerous patterns

    Example Code Patterns:
    --------------------
    ```python
    # Simple tool call
    result = tools.calculator(expression="10 * 5")
    print(f"Result: {result}")

    # Chaining tool calls
    content = tools.file_read(path="data.txt", mode="view")
    analysis = tools.use_llm(prompt=f"Summarize: {content}")

    # Loop with tool calls
    for item in ["apple", "banana", "cherry"]:
        info = tools.http_request(url=f"https://api.example.com/info/{item}")
        print(f"{item}: {info}")

    # Conditional logic
    if tools.file_read(path="config.json", mode="stats"):
        config = tools.file_read(path="config.json", mode="view")
    ```

    Args:
        code: Python code to execute. The code has access to a `tools` object
            that provides callable methods for each available tool.
        allowed_tools: Optional list of tool names to make available. If None,
            all tools except programmatic_tool_caller are available.
        timeout: Maximum execution time in seconds. Default is 30.
        agent: The Strands agent instance (automatically injected).

    Returns:
        Dict with status and content:
        - status: "success" or "error"
        - content: List of dicts with "text" keys containing:
            - Captured output from code execution
            - Summary of tool calls made
            - Any error messages if execution failed

    Security Notes:
    -------------
    - Code is validated for potentially dangerous patterns before execution
    - User confirmation is required unless BYPASS_TOOL_CONSENT is set
    - The code runs with limited access to the Python environment
    - Tool access is restricted to the agent's registered tools

    Examples:
    --------
    Basic calculation:
    >>> agent.tool.programmatic_tool_caller(
    ...     code='result = tools.calculator(expression="2 + 2"); print(result)'
    ... )

    Multi-tool workflow:
    >>> agent.tool.programmatic_tool_caller(
    ...     code='''
    ...     files = tools.file_read(path=".", mode="find")
    ...     for f in files.split("\\n")[:5]:
    ...         print(f"File: {f}")
    ...     ''',
    ...     allowed_tools=["file_read"]
    ... )
    """
    console = console_util.create()

    # Check for development mode (bypass consent)
    bypass_consent = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

    try:
        # Validate agent is available
        if agent is None:
            return {
                "status": "error",
                "content": [{"text": "No agent context available. This tool requires an agent."}],
            }

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

        # Show available tools
        available_tools = set(agent.tool_registry.registry.keys()) - {"programmatic_tool_caller"}
        if allowed_tools:
            available_tools = available_tools & set(allowed_tools)

        tools_table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE)
        tools_table.add_column("Available Tools", style="green")
        tools_table.add_column("Description", style="dim")

        for tool_name in sorted(available_tools):
            tool_impl = agent.tool_registry.registry.get(tool_name)
            desc = ""
            if tool_impl and hasattr(tool_impl, "tool_spec"):
                desc = tool_impl.tool_spec.get("description", "")[:60]
                if len(tool_impl.tool_spec.get("description", "")) > 60:
                    desc += "..."
            tools_table.add_row(tool_name, desc)

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
            details_table.add_row("Timeout", f"{timeout} seconds")

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

        # Create the tool callback
        def tool_callback(name: str, kwargs: Dict[str, Any]) -> Any:
            return _execute_tool(agent, name, kwargs)

        # Create the tool proxy
        tools_proxy = ToolProxy(
            tool_registry=agent.tool_registry,
            callback=tool_callback,
            allowed_tools=list(allowed_tools) if allowed_tools else None,
        )

        # Create the execution namespace
        exec_namespace: Dict[str, Any] = {
            "tools": tools_proxy,
            "__builtins__": __builtins__,
            # Add some safe imports
            "json": __import__("json"),
            "re": __import__("re"),
            "datetime": __import__("datetime"),
            "math": __import__("math"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
        }

        # Execute with output capture
        start_time = datetime.now()
        output_capture = OutputCapture()

        console.print("[green]Executing code...[/]")

        with output_capture:
            exec(code, exec_namespace)

        execution_time = (datetime.now() - start_time).total_seconds()
        captured_output = output_capture.get_output()

        # Get tool call history
        call_history = tools_proxy.get_call_history()

        # Build result summary
        result_parts: List[str] = []

        if captured_output.strip():
            result_parts.append(f"Output:\n{captured_output}")

        if call_history:
            calls_summary = f"\nTool calls made: {len(call_history)}"
            for i, call in enumerate(call_history, 1):
                calls_summary += f"\n  {i}. {call['tool_name']}({call['input']}) -> {call.get('status', 'unknown')}"
            result_parts.append(calls_summary)

        result_parts.append(f"\nExecution time: {execution_time:.2f}s")

        # Show success panel
        success_msg = f"✓ Code executed successfully ({execution_time:.2f}s, {len(call_history)} tool calls)"
        console.print(f"[bold green]{success_msg}[/]")

        if captured_output.strip():
            console.print(
                Panel(
                    captured_output,
                    title="[bold green]Execution Output[/]",
                    border_style="green",
                )
            )

        return {
            "status": "success",
            "content": [{"text": "\n".join(result_parts)}],
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
