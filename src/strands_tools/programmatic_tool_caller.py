"""
Programmatic Tool Calling for Strands Agents.

This module provides a tool that enables programmatic/code-based tool invocation,
similar to Anthropic's Programmatic Tool Calling feature. It allows an agent to
write Python code that calls other tools as functions, reducing API round-trips
and enabling complex orchestration logic.

Tools are exposed as async functions (e.g., `await calculator(expression="2+2")`).
The code runs in an async context automatically - no boilerplate needed.

Usage:
```python
from strands import Agent
from strands_tools import programmatic_tool_caller, calculator

agent = Agent(tools=[programmatic_tool_caller, calculator])

result = agent.tool.programmatic_tool_caller(
    code='''
result = await calculator(expression="2 + 2")
print(f"Result: {result}")

# Parallel execution
results = await asyncio.gather(
    calculator(expression="10 * 1"),
    calculator(expression="10 * 2"),
)
print(f"Parallel: {results}")
'''
)
```

Environment Variables:
- PROGRAMMATIC_TOOL_CALLER_ALLOWED_TOOLS: Comma-separated list of allowed tools
- BYPASS_TOOL_CONSENT: Skip user confirmation if "true"

Custom Executors:
```python
from strands_tools.programmatic_tool_caller import programmatic_tool_caller, Executor

class MyExecutor(Executor):
    def execute(self, code: str, namespace: dict) -> str:
        # Custom execution logic
        ...

# Set custom executor
programmatic_tool_caller.executor = MyExecutor()
```
"""

import asyncio
import logging
import os
import sys
import textwrap
import traceback
from abc import ABC, abstractmethod
from io import StringIO
from typing import Any, Callable, Dict, List, Optional

from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from strands import tool
from strands.types.tools import ToolContext

from strands_tools.utils import console_util
from strands_tools.utils.user_input import get_user_input

logger = logging.getLogger(__name__)


# =============================================================================
# Executor Interface & Implementations
# =============================================================================


class Executor(ABC):
    """Abstract base class for code executors.

    Implement this interface to provide custom execution environments
    (e.g., Docker, Lambda, smolagents, Code Interpreter).
    """

    @abstractmethod
    def execute(self, code: str, namespace: Dict[str, Any]) -> str:
        """Execute code and return captured output.

        Args:
            code: Python code to execute (already wrapped in async context)
            namespace: Execution namespace with tools and builtins

        Returns:
            Captured stdout/stderr output

        Raises:
            SyntaxError: If code has syntax errors
            Exception: If execution fails
        """
        pass


class LocalAsyncExecutor(Executor):
    """Default executor - runs code locally with asyncio."""

    def execute(self, code: str, namespace: Dict[str, Any]) -> str:
        """Execute code in local async context."""
        # Wrap user code in async function
        indented_code = textwrap.indent(code, "    ")
        wrapped_code = f"async def __user_code__():\n{indented_code}\n"

        # Capture output
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Define and run the async function
            exec(wrapped_code, namespace)
            asyncio.run(namespace["__user_code__"]())

            output = stdout_capture.getvalue()
            errors = stderr_capture.getvalue()
            if errors:
                output += f"\n[stderr]\n{errors}"
            return output

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# =============================================================================
# Tool Execution Helpers
# =============================================================================


def _execute_tool(agent: Any, tool_name: str, tool_input: Dict[str, Any]) -> Any:
    """Execute a tool through the agent's tool caller.

    Uses agent.tool.<name>() which properly handles all tool types including MCP tools.
    """
    if agent is None:
        raise RuntimeError("No agent available for tool execution")

    try:
        # Use agent.tool.<name>() which works for ALL tool types (including MCP tools)
        # record_direct_tool_call=False prevents polluting message history during programmatic calls
        tool_func = getattr(agent.tool, tool_name)
        result = tool_func(record_direct_tool_call=False, **tool_input)

        if isinstance(result, dict):
            if result.get("status") == "error":
                error_content = result.get("content", [{"text": "Unknown error"}])
                error_text = error_content[0].get("text", "Unknown error") if error_content else "Unknown error"
                raise RuntimeError(f"Tool error: {error_text}")

            content = result.get("content", [])
            if content and isinstance(content, list):
                text_parts = [item["text"] for item in content if isinstance(item, dict) and "text" in item]
                if text_parts:
                    return "\n".join(text_parts)
            return str(result)

        return result

    except AttributeError as e:
        raise RuntimeError(f"Tool '{tool_name}' not found in registry") from e
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}")
        raise RuntimeError(f"Failed to execute tool '{tool_name}': {e}") from e


def _create_async_tool_function(agent: Any, tool_name: str) -> Callable:
    """Create an async function wrapper for a tool."""

    async def tool_function(**kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: _execute_tool(agent, tool_name, kwargs))

    return tool_function


def _validate_code(code: str) -> List[str]:
    """Validate Python code for potential security issues."""
    warnings: List[str] = []
    dangerous_patterns = [
        r"\bimport\s+subprocess\b",
        r"\bfrom\s+subprocess\b",
        r"\b__import__\s*\(",
        r"\bexec\s*\(",
        r"\beval\s*\(",
        r"\bcompile\s*\(",
        r"\bglobals\s*\(",
        r"\blocals\s*\(",
        r"open\s*\(",
        r"os\.remove",
        r"os\.unlink",
        r"os\.rmdir",
    ]

    import re

    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            warnings.append(f"Potentially dangerous pattern: {pattern}")

    return warnings


def _get_allowed_tools(agent: Any) -> set:
    """Get allowed tools from env var or default to all (except self)."""
    all_tools = set(agent.tool_registry.registry.keys()) - {"programmatic_tool_caller"}

    env_allowed = os.environ.get("PROGRAMMATIC_TOOL_CALLER_ALLOWED_TOOLS", "").strip()
    if env_allowed:
        allowed_list = [t.strip() for t in env_allowed.split(",") if t.strip()]
        return all_tools & set(allowed_list)

    return all_tools


# =============================================================================
# Main Tool
# =============================================================================

# Default executor - can be swapped by users
_default_executor = LocalAsyncExecutor()


@tool(context=True)
def programmatic_tool_caller(
    code: str,
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """
    Execute Python code with access to agent tools as async functions.

    Tools are available as async functions - use `await` to call them.
    Code runs in async context automatically, no boilerplate needed.

    Example:
        ```python
        # Simple tool call
        result = await calculator(expression="2 + 2")
        print(result)

        # Loop with tool calls
        for i in range(3):
            r = await calculator(expression=f"{i} * 10")
            print(r)

        # Parallel execution
        results = await asyncio.gather(
            calculator(expression="1+1"),
            calculator(expression="2+2"),
        )
        print(results)
        ```

    Environment Variables:
        PROGRAMMATIC_TOOL_CALLER_ALLOWED_TOOLS: Comma-separated list of tools to expose
        BYPASS_TOOL_CONSENT: Skip confirmation if "true"

    Args:
        code: Python code to execute. Use `await tool_name(...)` to call tools.
        tool_context: Injected automatically.

    Returns:
        Dict with status and print() output only.
    """
    console = console_util.create()
    bypass_consent = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

    try:
        if tool_context is None or tool_context.agent is None:
            return {
                "status": "error",
                "content": [{"text": "No agent context available. This tool requires an agent."}],
            }

        agent = tool_context.agent
        code_warnings = _validate_code(code)

        # Show code preview
        console.print(
            Panel(
                Syntax(code, "python", theme="monokai", line_numbers=True),
                title="[bold blue]Programmatic Tool Calling[/]",
                border_style="blue",
            )
        )

        # Get allowed tools
        available_tools = _get_allowed_tools(agent)

        tools_table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE)
        tools_table.add_column("Available Tools", style="green")
        for tool_name in sorted(available_tools):
            tools_table.add_row(f"await {tool_name}(...)")
        console.print(tools_table)

        if code_warnings:
            console.print(
                Panel(
                    "\n".join(f"⚠️ {w}" for w in code_warnings),
                    title="[bold yellow]Warnings[/]",
                    border_style="yellow",
                )
            )

        # User confirmation
        if not bypass_consent:
            user_input = get_user_input("<yellow><bold>Execute this code?</bold> [y/*]</yellow>")
            if user_input.lower().strip() != "y":
                cancel_reason = user_input if user_input.strip() != "n" else get_user_input("Reason:")
                return {
                    "status": "error",
                    "content": [{"text": f"Cancelled. Reason: {cancel_reason}"}],
                }

        # Build execution namespace
        exec_namespace: Dict[str, Any] = {
            "__builtins__": __builtins__,
            "asyncio": asyncio,
            "json": __import__("json"),
            "re": __import__("re"),
            "math": __import__("math"),
        }

        # Inject tools as async functions
        for tool_name in available_tools:
            exec_namespace[tool_name] = _create_async_tool_function(agent, tool_name)

        console.print("[green]Executing...[/]")

        # Use the executor
        executor = getattr(programmatic_tool_caller, "executor", _default_executor)
        captured_output = executor.execute(code, exec_namespace)

        console.print("[bold green]✓ Done[/]")
        if captured_output.strip():
            console.print(Panel(captured_output, title="[bold green]Output[/]", border_style="green"))

        return {
            "status": "success",
            "content": [{"text": captured_output.strip() if captured_output.strip() else "(no output)"}],
        }

    except SyntaxError:
        error_msg = f"Syntax error:\n{traceback.format_exc()}"
        console.print(Panel(error_msg, title="[bold red]Error[/]", border_style="red"))
        return {"status": "error", "content": [{"text": error_msg}]}

    except Exception:
        error_msg = f"Execution error:\n{traceback.format_exc()}"
        console.print(Panel(error_msg, title="[bold red]Error[/]", border_style="red"))
        return {"status": "error", "content": [{"text": error_msg}]}


# Allow setting custom executor
programmatic_tool_caller.executor = _default_executor
