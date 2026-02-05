"""
Programmatic Tool Calling for Strands Agents.

This module provides a tool that enables programmatic/code-based tool invocation,
similar to Anthropic's Programmatic Tool Calling feature. It allows an agent to
write Python code that calls other tools as functions, reducing API round-trips
and enabling complex orchestration logic.

Tools are exposed as async functions (e.g., `await calculator(expression="2+2")`).
The code runs in an async context automatically - no boilerplate needed.

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import programmatic_tool_caller, calculator

agent = Agent(tools=[programmatic_tool_caller, calculator])

result = agent.tool.programmatic_tool_caller(
    code='''
# Tools are async - use await
result = await calculator(expression="2 + 2")
print(f"Result: {result}")

# Parallel execution with asyncio.gather
results = await asyncio.gather(
    calculator(expression="10 * 1"),
    calculator(expression="10 * 2"),
    calculator(expression="10 * 3"),
)
print(f"Parallel results: {results}")
'''
)
```

Note: Only print() output is returned to the agent.
"""

import asyncio
import logging
import os
import re
import sys
import textwrap
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
        self.stdout = StringIO()
        self.stderr = StringIO()
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def __enter__(self) -> "OutputCapture":
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def get_output(self) -> str:
        output = self.stdout.getvalue()
        errors = self.stderr.getvalue()
        if errors:
            output += f"\n[stderr]\n{errors}"
        return output


def _execute_tool(agent: Any, tool_name: str, tool_input: Dict[str, Any]) -> Any:
    """Execute a tool through the agent's tool registry."""
    if agent is None:
        raise RuntimeError("No agent available for tool execution")

    tool_impl = agent.tool_registry.registry.get(tool_name)
    if tool_impl is None:
        raise RuntimeError(f"Tool '{tool_name}' not found in registry")

    try:
        if callable(tool_impl):
            result = tool_impl(**tool_input)
        else:
            raise RuntimeError(f"Tool '{tool_name}' is not callable")

        if isinstance(result, dict):
            if result.get("status") == "error":
                error_content = result.get("content", [{"text": "Unknown error"}])
                error_text = error_content[0].get("text", "Unknown error") if error_content else "Unknown error"
                raise RuntimeError(f"Tool error: {error_text}")

            content = result.get("content", [])
            if content and isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                if text_parts:
                    return "\n".join(text_parts)
            return str(result)

        return result

    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}")
        raise RuntimeError(f"Failed to execute tool '{tool_name}': {e}") from e


def _create_async_tool_function(agent: Any, tool_name: str) -> Any:
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

    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            warnings.append(f"Potentially dangerous pattern: {pattern}")

    return warnings


@tool(context=True)
def programmatic_tool_caller(
    code: str,
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """
    Execute Python code with access to agent tools as async functions.

    Tools are available as async functions - use `await` to call them.
    The code runs in an async context automatically, no boilerplate needed.

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

        # Get available tools (excluding self)
        available_tools = set(agent.tool_registry.registry.keys()) - {"programmatic_tool_caller"}

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

        # Wrap user code in async function and run it
        # Indent user code to be inside the async function
        indented_code = textwrap.indent(code, "    ")
        wrapped_code = f"async def __user_code__():\n{indented_code}\n"

        console.print("[green]Executing...[/]")

        output_capture = OutputCapture()
        with output_capture:
            # First exec to define the async function
            exec(wrapped_code, exec_namespace)
            # Then run it
            asyncio.run(exec_namespace["__user_code__"]())

        captured_output = output_capture.get_output()

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
