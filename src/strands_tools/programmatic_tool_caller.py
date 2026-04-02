"""Programmatic Tool Calling for Strands Agents.

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
- PROGRAMMATIC_TOOL_CALLER_EXTRA_MODULES: Comma-separated list of extra modules to inject
  into the namespace (e.g., "json,re,math,collections"). `asyncio` is always available.
- BYPASS_TOOL_CONSENT: Skip user confirmation if "true"

Namespace:
    The execution namespace matches python_repl's base: `{"__name__": "__main__"}`.
    `asyncio` is always injected (required for async tool calls).
    Additional modules can be added via PROGRAMMATIC_TOOL_CALLER_EXTRA_MODULES env var.
    Tool functions are injected as async callables (e.g., `await shell(command="ls")`).

Limitations: Tools that use interrupts (human-in-the-loop) are not supported. The SDK
blocks interrupts during direct/programmatic tool calls — there is no mechanism to pause
execution, collect human input, and resume in this context. If an interrupt-capable tool
is called, it will raise a RuntimeError which surfaces as a failed tool result back to
the agent.
"""

import asyncio
import importlib
import logging
import os
import sys
import textwrap
import traceback
from io import StringIO
from typing import Any, Callable, Dict, Optional

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


def _get_allowed_tools(agent: Any) -> set[str]:
    """Get allowed tools from env var or default to all (except self)."""
    all_tools = set(agent.tool_registry.registry.keys()) - {"programmatic_tool_caller"}

    env_allowed = os.environ.get("PROGRAMMATIC_TOOL_CALLER_ALLOWED_TOOLS", "").strip()
    if env_allowed:
        allowed_list = [t.strip() for t in env_allowed.split(",") if t.strip()]
        return all_tools & set(allowed_list)

    return all_tools


def _build_namespace(available_tools: set[str], agent: Any) -> Dict[str, Any]:
    """Build the execution namespace.

    Base namespace matches python_repl: ``{"__name__": "__main__"}``.
    ``asyncio`` is always injected (required for async tool wrappers).
    Additional stdlib modules can be injected via the
    ``PROGRAMMATIC_TOOL_CALLER_EXTRA_MODULES`` environment variable
    (comma-separated module names, e.g. ``json,re,math,collections``).
    Tool functions are injected as async callables.

    Returns:
        Namespace dict ready for ``exec()``.
    """
    # Base namespace — matches python_repl
    namespace: Dict[str, Any] = {
        "__name__": "__main__",
    }

    # asyncio is always required (async wrapper)
    namespace["asyncio"] = asyncio

    # Extra modules from env var
    extra_modules = os.environ.get("PROGRAMMATIC_TOOL_CALLER_EXTRA_MODULES", "").strip()
    if extra_modules:
        for mod_name in extra_modules.split(","):
            mod_name = mod_name.strip()
            if not mod_name:
                continue
            try:
                namespace[mod_name] = importlib.import_module(mod_name)
            except ImportError:
                logger.warning(f"Could not import extra module '{mod_name}', skipping")

    # Inject tools as async functions
    for tool_name in available_tools:
        namespace[tool_name] = _create_async_tool_function(agent, tool_name)

    return namespace


# =============================================================================
# Main Tool
# =============================================================================


@tool(context=True)
def programmatic_tool_caller(
    code: str,
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """Execute Python code with access to agent tools as async functions.

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
        PROGRAMMATIC_TOOL_CALLER_EXTRA_MODULES: Comma-separated list of extra modules
            to inject into the namespace (e.g., "json,re,math")
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

        # User confirmation
        if not bypass_consent:
            user_input = get_user_input("<yellow><bold>Execute this code?</bold> [y/*]</yellow>")
            if user_input.lower().strip() != "y":
                cancel_reason = user_input if user_input.strip() != "n" else get_user_input("Reason:")
                return {
                    "status": "error",
                    "content": [{"text": f"Cancelled. Reason: {cancel_reason}"}],
                }

        # Build execution namespace (matches python_repl base + tools)
        exec_namespace = _build_namespace(available_tools, agent)

        console.print("[green]Executing...[/]")

        # Execute code in async context
        # Wrap user code in async function for await support
        indented_code = textwrap.indent(code, "    ")
        wrapped_code = f"async def __user_code__():\n{indented_code}\n"

        # Capture output
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Use compile() for better error tracebacks
            compiled = compile(wrapped_code, "<programmatic_tool_caller>", "exec")
            exec(compiled, exec_namespace)
            asyncio.run(exec_namespace["__user_code__"]())

            captured_output = stdout_capture.getvalue()
            errors = stderr_capture.getvalue()
            if errors:
                captured_output += f"\n[stderr]\n{errors}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

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
