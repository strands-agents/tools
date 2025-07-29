"""
Interactive shell tool with PTY support for real-time command execution and interaction.

This module provides a powerful shell interface for executing commands through a Strands Agent.
It supports various execution modes, including sequential and parallel command execution,
directory operations, and interactive PTY support for real-time feedback.

Features:
- Multiple command formats (string, array, or detailed objects)
- Sequential or parallel execution
- Real-time interactive terminal emulation
- Error handling and timeout control
- Working directory specification

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import shell

# Register the shell tool with the agent
agent = Agent(tools=[shell])

# Execute a single command
result = agent.tool.shell(command="ls -la")

# Execute multiple commands sequentially
result = agent.tool.shell(command=["cd /path", "ls -la", "pwd"])

# Execute with specific working directory
result = agent.tool.shell(command="npm install", work_dir="/app/path")

# Execute commands with custom timeout and error handling
result = agent.tool.shell(
    command=[{"command": "git clone https://github.com/example/repo", "timeout": 60}],
    ignore_errors=True
)

# Execute commands in parallel
result = agent.tool.shell(command=["task1", "task2"], parallel=True)
```

Configuration:
- STRANDS_NON_INTERACTIVE (environment variable): Set to "true" to run the tool
  in a non-interactive mode, suppressing all user prompts for confirmation.
- BYPASS_TOOL_CONSENT (environment variable): Set to "true" to bypass only the
  user confirmation prompt, even in an otherwise interactive session.

"""

import json
import logging
import os
import pty
import queue
import select
import signal
import sys
import termios
import time
import tty
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Literal, Tuple, Union

from rich import box
from rich.box import ROUNDED
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from strands import tool

from strands_tools.utils import console_util
from strands_tools.utils.user_input import get_user_input

# Initialize logging
logger = logging.getLogger(__name__)


def read_output(fd: int) -> str:
    """Read output from fd, handling both UTF-8 and other encodings."""
    try:
        data = os.read(fd, 1024)
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")
    except OSError:
        return ""


def validate_command(command: Union[str, Dict]) -> Tuple[str, Dict]:
    """Validate and normalize command input."""
    if isinstance(command, str):
        return command, {}
    elif isinstance(command, dict):
        cmd = command.get("command")
        if not cmd or not isinstance(cmd, str):
            raise ValueError("Command object must contain a 'command' string")
        return cmd, command
    else:
        raise ValueError("Command must be string or dict")


class CommandExecutor:
    """Handles execution of shell commands with timeout."""

    def __init__(self, timeout: int = None) -> None:
        self.timeout = int(os.environ.get("SHELL_DEFAULT_TIMEOUT", "900")) if timeout is None else timeout
        self.output_queue: queue.Queue = queue.Queue()
        self.exit_code = None
        self.error = None

    def execute_with_pty(self, command: str, cwd: str, non_interactive_mode: bool) -> Tuple[int, str, str]:
        """Execute command with PTY and timeout support."""
        output = []
        start_time = time.time()
        old_tty = None
        pid = -1
        # Save original terminal settings
        if not non_interactive_mode:
            try:
                old_tty = termios.tcgetattr(sys.stdin)
            except BaseException:
                non_interactive_mode = True
        try:
            # Fork a new PTY
            pid, fd = pty.fork()

            if pid == 0:  # Child process
                try:
                    os.chdir(cwd)
                    os.execvp("/bin/sh", ["/bin/sh", "-c", command])
                except Exception as e:
                    logger.debug(f"Error in child: {e}")
                    sys.exit(1)
            else:  # Parent process
                if not non_interactive_mode and old_tty:
                    tty.setraw(sys.stdin.fileno())
                while True:
                    if time.time() - start_time > self.timeout:
                        try:
                            # This kill entire group, not just parent shell.
                            os.killpg(os.getpgid(pid), signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        raise TimeoutError(f"Command timed out after {self.timeout} seconds")

                    fds_to_watch = [fd]
                    if not non_interactive_mode:
                        fds_to_watch.append(sys.stdin)

                    try:
                        readable, _, _ = select.select(fds_to_watch, [], [], 0.1)
                    except (select.error, ValueError):
                        logger.debug("select() failed, assuming process ended.")
                        break

                    if fd in readable:
                        try:
                            data = read_output(fd)
                            if not data:
                                break
                            output.append(data)
                            sys.stdout.write(data)
                            sys.stdout.flush()
                        except OSError:
                            break

                    # Handle interactive input from user
                    if not non_interactive_mode and sys.stdin in readable:
                        try:
                            stdin_data = os.read(sys.stdin.fileno(), 1024)
                            os.write(fd, stdin_data)
                        except OSError:
                            break
                try:
                    _, status = os.waitpid(pid, 0)
                    if os.WIFEXITED(status):
                        exit_code = os.WEXITSTATUS(status)
                    else:
                        exit_code = -1  # Process was terminated by a signal
                except OSError:
                    exit_code = -1  # waitpid failed

                # In non_interactive_mode, we should not print the live output to the console.
                # The captured output is returned for the agent to process.
                return exit_code, "".join(output), ""

        finally:
            # Restore terminal settings only if they were saved and changed.
            if not non_interactive_mode and old_tty:
                termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, old_tty)


def execute_single_command(
    command: Union[str, Dict], work_dir: str, timeout: int, non_interactive_mode: bool
) -> Dict[str, Any]:
    """Execute a single command and return its results."""
    cmd_str, cmd_opts = validate_command(command)
    executor = CommandExecutor(timeout=timeout)

    try:
        exit_code, output, error = executor.execute_with_pty(
            cmd_str, work_dir, non_interactive_mode=non_interactive_mode
        )

        result = {
            "command": cmd_str,
            "exit_code": exit_code,
            "output": output,
            "error": error,
            "status": "success" if exit_code == 0 else "error",
        }

        if cmd_opts:
            result["options"] = cmd_opts

        return result

    except Exception as e:
        return {
            "command": cmd_str,
            "exit_code": 1,
            "output": "",
            "error": str(e),
            "status": "error",
        }


class CommandContext:
    """Maintains command execution context including working directory."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = os.path.abspath(base_dir)
        self.current_dir = self.base_dir
        self._dir_stack: List[str] = []

    def push_dir(self) -> None:
        """Save current directory to stack."""
        self._dir_stack.append(self.current_dir)

    def pop_dir(self) -> None:
        """Restore previous directory from stack."""
        if self._dir_stack:
            self.current_dir = self._dir_stack.pop()

    def update_dir(self, command: str) -> None:
        """Update current directory based on cd command."""
        if command.strip().startswith("cd "):
            new_dir = command.split("cd ", 1)[1].strip()
            if new_dir.startswith("/"):
                # Absolute path
                self.current_dir = os.path.abspath(new_dir)
            else:
                # Relative path
                self.current_dir = os.path.abspath(os.path.join(self.current_dir, new_dir))


def execute_commands(
    commands: List[Union[str, Dict]],
    parallel: bool,
    ignore_errors: bool,
    work_dir: str,
    timeout: int,
    non_interactive_mode: bool,
) -> List[Dict[str, Any]]:
    """Execute multiple commands either sequentially or in parallel."""
    results = []
    context = CommandContext(work_dir)

    if parallel:
        # For parallel execution, use the initial work_dir for all commands
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(execute_single_command, cmd, work_dir, timeout, non_interactive_mode)
                for cmd in commands
            ]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if not ignore_errors and result["status"] == "error":
                    # Cancel remaining futures if error handling is strict
                    for f in futures:
                        f.cancel()
                    break
    else:
        # For sequential execution, maintain directory context
        for cmd in commands:
            cmd_str = cmd if isinstance(cmd, str) else cmd.get("command", "")

            # Execute in current context directory
            result = execute_single_command(
                cmd, context.current_dir, timeout, non_interactive_mode=non_interactive_mode
            )
            results.append(result)

            # Update context if command was successful
            if result["status"] == "success":
                context.update_dir(cmd_str)

            if not ignore_errors and result["status"] == "error":
                break

    return results


def normalize_commands(
    command: Union[str, List[Union[str, Dict[Any, Any]]], Dict[Any, Any]],
) -> List[Union[str, Dict]]:
    """Convert command input into a normalized list of commands."""
    if isinstance(command, list):
        return command
    return [command]


def format_command_preview(command: Union[str, Dict], parallel: bool, ignore_errors: bool, work_dir: str) -> Panel:
    """Create rich preview panel for command execution."""
    details = Table(show_header=False, box=box.SIMPLE)
    details.add_column("Property", style="cyan", justify="right")
    details.add_column("Value", style="green")

    # Format command info
    cmd_str = command if isinstance(command, str) else command.get("command", "")
    details.add_row("🔷 Command", Syntax(cmd_str, "bash", theme="monokai", line_numbers=False))
    details.add_row("📁 Working Dir", work_dir)
    details.add_row("⚡ Parallel Mode", "✓ Yes" if parallel else "✗ No")
    details.add_row("🛡️ Ignore Errors", "✓ Yes" if ignore_errors else "✗ No")

    return Panel(
        details,
        title="[bold blue]🚀 Command Execution Preview",
        border_style="blue",
        box=ROUNDED,
    )


def format_execution_result(result: Dict[str, Any]) -> Panel:
    """Format command execution result as a rich panel."""
    result_table = Table(show_header=False, box=box.SIMPLE)
    result_table.add_column("Property", style="cyan", justify="right")
    result_table.add_column("Value")

    # Status with appropriate styling
    status_style = "green" if result["status"] == "success" else "red"
    status_icon = "✓" if result["status"] == "success" else "✗"

    result_table.add_row(
        "Status",
        f"[{status_style}]{status_icon} {result['status'].capitalize()}[/{status_style}]",
    )
    result_table.add_row("Exit Code", f"{result['exit_code']}")

    # Add command with syntax highlighting
    result_table.add_row(
        "Command",
        Syntax(result["command"], "bash", theme="monokai", line_numbers=False),
    )

    # Output (truncate if too long)
    output = result["output"]
    if len(output) > 500:
        output = output[:500] + "...\n[dim](output truncated)[/dim]"
    result_table.add_row("Output", output)

    # Error (if any)
    if result["error"]:
        result_table.add_row("Error", f"[red]{result['error']}[/red]")

    border_style = "green" if result["status"] == "success" else "red"
    icon = "🟢" if result["status"] == "success" else "🔴"

    return Panel(
        result_table,
        title=f"[bold {border_style}]{icon} Command Result",
        border_style=border_style,
        box=ROUNDED,
    )


def format_summary(results: List[Dict[str, Any]], parallel: bool) -> Panel:
    """Format execution summary as a rich panel."""
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = len(results) - success_count

    summary_table = Table(show_header=False, box=box.SIMPLE)
    summary_table.add_column("Property", style="cyan", justify="right")
    summary_table.add_column("Value")

    summary_table.add_row("Total Commands", f"{len(results)}")
    summary_table.add_row("Successful", f"[green]{success_count}[/green]")
    summary_table.add_row("Failed", f"[red]{error_count}[/red]")
    summary_table.add_row("Execution Mode", "Parallel" if parallel else "Sequential")

    status = "success" if error_count == 0 else "warning" if error_count < len(results) else "error"
    icons = {"success": "✅", "warning": "⚠️", "error": "❌"}
    colors = {"success": "green", "warning": "yellow", "error": "red"}

    return Panel(
        summary_table,
        title=f"[bold {colors[status]}]{icons[status]} Execution Summary",
        border_style=colors[status],
        box=ROUNDED,
    )


@tool
def shell(
    command: Union[str, List[Union[str, Dict[str, Any]]]],
    parallel: bool = False,
    ignore_errors: bool = False,
    timeout: int = None,
    work_dir: str = None,
    non_interactive: bool = False,
) -> Dict[str, Any]:
    """Interactive shell with PTY support for real-time command execution and interaction. Features:

    1. Command Formats:
       • Single Command (string):
         command: "ls -la"

       • Multiple Commands (array):
         command: ["cd /path", "git status"]

       • Detailed Command Objects:
         command: [{
           "command": "git clone repo",
           "timeout": 60,
           "work_dir": "/specific/path"
         }]

    2. Execution Modes:
       • Sequential (default): Commands run in order
       • Parallel: Multiple commands execute simultaneously
       • Error Handling: Stop on error or continue with ignore_errors

    3. Real-time Features:
       • Live Output: See command output as it happens
       • Interactive Input: Send input to running commands
       • PTY Support: Full terminal emulation
       • Timeout Control: Prevent hanging commands

    4. Common Patterns:
       • Directory Operations:
         command: ["mkdir -p dir", "cd dir", "git init"]
       • Git Operations:
         command: {"command": "git pull", "work_dir": "/repo/path"}
       • Build Commands:
         command: "npm install", work_dir: "/app/path"

    5. Best Practices:
       • Use arrays for multiple commands
       • Set appropriate timeouts
       • Specify work_dir when needed
       • Enable ignore_errors for resilient scripts
       • Use parallel execution for independent commands

    Example Usage:
    1. Simple command:
       {"command": "ls -la"}

    2. Multiple commands:
       {"command": ["mkdir test", "cd test", "touch file.txt"]}

    3. Parallel execution:
       {"command": ["task1", "task2"], "parallel": true}

    4. With error handling:
       {"command": ["risky-command"], "ignore_errors": true}

    5. Custom directory:
       {"command": "npm install", "work_dir": "/app/path"}

    Args:
        command: The shell command(s) to execute interactively. Can be a single command string or array of commands
        parallel: Whether to execute multiple commands in parallel (default: False)
        ignore_errors: Continue execution even if some commands fail (default: False)
        timeout: Timeout in seconds for each command (default: controlled by SHELL_DEFAULT_TIMEOUT environment variable)
        work_dir: Working directory for command execution (default: current)
        non_interactive: Run in non-interactive mode without user prompts (default: False)

    Returns:
        Dict containing status and response content
    """
    console = console_util.create()

    is_strands_non_interactive = os.environ.get("STRANDS_NON_INTERACTIVE", "").lower() == "true"
    # Here we keep both doors open, but we only prompt env STRANDS_NON_INTERACTIVE in our doc.
    non_interactive_mode = is_strands_non_interactive or non_interactive

    # Validate command parameter
    if command is None:
        return {
            "status": "error",
            "content": [{"text": "Command is required"}],
        }

    # Fix for array input: if the command is a string that looks like JSON array, parse it
    if isinstance(command, str) and command.strip().startswith("[") and command.strip().endswith("]"):
        try:
            command = json.loads(command)
        except json.JSONDecodeError:
            # If it fails to parse, keep it as a string
            pass

    commands = normalize_commands(command)

    # Set defaults for parameters
    if timeout is None:
        timeout = int(os.environ.get("SHELL_DEFAULT_TIMEOUT", "900"))
    if work_dir is None:
        work_dir = os.getcwd()

    # Development mode check
    STRANDS_BYPASS_TOOL_CONSENT = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

    # Only show UI elements in interactive mode
    if not non_interactive_mode:
        # Show command previews
        console.print("\n[bold blue]Command Execution Plan[/bold blue]\n")

        # Show preview for each command
        for i, cmd in enumerate(commands):
            console.print(format_command_preview(cmd, parallel, ignore_errors, work_dir))

            # Add spacing between multiple commands
            if i < len(commands) - 1:
                console.print()

    if not STRANDS_BYPASS_TOOL_CONSENT and not non_interactive_mode:
        console.print()  # Empty line for spacing
        confirm = get_user_input("<yellow><bold>Do you want to proceed with execution?</bold> [y/*]</yellow>")
        if confirm.lower() != "y":
            console.print(
                Panel(
                    f"[bold blue]Operation cancelled. Reason: {confirm}[/bold blue]",
                    title="[bold blue]❌ Cancelled",
                    border_style="blue",
                    box=ROUNDED,
                )
            )
            return {
                "status": "error",
                "content": [{"text": f"Command execution cancelled by user. Input: {confirm}"}],
            }

    try:
        if not non_interactive_mode:
            console.print("\n[bold green]⏳ Starting Command Execution...[/bold green]\n")

        results = execute_commands(
            commands, parallel, ignore_errors, work_dir, timeout, non_interactive_mode=non_interactive_mode
        )

        if not non_interactive_mode:
            console.print("\n[bold green]✅ Command Execution Complete[/bold green]\n")

            # Display formatted results
            console.print(format_summary(results, parallel))
            console.print()  # Empty line for spacing

            for result in results:
                console.print(format_execution_result(result))
                console.print()  # Empty line for spacing

        # Process results for tool output
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = len(results) - success_count

        content = []
        for result in results:
            content.append(
                {
                    "text": f"Command: {result['command']}\n"
                    f"Status: {result['status']}\n"
                    f"Exit Code: {result['exit_code']}\n"
                    f"Output: {result['output']}\n"
                    f"Error: {result['error']}"
                }
            )

        content.insert(
            0,
            {
                "text": f"Execution Summary:\n"
                f"Total commands: {len(results)}\n"
                f"Successful: {success_count}\n"
                f"Failed: {error_count}"
            },
        )

        status: Literal["success", "error"] = "success" if error_count == 0 or ignore_errors else "error"

        return {"status": status, "content": content}

    except Exception as e:
        if not non_interactive_mode:
            console.print(
                Panel(
                    f"[bold red]Error: {str(e)}[/bold red]",
                    title="[bold red]❌ Execution Failed",
                    border_style="red",
                    box=ROUNDED,
                )
            )
        return {
            "status": "error",
            "content": [{"text": f"Interactive shell error: {str(e)}"}],
        }
