"""
Execute Python code in a REPL environment with PTY support and state persistence.

This module provides a tool for running Python code through a Strands Agent, with features like:
- Persistent state between executions
- Interactive PTY support for real-time feedback (Unix) or subprocess (Windows)
- Output capturing and formatting
- Error handling and logging
- State reset capabilities
- User confirmation for code execution
- Cross-platform support (Windows, Linux, macOS)

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import python_repl

# Register the python_repl tool with the agent
agent = Agent(tools=[python_repl])

# Execute Python code
result = agent.tool.python_repl(code="print('Hello, world!')")

# Execute with state persistence (variables remain available between calls)
agent.tool.python_repl(code="x = 10")
agent.tool.python_repl(code="print(x * 2)")  # Will print: 20

# Use interactive mode (default is True)
agent.tool.python_repl(code="input('Enter your name: ')", interactive=True)

# Reset the REPL state if needed
agent.tool.python_repl(code="print('Fresh start')", reset_state=True)
```
"""

import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import traceback
import types
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import dill
from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from strands.types.tools import ToolResult, ToolUse

from strands_tools.utils import console_util
from strands_tools.utils.user_input import get_user_input

# Platform-specific imports
IS_WINDOWS = platform.system() == "Windows"
IS_POSIX = not IS_WINDOWS

if IS_POSIX:
    import fcntl
    import pty
    import select
    import signal
    import struct
    import termios

# Initialize logging and set paths
logger = logging.getLogger(__name__)

# Tool specification
TOOL_SPEC = {
    "name": "python_repl",
    "description": "Execute Python code in a REPL environment with interactive support and state persistence.\n\n"
    "IMPORTANT SAFETY FEATURES:\n"
    "1. User Confirmation: Requires explicit approval before executing code\n"
    "2. Code Preview: Shows syntax-highlighted code before execution\n"
    "3. State Management: Maintains variables between executions, default controlled by PYTHON_REPL_RESET_STATE\n"
    "4. Error Handling: Captures and formats errors with suggestions\n"
    "5. Development Mode: Can bypass confirmation in BYPASS_TOOL_CONSENT environments\n"
    "6. Interactive Control: Can enable/disable interactive mode in PYTHON_REPL_INTERACTIVE environments\n"
    "7. Cross-Platform: Works on Windows, Linux, and macOS\n\n"
    "Key Features:\n"
    "- Persistent state between executions\n"
    "- Interactive support for real-time feedback\n"
    "- Output capturing and formatting\n"
    "- Error handling and logging\n"
    "- State reset capabilities\n\n"
    "Example Usage:\n"
    "1. Basic execution: code=\"print('Hello, world!')\"\n"
    '2. With state: First call code="x = 10", then code="print(x * 2)"\n'
    "3. Reset state: code=\"print('Fresh start')\", reset_state=True",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The Python code to execute"},
                "interactive": {
                    "type": "boolean",
                    "description": (
                        "Whether to enable interactive mode. "
                        "Default controlled by PYTHON_REPL_INTERACTIVE environment variable."
                    ),
                    "default": True,
                },
                "reset_state": {
                    "type": "boolean",
                    "description": (
                        "Whether to reset the REPL state before execution. "
                        "Default controlled by PYTHON_REPL_RESET_STATE environment variable."
                    ),
                    "default": False,
                },
            },
            "required": ["code"],
        }
    },
}


class OutputCapture:
    """Captures stdout and stderr output."""

    def __init__(self) -> None:
        self.stdout = StringIO()
        self.stderr = StringIO()
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def __enter__(self) -> "OutputCapture":
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def get_output(self) -> str:
        """Get captured output from both stdout and stderr."""
        output = self.stdout.getvalue()
        errors = self.stderr.getvalue()
        if errors:
            output += f"\nErrors:\n{errors}"
        return output


class ReplState:
    """Manages persistent Python REPL state."""

    def __init__(self) -> None:
        # Initialize namespace
        self._namespace = {
            "__name__": "__main__",
        }
        # Check if persistence directory path is defined in env variable
        if "PYTHON_REPL_PERSISTENCE_DIR" in os.environ:
            dir_path = os.environ.get("PYTHON_REPL_PERSISTENCE_DIR")
            # Test directory for validation and security
            try:
                path = Path(dir_path).resolve()

                # Check if path exists
                if not path.exists():
                    raise ValueError(f"Directory does not exist: {path}")

                # Check if directory or file
                if not path.is_dir():
                    raise ValueError(f"Path exists but is not a directory: {path}")

                # Check if directory is writable
                if not os.access(path, os.W_OK):
                    raise PermissionError(f"Directory is not writable: {path}")

                # If all validations pass, set path
                self.persistence_dir = os.path.join(path, "repl_state")
                logger.debug(f"Using validated persistence directory: {self.persistence_dir}")

            except Exception as e:
                # If validation fails, use original default path
                logger.warning(f"Invalid path set : {e}. Using default path")
                self.persistence_dir = os.path.join(Path.cwd(), "repl_state")
        else:
            self.persistence_dir = os.path.join(Path.cwd(), "repl_state")
        os.makedirs(self.persistence_dir, exist_ok=True)
        self.state_file = os.path.join(self.persistence_dir, "repl_state.pkl")
        self.load_state()

    def load_state(self) -> None:
        """Load persisted state with reset on failure."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "rb") as f:
                    saved_state = dill.load(f)
                self._namespace.update(saved_state)
                logger.debug("Successfully loaded REPL state")
            except Exception as e:
                # On error, remove the corrupted state file
                logger.debug(f"Error loading state: {e}. Removing corrupted state file.")
                try:
                    os.remove(self.state_file)
                    logger.debug("Removed corrupted state file")
                except Exception as remove_error:
                    logger.debug(f"Error removing state file: {remove_error}")

                # Initialize fresh state
                logger.debug("Initializing fresh REPL state")

    def save_state(self, code: Optional[str] = None) -> None:
        """Save current state."""
        try:
            # Execute new code if provided
            if code:
                exec(code, self._namespace)

            # Filter namespace for persistence
            save_dict = {}
            for name, value in self._namespace.items():
                if not name.startswith("_"):
                    try:
                        # Try to pickle the value
                        dill.dumps(value)
                        save_dict[name] = value
                    except BaseException:
                        continue

            # Save state
            with open(self.state_file, "wb") as f:
                dill.dump(save_dict, f)
            logger.debug("Successfully saved REPL state")

        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def execute(self, code: str) -> None:
        """Execute code and save state."""
        exec(code, self._namespace)
        self.save_state()

    def get_namespace(self) -> dict:
        """Get current namespace."""
        return dict(self._namespace)

    def clear_state(self) -> None:
        """Clear the current state and remove state file."""
        try:
            # Clear namespace to defaults
            self._namespace = {
                "__name__": "__main__",
            }

            # Remove state file if it exists
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
                logger.info("REPL state cleared and file removed")

            # Save fresh state
            self.save_state()

        except Exception as e:
            logger.error(f"Error clearing state: {e}")

    def get_user_objects(self) -> Dict[str, str]:
        """Get user-defined objects for display."""
        objects = {}
        for name, value in self._namespace.items():
            # Skip special/internal objects
            if name.startswith("_"):
                continue

            # Handle each type separately to avoid unreachable code
            if isinstance(value, (int, float, str, bool)):
                objects[name] = repr(value)

        return objects


# Create global state instance
repl_state = ReplState()


def clean_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


class SubprocessExecutor:
    """Cross-platform subprocess-based Python execution with state synchronization."""

    def __init__(self, callback: Optional[Callable] = None):
        self.output_buffer: List[str] = []
        self.callback = callback
        self.process: Optional[subprocess.Popen] = None

    def start(self, code: str) -> int:
        """Start subprocess session with code execution."""
        # Create a temporary script that loads state, executes code, and saves state
        script_code = f"""
import sys
import os
import dill
from pathlib import Path

# Load state
persistence_dir = os.path.join(Path.cwd(), "repl_state")
state_file = os.path.join(persistence_dir, "repl_state.pkl")

namespace = {{"__name__": "__main__"}}

if os.path.exists(state_file):
    try:
        with open(state_file, "rb") as f:
            saved_state = dill.load(f)
        namespace.update(saved_state)
    except Exception:
        pass

# Execute user code
try:
    exec('''
{code}
''', namespace)
    
    # Save state
    save_dict = {{}}
    for name, value in namespace.items():
        if not name.startswith("_"):
            try:
                dill.dumps(value)
                save_dict[name] = value
            except:
                continue
    
    with open(state_file, "wb") as f:
        dill.dump(save_dict, f)
    
    sys.exit(0)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""

        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            script_path = f.name
            f.write(script_code)

        try:
            # Execute the script
            self.process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )

            # Read output in separate thread
            output_thread = threading.Thread(target=self._read_output)
            output_thread.daemon = True
            output_thread.start()

            # Wait for completion
            exit_code = self.process.wait()

            # Wait for output thread to finish
            output_thread.join(timeout=1.0)

            return exit_code

        finally:
            # Clean up temporary file
            try:
                os.unlink(script_path)
            except Exception:
                pass

    def _read_output(self) -> None:
        """Read and process subprocess output."""
        if not self.process or not self.process.stdout:
            return

        try:
            for line in iter(self.process.stdout.readline, ''):
                if not line:
                    break
                
                cleaned = clean_ansi(line)
                self.output_buffer.append(cleaned)

                # Stream if callback exists
                if self.callback:
                    try:
                        self.callback(cleaned)
                    except Exception as callback_error:
                        logger.warning(f"Error in output callback: {callback_error}")

        except Exception as e:
            logger.warning(f"Error reading subprocess output: {e}")

    def get_output(self) -> str:
        """Get complete output with ANSI codes removed and binary content truncated."""
        raw = "".join(self.output_buffer)
        clean = clean_ansi(raw)

        # Handle binary content
        def format_binary(text: str, max_len: int = None) -> str:
            if max_len is None:
                max_len = int(os.environ.get("PYTHON_REPL_BINARY_MAX_LEN", "100"))
            if "\\x" in text and len(text) > max_len:
                return f"{text[:max_len]}... [binary content truncated]"
            return text

        return format_binary(clean)

    def stop(self) -> None:
        """Stop subprocess and clean up resources."""
        if self.process:
            try:
                if self.process.poll() is None:  # Process still running
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
            except Exception as e:
                logger.debug(f"Error stopping subprocess: {e}")


if IS_POSIX:
    class PtyManager:
        """Manages PTY-based Python execution with state synchronization (Unix only)."""

        def __init__(self, callback: Optional[Callable] = None):
            self.supervisor_fd = -1
            self.worker_fd = -1
            self.pid = -1
            self.output_buffer: List[str] = []
            self.input_buffer: List[str] = []
            self.stop_event = threading.Event()
            self.callback = callback

        def start(self, code: str) -> None:
            """Start PTY session with code execution."""
            # Create PTY
            self.supervisor_fd, self.worker_fd = pty.openpty()

            # Set terminal size
            term_size = struct.pack("HHHH", 24, 80, 0, 0)
            fcntl.ioctl(self.worker_fd, termios.TIOCSWINSZ, term_size)

            # Fork process
            self.pid = os.fork()

            if self.pid == 0:  # Child process
                try:
                    # Setup PTY
                    os.close(self.supervisor_fd)
                    os.dup2(self.worker_fd, 0)
                    os.dup2(self.worker_fd, 1)
                    os.dup2(self.worker_fd, 2)

                    # Execute in REPL namespace
                    namespace = repl_state.get_namespace()
                    exec(code, namespace)

                    os._exit(0)

                except Exception:
                    traceback.print_exc(file=sys.stderr)
                    os._exit(1)

            else:  # Parent process
                os.close(self.worker_fd)

                # Start output reader
                reader = threading.Thread(target=self._read_output)
                reader.daemon = True
                reader.start()

                # Start input handler
                input_handler = threading.Thread(target=self._handle_input)
                input_handler.daemon = True
                input_handler.start()

        def _read_output(self) -> None:
            """Read and process PTY output with improved error handling."""
            buffer = ""
            incomplete_bytes = b""

            while not self.stop_event.is_set():
                try:
                    if self.supervisor_fd < 0:
                        logger.debug("Invalid file descriptor, stopping output reader")
                        break

                    try:
                        r, _, _ = select.select([self.supervisor_fd], [], [], 0.1)
                    except (OSError, ValueError) as e:
                        logger.debug(f"File descriptor error during select: {e}")
                        break

                    if self.supervisor_fd in r:
                        try:
                            raw_data = os.read(self.supervisor_fd, 1024)
                        except (OSError, ValueError) as e:
                            if hasattr(e, 'errno') and e.errno == 9:
                                logger.debug("PTY closed, stopping output reader")
                            else:
                                logger.warning(f"Error reading from PTY: {e}")
                            break

                        if not raw_data:
                            logger.debug("EOF reached, PTY closed")
                            break

                        full_data = incomplete_bytes + raw_data

                        try:
                            data = full_data.decode("utf-8")
                            incomplete_bytes = b""

                        except UnicodeDecodeError as e:
                            if e.start > 0:
                                data = full_data[: e.start].decode("utf-8")
                                incomplete_bytes = full_data[e.start :]
                            else:
                                incomplete_bytes = full_data
                                continue

                        if data:
                            buffer += data

                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                cleaned = clean_ansi(line + "\n")
                                self.output_buffer.append(cleaned)

                                if self.callback:
                                    try:
                                        self.callback(cleaned)
                                    except Exception as callback_error:
                                        logger.warning(f"Error in output callback: {callback_error}")

                            if buffer:
                                cleaned = clean_ansi(buffer)
                                if self.callback:
                                    try:
                                        self.callback(cleaned)
                                    except Exception as callback_error:
                                        logger.warning(f"Error in output callback: {callback_error}")

                except (OSError, IOError) as e:
                    if hasattr(e, "errno") and e.errno == 9:
                        logger.debug("PTY file descriptor closed, stopping reader")
                        break
                    else:
                        logger.warning(f"I/O error reading PTY output: {e}")
                        continue

                except Exception as e:
                    logger.error(f"Unexpected error in _read_output: {e}")
                    break

            # Handle remaining buffer
            if buffer:
                try:
                    cleaned = clean_ansi(buffer)
                    self.output_buffer.append(cleaned)
                    if self.callback:
                        self.callback(cleaned)
                except Exception as e:
                    logger.warning(f"Error processing final buffer: {e}")

            if incomplete_bytes:
                try:
                    final_data = incomplete_bytes.decode("utf-8", errors="replace")
                    if final_data:
                        cleaned = clean_ansi(final_data)
                        self.output_buffer.append(cleaned)
                        if self.callback:
                            self.callback(cleaned)
                except Exception as e:
                    logger.warning(f"Failed to process remaining bytes: {e}")

            logger.debug("PTY output reader thread finished")

        def _handle_input(self) -> None:
            """Handle interactive user input."""
            while not self.stop_event.is_set():
                try:
                    r, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if sys.stdin in r:
                        input_data = ""
                        while True:
                            char = sys.stdin.read(1)
                            if not char or char == "\n":
                                input_data += "\n"
                                break
                            input_data += char

                        if input_data:
                            if input_data not in self.input_buffer:
                                self.input_buffer.append(input_data)
                                os.write(self.supervisor_fd, input_data.encode())

                except (OSError, IOError):
                    break

        def get_output(self) -> str:
            """Get complete output with ANSI codes removed."""
            raw = "".join(self.output_buffer)
            clean = clean_ansi(raw)

            def format_binary(text: str, max_len: int = None) -> str:
                if max_len is None:
                    max_len = int(os.environ.get("PYTHON_REPL_BINARY_MAX_LEN", "100"))
                if "\\x" in text and len(text) > max_len:
                    return f"{text[:max_len]}... [binary content truncated]"
                return text

            return format_binary(clean)

        def stop(self) -> None:
            """Stop PTY session and clean up resources."""
            logger.debug("Stopping PTY session...")
            self.stop_event.set()

            if self.pid > 0:
                try:
                    os.kill(self.pid, signal.SIGTERM)

                    try:
                        pid, status = os.waitpid(self.pid, os.WNOHANG)
                        if pid == 0:
                            import time
                            time.sleep(0.1)
                            pid, status = os.waitpid(self.pid, os.WNOHANG)
                            if pid == 0:
                                logger.debug("Forcing process termination")
                                os.kill(self.pid, signal.SIGKILL)
                                os.waitpid(self.pid, 0)

                    except OSError as e:
                        logger.debug(f"Process cleanup error: {e}")

                except (OSError, ProcessLookupError) as e:
                    logger.debug(f"Process termination error: {e}")

                finally:
                    self.pid = -1

            if self.supervisor_fd >= 0:
                try:
                    os.close(self.supervisor_fd)
                    logger.debug("PTY supervisor file descriptor closed")
                except OSError as e:
                    logger.debug(f"Error closing supervisor fd: {e}")
                finally:
                    self.supervisor_fd = -1

            logger.debug("PTY session cleanup completed")


def python_repl(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Execute Python code with persistent state and output streaming."""
    console = console_util.create()

    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    code = tool_input["code"]
    interactive = os.environ.get("PYTHON_REPL_INTERACTIVE", str(tool_input.get("interactive", True))).lower() == "true"
    reset_state = os.environ.get("PYTHON_REPL_RESET_STATE", str(tool_input.get("reset_state", False))).lower() == "true"

    # Check for development mode
    strands_dev = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

    # Check for non_interactive_mode parameter
    non_interactive_mode = kwargs.get("non_interactive_mode", False)

    try:
        # Handle state reset if requested
        if reset_state:
            console.print("[yellow]Resetting REPL state...[/]")
            repl_state.clear_state()
            console.print("[green]REPL state reset complete[/]")

        # Show code preview
        console.print(
            Panel(
                Syntax(code, "python", theme="monokai"),
                title="[bold blue]Executing Python Code[/]",
            )
        )

        # Add permissions check
        if not strands_dev and not non_interactive_mode:
            details_table = Table(show_header=False, box=box.SIMPLE)
            details_table.add_column("Property", style="cyan", justify="right")
            details_table.add_column("Value", style="green")

            details_table.add_row("Code Length", f"{len(code)} characters")
            details_table.add_row("Line Count", f"{len(code.splitlines())} lines")
            details_table.add_row("Mode", "Interactive" if interactive else "Standard")
            details_table.add_row("Reset State", "Yes" if reset_state else "No")
            details_table.add_row("Platform", platform.system())

            console.print(
                Panel(
                    details_table,
                    title="[bold blue]🐍 Python Code Execution Preview",
                    border_style="blue",
                    box=box.ROUNDED,
                )
            )
            
            user_input = get_user_input(
                "<yellow><bold>Do you want to proceed with Python code execution?</bold> [y/*]</yellow>"
            )
            if user_input.lower().strip() != "y":
                cancellation_reason = (
                    user_input
                    if user_input.strip() != "n"
                    else get_user_input("Please provide a reason for cancellation:")
                )
                error_message = f"Python code execution cancelled by the user. Reason: {cancellation_reason}"
                error_panel = Panel(
                    f"[bold blue]{error_message}[/bold blue]",
                    title="[bold blue]❌ Cancelled",
                    border_style="blue",
                    box=box.ROUNDED,
                )
                console.print(error_panel)
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": error_message}],
                }

        # Track execution time and capture output
        start_time = datetime.now()
        output = None
        exit_status = 0

        try:
            # On Windows or when PTY is not available, use subprocess
            if IS_WINDOWS or not interactive:
                if interactive:
                    console.print("[green]Running in interactive mode (subprocess)...[/]")
                else:
                    console.print("[blue]Running in standard mode...[/]")
                
                if interactive:
                    # Use subprocess executor for interactive mode on Windows
                    executor = SubprocessExecutor()
                    exit_status = executor.start(code)
                    output = executor.get_output()
                    executor.stop()
                else:
                    # Use direct execution for non-interactive mode
                    captured = OutputCapture()
                    with captured as output_capture:
                        repl_state.execute(code)
                        output = output_capture.get_output()
                        if output:
                            console.print("[cyan]Output:[/]")
                            console.print(output)

            # On Unix systems, use PTY for better interactive support
            elif IS_POSIX and interactive:
                console.print("[green]Running in interactive mode (PTY)...[/]")
                pty_mgr = PtyManager()
                pty_mgr.start(code)

                # Wait for completion
                while True:
                    try:
                        pid, exit_status = os.waitpid(pty_mgr.pid, os.WNOHANG)
                        if pid != 0:
                            break
                    except OSError:
                        break

                # Get output and clean up
                output = pty_mgr.get_output()
                pty_mgr.stop()

                # Save state if execution succeeded
                if exit_status == 0:
                    repl_state.save_state(code)

            # Show execution stats
            duration = (datetime.now() - start_time).total_seconds()
            user_objects = repl_state.get_user_objects()

            status = f"✓ Code executed successfully ({duration:.2f}s)"
            if user_objects:
                status += f"\nUser objects in namespace: {len(user_objects)} items"
                for name, value in user_objects.items():
                    status += f"\n - {name} = {value}"
            console.print(f"[bold green]{status}[/]")

            # Return result with output
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": output if output else "Code executed successfully"}],
            }

        except RecursionError:
            console.print("[yellow]Recursion error detected - resetting state...[/]")
            repl_state.clear_state()
            raise

    except Exception as e:
        error_tb = traceback.format_exc()
        error_time = datetime.now()

        console.print(
            Panel(
                Syntax(error_tb, "python", theme="monokai"),
                title="[bold red]Python Error[/]",
                border_style="red",
            )
        )

        # Log error with details
        errors_dir = os.path.join(Path.cwd(), "errors")
        os.makedirs(errors_dir, exist_ok=True)
        error_file = os.path.join(errors_dir, "errors.txt")

        error_msg = f"\n[{error_time.isoformat()}] Python REPL Error:\nCode:\n{code}\nError:\n{error_tb}\n"

        with open(error_file, "a") as f:
            f.write(error_msg)
        logger.debug(error_msg)

        # If it's a recursion error, suggest resetting state
        suggestion = ""
        if isinstance(e, RecursionError):
            suggestion = "\nTo fix this, try running with reset_state=True"

        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"{error_msg}{suggestion}"}],
        }