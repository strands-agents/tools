"""
Execute Python code in a REPL environment with PTY support and state persistence.

This module provides a tool for running Python code through a Strands Agent, with features like:
- Persistent state between executions
- Interactive PTY support for real-time feedback
- Output capturing and formatting
- Error handling and logging
- State reset capabilities
- User confirmation for code execution

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

import ast
import fcntl
import logging
import os
import pty
import re
import resource
import select
import signal
import struct
import sys
import termios
import threading
import traceback
import types
from datetime import datetime
from enum import Enum
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

# Initialize logging and set paths
logger = logging.getLogger(__name__)


# Security system components
class SecurityMode(Enum):
    """Security modes for Python REPL execution."""

    NORMAL = "normal"
    RESTRICTED = "restricted"


class SecurityViolation(Exception):
    """Raised when code violates security policies."""

    pass


class ASTSecurityValidator:
    """AST-based security validator for Python code."""

    # Dangerous imports that are blocked in restricted mode
    DANGEROUS_IMPORTS = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "http",
        "ftplib",
        "telnetlib",
        "smtplib",
        "poplib",
        "imaplib",
        "shutil",
        "tempfile",
        "glob",
        "pickle",
        "marshal",
        "shelve",
        "dbm",
        "sqlite3",
        "ctypes",
        "__future__",
    }

    # Dangerous built-in functions
    DANGEROUS_BUILTINS = {"eval", "exec", "compile"}

    # Dangerous attributes
    DANGEROUS_ATTRIBUTES = {
        "__builtins__",
        "__globals__",
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__dict__",
        "__class__",
        "func_globals",
    }

    def __init__(self, memory_limit_mb: int = None, timeout: int = None):
        """Initialize the security validator."""
        # Memory limit in bytes
        self.memory_limit = (memory_limit_mb or int(os.environ.get("PYTHON_REPL_MEMORY_LIMIT_MB", "100"))) * 1024 * 1024

        # Execution timeout in seconds
        self.timeout = timeout or int(os.environ.get("PYTHON_REPL_TIMEOUT", "30"))

        # Setup allowed file paths
        self.allowed_paths = []

        # Add current directory if allowed
        allow_current_dir = os.environ.get("PYTHON_REPL_ALLOW_CURRENT_DIR", "true").lower() in (
            "true",
            "1",
            "yes",
            "on",
        )
        if allow_current_dir:
            self.allowed_paths.append(os.path.abspath(os.getcwd()))

        # Add custom allowed paths
        custom_paths = os.environ.get("PYTHON_REPL_ALLOWED_PATHS", "")
        if custom_paths:
            for path in custom_paths.split(","):
                path = path.strip()
                if path:
                    self.allowed_paths.append(os.path.abspath(path))

    def validate_code(self, code: str) -> None:
        """Validate code against security policies."""
        try:
            tree = ast.parse(code)
            self._validate_ast(tree)
        except SyntaxError as e:
            raise SecurityViolation(f"Syntax error in code: {e}") from e

    def _validate_ast(self, node: ast.AST) -> None:
        """Recursively validate AST nodes."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in self.DANGEROUS_IMPORTS:
                    raise SecurityViolation(f"Import of module '{alias.name}' is blocked")

        elif isinstance(node, ast.ImportFrom):
            if node.module in self.DANGEROUS_IMPORTS:
                raise SecurityViolation(f"Import from module '{node.module}' is blocked")

        elif isinstance(node, ast.Call):
            # Check dangerous function calls
            if isinstance(node.func, ast.Name):
                if node.func.id in self.DANGEROUS_BUILTINS:
                    raise SecurityViolation(f"Call to built-in function '{node.func.id}' is blocked")

                # Check __import__ calls
                if node.func.id == "__import__":
                    if node.args and isinstance(node.args[0], ast.Constant):
                        module_name = node.args[0].value
                        if module_name in self.DANGEROUS_IMPORTS:
                            raise SecurityViolation(f"Import of module '{module_name}' via __import__ is blocked")

                # Check range calls for large ranges
                if node.func.id == "range":
                    if node.args and isinstance(node.args[0], ast.Constant):
                        if isinstance(node.args[0].value, int) and node.args[0].value > 1000000:
                            raise SecurityViolation(f"Large range detected: {node.args[0].value}")

                # Check open() calls with path validation
                if node.func.id == "open" and node.args:
                    if isinstance(node.args[0], ast.Constant):
                        filepath = node.args[0].value
                        self._validate_file_access(filepath)

        elif isinstance(node, ast.Attribute):
            if node.attr in self.DANGEROUS_ATTRIBUTES:
                raise SecurityViolation(f"Access to attribute '{node.attr}' is blocked")

        elif isinstance(node, ast.While):
            # Check for infinite loops
            if isinstance(node.test, ast.Constant) and node.test.value is True:
                raise SecurityViolation("Infinite loop detected (while True)")

        # Check nesting depth
        self._check_nesting_depth(node)

        # Recursively validate child nodes
        for child in ast.iter_child_nodes(node):
            self._validate_ast(child)

    def _validate_file_access(self, filepath: str) -> None:
        """Validate file access against allowed paths."""
        if not self.allowed_paths:
            # If no allowed paths configured, allow all (for backwards compatibility)
            return

        try:
            # Resolve the absolute path
            abs_path = os.path.abspath(filepath)

            # Check if the path is within any allowed directory
            for allowed_path in self.allowed_paths:
                try:
                    # Use os.path.commonpath to check if file is under allowed directory
                    common = os.path.commonpath([abs_path, allowed_path])
                    if common == allowed_path:
                        return  # Access allowed
                except ValueError:
                    # Paths are on different drives (Windows) or other path issues
                    continue

            # If we get here, the path is not allowed
            raise SecurityViolation(f"File access to '{filepath}' is blocked")

        except (OSError, ValueError) as e:
            raise SecurityViolation(f"Invalid file path '{filepath}': {e}") from e

    def _check_nesting_depth(self, node: ast.AST, depth: int = 0) -> None:
        """Check for excessive nesting depth."""
        if depth > 10:  # Maximum nesting depth
            raise SecurityViolation(f"Excessive nesting depth detected: {depth}")

        # Only check certain node types that contribute to nesting
        nesting_nodes = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)

        for child in ast.iter_child_nodes(node):
            if isinstance(child, nesting_nodes):
                self._check_nesting_depth(child, depth + 1)
            else:
                self._check_nesting_depth(child, depth)

    def create_safe_namespace(self, base_namespace: dict) -> dict:
        """Create a safe namespace with restricted built-ins."""
        safe_namespace = base_namespace.copy()

        # Create restricted built-ins
        safe_builtins = {}

        # Safe built-ins to include
        safe_builtin_names = {
            "abs",
            "all",
            "any",
            "ascii",
            "bin",
            "bool",
            "bytearray",
            "bytes",
            "callable",
            "chr",
            "complex",
            "dict",
            "dir",
            "divmod",
            "enumerate",
            "filter",
            "float",
            "format",
            "frozenset",
            "getattr",
            "globals",
            "hasattr",
            "hash",
            "hex",
            "id",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "locals",
            "map",
            "max",
            "min",
            "next",
            "object",
            "oct",
            "ord",
            "pow",
            "print",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "setattr",
            "slice",
            "sorted",
            "str",
            "sum",
            "tuple",
            "type",
            "vars",
            "zip",
            "__import__",
            "open",
        }

        # Copy safe built-ins from the original __builtins__
        original_builtins = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__

        for name in safe_builtin_names:
            if name in original_builtins:
                safe_builtins[name] = original_builtins[name]

        safe_namespace["__builtins__"] = safe_builtins
        return safe_namespace

    def execute_with_limits(self, code: str, namespace: dict) -> None:
        """Execute code with resource limits."""
        # Set memory limit
        try:
            current_mem_limit = resource.getrlimit(resource.RLIMIT_AS)[1]
            if current_mem_limit == resource.RLIM_INFINITY or self.memory_limit < current_mem_limit:
                resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, current_mem_limit))
        except (ValueError, OSError) as e:
            logger.debug(f"Could not set memory limit: {e}")

        # Set CPU time limit
        try:
            current_cpu_limit = resource.getrlimit(resource.RLIMIT_CPU)[1]
            if current_cpu_limit == resource.RLIM_INFINITY or self.timeout < current_cpu_limit:
                resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, current_cpu_limit))
        except (ValueError, OSError) as e:
            logger.debug(f"Could not set CPU time limit: {e}")

        # Execute the code
        exec(code, namespace)


class SecurityManager:
    """Manages security policies for Python REPL execution."""

    def __init__(self):
        """Initialize security manager."""
        # Determine security mode from environment
        restricted_mode = os.environ.get("PYTHON_REPL_RESTRICTED_MODE", "false").lower()
        self.mode = SecurityMode.RESTRICTED if restricted_mode in ("true", "1", "yes", "on") else SecurityMode.NORMAL

        # Initialize validator for restricted mode
        if self.mode == SecurityMode.RESTRICTED:
            self.validator = ASTSecurityValidator()
        else:
            self.validator = None

    def validate_and_execute(self, code: str, namespace: dict) -> None:
        """Validate and execute code according to security mode."""
        if self.mode == SecurityMode.RESTRICTED:
            # Validate code first
            self.validator.validate_code(code)

            # Create safe namespace
            safe_namespace = self.validator.create_safe_namespace(namespace)

            # Execute with limits
            self.validator.execute_with_limits(code, safe_namespace)

            # Update original namespace with safe results (excluding __builtins__)
            for key, value in safe_namespace.items():
                if key != "__builtins__" and not key.startswith("_"):
                    namespace[key] = value
        else:
            # Normal mode - execute directly
            exec(code, namespace)


# Create global security manager instance
security_manager = SecurityManager()

# Tool specification
TOOL_SPEC = {
    "name": "python_repl",
    "description": "Execute Python code in a REPL environment with interactive PTY support and state persistence.\n\n"
    "IMPORTANT SAFETY FEATURES:\n"
    "1. User Confirmation: Requires explicit approval before executing code\n"
    "2. Code Preview: Shows syntax-highlighted code before execution\n"
    "3. State Management: Maintains variables between executions\n"
    "4. Error Handling: Captures and formats errors with suggestions\n"
    "5. Development Mode: Can bypass confirmation in BYPASS_TOOL_CONSENT environments\n\n"
    "Key Features:\n"
    "- Persistent state between executions\n"
    "- Interactive PTY support for real-time feedback\n"
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
                        "Whether to enable interactive PTY mode. "
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
        # Setup state persistence
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
        # Use security manager for execution
        security_manager.validate_and_execute(code, self._namespace)
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


class PtyManager:
    """Manages PTY-based Python execution with state synchronization."""

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

                # Execute in REPL namespace with security validation
                namespace = repl_state.get_namespace()
                security_manager.validate_and_execute(code, namespace)

                # Update the global state with the modified namespace
                for key, value in namespace.items():
                    if not key.startswith("_"):
                        repl_state._namespace[key] = value

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
        """Read and process PTY output with improved error handling and file descriptor management."""
        buffer = ""
        incomplete_bytes = b""  # Buffer for incomplete UTF-8 sequences

        while not self.stop_event.is_set():
            try:
                # Check if file descriptor is still valid
                if self.supervisor_fd < 0:
                    logger.debug("Invalid file descriptor, stopping output reader")
                    break

                # Use select with timeout to avoid blocking
                try:
                    r, _, _ = select.select([self.supervisor_fd], [], [], 0.1)
                except (OSError, ValueError) as e:
                    # File descriptor became invalid during select
                    logger.debug(f"File descriptor error during select: {e}")
                    break

                if self.supervisor_fd in r:
                    try:
                        raw_data = os.read(self.supervisor_fd, 1024)
                    except (OSError, ValueError) as e:
                        # Handle closed file descriptor or other OS errors
                        if e.errno == 9:  # Bad file descriptor
                            logger.debug("PTY closed, stopping output reader")
                        else:
                            logger.warning(f"Error reading from PTY: {e}")
                        break

                    if not raw_data:
                        # EOF reached, PTY closed
                        logger.debug("EOF reached, PTY closed")
                        break

                    # Combine with any incomplete bytes from previous read
                    full_data = incomplete_bytes + raw_data

                    try:
                        # Try to decode the data
                        data = full_data.decode("utf-8")
                        incomplete_bytes = b""  # Clear incomplete buffer on success

                    except UnicodeDecodeError as e:
                        # Handle incomplete UTF-8 sequence at the end
                        if e.start > 0:
                            # We can decode part of the data
                            data = full_data[: e.start].decode("utf-8")
                            incomplete_bytes = full_data[e.start :]
                        else:
                            # Can't decode anything, save for next iteration
                            incomplete_bytes = full_data
                            continue

                    if data:
                        # Append to buffer
                        buffer += data

                        # Process complete lines
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            # Clean and store output
                            cleaned = clean_ansi(line + "\n")
                            self.output_buffer.append(cleaned)

                            # Stream if callback exists
                            if self.callback:
                                try:
                                    self.callback(cleaned)
                                except Exception as callback_error:
                                    logger.warning(f"Error in output callback: {callback_error}")

                        # Handle remaining buffer (usually prompts)
                        if buffer:
                            cleaned = clean_ansi(buffer)
                            if self.callback:
                                try:
                                    self.callback(cleaned)
                                except Exception as callback_error:
                                    logger.warning(f"Error in output callback: {callback_error}")

            except (OSError, IOError) as e:
                # Handle file descriptor errors gracefully
                if hasattr(e, "errno") and e.errno == 9:  # Bad file descriptor
                    logger.debug("PTY file descriptor closed, stopping reader")
                    break
                else:
                    logger.warning(f"I/O error reading PTY output: {e}")
                    # Don't break immediately, try to continue
                    continue

            except UnicodeDecodeError as e:
                # This shouldn't happen anymore with our improved handling, but just in case
                logger.warning(f"Unicode decode error: {e}")
                incomplete_bytes = b""
                continue

            except Exception as e:
                # Catch any other unexpected errors
                logger.error(f"Unexpected error in _read_output: {e}")
                break

        # Clean shutdown - handle any remaining buffer
        if buffer:
            try:
                cleaned = clean_ansi(buffer)
                self.output_buffer.append(cleaned)
                if self.callback:
                    self.callback(cleaned)
            except Exception as e:
                logger.warning(f"Error processing final buffer: {e}")

        # Handle any remaining incomplete bytes at shutdown
        if incomplete_bytes:
            try:
                # Try to decode with error handling
                final_data = incomplete_bytes.decode("utf-8", errors="replace")
                if final_data:
                    cleaned = clean_ansi(final_data)
                    self.output_buffer.append(cleaned)
                    if self.callback:
                        self.callback(cleaned)
            except Exception as e:
                logger.warning(f"Failed to process remaining bytes at shutdown: {e}")

        logger.debug("PTY output reader thread finished")

    def _handle_input(self) -> None:
        """Handle interactive user input with improved buffering."""
        while not self.stop_event.is_set():
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if sys.stdin in r:
                    # Read all available input
                    input_data = ""
                    while True:
                        char = sys.stdin.read(1)
                        if not char or char == "\n":
                            input_data += "\n"
                            break
                        input_data += char

                    if input_data:
                        # Only store input once
                        if input_data not in self.input_buffer:
                            self.input_buffer.append(input_data)
                            # Send to PTY with proper line ending
                            os.write(self.supervisor_fd, input_data.encode())

            except (OSError, IOError):
                break

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
        """Stop PTY session and clean up resources properly."""
        logger.debug("Stopping PTY session...")

        # Signal threads to stop
        self.stop_event.set()

        # Clean up child process
        if self.pid > 0:
            try:
                # Try graceful termination first
                os.kill(self.pid, signal.SIGTERM)

                # Wait briefly for graceful shutdown
                try:
                    pid, status = os.waitpid(self.pid, os.WNOHANG)
                    if pid == 0:  # Process still running
                        # Give it a moment
                        import time

                        time.sleep(0.1)
                        # Try again
                        pid, status = os.waitpid(self.pid, os.WNOHANG)
                        if pid == 0:
                            # Force kill if still running
                            logger.debug("Forcing process termination")
                            os.kill(self.pid, signal.SIGKILL)
                            os.waitpid(self.pid, 0)

                except OSError as e:
                    # Process might have already exited
                    logger.debug(f"Process cleanup error (likely already exited): {e}")

            except (OSError, ProcessLookupError) as e:
                # Process doesn't exist or already terminated
                logger.debug(f"Process termination error (likely already gone): {e}")

            finally:
                self.pid = -1

        # Clean up file descriptor
        if self.supervisor_fd >= 0:
            try:
                os.close(self.supervisor_fd)
                logger.debug("PTY supervisor file descriptor closed")
            except OSError as e:
                logger.debug(f"Error closing supervisor fd: {e}")
            finally:
                self.supervisor_fd = -1

        logger.debug("PTY session cleanup completed")


output_buffer: List[str] = []


def python_repl(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Execute Python code with persistent state and output streaming."""
    console = console_util.create()

    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    code = tool_input["code"]
    interactive = tool_input.get("interactive", True)
    reset_state = tool_input.get("reset_state", False)

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

        # Add permissions check - only show confirmation dialog if not
        # in BYPASS_TOOL_CONSENT mode and not in non_interactive mode
        if not strands_dev and not non_interactive_mode:
            # Create a table with code details for better visualization
            details_table = Table(show_header=False, box=box.SIMPLE)
            details_table.add_column("Property", style="cyan", justify="right")
            details_table.add_column("Value", style="green")

            # Add code details
            details_table.add_row("Code Length", f"{len(code)} characters")
            details_table.add_row("Line Count", f"{len(code.splitlines())} lines")
            details_table.add_row("Mode", "Interactive" if interactive else "Standard")
            details_table.add_row("Reset State", "Yes" if reset_state else "No")

            # Show confirmation panel
            console.print(
                Panel(
                    details_table,
                    title="[bold blue]🐍 Python Code Execution Preview",
                    border_style="blue",
                    box=box.ROUNDED,
                )
            )
            # Get user confirmation
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

        try:
            if interactive:
                console.print("[green]Running in interactive mode...[/]")
                pty_mgr = PtyManager()
                pty_mgr.start(code)

                # Wait for completion
                exit_status = None  # Initialize exit_status variable
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
            else:
                console.print("[blue]Running in standard mode...[/]")
                captured = OutputCapture()
                with captured as output_capture:
                    repl_state.execute(code)
                    output = output_capture.get_output()
                    if output:
                        console.print("[cyan]Output:[/]")
                        console.print(output)

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
            # Re-raise the exception after cleanup
            raise

        except SecurityViolation as e:
            # Handle security violations separately
            error_msg = f"Security Violation: {str(e)}"
            console.print(
                Panel(
                    f"[bold red]{error_msg}[/bold red]",
                    title="[bold red]Security Error[/]",
                    border_style="red",
                )
            )

            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": error_msg}],
            }

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
