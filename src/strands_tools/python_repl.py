"""
Execute Python code in a REPL environment with PTY support and state persistence.

This module provides a tool for running Python code through a Strands Agent, with features like:
- Persistent state between executions
- Interactive PTY support for real-time feedback
- Output capturing and formatting
- Error handling and logging
- State reset capabilities
- User confirmation for code execution
- Configurable security modes (normal by default, restricted when enabled)

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
import time
import traceback
import types
from datetime import datetime
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

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

# Tool specification
TOOL_SPEC = {
    "name": "python_repl",
    "description": "Execute Python code in a REPL environment with interactive PTY support and state persistence.\n\n"
    "IMPORTANT SAFETY FEATURES:\n"
    "1. User Confirmation: Requires explicit approval before executing code\n"
    "2. Code Preview: Shows syntax-highlighted code before execution\n"
    "3. State Management: Maintains variables between executions\n"
    "4. Error Handling: Captures and formats errors with suggestions\n"
    "5. Development Mode: Can bypass confirmation in BYPASS_TOOL_CONSENT environments\n"
    "6. Security Modes: Normal (default) and Restricted (enabled with PYTHON_REPL_RESTRICTED_MODE=true)\n\n"
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


class SecurityMode(Enum):
    """Security modes for Python REPL execution."""

    NORMAL = "normal"
    RESTRICTED = "restricted"


class SecurityViolation(Exception):
    """Raised when code violates security policies."""

    pass


class ASTSecurityValidator:
    """AST-based security validator for Python code."""

    # Dangerous imports that should be blocked
    DANGEROUS_IMPORTS = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "http",
        "ftplib",
        "smtplib",
        "poplib",
        "imaplib",
        "telnetlib",
        "webbrowser",
        "ctypes",
        "imp",
        "importlib",
        "pkgutil",
        "runpy",
        "code",
        "codeop",
        "compile",
        "eval",
        "exec",
        "shutil",
        "tempfile",
        "glob",
        "platform",
        "getpass",
        "multiprocessing",
        "threading",
        "asyncio",
        "concurrent",
        "pickle",
        "dill",
        "marshal",
        "shelve",
        "dbm",
        "sqlite3",
        "pip",
        "conda",
        "setuptools",
        "distutils",
    }

    # Dangerous built-in functions
    DANGEROUS_BUILTINS = {
        "eval",
        "exec",
        "compile",
        "input",
        "raw_input",
        "file",
        "execfile",
        "reload",
        "vars",
        "globals",
        "locals",
        "dir",
        "hasattr",
        "getattr",
        "setattr",
        "delattr",
        "callable",
        "isinstance",
        "issubclass",
    }

    # Dangerous attributes
    DANGEROUS_ATTRIBUTES = {
        "__builtins__",
        "__globals__",
        "__locals__",
        "__dict__",
        "__class__",
        "__bases__",
        "__mro__",
        "__subclasses__",
        "__init__",
        "__new__",
        "__del__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__getattr__",
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "__call__",
        "__enter__",
        "__exit__",
    }

    # Safe built-in functions that are allowed
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "chr",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hex",
        "id",
        "int",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "open",  # Allowed with path restrictions
        "__import__",  # Allowed to import modules, but restricted in usage
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
    }

    def __init__(self, timeout: int = 30, memory_limit_mb: int = 100):
        """Initialize AST security validator.

        Args:
            timeout: Maximum execution time in seconds
            memory_limit_mb: Maximum memory usage in MB
        """
        self.timeout = timeout
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes

        # Configure allowed paths for file operations
        self.allowed_paths = self._get_allowed_paths()

        # Map modules to their security risks
        self.danger_reasons = {
            "os": "system access and file operations",
            "sys": "system internals and interpreter access",
            "subprocess": "process execution and shell commands",
            "socket": "network access and communication",
            "urllib": "network requests and web access",
            "requests": "HTTP requests and web access",
            "http": "HTTP server and client operations",
            "ftplib": "FTP protocol access",
            "smtplib": "email sending capabilities",
            "poplib": "email retrieval capabilities",
            "imaplib": "email server access",
            "telnetlib": "remote terminal access",
            "webbrowser": "web browser control",
            "ctypes": "low-level system calls and memory access",
            "imp": "dynamic module import capabilities",
            "importlib": "advanced import system manipulation",
            "pkgutil": "package utilities and import manipulation",
            "runpy": "script execution capabilities",
            "code": "interactive code execution",
            "codeop": "code compilation utilities",
            "compile": "dynamic code compilation",
            "eval": "dynamic expression evaluation",
            "exec": "dynamic code execution",
            "shutil": "file and directory operations",
            "tempfile": "temporary file creation",
            "glob": "file pattern matching and filesystem access",
            "platform": "system information disclosure",
            "getpass": "user credential access",
            "multiprocessing": "parallel process creation",
            "threading": "concurrent execution capabilities",
            "asyncio": "asynchronous execution framework",
            "concurrent": "concurrent execution utilities",
            "pickle": "unsafe object serialization",
            "dill": "advanced object serialization",
            "marshal": "low-level object serialization",
            "shelve": "persistent object storage",
            "dbm": "database file access",
            "sqlite3": "database access",
            "pip": "package installation capabilities",
            "conda": "package and environment management",
            "setuptools": "package installation and distribution",
            "distutils": "package building and installation",
        }

        # Map built-ins to their security risks
        self.builtin_reasons = {
            "eval": "dynamic expression evaluation",
            "exec": "dynamic code execution",
            "compile": "code compilation and execution",
            "open": "file system access",
            "input": "user input collection",
            "raw_input": "user input collection",
            "file": "file system access",
            "execfile": "file execution",
            "reload": "module reloading and manipulation",
            "vars": "variable namespace access",
            "globals": "global namespace access",
            "locals": "local namespace access",
            "dir": "object introspection",
            "hasattr": "attribute existence checking",
            "getattr": "dynamic attribute access",
            "setattr": "dynamic attribute modification",
            "delattr": "attribute deletion",
            "callable": "function call introspection",
            "isinstance": "type checking and introspection",
            "issubclass": "inheritance introspection",
        }

        # Map attributes to their security risks
        self.attribute_reasons = {
            "__builtins__": "built-in function access",
            "__globals__": "global namespace access",
            "__locals__": "local namespace access",
            "__dict__": "object dictionary access",
            "__class__": "class introspection",
            "__bases__": "inheritance introspection",
            "__mro__": "method resolution order access",
            "__subclasses__": "subclass enumeration",
            "__init__": "constructor access",
            "__new__": "object creation control",
            "__del__": "destructor access",
            "__getattribute__": "attribute access control",
            "__setattr__": "attribute modification control",
            "__delattr__": "attribute deletion control",
            "__getattr__": "missing attribute access",
            "__getitem__": "item access control",
            "__setitem__": "item modification control",
            "__delitem__": "item deletion control",
            "__call__": "callable object control",
            "__enter__": "context manager entry",
            "__exit__": "context manager exit",
        }

    def _get_allowed_paths(self) -> List[str]:
        """Get list of allowed paths for file operations.

        Returns:
            List of allowed absolute paths
        """
        allowed_paths = []

        # Add current working directory if allowed
        allow_current = os.environ.get("PYTHON_REPL_ALLOW_CURRENT_DIR", "true").lower()
        if allow_current in ("true", "1", "yes", "on"):
            allowed_paths.append(os.getcwd())

        # Add custom allowed paths from environment
        custom_paths = os.environ.get("PYTHON_REPL_ALLOWED_PATHS", "")
        if custom_paths:
            for path in custom_paths.split(","):
                path = path.strip()
                if path:
                    # Convert to absolute path and normalize
                    abs_path = os.path.abspath(os.path.expanduser(path))
                    if os.path.exists(abs_path) and os.path.isdir(abs_path):
                        allowed_paths.append(abs_path)
                    else:
                        logger.warning(f"Allowed path does not exist or is not a directory: {abs_path}")

        # Add some safe default temporary directories if no paths configured
        if not allowed_paths:
            temp_dirs = ["/tmp", "/var/tmp"]
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
                    allowed_paths.append(temp_dir)
                    break

        logger.debug(f"Allowed paths for file operations: {allowed_paths}")
        return allowed_paths

    def _is_path_allowed(self, file_path: str) -> bool:
        """Check if a file path is within allowed directories.

        Args:
            file_path: Path to check

        Returns:
            True if path is allowed, False otherwise
        """
        if not self.allowed_paths:
            return False

        try:
            # Convert to absolute path and resolve symlinks
            abs_path = os.path.abspath(os.path.expanduser(file_path))
            real_path = os.path.realpath(abs_path)

            # Check if the path is within any allowed directory
            for allowed_path in self.allowed_paths:
                allowed_real = os.path.realpath(allowed_path)
                # Check if the file path starts with the allowed path
                if real_path.startswith(allowed_real + os.sep) or real_path == allowed_real:
                    return True

            return False

        except (OSError, ValueError):
            # If there's any error resolving the path, deny access
            return False

    def validate_code(self, code: str) -> None:
        """Validate Python code using AST analysis.

        Args:
            code: Python code to validate

        Raises:
            SecurityViolation: If code violates security policies
            SyntaxError: If code has syntax errors
        """
        try:
            # Parse code into AST
            tree = ast.parse(code)

            # Walk through AST nodes and check for violations
            for node in ast.walk(tree):
                self._check_node(node)

        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in code: {e}") from e
        except SecurityViolation:
            raise
        except Exception as e:
            raise SecurityViolation(f"Code analysis failed: {e}") from e

    def _check_node(self, node: ast.AST) -> None:
        """Check individual AST node for security violations.

        Args:
            node: AST node to check

        Raises:
            SecurityViolation: If node violates security policies
        """
        # Check imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self._check_imports(node)

        # Check function calls
        elif isinstance(node, ast.Call):
            self._check_function_call(node)

        # Check attribute access
        elif isinstance(node, ast.Attribute):
            self._check_attribute_access(node)

        # Check name access
        elif isinstance(node, ast.Name):
            self._check_name_access(node)

        # Check for suspicious patterns
        self._check_suspicious_patterns(node)

    def _check_imports(self, node: Union[ast.Import, ast.ImportFrom]) -> None:
        """Check import statements for dangerous modules.

        Args:
            node: Import AST node

        Raises:
            SecurityViolation: If importing dangerous module
        """
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if module_name in self.DANGEROUS_IMPORTS:
                    reason = self.danger_reasons.get(module_name, "potentially dangerous operations")
                    raise SecurityViolation(
                        f"Import of module '{module_name}' is blocked: {reason}. "
                        f"This module is not allowed in restricted security mode."
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split(".")[0]
                if module_name in self.DANGEROUS_IMPORTS:
                    reason = self.danger_reasons.get(module_name, "potentially dangerous operations")
                    raise SecurityViolation(
                        f"Import from module '{module_name}' is blocked: {reason}. "
                        f"This module is not allowed in restricted security mode."
                    )

    def _check_function_call(self, node: ast.Call) -> None:
        """Check function calls for dangerous operations.

        Args:
            node: Call AST node

        Raises:
            SecurityViolation: If calling dangerous function
        """
        # Check direct function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Special handling for open() function to check file paths
            if func_name == "open" and len(node.args) > 0:
                self._check_file_access(node)
                return

            # Special handling for __import__() function to check module names
            if func_name == "__import__" and len(node.args) > 0:
                self._check_import_call(node)
                return

            if func_name in self.DANGEROUS_BUILTINS:
                reason = self.builtin_reasons.get(func_name, "potentially dangerous operations")
                raise SecurityViolation(
                    f"Call to built-in function '{func_name}' is blocked: {reason}. "
                    f"This function is not allowed in restricted security mode."
                )

        # Check method calls that might be dangerous
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in self.DANGEROUS_ATTRIBUTES:
                reason = self.attribute_reasons.get(attr_name, "potentially dangerous operations")
                raise SecurityViolation(
                    f"Call to method '{attr_name}' is blocked: {reason}. "
                    f"This method is not allowed in restricted security mode."
                )

    def _check_file_access(self, node: ast.Call) -> None:
        """Check file access operations for path restrictions.

        Args:
            node: Call AST node for file operations

        Raises:
            SecurityViolation: If file access violates path restrictions
        """
        if not node.args:
            return

        # Get the file path argument
        file_arg = node.args[0]

        # Only check string literals for now (could be extended to variables)
        if isinstance(file_arg, ast.Constant) and isinstance(file_arg.value, str):
            file_path = file_arg.value

            if not self._is_path_allowed(file_path):
                allowed_paths_str = ", ".join(self.allowed_paths) if self.allowed_paths else "none"
                raise SecurityViolation(
                    f"File access to '{file_path}' is blocked: path is outside allowed directories. "
                    f"Allowed paths: {allowed_paths_str}. "
                    f"Configure PYTHON_REPL_ALLOWED_PATHS and PYTHON_REPL_ALLOW_CURRENT_DIR to modify restrictions."
                )

    def _check_import_call(self, node: ast.Call) -> None:
        """Check __import__() function calls for dangerous modules.

        Args:
            node: Call AST node for __import__ operations

        Raises:
            SecurityViolation: If importing dangerous module
        """
        if not node.args:
            return

        # Get the module name argument
        module_arg = node.args[0]

        # Only check string literals for now (could be extended to variables)
        if isinstance(module_arg, ast.Constant) and isinstance(module_arg.value, str):
            module_name = module_arg.value.split(".")[0]  # Get top-level module

            if module_name in self.DANGEROUS_IMPORTS:
                reason = self.danger_reasons.get(module_name, "potentially dangerous operations")
                raise SecurityViolation(
                    f"Import of module '{module_name}' is blocked: {reason}. "
                    f"This module is not allowed in restricted security mode."
                )

    def _check_attribute_access(self, node: ast.Attribute) -> None:
        """Check attribute access for dangerous attributes.

        Args:
            node: Attribute AST node

        Raises:
            SecurityViolation: If accessing dangerous attribute
        """
        if node.attr in self.DANGEROUS_ATTRIBUTES:
            reason = self.attribute_reasons.get(node.attr, "potentially dangerous operations")
            raise SecurityViolation(
                f"Access to attribute '{node.attr}' is blocked: {reason}. "
                f"This attribute is not allowed in restricted security mode."
            )

    def _check_name_access(self, node: ast.Name) -> None:
        """Check name access for dangerous built-ins.

        Args:
            node: Name AST node

        Raises:
            SecurityViolation: If accessing dangerous name
        """
        if isinstance(node.ctx, ast.Load) and node.id in self.DANGEROUS_BUILTINS:
            reason = self.builtin_reasons.get(node.id, "potentially dangerous operations")
            raise SecurityViolation(
                f"Access to built-in '{node.id}' is blocked: {reason}. "
                f"This built-in is not allowed in restricted security mode."
            )

    def _check_suspicious_patterns(self, node: ast.AST) -> None:
        """Check for suspicious code patterns that could be dangerous.

        Args:
            node: AST node to check

        Raises:
            SecurityViolation: If suspicious pattern is detected
        """
        # Check for infinite loops
        if isinstance(node, ast.While):
            if isinstance(node.test, ast.Constant) and node.test.value is True:
                raise SecurityViolation(
                    "Infinite loop detected (while True): can cause system hang. " "Use bounded loops instead."
                )
            elif isinstance(node.test, ast.NameConstant) and node.test.value is True:
                raise SecurityViolation(
                    "Infinite loop detected (while True): can cause system hang. " "Use bounded loops instead."
                )

        # Check for excessively large ranges
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "range" and len(node.args) > 0:
                # Check the last argument (stop value) for excessive size
                last_arg = node.args[-1]
                if isinstance(last_arg, ast.Constant) and isinstance(last_arg.value, int):
                    if last_arg.value > 1000000:  # 1 million limit
                        raise SecurityViolation(
                            f"Large range detected ({last_arg.value:,}): can cause memory exhaustion. "
                            f"Maximum allowed range is 1,000,000."
                        )
                elif isinstance(last_arg, ast.Num) and last_arg.n > 1000000:
                    raise SecurityViolation(
                        f"Large range detected ({last_arg.n:,}): can cause memory exhaustion. "
                        f"Maximum allowed range is 1,000,000."
                    )

        # Check for deeply nested structures
        elif isinstance(node, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
            # Count nesting depth starting from this node (depth = 1)
            depth = self._get_nesting_depth(node, 1)
            if depth > 10:  # Maximum nesting depth
                raise SecurityViolation(
                    f"Excessive nesting depth ({depth}): can cause stack overflow. "
                    f"Maximum allowed nesting depth is 10."
                )

    def _get_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate the maximum nesting depth from a given node.

        Args:
            node: Starting AST node
            current_depth: Current depth level

        Returns:
            Maximum nesting depth found
        """
        max_depth = current_depth

        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While, ast.If, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)):
                child_depth = self._get_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)

        return max_depth

    def create_safe_namespace(self, base_namespace: Dict[str, Any]) -> Dict[str, Any]:
        """Create a safe namespace with restricted built-ins.

        Args:
            base_namespace: Base namespace to filter

        Returns:
            Filtered namespace with only safe built-ins
        """
        # Start with base namespace
        safe_namespace = base_namespace.copy()

        # Create restricted __builtins__
        safe_builtins = {}
        import builtins

        for name in self.SAFE_BUILTINS:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)

        # Add safe namespace items
        safe_namespace["__builtins__"] = safe_builtins
        safe_namespace["__name__"] = "__main__"

        return safe_namespace

    def execute_with_limits(self, code: str, namespace: Dict[str, Any]) -> None:
        """Execute code with resource limits.

        Args:
            code: Python code to execute
            namespace: Execution namespace

        Raises:
            SecurityViolation: If execution violates limits
        """
        # Set memory limit
        try:
            current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
            logger.debug(f"Current memory limits: soft={current_soft}, hard={current_hard}")

            if self.memory_limit > current_hard and current_hard != resource.RLIM_INFINITY:
                new_limit = min(self.memory_limit, current_hard)
                logger.warning(
                    f"Requested memory limit {self.memory_limit} exceeds system limit {current_hard}, using {new_limit}"
                )
                resource.setrlimit(resource.RLIMIT_AS, (new_limit, current_hard))
            else:
                resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, self.memory_limit))

        except (OSError, ValueError) as e:
            logger.warning(f"Could not set memory limit: {e}")

        # Set CPU time limit
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
        except (OSError, ValueError) as e:
            logger.warning(f"Could not set CPU time limit: {e}")

        # Execute with timeout
        start_time = time.time()

        try:
            exec(code, namespace)
        except MemoryError as e:
            raise SecurityViolation(f"Code exceeded memory limit of {self.memory_limit // (1024 * 1024)}MB") from e
        except KeyboardInterrupt as e:
            raise SecurityViolation(f"Code execution timed out after {self.timeout} seconds") from e
        except Exception as e:
            # Check if it's a timeout
            if time.time() - start_time > self.timeout:
                raise SecurityViolation(f"Code execution timed out after {self.timeout} seconds") from e
            raise


class SecurityManager:
    """Manages security modes for Python REPL."""

    def __init__(self):
        """Initialize security manager."""
        self.mode = self._get_security_mode()
        self.ast_validator = ASTSecurityValidator(
            timeout=int(os.environ.get("PYTHON_REPL_TIMEOUT", "30")),
            memory_limit_mb=int(os.environ.get("PYTHON_REPL_MEMORY_LIMIT_MB", "100")),
        )

        # Log security mode
        logger.info(f"Python REPL security mode: {self.mode.value}")

    def _get_security_mode(self) -> SecurityMode:
        """Get security mode from environment variable.

        Returns:
            SecurityMode enum value
        """
        # Check if restricted mode is enabled via boolean flag
        restricted_mode = os.environ.get("PYTHON_REPL_RESTRICTED_MODE", "false").lower()

        if restricted_mode in ("true", "1", "yes", "on"):
            return SecurityMode.RESTRICTED
        else:
            return SecurityMode.NORMAL

    def validate_and_execute(self, code: str, namespace: Dict[str, Any]) -> None:
        """Validate and execute code based on security mode.

        Args:
            code: Python code to execute
            namespace: Execution namespace

        Raises:
            SecurityViolation: If code violates security policies
            SyntaxError: If code has syntax errors
        """
        if self.mode == SecurityMode.NORMAL:
            # Normal mode - execute without restrictions
            exec(code, namespace)

        elif self.mode == SecurityMode.RESTRICTED:
            # Restricted mode - validate then execute with restrictions
            self.ast_validator.validate_code(code)
            safe_namespace = self.ast_validator.create_safe_namespace(namespace)
            self.ast_validator.execute_with_limits(code, safe_namespace)

            # Update original namespace with safe results
            for key, value in safe_namespace.items():
                if not key.startswith("__") and key not in ["__builtins__", "__name__"]:
                    namespace[key] = value


# Create global security manager
security_manager = SecurityManager()


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
        """Execute code using security manager and save state."""
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
        """Start PTY session with code execution using security manager."""
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

                # Execute using security manager with REPL namespace
                namespace = repl_state.get_namespace()
                security_manager.validate_and_execute(code, namespace)

                os._exit(0)

            except SecurityViolation as e:
                logger.warning(f"Security violation: {e}")
                os._exit(1)
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

        # Display security mode info
        security_info = f"Security Mode: {security_manager.mode.value.upper()}"
        if security_manager.mode == SecurityMode.RESTRICTED:
            memory_mb = security_manager.ast_validator.memory_limit // (1024 * 1024)
            security_info += f" (Timeout: {security_manager.ast_validator.timeout}s, Memory: {memory_mb}MB)"
        console.print(f"[yellow]{security_info}[/]")

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
        # Handle security violations specifically
        error_time = datetime.now()

        console.print(
            Panel(
                f"[bold red]🔒 Security Violation:[/bold red]\n{str(e)}",
                title="[bold red]Security Policy Violation[/]",
                border_style="red",
            )
        )

        # Log security violation
        logger.warning(f"Security violation: {e}")

        error_msg = f"[{error_time.isoformat()}] Security Violation: {str(e)}"

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
