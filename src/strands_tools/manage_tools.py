"""Runtime tool management for Strands Agents.

This module provides dynamic tool management capabilities, enabling agents to expand
their functionality at runtime without restarts. It serves as a package manager for
AI agent tools, supporting installation, creation, discovery, and lifecycle management.

Features:
    - Dynamic Loading: Load tools from packages, files, URLs, or GitHub repositories
    - Runtime Creation: Generate new tools from Python code on-the-fly
    - Module Discovery: Introspect packages to find available @tool decorated functions
    - Sandbox Testing: Validate code in isolated subprocess before loading
    - Auto-Installation: Optionally install missing PyPI packages via pip or uv

Environment Variables:
    STRANDS_AUTO_INSTALL_TOOLS (str): Enable automatic package installation.
        Set to "true" to enable. Default: "false"
    STRANDS_PACKAGE_INSTALLER (str): Package installer to use.
        Options: "pip" (default), "uv"
    STRANDS_INSTALL_TIMEOUT (int): Package installation timeout in seconds.
        Default: 300
    STRANDS_DISABLE_LOAD_TOOL (str): Disable all dynamic tool loading.
        Set to "true" to disable. Default: "false"
    STRANDS_TOOLS_CACHE_DIR (str): Directory for caching fetched/created tools.
        Default: system temp directory / "strands_tools_cache"

Example:
    Basic usage with an agent::

        from strands import Agent
        from strands_tools import manage_tools

        agent = Agent(tools=[manage_tools])

        # List registered tools
        agent.tool.manage_tools(action="list")

        # Add a tool from a package
        agent.tool.manage_tools(action="add", tools="strands_tools.calculator")

        # Discover available tools in a module
        agent.tool.manage_tools(action="discover", tools="strands_tools")

        # Create a custom tool at runtime
        agent.tool.manage_tools(
            action="create",
            code='''
            from strands import tool

            @tool
            def multiply(a: int, b: int) -> int:
                \"\"\"Multiply two integers.\"\"\"
                return a * b
            '''
        )
"""

import hashlib
import importlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from strands import tool

logger = logging.getLogger(__name__)

# Environment variable names
ENV_INSTALL_TIMEOUT = "STRANDS_INSTALL_TIMEOUT"
ENV_PACKAGE_INSTALLER = "STRANDS_PACKAGE_INSTALLER"
ENV_AUTO_INSTALL = "STRANDS_AUTO_INSTALL_TOOLS"
ENV_DISABLE_LOAD = "STRANDS_DISABLE_LOAD_TOOL"
ENV_TOOLS_CACHE_DIR = "STRANDS_TOOLS_CACHE_DIR"

# Default configuration values
DEFAULT_INSTALL_TIMEOUT = 300
DEFAULT_PACKAGE_INSTALLER = "pip"
DEFAULT_TOOLS_CACHE_DIR = Path(tempfile.gettempdir()) / "strands_tools_cache"


def _get_install_timeout() -> int:
    """Retrieve the package installation timeout from environment.

    Returns:
        The timeout value in seconds. Defaults to 300 if not set or invalid.
    """
    try:
        return int(os.environ.get(ENV_INSTALL_TIMEOUT, DEFAULT_INSTALL_TIMEOUT))
    except ValueError:
        return DEFAULT_INSTALL_TIMEOUT


def _get_installer_command() -> List[str]:
    """Determine the package installer command based on environment configuration.

    Supports two installers:
        - pip: Standard Python package installer (default)
        - uv: Fast Rust-based package installer

    The installer is selected via the STRANDS_PACKAGE_INSTALLER environment variable.
    Falls back to pip if uv is requested but not available in PATH.

    Returns:
        A list of command arguments for the selected package installer.

    Example:
        >>> os.environ["STRANDS_PACKAGE_INSTALLER"] = "uv"
        >>> _get_installer_command()
        ["uv", "pip"]
    """
    installer = os.environ.get(ENV_PACKAGE_INSTALLER, DEFAULT_PACKAGE_INSTALLER).lower()

    if installer == "uv":
        if shutil.which("uv"):
            return ["uv", "pip"]
        logger.warning("uv not found in PATH, falling back to pip")

    return [sys.executable, "-m", "pip"]


def _extract_package_name(tool_spec: str) -> Optional[str]:
    """Extract the PyPI package name from a tool specification string.

    Converts Python module paths to their corresponding pip package names by
    extracting the top-level package and converting underscores to hyphens.

    Args:
        tool_spec: A tool specification string in one of these formats:
            - Module path: "strands_tools.calculator"
            - Module with function: "my_package.module:my_function"
            - File path: "./my_tool.py" or "/absolute/path.py"

    Returns:
        The pip package name (e.g., "strands-tools"), or None if the spec
        is a file path rather than a module reference.

    Example:
        >>> _extract_package_name("strands_tools.calculator")
        "strands-tools"
        >>> _extract_package_name("./local_tool.py")
        None
    """
    if tool_spec.startswith((".", "/", "~")):
        return None

    base = tool_spec.split(".")[0].split(":")[0]
    return base.replace("_", "-")


def _install_packages(packages: List[str]) -> Dict[str, bool]:
    """Install Python packages using the configured package installer.

    Attempts to install each package using either pip or uv, depending on
    the STRANDS_PACKAGE_INSTALLER environment variable setting.

    Args:
        packages: A list of package names to install (e.g., ["strands-tools"]).

    Returns:
        A dictionary mapping package names to their installation success status.

    Example:
        >>> results = _install_packages(["requests", "nonexistent-pkg"])
        >>> results
        {"requests": True, "nonexistent-pkg": False}
    """
    results = {}
    timeout = _get_install_timeout()
    installer_cmd = _get_installer_command()
    installer_name = "uv" if "uv" in installer_cmd else "pip"

    for package in packages:
        try:
            logger.info(f"Installing package via {installer_name}: {package}")

            result = subprocess.run(
                [*installer_cmd, "install", package, "-q"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                results[package] = True
                logger.info(f"Successfully installed: {package}")
            else:
                results[package] = False
                logger.warning(f"Failed to install {package}: {result.stderr}")

        except subprocess.TimeoutExpired:
            results[package] = False
            logger.warning(f"Installation timeout ({timeout}s) for package: {package}")
        except Exception as e:
            results[package] = False
            logger.warning(f"Failed to install {package}: {e}")

    return results


def _get_tools_cache_dir() -> Path:
    """Get or create the cache directory for storing fetched and created tools.

    The cache directory location can be customized via the STRANDS_TOOLS_CACHE_DIR
    environment variable. If not set, uses a subdirectory in the system temp folder.

    Returns:
        Path to the tools cache directory (created if it doesn't exist).
    """
    cache_dir = Path(os.environ.get(ENV_TOOLS_CACHE_DIR, DEFAULT_TOOLS_CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _fetch_from_url(url: str) -> str:
    """Fetch Python source code from a remote URL.

    Supports direct raw URLs as well as GitHub blob URLs, which are automatically
    converted to their raw.githubusercontent.com equivalents.

    Args:
        url: The URL to fetch from. Supported formats:
            - Raw URLs: "https://example.com/tool.py"
            - GitHub blob: "https://github.com/user/repo/blob/main/tool.py"
            - Gist raw: "https://gist.githubusercontent.com/user/id/raw/file.py"

    Returns:
        The fetched source code as a string.

    Raises:
        RuntimeError: If the fetch operation fails for any reason.

    Example:
        >>> code = _fetch_from_url("https://github.com/user/repo/blob/main/tool.py")
        >>> print(code[:50])
        "from strands import tool..."
    """
    timeout = _get_install_timeout()

    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    logger.info(f"Fetching tool from: {url}")

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch from {url}: {e}") from e


def _create_tool_file(code: str, name: Optional[str] = None) -> Path:
    """Create a Python tool file in the cache directory.

    Writes the provided source code to a .py file in the tools cache directory.
    If no name is provided, attempts to extract the function name from the code
    or generates a hash-based name as a fallback.

    Args:
        code: Python source code containing a @tool decorated function.
        name: Optional filename (without extension). If not provided, the name
            is extracted from the first function definition in the code.

    Returns:
        Path to the created tool file.

    Example:
        >>> code = "from strands import tool\\n@tool\\ndef greet(): pass"
        >>> path = _create_tool_file(code)
        >>> path.name
        "greet.py"
    """
    cache_dir = _get_tools_cache_dir()

    if not name:
        for line in code.split("\n"):
            line = line.strip()
            if line.startswith("def ") and "(" in line:
                name = line[4 : line.index("(")].strip()
                break

    if not name:
        name = f"tool_{hashlib.md5(code.encode()).hexdigest()[:8]}"

    tool_path = cache_dir / f"{name}.py"
    tool_path.write_text(code)
    logger.info(f"Created tool file: {tool_path}")

    return tool_path


@dataclass
class ToolInfo:
    """Metadata container for a discovered tool.

    Attributes:
        name: The tool's registered name.
        description: Brief description of the tool's functionality.
        module: Fully qualified module path where the tool is defined.
        parameters: List of parameter metadata dictionaries.
        required_params: Names of required parameters.
        optional_params: Names of optional parameters.
    """

    name: str
    description: str
    module: str
    parameters: List[Dict[str, Any]]
    required_params: List[str]
    optional_params: List[str]


def _discover_tools_in_module(module_name: str, recursive: bool = True) -> Dict[str, ToolInfo]:
    """Discover all @tool decorated functions within a module and its submodules.

    Scans the specified module for functions decorated with @tool and extracts
    their metadata including parameters, descriptions, and type information.

    Args:
        module_name: The fully qualified module name to scan (e.g., "strands_tools").
        recursive: If True, recursively scans all submodules. Default: True.

    Returns:
        A dictionary mapping tool names to their ToolInfo metadata objects.

    Raises:
        RuntimeError: If the specified module cannot be imported.

    Example:
        >>> tools = _discover_tools_in_module("strands_tools")
        >>> list(tools.keys())
        ["calculator", "shell", "editor", ...]
    """
    tools = {}

    def _extract_tool_info(obj: Any, source_module: str) -> Optional[ToolInfo]:
        """Extract metadata from a tool-decorated callable."""
        try:
            if not (hasattr(obj, "tool_name") and hasattr(obj, "tool_spec")):
                if hasattr(obj, "__wrapped__"):
                    obj = obj.__wrapped__
                    if not hasattr(obj, "tool_name"):
                        return None
                else:
                    return None

            spec = getattr(obj, "tool_spec", {})
            tool_name = getattr(obj, "tool_name", None)
            if not tool_name:
                return None

            desc = spec.get("description", "")
            if not desc:
                desc = getattr(obj, "__doc__", "") or "No description"
            desc = desc.split("\n")[0].strip()
            if len(desc) > 120:
                desc = desc[:117] + "..."

            input_schema = spec.get("inputSchema", {})
            # Handle both legacy (direct properties) and current (nested under "json") schema structures
            if "json" in input_schema:
                schema_data = input_schema.get("json", {})
            else:
                schema_data = input_schema
            properties = schema_data.get("properties", {})
            required = schema_data.get("required", [])

            parameters = []
            required_params = []
            optional_params = []

            for param_name, param_info in properties.items():
                if param_name == "agent":
                    continue

                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                if len(param_desc) > 50:
                    param_desc = param_desc[:47] + "..."

                parameters.append(
                    {
                        "name": param_name,
                        "type": param_type,
                        "description": param_desc,
                        "required": param_name in required,
                    }
                )

                if param_name in required:
                    required_params.append(param_name)
                else:
                    optional_params.append(param_name)

            return ToolInfo(
                name=tool_name,
                description=desc,
                module=source_module,
                parameters=parameters,
                required_params=required_params,
                optional_params=optional_params,
            )

        except Exception as e:
            logger.debug(f"Failed to extract tool info: {e}")
            return None

    def _scan_module(mod_name: str):
        """Scan a single module for tool-decorated functions."""
        try:
            module = importlib.import_module(mod_name)
        except ImportError as e:
            logger.debug(f"Cannot import {mod_name}: {e}")
            return

        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(module, attr_name)
                info = _extract_tool_info(attr, mod_name)
                if info and info.name not in tools:
                    tools[info.name] = info
            except Exception as e:
                logger.debug(f"Error inspecting {mod_name}.{attr_name}: {e}")
                continue

    try:
        main_module = importlib.import_module(module_name)
    except ImportError as e:
        raise RuntimeError(f"Cannot import module {module_name}: {e}") from e

    _scan_module(module_name)

    if recursive and hasattr(main_module, "__path__"):
        try:
            import pkgutil

            for _importer, submod_name, _is_pkg in pkgutil.walk_packages(
                main_module.__path__, prefix=f"{module_name}."
            ):
                try:
                    _scan_module(submod_name)
                except Exception as e:
                    logger.debug(f"Failed to scan submodule {submod_name}: {e}")
                    continue
        except Exception as e:
            logger.debug(f"Failed to walk submodules: {e}")

    return tools


def _format_discovered_tools(tools: Dict[str, ToolInfo], module_name: str, verbose: bool = False) -> str:
    """Format discovered tools into a human-readable string.

    Args:
        tools: Dictionary of tool names to ToolInfo objects.
        module_name: The module that was scanned.
        verbose: If True, include parameter details in the output.

    Returns:
        A formatted string suitable for display to users.
    """
    if not tools:
        return f"No @tool decorated functions found in {module_name}"

    lines = [f"🔍 **{len(tools)} tools discovered in {module_name}:**\n"]

    by_module: Dict[str, List[ToolInfo]] = {}
    for info in tools.values():
        mod = info.module
        if mod not in by_module:
            by_module[mod] = []
        by_module[mod].append(info)

    for mod_name in sorted(by_module.keys()):
        mod_tools = sorted(by_module[mod_name], key=lambda x: x.name)

        if len(by_module) > 1:
            short_mod = mod_name.replace(module_name + ".", "")
            lines.append(f"\n**📦 {short_mod}:**")

        for info in mod_tools:
            lines.append(f"  • **{info.name}**: {info.description}")

            if verbose and info.parameters:
                params_str = []
                for p in info.parameters:
                    marker = "•" if p["required"] else "○"
                    params_str.append(f"{marker}{p['name']}:{p['type']}")
                if params_str:
                    lines.append(f"    └─ params: {', '.join(params_str)}")

    lines.append(f"\n💡 Load: `manage_tools(action='add', tools='{module_name}.TOOL_NAME')`")

    if verbose:
        lines.append("    (• = required, ○ = optional)")

    return "\n".join(lines)


def _sandbox_test(code: str) -> Dict[str, Any]:
    """Execute tool code in an isolated subprocess for validation.

    Tests the provided code for syntax errors, import issues, and verifies
    the presence of @tool decorated functions without affecting the current
    Python environment.

    Args:
        code: Python source code to validate.

    Returns:
        A dictionary containing:
            - success (bool): Whether the code passed all validation checks.
            - output (str): Combined stdout/stderr from the validation process.
            - has_tools (bool): Whether @tool decorated functions were detected.

    Example:
        >>> result = _sandbox_test("from strands import tool\\n@tool\\ndef test(): pass")
        >>> result["success"]
        True
        >>> result["has_tools"]
        True
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        test_code = f'''
import sys
sys.path.insert(0, "{Path(temp_path).parent}")
try:
    import ast
    with open("{temp_path}") as f:
        ast.parse(f.read())
    print("SYNTAX_OK")

    import importlib.util
    spec = importlib.util.spec_from_file_location("test_tool", "{temp_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("IMPORT_OK")

    tools_found = []
    for name in dir(module):
        obj = getattr(module, name)
        if hasattr(obj, "tool_name"):
            tools_found.append(obj.tool_name)
    if tools_found:
        print(f"TOOLS_FOUND: {{tools_found}}")
    else:
        print("NO_TOOLS_FOUND")
except SyntaxError as e:
    print(f"SYNTAX_ERROR: {{e}}")
except Exception as e:
    print(f"ERROR: {{e}}")
'''

        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = result.stdout + result.stderr
        success = "SYNTAX_OK" in output and "IMPORT_OK" in output

        return {
            "success": success,
            "output": output.strip(),
            "has_tools": "TOOLS_FOUND:" in output,
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "output": "Sandbox test timed out", "has_tools": False}
    except Exception as e:
        return {"success": False, "output": str(e), "has_tools": False}
    finally:
        Path(temp_path).unlink(missing_ok=True)


@tool
def manage_tools(
    action: str,
    tools: Optional[str] = None,
    code: Optional[str] = None,
    url: Optional[str] = None,
    name: Optional[str] = None,
    install: bool = False,
    verbose: bool = False,
    agent: Optional[Any] = None,
) -> Dict[str, Any]:
    """Manage agent tools at runtime - add, remove, create, fetch, discover.

    This tool provides god-level abilities to dynamically expand agent capabilities:
    - Load tools from packages, files, URLs, or GitHub
    - Create new tools on the fly from code
    - Discover available tools in any module
    - Sandbox test code before loading

    Supported Tool Formats (via SDK's process_tools):
    - Module path: "strands_tools.calculator"
    - Module with function: "my_package.module:my_function"
    - File path: "./tools/my_tool.py" or "/absolute/path/tool.py"
    - Multiple tools: "strands_tools.shell,strands_tools.editor"

    Args:
        action: The operation to perform. Valid values:
            - "list": Display all currently registered tools.
            - "add": Load tools from the specified tool specs.
            - "remove": Unregister tools by name.
            - "reload": Hot-reload tools to pick up code changes.
            - "create": Create and load a new tool from source code.
            - "fetch": Download and load a tool from a URL.
            - "discover": List all available tools in a module.
            - "sandbox": Test code in isolation without loading.
        tools: Tool specification(s), comma-separated. Usage varies by action:
            - For "add": Module paths, file paths, or module:function specs.
            - For "remove"/"reload": Tool names to operate on.
            - For "discover": Module name to introspect.
        code: Python source code for "create" and "sandbox" actions.
            Must contain at least one @tool decorated function.
        url: Remote URL for the "fetch" action. Supports GitHub blob URLs,
            Gist URLs, and direct raw file URLs.
        name: Optional custom name for tools created via "create" or "fetch".
            If not provided, the name is extracted from the source code.
        install: If True, automatically install missing PyPI packages before
            loading tools. Can also be enabled globally via the
            STRANDS_AUTO_INSTALL_TOOLS environment variable.
        verbose: If True, include detailed parameter information in "discover"
            output, showing types and required/optional status.
        agent: The Agent instance (automatically injected by the framework).

    Returns:
        A dictionary with the following structure:
            - status (str): "success" or "error"
            - content (list): List of dictionaries containing response text

    Raises:
        No exceptions are raised; errors are returned in the response dictionary.

    Example:
        List all registered tools::

            manage_tools(action="list")

        Add tools from a package with auto-installation::

            manage_tools(
                action="add",
                tools="strands_mlx.dataset_splitter",
                install=True
            )

        Fetch and load a tool from GitHub::

            manage_tools(
                action="fetch",
                url="https://github.com/user/repo/blob/main/my_tool.py"
            )

        Create a tool at runtime::

            manage_tools(
                action="create",
                code='''
                from strands import tool

                @tool
                def greet(name: str) -> str:
                    \"\"\"Greet someone by name.\"\"\"
                    return f"Hello, {name}!"
                '''
            )

        Discover tools with parameter details::

            manage_tools(
                action="discover",
                tools="strands_tools",
                verbose=True
            )

        Validate code before loading::

            manage_tools(
                action="sandbox",
                code="from strands import tool\\n@tool\\ndef test(): pass"
            )
    """
    if os.environ.get(ENV_DISABLE_LOAD, "").lower() == "true":
        if action in ("add", "reload", "create", "fetch"):
            return {
                "status": "error",
                "content": [{"text": f"⚠️ Dynamic tool loading disabled ({ENV_DISABLE_LOAD}=true)"}],
            }

    if not agent:
        return {"status": "error", "content": [{"text": "Agent not available"}]}

    registry = agent.tool_registry

    # =========================================================================
    # ACTION: list - Display all registered tools
    # =========================================================================
    if action == "list":
        tool_list = sorted(registry.registry.keys())
        dynamic = sorted(registry.dynamic_tools.keys())

        text = f"**{len(tool_list)} tools registered:**\n"
        text += "\n".join(f"  • {t}" + (" (dynamic)" if t in dynamic else "") for t in tool_list)

        return {"status": "success", "content": [{"text": text}]}

    # =========================================================================
    # ACTION: create - Generate new tools from source code
    # =========================================================================
    elif action == "create":
        if not code:
            return {"status": "error", "content": [{"text": "Required: 'code' parameter with Python source"}]}

        try:
            test_result = _sandbox_test(code)
            if not test_result["success"]:
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Code failed sandbox test:\n{test_result['output']}"}],
                }

            if not test_result["has_tools"]:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "❌ No @tool decorated functions found in code.\n\n"
                            "Example:\n```python\nfrom strands import tool\n\n"
                            "@tool\ndef my_tool(arg: str) -> str:\n"
                            "    '''Description'''\n    return result\n```"
                        }
                    ],
                }

            tool_path = _create_tool_file(code, name)
            added = registry.process_tools([str(tool_path)])

            if added:
                return {
                    "status": "success",
                    "content": [
                        {"text": f"✅ Created and loaded tool(s): {', '.join(added)}\n📁 Saved to: {tool_path}"}
                    ],
                }
            return {"status": "error", "content": [{"text": "Tool created but failed to load"}]}

        except Exception as e:
            logger.exception("Failed to create tool")
            return {"status": "error", "content": [{"text": f"❌ Failed to create tool: {e}"}]}

    # =========================================================================
    # ACTION: fetch - Download and load tools from remote URLs
    # =========================================================================
    elif action == "fetch":
        if not url:
            return {
                "status": "error",
                "content": [
                    {
                        "text": "Required: 'url' parameter\n\n"
                        "Supported formats:\n"
                        "  • GitHub: https://github.com/user/repo/blob/main/tool.py\n"
                        "  • Gist: https://gist.githubusercontent.com/user/id/raw/file.py\n"
                        "  • Raw: https://example.com/tool.py"
                    }
                ],
            }

        try:
            fetched_code = _fetch_from_url(url)

            test_result = _sandbox_test(fetched_code)
            if not test_result["success"]:
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Fetched code failed sandbox test:\n{test_result['output']}"}],
                }

            tool_path = _create_tool_file(fetched_code, name)
            added = registry.process_tools([str(tool_path)])

            if added:
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": f"✅ Fetched and loaded tool(s): {', '.join(added)}\n"
                            f"🌐 Source: {url}\n"
                            f"📁 Cached: {tool_path}"
                        }
                    ],
                }
            return {"status": "error", "content": [{"text": "Tool fetched but failed to load"}]}

        except Exception as e:
            logger.exception("Failed to fetch tool")
            return {"status": "error", "content": [{"text": f"❌ Failed to fetch tool: {e}"}]}

    # =========================================================================
    # ACTION: discover - Introspect modules for available tools
    # =========================================================================
    elif action == "discover":
        if not tools:
            return {
                "status": "error",
                "content": [{"text": "Required: 'tools' parameter with module name (e.g., 'strands_tools')"}],
            }

        module_name = tools.strip()
        auto_install = install or os.environ.get(ENV_AUTO_INSTALL, "").lower() == "true"

        try:
            discovered = _discover_tools_in_module(module_name, recursive=True)
        except RuntimeError as e:
            if not auto_install:
                return {"status": "error", "content": [{"text": str(e)}]}

            pkg_name = _extract_package_name(module_name)
            if pkg_name:
                install_results = _install_packages([pkg_name])
                if install_results.get(pkg_name):
                    try:
                        importlib.invalidate_caches()
                        discovered = _discover_tools_in_module(module_name, recursive=True)
                    except Exception as e2:
                        return {
                            "status": "error",
                            "content": [{"text": f"❌ Installed {pkg_name} but still can't import: {e2}"}],
                        }
                else:
                    return {"status": "error", "content": [{"text": f"❌ Failed to install {pkg_name}"}]}
            else:
                return {"status": "error", "content": [{"text": str(e)}]}

        output = _format_discovered_tools(discovered, module_name, verbose=verbose)
        return {"status": "success", "content": [{"text": output}]}

    # =========================================================================
    # ACTION: sandbox - Validate code in isolation
    # =========================================================================
    elif action == "sandbox":
        if not code:
            return {"status": "error", "content": [{"text": "Required: 'code' parameter with Python source to test"}]}

        result = _sandbox_test(code)

        if result["success"]:
            status_icon = "✅"
            msg = "Code passed sandbox test!"
            if result["has_tools"]:
                msg += "\n🔧 @tool decorated functions detected - ready to load"
            else:
                msg += "\n⚠️ No @tool decorated functions found"
        else:
            status_icon = "❌"
            msg = f"Code failed sandbox test:\n{result['output']}"

        return {"status": "success" if result["success"] else "error", "content": [{"text": f"{status_icon} {msg}"}]}

    # =========================================================================
    # ACTION: add - Load tools into the registry
    # =========================================================================
    elif action == "add":
        if not tools:
            return {"status": "error", "content": [{"text": "Required: tools parameter (comma-separated specs)"}]}

        tool_specs = [t.strip() for t in tools.split(",") if t.strip()]
        auto_install = install or os.environ.get(ENV_AUTO_INSTALL, "").lower() == "true"

        try:
            added = registry.process_tools(tool_specs)

            if added:
                return {
                    "status": "success",
                    "content": [{"text": f"✅ Added {len(added)} tools: {', '.join(added)}"}],
                }
            return {"status": "success", "content": [{"text": "No tools added"}]}

        except Exception as e:
            if not auto_install:
                logger.exception("Failed to add tools")
                return {"status": "error", "content": [{"text": f"❌ Failed to add tools: {e}"}]}

            packages = set()
            for spec in tool_specs:
                pkg = _extract_package_name(spec)
                if pkg:
                    packages.add(pkg)

            if not packages:
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Failed to add tools (no packages): {e}"}],
                }

            logger.info(f"Attempting to install packages: {packages}")
            install_results = _install_packages(list(packages))

            installed = [p for p, ok in install_results.items() if ok]
            failed_installs = [p for p, ok in install_results.items() if not ok]

            if not installed:
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Failed to install packages: {', '.join(failed_installs)}"}],
                }

            try:
                added = registry.process_tools(tool_specs)

                msg_parts = []
                if installed:
                    msg_parts.append(f"📦 Installed: {', '.join(installed)}")
                if added:
                    msg_parts.append(f"✅ Added {len(added)} tools: {', '.join(added)}")
                if failed_installs:
                    msg_parts.append(f"⚠️ Failed to install: {', '.join(failed_installs)}")

                return {"status": "success", "content": [{"text": "\n".join(msg_parts)}]}

            except Exception as e2:
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Installed packages but still failed to load: {e2}"}],
                }

    # =========================================================================
    # ACTION: remove - Unregister tools from the registry
    # =========================================================================
    elif action == "remove":
        if not tools:
            return {"status": "error", "content": [{"text": "Required: tools parameter (comma-separated names)"}]}

        names = [t.strip() for t in tools.split(",") if t.strip()]
        removed = []

        for tool_name in names:
            if tool_name in registry.registry:
                del registry.registry[tool_name]
                if tool_name in registry.dynamic_tools:
                    del registry.dynamic_tools[tool_name]
                removed.append(tool_name)
                logger.info(f"Removed tool: {tool_name}")

        if removed:
            return {
                "status": "success",
                "content": [{"text": f"✅ Removed {len(removed)} tools: {', '.join(removed)}"}],
            }
        return {"status": "success", "content": [{"text": f"Tools not found: {', '.join(names)}"}]}

    # =========================================================================
    # ACTION: reload - Hot-reload tools to pick up code changes
    # =========================================================================
    elif action == "reload":
        if not tools:
            return {"status": "error", "content": [{"text": "Required: tools parameter (comma-separated names)"}]}

        names = [t.strip() for t in tools.split(",") if t.strip()]
        reloaded = []
        errors = []

        for tool_name in names:
            if tool_name not in registry.registry:
                errors.append(f"{tool_name}: not found")
                continue

            tool_file = None
            for tools_dir in registry.get_tools_dirs():
                candidate = tools_dir / f"{tool_name}.py"
                if candidate.exists():
                    tool_file = candidate
                    break

            if tool_file:
                try:
                    registry.reload_tool(tool_name)
                    reloaded.append(tool_name)
                    logger.info(f"Reloaded file-based tool: {tool_name} from {tool_file}")
                except Exception as e:
                    errors.append(f"{tool_name}: {e}")
                continue

            tool_func = registry.registry.get(tool_name)
            module = getattr(tool_func, "__module__", None)

            if not module or module.startswith("__"):
                errors.append(f"{tool_name}: no module path available")
                continue

            module_path = module if module.endswith(f".{tool_name}") else f"{module}.{tool_name}"

            del registry.registry[tool_name]

            added = registry.process_tools([module_path])
            if added:
                reloaded.append(f"{tool_name} (reimported)")
                logger.info(f"Reloaded package tool: {tool_name} from {module_path}")
            else:
                errors.append(f"{tool_name}: reimport failed")

        parts = []
        if reloaded:
            parts.append(f"✅ Reloaded: {', '.join(reloaded)}")
        if errors:
            parts.append(f"❌ Failed: {'; '.join(errors)}")

        return {"status": "success" if reloaded else "error", "content": [{"text": "\n".join(parts)}]}

    # =========================================================================
    # Invalid action
    # =========================================================================
    else:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"Unknown action: {action}. "
                    "Valid: list, add, remove, reload, create, fetch, discover, sandbox"
                }
            ],
        }
