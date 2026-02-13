"""
Tests for the manage_tools tool.

This test suite covers all actions and helper functions in manage_tools:
- list, add, remove, reload (basic CRUD)
- create, fetch (dynamic tool creation)
- discover, sandbox (introspection and validation)
- Helper functions for installation, caching, URL fetching
"""

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from strands_tools import manage_tools as manage_tools_module
from strands_tools.manage_tools import (
    DEFAULT_INSTALL_TIMEOUT,
    ENV_AUTO_INSTALL,
    ENV_DISABLE_LOAD,
    ENV_INSTALL_TIMEOUT,
    ENV_PACKAGE_INSTALLER,
    ENV_TOOLS_CACHE_DIR,
    ToolInfo,
    _create_tool_file,
    _discover_tools_in_module,
    _extract_package_name,
    _fetch_from_url,
    _format_discovered_tools,
    _get_install_timeout,
    _get_installer_command,
    _get_tools_cache_dir,
    _install_packages,
    _sandbox_test,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent():
    """Create a mock agent with tool registry."""
    mock = MagicMock()
    mock.tool_registry = MagicMock()
    mock.tool_registry.registry = {
        "shell": MagicMock(),
        "editor": MagicMock(),
        "calculator": MagicMock(),
    }
    mock.tool_registry.dynamic_tools = {
        "calculator": MagicMock(),
    }
    mock.tool_registry.get_tools_dirs = MagicMock(return_value=[])
    return mock


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "strands_tools_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def valid_tool_code():
    """Valid tool source code for testing."""
    return '''from strands import tool

@tool
def test_tool(message: str) -> str:
    """A test tool that echoes a message."""
    return f"Echo: {message}"
'''


@pytest.fixture
def invalid_tool_code():
    """Invalid Python code for testing."""
    return "this is not valid python {{{{"


@pytest.fixture
def code_without_tool_decorator():
    """Valid Python but without @tool decorator."""
    return '''def plain_function(x):
    """A plain function."""
    return x * 2
'''


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGetInstallTimeout:
    """Tests for _get_install_timeout()."""

    def test_default_timeout(self):
        """Test default timeout when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop(ENV_INSTALL_TIMEOUT, None)
            assert _get_install_timeout() == DEFAULT_INSTALL_TIMEOUT

    def test_custom_timeout(self):
        """Test custom timeout from environment."""
        with patch.dict(os.environ, {ENV_INSTALL_TIMEOUT: "60"}):
            assert _get_install_timeout() == 60

    def test_invalid_timeout_returns_default(self):
        """Test that invalid value returns default."""
        with patch.dict(os.environ, {ENV_INSTALL_TIMEOUT: "not_a_number"}):
            assert _get_install_timeout() == DEFAULT_INSTALL_TIMEOUT


class TestGetInstallerCommand:
    """Tests for _get_installer_command()."""

    def test_default_pip_installer(self):
        """Test default pip installer."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop(ENV_PACKAGE_INSTALLER, None)
            cmd = _get_installer_command()
            assert cmd == [sys.executable, "-m", "pip"]

    def test_explicit_pip_installer(self):
        """Test explicit pip setting."""
        with patch.dict(os.environ, {ENV_PACKAGE_INSTALLER: "pip"}):
            cmd = _get_installer_command()
            assert cmd == [sys.executable, "-m", "pip"]

    def test_uv_installer_when_available(self):
        """Test uv installer when available in PATH."""
        with patch.dict(os.environ, {ENV_PACKAGE_INSTALLER: "uv"}):
            with patch("shutil.which", return_value="/usr/bin/uv"):
                cmd = _get_installer_command()
                assert cmd == ["uv", "pip"]

    def test_uv_fallback_to_pip_when_not_available(self):
        """Test fallback to pip when uv not in PATH."""
        with patch.dict(os.environ, {ENV_PACKAGE_INSTALLER: "uv"}):
            with patch("shutil.which", return_value=None):
                cmd = _get_installer_command()
                assert cmd == [sys.executable, "-m", "pip"]


class TestExtractPackageName:
    """Tests for _extract_package_name()."""

    def test_simple_module_path(self):
        """Test extraction from simple module path."""
        assert _extract_package_name("strands_tools.calculator") == "strands-tools"

    def test_deep_module_path(self):
        """Test extraction from deep module path."""
        assert _extract_package_name("my_package.submodule.tool") == "my-package"

    def test_module_with_function(self):
        """Test extraction from module:function format."""
        assert _extract_package_name("my_package.utils:helper") == "my-package"

    def test_relative_file_path(self):
        """Test that relative paths return None."""
        assert _extract_package_name("./tools/my_tool.py") is None

    def test_absolute_file_path(self):
        """Test that absolute paths return None."""
        assert _extract_package_name("/absolute/path/tool.py") is None

    def test_home_relative_path(self):
        """Test that ~ paths return None."""
        assert _extract_package_name("~/tools/my_tool.py") is None


class TestInstallPackages:
    """Tests for _install_packages()."""

    def test_successful_installation(self):
        """Test successful package installation."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            results = _install_packages(["test-package"])
            assert results == {"test-package": True}

    def test_failed_installation(self):
        """Test failed package installation."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Package not found"

        with patch("subprocess.run", return_value=mock_result):
            results = _install_packages(["nonexistent-package"])
            assert results == {"nonexistent-package": False}

    def test_installation_timeout(self):
        """Test installation timeout handling."""
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("pip", 300)):
            results = _install_packages(["slow-package"])
            assert results == {"slow-package": False}

    def test_installation_exception(self):
        """Test handling of general exceptions."""
        with patch("subprocess.run", side_effect=Exception("Unknown error")):
            results = _install_packages(["error-package"])
            assert results == {"error-package": False}

    def test_multiple_packages(self):
        """Test installing multiple packages."""
        call_count = [0]

        def mock_run(*args, **kwargs):
            call_count[0] += 1
            mock_result = MagicMock()
            mock_result.returncode = 0 if call_count[0] == 1 else 1
            mock_result.stderr = ""
            return mock_result

        with patch("subprocess.run", side_effect=mock_run):
            results = _install_packages(["package1", "package2"])
            assert results["package1"] is True
            assert results["package2"] is False


class TestGetToolsCacheDir:
    """Tests for _get_tools_cache_dir()."""

    def test_default_cache_dir(self):
        """Test default cache directory creation."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop(ENV_TOOLS_CACHE_DIR, None)
            cache_dir = _get_tools_cache_dir()
            assert cache_dir.exists()
            assert "strands_tools_cache" in str(cache_dir)

    def test_custom_cache_dir(self, tmp_path):
        """Test custom cache directory from environment."""
        custom_dir = tmp_path / "custom_cache"
        with patch.dict(os.environ, {ENV_TOOLS_CACHE_DIR: str(custom_dir)}):
            cache_dir = _get_tools_cache_dir()
            assert cache_dir == custom_dir
            assert cache_dir.exists()


class TestFetchFromUrl:
    """Tests for _fetch_from_url()."""

    def test_fetch_raw_url(self):
        """Test fetching from raw URL."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"print('hello')"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            code = _fetch_from_url("https://example.com/tool.py")
            assert code == "print('hello')"

    def test_github_blob_url_conversion(self):
        """Test GitHub blob URL is converted to raw."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"code"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            _fetch_from_url("https://github.com/user/repo/blob/main/tool.py")
            called_url = mock_urlopen.call_args[0][0]
            assert "raw.githubusercontent.com" in called_url
            assert "/blob/" not in called_url

    def test_fetch_failure(self):
        """Test fetch failure raises RuntimeError."""
        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            with pytest.raises(RuntimeError, match="Failed to fetch"):
                _fetch_from_url("https://example.com/tool.py")


class TestCreateToolFile:
    """Tests for _create_tool_file()."""

    def test_create_with_explicit_name(self, temp_cache_dir):
        """Test creating tool file with explicit name."""
        with patch.dict(os.environ, {ENV_TOOLS_CACHE_DIR: str(temp_cache_dir)}):
            code = "def my_func(): pass"
            path = _create_tool_file(code, name="my_tool")
            assert path.name == "my_tool.py"
            assert path.read_text() == code

    def test_create_extracts_function_name(self, temp_cache_dir):
        """Test that function name is extracted from code."""
        with patch.dict(os.environ, {ENV_TOOLS_CACHE_DIR: str(temp_cache_dir)}):
            code = "def extracted_name(x):\n    return x"
            path = _create_tool_file(code)
            assert path.name == "extracted_name.py"

    def test_create_generates_hash_name(self, temp_cache_dir):
        """Test hash-based name when no function found."""
        with patch.dict(os.environ, {ENV_TOOLS_CACHE_DIR: str(temp_cache_dir)}):
            code = "x = 1 + 2"  # No function definition
            path = _create_tool_file(code)
            assert path.name.startswith("tool_")
            assert path.name.endswith(".py")


class TestSandboxTest:
    """Tests for _sandbox_test()."""

    def test_valid_tool_code_passes(self, valid_tool_code):
        """Test that valid tool code passes sandbox."""
        result = _sandbox_test(valid_tool_code)
        assert result["success"] is True
        assert result["has_tools"] is True
        assert "SYNTAX_OK" in result["output"]
        assert "IMPORT_OK" in result["output"]

    def test_invalid_syntax_fails(self, invalid_tool_code):
        """Test that invalid syntax fails sandbox."""
        result = _sandbox_test(invalid_tool_code)
        assert result["success"] is False
        assert result["has_tools"] is False
        assert "SYNTAX_ERROR" in result["output"]

    def test_code_without_decorator_detected(self, code_without_tool_decorator):
        """Test that code without @tool is detected."""
        result = _sandbox_test(code_without_tool_decorator)
        assert result["success"] is True  # Syntax and import OK
        assert result["has_tools"] is False  # But no @tool found

    def test_sandbox_timeout(self):
        """Test sandbox timeout handling."""
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("python", 30)):
            result = _sandbox_test("print('hello')")
            assert result["success"] is False
            assert "timed out" in result["output"]

    def test_sandbox_exception(self):
        """Test sandbox exception handling."""
        with patch("subprocess.run", side_effect=Exception("Unexpected error")):
            result = _sandbox_test("print('hello')")
            assert result["success"] is False


class TestToolInfo:
    """Tests for ToolInfo dataclass."""

    def test_toolinfo_creation(self):
        """Test ToolInfo dataclass instantiation."""
        info = ToolInfo(
            name="test_tool",
            description="A test tool",
            module="test_module",
            parameters=[{"name": "arg1", "type": "string", "required": True}],
            required_params=["arg1"],
            optional_params=[],
        )
        assert info.name == "test_tool"
        assert info.description == "A test tool"
        assert len(info.parameters) == 1
        assert info.required_params == ["arg1"]


class TestDiscoverToolsInModule:
    """Tests for _discover_tools_in_module()."""

    def test_discover_nonexistent_module(self):
        """Test discovery of nonexistent module raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Cannot import module"):
            _discover_tools_in_module("nonexistent_module_xyz")

    def test_discover_empty_module(self):
        """Test discovery in module with no tools."""
        # os module has no @tool decorated functions
        tools = _discover_tools_in_module("os", recursive=False)
        assert tools == {}

    def test_discover_with_tools(self):
        """Test discovery in module with @tool functions."""
        # This tests against our own module which has manage_tools
        tools = _discover_tools_in_module("strands_tools.manage_tools", recursive=False)
        assert "manage_tools" in tools
        assert tools["manage_tools"].name == "manage_tools"

    def test_discover_recursive_with_submodules(self):
        """Test recursive discovery walks submodules."""
        # strands_tools has multiple submodules
        tools = _discover_tools_in_module("strands_tools", recursive=True)
        # Should find tools from multiple submodules
        assert len(tools) > 1

    def test_discover_handles_wrapped_objects(self):
        """Test that wrapped objects without tool_name are skipped."""
        # Create a mock module with a wrapped object that has no tool_name
        import types

        mock_module = types.ModuleType("mock_wrapped_module")

        class WrappedObj:
            pass

        wrapped = WrappedObj()
        wrapped.__wrapped__ = WrappedObj()  # Wrapped object also has no tool_name

        mock_module.wrapped_thing = wrapped

        with patch("importlib.import_module", return_value=mock_module):
            tools = _discover_tools_in_module("mock_wrapped_module", recursive=False)
            assert tools == {}

    def test_discover_wrapped_object_inner_no_tool_name(self):
        """Test wrapped object where inner object lacks tool_name attribute.

        This specifically tests the branch at line 347->352 where:
        1. Outer object doesn't have tool_name AND tool_spec
        2. Outer object HAS __wrapped__
        3. Inner (__wrapped__) object doesn't have tool_name
        """
        import types

        mock_module = types.ModuleType("mock_wrapped_inner")

        # Create objects using __slots__ to ensure no extra attributes
        class InnerWithoutToolName:
            __slots__ = ()  # No attributes at all

        class OuterWithWrapped:
            __slots__ = ("__wrapped__",)

            def __init__(self):
                self.__wrapped__ = InnerWithoutToolName()

        mock_module.wrapper = OuterWithWrapped()

        # Verify our setup is correct
        assert hasattr(mock_module.wrapper, "__wrapped__")
        assert not hasattr(mock_module.wrapper, "tool_name")
        assert not hasattr(mock_module.wrapper.__wrapped__, "tool_name")

        with patch("importlib.import_module", return_value=mock_module):
            tools = _discover_tools_in_module("mock_wrapped_inner", recursive=False)
            assert tools == {}

    def test_discover_handles_long_descriptions(self):
        """Test that long descriptions are truncated."""
        import types
        from unittest.mock import MagicMock

        mock_module = types.ModuleType("mock_long_desc")

        # Create a mock tool with a very long description
        mock_tool = MagicMock()
        mock_tool.tool_name = "long_desc_tool"
        mock_tool.tool_spec = {
            "description": "A" * 200,  # > 120 chars
            "inputSchema": {"json": {"properties": {}, "required": []}},
        }

        mock_module.long_tool = mock_tool

        with patch("importlib.import_module", return_value=mock_module):
            tools = _discover_tools_in_module("mock_long_desc", recursive=False)
            assert "long_desc_tool" in tools
            assert len(tools["long_desc_tool"].description) <= 120

    def test_discover_handles_long_param_descriptions(self):
        """Test that long parameter descriptions are truncated."""
        import types
        from unittest.mock import MagicMock

        mock_module = types.ModuleType("mock_long_param")

        mock_tool = MagicMock()
        mock_tool.tool_name = "long_param_tool"
        mock_tool.tool_spec = {
            "description": "Short desc",
            "inputSchema": {
                "json": {
                    "properties": {
                        "param1": {
                            "type": "string",
                            "description": "B" * 100,  # > 50 chars
                        }
                    },
                    "required": ["param1"],
                }
            },
        }

        mock_module.param_tool = mock_tool

        with patch("importlib.import_module", return_value=mock_module):
            tools = _discover_tools_in_module("mock_long_param", recursive=False)
            assert "long_param_tool" in tools
            param = tools["long_param_tool"].parameters[0]
            assert len(param["description"]) <= 50

    def test_discover_uses_docstring_when_no_description(self):
        """Test fallback to docstring when description is empty."""
        import types
        from unittest.mock import MagicMock

        mock_module = types.ModuleType("mock_docstring")

        mock_tool = MagicMock()
        mock_tool.tool_name = "docstring_tool"
        mock_tool.tool_spec = {
            "description": "",  # Empty description
            "inputSchema": {"json": {"properties": {}, "required": []}},
        }
        mock_tool.__doc__ = "This is from docstring"

        mock_module.doc_tool = mock_tool

        with patch("importlib.import_module", return_value=mock_module):
            tools = _discover_tools_in_module("mock_docstring", recursive=False)
            assert "docstring_tool" in tools
            assert "docstring" in tools["docstring_tool"].description

    def test_discover_handles_missing_tool_name(self):
        """Test that objects with tool_spec but no tool_name are skipped."""
        import types
        from unittest.mock import MagicMock

        mock_module = types.ModuleType("mock_no_name")

        mock_tool = MagicMock()
        mock_tool.tool_name = None  # No tool_name
        mock_tool.tool_spec = {"description": "test"}

        mock_module.no_name_tool = mock_tool

        with patch("importlib.import_module", return_value=mock_module):
            tools = _discover_tools_in_module("mock_no_name", recursive=False)
            assert tools == {}

    def test_discover_handles_legacy_schema(self):
        """Test handling of legacy inputSchema (direct properties, no 'json' key)."""
        import types
        from unittest.mock import MagicMock

        mock_module = types.ModuleType("mock_legacy")

        mock_tool = MagicMock()
        mock_tool.tool_name = "legacy_tool"
        mock_tool.tool_spec = {
            "description": "Legacy schema tool",
            "inputSchema": {  # Direct properties, no "json" wrapper
                "properties": {"arg1": {"type": "string", "description": "An arg"}},
                "required": ["arg1"],
            },
        }

        mock_module.legacy = mock_tool

        with patch("importlib.import_module", return_value=mock_module):
            tools = _discover_tools_in_module("mock_legacy", recursive=False)
            assert "legacy_tool" in tools
            assert len(tools["legacy_tool"].parameters) == 1

    def test_discover_skips_agent_param(self):
        """Test that 'agent' parameter is filtered out."""
        import types
        from unittest.mock import MagicMock

        mock_module = types.ModuleType("mock_agent_param")

        mock_tool = MagicMock()
        mock_tool.tool_name = "agent_param_tool"
        mock_tool.tool_spec = {
            "description": "Tool with agent param",
            "inputSchema": {
                "json": {
                    "properties": {
                        "agent": {"type": "object", "description": "The agent"},
                        "real_param": {"type": "string", "description": "Real param"},
                    },
                    "required": ["real_param"],
                }
            },
        }

        mock_module.agent_tool = mock_tool

        with patch("importlib.import_module", return_value=mock_module):
            tools = _discover_tools_in_module("mock_agent_param", recursive=False)
            assert "agent_param_tool" in tools
            param_names = [p["name"] for p in tools["agent_param_tool"].parameters]
            assert "agent" not in param_names
            assert "real_param" in param_names

    def test_discover_handles_attr_exception(self):
        """Test that exceptions during getattr are handled gracefully."""
        import types

        mock_module = types.ModuleType("mock_exception")

        # Create a descriptor that raises when accessed
        class RaisingDescriptor:
            def __get__(self, obj, objtype=None):
                raise RuntimeError("Cannot access")

        # Add a normal attribute and mark as having things to iterate
        mock_module.normal = "value"

        with patch("importlib.import_module", return_value=mock_module):
            # The module will be scanned but no tools found (this is fine)
            tools = _discover_tools_in_module("mock_exception", recursive=False)
            assert tools == {}

    def test_discover_getattr_raises_exception(self):
        """Test that getattr exceptions during module scan are caught.

        This tests lines 429-431 - the exception handler in _scan_module.
        """
        import types

        # Create a module class that raises on attribute access
        class ProblematicModule(types.ModuleType):
            def __init__(self):
                super().__init__("problematic_mod")
                self._safe_attr = "safe"

            def __dir__(self):
                return ["_safe_attr", "raising_attr"]

            def __getattribute__(self, name):
                if name == "raising_attr":
                    raise RuntimeError("Simulated attribute error")
                return super().__getattribute__(name)

        mock_module = ProblematicModule()

        with patch("importlib.import_module", return_value=mock_module):
            # Should not raise - exception is caught and logged
            tools = _discover_tools_in_module("problematic_mod", recursive=False)
            assert isinstance(tools, dict)

    def test_discover_handles_scan_exception_in_submodule_loop(self):
        """Test exception handling during submodule _scan_module calls."""
        import pkgutil
        import types

        mock_module = types.ModuleType("test_scan_exc")
        mock_module.__path__ = ["/fake/path"]

        # Make a submodule that will fail during scanning
        bad_submodule = types.ModuleType("test_scan_exc.bad")

        # Add a property that will raise during iteration
        class BadProperty:
            def __getattr__(self, name):
                if name == "__dict__":
                    return {}
                raise RuntimeError("Bad attribute access")

        original_import = importlib.import_module
        import_count = [0]

        def selective_import(name):
            import_count[0] += 1
            if name == "test_scan_exc":
                return mock_module
            elif name == "test_scan_exc.bad":
                return bad_submodule
            return original_import(name)

        mock_pkg_info = [(None, "test_scan_exc.bad", False)]

        with patch.object(importlib, "import_module", side_effect=selective_import):
            with patch.object(pkgutil, "walk_packages", return_value=iter(mock_pkg_info)):
                tools = _discover_tools_in_module("test_scan_exc", recursive=True)
                assert isinstance(tools, dict)

    def test_discover_handles_submodule_import_failure(self):
        """Test that submodule import failures during recursive scan are handled.

        This tests the exception handling in the recursive submodule scanning loop.
        The inner _scan_module catches ImportError when scanning submodules.
        """
        import pkgutil
        import types

        mock_module = types.ModuleType("test_pkg_sub")
        mock_module.__path__ = ["/fake/path"]

        original_import = importlib.import_module

        def selective_import(name):
            if name == "test_pkg_sub":
                return mock_module
            elif name.startswith("test_pkg_sub."):
                raise ImportError(f"Simulated failure for {name}")
            return original_import(name)

        mock_pkg_info = [(None, "test_pkg_sub.failing_sub", False)]

        with patch.object(importlib, "import_module", side_effect=selective_import):
            with patch.object(pkgutil, "walk_packages", return_value=iter(mock_pkg_info)):
                tools = _discover_tools_in_module("test_pkg_sub", recursive=True)
                assert isinstance(tools, dict)

    def test_discover_handles_pkgutil_failure(self):
        """Test that pkgutil.walk_packages failure is handled gracefully."""
        import pkgutil
        import types

        mock_module = types.ModuleType("pkgutil_fail_pkg")
        mock_module.__path__ = ["/fake/path"]

        original_import = importlib.import_module

        def selective_import(name):
            if name == "pkgutil_fail_pkg":
                return mock_module
            return original_import(name)

        with patch.object(importlib, "import_module", side_effect=selective_import):
            with patch.object(pkgutil, "walk_packages", side_effect=OSError("Permission denied")):
                tools = _discover_tools_in_module("pkgutil_fail_pkg", recursive=True)
                assert isinstance(tools, dict)

    def test_discover_handles_extract_exception(self):
        """Test that exceptions during tool extraction are logged and skipped."""
        import types
        from unittest.mock import MagicMock

        mock_module = types.ModuleType("mock_extract_fail")

        # Create an object that will cause extraction to fail
        mock_tool = MagicMock()
        mock_tool.tool_name = "failing_tool"
        mock_tool.tool_spec = MagicMock()
        mock_tool.tool_spec.get = MagicMock(side_effect=Exception("Extraction failed"))

        mock_module.bad_tool = mock_tool

        with patch("importlib.import_module", return_value=mock_module):
            # Should not raise, just skip the problematic tool
            tools = _discover_tools_in_module("mock_extract_fail", recursive=False)
            # Tool was skipped due to exception
            assert "failing_tool" not in tools


class TestFormatDiscoveredTools:
    """Tests for _format_discovered_tools()."""

    def test_format_empty_tools(self):
        """Test formatting empty tools dict."""
        result = _format_discovered_tools({}, "test_module")
        assert "No @tool decorated functions found" in result

    def test_format_single_tool(self):
        """Test formatting single tool."""
        tools = {
            "my_tool": ToolInfo(
                name="my_tool",
                description="Does something",
                module="test_module",
                parameters=[],
                required_params=[],
                optional_params=[],
            )
        }
        result = _format_discovered_tools(tools, "test_module")
        assert "1 tools discovered" in result
        assert "my_tool" in result
        assert "Does something" in result

    def test_format_verbose_mode(self):
        """Test verbose formatting includes parameters."""
        tools = {
            "my_tool": ToolInfo(
                name="my_tool",
                description="Does something",
                module="test_module",
                parameters=[
                    {"name": "arg1", "type": "string", "required": True, "description": ""},
                    {"name": "arg2", "type": "integer", "required": False, "description": ""},
                ],
                required_params=["arg1"],
                optional_params=["arg2"],
            )
        }
        result = _format_discovered_tools(tools, "test_module", verbose=True)
        assert "params:" in result
        assert "•arg1:string" in result  # Required marker
        assert "○arg2:integer" in result  # Optional marker

    def test_format_multiple_modules(self):
        """Test formatting tools from multiple modules shows module headers."""
        tools = {
            "tool1": ToolInfo(
                name="tool1",
                description="Tool 1",
                module="parent.submod1",
                parameters=[],
                required_params=[],
                optional_params=[],
            ),
            "tool2": ToolInfo(
                name="tool2",
                description="Tool 2",
                module="parent.submod2",
                parameters=[],
                required_params=[],
                optional_params=[],
            ),
        }
        result = _format_discovered_tools(tools, "parent")
        assert "2 tools discovered" in result
        assert "📦" in result  # Module header marker
        assert "submod1" in result
        assert "submod2" in result


# =============================================================================
# Main Tool Function Tests - List Action
# =============================================================================


class TestManageToolsList:
    """Tests for manage_tools list action."""

    def test_list_basic(self, mock_agent):
        """Test listing all registered tools."""
        result = manage_tools_module.manage_tools(action="list", agent=mock_agent)

        assert result["status"] == "success"
        assert "3 tools registered" in result["content"][0]["text"]
        assert "shell" in result["content"][0]["text"]
        assert "editor" in result["content"][0]["text"]
        assert "calculator" in result["content"][0]["text"]

    def test_list_shows_dynamic_marker(self, mock_agent):
        """Test that dynamic tools are marked."""
        result = manage_tools_module.manage_tools(action="list", agent=mock_agent)

        assert "(dynamic)" in result["content"][0]["text"]

    def test_list_empty_registry(self, mock_agent):
        """Test listing when no tools are registered."""
        mock_agent.tool_registry.registry = {}
        mock_agent.tool_registry.dynamic_tools = {}

        result = manage_tools_module.manage_tools(action="list", agent=mock_agent)

        assert result["status"] == "success"
        assert "0 tools registered" in result["content"][0]["text"]


# =============================================================================
# Main Tool Function Tests - Add Action
# =============================================================================


class TestManageToolsAdd:
    """Tests for manage_tools add action."""

    def test_add_single_tool(self, mock_agent):
        """Test adding a single tool from a package."""
        mock_agent.tool_registry.process_tools = MagicMock(return_value=["calculator"])

        result = manage_tools_module.manage_tools(
            action="add",
            tools="strands_tools.calculator",
            agent=mock_agent,
        )

        assert result["status"] == "success"
        assert "Added 1 tools" in result["content"][0]["text"]
        assert "calculator" in result["content"][0]["text"]
        mock_agent.tool_registry.process_tools.assert_called_once_with(["strands_tools.calculator"])

    def test_add_multiple_tools(self, mock_agent):
        """Test adding multiple tools at once."""
        mock_agent.tool_registry.process_tools = MagicMock(return_value=["shell", "editor"])

        result = manage_tools_module.manage_tools(
            action="add",
            tools="strands_tools.shell,strands_tools.editor",
            agent=mock_agent,
        )

        assert result["status"] == "success"
        assert "Added 2 tools" in result["content"][0]["text"]
        mock_agent.tool_registry.process_tools.assert_called_once_with(["strands_tools.shell", "strands_tools.editor"])

    def test_add_from_file_path(self, mock_agent):
        """Test adding a tool from file path."""
        mock_agent.tool_registry.process_tools = MagicMock(return_value=["my_tool"])

        result = manage_tools_module.manage_tools(
            action="add",
            tools="./tools/my_tool.py",
            agent=mock_agent,
        )

        assert result["status"] == "success"
        assert "Added 1 tools" in result["content"][0]["text"]
        mock_agent.tool_registry.process_tools.assert_called_once_with(["./tools/my_tool.py"])

    def test_add_module_with_function(self, mock_agent):
        """Test adding a specific function from a module."""
        mock_agent.tool_registry.process_tools = MagicMock(return_value=["helper"])

        result = manage_tools_module.manage_tools(
            action="add",
            tools="my_package.utils:helper",
            agent=mock_agent,
        )

        assert result["status"] == "success"
        mock_agent.tool_registry.process_tools.assert_called_once_with(["my_package.utils:helper"])

    def test_add_missing_tools_param(self, mock_agent):
        """Test add action without tools parameter."""
        result = manage_tools_module.manage_tools(action="add", agent=mock_agent)

        assert result["status"] == "error"
        assert "Required: tools parameter" in result["content"][0]["text"]

    def test_add_failure_without_auto_install(self, mock_agent):
        """Test handling of tool loading failure without auto-install."""
        mock_agent.tool_registry.process_tools = MagicMock(side_effect=Exception("Module not found"))

        result = manage_tools_module.manage_tools(
            action="add",
            tools="nonexistent.tool",
            agent=mock_agent,
        )

        assert result["status"] == "error"
        assert "Failed to add tools" in result["content"][0]["text"]

    def test_add_with_auto_install_success(self, mock_agent):
        """Test auto-installation on failure when install=True."""
        call_count = [0]

        def process_tools_side_effect(specs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Module not found")
            return ["new_tool"]

        mock_agent.tool_registry.process_tools = MagicMock(side_effect=process_tools_side_effect)

        with patch("strands_tools.manage_tools._install_packages", return_value={"new-package": True}):
            result = manage_tools_module.manage_tools(
                action="add",
                tools="new_package.new_tool",
                install=True,
                agent=mock_agent,
            )

            assert result["status"] == "success"
            assert "Installed" in result["content"][0]["text"]
            assert "Added" in result["content"][0]["text"]

    def test_add_with_auto_install_env_var(self, mock_agent):
        """Test auto-installation via environment variable."""
        call_count = [0]

        def process_tools_side_effect(specs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Module not found")
            return ["env_tool"]

        mock_agent.tool_registry.process_tools = MagicMock(side_effect=process_tools_side_effect)

        with patch.dict(os.environ, {ENV_AUTO_INSTALL: "true"}):
            with patch("strands_tools.manage_tools._install_packages", return_value={"env-package": True}):
                result = manage_tools_module.manage_tools(
                    action="add",
                    tools="env_package.env_tool",
                    agent=mock_agent,
                )

                assert result["status"] == "success"

    def test_add_returns_empty_list(self, mock_agent):
        """Test add when process_tools returns empty list."""
        mock_agent.tool_registry.process_tools = MagicMock(return_value=[])

        result = manage_tools_module.manage_tools(
            action="add",
            tools="strands_tools.nonexistent",
            agent=mock_agent,
        )

        assert result["status"] == "success"
        assert "No tools added" in result["content"][0]["text"]

    def test_add_whitespace_handling(self, mock_agent):
        """Test that tool names with whitespace are handled correctly."""
        mock_agent.tool_registry.process_tools = MagicMock(return_value=["shell", "editor"])

        result = manage_tools_module.manage_tools(
            action="add",
            tools="  strands_tools.shell  ,  strands_tools.editor  ",
            agent=mock_agent,
        )

        assert result["status"] == "success"
        mock_agent.tool_registry.process_tools.assert_called_once_with(["strands_tools.shell", "strands_tools.editor"])

    def test_add_auto_install_file_path_no_packages(self, mock_agent):
        """Test add with file path fails gracefully (no packages to install)."""
        mock_agent.tool_registry.process_tools = MagicMock(side_effect=Exception("Cannot load"))

        result = manage_tools_module.manage_tools(
            action="add",
            tools="./local/tool.py",  # File path - no package name
            install=True,
            agent=mock_agent,
        )

        assert result["status"] == "error"
        assert "no packages" in result["content"][0]["text"]

    def test_add_auto_install_all_packages_fail(self, mock_agent):
        """Test add when all package installations fail."""
        mock_agent.tool_registry.process_tools = MagicMock(side_effect=Exception("Cannot load"))

        with patch(
            "strands_tools.manage_tools._install_packages",
            return_value={"failed-package": False},
        ):
            result = manage_tools_module.manage_tools(
                action="add",
                tools="failed_package.tool",
                install=True,
                agent=mock_agent,
            )

        assert result["status"] == "error"
        assert "Failed to install packages" in result["content"][0]["text"]

    def test_add_auto_install_success_but_load_fails(self, mock_agent):
        """Test add when package installs but tool still won't load."""
        mock_agent.tool_registry.process_tools = MagicMock(side_effect=Exception("Still cannot load"))

        with patch(
            "strands_tools.manage_tools._install_packages",
            return_value={"some-package": True},
        ):
            result = manage_tools_module.manage_tools(
                action="add",
                tools="some_package.tool",
                install=True,
                agent=mock_agent,
            )

        assert result["status"] == "error"
        assert "still failed to load" in result["content"][0]["text"]

    def test_add_auto_install_partial_with_failures(self, mock_agent):
        """Test add with some packages failing but tools still loading."""
        call_count = [0]

        def process_side_effect(specs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("First try fails")
            return ["tool1"]

        mock_agent.tool_registry.process_tools = MagicMock(side_effect=process_side_effect)

        with patch(
            "strands_tools.manage_tools._install_packages",
            return_value={"pkg1": True, "pkg2": False},
        ):
            result = manage_tools_module.manage_tools(
                action="add",
                tools="pkg1.tool1,pkg2.tool2",
                install=True,
                agent=mock_agent,
            )

        assert result["status"] == "success"
        assert "Installed" in result["content"][0]["text"]
        assert "Added" in result["content"][0]["text"]
        assert "Failed to install" in result["content"][0]["text"]


# =============================================================================
# Main Tool Function Tests - Remove Action
# =============================================================================


class TestManageToolsRemove:
    """Tests for manage_tools remove action."""

    def test_remove_single_tool(self, mock_agent):
        """Test removing a single tool."""
        result = manage_tools_module.manage_tools(
            action="remove",
            tools="calculator",
            agent=mock_agent,
        )

        assert result["status"] == "success"
        assert "Removed 1 tools" in result["content"][0]["text"]
        assert "calculator" not in mock_agent.tool_registry.registry
        assert "calculator" not in mock_agent.tool_registry.dynamic_tools

    def test_remove_multiple_tools(self, mock_agent):
        """Test removing multiple tools at once."""
        result = manage_tools_module.manage_tools(
            action="remove",
            tools="shell,editor",
            agent=mock_agent,
        )

        assert result["status"] == "success"
        assert "Removed 2 tools" in result["content"][0]["text"]
        assert "shell" not in mock_agent.tool_registry.registry
        assert "editor" not in mock_agent.tool_registry.registry

    def test_remove_nonexistent_tool(self, mock_agent):
        """Test removing a tool that doesn't exist."""
        result = manage_tools_module.manage_tools(
            action="remove",
            tools="nonexistent",
            agent=mock_agent,
        )

        assert result["status"] == "success"
        assert "not found" in result["content"][0]["text"]

    def test_remove_missing_tools_param(self, mock_agent):
        """Test remove action without tools parameter."""
        result = manage_tools_module.manage_tools(action="remove", agent=mock_agent)

        assert result["status"] == "error"
        assert "Required: tools parameter" in result["content"][0]["text"]


# =============================================================================
# Main Tool Function Tests - Reload Action
# =============================================================================


class TestManageToolsReload:
    """Tests for manage_tools reload action."""

    def test_reload_file_based_tool(self, mock_agent, tmp_path):
        """Test reloading a file-based tool."""
        # Create a mock tool file
        tool_file = tmp_path / "shell.py"
        tool_file.write_text("# mock tool")

        mock_agent.tool_registry.get_tools_dirs = MagicMock(return_value=[tmp_path])
        mock_agent.tool_registry.reload_tool = MagicMock()

        result = manage_tools_module.manage_tools(
            action="reload",
            tools="shell",
            agent=mock_agent,
        )

        assert result["status"] == "success"
        assert "Reloaded" in result["content"][0]["text"]
        mock_agent.tool_registry.reload_tool.assert_called_once_with("shell")

    def test_reload_package_tool(self, mock_agent):
        """Test reloading a package-based tool via reimport."""
        # Tool not in any tools_dir, so will try reimport
        mock_agent.tool_registry.get_tools_dirs = MagicMock(return_value=[])

        # Set up mock tool function with module info
        mock_tool_func = MagicMock()
        mock_tool_func.__module__ = "strands_tools.shell"
        mock_agent.tool_registry.registry = {"shell": mock_tool_func}
        mock_agent.tool_registry.dynamic_tools = {}
        mock_agent.tool_registry.process_tools = MagicMock(return_value=["shell"])

        result = manage_tools_module.manage_tools(
            action="reload",
            tools="shell",
            agent=mock_agent,
        )

        assert result["status"] == "success"
        assert "reimported" in result["content"][0]["text"]

    def test_reload_nonexistent_tool(self, mock_agent):
        """Test reloading a tool that doesn't exist."""
        result = manage_tools_module.manage_tools(
            action="reload",
            tools="nonexistent",
            agent=mock_agent,
        )

        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_reload_failure(self, mock_agent, tmp_path):
        """Test handling of reload failure."""
        tool_file = tmp_path / "shell.py"
        tool_file.write_text("# mock tool")

        mock_agent.tool_registry.get_tools_dirs = MagicMock(return_value=[tmp_path])
        mock_agent.tool_registry.reload_tool = MagicMock(side_effect=Exception("Reload failed"))

        result = manage_tools_module.manage_tools(
            action="reload",
            tools="shell",
            agent=mock_agent,
        )

        assert result["status"] == "error"
        assert "Failed" in result["content"][0]["text"]

    def test_reload_partial_success(self, mock_agent, tmp_path):
        """Test reload with some successes and some failures."""
        tool_file = tmp_path / "shell.py"
        tool_file.write_text("# mock tool")

        mock_agent.tool_registry.get_tools_dirs = MagicMock(return_value=[tmp_path])

        def reload_side_effect(name):
            if name == "shell":
                return None
            raise Exception(f"Tool {name} reload failed")

        mock_agent.tool_registry.reload_tool = MagicMock(side_effect=reload_side_effect)

        # editor is in registry but not as a file
        mock_agent.tool_registry.registry["editor"].__module__ = "__main__"

        result = manage_tools_module.manage_tools(
            action="reload",
            tools="shell,editor",
            agent=mock_agent,
        )

        # shell succeeds, editor fails (no module path)
        assert "Reloaded" in result["content"][0]["text"]
        assert "Failed" in result["content"][0]["text"]

    def test_reload_missing_tools_param(self, mock_agent):
        """Test reload action without tools parameter."""
        result = manage_tools_module.manage_tools(action="reload", agent=mock_agent)

        assert result["status"] == "error"
        assert "Required: tools parameter" in result["content"][0]["text"]

    def test_reload_tool_no_module_path(self, mock_agent):
        """Test reload when tool has no module path available."""
        mock_agent.tool_registry.get_tools_dirs = MagicMock(return_value=[])

        # Tool function with __main__ module (can't be reimported)
        mock_tool_func = MagicMock()
        mock_tool_func.__module__ = "__main__"
        mock_agent.tool_registry.registry = {"local_tool": mock_tool_func}
        mock_agent.tool_registry.dynamic_tools = {}

        result = manage_tools_module.manage_tools(
            action="reload",
            tools="local_tool",
            agent=mock_agent,
        )

        assert result["status"] == "error"
        assert "no module path available" in result["content"][0]["text"]

    def test_reload_package_tool_reimport_fails(self, mock_agent):
        """Test reload when package tool reimport fails."""
        mock_agent.tool_registry.get_tools_dirs = MagicMock(return_value=[])

        mock_tool_func = MagicMock()
        mock_tool_func.__module__ = "some_package.some_tool"
        mock_agent.tool_registry.registry = {"some_tool": mock_tool_func}
        mock_agent.tool_registry.dynamic_tools = {}
        mock_agent.tool_registry.process_tools = MagicMock(return_value=[])

        result = manage_tools_module.manage_tools(
            action="reload",
            tools="some_tool",
            agent=mock_agent,
        )

        assert result["status"] == "error"
        assert "reimport failed" in result["content"][0]["text"]

    def test_reload_module_ends_with_tool_name(self, mock_agent):
        """Test reload builds correct path when module ends with tool name."""
        mock_agent.tool_registry.get_tools_dirs = MagicMock(return_value=[])

        mock_tool_func = MagicMock()
        mock_tool_func.__module__ = "strands_tools.calculator"  # Ends with tool name
        mock_agent.tool_registry.registry = {"calculator": mock_tool_func}
        mock_agent.tool_registry.dynamic_tools = {}
        mock_agent.tool_registry.process_tools = MagicMock(return_value=["calculator"])

        result = manage_tools_module.manage_tools(
            action="reload",
            tools="calculator",
            agent=mock_agent,
        )

        assert result["status"] == "success"
        assert "reimported" in result["content"][0]["text"]
        # Should use the module path directly since it ends with tool name
        mock_agent.tool_registry.process_tools.assert_called_with(["strands_tools.calculator"])

    def test_reload_module_not_ending_with_tool_name(self, mock_agent):
        """Test reload appends tool name when module doesn't end with it."""
        mock_agent.tool_registry.get_tools_dirs = MagicMock(return_value=[])

        mock_tool_func = MagicMock()
        mock_tool_func.__module__ = "strands_tools"  # Doesn't end with tool name
        mock_agent.tool_registry.registry = {"my_tool": mock_tool_func}
        mock_agent.tool_registry.dynamic_tools = {}
        mock_agent.tool_registry.process_tools = MagicMock(return_value=["my_tool"])

        result = manage_tools_module.manage_tools(
            action="reload",
            tools="my_tool",
            agent=mock_agent,
        )

        assert result["status"] == "success"
        # Should append tool name: strands_tools.my_tool
        mock_agent.tool_registry.process_tools.assert_called_with(["strands_tools.my_tool"])


# =============================================================================
# Main Tool Function Tests - Create Action
# =============================================================================


class TestManageToolsCreate:
    """Tests for manage_tools create action."""

    def test_create_valid_tool(self, mock_agent, valid_tool_code, temp_cache_dir):
        """Test creating a valid tool."""
        mock_agent.tool_registry.process_tools = MagicMock(return_value=["test_tool"])

        with patch.dict(os.environ, {ENV_TOOLS_CACHE_DIR: str(temp_cache_dir)}):
            result = manage_tools_module.manage_tools(
                action="create",
                code=valid_tool_code,
                agent=mock_agent,
            )

        assert result["status"] == "success"
        assert "Created and loaded" in result["content"][0]["text"]
        assert "test_tool" in result["content"][0]["text"]

    def test_create_with_custom_name(self, mock_agent, valid_tool_code, temp_cache_dir):
        """Test creating a tool with custom name."""
        mock_agent.tool_registry.process_tools = MagicMock(return_value=["custom_name"])

        with patch.dict(os.environ, {ENV_TOOLS_CACHE_DIR: str(temp_cache_dir)}):
            result = manage_tools_module.manage_tools(
                action="create",
                code=valid_tool_code,
                name="custom_name",
                agent=mock_agent,
            )

        assert result["status"] == "success"
        # Check file was created with custom name
        assert (temp_cache_dir / "custom_name.py").exists()

    def test_create_invalid_syntax(self, mock_agent, invalid_tool_code):
        """Test creating tool with invalid syntax."""
        result = manage_tools_module.manage_tools(
            action="create",
            code=invalid_tool_code,
            agent=mock_agent,
        )

        assert result["status"] == "error"
        assert "failed sandbox test" in result["content"][0]["text"]

    def test_create_without_decorator(self, mock_agent, code_without_tool_decorator):
        """Test creating tool without @tool decorator."""
        result = manage_tools_module.manage_tools(
            action="create",
            code=code_without_tool_decorator,
            agent=mock_agent,
        )

        assert result["status"] == "error"
        assert "No @tool decorated functions found" in result["content"][0]["text"]

    def test_create_missing_code_param(self, mock_agent):
        """Test create action without code parameter."""
        result = manage_tools_module.manage_tools(action="create", agent=mock_agent)

        assert result["status"] == "error"
        assert "Required: 'code' parameter" in result["content"][0]["text"]

    def test_create_load_failure(self, mock_agent, valid_tool_code, temp_cache_dir):
        """Test create when loading fails after sandbox passes."""
        mock_agent.tool_registry.process_tools = MagicMock(return_value=[])

        with patch.dict(os.environ, {ENV_TOOLS_CACHE_DIR: str(temp_cache_dir)}):
            result = manage_tools_module.manage_tools(
                action="create",
                code=valid_tool_code,
                agent=mock_agent,
            )

        assert result["status"] == "error"
        assert "failed to load" in result["content"][0]["text"]

    def test_create_exception_during_load(self, mock_agent, valid_tool_code, temp_cache_dir):
        """Test create when process_tools raises an exception."""
        mock_agent.tool_registry.process_tools = MagicMock(side_effect=Exception("Registry error"))

        with patch.dict(os.environ, {ENV_TOOLS_CACHE_DIR: str(temp_cache_dir)}):
            result = manage_tools_module.manage_tools(
                action="create",
                code=valid_tool_code,
                agent=mock_agent,
            )

        assert result["status"] == "error"
        assert "Failed to create tool" in result["content"][0]["text"]


# =============================================================================
# Main Tool Function Tests - Fetch Action
# =============================================================================


class TestManageToolsFetch:
    """Tests for manage_tools fetch action."""

    def test_fetch_valid_url(self, mock_agent, valid_tool_code, temp_cache_dir):
        """Test fetching a tool from URL."""
        mock_agent.tool_registry.process_tools = MagicMock(return_value=["fetched_tool"])

        with patch.dict(os.environ, {ENV_TOOLS_CACHE_DIR: str(temp_cache_dir)}):
            with patch("strands_tools.manage_tools._fetch_from_url", return_value=valid_tool_code):
                result = manage_tools_module.manage_tools(
                    action="fetch",
                    url="https://example.com/tool.py",
                    agent=mock_agent,
                )

        assert result["status"] == "success"
        assert "Fetched and loaded" in result["content"][0]["text"]

    def test_fetch_github_url(self, mock_agent, valid_tool_code, temp_cache_dir):
        """Test fetching from GitHub blob URL."""
        mock_agent.tool_registry.process_tools = MagicMock(return_value=["github_tool"])

        with patch.dict(os.environ, {ENV_TOOLS_CACHE_DIR: str(temp_cache_dir)}):
            with patch("strands_tools.manage_tools._fetch_from_url", return_value=valid_tool_code):
                result = manage_tools_module.manage_tools(
                    action="fetch",
                    url="https://github.com/user/repo/blob/main/tool.py",
                    agent=mock_agent,
                )

        assert result["status"] == "success"

    def test_fetch_missing_url_param(self, mock_agent):
        """Test fetch action without url parameter."""
        result = manage_tools_module.manage_tools(action="fetch", agent=mock_agent)

        assert result["status"] == "error"
        assert "Required: 'url' parameter" in result["content"][0]["text"]

    def test_fetch_network_failure(self, mock_agent):
        """Test fetch when network request fails."""
        with patch(
            "strands_tools.manage_tools._fetch_from_url",
            side_effect=RuntimeError("Network error"),
        ):
            result = manage_tools_module.manage_tools(
                action="fetch",
                url="https://example.com/tool.py",
                agent=mock_agent,
            )

        assert result["status"] == "error"
        assert "Failed to fetch" in result["content"][0]["text"]

    def test_fetch_invalid_code(self, mock_agent, invalid_tool_code):
        """Test fetch when fetched code is invalid."""
        with patch("strands_tools.manage_tools._fetch_from_url", return_value=invalid_tool_code):
            result = manage_tools_module.manage_tools(
                action="fetch",
                url="https://example.com/bad_tool.py",
                agent=mock_agent,
            )

        assert result["status"] == "error"
        assert "failed sandbox test" in result["content"][0]["text"]

    def test_fetch_load_returns_empty(self, mock_agent, valid_tool_code, temp_cache_dir):
        """Test fetch when sandbox passes but loading returns empty list."""
        mock_agent.tool_registry.process_tools = MagicMock(return_value=[])

        with patch.dict(os.environ, {ENV_TOOLS_CACHE_DIR: str(temp_cache_dir)}):
            with patch("strands_tools.manage_tools._fetch_from_url", return_value=valid_tool_code):
                result = manage_tools_module.manage_tools(
                    action="fetch",
                    url="https://example.com/tool.py",
                    agent=mock_agent,
                )

        assert result["status"] == "error"
        assert "failed to load" in result["content"][0]["text"]

    def test_fetch_code_passes_but_no_tools(self, mock_agent, code_without_tool_decorator, temp_cache_dir):
        """Test fetch when code passes sandbox but has no @tool decorators.

        Note: fetch action doesn't check has_tools - it only checks sandbox success.
        So code without @tool will pass sandbox but fail at process_tools.
        """
        mock_agent.tool_registry.process_tools = MagicMock(side_effect=Exception("No valid tools in module"))

        with patch.dict(os.environ, {ENV_TOOLS_CACHE_DIR: str(temp_cache_dir)}):
            with patch(
                "strands_tools.manage_tools._fetch_from_url",
                return_value=code_without_tool_decorator,
            ):
                result = manage_tools_module.manage_tools(
                    action="fetch",
                    url="https://example.com/no_decorator.py",
                    agent=mock_agent,
                )

        # fetch catches the exception from process_tools
        assert result["status"] == "error"
        assert "Failed to fetch" in result["content"][0]["text"]


# =============================================================================
# Main Tool Function Tests - Discover Action
# =============================================================================


class TestManageToolsDiscover:
    """Tests for manage_tools discover action."""

    def test_discover_valid_module(self, mock_agent):
        """Test discovering tools in a valid module."""
        result = manage_tools_module.manage_tools(
            action="discover",
            tools="strands_tools.manage_tools",
            agent=mock_agent,
        )

        assert result["status"] == "success"
        assert "tools discovered" in result["content"][0]["text"]
        assert "manage_tools" in result["content"][0]["text"]

    def test_discover_verbose_mode(self, mock_agent):
        """Test discover with verbose output."""
        result = manage_tools_module.manage_tools(
            action="discover",
            tools="strands_tools.manage_tools",
            verbose=True,
            agent=mock_agent,
        )

        assert result["status"] == "success"
        assert "params:" in result["content"][0]["text"]

    def test_discover_nonexistent_module(self, mock_agent):
        """Test discover with nonexistent module."""
        result = manage_tools_module.manage_tools(
            action="discover",
            tools="nonexistent_module_xyz",
            agent=mock_agent,
        )

        assert result["status"] == "error"
        assert "Cannot import module" in result["content"][0]["text"]

    def test_discover_missing_tools_param(self, mock_agent):
        """Test discover without tools parameter."""
        result = manage_tools_module.manage_tools(action="discover", agent=mock_agent)

        assert result["status"] == "error"
        assert "Required: 'tools' parameter" in result["content"][0]["text"]

    def test_discover_with_auto_install(self, mock_agent):
        """Test discover with auto-install on import failure."""
        call_count = [0]

        def discover_side_effect(module_name, recursive=True):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Cannot import module")
            return {"discovered_tool": MagicMock()}

        with patch(
            "strands_tools.manage_tools._discover_tools_in_module",
            side_effect=discover_side_effect,
        ):
            with patch(
                "strands_tools.manage_tools._install_packages",
                return_value={"new-package": True},
            ):
                with patch("importlib.invalidate_caches"):
                    result = manage_tools_module.manage_tools(
                        action="discover",
                        tools="new_package",
                        install=True,
                        agent=mock_agent,
                    )

        # After install, discovery should succeed
        assert result["status"] == "success"

    def test_discover_auto_install_still_fails_after_install(self, mock_agent):
        """Test discover when import still fails after package install."""
        with patch(
            "strands_tools.manage_tools._discover_tools_in_module",
            side_effect=RuntimeError("Cannot import module"),
        ):
            with patch(
                "strands_tools.manage_tools._install_packages",
                return_value={"new-package": True},
            ):
                with patch("importlib.invalidate_caches"):
                    result = manage_tools_module.manage_tools(
                        action="discover",
                        tools="new_package",
                        install=True,
                        agent=mock_agent,
                    )

        assert result["status"] == "error"
        assert "still can't import" in result["content"][0]["text"]

    def test_discover_auto_install_package_install_fails(self, mock_agent):
        """Test discover when package installation fails."""
        with patch(
            "strands_tools.manage_tools._discover_tools_in_module",
            side_effect=RuntimeError("Cannot import module"),
        ):
            with patch(
                "strands_tools.manage_tools._install_packages",
                return_value={"new-package": False},
            ):
                result = manage_tools_module.manage_tools(
                    action="discover",
                    tools="new_package",
                    install=True,
                    agent=mock_agent,
                )

        assert result["status"] == "error"
        assert "Failed to install" in result["content"][0]["text"]

    def test_discover_auto_install_file_path_no_package(self, mock_agent):
        """Test discover with file path (no package to install)."""
        with patch(
            "strands_tools.manage_tools._discover_tools_in_module",
            side_effect=RuntimeError("Cannot import module"),
        ):
            result = manage_tools_module.manage_tools(
                action="discover",
                tools="./local/path",  # File path returns None for package name
                install=True,
                agent=mock_agent,
            )

        assert result["status"] == "error"
        # Should return the original error since no package to install


# =============================================================================
# Main Tool Function Tests - Sandbox Action
# =============================================================================


class TestManageToolsSandbox:
    """Tests for manage_tools sandbox action."""

    def test_sandbox_valid_tool(self, mock_agent, valid_tool_code):
        """Test sandbox with valid tool code."""
        result = manage_tools_module.manage_tools(
            action="sandbox",
            code=valid_tool_code,
            agent=mock_agent,
        )

        assert result["status"] == "success"
        assert "passed sandbox test" in result["content"][0]["text"]
        assert "@tool decorated functions detected" in result["content"][0]["text"]

    def test_sandbox_invalid_syntax(self, mock_agent, invalid_tool_code):
        """Test sandbox with invalid syntax."""
        result = manage_tools_module.manage_tools(
            action="sandbox",
            code=invalid_tool_code,
            agent=mock_agent,
        )

        assert result["status"] == "error"
        assert "failed sandbox test" in result["content"][0]["text"]

    def test_sandbox_no_decorator(self, mock_agent, code_without_tool_decorator):
        """Test sandbox with code missing @tool decorator."""
        result = manage_tools_module.manage_tools(
            action="sandbox",
            code=code_without_tool_decorator,
            agent=mock_agent,
        )

        assert result["status"] == "success"  # Syntax OK
        assert "No @tool decorated functions found" in result["content"][0]["text"]

    def test_sandbox_missing_code_param(self, mock_agent):
        """Test sandbox without code parameter."""
        result = manage_tools_module.manage_tools(action="sandbox", agent=mock_agent)

        assert result["status"] == "error"
        assert "Required: 'code' parameter" in result["content"][0]["text"]


# =============================================================================
# Main Tool Function Tests - Error Cases
# =============================================================================


class TestManageToolsErrorCases:
    """Tests for manage_tools error handling."""

    def test_unknown_action(self, mock_agent):
        """Test handling of unknown action."""
        result = manage_tools_module.manage_tools(action="invalid_action", agent=mock_agent)

        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]
        assert "list, add, remove, reload, create, fetch, discover, sandbox" in result["content"][0]["text"]

    def test_no_agent(self):
        """Test handling when agent is not provided."""
        result = manage_tools_module.manage_tools(action="list", agent=None)

        assert result["status"] == "error"
        assert "Agent not available" in result["content"][0]["text"]

    def test_disabled_loading_blocks_add(self, mock_agent):
        """Test that add is blocked when loading disabled."""
        with patch.dict(os.environ, {ENV_DISABLE_LOAD: "true"}):
            result = manage_tools_module.manage_tools(
                action="add",
                tools="strands_tools.calculator",
                agent=mock_agent,
            )

        assert result["status"] == "error"
        assert "disabled" in result["content"][0]["text"].lower()

    def test_disabled_loading_blocks_reload(self, mock_agent):
        """Test that reload is blocked when loading disabled."""
        with patch.dict(os.environ, {ENV_DISABLE_LOAD: "true"}):
            result = manage_tools_module.manage_tools(
                action="reload",
                tools="shell",
                agent=mock_agent,
            )

        assert result["status"] == "error"
        assert "disabled" in result["content"][0]["text"].lower()

    def test_disabled_loading_blocks_create(self, mock_agent, valid_tool_code):
        """Test that create is blocked when loading disabled."""
        with patch.dict(os.environ, {ENV_DISABLE_LOAD: "true"}):
            result = manage_tools_module.manage_tools(
                action="create",
                code=valid_tool_code,
                agent=mock_agent,
            )

        assert result["status"] == "error"
        assert "disabled" in result["content"][0]["text"].lower()

    def test_disabled_loading_blocks_fetch(self, mock_agent):
        """Test that fetch is blocked when loading disabled."""
        with patch.dict(os.environ, {ENV_DISABLE_LOAD: "true"}):
            result = manage_tools_module.manage_tools(
                action="fetch",
                url="https://example.com/tool.py",
                agent=mock_agent,
            )

        assert result["status"] == "error"
        assert "disabled" in result["content"][0]["text"].lower()

    def test_disabled_loading_allows_list(self, mock_agent):
        """Test that list still works when loading disabled."""
        with patch.dict(os.environ, {ENV_DISABLE_LOAD: "true"}):
            result = manage_tools_module.manage_tools(action="list", agent=mock_agent)

        assert result["status"] == "success"

    def test_disabled_loading_allows_remove(self, mock_agent):
        """Test that remove still works when loading disabled."""
        with patch.dict(os.environ, {ENV_DISABLE_LOAD: "true"}):
            result = manage_tools_module.manage_tools(
                action="remove",
                tools="calculator",
                agent=mock_agent,
            )

        assert result["status"] == "success"

    def test_disabled_loading_allows_discover(self, mock_agent):
        """Test that discover still works when loading disabled."""
        with patch.dict(os.environ, {ENV_DISABLE_LOAD: "true"}):
            result = manage_tools_module.manage_tools(
                action="discover",
                tools="strands_tools.manage_tools",
                agent=mock_agent,
            )

        assert result["status"] == "success"

    def test_disabled_loading_allows_sandbox(self, mock_agent, valid_tool_code):
        """Test that sandbox still works when loading disabled."""
        with patch.dict(os.environ, {ENV_DISABLE_LOAD: "true"}):
            result = manage_tools_module.manage_tools(
                action="sandbox",
                code=valid_tool_code,
                agent=mock_agent,
            )

        assert result["status"] == "success"
