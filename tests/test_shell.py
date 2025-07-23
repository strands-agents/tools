"""
Tests for the shell tool including execution, formatting, and various modes.
"""

import os
import signal
import sys
from unittest.mock import MagicMock, patch

import pytest
from rich.panel import Panel
from strands import Agent

if os.name == "nt":
    pytest.skip("skipping on windows until issue #17 is resolved", allow_module_level=True)

import termios

from strands_tools import shell


@pytest.fixture
def agent():
    """Create an agent with the shell tool loaded."""
    return Agent(tools=[shell])


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


# Basic functionality tests
@patch("strands_tools.shell.execute_commands")
@patch("strands_tools.shell.get_user_input")
def test_shell_tool_direct(mock_get_user_input, mock_execute_commands):
    """Test direct invocation of the shell tool."""
    # Mock the user input to 'y' (yes)
    mock_get_user_input.return_value = "y"

    # Mock the execute_commands to return a successful result
    mock_execute_commands.return_value = [
        {
            "command": "echo test",
            "exit_code": 0,
            "output": "test\n",
            "error": "",
            "status": "success",
        }
    ]

    # Call the shell function directly with new @tool decorator signature
    result = shell.shell("echo test")

    # Verify the result has the expected structure
    assert result["status"] == "success"
    assert "Total commands: 1" in result["content"][0]["text"]
    assert "Successful: 1" in result["content"][0]["text"]

    # Check that execute_commands was called with the correct arguments
    mock_execute_commands.assert_called_once()
    args, kwargs = mock_execute_commands.call_args
    assert args[0] == ["echo test"]  # commands
    assert args[1] is False  # parallel
    assert args[2] is False  # ignore_errors


@patch("os.environ")
@patch("strands_tools.shell.execute_commands")
@patch("strands_tools.shell.get_user_input")
def test_shell_non_interactive_mode(mock_get_user_input, mock_execute_commands, mock_environ):
    """Test shell tool in non-interactive mode."""

    # Mock execute_commands to return a successful result
    def mock_env_get(key, default=""):
        if key == "STRANDS_NON_INTERACTIVE":
            return "true"
        if key == "BYPASS_TOOL_CONSENT":
            return "false"
        return default

    mock_environ.get.side_effect = mock_env_get

    # Mock execute_commands to return a successful result
    mock_execute_commands.return_value = [
        {
            "command": "ls",
            "exit_code": 0,
            "output": "file1\nfile2\n",
            "error": "",
            "status": "success",
        }
    ]

    # Call the shell function with new @tool decorator signature
    result = shell.shell("ls")

    assert result["status"] == "success"

    # Verify that get_user_input was not called because the env var forces non-interactive mode
    mock_get_user_input.assert_not_called()


@patch("strands_tools.shell.execute_commands")
@patch("strands_tools.shell.get_user_input")
def test_shell_cancel_execution(mock_get_user_input, mock_execute_commands):
    """Test cancellation of shell execution."""
    # Mock the user input to 'n' (no)
    mock_get_user_input.return_value = "n"

    # Ensure BYPASS_TOOL_CONSENT mode is disabled to force confirmation
    current_dev = os.environ.get("BYPASS_TOOL_CONSENT", None)
    if current_dev:
        os.environ.pop("BYPASS_TOOL_CONSENT")

    # Call the shell function with new @tool decorator signature
    result = shell.shell("dangerous_command")

    # Restore BYPASS_TOOL_CONSENT mode if it was set
    if current_dev:
        os.environ["BYPASS_TOOL_CONSENT"] = current_dev

    # Verify the result shows cancellation
    assert result["status"] == "error"
    assert "cancelled" in result["content"][0]["text"].lower()

    # Verify execute_commands was not called
    mock_execute_commands.assert_not_called()


@patch("strands_tools.shell.execute_commands")
@patch("strands_tools.shell.get_user_input")
def test_shell_multiple_commands(mock_get_user_input, mock_execute_commands):
    """Test shell tool with multiple commands."""
    # Mock the user input and commands execution
    mock_get_user_input.return_value = "y"
    mock_execute_commands.return_value = [
        {
            "command": "echo hello",
            "exit_code": 0,
            "output": "hello\n",
            "error": "",
            "status": "success",
        },
        {
            "command": "echo world",
            "exit_code": 0,
            "output": "world\n",
            "error": "",
            "status": "success",
        },
    ]

    # Call the shell function with new @tool decorator signature
    result = shell.shell(["echo hello", "echo world"])

    # Verify the result
    assert result["status"] == "success"
    assert "Total commands: 2" in result["content"][0]["text"]
    assert "Successful: 2" in result["content"][0]["text"]
    assert len(result["content"]) == 3  # Summary + 2 command results


@patch("strands_tools.shell.execute_commands")
@patch("strands_tools.shell.get_user_input")
def test_shell_command_failure(mock_get_user_input, mock_execute_commands):
    """Test shell tool with a failing command."""
    # Mock the user input and commands execution
    mock_get_user_input.return_value = "y"
    mock_execute_commands.return_value = [
        {
            "command": "invalid_command",
            "exit_code": 127,
            "output": "",
            "error": "command not found",
            "status": "error",
        }
    ]

    # Call the shell function with new @tool decorator signature
    result = shell.shell("invalid_command")

    # Verify the result
    assert result["status"] == "error"
    assert "Failed: 1" in result["content"][0]["text"]


@patch("strands_tools.shell.execute_commands")
@patch("strands_tools.shell.get_user_input")
def test_shell_ignore_errors(mock_get_user_input, mock_execute_commands):
    """Test shell tool with ignore_errors flag."""
    # Mock the user input and commands execution
    mock_get_user_input.return_value = "y"
    mock_execute_commands.return_value = [
        {
            "command": "invalid_command",
            "exit_code": 127,
            "output": "",
            "error": "command not found",
            "status": "error",
        }
    ]

    # Call the shell function with new @tool decorator signature and ignore_errors=True
    result = shell.shell("invalid_command", ignore_errors=True)

    # Verify the result - status should be success due to ignore_errors
    assert result["status"] == "success"
    assert "Failed: 1" in result["content"][0]["text"]


@patch("strands_tools.shell.execute_commands")
@patch("strands_tools.shell.get_user_input")
def test_shell_exception_handling(mock_get_user_input, mock_execute_commands):
    """Test shell tool exception handling."""
    # Mock the user input
    mock_get_user_input.return_value = "y"

    # Make execute_commands raise an exception
    mock_execute_commands.side_effect = Exception("Test exception")

    # Call the shell function with new @tool decorator signature
    result = shell.shell("echo test")

    # Verify the result
    assert result["status"] == "error"
    assert "Interactive shell error:" in result["content"][0]["text"]
    assert "Test exception" in result["content"][0]["text"]


# Utility function tests
def test_normalize_commands():
    """Test the normalize_commands function."""
    # Test with a string
    assert shell.normalize_commands("echo test") == ["echo test"]

    # Test with a list of strings
    assert shell.normalize_commands(["ls", "pwd"]) == ["ls", "pwd"]

    # Test with a dictionary
    cmd_dict = {"command": "git status", "timeout": 30}
    assert shell.normalize_commands(cmd_dict) == [cmd_dict]


def test_validate_command():
    """Test the validate_command function."""
    # Test with a string command
    cmd, opts = shell.validate_command("echo test")
    assert cmd == "echo test"
    assert opts == {}

    # Test with a dictionary command
    cmd_dict = {"command": "git status", "timeout": 30}
    cmd, opts = shell.validate_command(cmd_dict)
    assert cmd == "git status"
    assert opts == cmd_dict

    # Test with an invalid dictionary (no command key)
    with pytest.raises(ValueError):
        shell.validate_command({"timeout": 30})

    # Test with an invalid type
    with pytest.raises(ValueError):
        shell.validate_command(123)


def test_command_context():
    """Test the CommandContext class."""
    # Create a context with the current directory
    context = shell.CommandContext(os.getcwd())

    # Save the current directory
    original_dir = context.current_dir
    context.push_dir()

    # Change to a new directory
    context.update_dir("cd /tmp")
    assert context.current_dir == "/tmp"

    # Restore the previous directory
    context.pop_dir()
    assert context.current_dir == original_dir

    # Test with a relative path
    context.update_dir("cd ..")
    assert context.current_dir == os.path.abspath(os.path.join(original_dir, ".."))

    # Test with a non-cd command (should not change directory)
    current_dir = context.current_dir
    context.update_dir("ls -la")
    assert context.current_dir == current_dir

    # Test with an empty directory stack
    context = shell.CommandContext(os.getcwd())
    context.pop_dir()  # Should not raise an error


# Tests from test_shell_additional.py
def test_read_output():
    """Test the read_output function with different encodings."""
    with patch("os.read") as mock_read:
        # Test UTF-8 decoding
        mock_read.return_value = b"test output"
        assert shell.read_output(5) == "test output"

        # Test Latin-1 fallback
        mock_read.return_value = b"\xff\xfe test with invalid utf-8"
        assert shell.read_output(5) == "ÿþ test with invalid utf-8"

        # Test OSError handling
        mock_read.side_effect = OSError("Test error")
        assert shell.read_output(5) == ""


@patch("os.execvp")
@patch("os.chdir")
@patch("pty.fork")
@patch("termios.tcgetattr")
@patch("tty.setraw")
@patch("select.select")
@patch("os.read")
@patch("os.write")
@patch("os.waitpid")
@patch("termios.tcsetattr")
def test_command_executor_execute_with_pty(
    mock_tcsetattr,
    mock_waitpid,
    mock_write,
    mock_read,
    mock_select,
    mock_setraw,
    mock_tcgetattr,
    mock_fork,
    mock_chdir,
    mock_execvp,
):
    """Test the CommandExecutor execute_with_pty method."""
    # Mock setup
    mock_tcgetattr.return_value = "old_tty_settings"
    mock_fork.return_value = (123, 5)  # pid, fd
    mock_select.return_value = ([5], [], [])  # fd is ready for reading
    mock_read.side_effect = [b"test output", b""]  # Return output then EOF
    mock_waitpid.return_value = (123, 0)  # pid, status 0 (success)

    # Mock sys.stdin to handle fileno() calls in pytest environment
    with patch("sys.stdin") as mock_stdin:
        mock_stdin.fileno.return_value = 0  # Fake file descriptor for stdin

        # Create the executor
        executor = shell.CommandExecutor(timeout=10)

        # Execute a command
        exit_code, output, error = executor.execute_with_pty("echo test", "/tmp", non_interactive_mode=False)

        # Verify the results
        assert exit_code == 0
        assert output == "test output"
        assert error == ""

        # Verify the method calls
        mock_fork.assert_called_once()
        mock_chdir.assert_not_called()  # Not called in parent process

        # Allow tcgetattr to be called multiple times (but at least once)
        assert mock_tcgetattr.call_count >= 1
        # Check at least one call was with mock_stdin
        mock_tcgetattr.assert_any_call(mock_stdin)

        # In pytest environment, setraw may fail due to stdin redirection
        # Only verify it was attempted if stdin has fileno
        if hasattr(sys.stdin, "fileno"):
            mock_setraw.assert_called_once_with(mock_stdin.fileno())

        mock_select.assert_called()
        mock_read.assert_called()
        mock_waitpid.assert_called_once_with(123, 0)
        mock_tcsetattr.assert_called_once_with(mock_stdin, termios.TCSAFLUSH, "old_tty_settings")


@patch("os.execvp")
@patch("os.chdir")
@patch("os.getpgid")
@patch("pty.fork")
@patch("termios.tcgetattr")
@patch("tty.setraw")
@patch("select.select")
@patch("os.read")
@patch("os.waitpid")
@patch("termios.tcsetattr")
@patch("time.time")
def test_command_executor_execute_with_pty_timeout(
    mock_time,
    mock_tcsetattr,
    mock_waitpid,
    mock_read,
    mock_select,
    mock_setraw,
    mock_tcgetattr,
    mock_fork,
    mock_getpgid,
    mock_chdir,
    mock_execvp,
):
    """Test the CommandExecutor execute_with_pty method with timeout."""
    # Mock setup
    mock_tcgetattr.return_value = "old_tty_settings"
    mock_fork.return_value = (123, 5)  # pid, fd
    mock_getpgid.return_value = 123

    # Set up time mock to simulate passing the timeout
    mock_time.side_effect = [
        10,
        20,
        30,
    ]  # Start time, check time (which exceeds timeout)

    # Mock sys.stdin
    with patch("sys.stdin") as mock_stdin, patch("os.killpg") as mock_killpg:
        mock_stdin.fileno.return_value = 0

        # Create the executor with a very short timeout
        executor = shell.CommandExecutor(timeout=5)

        # Execute command - should time out
        with pytest.raises(TimeoutError):
            executor.execute_with_pty("sleep 10", "/tmp", non_interactive_mode=False)

        # Verify kill was called with SIGTERM
        mock_killpg.assert_called_once_with(123, signal.SIGTERM)


@patch("os.execvp")
@patch("os.chdir")
@patch("pty.fork")
@patch("termios.tcgetattr")
@patch("termios.tcsetattr")
@patch("tty.setraw")
def test_command_executor_execute_with_pty_tcsetattr_exception(
    mock_setraw, mock_tcsetattr, mock_tcgetattr, mock_fork, mock_chdir, mock_execvp
):
    """Test the CommandExecutor execute_with_pty method with tcsetattr exception."""
    # Mock setup
    mock_tcgetattr.return_value = "old_tty_settings"
    mock_fork.return_value = (123, 5)  # pid, fd

    # Simulate tcsetattr() error
    mock_tcsetattr.side_effect = Exception("Test tcsetattr error")

    # Mock additional functions
    with (
        patch("sys.stdin") as mock_stdin,
        patch("select.select") as mock_select,
        patch("os.read") as mock_read,
        patch("os.waitpid") as mock_waitpid,
    ):
        mock_stdin.fileno.return_value = 0
        mock_select.return_value = ([5], [], [])
        mock_read.side_effect = [b"test output", b""]
        mock_waitpid.return_value = (123, 0)

        # Create the executor
        executor = shell.CommandExecutor(timeout=10)

        # Execute command - should catch tcsetattr exception and call stty sane
        with pytest.raises(Exception, match="Test tcsetattr error"):
            executor.execute_with_pty("echo test", "/tmp", non_interactive_mode=False)

        # Verify tty.setraw was attempted
        mock_setraw.assert_called_once()
        # Verify tcsetattr was called to restore (which then raised the error)
        mock_tcsetattr.assert_called_once()


@patch("strands_tools.shell.execute_single_command")
def test_execute_commands_parallel(mock_execute_single_command):
    """Test execute_commands with parallel execution."""
    # Setup mock
    mock_execute_single_command.side_effect = [
        {
            "command": "cmd1",
            "exit_code": 0,
            "output": "output1",
            "error": "",
            "status": "success",
        },
        {
            "command": "cmd2",
            "exit_code": 0,
            "output": "output2",
            "error": "",
            "status": "success",
        },
    ]

    # Execute commands in parallel
    commands = ["cmd1", "cmd2"]
    results = shell.execute_commands(
        commands=commands, parallel=True, ignore_errors=False, work_dir="/tmp", timeout=10, non_interactive_mode=False
    )

    # Verify results
    assert len(results) == 2
    assert results[0]["command"] == "cmd1"
    assert results[1]["command"] == "cmd2"

    # Verify parallel execution called execute_single_command with the same work_dir
    assert mock_execute_single_command.call_count == 2

    # Check parameter locations in kwargs instead of positional args
    for call_args in mock_execute_single_command.call_args_list:
        # Get the kwargs dictionary or positional args based on the function signature
        if len(call_args[0]) >= 3:  # If called with positional args
            assert call_args[0][1] == "/tmp"  # work_dir parameter is the second arg
        else:  # If called with keyword args
            kwargs = call_args[1] if len(call_args) > 1 else {}
            assert kwargs.get("work_dir") == "/tmp"


@patch("strands_tools.shell.execute_single_command")
def test_execute_commands_parallel_with_error(mock_execute_single_command):
    """Test execute_commands with parallel execution and error handling."""
    # Setup mock to simulate one command failing
    mock_execute_single_command.side_effect = [
        {
            "command": "cmd1",
            "exit_code": 1,
            "output": "",
            "error": "error",
            "status": "error",
        },
        {
            "command": "cmd2",
            "exit_code": 0,
            "output": "output",
            "error": "",
            "status": "success",
        },
    ]

    # Execute commands in parallel with ignore_errors=False
    commands = ["cmd1", "cmd2"]
    results = shell.execute_commands(
        commands=commands, parallel=True, ignore_errors=False, work_dir="/tmp", timeout=10, non_interactive_mode=False
    )

    # Verify only the first result is returned (second should be canceled)
    assert len(results) == 1
    assert results[0]["status"] == "error"


@patch("strands_tools.shell.execute_single_command")
def test_execute_commands_parallel_with_ignore_errors(mock_execute_single_command):
    """Test execute_commands with parallel execution and ignored errors."""
    # Setup mock to simulate one command failing
    mock_execute_single_command.side_effect = [
        {
            "command": "cmd1",
            "exit_code": 1,
            "output": "",
            "error": "error",
            "status": "error",
        },
        {
            "command": "cmd2",
            "exit_code": 0,
            "output": "output",
            "error": "",
            "status": "success",
        },
    ]

    # Execute commands in parallel with ignore_errors=True
    commands = ["cmd1", "cmd2"]
    results = shell.execute_commands(
        commands=commands, parallel=True, ignore_errors=True, work_dir="/tmp", timeout=10, non_interactive_mode=False
    )

    # Verify both results are returned despite the first one failing
    assert len(results) == 2
    assert results[0]["status"] == "error"
    assert results[1]["status"] == "success"


@patch("strands_tools.shell.execute_single_command")
def test_execute_commands_sequential_with_cd(mock_execute_single_command):
    """Test execute_commands with sequential execution and directory changes."""
    # Setup mock
    mock_execute_single_command.side_effect = [
        {
            "command": "cd /new/dir",
            "exit_code": 0,
            "output": "",
            "error": "",
            "status": "success",
        },
        {
            "command": "pwd",
            "exit_code": 0,
            "output": "/new/dir",
            "error": "",
            "status": "success",
        },
    ]

    # Execute commands sequentially
    commands = ["cd /new/dir", "pwd"]
    results = shell.execute_commands(
        commands=commands, parallel=False, ignore_errors=False, work_dir="/tmp", timeout=10, non_interactive_mode=False
    )

    # Verify results
    assert len(results) == 2
    assert results[0]["command"] == "cd /new/dir"
    assert results[1]["command"] == "pwd"

    # Verify the second command was executed in the new directory
    # The first call should use original work_dir, second call should use updated dir
    assert mock_execute_single_command.call_args_list[0][0][1] == "/tmp"
    assert mock_execute_single_command.call_args_list[1][0][1] == "/new/dir"


@patch("strands_tools.shell.execute_single_command")
def test_execute_commands_sequential_with_error(mock_execute_single_command):
    """Test execute_commands with sequential execution and error handling."""
    # Setup mock
    mock_execute_single_command.side_effect = [
        {
            "command": "cmd1",
            "exit_code": 0,
            "output": "output1",
            "error": "",
            "status": "success",
        },
        {
            "command": "cmd2",
            "exit_code": 1,
            "output": "",
            "error": "error",
            "status": "error",
        },
        {
            "command": "cmd3",
            "exit_code": 0,
            "output": "output3",
            "error": "",
            "status": "success",
        },
    ]

    # Execute commands sequentially with ignore_errors=False
    commands = ["cmd1", "cmd2", "cmd3"]
    results = shell.execute_commands(
        commands=commands, parallel=False, ignore_errors=False, work_dir="/tmp", timeout=10, non_interactive_mode=False
    )

    # Verify only the first two commands were executed (stopped after error)
    assert len(results) == 2
    assert results[0]["status"] == "success"
    assert results[1]["status"] == "error"
    assert mock_execute_single_command.call_count == 2


# Rich formatting tests
def test_format_command_preview():
    """Test the format_command_preview function."""
    # Test with string command
    panel = shell.format_command_preview(command="echo test", parallel=False, ignore_errors=True, work_dir="/tmp")

    # Verify the result is a Panel object
    assert isinstance(panel, Panel)

    # Test with dictionary command
    panel = shell.format_command_preview(
        command={"command": "git pull", "timeout": 60},
        parallel=True,
        ignore_errors=False,
        work_dir="/repo",
    )

    assert isinstance(panel, Panel)


def test_format_summary():
    """Test the format_summary function."""
    # Test all success
    results = [
        {"command": "cmd1", "status": "success", "exit_code": 0},
        {"command": "cmd2", "status": "success", "exit_code": 0},
    ]

    panel = shell.format_summary(results, parallel=False)
    assert isinstance(panel, Panel)
    assert panel.border_style == "green"

    # Test mixed results
    results = [
        {"command": "cmd1", "status": "success", "exit_code": 0},
        {"command": "cmd2", "status": "error", "exit_code": 1},
    ]

    panel = shell.format_summary(results, parallel=True)
    assert isinstance(panel, Panel)
    assert panel.border_style == "yellow"  # Warning for mixed success/failure

    # Test all failures
    results = [
        {"command": "cmd1", "status": "error", "exit_code": 1},
        {"command": "cmd2", "status": "error", "exit_code": 1},
    ]

    panel = shell.format_summary(results, parallel=False)
    assert isinstance(panel, Panel)
    assert panel.border_style == "red"


# Additional configuration tests
@patch("strands_tools.shell.execute_commands")
@patch("strands_tools.shell.get_user_input")
def test_shell_with_timeout(mock_get_user_input, mock_execute_commands):
    """Test shell tool with custom timeout."""
    # Mock setup
    mock_get_user_input.return_value = "y"
    mock_execute_commands.return_value = [
        {
            "command": "sleep 1",
            "exit_code": 0,
            "output": "",
            "error": "",
            "status": "success",
        }
    ]

    # Call the shell function with new @tool decorator signature and timeout=60
    shell.shell("sleep 1", timeout=60)

    # Verify execute_commands was called with the custom timeout
    mock_execute_commands.assert_called_once()
    args, _kwargs = mock_execute_commands.call_args
    assert args[4] == 60  # timeout parameter


@patch("strands_tools.shell.execute_commands")
@patch("strands_tools.shell.get_user_input")
def test_shell_with_work_dir(mock_get_user_input, mock_execute_commands):
    """Test shell tool with custom working directory."""
    # Mock setup
    mock_get_user_input.return_value = "y"
    mock_execute_commands.return_value = [
        {
            "command": "pwd",
            "exit_code": 0,
            "output": "/custom/dir\n",
            "error": "",
            "status": "success",
        }
    ]

    # Call the shell function with new @tool decorator signature and work_dir="/custom/dir"
    shell.shell("pwd", work_dir="/custom/dir")

    # Verify execute_commands was called with the custom work_dir
    mock_execute_commands.assert_called_once()
    args, kwargs = mock_execute_commands.call_args
    assert args[3] == "/custom/dir"  # work_dir parameter


@patch("os.environ")
@patch("strands_tools.shell.execute_commands")
@patch("strands_tools.shell.get_user_input")
def test_shell_dev_mode(mock_get_user_input, mock_execute_commands, mock_environ):
    """Test shell tool in BYPASS_TOOL_CONSENT mode (skips confirmation)."""

    # Configure mock to handle different environment variables appropriately
    def mock_env_get(key, default=""):
        if key == "BYPASS_TOOL_CONSENT":
            return "true"
        if key == "SHELL_DEFAULT_TIMEOUT":
            return "900"
        return default

    mock_environ.get.side_effect = mock_env_get

    # Mock setup
    mock_execute_commands.return_value = [
        {
            "command": "echo test",
            "exit_code": 0,
            "output": "test\n",
            "error": "",
            "status": "success",
        }
    ]

    # Call the shell function with new @tool decorator signature
    result = shell.shell("echo test")

    # Verify the result
    assert result["status"] == "success"

    # Verify get_user_input was not called (no confirmation in BYPASS_TOOL_CONSENT mode)
    mock_get_user_input.assert_not_called()


@patch("strands_tools.shell.execute_commands")
def test_shell_with_json_array_string(mock_execute_commands):
    """Test shell tool with a JSON array string as the command."""
    # Setup mock
    mock_execute_commands.return_value = [
        {
            "command": "cmd1",
            "exit_code": 0,
            "output": "output1",
            "error": "",
            "status": "success",
        },
        {
            "command": "cmd2",
            "exit_code": 0,
            "output": "output2",
            "error": "",
            "status": "success",
        },
    ]
    # Call the shell function with new @tool decorator signature and non_interactive=True to skip confirmation
    shell.shell('["cmd1", "cmd2"]', non_interactive=True)

    # Verify execute_commands was called with parsed array
    mock_execute_commands.assert_called_once()
    args, _kwargs = mock_execute_commands.call_args
    assert args[0] == ["cmd1", "cmd2"]  # Parsed as array


@patch("strands_tools.shell.execute_commands")
def test_shell_with_invalid_json_array_string(mock_execute_commands):
    """Test shell tool with an invalid JSON array string."""
    # Setup mock
    mock_execute_commands.return_value = [
        {
            "command": "[invalid json array]",
            "exit_code": 0,
            "output": "output",
            "error": "",
            "status": "success",
        }
    ]

    # Call the shell function with new @tool decorator signature and non_interactive=True
    shell.shell("[invalid json array]", non_interactive=True)

    # Verify execute_commands was called with the original string (fallback)
    mock_execute_commands.assert_called_once()
    args, _kwargs = mock_execute_commands.call_args
    assert args[0] == ["[invalid json array]"]  # Kept as string


@patch("strands_tools.shell.execute_commands")
def test_shell_with_complex_command_objects(mock_execute_commands):
    """Test shell tool with complex command objects."""
    # Setup mock
    mock_execute_commands.return_value = [
        {
            "command": "git clone",
            "exit_code": 0,
            "output": "Cloning...",
            "error": "",
            "status": "success",
            "options": {"command": "git clone", "timeout": 120},
        }
    ]

    # Call the shell function with new @tool decorator signature - complex command object
    command_obj = {"command": "git clone", "timeout": 120}
    shell.shell(command_obj, non_interactive=True)

    # Verify execute_commands was called with the complex object
    mock_execute_commands.assert_called_once()
    args, _kwargs = mock_execute_commands.call_args
    assert args[0] == [command_obj]


def test_execute_single_command():
    """Test execute_single_command function."""
    with patch.object(shell.CommandExecutor, "execute_with_pty") as mock_execute:
        # Mock successful execution
        mock_execute.return_value = (0, "command output", "")

        # Test with string command
        result = shell.execute_single_command("echo test", "/tmp", 10, non_interactive_mode=False)
        assert result["command"] == "echo test"
        assert result["exit_code"] == 0
        assert result["output"] == "command output"
        assert result["status"] == "success"

        # Test with dictionary command
        cmd_dict = {"command": "git status", "timeout": 30}
        result = shell.execute_single_command(cmd_dict, "/tmp", 10, non_interactive_mode=False)
        assert result["command"] == "git status"
        assert result["options"] == cmd_dict

        # Test with exception
        mock_execute.side_effect = Exception("Test error")
        result = shell.execute_single_command("failing command", "/tmp", 10, non_interactive_mode=False)
        assert result["command"] == "failing command"
        assert result["status"] == "error"
        assert "Test error" in result["error"]


@patch("strands_tools.shell.console_util")
def test_shell_console_output(mock_console_util):
    """Test console output in interactive mode."""
    mock_console = mock_console_util.create.return_value

    # Setup mocks
    with (
        patch("strands_tools.shell.execute_commands") as mock_execute_commands,
        patch("strands_tools.shell.get_user_input") as mock_get_user_input,
    ):
        mock_get_user_input.return_value = "y"
        mock_execute_commands.return_value = [
            {
                "command": "echo test",
                "exit_code": 0,
                "output": "test",
                "error": "",
                "status": "success",
            }
        ]

        # Call the shell function in interactive mode
        shell.shell("echo test")

        assert mock_console.print.call_count >= 3

        # Reset mock for non-interactive test
        mock_console.reset_mock()

        # Call with non_interactive_mode=True
        shell.shell("echo test", non_interactive=True)

        # Verify console.print was not called for UI elements in non-interactive mode
        assert mock_console.print.call_count == 0


@patch("strands_tools.shell.console_util")
@patch("strands_tools.shell.get_user_input")
@patch("strands_tools.shell.execute_commands")
def test_shell_exception_ui_output(mock_execute_commands, mock_get_user_input, mock_console_util):
    """Test console error output when exception occurs."""
    # Setup mocks
    mock_console = mock_console_util.create.return_value
    mock_get_user_input.return_value = "y"
    mock_execute_commands.side_effect = Exception("Test error")

    # Create a Panel mock for the console.print call
    error_panel = Panel("Test content")
    mock_panel = MagicMock(return_value=error_panel)

    with patch("rich.panel.Panel", mock_panel):
        # Call the shell function
        shell.shell("echo test")

        # Verify console.print was called at least once
        assert mock_console.print.call_count >= 1

        # Reset mock for non-interactive test
        mock_console.reset_mock()

        # Call with non_interactive_mode=True
        shell.shell("echo test", non_interactive=True)

        # Verify console.print was not called in non-interactive mode
        assert mock_console.print.call_count == 0


@patch("strands_tools.shell.execute_commands")
@patch("strands_tools.shell.get_user_input")
def test_shell_non_interactive_parameter(mock_get_user_input, mock_execute_commands):
    """Test shell tool with non_interactive parameter."""
    # Mock execute_commands to return a successful result
    mock_execute_commands.return_value = [
        {
            "command": "echo test",
            "exit_code": 0,
            "output": "test\n",
            "error": "",
            "status": "success",
        }
    ]

    # Call the shell function with non_interactive=True
    result = shell.shell("echo test", non_interactive=True)

    assert result["status"] == "success"

    # Verify that get_user_input was not called because non_interactive=True
    mock_get_user_input.assert_not_called()
