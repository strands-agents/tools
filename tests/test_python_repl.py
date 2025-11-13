"""Tests for the python_repl tool using the Agent interface."""

import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import dill
import pytest
from strands import Agent

if os.name == "nt":
    pytest.skip("skipping on windows until issue #17 is resolved", allow_module_level=True)

from strands_tools import python_repl


@pytest.fixture
def agent():
    """Create an agent with the python_repl tool loaded."""
    return Agent(tools=[python_repl])


@pytest.fixture
def mock_console():
    """Mock the rich Console to prevent terminal output during tests."""
    with patch("strands_tools.python_repl.console_util") as mock_console_util:
        yield mock_console_util.create.return_value


@pytest.fixture
def temp_repl_state_dir():
    """Create a temporary directory for REPL state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = python_repl.repl_state.persistence_dir
        python_repl.repl_state.persistence_dir = tmpdir
        python_repl.repl_state.state_file = os.path.join(tmpdir, "repl_state.pkl")
        yield tmpdir
        # Restore original directory
        python_repl.repl_state.persistence_dir = original_dir
        python_repl.repl_state.state_file = os.path.join(original_dir, "repl_state.pkl")


class TestOutputCapture:
    """Test the OutputCapture class."""

    def test_capture_stdout(self):
        """Test capturing standard output."""
        capture = python_repl.OutputCapture()
        with capture:
            print("Hello, world!")

        assert "Hello, world!" in capture.get_output()

    def test_capture_stderr(self):
        """Test capturing standard error."""
        capture = python_repl.OutputCapture()
        with capture:
            print("Error message", file=sys.stderr)

        assert "Error message" in capture.get_output()
        assert "Errors:" in capture.get_output()

    def test_capture_both(self):
        """Test capturing both stdout and stderr."""
        capture = python_repl.OutputCapture()
        with capture:
            print("Standard output")
            print("Standard error", file=sys.stderr)

        output = capture.get_output()
        assert "Standard output" in output
        assert "Standard error" in output


class TestReplState:
    """Test the ReplState class."""

    def test_execute_and_namespace(self, temp_repl_state_dir):
        """Test executing code and updating namespace."""
        repl = python_repl.ReplState()
        repl.execute("x = 42")

        namespace = repl.get_namespace()
        assert namespace["x"] == 42

    def test_valid_persistence_dir_from_env(self):
        """Test that a valid PYTHON_REPL_PERSISTENCE_DIR is accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(tmpdir, exist_ok=True)
            with patch.dict(os.environ, {"PYTHON_REPL_PERSISTENCE_DIR": tmpdir}):
                repl = python_repl.ReplState()
                expected_dir = os.path.join(Path(tmpdir).resolve(), "repl_state")
                assert repl.persistence_dir == expected_dir

    def test_nonexistent_persistence_dir_falls_back(self):
        """Test that a nonexistent directory changes to default."""
        fake_dir = "/badpath"
        with patch.dict(os.environ, {"PYTHON_REPL_PERSISTENCE_DIR": fake_dir}):
            repl = python_repl.ReplState()
            assert repl.persistence_dir != fake_dir
            assert "repl_state" in repl.persistence_dir

    def test_file_instead_of_dir_defaults(self):
        """Test that a file path instead of directory changes to default."""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            try:
                with patch.dict(os.environ, {"PYTHON_REPL_PERSISTENCE_DIR": tmpfile.name}):
                    repl = python_repl.ReplState()
                    assert repl.persistence_dir != tmpfile.name
                    assert "repl_state" in repl.persistence_dir
            finally:
                os.unlink(tmpfile.name)

    def test_unwritable_persistence_dir_falls_back(self):
        """Test that an unwritable directory changes to default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock os.access to simulate unwritable directory
            with (
                patch.dict(os.environ, {"PYTHON_REPL_PERSISTENCE_DIR": tmpdir}),
                patch("os.access", return_value=False),
            ):
                repl = python_repl.ReplState()
                # Should fall back to default
                assert repl.persistence_dir != tmpdir
                assert "repl_state" in repl.persistence_dir

    def test_no_env_var_uses_default(self):
        """Test that missing env var uses default directory."""
        with patch.dict(os.environ, {"PYTHON_REPL_PERSISTENCE_DIR": ""}):
            repl = python_repl.ReplState()
            assert "repl_state" in repl.persistence_dir
            assert os.path.exists(repl.persistence_dir)

    def test_clear_state(self, temp_repl_state_dir):
        """Test clearing state."""
        repl = python_repl.ReplState()
        repl.execute("z = 100")
        assert repl.get_namespace()["z"] == 100

        repl.clear_state()
        assert "z" not in repl.get_namespace()

    def test_save_state(self, temp_repl_state_dir):
        """Test saving state to file."""
        # Create a clean ReplState for this test
        repl = python_repl.ReplState()

        # Ensure we're starting with a clean state
        repl.clear_state()

        # Test in isolation - mock save_state to avoid namespace conflicts
        with patch.object(repl, "save_state") as mock_save:
            # Execute our code
            repl.execute("test_save = 'saved value'")

            # Verify save_state was called during execute
            mock_save.assert_called()

        # Verify the state file exists
        assert os.path.exists(repl.state_file)

    def test_save_state_and_load(self, temp_repl_state_dir):
        """Test saving and loading state from a file."""
        # Create a new state file with our own content
        test_state = {"test_var": "test value"}
        state_file_path = os.path.join(temp_repl_state_dir, "repl_state.pkl")
        with open(state_file_path, "wb") as f:
            dill.dump(test_state, f)

        # Force loading of our state file
        repl = python_repl.ReplState()
        # Explicitly load the state to ensure it picks up our file
        with open(state_file_path, "rb") as f:
            saved_state = dill.load(f)
        repl._namespace.update(saved_state)

        # Verify our variable is in the namespace
        assert "test_var" in repl.get_namespace()
        assert repl.get_namespace()["test_var"] == "test value"

    def test_save_state_with_code(self, temp_repl_state_dir):
        """Test saving state with code execution."""
        repl = python_repl.ReplState()

        # Make sure we have a clean state
        repl.clear_state()

        # Run and save code
        repl.save_state("test_var = 'directly saved'")

        # Verify the code was executed
        assert "test_var" in repl.get_namespace()
        assert repl.get_namespace()["test_var"] == "directly saved"

    def test_save_state_with_unpicklable_objects(self, temp_repl_state_dir):
        """Test saving state with objects that can't be pickled."""
        # Create a clean state
        repl = python_repl.ReplState()
        repl.clear_state()

        # Patch the dill.dumps function to simulate pickling failures for specific objects
        with patch("dill.dumps") as mock_dumps:
            # Configure mock to raise for unpicklable but work for regular values
            def side_effect(obj):
                if isinstance(obj, dict) and "unpicklable" in obj:
                    # The whole dict can be pickled, but we'll test the filtering logic
                    return bytes("mocked pickle", "utf-8")
                elif obj == "unpicklable_value":
                    raise TypeError("Cannot pickle 'unpicklable_value'")
                return bytes("mocked pickle", "utf-8")

            mock_dumps.side_effect = side_effect

            # Add objects to namespace
            repl._namespace["regular"] = "this should be saved"
            repl._namespace["unpicklable"] = "unpicklable_value"

            # Save state
            repl.save_state()

        # Verify that save_dict would only contain pickable objects
        mock_calls = mock_dumps.call_args_list
        found_unpicklable_rejection = False

        # dill.dumps will be called for both individual values and the final save_dict
        for call in mock_calls:
            args = call[0]
            if args[0] == "unpicklable_value":
                # This should attempt to pickle but raise an exception
                found_unpicklable_rejection = True

        assert found_unpicklable_rejection, "The code should attempt to pickle 'unpicklable_value' but fail"

    def test_error_during_state_removal(self, temp_repl_state_dir):
        """Test handling error when removing corrupted state file."""
        # Create a corrupted state file
        with open(os.path.join(temp_repl_state_dir, "repl_state.pkl"), "wb") as f:
            f.write(b"This is not valid pickle data")

        # Mock os.remove to raise an exception
        with patch("os.remove", side_effect=PermissionError("Permission denied")):
            # Should handle the error gracefully
            repl = python_repl.ReplState()
            # Still have a valid state even though removal failed
            assert "__name__" in repl.get_namespace()


class TestPythonRepl:
    """Test the main python_repl function."""

    def test_successful_execution(self, mock_console):
        """Test successful code execution."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "result = 2 + 2", "interactive": False},
        }

        # Mock user_input to simulate user confirmation
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result = python_repl.python_repl(tool=tool_use)

        assert result["toolUseId"] == "test-id"
        assert result["status"] == "success"
        assert "Code executed successfully" in result["content"][0]["text"]

        # Verify the code was actually executed
        assert python_repl.repl_state.get_namespace()["result"] == 4

    def test_syntax_error(self, mock_console):
        """Test handling of syntax errors."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "this is not valid python code", "interactive": False},
        }

        # Mock user_input to simulate user confirmation
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result = python_repl.python_repl(tool=tool_use)

        assert result["toolUseId"] == "test-id"
        assert result["status"] == "error"
        assert "SyntaxError" in result["content"][0]["text"]

    def test_runtime_error(self, mock_console):
        """Test handling of runtime errors."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {
                "code": "1/0",  # ZeroDivisionError
                "interactive": False,
            },
        }

        # Mock user_input to simulate user confirmation
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result = python_repl.python_repl(tool=tool_use)

        assert result["toolUseId"] == "test-id"
        assert result["status"] == "error"
        assert "ZeroDivisionError" in result["content"][0]["text"]

    def test_reset_state(self, temp_repl_state_dir, mock_console):
        """Test resetting the REPL state."""
        # First set a variable
        setup_tool = {
            "toolUseId": "setup-id",
            "input": {"code": "test_var = 'should be cleared'", "interactive": False},
        }

        # Mock user_input to simulate user confirmation
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            python_repl.python_repl(tool=setup_tool)

        # Now reset the state
        reset_tool = {
            "toolUseId": "reset-id",
            "input": {
                "code": "print('After reset')",
                "interactive": False,
                "reset_state": True,
            },
        }

        # Mock user_input to simulate user confirmation for reset operation
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            python_repl.python_repl(tool=reset_tool)

        # Verify the variable was cleared
        assert "test_var" not in python_repl.repl_state.get_namespace()

    def test_dev_mode_bypass_confirmation(self, mock_console):
        """Test that BYPASS_TOOL_CONSENT mode bypasses the confirmation dialog."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "dev_mode_test = 42", "interactive": False},
        }

        # Set BYPASS_TOOL_CONSENT environment variable to true
        with patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"}):
            # We shouldn't need to mock get_user_input here since BYPASS_TOOL_CONSENT mode should bypass it
            result = python_repl.python_repl(tool=tool_use)

            assert result["status"] == "success"
            assert python_repl.repl_state.get_namespace()["dev_mode_test"] == 42

    def test_non_interactive_mode_bypass_confirmation(self, mock_console):
        """Test that non_interactive_mode bypasses the confirmation dialog."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "non_interactive_test = 'passed'", "interactive": False},
        }

        # Pass non_interactive_mode as True
        result = python_repl.python_repl(tool=tool_use, non_interactive_mode=True)

        assert result["status"] == "success"
        assert python_repl.repl_state.get_namespace()["non_interactive_test"] == "passed"

    def test_user_rejection_cancels_execution(self, mock_console):
        """Test that user rejection properly cancels execution."""
        # Clear REPL state to ensure clean test environment
        python_repl.repl_state.clear_state()

        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "should_not_execute = True", "interactive": False},
        }

        # Mock user rejecting the execution and ensure bypass conditions are disabled
        with (
            patch("strands_tools.python_repl.get_user_input", side_effect=["n", "Testing rejection"]),
            patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "false"}, clear=False),
        ):
            result = python_repl.python_repl(tool=tool_use)

            assert result["status"] == "error"
            assert "cancelled by the user" in result["content"][0]["text"]
            assert "should_not_execute" not in python_repl.repl_state.get_namespace()

    def test_custom_rejection_message(self, mock_console):
        """Test that custom rejection message is included."""
        # Clear REPL state to ensure clean test environment
        python_repl.repl_state.clear_state()

        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "print('Should not run')", "interactive": False},
        }

        # Mock user providing custom rejection reason and ensure bypass conditions are disabled
        with (
            patch("strands_tools.python_repl.get_user_input", side_effect=["custom reason", ""]),
            patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "false"}, clear=False),
        ):
            result = python_repl.python_repl(tool=tool_use)

            assert result["status"] == "error"
            assert "Reason: custom reason" in result["content"][0]["text"]

    def test_recursion_error(self, mock_console, temp_repl_state_dir):
        """Test handling of recursion errors."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {
                "code": "def recurse(): recurse(); recurse()",  # Set up recursion
                "interactive": False,
            },
        }

        # First define the recursive function
        # Pass non_interactive_mode=True to bypass confirmation
        python_repl.python_repl(tool=tool_use, non_interactive_mode=True)

        # Now trigger the recursion error
        error_tool = {
            "toolUseId": "error-id",
            "input": {
                "code": "recurse()",  # This will cause a recursion error
                "interactive": False,
            },
        }

        # Mock the clear_state method to verify it gets called
        with patch.object(
            python_repl.repl_state,
            "clear_state",
            wraps=python_repl.repl_state.clear_state,
        ) as mock_clear:
            # Pass non_interactive_mode=True to bypass confirmation
            result = python_repl.python_repl(tool=error_tool, non_interactive_mode=True)

            # Verify clear_state was called
            mock_clear.assert_called_once()

            # Verify error message and suggestion
            assert result["status"] == "error"
            assert "RecursionError" in result["content"][0]["text"]
            assert "reset_state=True" in result["content"][0]["text"]

    def test_interactive_mode(self, mock_console):
        """Test interactive mode with PTY simulation."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {
                "code": "print('Interactive test')",
                "interactive": True,
            },
        }

        # Mock PtyManager to avoid actual PTY operations
        with patch("strands_tools.python_repl.PtyManager") as mock_pty:
            # Configure mocks
            mock_pty_instance = mock_pty.return_value
            mock_pty_instance.pid = 12345
            mock_pty_instance.get_output.return_value = "Interactive test\n"

            # Mock os.waitpid to simulate process completion
            with patch("os.waitpid") as mock_waitpid:
                mock_waitpid.side_effect = [(12345, 0)]  # Return pid and exit status 0

                # Pass non_interactive_mode=True to bypass confirmation
                result = python_repl.python_repl(tool=tool_use, non_interactive_mode=True)

                # Verify PtyManager was used
                mock_pty.assert_called_once()
                mock_pty_instance.start.assert_called_once_with("print('Interactive test')")
                mock_pty_instance.get_output.assert_called_once()
                mock_pty_instance.stop.assert_called_once()

                # Verify result
                assert result["status"] == "success"
                assert "Interactive test" in result["content"][0]["text"]

    def test_interactive_mode_error(self, mock_console):
        """Test interactive mode with process error."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {
                "code": "raise ValueError('Test error')",
                "interactive": True,
            },
        }

        # Mock PtyManager to avoid actual PTY operations
        with patch("strands_tools.python_repl.PtyManager") as mock_pty:
            # Configure mocks
            mock_pty_instance = mock_pty.return_value
            mock_pty_instance.pid = 12345
            mock_pty_instance.get_output.return_value = "Traceback... ValueError: Test error"

            # Mock os.waitpid to simulate process error
            with patch("os.waitpid") as mock_waitpid:
                mock_waitpid.side_effect = [(12345, 1)]  # Return pid and non-zero exit status

                # Pass non_interactive_mode=True to bypass confirmation
                result = python_repl.python_repl(tool=tool_use, non_interactive_mode=True)

                # Verify PtyManager was used and stopped
                mock_pty_instance.stop.assert_called_once()

                # State shouldn't be saved on error
                assert result["status"] == "success"  # Tool still succeeds as it captured output
                assert "Test error" in result["content"][0]["text"]

    def test_interactive_mode_os_error(self, mock_console):
        """Test interactive mode with OSError when waiting for process."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {
                "code": "print('test')",
                "interactive": True,
            },
        }

        # Mock PtyManager
        with patch("strands_tools.python_repl.PtyManager") as mock_pty:
            mock_pty_instance = mock_pty.return_value
            mock_pty_instance.pid = 12345
            mock_pty_instance.get_output.return_value = "test output"

            # Mock os.waitpid to raise OSError
            with patch("os.waitpid", side_effect=OSError("No such process")):
                # Pass non_interactive_mode=True to bypass confirmation
                result = python_repl.python_repl(tool=tool_use, non_interactive_mode=True)

                # Verify PtyManager was stopped and cleaned up
                mock_pty_instance.stop.assert_called_once()

                # Verify the output is returned successfully
                assert "test output" in result["content"][0]["text"]
                assert result["status"] == "success"


@pytest.mark.parametrize(
    "code,expected",
    [
        ("print('Hello')", "Hello"),
        ("import sys; print('Python version:', sys.version)", "Python version:"),
        ("for i in range(3): print(i)", "0\n1\n2"),
    ],
)
def test_agent_interface(agent, code, expected):
    """Test calling python_repl through the Agent interface."""
    # Use non_interactive_mode to bypass confirmation
    result = agent.tool.python_repl(code=code, interactive=False, non_interactive_mode=True)

    # Extract the response text
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        content = result["content"][0]["text"]
    else:
        content = str(result)

    assert expected in content


def test_clean_ansi():
    """Test the clean_ansi function."""
    ansi_text = "\033[31mRed text\033[0m \033[1mBold text\033[0m"
    clean_text = python_repl.clean_ansi(ansi_text)
    assert clean_text == "Red text Bold text"


class TestPtyManager:
    """Test the PtyManager class."""

    @pytest.mark.skip("broken")
    def test_start_and_get_output(self):
        """Test starting a PTY and getting output."""
        if sys.platform != "linux":
            pytest.skip("PTY tests require Linux")

        pty_mgr = python_repl.PtyManager()
        pty_mgr.start("print('PTY test')")

        # Allow time for execution
        import time

        time.sleep(0.5)

        output = pty_mgr.get_output()
        pty_mgr.stop()

        assert "PTY test" in output

    def test_stop_cleanup(self):
        """Test that stop cleans up resources."""
        if sys.platform != "linux":
            pytest.skip("PTY tests require Linux")

        pty_mgr = python_repl.PtyManager()
        pty_mgr.start("import time; time.sleep(0.5)")

        # Store the PID
        pid = pty_mgr.pid

        # Stop should kill the process
        pty_mgr.stop()

        # Check if process still exists
        try:
            os.kill(pid, 0)  # Signal 0 tests if process exists
            process_exists = True
        except ProcessLookupError:
            process_exists = False

        assert not process_exists, "Process should have been terminated"

    def test_pty_manager_with_callback(self):
        """Test PtyManager with callback function."""
        callback_outputs = []

        def callback(output):
            callback_outputs.append(output)

        # Mock the select and os.read functions to simulate output
        with (
            patch("select.select") as mock_select,
            patch("os.read") as mock_read,
            patch("os.close"),
            patch("os.fork", return_value=12345),
            patch("os.kill"),
            patch("os.waitpid"),
        ):
            # Configure mock select to return that supervisor_fd is ready
            mock_select.side_effect = lambda r, w, x, timeout: ([r[0]], [], [])

            # Configure mock read to return some data and then raise OSError to exit the loop
            mock_read.side_effect = [b"Line 1\n", b"Line 2\n", OSError()]

            pty_mgr = python_repl.PtyManager(callback=callback)

            # Set file descriptors directly to avoid actual PTY creation
            pty_mgr.supervisor_fd = 10
            pty_mgr.worker_fd = 11

            # Start the thread to read output
            read_thread = threading.Thread(target=pty_mgr._read_output)
            read_thread.daemon = True
            read_thread.start()

            # Allow thread to run
            time.sleep(0.1)

            # Stop the thread
            pty_mgr.stop_event.set()
            read_thread.join(timeout=1.0)

            # Verify callback received output
            assert len(callback_outputs) >= 1  # Should have at least one output
            assert any("Line 1" in output for output in callback_outputs)

    def test_handle_input(self):
        """Test PtyManager's input handling."""
        # Mock required functions
        with (
            patch("select.select") as mock_select,
            patch("sys.stdin.read") as mock_read,
            patch("os.write") as _,
        ):
            # Configure mocks
            mock_select.side_effect = [
                ([sys.stdin], [], []),
                OSError(),
            ]  # First return stdin is ready, then exit
            mock_read.side_effect = ["t", "e", "s", "t", "\n"]  # Simulate typing "test"

            pty_mgr = python_repl.PtyManager()
            pty_mgr.supervisor_fd = 10  # Arbitrary FD for testing

            # Start the input handler thread
            input_thread = threading.Thread(target=pty_mgr._handle_input)
            input_thread.daemon = True
            input_thread.start()

            # Allow thread to run
            time.sleep(0.1)

            # Stop the thread
            pty_mgr.stop_event.set()
            input_thread.join(timeout=1.0)

            # Just verify no exceptions occurred
            # Input handling is difficult to test due to stdin interaction

    def test_get_output_binary_truncation(self):
        """Test that binary content is truncated in get_output."""
        pty_mgr = python_repl.PtyManager()

        # Add binary-looking content to the output buffer
        binary_content = "\\x00\\x01" * 100  # Long binary content
        pty_mgr.output_buffer = [binary_content]

        output = pty_mgr.get_output()

        # Verify truncation occurred
        assert "[binary content truncated]" in output
        assert len(output) < len(binary_content)
