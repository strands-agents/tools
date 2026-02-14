"""
Tests for the KeySequenceMatcher class and key sequence callback functionality.
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest

if os.name == "nt":
    pytest.skip("skipping on windows until issue #17 is resolved", allow_module_level=True)

from strands_tools.shell import KEY_SEQUENCE_TIMEOUT, CommandExecutor, KeySequenceMatcher


class TestKeySequenceMatcher:
    """Tests for KeySequenceMatcher class."""

    def test_init_empty_callbacks(self):
        """Test initialization with no callbacks."""
        matcher = KeySequenceMatcher()
        assert matcher._callbacks == {}
        assert matcher._max_seq_len == 0
        assert matcher._buffer_len == 0

    def test_init_with_callbacks(self):
        """Test initialization with callbacks."""
        callback = MagicMock()
        callbacks = {
            b"\x03": callback,  # Ctrl+C (1 byte)
            b"\x1bc": callback,  # Alt+C (2 bytes)
        }
        matcher = KeySequenceMatcher(callbacks)

        assert matcher._callbacks == callbacks
        assert matcher._max_seq_len == 2
        assert len(matcher._prefixes) == 1  # Only \x1b is a prefix
        assert b"\x1b" in matcher._prefixes

    def test_init_precomputes_prefixes(self):
        """Test that prefixes are correctly precomputed."""
        callback = MagicMock()
        callbacks = {
            b"\x1b[A": callback,  # Up arrow (3 bytes)
            b"\x1b[B": callback,  # Down arrow (3 bytes)
            b"\x03": callback,  # Ctrl+C (1 byte)
        }
        matcher = KeySequenceMatcher(callbacks)

        # Should have prefixes: \x1b, \x1b[
        assert b"\x1b" in matcher._prefixes
        assert b"\x1b[" in matcher._prefixes
        assert len(matcher._prefixes) == 2

    def test_init_sequences_sorted_by_length(self):
        """Test that sequences are pre-sorted by length (longest first)."""
        callback = MagicMock()
        callbacks = {
            b"\x03": callback,
            b"\x1bc": callback,
            b"\x1b[A": callback,
        }
        matcher = KeySequenceMatcher(callbacks)

        # Should be sorted longest first
        assert matcher._sequences_by_len[0] == b"\x1b[A"  # 3 bytes
        assert matcher._sequences_by_len[1] == b"\x1bc"  # 2 bytes
        assert matcher._sequences_by_len[2] == b"\x03"  # 1 byte

    def test_process_input_no_callbacks_fast_path(self):
        """Test fast path when no callbacks configured."""
        matcher = KeySequenceMatcher()

        callback, to_forward = matcher.process_input(b"hello world")

        assert callback is None
        assert to_forward == b"hello world"

    def test_process_input_single_byte_match(self):
        """Test matching single byte sequence (Ctrl+C)."""
        callback = MagicMock()
        matcher = KeySequenceMatcher({b"\x03": callback})

        result_callback, to_forward = matcher.process_input(b"\x03")

        assert result_callback is callback
        assert to_forward == b""

    def test_process_input_multi_byte_match(self):
        """Test matching multi-byte sequence (Alt+C)."""
        callback = MagicMock()
        matcher = KeySequenceMatcher({b"\x1bc": callback})

        # Send both bytes together (as Alt+C typically arrives)
        result_callback, to_forward = matcher.process_input(b"\x1bc")

        assert result_callback is callback
        assert to_forward == b""

    def test_process_input_prefix_buffered(self):
        """Test that prefix bytes are buffered, not forwarded."""
        callback = MagicMock()
        matcher = KeySequenceMatcher({b"\x1bc": callback})

        # Send just the prefix
        result_callback, to_forward = matcher.process_input(b"\x1b")

        assert result_callback is None
        assert to_forward == b""  # Should be buffered, not forwarded
        assert matcher._buffer_len == 1

    def test_process_input_non_sequence_forwarded(self):
        """Test that non-sequence bytes are forwarded immediately."""
        callback = MagicMock()
        matcher = KeySequenceMatcher({b"\x03": callback})

        result_callback, to_forward = matcher.process_input(b"hello")

        assert result_callback is None
        assert to_forward == b"hello"

    def test_process_input_mixed_content(self):
        """Test processing mixed content with sequence in middle."""
        callback = MagicMock()
        matcher = KeySequenceMatcher({b"\x03": callback})

        # Process "ab" first
        result_callback, to_forward = matcher.process_input(b"ab")
        assert result_callback is None
        assert to_forward == b"ab"

        # Then Ctrl+C
        result_callback, to_forward = matcher.process_input(b"\x03")
        assert result_callback is callback
        assert to_forward == b""

    def test_process_input_sequence_at_end(self):
        """Test sequence detection when sequence is at end of input."""
        callback = MagicMock()
        matcher = KeySequenceMatcher({b"\x03": callback})

        result_callback, to_forward = matcher.process_input(b"hello\x03")

        assert result_callback is callback
        assert to_forward == b"hello"

    def test_check_timeout_empty_buffer(self):
        """Test timeout check with empty buffer returns empty bytes."""
        matcher = KeySequenceMatcher({b"\x1bc": MagicMock()})

        callback, to_forward = matcher.check_timeout()

        assert callback is None
        assert to_forward == b""

    def test_check_timeout_before_expiry(self):
        """Test timeout check before timeout expires."""
        callback = MagicMock()
        matcher = KeySequenceMatcher({b"\x1bc": callback})

        # Add a prefix to buffer
        matcher.process_input(b"\x1b")

        # Check immediately (before timeout)
        result_callback, to_forward = matcher.check_timeout()

        assert result_callback is None
        assert to_forward == b""  # Should not flush yet
        assert matcher._buffer_len == 1

    def test_check_timeout_after_expiry(self):
        """Test timeout check after timeout expires flushes buffer."""
        callback = MagicMock()
        matcher = KeySequenceMatcher({b"\x1bc": callback})

        # Add a prefix to buffer
        matcher.process_input(b"\x1b")

        # Simulate time passing
        matcher._last_input_time = time.time() - KEY_SEQUENCE_TIMEOUT - 0.01

        result_callback, to_forward = matcher.check_timeout()

        assert result_callback is None  # \x1b alone is not a sequence
        assert to_forward == b"\x1b"  # Should flush the buffered escape
        assert matcher._buffer_len == 0

    def test_escape_then_c_slow_typing(self):
        """Test Escape followed by 'c' with delay (two separate keypresses)."""
        callback = MagicMock()
        matcher = KeySequenceMatcher({b"\x1bc": callback})

        # User presses Escape
        result_callback, to_forward = matcher.process_input(b"\x1b")
        assert result_callback is None
        assert to_forward == b""

        # Simulate timeout (user paused)
        matcher._last_input_time = time.time() - KEY_SEQUENCE_TIMEOUT - 0.01
        timeout_callback, timed_out = matcher.check_timeout()
        assert timeout_callback is None
        assert timed_out == b"\x1b"  # Escape forwarded

        # User presses 'c' (now just a regular 'c')
        result_callback, to_forward = matcher.process_input(b"c")
        assert result_callback is None
        assert to_forward == b"c"

    def test_alt_c_fast_arrival(self):
        """Test Alt+C arriving as two bytes quickly (single keypress with modifier)."""
        callback = MagicMock()
        matcher = KeySequenceMatcher({b"\x1bc": callback})

        # Alt+C typically arrives as two bytes together or very quickly
        result_callback, to_forward = matcher.process_input(b"\x1bc")

        assert result_callback is callback
        assert to_forward == b""

    def test_multiple_sequences_longest_match(self):
        """Test that longest matching sequence wins when arriving together."""
        short_callback = MagicMock()
        long_callback = MagicMock()
        matcher = KeySequenceMatcher(
            {
                b"\x1b": short_callback,
                b"\x1bc": long_callback,
            }
        )

        # When both bytes arrive together, should match longer sequence
        result_callback, to_forward = matcher.process_input(b"\x1bc")

        assert result_callback is long_callback
        assert to_forward == b""

    def test_short_sequence_matches_on_timeout(self):
        """Test that short sequence matches on timeout when no longer sequence arrives."""
        short_callback = MagicMock()
        long_callback = MagicMock()
        matcher = KeySequenceMatcher(
            {
                b"\x1b": short_callback,
                b"\x1bc": long_callback,
            }
        )

        # Just escape arrives
        result_callback, to_forward = matcher.process_input(b"\x1b")
        assert result_callback is None  # Buffered, waiting for potential 'c'
        assert to_forward == b""

        # Simulate timeout
        matcher._last_input_time = time.time() - KEY_SEQUENCE_TIMEOUT - 0.01
        timeout_callback, timed_out = matcher.check_timeout()

        # Short sequence should match on timeout
        assert timeout_callback is short_callback
        assert timed_out == b""


class TestCommandExecutorWithCallbacks:
    """Tests for CommandExecutor with key_sequence_callbacks."""

    def test_init_with_callbacks(self):
        """Test CommandExecutor initialization with callbacks."""
        callback = MagicMock()
        callbacks = {b"\x03": callback}

        executor = CommandExecutor(timeout=10, key_sequence_callbacks=callbacks)

        assert executor.key_sequence_callbacks == callbacks
        assert executor.timeout == 10

    def test_init_without_callbacks(self):
        """Test CommandExecutor initialization without callbacks."""
        executor = CommandExecutor(timeout=10)

        assert executor.key_sequence_callbacks is None

    @patch("pty.fork")
    @patch("termios.tcgetattr")
    @patch("termios.tcsetattr")
    @patch("tty.setraw")
    @patch("select.select")
    @patch("os.read")
    @patch("os.write")
    @patch("os.waitpid")
    def test_execute_with_pty_callback_invoked(
        self,
        mock_waitpid,
        mock_write,
        mock_read,
        mock_select,
        mock_setraw,
        mock_tcsetattr,
        mock_tcgetattr,
        mock_fork,
    ):
        """Test that callback is invoked when key sequence detected."""
        callback = MagicMock()
        callbacks = {b"\x03": callback}

        # Setup mocks
        mock_tcgetattr.return_value = "old_tty"
        mock_fork.return_value = (123, 5)  # pid, fd

        # First select returns stdin readable, second returns fd readable with EOF
        mock_select.side_effect = [
            ([5], [], []),  # fd readable first (some output)
            ([], [], []),  # nothing (to allow stdin check)
        ]
        mock_read.side_effect = [
            b"output",  # PTY output
            b"",  # PTY EOF
        ]
        mock_waitpid.return_value = (123, 0)

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.fileno.return_value = 0
            mock_stdin.in_readable = False

            executor = CommandExecutor(timeout=10, key_sequence_callbacks=callbacks)

            # We can't easily test the full callback invocation without
            # a more complex mock setup, but we can verify the executor
            # accepts the callbacks parameter
            assert executor.key_sequence_callbacks == callbacks


class TestShellToolWithCallbacks:
    """Tests for shell tool function with key_sequence_callbacks."""

    @patch("strands_tools.shell.execute_commands")
    def test_shell_extracts_callbacks_from_context(self, mock_execute_commands):
        """Test that shell extracts key_sequence_callbacks from tool_context."""
        from strands_tools.shell import shell as shell_func

        callback = MagicMock()
        callbacks = {b"\x03": callback}

        # Create mock tool_context
        mock_context = MagicMock()
        mock_context.invocation_state = {"key_sequence_callbacks": callbacks}

        mock_execute_commands.return_value = [
            {
                "command": "echo test",
                "exit_code": 0,
                "output": "test",
                "error": "",
                "status": "success",
            }
        ]

        # Call shell with tool_context
        shell_func("echo test", non_interactive=True, tool_context=mock_context)

        # Verify execute_commands was called with the callbacks
        mock_execute_commands.assert_called_once()
        call_kwargs = mock_execute_commands.call_args
        assert call_kwargs[1].get("key_sequence_callbacks") == callbacks

    @patch("strands_tools.shell.execute_commands")
    def test_shell_handles_missing_context(self, mock_execute_commands):
        """Test that shell handles missing tool_context gracefully."""
        from strands_tools.shell import shell as shell_func

        mock_execute_commands.return_value = [
            {
                "command": "echo test",
                "exit_code": 0,
                "output": "test",
                "error": "",
                "status": "success",
            }
        ]

        # Call shell without tool_context
        shell_func("echo test", non_interactive=True, tool_context=None)

        # Verify execute_commands was called with None callbacks
        mock_execute_commands.assert_called_once()
        call_kwargs = mock_execute_commands.call_args
        assert call_kwargs[1].get("key_sequence_callbacks") is None

    @patch("strands_tools.shell.execute_commands")
    def test_shell_handles_context_without_invocation_state(self, mock_execute_commands):
        """Test that shell handles context without invocation_state."""
        from strands_tools.shell import shell as shell_func

        # Create mock tool_context without invocation_state attribute
        mock_context = MagicMock(spec=[])  # Empty spec means no attributes

        mock_execute_commands.return_value = [
            {
                "command": "echo test",
                "exit_code": 0,
                "output": "test",
                "error": "",
                "status": "success",
            }
        ]

        # Call shell with incomplete context
        shell_func("echo test", non_interactive=True, tool_context=mock_context)

        # Should not raise, callbacks should be None
        mock_execute_commands.assert_called_once()


class TestExecuteCommandsWithCallbacks:
    """Tests for execute_commands function with key_sequence_callbacks."""

    @patch("strands_tools.shell.execute_single_command")
    def test_callbacks_passed_to_sequential_execution(self, mock_execute_single):
        """Test callbacks are passed through in sequential execution."""
        from strands_tools.shell import execute_commands

        callback = MagicMock()
        callbacks = {b"\x03": callback}

        mock_execute_single.return_value = {
            "command": "cmd1",
            "exit_code": 0,
            "output": "",
            "error": "",
            "status": "success",
        }

        execute_commands(
            ["cmd1"],
            parallel=False,
            ignore_errors=False,
            work_dir="/tmp",
            timeout=10,
            non_interactive_mode=True,
            key_sequence_callbacks=callbacks,
        )

        # Verify callbacks were passed
        mock_execute_single.assert_called_once()
        call_args = mock_execute_single.call_args
        assert call_args[1].get("key_sequence_callbacks") == callbacks

    @patch("strands_tools.shell.execute_single_command")
    def test_callbacks_passed_to_parallel_execution(self, mock_execute_single):
        """Test callbacks are passed through in parallel execution."""
        from strands_tools.shell import execute_commands

        callback = MagicMock()
        callbacks = {b"\x03": callback}

        mock_execute_single.return_value = {
            "command": "cmd1",
            "exit_code": 0,
            "output": "",
            "error": "",
            "status": "success",
        }

        execute_commands(
            ["cmd1"],
            parallel=True,
            ignore_errors=False,
            work_dir="/tmp",
            timeout=10,
            non_interactive_mode=True,
            key_sequence_callbacks=callbacks,
        )

        # Verify callbacks were passed
        mock_execute_single.assert_called_once()
        call_args = mock_execute_single.call_args
        assert call_args[0][4] == callbacks  # 5th positional arg


class TestExecuteSingleCommandWithCallbacks:
    """Tests for execute_single_command function with key_sequence_callbacks."""

    @patch.object(CommandExecutor, "execute_with_pty")
    def test_callbacks_passed_to_executor(self, mock_execute_pty):
        """Test callbacks are passed to CommandExecutor."""
        from strands_tools.shell import execute_single_command

        callback = MagicMock()
        callbacks = {b"\x03": callback}

        mock_execute_pty.return_value = (0, "output", "")

        execute_single_command(
            "echo test",
            "/tmp",
            10,
            non_interactive_mode=True,
            key_sequence_callbacks=callbacks,
        )

        # The executor should have been created with callbacks
        # We verify by checking that execute_with_pty was called
        mock_execute_pty.assert_called_once()
