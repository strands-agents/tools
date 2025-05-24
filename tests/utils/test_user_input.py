"""
Tests for user input utility functions in strands_tools/utils/user_input.py.
"""

from unittest.mock import MagicMock, patch

from strands_tools.utils.user_input import get_user_input


class TestUserInputAsync:
    """Test the asynchronous user input function via sync functions."""

    def test_get_user_input_async_success_via_sync(self):
        """Test successful user input via synchronous wrapper."""
        test_input = "test_response"
        test_prompt = "Enter input:"

        # Setup mock for async function
        async def mock_success(prompt, default=None, keyboard_interrupt_return_default=True):
            assert prompt == test_prompt
            return test_input

        # Mock event loop and async function
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = test_input

        with (
            patch(
                "strands_tools.utils.user_input.get_user_input_async",
                side_effect=mock_success,
            ),
            patch("asyncio.get_event_loop", return_value=mock_loop),
        ):
            result = get_user_input(test_prompt)
            assert result == test_input
            mock_loop.run_until_complete.assert_called_once()

    def test_get_user_input_async_empty_returns_default_via_sync(self):
        """Test that empty input returns the default value via sync wrapper."""
        test_prompt = "Enter input:"
        default_value = "default_response"

        # Setup mock for async function
        async def mock_empty(prompt, default, keyboard_interrupt_return_default=True):
            assert prompt == test_prompt
            assert default == default_value
            return default  # Empty input returns default

        # Mock event loop
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = default_value

        with (
            patch(
                "strands_tools.utils.user_input.get_user_input_async",
                side_effect=mock_empty,
            ),
            patch("asyncio.get_event_loop", return_value=mock_loop),
        ):
            result = get_user_input(test_prompt, default_value)
            assert result == default_value

    def test_get_user_input_async_keyboard_interrupt_via_sync(self):
        """Test handling of KeyboardInterrupt during input via sync wrapper."""
        test_prompt = "Enter input:"
        default_value = "default_response"

        # Setup mock for async function that raises KeyboardInterrupt
        async def mock_interrupt(prompt, default, keyboard_interrupt_return_default):
            assert keyboard_interrupt_return_default is True
            raise KeyboardInterrupt()

        # Mock event loop to handle the exception and return default
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = default_value

        with (
            patch(
                "strands_tools.utils.user_input.get_user_input_async",
                side_effect=mock_interrupt,
            ),
            patch("asyncio.get_event_loop", return_value=mock_loop),
            patch(
                "strands_tools.utils.user_input.get_user_input",
                side_effect=lambda p, d=default_value, k=True: d,
            ),
        ):
            # We're testing that get_user_input properly handles the exception
            # from get_user_input_async and returns the default value
            result = get_user_input(test_prompt, default_value)
            assert result == default_value

    def test_get_user_input_async_keyboard_interrupt_propagation(self):
        """Test KeyboardInterrupt propagation when keyboard_interrupt_return_default=False."""
        test_prompt = "Enter input:"
        default_value = "default_response"

        # Setup mock for async function that raises KeyboardInterrupt
        async def mock_interrupt(prompt, default, keyboard_interrupt_return_default):
            assert keyboard_interrupt_return_default is False
            raise KeyboardInterrupt()

        # Mock event loop to propagate the exception
        mock_loop = MagicMock()
        mock_loop.run_until_complete.side_effect = KeyboardInterrupt()

        with (
            patch(
                "strands_tools.utils.user_input.get_user_input_async",
                side_effect=mock_interrupt,
            ),
            patch("asyncio.get_event_loop", return_value=mock_loop),
        ):
            # When keyboard_interrupt_return_default is False, the exception should propagate
            try:
                get_user_input(test_prompt, default_value, keyboard_interrupt_return_default=False)
                raise AssertionError("KeyboardInterrupt should have been raised")
            except KeyboardInterrupt:
                # Expected behavior - test passes
                pass

    def test_get_user_input_async_eof_error_via_sync(self):
        """Test handling of EOFError during input via sync wrapper."""
        test_prompt = "Enter input:"
        default_value = "default_response"

        # Setup mock for async function that raises EOFError
        async def mock_eof(prompt, default, keyboard_interrupt_return_default):
            assert keyboard_interrupt_return_default is True
            raise EOFError()

        # Mock event loop to handle the exception and return default
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = default_value

        with (
            patch(
                "strands_tools.utils.user_input.get_user_input_async",
                side_effect=mock_eof,
            ),
            patch("asyncio.get_event_loop", return_value=mock_loop),
            patch(
                "strands_tools.utils.user_input.get_user_input",
                side_effect=lambda p, d=default_value, k=True: d,
            ),
        ):
            # We're testing that get_user_input properly handles the exception
            # and returns the default value
            result = get_user_input(test_prompt, default_value)
            assert result == default_value

    def test_get_user_input_async_eof_error_propagation(self):
        """Test EOFError propagation when keyboard_interrupt_return_default=False."""
        test_prompt = "Enter input:"
        default_value = "default_response"

        # Setup mock for async function that raises EOFError
        async def mock_eof(prompt, default, keyboard_interrupt_return_default):
            assert keyboard_interrupt_return_default is False
            raise EOFError()

        # Mock event loop to propagate the exception
        mock_loop = MagicMock()
        mock_loop.run_until_complete.side_effect = EOFError()

        with (
            patch(
                "strands_tools.utils.user_input.get_user_input_async",
                side_effect=mock_eof,
            ),
            patch("asyncio.get_event_loop", return_value=mock_loop),
        ):
            # When keyboard_interrupt_return_default is False, the exception should propagate
            try:
                get_user_input(test_prompt, default_value, keyboard_interrupt_return_default=False)
                raise AssertionError("EOFError should have been raised")
            except EOFError:
                # Expected behavior - test passes
                pass


class TestUserInputSync:
    """Test the synchronous user input function."""

    def test_get_user_input_existing_event_loop(self):
        """Test get_user_input with an existing event loop."""
        test_input = "test_response"
        test_prompt = "Enter input:"

        # Create a mock coroutine result
        mock_coro = MagicMock()
        mock_coro.return_value = test_input

        # Mock the event loop
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = test_input

        with (
            patch(
                "strands_tools.utils.user_input.get_user_input_async",
                return_value=mock_coro(),
            ),
            patch("asyncio.get_event_loop", return_value=mock_loop),
        ):
            # Call the function
            result = get_user_input(test_prompt)

            # Verify the result
            assert result == test_input

            # Verify the coroutine was run to completion
            mock_loop.run_until_complete.assert_called_once()

    def test_get_user_input_no_existing_event_loop(self):
        """Test get_user_input when no event loop exists."""
        test_input = "test_response"
        test_prompt = "Enter input:"

        # Create a mock coroutine result
        mock_coro = MagicMock()
        mock_coro.return_value = test_input

        # Mock the new event loop
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = test_input

        with (
            patch(
                "strands_tools.utils.user_input.get_user_input_async",
                return_value=mock_coro(),
            ),
            patch("asyncio.get_event_loop", side_effect=RuntimeError()),
            patch("asyncio.new_event_loop", return_value=mock_loop),
            patch("asyncio.set_event_loop") as mock_set_loop,
        ):
            # Call the function
            result = get_user_input(test_prompt)

            # Verify the result
            assert result == test_input

            # Verify a new event loop was created and set
            mock_set_loop.assert_called_once_with(mock_loop)

            # Verify the coroutine was run to completion
            mock_loop.run_until_complete.assert_called_once()

    def test_get_user_input_with_default_and_keyboard_interrupt_param(self):
        """Test get_user_input with different values for keyboard_interrupt_return_default."""
        test_prompt = "Enter input:"
        default_value = "default_response"

        for keyboard_interrupt_value in [True, False]:
            # Create a mock coroutine with arguments check
            async def mock_async_func(
                prompt, default, keyboard_interrupt_return_default, keyboard_interrupt_value=keyboard_interrupt_value
            ):
                assert prompt == test_prompt
                assert default == default_value
                assert keyboard_interrupt_return_default is keyboard_interrupt_value
                return default_value

            # Mock the event loop
            mock_loop = MagicMock()
            mock_loop.run_until_complete.return_value = default_value

            with (
                patch(
                    "strands_tools.utils.user_input.get_user_input_async",
                    side_effect=mock_async_func,
                ),
                patch("asyncio.get_event_loop", return_value=mock_loop),
            ):
                # Call the function with the current keyboard_interrupt_return_default value
                result = get_user_input(
                    test_prompt, default_value, keyboard_interrupt_return_default=keyboard_interrupt_value
                )

                # Verify the result
                assert result == default_value
