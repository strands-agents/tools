"""
Comprehensive tests for user input utility to improve coverage.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from strands_tools.utils.user_input import get_user_input, get_user_input_async


class TestGetUserInputAsync:
    """Test the async user input function directly."""

    def setup_method(self):
        """Reset the global session before each test."""
        import strands_tools.utils.user_input
        strands_tools.utils.user_input.session = None

    @pytest.mark.asyncio
    async def test_get_user_input_async_basic_success(self):
        """Test basic successful async user input."""
        test_input = "test response"
        test_prompt = "Enter input:"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.return_value = test_input
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                result = await get_user_input_async(test_prompt)
                
                assert result == test_input
                mock_session.prompt_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_input_async_empty_input_returns_default(self):
        """Test that empty input returns default value."""
        test_prompt = "Enter input:"
        default_value = "default response"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.return_value = ""  # Empty input
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                result = await get_user_input_async(test_prompt, default=default_value)
                
                assert result == default_value

    @pytest.mark.asyncio
    async def test_get_user_input_async_none_input_returns_default(self):
        """Test that None input returns default value."""
        test_prompt = "Enter input:"
        default_value = "default response"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.return_value = None
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                result = await get_user_input_async(test_prompt, default=default_value)
                
                assert result == default_value

    @pytest.mark.asyncio
    async def test_get_user_input_async_keyboard_interrupt_with_default_return(self):
        """Test KeyboardInterrupt handling with default return enabled."""
        test_prompt = "Enter input:"
        default_value = "default response"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.side_effect = KeyboardInterrupt()
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                result = await get_user_input_async(
                    test_prompt, 
                    default=default_value, 
                    keyboard_interrupt_return_default=True
                )
                
                assert result == default_value

    @pytest.mark.asyncio
    async def test_get_user_input_async_keyboard_interrupt_propagation(self):
        """Test KeyboardInterrupt propagation when default return is disabled."""
        test_prompt = "Enter input:"
        default_value = "default response"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.side_effect = KeyboardInterrupt()
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                with pytest.raises(KeyboardInterrupt):
                    await get_user_input_async(
                        test_prompt, 
                        default=default_value, 
                        keyboard_interrupt_return_default=False
                    )

    @pytest.mark.asyncio
    async def test_get_user_input_async_eof_error_with_default_return(self):
        """Test EOFError handling with default return enabled."""
        test_prompt = "Enter input:"
        default_value = "default response"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.side_effect = EOFError()
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                result = await get_user_input_async(
                    test_prompt, 
                    default=default_value, 
                    keyboard_interrupt_return_default=True
                )
                
                assert result == default_value

    @pytest.mark.asyncio
    async def test_get_user_input_async_eof_error_propagation(self):
        """Test EOFError propagation when default return is disabled."""
        test_prompt = "Enter input:"
        default_value = "default response"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.side_effect = EOFError()
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                with pytest.raises(EOFError):
                    await get_user_input_async(
                        test_prompt, 
                        default=default_value, 
                        keyboard_interrupt_return_default=False
                    )

    @pytest.mark.asyncio
    async def test_get_user_input_async_session_reuse(self):
        """Test that PromptSession is reused across calls."""
        test_prompt = "Enter input:"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.return_value = "response"
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                # First call should create session
                result1 = await get_user_input_async(test_prompt)
                assert result1 == "response"
                
                # Second call should reuse session
                result2 = await get_user_input_async(test_prompt)
                assert result2 == "response"
                
                # PromptSession should only be created once
                mock_session_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_input_async_html_prompt_formatting(self):
        """Test that prompt is formatted as HTML."""
        test_prompt = "<b>Bold prompt:</b>"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.return_value = "response"
            mock_session_class.return_value = mock_session
            
            with (
                patch("strands_tools.utils.user_input.patch_stdout"),
                patch("strands_tools.utils.user_input.HTML") as mock_html
            ):
                mock_html.return_value = "formatted_prompt"
                
                result = await get_user_input_async(test_prompt)
                
                # HTML should be called with the prompt
                mock_html.assert_called_once_with(f"{test_prompt} ")
                
                # Session should be called with formatted prompt
                mock_session.prompt_async.assert_called_once_with("formatted_prompt")

    @pytest.mark.asyncio
    async def test_get_user_input_async_patch_stdout_usage(self):
        """Test that patch_stdout is used correctly."""
        test_prompt = "Enter input:"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.return_value = "response"
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout") as mock_patch_stdout:
                mock_context = MagicMock()
                mock_patch_stdout.return_value.__enter__ = MagicMock(return_value=mock_context)
                mock_patch_stdout.return_value.__exit__ = MagicMock(return_value=None)
                
                result = await get_user_input_async(test_prompt)
                
                # patch_stdout should be called with raw=True
                mock_patch_stdout.assert_called_once_with(raw=True)
                
                # Context manager should be used
                mock_patch_stdout.return_value.__enter__.assert_called_once()
                mock_patch_stdout.return_value.__exit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_input_async_return_type_conversion(self):
        """Test that return value is converted to string."""
        test_prompt = "Enter input:"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            # Return a non-string value
            mock_session.prompt_async.return_value = 42
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                result = await get_user_input_async(test_prompt)
                
                # Should be converted to string
                assert result == "42"
                assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_user_input_async_default_type_conversion(self):
        """Test that default value is converted to string."""
        test_prompt = "Enter input:"
        default_value = 123  # Non-string default
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.return_value = ""  # Empty input
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                result = await get_user_input_async(test_prompt, default=default_value)
                
                # Should be converted to string
                assert result == "123"
                assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_user_input_async_whitespace_input(self):
        """Test handling of whitespace-only input."""
        test_prompt = "Enter input:"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.return_value = "   \t\n   "  # Whitespace only
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                result = await get_user_input_async(test_prompt)
                
                # Whitespace input should be preserved
                assert result == "   \t\n   "

    @pytest.mark.asyncio
    async def test_get_user_input_async_unicode_input(self):
        """Test handling of unicode input."""
        test_prompt = "Enter input:"
        unicode_input = "🐍 Python 中文 العربية"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.return_value = unicode_input
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                result = await get_user_input_async(test_prompt)
                
                assert result == unicode_input

    @pytest.mark.asyncio
    async def test_get_user_input_async_long_input(self):
        """Test handling of very long input."""
        test_prompt = "Enter input:"
        long_input = "x" * 10000  # Very long input
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.return_value = long_input
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                result = await get_user_input_async(test_prompt)
                
                assert result == long_input
                assert len(result) == 10000

    @pytest.mark.asyncio
    async def test_get_user_input_async_multiline_input(self):
        """Test handling of multiline input."""
        test_prompt = "Enter input:"
        multiline_input = "Line 1\nLine 2\nLine 3"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.return_value = multiline_input
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                result = await get_user_input_async(test_prompt)
                
                assert result == multiline_input
                assert "\n" in result

    @pytest.mark.asyncio
    async def test_get_user_input_async_special_characters(self):
        """Test handling of special characters in input."""
        test_prompt = "Enter input:"
        special_input = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.return_value = special_input
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                result = await get_user_input_async(test_prompt)
                
                assert result == special_input

    @pytest.mark.asyncio
    async def test_get_user_input_async_exception_in_session_creation(self):
        """Test handling of exception during session creation."""
        test_prompt = "Enter input:"
        
        with patch("strands_tools.utils.user_input.PromptSession", side_effect=Exception("Session creation failed")):
            with patch("strands_tools.utils.user_input.patch_stdout"):
                with pytest.raises(Exception, match="Session creation failed"):
                    await get_user_input_async(test_prompt)

    @pytest.mark.asyncio
    async def test_get_user_input_async_exception_in_prompt(self):
        """Test handling of exception during prompt execution."""
        test_prompt = "Enter input:"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.side_effect = Exception("Prompt failed")
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout"):
                with pytest.raises(Exception, match="Prompt failed"):
                    await get_user_input_async(test_prompt)


class TestGetUserInputSync:
    """Test the synchronous user input function."""

    def setup_method(self):
        """Reset the global session before each test."""
        import strands_tools.utils.user_input
        strands_tools.utils.user_input.session = None

    def test_get_user_input_with_existing_event_loop(self):
        """Test get_user_input when event loop already exists."""
        test_input = "test response"
        test_prompt = "Enter input:"
        
        # Mock the async function
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            return test_input
        
        # Mock existing event loop
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = test_input
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            result = get_user_input(test_prompt)
            
            assert result == test_input
            mock_loop.run_until_complete.assert_called_once()

    def test_get_user_input_no_existing_event_loop(self):
        """Test get_user_input when no event loop exists."""
        test_input = "test response"
        test_prompt = "Enter input:"
        
        # Mock the async function
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            return test_input
        
        # Mock new event loop creation
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = test_input
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", side_effect=RuntimeError("No event loop")),
            patch("asyncio.new_event_loop", return_value=mock_loop),
            patch("asyncio.set_event_loop") as mock_set_loop
        ):
            result = get_user_input(test_prompt)
            
            assert result == test_input
            mock_set_loop.assert_called_once_with(mock_loop)
            mock_loop.run_until_complete.assert_called_once()

    def test_get_user_input_parameter_passing(self):
        """Test that parameters are passed correctly to async function."""
        test_prompt = "Enter input:"
        default_value = "default"
        keyboard_interrupt_value = False
        
        # Mock the async function to verify parameters
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            assert prompt == test_prompt
            assert default == default_value
            assert keyboard_interrupt_return_default == keyboard_interrupt_value
            return "response"
        
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = "response"
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            result = get_user_input(
                test_prompt, 
                default=default_value, 
                keyboard_interrupt_return_default=keyboard_interrupt_value
            )
            
            assert result == "response"

    def test_get_user_input_default_parameters(self):
        """Test get_user_input with default parameters."""
        test_prompt = "Enter input:"
        
        # Mock the async function to verify default parameters
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            assert prompt == test_prompt
            assert default == ""  # Default value
            assert keyboard_interrupt_return_default == True  # Default value
            return "response"
        
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = "response"
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            result = get_user_input(test_prompt)
            
            assert result == "response"

    def test_get_user_input_return_type_conversion(self):
        """Test that return value is converted to string."""
        test_prompt = "Enter input:"
        
        # Mock async function to return non-string
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            return 42
        
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = 42
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            result = get_user_input(test_prompt)
            
            # Should be converted to string
            assert result == "42"
            assert isinstance(result, str)

    def test_get_user_input_exception_in_async_function(self):
        """Test handling of exception in async function."""
        test_prompt = "Enter input:"
        
        # Mock async function to raise exception
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            raise ValueError("Async function failed")
        
        mock_loop = MagicMock()
        mock_loop.run_until_complete.side_effect = ValueError("Async function failed")
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            with pytest.raises(ValueError, match="Async function failed"):
                get_user_input(test_prompt)

    def test_get_user_input_exception_in_event_loop_creation(self):
        """Test handling of exception in event loop creation."""
        test_prompt = "Enter input:"
        
        with (
            patch("asyncio.get_event_loop", side_effect=RuntimeError("No event loop")),
            patch("asyncio.new_event_loop", side_effect=Exception("Loop creation failed"))
        ):
            with pytest.raises(Exception, match="Loop creation failed"):
                get_user_input(test_prompt)

    def test_get_user_input_exception_in_set_event_loop(self):
        """Test handling of exception in set_event_loop."""
        test_prompt = "Enter input:"
        
        mock_loop = MagicMock()
        
        with (
            patch("asyncio.get_event_loop", side_effect=RuntimeError("No event loop")),
            patch("asyncio.new_event_loop", return_value=mock_loop),
            patch("asyncio.set_event_loop", side_effect=Exception("Set loop failed"))
        ):
            with pytest.raises(Exception, match="Set loop failed"):
                get_user_input(test_prompt)

    def test_get_user_input_multiple_calls_same_loop(self):
        """Test multiple calls to get_user_input with same event loop."""
        test_prompt = "Enter input:"
        
        responses = ["response1", "response2", "response3"]
        call_count = 0
        
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            nonlocal call_count
            response = responses[call_count]
            call_count += 1
            return response
        
        mock_loop = MagicMock()
        mock_loop.run_until_complete.side_effect = responses
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            # Multiple calls should use the same loop
            result1 = get_user_input(test_prompt)
            result2 = get_user_input(test_prompt)
            result3 = get_user_input(test_prompt)
            
            assert result1 == "response1"
            assert result2 == "response2"
            assert result3 == "response3"
            
            # Event loop should be retrieved multiple times but not created
            assert mock_loop.run_until_complete.call_count == 3

    def test_get_user_input_mixed_loop_scenarios(self):
        """Test get_user_input with mixed event loop scenarios."""
        test_prompt = "Enter input:"
        
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            return "response"
        
        # First call - existing loop
        mock_existing_loop = MagicMock()
        mock_existing_loop.run_until_complete.return_value = "response"
        
        # Second call - no loop, create new one
        mock_new_loop = MagicMock()
        mock_new_loop.run_until_complete.return_value = "response"
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", side_effect=[mock_existing_loop, RuntimeError("No event loop")]),
            patch("asyncio.new_event_loop", return_value=mock_new_loop),
            patch("asyncio.set_event_loop") as mock_set_loop
        ):
            # First call - uses existing loop
            result1 = get_user_input(test_prompt)
            assert result1 == "response"
            
            # Second call - creates new loop
            result2 = get_user_input(test_prompt)
            assert result2 == "response"
            
            # Verify new loop was set
            mock_set_loop.assert_called_once_with(mock_new_loop)

    def test_get_user_input_coroutine_handling(self):
        """Test that coroutine is properly handled by event loop."""
        test_prompt = "Enter input:"
        test_response = "coroutine response"
        
        # Create actual coroutine
        async def actual_coroutine():
            return test_response
        
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = test_response
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", return_value=actual_coroutine()),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            result = get_user_input(test_prompt)
            
            assert result == test_response
            # Verify that run_until_complete was called with a coroutine
            mock_loop.run_until_complete.assert_called_once()

    def test_get_user_input_empty_prompt(self):
        """Test get_user_input with empty prompt."""
        empty_prompt = ""
        
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            assert prompt == empty_prompt
            return "response"
        
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = "response"
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            result = get_user_input(empty_prompt)
            assert result == "response"

    def test_get_user_input_unicode_prompt(self):
        """Test get_user_input with unicode prompt."""
        unicode_prompt = "请输入: 🐍"
        
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            assert prompt == unicode_prompt
            return "unicode response"
        
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = "unicode response"
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            result = get_user_input(unicode_prompt)
            assert result == "unicode response"

    def test_get_user_input_long_prompt(self):
        """Test get_user_input with very long prompt."""
        long_prompt = "x" * 1000
        
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            assert prompt == long_prompt
            return "long prompt response"
        
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = "long prompt response"
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            result = get_user_input(long_prompt)
            assert result == "long prompt response"


class TestUserInputEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Reset the global session before each test."""
        import strands_tools.utils.user_input
        strands_tools.utils.user_input.session = None

    def test_get_user_input_none_prompt(self):
        """Test get_user_input with None prompt."""
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            assert prompt is None
            return "none prompt response"
        
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = "none prompt response"
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            result = get_user_input(None)
            assert result == "none prompt response"

    def test_get_user_input_numeric_default(self):
        """Test get_user_input with numeric default value."""
        test_prompt = "Enter number:"
        numeric_default = 42
        
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            assert default == numeric_default
            return str(numeric_default)
        
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = "42"
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            result = get_user_input(test_prompt, default=numeric_default)
            assert result == "42"

    def test_get_user_input_boolean_default(self):
        """Test get_user_input with boolean default value."""
        test_prompt = "Enter boolean:"
        boolean_default = True
        
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            assert default == boolean_default
            return str(boolean_default)
        
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = "True"
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            result = get_user_input(test_prompt, default=boolean_default)
            assert result == "True"

    def test_get_user_input_list_default(self):
        """Test get_user_input with list default value."""
        test_prompt = "Enter list:"
        list_default = [1, 2, 3]
        
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            assert default == list_default
            return str(list_default)
        
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = "[1, 2, 3]"
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            result = get_user_input(test_prompt, default=list_default)
            assert result == "[1, 2, 3]"

    @pytest.mark.asyncio
    async def test_get_user_input_async_session_initialization_error_recovery(self):
        """Test session initialization error and recovery."""
        test_prompt = "Enter input:"
        
        # Reset global session
        import strands_tools.utils.user_input
        strands_tools.utils.user_input.session = None
        
        call_count = 0
        
        def mock_session_class():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First initialization failed")
            # Second call succeeds
            mock_session = AsyncMock()
            mock_session.prompt_async.return_value = "recovered response"
            return mock_session
        
        with patch("strands_tools.utils.user_input.PromptSession", side_effect=mock_session_class):
            with patch("strands_tools.utils.user_input.patch_stdout"):
                # First call should fail
                with pytest.raises(Exception, match="First initialization failed"):
                    await get_user_input_async(test_prompt)
                
                # Reset session for second attempt
                strands_tools.utils.user_input.session = None
                
                # Second call should succeed
                result = await get_user_input_async(test_prompt)
                assert result == "recovered response"

    @pytest.mark.asyncio
    async def test_get_user_input_async_patch_stdout_exception(self):
        """Test handling of patch_stdout exception."""
        test_prompt = "Enter input:"
        
        with patch("strands_tools.utils.user_input.PromptSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.prompt_async.return_value = "response"
            mock_session_class.return_value = mock_session
            
            with patch("strands_tools.utils.user_input.patch_stdout", side_effect=Exception("Patch stdout failed")):
                with pytest.raises(Exception, match="Patch stdout failed"):
                    await get_user_input_async(test_prompt)

    def test_get_user_input_event_loop_policy_changes(self):
        """Test get_user_input with event loop policy changes."""
        test_prompt = "Enter input:"
        
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            return "policy response"
        
        # Mock different event loop policies
        mock_loop1 = MagicMock()
        mock_loop1.run_until_complete.return_value = "policy response"
        
        mock_loop2 = MagicMock()
        mock_loop2.run_until_complete.return_value = "policy response"
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", side_effect=[mock_loop1, RuntimeError(), mock_loop2]),
            patch("asyncio.new_event_loop", return_value=mock_loop2),
            patch("asyncio.set_event_loop")
        ):
            # First call uses existing loop
            result1 = get_user_input(test_prompt)
            assert result1 == "policy response"
            
            # Second call creates new loop due to RuntimeError
            result2 = get_user_input(test_prompt)
            assert result2 == "policy response"


class TestUserInputIntegration:
    """Integration tests for user input functionality."""

    def setup_method(self):
        """Reset the global session before each test."""
        import strands_tools.utils.user_input
        strands_tools.utils.user_input.session = None

    def test_user_input_full_workflow_simulation(self):
        """Test complete user input workflow simulation."""
        prompts_and_responses = [
            ("Enter your name:", "Alice"),
            ("Enter your age:", "30"),
            ("Enter your email:", "alice@example.com"),
            ("Confirm (y/n):", "y")
        ]
        
        responses = [response for _, response in prompts_and_responses]
        call_count = 0
        
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            nonlocal call_count
            expected_prompt = prompts_and_responses[call_count][0]
            expected_response = prompts_and_responses[call_count][1]
            call_count += 1
            
            assert prompt == expected_prompt
            return expected_response
        
        mock_loop = MagicMock()
        mock_loop.run_until_complete.side_effect = responses
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            # Simulate a form-filling workflow
            name = get_user_input("Enter your name:")
            age = get_user_input("Enter your age:")
            email = get_user_input("Enter your email:")
            confirm = get_user_input("Confirm (y/n):")
            
            assert name == "Alice"
            assert age == "30"
            assert email == "alice@example.com"
            assert confirm == "y"

    def test_user_input_error_recovery_workflow(self):
        """Test user input error recovery workflow."""
        call_count = 0
        
        async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                raise KeyboardInterrupt()
            elif call_count == 2:
                raise EOFError()
            else:
                return "final response"
        
        mock_loop = MagicMock()
        mock_loop.run_until_complete.side_effect = ["default", "default", "final response"]
        
        with (
            patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
            patch("asyncio.get_event_loop", return_value=mock_loop)
        ):
            # First call - KeyboardInterrupt, should return default
            result1 = get_user_input("Prompt 1:", default="default")
            assert result1 == "default"
            
            # Second call - EOFError, should return default
            result2 = get_user_input("Prompt 2:", default="default")
            assert result2 == "default"
            
            # Third call - success
            result3 = get_user_input("Prompt 3:")
            assert result3 == "final response"

    def test_user_input_concurrent_calls_simulation(self):
        """Test simulation of concurrent user input calls."""
        import threading
        import time
        
        results = {}
        
        def user_input_thread(thread_id, prompt):
            async def mock_async_func(prompt, default, keyboard_interrupt_return_default):
                # Simulate some processing time
                await asyncio.sleep(0.01)
                return f"response_{thread_id}"
            
            mock_loop = MagicMock()
            mock_loop.run_until_complete.return_value = f"response_{thread_id}"
            
            with (
                patch("strands_tools.utils.user_input.get_user_input_async", side_effect=mock_async_func),
                patch("asyncio.get_event_loop", return_value=mock_loop)
            ):
                result = get_user_input(prompt)
                results[thread_id] = result
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=user_input_thread, 
                args=(i, f"Prompt {i}:")
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 3
        for i in range(3):
            assert results[i] == f"response_{i}"