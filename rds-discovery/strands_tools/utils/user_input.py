"""
Unified user input handling module for STRANDS tools.
Uses prompt_toolkit for input features and rich.console for styling.
"""

import asyncio

from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

# Lazy initialize to avoid import errors for tests on windows without a terminal
session: PromptSession | None = None


async def get_user_input_async(prompt: str, default: str = "", keyboard_interrupt_return_default: bool = True) -> str:
    """
    Asynchronously get user input with prompt_toolkit's features (history, arrow keys, styling, etc.).

    Args:
        prompt: The prompt to show
        default: Default response (default is 'n')
        keyboard_interrupt_return_default: Return default value on keyboard interrupt or EOF error (default is True)

    Returns:
        str: The user's input response
    """

    async def _get_input():
        global session

        with patch_stdout(raw=True):
            if session is None:
                session = PromptSession()

            response = await session.prompt_async(HTML(f"{prompt} "))

        if not response:
            return str(default)

        return str(response)

    if keyboard_interrupt_return_default:
        try:
            return await _get_input()
        except (KeyboardInterrupt, EOFError):
            return default

    return await _get_input()


def get_user_input(prompt: str, default: str = "", keyboard_interrupt_return_default: bool = True) -> str:
    """
    Synchronous wrapper for get_user_input_async.

    Args:
        prompt: The prompt to show
        default: Default response shown in prompt (default is 'n')
        keyboard_interrupt_return_default: Return default value on keyboard interrupt or EOF error (default is True)

    Returns:
        str: The user's input response
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Get result and ensure it's returned as a string
    result = loop.run_until_complete(get_user_input_async(prompt, default, keyboard_interrupt_return_default))
    return str(result)
