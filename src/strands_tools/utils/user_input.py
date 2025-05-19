"""
Unified user input handling module for STRANDS tools.
Uses prompt_toolkit for input features and rich.console for styling.
"""

import asyncio

from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

# Lazy initialize to avoid import errors for tests on windows without a terminal
session: PromptSession | None = None


async def get_user_input_async(prompt: str, default: str = "n") -> str:
    global session

    """
    Asynchronously get user input with prompt_toolkit's features (history, arrow keys, styling, etc).

    Args:
        prompt: The prompt to show
        default: Default response (default is 'n')

    Returns:
        str: The user's input response
    """

    try:
        with patch_stdout(raw=True):
            if session is None:
                session = PromptSession()

            response = await session.prompt_async(HTML(f"{prompt} "))

        if not response:
            return str(default)

        return str(response)
    except (KeyboardInterrupt, EOFError):
        return default


def get_user_input(prompt: str, default: str = "n") -> str:
    """
    Synchronous wrapper for get_user_input_async.

    Args:
        prompt: The prompt to show
        default: Default response shown in prompt (default is 'n')

    Returns:
        str: The user's input response
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Get result and ensure it's returned as a string
    result = loop.run_until_complete(get_user_input_async(prompt, default))
    return str(result)
