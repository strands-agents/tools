import time
from typing import Union

from strands import tool


@tool
def sleep(seconds: Union[int, float]) -> str:
    """
    Pause execution for the specified number of seconds.

    This tool pauses the execution flow for the given number of seconds.
    It can be interrupted with SIGINT (Ctrl+C).

    Args:
        seconds (Union[int, float]): Number of seconds to sleep.
            Must be a positive number.

    Returns:
        str: A message indicating the sleep completed or was interrupted.

    Raises:
        ValueError: If seconds is negative or not a number.

    Examples:
        >>> sleep(5)  # Sleeps for 5 seconds
        'Slept for 5.0 seconds'

        >>> sleep(0.5)  # Sleeps for half a second
        'Slept for 0.5 seconds'
    """
    # Validate input
    if not isinstance(seconds, (int, float)):
        raise ValueError("Sleep duration must be a number")

    if seconds < 0:
        raise ValueError("Sleep duration cannot be negative")

    try:
        time.sleep(seconds)
        return f"Slept for {float(seconds)} seconds"
    except KeyboardInterrupt:
        return "Sleep interrupted by user"
