import time
from datetime import datetime
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
            Must be a positive number greater than 0.

    Returns:
        str: A message indicating the sleep completed or was interrupted.

    Raises:
        ValueError: If seconds is not positive or not a number.

    Examples:
        >>> sleep(5)  # Sleeps for 5 seconds
        'Started sleep at 2025-05-30 11:30:00, slept for 5.0 seconds'

        >>> sleep(0.5)  # Sleeps for half a second
        'Started sleep at 2025-05-30 11:30:00, slept for 0.5 seconds'
    """
    # Validate input
    if not isinstance(seconds, (int, float)):
        raise ValueError("Sleep duration must be a number")

    if seconds <= 0:
        raise ValueError("Sleep duration must be greater than 0")

    try:
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time.sleep(seconds)
        return f"Started sleep at {start_time}, slept for {float(seconds)} seconds"
    except KeyboardInterrupt:
        return "Sleep interrupted by user"
