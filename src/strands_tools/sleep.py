import logging
import os
import time
from datetime import datetime
from typing import Union

from strands import tool
from typing_extensions import deprecated

logger = logging.getLogger(__name__)

# Default maximum sleep time (5 minutes)
max_sleep_seconds = int(os.environ.get("MAX_SLEEP_SECONDS", "300"))

_DEPRECATION_MESSAGE = (
    "sleep is deprecated. This warning becomes an error log in v0.9.0. "
    "Migration path: use the sleep tool vended by strands-agents "
    "(from strands.vended_tools import sleep)."
)


# @deprecated surfaces in IDEs and type checkers; the logger.warning below is what
# users actually see, since DeprecationWarning raised from inside the SDK's tool
# invocation path is suppressed by Python's default warning filter. The message is
# spelled out here rather than passed as _DEPRECATION_MESSAGE because mypy only
# reports @deprecated when the argument is a string literal.
@tool
@deprecated(
    "sleep is deprecated. This warning becomes an error log in v0.9.0. "
    "Migration path: use the sleep tool vended by strands-agents "
    "(from strands.vended_tools import sleep)."
)
def sleep(seconds: Union[int, float]) -> str:
    """
    Pause execution for the specified number of seconds.

    This tool pauses the execution flow for the given number of seconds.
    It can be interrupted with SIGINT (Ctrl+C).

    Args:
        seconds (Union[int, float]): Number of seconds to sleep.
            Must be a positive number greater than 0 and less than or equal to
            the maximum allowed value (default: 300 seconds, configurable via
            MAX_SLEEP_SECONDS environment variable).

    Returns:
        str: A message indicating the sleep completed or was interrupted.

    Raises:
        ValueError: If seconds is not positive, exceeds the maximum allowed value,
                   or is not a number.

    Examples:
        >>> sleep(5)  # Sleeps for 5 seconds
        'Started sleep at 2025-05-30 11:30:00, slept for 5.0 seconds'

        >>> sleep(0.5)  # Sleeps for half a second
        'Started sleep at 2025-05-30 11:30:00, slept for 0.5 seconds'
    """
    logger.warning("DEPRECATION WARNING: %s", _DEPRECATION_MESSAGE)

    # Validate input
    if not isinstance(seconds, (int, float)):
        raise ValueError("Sleep duration must be a number")

    if seconds <= 0:
        raise ValueError("Sleep duration must be greater than 0")

    if seconds > max_sleep_seconds:
        raise ValueError(f"Sleep duration cannot exceed {max_sleep_seconds} seconds")

    try:
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time.sleep(seconds)
        return f"Started sleep at {start_time}, slept for {float(seconds)} seconds"
    except KeyboardInterrupt:
        return "Sleep interrupted by user"
