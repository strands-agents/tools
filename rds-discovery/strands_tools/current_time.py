import os
from datetime import datetime
from datetime import timezone as tz
from typing import Any
from zoneinfo import ZoneInfo

from strands import tool


@tool
def current_time(timezone: str = None) -> str:
    """
    Get the current time in ISO 8601 format.

    This tool returns the current date and time in ISO 8601 format (e.g., 2023-04-15T14:32:16.123456+00:00)
    for the specified timezone. If no timezone is provided, the value from the DEFAULT_TIMEZONE
    environment variable is used (defaults to 'UTC' if not set).

    Args:
        timezone (str, optional): The timezone to use (e.g., 'UTC', 'US/Pacific', 'Europe/London', 'Asia/Tokyo').
            Defaults to environment variable DEFAULT_TIMEZONE ('UTC' if not set).

    Returns:
        str: The current time in ISO 8601 format.

    Raises:
        ValueError: If an invalid timezone is provided.

    Examples:
        >>> current_time()  # Returns current time in default timezone (from DEFAULT_TIMEZONE or UTC)
        '2023-04-15T14:32:16.123456+00:00'

        >>> current_time(timezone="US/Pacific")  # Returns current time in Pacific timezone
        '2023-04-15T07:32:16.123456-07:00'
    """
    # Get environment variables at runtime
    default_timezone = os.getenv("DEFAULT_TIMEZONE", "UTC")

    # Use provided timezone or fall back to default
    timezone = timezone or default_timezone

    try:
        if timezone.upper() == "UTC":
            timezone_obj: Any = tz.utc
        else:
            timezone_obj = ZoneInfo(timezone)

        return datetime.now(timezone_obj).isoformat()
    except Exception as e:
        raise ValueError(f"Error getting current time: {str(e)}") from e
