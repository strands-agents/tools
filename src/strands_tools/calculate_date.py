"""Relative date calculation tool."""

import calendar
from datetime import date, timedelta

from strands import tool


def _add_months(d: date, months: int) -> date:
    """Add months to a date, clamping to the last valid day of the target month."""
    # Python's floor division/modulo handles negative months correctly
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    max_day = calendar.monthrange(year, month)[1]
    return d.replace(year=year, month=month, day=min(d.day, max_day))


@tool
def calculate_date(
    offset_type: str = "days",
    offset_value: int = 0,
    base_date: str | None = None,
) -> str:
    """Calculate a date relative to today or a given base date.

    Supports offsets in days, weeks, months, and years. Handles month-end
    clamping (e.g. Jan 31 + 1 month = Feb 28) and leap year transitions.

    Args:
        offset_type: One of 'days', 'weeks', 'months', 'years'. Defaults to 'days'.
        offset_value: Number of units to offset. Positive for future, negative for past.
        base_date: Starting date as YYYY-MM-DD. Defaults to today.

    Returns:
        str: The calculated date as YYYY-MM-DD.
    """
    if base_date is not None:
        try:
            base = date.fromisoformat(base_date)
        except ValueError as e:
            raise ValueError(f"Invalid base_date: {base_date}") from e
    else:
        base = date.today()

    if offset_type == "days":
        result = base + timedelta(days=offset_value)
    elif offset_type == "weeks":
        result = base + timedelta(weeks=offset_value)
    elif offset_type == "months":
        result = _add_months(base, offset_value)
    elif offset_type == "years":
        result = _add_months(base, offset_value * 12)
    else:
        raise ValueError(f"Invalid offset_type: {offset_type}")

    return result.isoformat()
