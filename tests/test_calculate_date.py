from datetime import date

import pytest

from strands_tools.calculate_date import _add_months, calculate_date


def _calc(**kwargs):
    """Bypass @tool decorator to test the function directly."""
    return calculate_date._tool_func(**kwargs)


def test_days_offset():
    assert _calc(offset_type="days", offset_value=0, base_date="2026-04-02") == "2026-04-02"
    assert _calc(offset_type="days", offset_value=90, base_date="2024-01-15") == "2024-04-14"
    assert _calc(offset_type="days", offset_value=-30, base_date="2026-04-02") == "2026-03-03"


def test_weeks_offset():
    assert _calc(offset_type="weeks", offset_value=2, base_date="2026-04-02") == "2026-04-16"
    assert _calc(offset_type="weeks", offset_value=-2, base_date="2026-04-16") == "2026-04-02"


def test_months_offset():
    assert _calc(offset_type="months", offset_value=0, base_date="2026-04-02") == "2026-04-02"
    assert _calc(offset_type="months", offset_value=3, base_date="2026-01-15") == "2026-04-15"
    assert _calc(offset_type="months", offset_value=-6, base_date="2026-04-02") == "2025-10-02"
    assert _calc(offset_type="months", offset_value=25, base_date="2025-01-15") == "2027-02-15"
    assert _calc(offset_type="months", offset_value=-25, base_date="2025-01-15") == "2022-12-15"


def test_years_offset():
    assert _calc(offset_type="years", offset_value=0, base_date="2026-04-02") == "2026-04-02"
    assert _calc(offset_type="years", offset_value=-1, base_date="2026-04-02") == "2025-04-02"


def test_month_end_clamping():
    # Jan 31 + 1 month -> Feb 28 (non-leap) or Feb 29 (leap)
    assert _calc(offset_type="months", offset_value=1, base_date="2025-01-31") == "2025-02-28"
    assert _calc(offset_type="months", offset_value=1, base_date="2024-01-31") == "2024-02-29"
    # Feb 29 + 1 year -> Feb 28
    assert _calc(offset_type="years", offset_value=1, base_date="2024-02-29") == "2025-02-28"


def test_year_and_december_boundary():
    # Crossing year boundary backwards
    assert _calc(offset_type="months", offset_value=-2, base_date="2026-01-15") == "2025-11-15"
    assert _calc(offset_type="months", offset_value=-1, base_date="2026-01-15") == "2025-12-15"
    # Crossing year boundary forwards
    assert _calc(offset_type="months", offset_value=3, base_date="2025-11-15") == "2026-02-15"


def test_default_base_date():
    assert _calc(offset_type="days", offset_value=0) == date.today().isoformat()


def test_invalid_inputs():
    with pytest.raises(ValueError):
        _calc(offset_type="hours", offset_value=1, base_date="2026-04-02")
    with pytest.raises(ValueError):
        _calc(offset_type="days", offset_value=1, base_date="not-a-date")


def test_add_months_helper():
    assert _add_months(date(2025, 1, 31), 1) == date(2025, 2, 28)
    assert _add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert _add_months(date(2026, 3, 15), -12) == date(2025, 3, 15)
