import re
from datetime import datetime
from typing import Any


def convert_datetime_to_str(obj: Any) -> Any:
    """
    Recursively converts datetime.datetime objects to strings in the desired format
    within a JSON-like object (dict or list).
    """
    desired_format = "%Y-%m-%d %H:%M:%S%z"

    if isinstance(obj, datetime):
        return obj.strftime(desired_format)
    elif isinstance(obj, dict):
        return {k: convert_datetime_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetime_to_str(item) for item in obj]
    else:
        return obj


def to_snake_case(text: str) -> str:
    pattern = re.compile(r"(?<!^)(?=[A-Z])")
    return pattern.sub("_", text).lower()
